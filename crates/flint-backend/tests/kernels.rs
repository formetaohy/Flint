//! Kernel conformance. Every kernel runs on the WGPU backend and must match
//! the CPU reference (`flint_kernel::cpu`) on identical inputs; the anchor
//! tests at the bottom pin the CPU reference itself to hand-computed math, so
//! the two backends cannot silently agree on wrong results.

use flint_backend::{Backend, Binding, Pass};
use flint_kernel::cpu;
use flint_tensor::{DType, Tensor, Weight};

// ================================================================ harness

struct Ctx {
    backend: Backend,
}

impl Ctx {
    fn new() -> Self {
        Self {
            backend: Backend::new().unwrap(),
        }
    }

    fn f32(&self, data: &[f32], shape: &[u32]) -> Tensor {
        self.backend.tensor_f32(data, shape.to_vec(), "t")
    }

    fn bf16(&self, data: &[f32], shape: &[u32]) -> Tensor {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        self.backend
            .tensor_bf16(&bytes, shape.to_vec(), "t")
            .unwrap()
    }

    fn zero(&self, shape: &[u32]) -> Tensor {
        self.backend.zero_tensor(shape, "z")
    }

    fn zero_bf16(&self, shape: &[u32]) -> Tensor {
        self.backend.zero_bf16_tensor(shape, "z")
    }

    /// One-u32 step-args buffer holding a position, for rope/kv_store/attn.
    fn arg(&self, pos: usize) -> Tensor {
        let t = Tensor::new(self.backend.storage(4, "args"), vec![1], DType::U32);
        self.backend.write_u32(&t.buf, &[pos as u32]);
        t
    }

    /// Reads a packed-bf16 tensor back as f32 (two elements per u32).
    fn read_bf16(&self, t: &Tensor) -> Vec<f32> {
        let raw = self
            .backend
            .read_f32(&t.buf, 0, (t.numel() / 2) as usize)
            .unwrap();
        let mut out = Vec::with_capacity(t.numel() as usize);
        for w in raw {
            let bits = w.to_bits();
            out.push(f32::from_bits((bits & 0xFFFF) << 16));
            out.push(f32::from_bits((bits >> 16) << 16));
        }
        out
    }

    fn dispatch(
        &mut self,
        name: &'static str,
        consts: &[(&'static str, f64)],
        bufs: &[Binding<'_>],
        groups: [u32; 3],
    ) {
        let mut enc = self.backend.encoder();
        {
            let mut pass = Pass::begin(&mut enc, "k");
            self.backend
                .dispatch(&mut pass, name, consts, bufs, groups)
                .unwrap();
        }
        self.backend.submit(enc);
    }

    fn read(&self, t: &Tensor) -> Vec<f32> {
        self.backend
            .read_f32(&t.buf, 0, t.numel() as usize)
            .unwrap()
    }
}

/// Deterministic PRNG for reproducible inputs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn bf16_round(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|x| f32::from_bits((x.to_bits() >> 16) << 16))
        .collect()
}

/// Asserts the WGPU and CPU results agree elementwise within tolerance.
fn agree(gpu: &[f32], cpu: &[f32], rel: f32, abs: f32) {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "length mismatch {} vs {}",
        gpu.len(),
        cpu.len()
    );
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let diff = (g - c).abs();
        assert!(
            diff <= rel * c.abs() + abs,
            "index {i}: gpu {g} vs cpu {c} (diff {diff})"
        );
    }
}

/// Group absmax quantization, written independently of `flint_model::loader`
/// (which cannot be depended on here) so it also cross-checks the loader math.
fn quant(data: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    for r in 0..rows {
        for g in 0..cols / group {
            let block = &data[r * cols + g * group..r * cols + (g + 1) * group];
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales.push(scale);
            for v in block {
                bytes.push((v / scale).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
    }
    (bytes, scales)
}

fn dequant(bytes: &[u8], scales: &[f32], cols: usize, group: usize) -> Vec<f32> {
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let row = i / cols;
            let g = (i % cols) / group;
            (b as i8) as f32 * scales[row * (cols / group) + g]
        })
        .collect()
}

// ================================================================ gemm
// The streaming two-row matmul is the production prefill path; it must agree
// with the CPU reference on every weight layout.

enum WType {
    Bf16,
    I8(usize), // group size
}

fn gemm_case(wt: WType, m: usize, n: usize, k: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);

    let (wb, sb, wdtype, group, cpu_w, rel, abs) = match wt {
        WType::Bf16 => {
            let wb = ctx.bf16(&w, &[n as u32, k as u32]);
            let sb = ctx.f32(&[0.0], &[1]);
            (wb, sb, 0.0, 128.0, bf16_round(&w), 2e-2, 5e-2)
        }
        WType::I8(group) => {
            let (wq, scales) = quant(&w, n, k, group);
            let cpu_w = dequant(&wq, &scales, k, group);
            let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32], "wq");
            let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
            (wb, sb, 1.0, group as f64, cpu_w, 1e-4, 1e-3)
        }
    };

    ctx.dispatch(
        "gemm",
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", wdtype),
            ("GROUP", group),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&sb),
            Binding::Full(&y),
        ],
        [(n / 16) as u32, (m / 2) as u32, 1],
    );
    agree(&ctx.read(&y), &cpu::gemm(&x, &cpu_w, m, n, k), rel, abs);
}

#[test]
fn gemm_bf16() {
    gemm_case(WType::Bf16, 16, 64, 128, 7);
}

#[test]
fn gemm_bf16_multi_tile_m() {
    gemm_case(WType::Bf16, 32, 32, 64, 29);
}

#[test]
fn gemm_i8_group128() {
    gemm_case(WType::I8(128), 16, 64, 256, 17);
}

#[test]
fn gemm_i8_group64() {
    // SmolLM-style 960-wide hidden: K not a multiple of 128.
    gemm_case(WType::I8(64), 16, 64, 960, 19);
}

#[test]
fn gemm_i8_group32() {
    gemm_case(WType::I8(32), 16, 32, 192, 23);
}

// ================================================================ gemv
// The split-K decode gemv plus its merge pass. SEGS=1 exercises the direct
// write; SEGS>1 the partial + merge path; every case must match the CPU
// reference.

fn gemv_case(wt: WType, n: usize, k: usize, seed: u64, segs: u32) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[k as u32]);
    let y = ctx.zero(&[n as u32]);
    let partial = ctx.zero(&[8, 65536]);

    let (wb, sb, wdtype, group, cpu_w, rel, abs) = match wt {
        WType::Bf16 => {
            let wb = ctx.bf16(&w, &[n as u32, k as u32]);
            let sb = ctx.f32(&[0.0], &[1]);
            (wb, sb, 0.0, 128.0, bf16_round(&w), 2e-2, 5e-2)
        }
        WType::I8(group) => {
            let (wq, scales) = quant(&w, n, k, group);
            let cpu_w = dequant(&wq, &scales, k, group);
            let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32], "wq");
            let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
            (wb, sb, 1.0, group as f64, cpu_w, 1e-4, 1e-3)
        }
    };

    let out = if segs == 1 {
        Binding::Full(&y)
    } else {
        Binding::Slice(&partial, 0, n as u64 * 4 * segs as u64)
    };
    ctx.dispatch(
        "gemv",
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", wdtype),
            ("GROUP", group),
            ("SEGS", segs as f64),
        ],
        &[Binding::Full(&xb), Binding::Full(&wb), Binding::Full(&sb), out],
        [(n / 16) as u32, segs, 1],
    );
    if segs > 1 {
        ctx.dispatch(
            "merge_gemv",
            &[("N", n as f64), ("SEGS", segs as f64)],
            &[
                Binding::Slice(&partial, 0, n as u64 * 4 * segs as u64),
                Binding::Full(&y),
            ],
            [(n / 16) as u32, 1, 1],
        );
    }
    agree(&ctx.read(&y), &cpu::gemv(&x, &cpu_w, n, k), rel, abs);
}

#[test]
fn gemv_bf16() {
    gemv_case(WType::Bf16, 32, 128, 31, 1);
}

#[test]
fn gemv_i8_group128() {
    gemv_case(WType::I8(128), 64, 256, 37, 1);
}

#[test]
fn gemv_i8_partial_chunk() {
    // K not a multiple of BK=128 exercises the final partial-chunk guard.
    gemv_case(WType::I8(32), 32, 192, 41, 1);
}

#[test]
fn gemv_i8_split4() {
    gemv_case(WType::I8(128), 64, 512, 43, 4);
}

#[test]
fn gemv_bf16_split8() {
    gemv_case(WType::Bf16, 32, 1024, 47, 8);
}

// ================================================================ embed

fn embed_case(rows: usize, dim: usize, scale: f32, seed: u64) {
    let mut ctx = Ctx::new();
    let vocab = 8usize;
    let mut rng = Rng(seed);
    let table = rng.fill(vocab * dim);
    let ids: Vec<u32> = (0..rows)
        .map(|r| (r * 3 + 1) as u32 % vocab as u32)
        .collect();

    let tb = ctx.bf16(&table, &[vocab as u32, dim as u32]);
    let ib = Tensor::new(
        ctx.backend.storage(rows as u64 * 4, "ids"),
        vec![rows as u32],
        DType::F32,
    );
    ctx.backend.write_u32(&ib.buf, &ids);
    let y = ctx.zero(&[16 as u32, dim as u32]);
    let fallback = ctx.f32(&[1.0], &[1]);
    let w = Weight::plain(tb);

    ctx.dispatch(
        "embed",
        &[
            ("ROWS", 16.0),
            ("DIM", dim as f64),
            ("SCALE", scale as f64),
            ("WDTYPE", 0.0),
            ("GROUP", 128.0),
        ],
        &[
            Binding::Full(&ib),
            Binding::Full(w.tensor()),
            Binding::Full(&fallback),
            Binding::Full(&y),
        ],
        [1, 1, 1],
    );
    let got = ctx.read(&y);
    agree(
        &got[..rows * dim],
        &cpu::embed(&ids, &bf16_round(&table), dim, scale),
        0.0,
        1e-6,
    );
}

#[test]
fn embed_unit_scale() {
    embed_case(4, 16, 1.0, 11);
}

#[test]
fn embed_gemma_scale() {
    embed_case(3, 16, 4.0, 13);
}

// ================================================================ norm

fn norm_case(mode: u32, rows: usize, dim: usize, w_dim: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(rows * dim);
    let w = rng.fill(w_dim);
    let z = rng.fill(rows * dim);
    let xb = ctx.f32(&x, &[rows as u32, dim as u32]);
    let wb = ctx.f32(&w, &[w_dim as u32]);
    let zb = ctx.f32(&z, &[rows as u32, dim as u32]);
    let y = ctx.zero(&[rows as u32, dim as u32]);

    ctx.dispatch(
        "norm",
        &[
            ("MODE", mode as f64),
            ("DIM", dim as f64),
            ("W_DIM", w_dim as f64),
            ("EPS", 1e-6),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&zb),
            Binding::Full(&y),
        ],
        [rows as u32, 1, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu::norm(mode, &x, &w, &z, rows, dim, w_dim, 1e-6),
        1e-4,
        1e-5,
    );
}

#[test]
fn norm_layer() {
    norm_case(3, 4, 64, 64, 29);
}

#[test]
fn norm_offset() {
    norm_case(0, 3, 64, 64, 21);
}

#[test]
fn norm_gated_weight_repeats_across_row() {
    norm_case(1, 4, 64, 8, 31);
}

#[test]
fn norm_direct() {
    norm_case(2, 2, 32, 32, 27);
}

// ================================================================ elementwise

#[test]
fn add() {
    let mut ctx = Ctx::new();
    let n = 100usize; // not a multiple of 256: exercises the bounds check
    let mut rng = Rng(33);
    let a = rng.fill(n);
    let b = rng.fill(n);
    let ab = ctx.f32(&a, &[n as u32]);
    let bb = ctx.f32(&b, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        "add",
        &[("N_ELEM", n as f64)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu::add(&a, &b), 0.0, 0.0);
}

#[test]
fn bias() {
    let mut ctx = Ctx::new();
    let (rows, dim) = (3usize, 16usize);
    let mut rng = Rng(35);
    let x = rng.fill(rows * dim);
    let b = rng.fill(dim);
    let xb = ctx.f32(&x, &[rows as u32, dim as u32]);
    let bb = ctx.f32(&b, &[dim as u32]);

    ctx.dispatch(
        "bias",
        &[("N_ELEM", (rows * dim) as f64), ("DIM", dim as f64)],
        &[Binding::Full(&xb), Binding::Full(&bb)],
        [1, 1, 1],
    );
    let mut cpu = x.clone();
    cpu::bias(&mut cpu, &b, dim);
    agree(&ctx.read(&xb), &cpu, 0.0, 1e-7);
}

#[test]
fn swiglu() {
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(41);
    let g = rng.fill(n);
    let u = rng.fill(n);
    let gb = ctx.f32(&g, &[n as u32]);
    let ub = ctx.f32(&u, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        "swiglu",
        &[("N_ELEM", n as f64), ("MODE", 0.0)],
        &[Binding::Full(&gb), Binding::Full(&ub), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu::swiglu(&g, &u, 0), 1e-5, 1e-6);
}

#[test]
fn swiglu_gelu_tanh() {
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(47);
    let g = rng.fill(n);
    let u = rng.fill(n);
    let gb = ctx.f32(&g, &[n as u32]);
    let ub = ctx.f32(&u, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        "swiglu",
        &[("N_ELEM", n as f64), ("MODE", 1.0)],
        &[Binding::Full(&gb), Binding::Full(&ub), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu::swiglu(&g, &u, 1), 1e-5, 1e-6);
}

#[test]
fn softcap() {
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(53);
    let mut x = rng.fill(n);
    for v in x.iter_mut() {
        *v *= 40.0; // push past the cap
    }
    let xb = ctx.f32(&x, &[n as u32]);
    ctx.dispatch(
        "softcap",
        &[("N_ELEM", n as f64), ("CAP", 30.0)],
        &[Binding::Full(&xb)],
        [1, 1, 1],
    );
    cpu::softcap(&mut x, 30.0);
    agree(&ctx.read(&xb), &x, 1e-5, 1e-6);
}

#[test]
fn mul_broadcast() {
    let mut ctx = Ctx::new();
    let n = 160usize;
    let mut rng = Rng(59);
    let a = rng.fill(n);
    let b = rng.fill(4);
    let ab = ctx.f32(&a, &[n as u32]);
    let bb = ctx.f32(&b, &[4 as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        "mul",
        &[("N", n as f64), ("M", 4.0)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu::mul(&a, &b, n, 4), 1e-6, 1e-7);
}

#[test]
fn expert_gather_scatter() {
    let mut ctx = Ctx::new();
    let hidden = 32usize;
    let mut rng = Rng(61);
    // 4 rows packed into 2 experts: rows [1, 3] -> expert 0, rows [0, 2] -> expert 1.
    let x = rng.fill(4 * hidden);
    let rows: Vec<u32> = vec![1, 3, 0, 2];
    let weights = [0.5f32, 1.5, 2.5, 3.5];
    let xb = ctx.f32(&x, &[4 as u32, hidden as u32]);
    let rb = Tensor::new(ctx.backend.storage(16, "rb"), vec![4], DType::U32);
    ctx.backend.write_u32(&rb.buf, &rows);
    let wb = ctx.f32(&weights, &[4 as u32]);
    let packed = ctx.zero(&[16 as u32, hidden as u32]);
    let acc = ctx.zero(&[4 as u32, hidden as u32]);

    ctx.dispatch(
        "expert_gather",
        &[("ROWS", 16.0), ("HIDDEN", hidden as f64), ("COUNT", 4.0)],
        &[
            Binding::Full(&xb),
            Binding::Slice(&rb, 0, 16),
            Binding::Full(&packed),
        ],
        [1, 1, 1],
    );
    ctx.dispatch(
        "expert_scatter",
        &[
            ("ROWS", 16.0),
            ("HIDDEN", hidden as f64),
            ("COUNT", 4.0),
        ],
        &[
            Binding::Full(&acc),
            Binding::Full(&packed),
            Binding::Full(&rb),
            Binding::Full(&wb),
        ],
        [1, 1, 1],
    );
    let gathered = cpu::expert_gather(&x, &rows, 16, hidden);
    let mut expect = vec![0f32; 4 * hidden];
    cpu::expert_scatter(&mut expect, &gathered, &rows, &weights, hidden);
    agree(&ctx.read(&acc), &expect, 1e-6, 1e-7);
}

#[test]
fn zero_rows() {
    let mut ctx = Ctx::new();
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let xb = ctx.f32(&x, &[4 as u32]);
    ctx.dispatch(
        "zero_rows",
        &[("N_ELEM", 3.0)],
        &[Binding::Full(&xb)],
        [1, 1, 1],
    );
    let mut expect = x.clone();
    cpu::zero_rows(&mut expect, 3);
    agree(&ctx.read(&xb), &expect, 0.0, 0.0);
}

#[test]
fn sigmoid_mul() {
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(43);
    let a = rng.fill(n);
    let b = rng.fill(n);
    let ab = ctx.f32(&a, &[n as u32]);
    let bb = ctx.f32(&b, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        "sigmoid_mul",
        &[("N_ELEM", n as f64)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu::sigmoid_mul(&a, &b), 1e-5, 1e-6);
}

#[test]
fn concat() {
    let mut ctx = Ctx::new();
    let (rows, d) = (2usize, 8usize);
    let mut rng = Rng(37);
    let a = rng.fill(rows * d);
    let b = rng.fill(rows * d);
    let ab = ctx.f32(&a, &[rows as u32, d as u32]);
    let bb = ctx.f32(&b, &[rows as u32, d as u32]);
    let y = ctx.zero(&[rows as u32, 2 * d as u32]);

    ctx.dispatch(
        "concat",
        &[("ROWS", rows as f64), ("D", d as f64)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu::concat(&a, &b, rows, d), 0.0, 0.0);
}

// ================================================================ rope

fn rope_case(m: usize, heads: usize, hd: usize, rot: usize, pos: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let half = rot / 2;
    let max_seq = pos + m + 2;
    let theta = 1e7f64;
    let mut rng = Rng(seed);
    let x = rng.fill(m * heads * hd);
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for p in 0..max_seq {
        for i in 0..half {
            let inv = 1.0 / theta.powf((2 * i) as f64 / rot as f64);
            let a = p as f64 * inv;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    let xb = ctx.f32(&x, &[m as u32, heads as u32, hd as u32]);
    let cb = ctx.f32(&cos, &[max_seq as u32, half as u32]);
    let sb = ctx.f32(&sin, &[max_seq as u32, half as u32]);
    let args = ctx.arg(pos);

    ctx.dispatch(
        "rope",
        &[
            ("HEADS", heads as f64),
            ("HEAD_DIM", hd as f64),
            ("ROT", rot as f64),
            ("COS_STRIDE", half as f64),
        ],
        &[
            Binding::Full(&cb),
            Binding::Full(&sb),
            Binding::Full(&xb),
            Binding::Full(&args),
        ],
        [m as u32, heads as u32, 1],
    );
    let mut cpu = x.clone();
    cpu::rope(&mut cpu, &cos, &sin, m, heads, hd, rot, pos);
    agree(&ctx.read(&xb), &cpu, 1e-5, 1e-6);
}

#[test]
fn rope_partial_rotation_multi_row() {
    rope_case(2, 2, 32, 16, 2, 51);
}

#[test]
fn rope_full_rotation() {
    rope_case(1, 1, 32, 32, 0, 53);
}

// ================================================================ conv1d

#[test]
fn conv1d_rolls_state_across_steps() {
    let mut ctx = Ctx::new();
    let dim = 8usize;
    let mut rng = Rng(61);
    let w = rng.fill(dim * 4);
    let mut state = rng.fill(dim * 3);
    let steps: Vec<Vec<f32>> = (0..3).map(|_| rng.fill(dim)).collect();

    let wb = ctx.f32(&w, &[dim as u32, 4]);
    let st = ctx.f32(&state, &[dim as u32, 3]);
    let xb = ctx.zero(&[dim as u32]);
    let y = ctx.zero(&[dim as u32]);

    for x in &steps {
        ctx.backend.write_f32(&xb.buf, x);
        ctx.dispatch(
            "conv1d",
            &[("DIM", dim as f64)],
            &[
                Binding::Full(&xb),
                Binding::Full(&wb),
                Binding::Full(&st),
                Binding::Full(&y),
            ],
            [1, 1, 1],
        );
        let cpu_y = cpu::conv1d(x, &w, &mut state);
        agree(&ctx.read(&y), &cpu_y, 1e-5, 1e-6);
    }
    agree(&ctx.read(&st), &state, 0.0, 1e-6);
}

// ================================================================ repeat_qk

/// Simulates the production sequence: conv1d writes a conv tile (per-row
/// slices) and repeat_qk then reads the whole tile in the same pass.
#[test]
fn repeat_qk_sees_convd_writes_in_same_pass() {
    let mut ctx = Ctx::new();
    // DIMS: conv_dim=192 (row stride 768 B) and exp row stride 256 B are
    // both multiples of wgpu's 256 B min_storage_buffer_offset_alignment.
    let (n_k, n_v, kd, vd) = (2usize, 2usize, 16usize, 64usize);
    let conv_dim = 2 * n_k * kd + n_v * vd;
    let rows = 3usize;
    let mut rng = Rng(101);
    let w = rng.fill(conv_dim * 4);
    let mut state = rng.fill(conv_dim * 3);
    let x: Vec<Vec<f32>> = (0..rows).map(|_| rng.fill(conv_dim)).collect();

    let wb = ctx.f32(&w, &[conv_dim as u32, 4]);
    let st = ctx.f32(&state, &[conv_dim as u32, 3]);
    let xb = ctx.zero(&[rows as u32, conv_dim as u32]);
    let convd = ctx.zero(&[rows as u32, conv_dim as u32]);
    let y = ctx.zero(&[rows as u32, (2 * n_v * kd) as u32]);

    // Stage the full input tile before the pass, like the gemm that feeds
    // conv1d in production.
    let flat: Vec<f32> = x.iter().flatten().copied().collect();
    ctx.backend.write_f32(&xb.buf, &flat);

    // Same pass: per-row conv1d writes, then a full-tile repeat_qk read.
    let mut enc = ctx.backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "k");
        let row = |t: usize| (t * conv_dim * 4) as u64;
        for t in 0..rows {
            ctx.backend
                .dispatch(
                    &mut pass,
                    "conv1d",
                    &[("DIM", conv_dim as f64)],
                    &[
                        Binding::Slice(&xb, row(t), conv_dim as u64 * 4),
                        Binding::Full(&wb),
                        Binding::Full(&st),
                        Binding::Slice(&convd, row(t), conv_dim as u64 * 4),
                    ],
                    [1, 1, 1],
                )
                .unwrap();
        }
        // Reference: conv each row sequentially, then repeat_qk the tile.
        let mut conv_cpu = Vec::with_capacity(rows * conv_dim);
        let mut st2 = state.clone();
        for xrow in &x {
            conv_cpu.extend(cpu::conv1d(xrow, &w, &mut st2));
        }
        ctx.backend
            .dispatch(
                &mut pass,
                "repeat_qk",
                &[
                    ("ROWS", rows as f64),
                    ("N_K", n_k as f64),
                    ("N_V", n_v as f64),
                    ("K_DIM", kd as f64),
                    ("RATIO", (n_v / n_k) as f64),
                    ("CONV_DIM", conv_dim as f64),
                ],
                &[
                    Binding::Slice(&convd, 0, (rows * conv_dim * 4) as u64),
                    Binding::Full(&y),
                ],
                [rows as u32, 1, 1],
            )
            .unwrap();
    }
    ctx.backend.submit(enc);

    // Reference: conv each row sequentially, then repeat_qk the tile.
    let mut conv_cpu = Vec::with_capacity(rows * conv_dim);
    for xrow in &x {
        conv_cpu.extend(cpu::conv1d(xrow, &w, &mut state));
    }
    let mut cpu_y = vec![0f32; rows * 2 * n_v * kd];
    cpu::repeat_qk(&conv_cpu, &mut cpu_y, rows, n_k, n_v, kd, vd);
    agree(&ctx.read(&y), &cpu_y, 1e-5, 1e-6);
}

fn repeat_qk_case(rows: usize, n_k: usize, n_v: usize, kd: usize, vd: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let conv_dim = 2 * n_k * kd + n_v * vd;
    let x = rng.fill(rows * conv_dim);
    let xb = ctx.f32(&x, &[rows as u32, conv_dim as u32]);
    let out_dim = 2 * n_v * kd;
    let y = ctx.zero(&[rows as u32, out_dim as u32]);

    ctx.dispatch(
        "repeat_qk",
        &[
            ("ROWS", rows as f64),
            ("N_K", n_k as f64),
            ("N_V", n_v as f64),
            ("K_DIM", kd as f64),
            ("RATIO", (n_v / n_k) as f64),
            ("CONV_DIM", conv_dim as f64),
        ],
        &[Binding::Full(&xb), Binding::Full(&y)],
        [rows as u32, 1, 1],
    );

    let mut cpu_y = vec![0f32; rows * out_dim];
    cpu::repeat_qk(&x, &mut cpu_y, rows, n_k, n_v, kd, vd);
    agree(&ctx.read(&y), &cpu_y, 0.0, 1e-6);
}

/// Exact production dims of the toy Qwen35 model (conv 384, exp 256).
#[test]
fn repeat_qk_production_dims() {
    repeat_qk_case(2, 16, 16, 8, 8, 211);
}

#[test]
fn repeat_qk_identity_ratio_one() {
    repeat_qk_case(3, 4, 4, 8, 16, 91);
}

#[test]
fn repeat_qk_expands_key_heads() {
    repeat_qk_case(2, 2, 4, 8, 16, 93);
    repeat_qk_case(3, 1, 3, 16, 8, 97);
}

#[test]
fn anchor_repeat_qk() {
    // One row, q=[1,2] k=[3,4] v=[5,6,7] at key-head width; value heads
    // double the q/k segments (repeat_interleave) and leave v untouched.
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut y = vec![0f32; 8];
    cpu::repeat_qk(&x, &mut y, 1, 1, 2, 2, 3);
    assert_eq!(y, [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
}

// ================================================================ delta_gate

#[test]
fn delta_gate_selects_chunk_rows() {
    let mut ctx = Ctx::new();
    let (rows, heads) = (3usize, 4usize);
    let mut rng = Rng(71);
    let b = rng.fill(rows * heads);
    let a = rng.fill(rows * heads);
    let alog = rng.fill(heads).iter().map(|v| v.abs()).collect::<Vec<_>>();
    let dt = rng.fill(heads);
    let bb = ctx.f32(&b, &[rows as u32, heads as u32]);
    let ab = ctx.f32(&a, &[rows as u32, heads as u32]);
    let lb = ctx.f32(&alog, &[heads as u32]);
    let db = ctx.f32(&dt, &[heads as u32]);
    let beta = ctx.zero(&[heads as u32]);
    let g = ctx.zero(&[heads as u32]);

    for row in [0usize, 2] {
        ctx.dispatch(
            "delta_gate",
            &[("HEADS", heads as f64), ("ROW_T", row as f64)],
            &[
                Binding::Full(&bb),
                Binding::Full(&ab),
                Binding::Full(&lb),
                Binding::Full(&db),
                Binding::Full(&beta),
                Binding::Full(&g),
            ],
            [1, 1, 1],
        );
        let (cb, cg) = cpu::delta_gate(
            &b[row * heads..(row + 1) * heads],
            &a[row * heads..(row + 1) * heads],
            &alog,
            &dt,
        );
        agree(&ctx.read(&beta), &cb, 0.0, 1e-5);
        agree(&ctx.read(&g), &cg, 0.0, 1e-5);
    }
}

// ================================================================ delta_recur

fn delta_recur_case(heads: usize, kd: usize, vd: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let q = rng.fill(heads * kd);
    let k = rng.fill(heads * kd);
    let v = rng.fill(heads * vd);
    let beta = rng.fill(heads).iter().map(|x| x.abs()).collect::<Vec<_>>();
    let g = rng.fill(heads);
    let s0 = rng.fill(heads * kd * vd);
    let qb = ctx.f32(&q, &[heads as u32, kd as u32]);
    let kb = ctx.f32(&k, &[heads as u32, kd as u32]);
    let vb = ctx.f32(&v, &[heads as u32, vd as u32]);
    let beb = ctx.f32(&beta, &[heads as u32]);
    let gb = ctx.f32(&g, &[heads as u32]);
    let st = ctx.f32(&s0, &[heads as u32, kd as u32, vd as u32]);
    let out = ctx.zero(&[heads as u32, vd as u32]);

    ctx.dispatch(
        "delta_recur",
        &[
            ("HEADS", heads as f64),
            ("K_DIM", kd as f64),
            ("V_DIM", vd as f64),
        ],
        &[
            Binding::Full(&qb),
            Binding::Full(&kb),
            Binding::Full(&vb),
            Binding::Full(&beb),
            Binding::Full(&gb),
            Binding::Full(&st),
            Binding::Full(&out),
        ],
        [heads as u32, 1, 1],
    );
    let mut state = s0.clone();
    let cpu = cpu::delta_recur(&q, &k, &v, &beta, &g, &mut state, heads, kd, vd);
    agree(&ctx.read(&st), &state, 1e-4, 1e-5);
    agree(&ctx.read(&out), &cpu, 1e-4, 1e-5);
}

#[test]
fn delta_recur_symmetric_dims() {
    delta_recur_case(2, 8, 8, 81);
}

#[test]
fn delta_recur_asymmetric_dims() {
    delta_recur_case(1, 16, 8, 83);
}

// ================================================================ attn
// The split-K GQA attention (per-segment kernel + merge) is the production
// path; it must match the CPU reference on GQA, sliding windows, empty
// segments (short prefixes) and 8:1 GQA ratios.

fn attn_case(
    m: usize,
    nq: usize,
    nkv: usize,
    max_seq: usize,
    pos: usize,
    window: usize,
    seed: u64,
) {
    let mut ctx = Ctx::new();
    let hd = 64usize;
    let mut rng = Rng(seed);
    let q = rng.fill(m * nq * hd);
    let kc = rng
        .fill(nkv * max_seq * hd)
        .iter()
        .map(|v| v * 0.1)
        .collect::<Vec<_>>();
    let vc = rng.fill(nkv * max_seq * hd);
    let qb = ctx.f32(&q, &[m as u32, nq as u32, hd as u32]);
    let kb = ctx.bf16(&kc, &[nkv as u32, max_seq as u32, hd as u32]);
    let vb = ctx.bf16(&vc, &[nkv as u32, max_seq as u32, hd as u32]);
    let y = ctx.zero(&[m as u32, nq as u32, hd as u32]);
    let scratch = ctx.zero(&[m as u32, nkv as u32, 32, 8, hd as u32 + 2]);
    let args = ctx.arg(pos);

    ctx.dispatch(
        "attn",
        &[
            ("N_HEADS", nq as f64),
            ("KV_HEADS", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("MAX_SEQ", max_seq as f64),
            ("SCALE", 1.0 / (hd as f64).sqrt()),
            ("WINDOW", window as f64),
            ("NQ_PER_KV", (nq / nkv) as f64),
            ("STRIDE", (hd as u32 + 2) as f64),
        ],
        &[
            Binding::Full(&qb),
            Binding::Full(&kb),
            Binding::Full(&vb),
            Binding::Full(&scratch),
            Binding::Full(&args),
        ],
        [m as u32, nkv as u32, 32],
    );
    ctx.dispatch(
        "merge_attn",
        &[
            ("N_HEADS", nq as f64),
            ("KV_HEADS", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("STRIDE", (hd as u32 + 2) as f64),
        ],
        &[Binding::Full(&scratch), Binding::Full(&y)],
        [m as u32, nkv as u32, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu::attn(&q, &kc, &vc, m, nq, nkv, hd, max_seq, pos, window),
        1e-2,
        1e-2,
    );
}

#[test]
fn attn_gqa_multi_row() {
    attn_case(3, 2, 1, 16, 5, 0, 91);
}

#[test]
fn attn_sliding_window() {
    // A 3-wide window over a prefix of 6..8 keys exercises the masked head/tail.
    attn_case(3, 2, 1, 16, 5, 3, 93);
}

#[test]
fn attn_single_query_per_head() {
    attn_case(1, 2, 2, 4, 0, 0, 95);
}

#[test]
fn attn_gqa_4x_multi_chunk() {
    attn_case(16, 4, 1, 64, 16, 0, 97);
}

#[test]
fn attn_split_short_prefix() {
    // pos=2 with 32 segments: most segments are empty.
    attn_case(1, 2, 1, 16, 2, 0, 101);
}

#[test]
fn attn_split_long_prefix() {
    attn_case(16, 4, 1, 1024, 512, 0, 103);
}

#[test]
fn attn_split_sliding_window() {
    attn_case(3, 2, 1, 1024, 700, 128, 105);
}

#[test]
fn attn_gqa_8x() {
    attn_case(2, 8, 1, 64, 10, 0, 107);
}

// ================================================================ layout

#[test]
fn split_qg() {
    let mut ctx = Ctx::new();
    let (rows, heads, hd) = (2usize, 2usize, 4usize);
    let mut rng = Rng(101);
    let x = rng.fill(rows * heads * 2 * hd);
    let xb = ctx.f32(&x, &[rows as u32, heads as u32, 2 * hd as u32]);
    let qb = ctx.zero(&[rows as u32, heads as u32, hd as u32]);
    let gb = ctx.zero(&[rows as u32, heads as u32, hd as u32]);

    ctx.dispatch(
        "split_qg",
        &[
            ("ROWS", rows as f64),
            ("HEADS", heads as f64),
            ("HD", hd as f64),
        ],
        &[Binding::Full(&xb), Binding::Full(&qb), Binding::Full(&gb)],
        [1, 1, 1],
    );
    let (cq, cg) = cpu::split_qg(&x, rows, heads, hd);
    agree(&ctx.read(&qb), &cq, 0.0, 1e-7);
    agree(&ctx.read(&gb), &cg, 0.0, 1e-7);
}

#[test]
fn kv_store_multi_row() {
    let mut ctx = Ctx::new();
    let (m, nkv, hd, max_seq, pos) = (3usize, 2usize, 4usize, 8usize, 3usize);
    let mut rng = Rng(111);
    let src = rng.fill(m * nkv * hd);
    let sb = ctx.f32(&src, &[m as u32, nkv as u32, hd as u32]);
    let cache = ctx.zero_bf16(&[nkv as u32, max_seq as u32, hd as u32]);
    let args = ctx.arg(pos);

    ctx.dispatch(
        "kv_store",
        &[
            ("N_KV", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("MAX_SEQ", max_seq as f64),
        ],
        &[
            Binding::Full(&sb),
            Binding::Full(&cache),
            Binding::Full(&args),
        ],
        [1, m as u32, 1],
    );
    let mut cpu_cache = vec![0f32; nkv * max_seq * hd];
    cpu::kv_store(&src, &mut cpu_cache, m, nkv, hd, max_seq, pos);
    agree(&ctx.read_bf16(&cache), &cpu_cache, 0.0, 1e-7);
}

// ================================================================ anchors
// Hand-computed cases pinning the CPU reference to independent arithmetic.

#[test]
fn anchor_gemm() {
    assert_eq!(
        cpu::gemm(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2),
        vec![17.0, 23.0, 39.0, 53.0]
    );
}

#[test]
fn anchor_embed() {
    let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(
        cpu::embed(&[2, 0], &table, 2, 2.0),
        vec![10.0, 12.0, 2.0, 4.0]
    );
}

#[test]
fn anchor_norm_modes() {
    let inv = (2.5f32 + 1e-6).sqrt().recip(); // RMS of [1, 2]
    let x = [1.0f32, 2.0];

    let direct = cpu::norm(2, &x, &[2.0, 3.0], &[], 1, 2, 2, 1e-6);
    assert_eq!(direct, vec![inv * 2.0, inv * 6.0]);

    let offset = cpu::norm(0, &x, &[0.5, -0.25], &[], 1, 2, 2, 1e-6);
    assert_eq!(offset, vec![inv * 1.5, inv * 1.5]);

    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let gated = cpu::norm(1, &x, &[2.0, 3.0], &[0.0, 1.0], 1, 2, 2, 1e-6);
    assert_eq!(gated[0], 0.0, "silu(0) gates the first element to zero");
    assert!((gated[1] - inv * 6.0 * silu1).abs() < 1e-6);
}

#[test]
fn anchor_elementwise() {
    assert_eq!(cpu::add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);

    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    cpu::bias(&mut x, &[10.0, 20.0], 2);
    assert_eq!(x, [11.0, 22.0, 13.0, 24.0]);

    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let swi = cpu::swiglu(&[0.0, 1.0], &[2.0, 3.0], 0);
    assert_eq!(swi[0], 0.0);
    assert!((swi[1] - silu1 * 3.0).abs() < 1e-6);

    let sm = cpu::sigmoid_mul(&[2.0, -1.0], &[0.0, 1.0]);
    assert!((sm[0] - 1.0).abs() < 1e-6);
    assert!((sm[1] + silu1).abs() < 1e-6);
}

#[test]
fn anchor_layout_ops() {
    assert_eq!(
        cpu::concat(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );

    let x: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let (q, g) = cpu::split_qg(&x, 1, 2, 2);
    assert_eq!(q, vec![0.0, 1.0, 4.0, 5.0]);
    assert_eq!(g, vec![2.0, 3.0, 6.0, 7.0]);

    let mut cache = vec![0.0f32; 8];
    cpu::kv_store(&[1.0, 2.0, 3.0, 4.0], &mut cache, 2, 1, 2, 4, 1);
    assert_eq!(cache, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
}

#[test]
fn anchor_rope() {
    // One head, full rotation of hd=4 at position 1, theta = 1e4.
    let inv1 = 1.0 / 10000f64.powf(2.0 / 4.0);
    let (c0, s0) = (1.0f64.cos() as f32, 1.0f64.sin() as f32);
    let (c1, s1) = (inv1.cos() as f32, inv1.sin() as f32);
    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    cpu::rope(
        &mut x,
        &[0.0, 0.0, c0, c1],
        &[0.0, 0.0, s0, s1],
        1,
        1,
        4,
        4,
        1,
    );
    let want = [
        1.0 * c0 - 3.0 * s0,
        2.0 * c1 - 4.0 * s1,
        3.0 * c0 + 1.0 * s0,
        4.0 * c1 + 2.0 * s1,
    ];
    for (got, want) in x.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

#[test]
fn anchor_attn_causal_and_window() {
    let k = [1.0, 0.0, 0.0, 1.0]; // two keys at t=0,1
    let v = [1.0, 2.0, 3.0, 4.0];
    let q = [1.0, 1.0];

    // pos=1 attends to both keys; equal scores -> mean of the values.
    let full = cpu::attn(&q, &k, &v, 1, 1, 1, 2, 2, 1, 0);
    assert!((full[0] - 2.0).abs() < 1e-6);
    assert!((full[1] - 3.0).abs() < 1e-6);

    // Window 1 keeps only the newest key -> its value verbatim.
    let win = cpu::attn(&q, &k, &v, 1, 1, 1, 2, 2, 1, 1);
    assert!((win[0] - 3.0).abs() < 1e-6);
    assert!((win[1] - 4.0).abs() < 1e-6);
}

#[test]
fn anchor_conv1d() {
    let mut state = [1.0f32, 1.0, 1.0];
    let y = cpu::conv1d(&[2.0], &[1.0, 2.0, 3.0, 4.0], &mut state);
    let v = 14.0f32;
    assert!((y[0] - v / (1.0 + (-v).exp())).abs() < 1e-5);
    assert_eq!(state, [1.0, 1.0, 2.0], "state shifts to [s1, s2, x]");
}

#[test]
fn anchor_delta_gate() {
    let (beta, g) = cpu::delta_gate(&[0.0], &[0.0], &[2.0f32.ln()], &[0.0]);
    assert!((beta[0] - 0.5).abs() < 1e-6);
    assert!(
        (g[0] + 2.0 * 2.0f32.ln()).abs() < 1e-5,
        "-exp(ln2) * softplus(0)"
    );
}

#[test]
fn anchor_delta_recur_two_steps() {
    let mut state = [0.0f32; 4];
    let r2 = 2.0f32.sqrt();

    // Step 1: axis-aligned q/k make the norms trivial; zero state means the
    // delta rule writes beta * (k-hat outer v) straight in.
    let out = cpu::delta_recur(
        &[4.0, 0.0],
        &[3.0, 0.0],
        &[5.0, 7.0],
        &[0.5],
        &[0.0],
        &mut state,
        1,
        2,
        2,
    );
    for (got, want) in state.iter().zip([2.5, 3.5, 0.0, 0.0]) {
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
    assert!((out[0] - 2.5 / r2).abs() < 1e-5);
    assert!((out[1] - 3.5 / r2).abs() < 1e-5);

    // Step 2: decay halves the state; k-hat = [0, 1] writes into row 1.
    let out = cpu::delta_recur(
        &[0.0, 4.0],
        &[0.0, 2.0],
        &[1.0, 1.0],
        &[1.0],
        &[-2.0f32.ln()],
        &mut state,
        1,
        2,
        2,
    );
    assert!((state[0] - 1.25).abs() < 1e-5);
    assert!((state[1] - 1.75).abs() < 1e-5);
    assert!((state[2] - 1.0).abs() < 1e-5);
    assert!((state[3] - 1.0).abs() < 1e-5);
    assert!((out[0] - 1.0 / r2).abs() < 1e-5);
    assert!((out[1] - 1.0 / r2).abs() < 1e-5);
}



