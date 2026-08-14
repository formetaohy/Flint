use flint_backend::{Backend, Binding, Commands};
use flint_kernel::{Act, NormMode, name};

mod support;
use support::cpu_ref;
use flint_tensor::{DType, Tensor, Weight};

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
        self.backend.tensor_f32(data, shape.to_vec())
    }

    fn bf16(&self, data: &[f32], shape: &[u32]) -> Tensor {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        self.backend.tensor_bf16(&bytes, shape.to_vec()).unwrap()
    }

    fn zero(&self, shape: &[u32]) -> Tensor {
        self.backend.zero_tensor(shape)
    }

    fn zero_bf16(&self, shape: &[u32]) -> Tensor {
        self.backend.zero_bf16_tensor(shape)
    }

    fn arg(&self, pos: usize, kv_len: usize) -> Tensor {
        let segs = kv_len.div_ceil(256).clamp(1, 32);
        let t = Tensor::new(self.backend.storage(8), vec![2], DType::U32);
        self.backend
            .write_u32(&t.buf, &[pos as u32, segs as u32]);
        t
    }

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
        let mut enc = self.backend.encoder().unwrap();
        {
            let mut pass = Commands::begin(&mut enc);
            self.backend
                .dispatch(&mut pass, name, consts, bufs, groups)
                .unwrap();
        }
        self.backend.submit(&mut enc).unwrap();
    }

    fn read(&self, t: &Tensor) -> Vec<f32> {
        self.backend
            .read_f32(&t.buf, 0, t.numel() as usize)
            .unwrap()
    }
}

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

fn quant(data: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    let mut row_major = Vec::new();
    let mut scales = vec![0f32; rows * (cols / group)];
    for r in 0..rows {
        for g in 0..cols / group {
            let block = &data[r * cols + g * group..r * cols + (g + 1) * group];
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales[g * rows + r] = scale;
            for v in block {
                row_major.push((v / scale).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
    }
    let mut bytes = vec![0u8; rows * cols];
    for kb in 0..cols / 16 {
        for r in 0..rows {
            for i in 0..16 {
                bytes[(kb * rows + r) * 16 + i] = row_major[r * cols + kb * 16 + i];
            }
        }
    }
    (bytes, scales)
}

fn dequant(bytes: &[u8], scales: &[f32], rows: usize, cols: usize, group: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for kb in 0..cols / 16 {
        for r in 0..rows {
            for i in 0..16 {
                let byte = bytes[(kb * rows + r) * 16 + i];
                let g = (kb * 16) / group;
                out[r * cols + kb * 16 + i] = (byte as i8) as f32 * scales[g * rows + r];
            }
        }
    }
    out
}

enum WType {
    Bf16,
    I8(usize),
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
            let cpu_w = dequant(&wq, &scales, n, k, group);
            let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
            let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
            (wb, sb, 1.0, group as f64, cpu_w, 1e-4, 1e-3)
        }
    };

    ctx.dispatch(
        name::GEMM,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("WDTYPE", wdtype),
            ("GROUP", group),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&sb),
            Binding::Full(&y),
        ],
        [n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::gemm(&x, &cpu_w, m, n, k), rel, abs);
}

#[test]
fn gemm_bf16() {
    gemm_case(WType::Bf16, 16, 64, 128, 7);
}

#[test]
fn gemm_coop_bf16() {
    let mut ctx = Ctx::new();
    let mut rng = Rng(7);
    let (m, n, k) = (16usize, 64usize, 128usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let sb = ctx.f32(&[0.0], &[1]);
    ctx.dispatch(
        name::GEMM_COOP,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("WDTYPE", 0.0),
            ("GROUP", 128.0),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&sb),
            Binding::Full(&y),
        ],
        [n.div_ceil(32) as u32, m.div_ceil(64) as u32, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &bf16_round(&w), m, n, k),
        2e-2,
        5e-2,
    );
}

#[test]
fn gemm_coop_i8_group32() {
    let mut ctx = Ctx::new();
    let mut rng = Rng(71);
    let (m, n, k, group) = (16usize, 64usize, 128usize, 32usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let (wq, scales) = quant(&w, n, k, group);
    let cpu_w = dequant(&wq, &scales, n, k, group);
    let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
    let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
    ctx.dispatch(
        name::GEMM_COOP,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("WDTYPE", 1.0),
            ("GROUP", group as f64),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&sb),
            Binding::Full(&y),
        ],
        [n.div_ceil(32) as u32, m.div_ceil(64) as u32, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::gemm(&x, &cpu_w, m, n, k), 1e-2, 1e-2);
}

#[test]
fn gemm_coop_bf16_segs() {
    let mut ctx = Ctx::new();
    let mut rng = Rng(77);
    let (m, n, k) = (16usize, 64usize, 8192usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let sb = ctx.f32(&[0.0], &[1]);
    let segs = 4u32;
    let partial = ctx.zero(&[segs * m as u32 * n as u32]);
    ctx.dispatch(
        name::GEMM_COOP,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", segs as f64),
            ("WDTYPE", 0.0),
            ("GROUP", 128.0),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&sb),
            Binding::Full(&partial),
        ],
        [n.div_ceil(32) as u32, m.div_ceil(64) as u32, segs],
    );
    ctx.dispatch(
        name::MERGE_GEMM,
        &[
            ("M", m as f64),
            ("N", n as f64),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
            ("SEGS", segs as f64),
            ("ACC", 0.0),
        ],
        &[Binding::Full(&partial), Binding::Full(&y)],
        [n.div_ceil(256) as u32, 1, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &bf16_round(&w), m, n, k),
        2e-2,
        5e-2,
    );
}

#[test]
fn gemm_coop_bf16_multi_tile() {
    let mut ctx = Ctx::new();
    let mut rng = Rng(29);
    let (m, n, k) = (80usize, 64usize, 256usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let sb = ctx.f32(&[0.0], &[1]);
    ctx.dispatch(
        name::GEMM_COOP,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("WDTYPE", 0.0),
            ("GROUP", 128.0),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&sb),
            Binding::Full(&y),
        ],
        [n.div_ceil(32) as u32, m.div_ceil(64) as u32, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &bf16_round(&w), m, n, k),
        2e-2,
        5e-2,
    );
}

#[test]
fn gemm_bf16_multi_tile_m() {
    gemm_case(WType::Bf16, 32, 32, 64, 29);
}

#[test]
fn gemm_i8_group128() {
    gemm_case(WType::I8(128), 16, 256, 256, 17);
}

#[test]
fn gemm_i8_group64() {
    gemm_case(WType::I8(64), 16, 64, 960, 19);
}

#[test]
fn gemm_i8_group32() {
    gemm_case(WType::I8(32), 16, 32, 192, 23);
}

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
            let cpu_w = dequant(&wq, &scales, n, k, group);
            let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
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
        name::GEMV,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", wdtype),
            ("GROUP", group),
            ("SEGS", segs as f64),
            ("ACC", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&sb),
            out,
        ],
        [(n.div_ceil(64)) as u32, segs, 1],
    );
    if segs > 1 {
        ctx.dispatch(
            name::MERGE_GEMV,
            &[("N", n as f64), ("SEGS", segs as f64), ("ACC", 0.0)],
            &[
                Binding::Slice(&partial, 0, n as u64 * 4 * segs as u64),
                Binding::Full(&y),
            ],
            [(n.div_ceil(256)) as u32, 1, 1],
        );
    }
    agree(&ctx.read(&y), &cpu_ref::gemv(&x, &cpu_w, n, k), rel, abs);
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

#[test]
fn gemv_bf16_split_segs_divide_k_blocks() {
    let k = 1088u32;
    let n = 5120u32;
    let mut ctx = Ctx::new();
    let mut rng = Rng(51);
    let x = rng.fill(k as usize);
    let w = rng.fill((n * k) as usize);
    let xb = ctx.f32(&x, &[k]);
    let wb = ctx.bf16(&w, &[n, k]);
    let wt = Weight::plain(wb);
    let y = ctx.zero(&[n]);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut pass = Commands::begin(&mut enc);
        ctx.backend
            .gemv(&mut pass, Binding::Full(&xb), &wt, Binding::Full(&y))
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    agree(
        &ctx.read(&y),
        &cpu_ref::gemv(&x, &w, n as usize, k as usize),
        2e-2,
        5e-2,
    );
}

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
        ctx.backend.storage(rows as u64 * 4),
        vec![rows as u32],
        DType::F32,
    );
    ctx.backend.write_u32(&ib.buf, &ids);
    let y = ctx.zero(&[16u32, dim as u32]);
    let fallback = ctx.f32(&[1.0], &[1]);
    let w = Weight::plain(tb);

    ctx.dispatch(
        name::EMBED,
        &[
            ("M", rows as f64),
            ("DIM", dim as f64),
            ("SCALE", scale as f64),
            ("WDTYPE", 0.0),
            ("GROUP", 128.0),
            ("SPLIT", u32::MAX as f64),
            ("ROWS", vocab as f64),
        ],
        &[
            Binding::Full(&ib),
            Binding::Full(w.tensor()),
            Binding::Full(&fallback),
            Binding::Full(&fallback),
            Binding::Full(&y),
        ],
        [(rows * dim).div_ceil(256) as u32, 1, 1],
    );
    let got = ctx.read(&y);
    agree(
        &got[..rows * dim],
        &cpu_ref::embed(&ids, &bf16_round(&table), dim, scale),
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

fn norm_case(mode: NormMode, rows: usize, dim: usize, w_dim: usize, seed: u64) {
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
        name::NORM,
        &[
            ("MODE", mode as u32 as f64),
            ("DIM", dim as f64),
            ("W_DIM", w_dim as f64),
            ("EPS", 1e-6),
            ("HEADS", 1.0),
            ("ROT", 2.0),
            ("COS_STRIDE", 1.0),
            ("STRIDE", dim as f64),
            ("PLE", 0.0),
            ("PLE_LAYERS", 0.0),
            ("PLE_STRIDE", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&zb),
            Binding::Full(&y),
            Binding::Full(&zb),
            Binding::Full(&zb),
            Binding::Full(&zb),
        ],
        [rows as u32, 1, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::norm(mode, &x, &w, &z, cpu_ref::NormArgs { rows, dim, w_dim, eps: 1e-6 }),
        1e-4,
        1e-5,
    );
}

struct NormRopeArgs {
    rows: usize,
    dim: usize,
    rot: usize,
    heads: usize,
    pos: usize,
    stride: usize,
    eps: f32,
}

fn norm_rope_cpu(x: &[f32], w: &[f32], cos: &[f32], sin: &[f32], spec: NormRopeArgs) -> Vec<f32> {
    let (rows, dim, rot, heads, pos, stride, eps) = (
        spec.rows,
        spec.dim,
        spec.rot,
        spec.heads,
        spec.pos,
        spec.stride,
        spec.eps,
    );
    let half = rot / 2;
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = (mean_sq + eps).sqrt().recip();
        let mut normed: Vec<f32> = (0..dim).map(|d| row[d] * inv * w[d]).collect();
        let pos_m = pos + r / heads;
        let orig = normed.clone();
        for d in 0..rot {
            let t = pos_m * stride + d % half;
            normed[d] = if d < half {
                orig[d] * cos[t] - orig[d + half] * sin[t]
            } else {
                orig[d] * cos[t] + orig[d - half] * sin[t]
            };
        }
        out[r * dim..(r + 1) * dim].copy_from_slice(&normed);
    }
    out
}

#[test]
fn norm_rope_mode4_pos72_repro() {
    let (rows, dim, rot, heads, pos) = (32usize, 128usize, 128usize, 16usize, 72usize);
    let mut ctx = Ctx::new();
    let mut rng = Rng(1234);
    let x = rng.fill(rows * dim);
    let w = rng.fill(dim);
    let half = rot / 2;
    let mut cos = Vec::with_capacity(4096 * half);
    let mut sin = Vec::with_capacity(4096 * half);
    for p in 0..4096usize {
        for i in 0..half {
            let inv = 1.0 / 1_000_000f64.powf((2 * i) as f64 / dim as f64);
            let angle = p as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    let xb = ctx.f32(&x, &[rows as u32, dim as u32]);
    let wb = ctx.f32(&w, &[dim as u32]);
    let y = ctx.zero(&[rows as u32, dim as u32]);
    let cb = ctx.f32(&cos, &[4096, half as u32]);
    let sb = ctx.f32(&sin, &[4096, half as u32]);
    let args = ctx.arg(pos, pos + 1);

    ctx.dispatch(
        name::NORM,
        &[
            ("MODE", 4.0),
            ("DIM", dim as f64),
            ("W_DIM", dim as f64),
            ("EPS", 1e-6),
            ("HEADS", heads as f64),
            ("ROT", rot as f64),
            ("COS_STRIDE", half as f64),
            ("STRIDE", dim as f64),
            ("PLE", 0.0),
            ("PLE_LAYERS", 0.0),
            ("PLE_STRIDE", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(&wb),
            Binding::Full(&xb),
            Binding::Full(&y),
            Binding::Full(&cb),
            Binding::Full(&sb),
            Binding::Full(&args),
        ],
        [rows as u32, 1, 1],
    );
    let got = ctx.read(&y);
    let want = norm_rope_cpu(
        &x,
        &w,
        &cos,
        &sin,
        NormRopeArgs {
            rows,
            dim,
            rot,
            heads,
            pos,
            stride: half,
            eps: 1e-6,
        },
    );
    agree(&got, &want, 1e-3, 1e-4);
    eprintln!("nan in gpu: {}", got.iter().filter(|v| v.is_nan()).count());
    eprintln!("nan in cpu: {}", want.iter().filter(|v| v.is_nan()).count());
}

#[test]
fn norm_layer() {
    norm_case(NormMode::Layer, 4, 64, 64, 29);
}

#[test]
fn norm_offset() {
    norm_case(NormMode::Offset, 3, 64, 64, 21);
}

#[test]
fn norm_gated_weight_repeats_across_row() {
    norm_case(NormMode::Gated, 4, 64, 8, 31);
}

#[test]
fn norm_direct() {
    norm_case(NormMode::Direct, 2, 32, 32, 27);
}

#[test]
fn add() {
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(33);
    let a = rng.fill(n);
    let b = rng.fill(n);
    let ab = ctx.f32(&a, &[n as u32]);
    let bb = ctx.f32(&b, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        name::ADD,
        &[("N_ELEM", n as f64)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::add(&a, &b), 0.0, 0.0);
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
        name::BIAS,
        &[("N_ELEM", (rows * dim) as f64), ("DIM", dim as f64)],
        &[Binding::Full(&xb), Binding::Full(&bb)],
        [1, 1, 1],
    );
    let mut cpu = x.clone();
    cpu_ref::bias(&mut cpu, &b, dim);
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
        name::SWIGLU,
        &[("N_ELEM", n as f64), ("MODE", 0.0)],
        &[Binding::Full(&gb), Binding::Full(&ub), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::swiglu(&g, &u, Act::Silu), 1e-5, 1e-6);
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
        name::SWIGLU,
        &[("N_ELEM", n as f64), ("MODE", 1.0)],
        &[Binding::Full(&gb), Binding::Full(&ub), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::swiglu(&g, &u, Act::GeluTanh),
        1e-5,
        1e-6,
    );
}

#[test]
fn softcap() {
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(53);
    let mut x = rng.fill(n);
    for v in x.iter_mut() {
        *v *= 40.0;
    }
    let xb = ctx.f32(&x, &[n as u32]);
    ctx.dispatch(
        name::SOFTCAP,
        &[("N_ELEM", n as f64), ("CAP", 30.0)],
        &[Binding::Full(&xb)],
        [1, 1, 1],
    );
    cpu_ref::softcap(&mut x, 30.0);
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
    let bb = ctx.f32(&b, &[4u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        name::MUL,
        &[
            ("N", n as f64),
            ("M", 4.0),
            ("MODE", 0.0),
            ("STRIDE", 0.0),
            ("OFFSET", 0.0),
        ],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::mul(&a, &b, n, 4), 1e-6, 1e-7);
}

#[test]
fn expert_gather_scatter() {
    let mut ctx = Ctx::new();
    let hidden = 32usize;
    let mut rng = Rng(61);

    let x = rng.fill(4 * hidden);
    let rows: Vec<u32> = vec![1, 3, 0, 2];
    let weights = [0.5f32, 1.5, 2.5, 3.5];
    let xb = ctx.f32(&x, &[4u32, hidden as u32]);
    let rb = Tensor::new(ctx.backend.storage(16), vec![4u32], DType::U32);
    ctx.backend.write_u32(&rb.buf, &rows);
    let wb = ctx.f32(&weights, &[4u32]);
    let packed = ctx.zero(&[16u32, hidden as u32]);
    let acc = ctx.zero(&[4u32, hidden as u32]);

    ctx.dispatch(
        name::EXPERT_GATHER,
        &[("HIDDEN", hidden as f64), ("COUNT", 4.0)],
        &[
            Binding::Full(&xb),
            Binding::Slice(&rb, 0, 16),
            Binding::Full(&packed),
        ],
        [1, 1, 1],
    );
    ctx.dispatch(
        name::EXPERT_SCATTER,
        &[("HIDDEN", hidden as f64), ("COUNT", 4.0)],
        &[
            Binding::Full(&acc),
            Binding::Full(&packed),
            Binding::Full(&rb),
            Binding::Full(&wb),
        ],
        [1, 1, 1],
    );
    let gathered = cpu_ref::expert_gather(&x, &rows, 16, hidden);
    let mut expect = vec![0f32; 4 * hidden];
    cpu_ref::expert_scatter(&mut expect, &gathered, &rows, &weights, hidden);
    agree(&ctx.read(&acc), &expect, 1e-6, 1e-7);
}

#[test]
fn zero_rows() {
    let mut ctx = Ctx::new();
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let xb = ctx.f32(&x, &[4u32]);
    ctx.dispatch(
        name::ZERO_ROWS,
        &[("N_ELEM", 3.0)],
        &[Binding::Full(&xb)],
        [1, 1, 1],
    );
    let mut expect = x.clone();
    cpu_ref::zero_rows(&mut expect, 3);
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
        name::SIGMOID_MUL,
        &[("N_ELEM", n as f64)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::sigmoid_mul(&a, &b), 1e-5, 1e-6);
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
        name::CONCAT,
        &[("ROWS", rows as f64), ("D", d as f64)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::concat(&a, &b, rows, d), 0.0, 0.0);
}

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
    let args = ctx.arg(pos, pos + m);

    ctx.dispatch(
        name::ROPE,
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
    cpu_ref::rope(&mut cpu, &cos, &sin, cpu_ref::RopeArgs { m, heads, hd, rot, pos });
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
            name::CONV1D,
            &[("DIM", dim as f64)],
            &[
                Binding::Full(&xb),
                Binding::Full(&wb),
                Binding::Full(&st),
                Binding::Full(&y),
            ],
            [1, 1, 1],
        );
        let cpu_y = cpu_ref::conv1d(x, &w, &mut state);
        agree(&ctx.read(&y), &cpu_y, 1e-5, 1e-6);
    }
    agree(&ctx.read(&st), &state, 0.0, 1e-6);
}

#[test]
fn repeat_qk_sees_convd_writes_in_same_pass() {
    let mut ctx = Ctx::new();

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
    let conv_out = ctx.zero(&[rows as u32, conv_dim as u32]);
    let y = ctx.zero(&[rows as u32, (2 * n_v * kd) as u32]);

    let flat: Vec<f32> = x.iter().flatten().copied().collect();
    ctx.backend.write_f32(&xb.buf, &flat);

    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut pass = Commands::begin(&mut enc);
        let row = |t: usize| (t * conv_dim * 4) as u64;
        for t in 0..rows {
            ctx.backend
                .dispatch(
                    &mut pass,
                    name::CONV1D,
                    &[("DIM", conv_dim as f64)],
                    &[
                        Binding::Slice(&xb, row(t), conv_dim as u64 * 4),
                        Binding::Full(&wb),
                        Binding::Full(&st),
                        Binding::Slice(&conv_out, row(t), conv_dim as u64 * 4),
                    ],
                    [1, 1, 1],
                )
                .unwrap();
        }

        let mut conv_cpu = Vec::with_capacity(rows * conv_dim);
        let mut st2 = state.clone();
        for xrow in &x {
            conv_cpu.extend(cpu_ref::conv1d(xrow, &w, &mut st2));
        }
        ctx.backend
            .dispatch(
                &mut pass,
                name::REPEAT_QK,
                &[
                    ("ROWS", rows as f64),
                    ("N_K", n_k as f64),
                    ("N_V", n_v as f64),
                    ("K_DIM", kd as f64),
                    ("RATIO", (n_v / n_k) as f64),
                    ("CONV_DIM", conv_dim as f64),
                ],
                &[
                    Binding::Slice(&conv_out, 0, (rows * conv_dim * 4) as u64),
                    Binding::Full(&y),
                ],
                [rows as u32, 1, 1],
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();

    let mut conv_cpu = Vec::with_capacity(rows * conv_dim);
    for xrow in &x {
        conv_cpu.extend(cpu_ref::conv1d(xrow, &w, &mut state));
    }
    let mut cpu_y = vec![0f32; rows * 2 * n_v * kd];
    cpu_ref::repeat_qk(&conv_cpu, &mut cpu_y, rows, n_k, n_v, kd, vd);
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
        name::REPEAT_QK,
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
    cpu_ref::repeat_qk(&x, &mut cpu_y, rows, n_k, n_v, kd, vd);
    agree(&ctx.read(&y), &cpu_y, 0.0, 1e-6);
}

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
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut y = vec![0f32; 8];
    cpu_ref::repeat_qk(&x, &mut y, 1, 1, 2, 2, 3);
    assert_eq!(y, [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
}

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
            name::DELTA_GATE,
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
        let (cb, cg) = cpu_ref::delta_gate(
            &b[row * heads..(row + 1) * heads],
            &a[row * heads..(row + 1) * heads],
            &alog,
            &dt,
        );
        agree(&ctx.read(&beta), &cb, 0.0, 1e-5);
        agree(&ctx.read(&g), &cg, 0.0, 1e-5);
    }
}

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
        name::DELTA_RECUR,
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
    let cpu = cpu_ref::delta_recur(&q, &k, &v, &beta, &g, &mut state, cpu_ref::DeltaRecurArgs { heads, kd, vd });
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

struct AttnCase {
    m: usize,
    nq: usize,
    nkv: usize,
    max_seq: usize,
    pos: usize,
    window: usize,
    seed: u64,
    k_scale: f32,
}

fn attn_case(spec: AttnCase) {
    let AttnCase {
        m,
        nq,
        nkv,
        max_seq,
        pos,
        window,
        seed,
        k_scale,
    } = spec;
    let hd = 64usize;
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let q = rng.fill(m * nq * hd);
    let kc = rng
        .fill(nkv * max_seq * hd)
        .iter()
        .map(|v| v * k_scale)
        .collect::<Vec<_>>();
    let vc = rng.fill(nkv * max_seq * hd);
    let qb = ctx.f32(&q, &[m as u32, nq as u32, hd as u32]);
    let kc = bf16_round(&kc);
    let vc = bf16_round(&vc);
    let kb = ctx.bf16(&kc, &[nkv as u32, max_seq as u32, hd as u32]);
    let vb = ctx.bf16(&vc, &[nkv as u32, max_seq as u32, hd as u32]);
    let y = ctx.zero(&[m as u32, nq as u32, hd as u32]);
    let scratch = ctx.zero(&[m as u32, nkv as u32, 32, 8, hd as u32 + 2]);
    let args = ctx.arg(pos, pos + m);

    ctx.dispatch(
        name::ATTN,
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
        name::MERGE_ATTN,
        &[
            ("N_HEADS", nq as f64),
            ("KV_HEADS", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("STRIDE", (hd as u32 + 2) as f64),
        ],
        &[
            Binding::Full(&scratch),
            Binding::Full(&y),
            Binding::Full(&args),
        ],
        [m as u32, nkv as u32, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::attn(&q, &kc, &vc, cpu_ref::AttnArgs { m, nq, nkv, hd, max_seq, pos, window }),
        1e-2,
        1e-2,
    );
}

#[test]
fn attn_decode_hd128() {
    for pos in [1usize, 63, 64, 65, 255, 256, 300, 511, 512, 1024] {
        let (m, nq, nkv, hd, max_seq) = (1usize, 32usize, 8usize, 128usize, 2048usize);
        let mut ctx = Ctx::new();
        let mut rng = Rng(9000 + pos as u64);
        let q = rng.fill(m * nq * hd);
        let kc = bf16_round(&rng.fill(nkv * max_seq * hd));
        let vc = bf16_round(&rng.fill(nkv * max_seq * hd));
        let qb = ctx.f32(&q, &[m as u32, nq as u32, hd as u32]);
        let kb = ctx.bf16(&kc, &[nkv as u32, max_seq as u32, hd as u32]);
        let vb = ctx.bf16(&vc, &[nkv as u32, max_seq as u32, hd as u32]);
        let y = ctx.zero(&[m as u32, nq as u32, hd as u32]);
        let scratch = ctx.zero(&[m as u32, nkv as u32, 32, 8, hd as u32 + 2]);
        let args = ctx.arg(pos, pos + m);

        ctx.dispatch(
            name::ATTN_DECODE,
            &[
                ("N_HEADS", nq as f64),
                ("KV_HEADS", nkv as f64),
                ("HEAD_DIM", hd as f64),
                ("MAX_SEQ", max_seq as f64),
                ("SCALE", 1.0 / (hd as f64).sqrt()),
                ("WINDOW", 0.0),
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
            [nkv as u32, 32, 1],
        );
        ctx.dispatch(
            name::MERGE_ATTN,
            &[
                ("N_HEADS", nq as f64),
                ("KV_HEADS", nkv as f64),
                ("HEAD_DIM", hd as f64),
                ("STRIDE", (hd as u32 + 2) as f64),
            ],
            &[
                Binding::Full(&scratch),
                Binding::Full(&y),
                Binding::Full(&args),
            ],
            [m as u32, nkv as u32, 1],
        );
        agree(
            &ctx.read(&y),
            &cpu_ref::attn(
                &q,
                &kc,
                &vc,
                cpu_ref::AttnArgs { m, nq, nkv, hd, max_seq, pos, window: 0 },
            ),
            1e-2,
            1e-2,
        );
    }
}

#[test]
fn kv_store_attn_hd128_gqa2_pos71() {
    let (m, nq, nkv, hd, max_seq, pos) = (2usize, 16usize, 8usize, 128usize, 4096usize, 71usize);
    let mut ctx = Ctx::new();
    let mut rng = Rng(4321);
    let k_src = rng.fill(m * nkv * hd);
    let v_src = rng.fill(m * nkv * hd);
    let q = rng.fill(m * nq * hd);
    let kc = vec![0f32; nkv * max_seq * hd];
    let vc = vec![0f32; nkv * max_seq * hd];
    let qb = ctx.f32(&q, &[m as u32, nq as u32, hd as u32]);
    let kb = ctx.f32(&k_src, &[m as u32, nkv as u32, hd as u32]);
    let vb = ctx.f32(&v_src, &[m as u32, nkv as u32, hd as u32]);
    let k_cache = ctx.zero_bf16(&[nkv as u32, max_seq as u32, hd as u32]);
    let v_cache = ctx.zero_bf16(&[nkv as u32, max_seq as u32, hd as u32]);
    let y = ctx.zero(&[m as u32, nq as u32, hd as u32]);
    let scratch = ctx.zero(&[m as u32, nkv as u32, 32, 8, hd as u32 + 2]);
    let args = ctx.arg(pos, pos + m);

    ctx.dispatch(
        name::KV_STORE,
        &[
            ("N_KV", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("MAX_SEQ", max_seq as f64),
        ],
        &[
            Binding::Full(&kb),
            Binding::Full(&vb),
            Binding::Full(&k_cache),
            Binding::Full(&v_cache),
            Binding::Full(&args),
        ],
        [(nkv * hd / 2) as u32, m as u32, 1],
    );
    ctx.dispatch(
        name::ATTN,
        &[
            ("N_HEADS", nq as f64),
            ("KV_HEADS", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("MAX_SEQ", max_seq as f64),
            ("SCALE", 1.0 / (hd as f64).sqrt()),
            ("WINDOW", 0.0),
            ("NQ_PER_KV", (nq / nkv) as f64),
            ("STRIDE", (hd as u32 + 2) as f64),
        ],
        &[
            Binding::Full(&qb),
            Binding::Full(&k_cache),
            Binding::Full(&v_cache),
            Binding::Full(&scratch),
            Binding::Full(&args),
        ],
        [m as u32, nkv as u32, 32],
    );
    ctx.dispatch(
        name::MERGE_ATTN,
        &[
            ("N_HEADS", nq as f64),
            ("KV_HEADS", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("STRIDE", (hd as u32 + 2) as f64),
        ],
        &[
            Binding::Full(&scratch),
            Binding::Full(&y),
            Binding::Full(&args),
        ],
        [m as u32, nkv as u32, 1],
    );
    let mut kc = kc;
    let mut vc = vc;
    for i in 0..m {
        for h in 0..nkv {
            for d in 0..hd {
                kc[(h * max_seq + pos + i) * hd + d] = k_src[(i * nkv + h) * hd + d];
                vc[(h * max_seq + pos + i) * hd + d] = v_src[(i * nkv + h) * hd + d];
            }
        }
    }
    let got = ctx.read(&y);
    let want = cpu_ref::attn(&q, &kc, &vc, cpu_ref::AttnArgs { m, nq, nkv, hd, max_seq, pos, window: 0 });
    agree(&got, &want, 1e-2, 1e-2);
    eprintln!("nan in gpu: {}", got.iter().filter(|v| v.is_nan()).count());
}

#[test]
fn attn_gqa_multi_row() {
    attn_case(AttnCase { m: 3, nq: 2, nkv: 1, max_seq: 16, pos: 5, window: 0, seed: 91, k_scale: 0.1 });
}

#[test]
fn attn_sliding_window() {
    attn_case(AttnCase { m: 3, nq: 2, nkv: 1, max_seq: 16, pos: 5, window: 3, seed: 93, k_scale: 0.1 });
}

#[test]
fn attn_single_query_per_head() {
    attn_case(AttnCase { m: 1, nq: 2, nkv: 2, max_seq: 4, pos: 0, window: 0, seed: 95, k_scale: 0.1 });
}

#[test]
fn attn_gqa_4x_multi_chunk() {
    attn_case(AttnCase { m: 16, nq: 4, nkv: 1, max_seq: 64, pos: 16, window: 0, seed: 97, k_scale: 0.1 });
}

#[test]
fn attn_split_short_prefix() {
    attn_case(AttnCase { m: 1, nq: 2, nkv: 1, max_seq: 16, pos: 2, window: 0, seed: 101, k_scale: 0.1 });
}

#[test]
fn attn_split_long_prefix() {
    attn_case(AttnCase { m: 16, nq: 4, nkv: 1, max_seq: 1024, pos: 512, window: 0, seed: 103, k_scale: 0.1 });
}

#[test]
fn attn_multi_round_boundaries() {
    for kv_len in [65, 127, 128, 191, 192, 255, 256, 257, 511, 512] {
        attn_case(AttnCase { m: 8, nq: 4, nkv: 1, max_seq: 1024, pos: kv_len, window: 0, seed: 2000 + kv_len as u64, k_scale: 1.0 });
    }
}

#[test]
fn attn_multi_round_tail() {
    for kv_len in [66, 129, 193, 258] {
        attn_case(AttnCase { m: 1, nq: 2, nkv: 1, max_seq: 1024, pos: kv_len, window: 0, seed: 3000 + kv_len as u64, k_scale: 1.0 });
    }
}

#[test]
fn attn_gemm_coexist() {
    let mut ctx = Ctx::new();
    let mut rng = Rng(2049);
    let (m, nq, nkv, max_seq, pos, hd) = (4usize, 4usize, 2usize, 256usize, 130usize, 64usize);
    let q = rng.fill(m * nq * hd);
    let kc = rng.fill(nkv * max_seq * hd);
    let vc = rng.fill(nkv * max_seq * hd);
    let qb = ctx.f32(&q, &[m as u32, nq as u32, hd as u32]);
    let kb = ctx.bf16(&kc, &[nkv as u32, max_seq as u32, hd as u32]);
    let vb = ctx.bf16(&vc, &[nkv as u32, max_seq as u32, hd as u32]);
    let y = ctx.zero(&[m as u32, nq as u32, hd as u32]);
    let scratch = ctx.zero(&[m as u32, nkv as u32, 32, 8, hd as u32 + 2]);
    let args = ctx.arg(pos, pos + m);

    let (k2, n2) = (96usize, 80usize);
    let x = rng.fill(m * k2);
    let w = rng.fill(n2 * k2);
    let xb = ctx.f32(&x, &[m as u32, k2 as u32]);
    let wb = ctx.bf16(&w, &[n2 as u32, k2 as u32]);
    let sb = ctx.f32(&[0.0], &[1]);
    let yg = ctx.zero(&[m as u32, n2 as u32]);

    for _ in 0..24 {
        let mut enc = ctx.backend.encoder().unwrap();
        {
            let mut pass = Commands::begin(&mut enc);
            ctx.backend
                .dispatch(
                    &mut pass,
                    name::ATTN,
                    &[
                        ("N_HEADS", nq as f64),
                        ("KV_HEADS", nkv as f64),
                        ("HEAD_DIM", hd as f64),
                        ("MAX_SEQ", max_seq as f64),
                        ("SCALE", 1.0 / (hd as f64).sqrt()),
                        ("WINDOW", 0.0),
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
                )
                .unwrap();
            ctx.backend
                .dispatch(
                    &mut pass,
                    name::MERGE_ATTN,
                    &[
                        ("N_HEADS", nq as f64),
                        ("KV_HEADS", nkv as f64),
                        ("HEAD_DIM", hd as f64),
                        ("STRIDE", (hd as u32 + 2) as f64),
                    ],
                    &[
                        Binding::Full(&scratch),
                        Binding::Full(&y),
                        Binding::Full(&args),
                    ],
                    [m as u32, nkv as u32, 1],
                )
                .unwrap();
            ctx.backend
                .dispatch(
                    &mut pass,
                    name::GEMM,
                    &[
                        ("N", n2 as f64),
                        ("K", k2 as f64),
                        ("M", m as f64),
                        ("SEGS", 1.0),
                        ("WDTYPE", 0.0),
                        ("GROUP", 128.0),
                        ("ACC", 0.0),
                        ("Y_STRIDE", n2 as f64),
                        ("Y_OFF", 0.0),
                    ],
                    &[
                        Binding::Full(&xb),
                        Binding::Full(&wb),
                        Binding::Full(&sb),
                        Binding::Full(&yg),
                    ],
                    [n2.div_ceil(128) as u32, m.div_ceil(64) as u32, 1],
                )
                .unwrap();
        }
        ctx.backend.submit(&mut enc).unwrap();
    }

    agree(
        &ctx.read(&y),
        &cpu_ref::attn(&q, &kc, &vc, cpu_ref::AttnArgs { m, nq, nkv, hd, max_seq, pos, window: 0 }),
        2e-3,
        1e-3,
    );
    agree(&ctx.read(&yg), &cpu_ref::gemm(&x, &w, m, n2, k2), 2e-2, 5e-2);
}

#[test]
fn attn_single_round_realistic() {
    for kv_len in [1, 32, 63, 64] {
        attn_case(AttnCase { m: 1, nq: 2, nkv: 1, max_seq: 1024, pos: kv_len, window: 0, seed: 4000 + kv_len as u64, k_scale: 1.0 });
    }
}

#[test]
fn attn_split_sliding_window() {
    attn_case(AttnCase { m: 3, nq: 2, nkv: 1, max_seq: 1024, pos: 700, window: 128, seed: 105, k_scale: 0.1 });
}

#[test]
fn attn_gqa_8x() {
    attn_case(AttnCase { m: 2, nq: 8, nkv: 1, max_seq: 64, pos: 10, window: 0, seed: 107, k_scale: 0.1 });
}

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
        name::SPLIT_QG,
        &[
            ("ROWS", rows as f64),
            ("HEADS", heads as f64),
            ("HD", hd as f64),
        ],
        &[Binding::Full(&xb), Binding::Full(&qb), Binding::Full(&gb)],
        [1, 1, 1],
    );
    let (cq, cg) = cpu_ref::split_qg(&x, rows, heads, hd);
    agree(&ctx.read(&qb), &cq, 0.0, 1e-7);
    agree(&ctx.read(&gb), &cg, 0.0, 1e-7);
}

#[test]
fn kv_store_writes_both_caches() {
    let mut ctx = Ctx::new();
    let (m, nkv, hd, max_seq, pos) = (3usize, 2usize, 4usize, 8usize, 3usize);
    let mut rng = Rng(111);
    let k_src = rng.fill(m * nkv * hd);
    let v_src = rng.fill(m * nkv * hd);
    let kb = ctx.f32(&k_src, &[m as u32, nkv as u32, hd as u32]);
    let vb = ctx.f32(&v_src, &[m as u32, nkv as u32, hd as u32]);
    let k_cache = ctx.zero_bf16(&[nkv as u32, max_seq as u32, hd as u32]);
    let v_cache = ctx.zero_bf16(&[nkv as u32, max_seq as u32, hd as u32]);
    let args = ctx.arg(pos, pos + m);

    ctx.dispatch(
        name::KV_STORE,
        &[
            ("N_KV", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("MAX_SEQ", max_seq as f64),
        ],
        &[
            Binding::Full(&kb),
            Binding::Full(&vb),
            Binding::Full(&k_cache),
            Binding::Full(&v_cache),
            Binding::Full(&args),
        ],
        [1, m as u32, 1],
    );
    let mut cpu_k = vec![0f32; nkv * max_seq * hd];
    let mut cpu_v = vec![0f32; nkv * max_seq * hd];
    cpu_ref::kv_store(&k_src, &mut cpu_k, m, nkv, hd, max_seq, pos);
    cpu_ref::kv_store(&v_src, &mut cpu_v, m, nkv, hd, max_seq, pos);
    agree(&ctx.read_bf16(&k_cache), &cpu_k, 0.0, 1e-7);
    agree(&ctx.read_bf16(&v_cache), &cpu_v, 0.0, 1e-7);
}

#[test]
fn anchor_gemm() {
    assert_eq!(
        cpu_ref::gemm(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2),
        vec![17.0, 23.0, 39.0, 53.0]
    );
}

#[test]
fn anchor_embed() {
    let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(
        cpu_ref::embed(&[2, 0], &table, 2, 2.0),
        vec![10.0, 12.0, 2.0, 4.0]
    );
}

#[test]
fn anchor_norm_modes() {
    let inv = (2.5f32 + 1e-6).sqrt().recip();
    let x = [1.0f32, 2.0];

    let direct = cpu_ref::norm(NormMode::Direct, &x, &[2.0, 3.0], &[], cpu_ref::NormArgs { rows: 1, dim: 2, w_dim: 2, eps: 1e-6 });
    assert_eq!(direct, vec![inv * 2.0, inv * 6.0]);

    let offset = cpu_ref::norm(NormMode::Offset, &x, &[0.5, -0.25], &[], cpu_ref::NormArgs { rows: 1, dim: 2, w_dim: 2, eps: 1e-6 });
    assert_eq!(offset, vec![inv * 1.5, inv * 1.5]);

    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let gated = cpu_ref::norm(NormMode::Gated, &x, &[2.0, 3.0], &[0.0, 1.0], cpu_ref::NormArgs { rows: 1, dim: 2, w_dim: 2, eps: 1e-6 });
    assert_eq!(gated[0], 0.0, "silu(0) gates the first element to zero");
    assert!((gated[1] - inv * 6.0 * silu1).abs() < 1e-6);
}

#[test]
fn anchor_elementwise() {
    assert_eq!(cpu_ref::add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);

    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    cpu_ref::bias(&mut x, &[10.0, 20.0], 2);
    assert_eq!(x, [11.0, 22.0, 13.0, 24.0]);

    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let swi = cpu_ref::swiglu(&[0.0, 1.0], &[2.0, 3.0], Act::Silu);
    assert_eq!(swi[0], 0.0);
    assert!((swi[1] - silu1 * 3.0).abs() < 1e-6);

    let sm = cpu_ref::sigmoid_mul(&[2.0, -1.0], &[0.0, 1.0]);
    assert!((sm[0] - 1.0).abs() < 1e-6);
    assert!((sm[1] + silu1).abs() < 1e-6);
}

#[test]
fn anchor_layout_ops() {
    assert_eq!(
        cpu_ref::concat(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );

    let x: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let (q, g) = cpu_ref::split_qg(&x, 1, 2, 2);
    assert_eq!(q, vec![0.0, 1.0, 4.0, 5.0]);
    assert_eq!(g, vec![2.0, 3.0, 6.0, 7.0]);

    let mut cache = vec![0.0f32; 8];
    cpu_ref::kv_store(&[1.0, 2.0, 3.0, 4.0], &mut cache, 2, 1, 2, 4, 1);
    assert_eq!(cache, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
}

#[test]
fn anchor_rope() {
    let inv1 = 1.0 / 10000f64.powf(2.0 / 4.0);
    let (c0, s0) = (1.0f64.cos() as f32, 1.0f64.sin() as f32);
    let (c1, s1) = (inv1.cos() as f32, inv1.sin() as f32);
    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    cpu_ref::rope(
        &mut x,
        &[0.0, 0.0, c0, c1],
        &[0.0, 0.0, s0, s1],
        cpu_ref::RopeArgs {
            m: 1,
            heads: 1,
            hd: 4,
            rot: 4,
            pos: 1,
        },
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
    let k = [1.0, 0.0, 0.0, 1.0];
    let v = [1.0, 2.0, 3.0, 4.0];
    let q = [1.0, 1.0];

    let full = cpu_ref::attn(&q, &k, &v, cpu_ref::AttnArgs { m: 1, nq: 1, nkv: 1, hd: 2, max_seq: 2, pos: 1, window: 0 });
    assert!((full[0] - 2.0).abs() < 1e-6);
    assert!((full[1] - 3.0).abs() < 1e-6);

    let win = cpu_ref::attn(&q, &k, &v, cpu_ref::AttnArgs { m: 1, nq: 1, nkv: 1, hd: 2, max_seq: 2, pos: 1, window: 1 });
    assert!((win[0] - 3.0).abs() < 1e-6);
    assert!((win[1] - 4.0).abs() < 1e-6);
}

#[test]
fn anchor_conv1d() {
    let mut state = [1.0f32, 1.0, 1.0];
    let y = cpu_ref::conv1d(&[2.0], &[1.0, 2.0, 3.0, 4.0], &mut state);
    let v = 14.0f32;
    assert!((y[0] - v / (1.0 + (-v).exp())).abs() < 1e-5);
    assert_eq!(state, [1.0, 1.0, 2.0], "state shifts to [s1, s2, x]");
}

#[test]
fn anchor_delta_gate() {
    let (beta, g) = cpu_ref::delta_gate(&[0.0], &[0.0], &[2.0f32.ln()], &[0.0]);
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

    let out = cpu_ref::delta_recur(
        &[4.0, 0.0],
        &[3.0, 0.0],
        &[5.0, 7.0],
        &[0.5],
        &[0.0],
        &mut state,
        cpu_ref::DeltaRecurArgs { heads: 1, kd: 2, vd: 2 },
    );
    for (got, want) in state.iter().zip([2.5, 3.5, 0.0, 0.0]) {
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
    assert!((out[0] - 2.5 / r2).abs() < 1e-5);
    assert!((out[1] - 3.5 / r2).abs() < 1e-5);

    let out = cpu_ref::delta_recur(
        &[0.0, 4.0],
        &[0.0, 2.0],
        &[1.0, 1.0],
        &[1.0],
        &[-2.0f32.ln()],
        &mut state,
        cpu_ref::DeltaRecurArgs { heads: 1, kd: 2, vd: 2 },
    );
    assert!((state[0] - 1.25).abs() < 1e-5);
    assert!((state[1] - 1.75).abs() < 1e-5);
    assert!((state[2] - 1.0).abs() < 1e-5);
    assert!((state[3] - 1.0).abs() < 1e-5);
    assert!((out[0] - 1.0 / r2).abs() < 1e-5);
    assert!((out[1] - 1.0 / r2).abs() < 1e-5);
}

#[test]
fn gemv_bf16_wide() {
    gemv_case(WType::Bf16, 256, 2560, 99, 1);
}

#[test]
fn gemm_i8_9728() {
    gemm_case(WType::I8(128), 16, 2560, 9728, 77);
}
