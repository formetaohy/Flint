use std::sync::{Mutex, MutexGuard};

use flint_backend::{Backend, Binding, Commands};
use flint_kernel::{Act, NormMode, shader};

mod support;
use flint_tensor::{DType, Tensor, Weight};
use support::cpu_ref;

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

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
        self.backend.zero_tensor(shape, DType::F32)
    }

    fn zero_bf16(&self, shape: &[u32]) -> Tensor {
        self.backend.zero_tensor(shape, DType::Bf16)
    }

    fn rows(&self, pos: usize, m: usize) -> Tensor {
        let t = Tensor::new(
            self.backend.storage(8 * m as u64 * 4),
            vec![8 * m as u32],
            DType::U32,
        );
        let mut data = vec![0u32; 8 * m];
        for i in 0..m {
            data[8 * i] = pos as u32 + i as u32;
            data[8 * i + 1] = 0;
        }
        self.backend.write_u32(&t.buf, &data);
        t
    }

    fn block_table(&self, max_seq: usize) -> Tensor {
        let pages = max_seq.div_ceil(flint_kernel::PAGE_LEN as usize);
        let t = Tensor::new(
            self.backend.storage(pages as u64 * 4),
            vec![pages as u32],
            DType::U32,
        );
        self.backend
            .write_u32(&t.buf, &(0..pages as u32).collect::<Vec<_>>());
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
        shader::GEMM,
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
    let _g = gpu();
    gemm_case(WType::Bf16, 16, 64, 128, 7);
}

struct GemmCase<'a> {
    x: &'a Tensor,
    w: &'a Weight,
    y: &'a Tensor,
    m: u32,
    n: u32,
    k: u32,
    acc: bool,
}

fn coop_gemm(ctx: &mut Ctx, case: &GemmCase<'_>) {
    let (x, w, y, m, n, k, acc) = (case.x, case.w, case.y, case.m, case.n, case.k, case.acc);
    let xf = ctx.backend.zero_tensor(&[m * k], DType::F16);
    ctx.dispatch(
        shader::TO_F16,
        &[("N_ELEM", (m * k) as f64)],
        &[Binding::Full(x), Binding::Full(&xf)],
        [(m * k / 4).div_ceil(256), 1, 1],
    );
    let unit = ctx.f32(&[1.0], &[1]);
    let scale = match w.scale() {
        Some(s) => Binding::Full(s),
        None => Binding::Full(&unit),
    };
    let wdtype = if w.scale().is_some() { 1.0 } else { 0.0 };
    let group = w.group().unwrap_or(128) as f64;
    ctx.dispatch(
        shader::GEMM_COOP,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("WDTYPE", wdtype),
            ("GROUP", group),
            ("ACC", acc as u32 as f64),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xf),
            Binding::Full(w.tensor()),
            scale,
            Binding::Full(y),
        ],
        [n.div_ceil(128), m.div_ceil(128), 1],
    );
}

#[test]
fn gemm_coop_bf16() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(7);
    let (m, n, k) = (128usize, 128usize, 256usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let weight = Weight::plain(wb);
    coop_gemm(
        &mut ctx,
        &GemmCase {
            x: &xb,
            w: &weight,
            y: &y,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            acc: false,
        },
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
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(71);
    let (m, n, k, group) = (128usize, 128usize, 256usize, 32usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let (wq, scales) = quant(&w, n, k, group);
    let cpu_w = dequant(&wq, &scales, n, k, group);
    let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
    let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
    let weight = Weight::quant(wb, sb, group as u32);
    coop_gemm(
        &mut ctx,
        &GemmCase {
            x: &xb,
            w: &weight,
            y: &y,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            acc: false,
        },
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &cpu_w, m, n, k),
        1e-2,
        1e-2,
    );
}

#[test]
fn gemm_coop_bf16_acc() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(13);
    let (m, n, k) = (128usize, 128usize, 256usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let y0 = rng.fill(m * n);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.f32(&y0, &[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let weight = Weight::plain(wb);
    coop_gemm(
        &mut ctx,
        &GemmCase {
            x: &xb,
            w: &weight,
            y: &y,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            acc: true,
        },
    );
    let mut expect = cpu_ref::gemm(&x, &bf16_round(&w), m, n, k);
    for i in 0..expect.len() {
        expect[i] += y0[i];
    }
    agree(&ctx.read(&y), &expect, 2e-2, 5e-2);
}

#[test]
fn gemm_coop_bf16_multi_tile() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(29);
    let (m, n, k) = (128usize, 256usize, 1024usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let weight = Weight::plain(wb);
    coop_gemm(
        &mut ctx,
        &GemmCase {
            x: &xb,
            w: &weight,
            y: &y,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            acc: false,
        },
    );

    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &bf16_round(&w), m, n, k),
        2e-2,
        5e-2,
    );
}

#[test]
fn gemm_acc_coop_full() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(31);
    let (m, n, k, group) = (128usize, 128usize, 256usize, 32usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let (wq, scales) = quant(&w, n, k, group);
    let cpu_w = dequant(&wq, &scales, n, k, group);
    let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
    let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
    let weight = Weight::quant(wb, sb, group as u32);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut commands = Commands::begin(&mut enc);
        ctx.backend
            .gemm_acc(
                &mut commands,
                Binding::Full(&xb),
                &weight,
                Binding::Full(&y),
                m as u32,
                false,
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &cpu_w, m, n, k),
        1e-2,
        1e-2,
    );
}

#[test]
fn gemm_acc_coop_tail() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(37);
    let (m, n, k, group) = (144usize, 128usize, 256usize, 32usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let y0 = rng.fill(m * n);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.f32(&y0, &[m as u32, n as u32]);
    let (wq, scales) = quant(&w, n, k, group);
    let cpu_w = dequant(&wq, &scales, n, k, group);
    let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
    let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
    let weight = Weight::quant(wb, sb, group as u32);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut commands = Commands::begin(&mut enc);
        ctx.backend
            .gemm_acc(
                &mut commands,
                Binding::Full(&xb),
                &weight,
                Binding::Full(&y),
                m as u32,
                false,
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &cpu_w, m, n, k),
        1e-2,
        1e-2,
    );
}

#[test]
fn gemm_acc_coop_tail_accumulates() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(39);
    let (m, n, k, group) = (144usize, 128usize, 256usize, 32usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let y0 = rng.fill(m * n);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.f32(&y0, &[m as u32, n as u32]);
    let (wq, scales) = quant(&w, n, k, group);
    let cpu_w = dequant(&wq, &scales, n, k, group);
    let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
    let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
    let weight = Weight::quant(wb, sb, group as u32);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut commands = Commands::begin(&mut enc);
        ctx.backend
            .gemm_acc(
                &mut commands,
                Binding::Full(&xb),
                &weight,
                Binding::Full(&y),
                m as u32,
                true,
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    let mut expect = cpu_ref::gemm(&x, &cpu_w, m, n, k);
    for i in 0..expect.len() {
        expect[i] += y0[i];
    }
    agree(&ctx.read(&y), &expect, 1e-2, 1e-2);
}

#[test]
fn gemm_acc_coop_long_k() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(41);
    let (m, n, k, group) = (128usize, 128usize, 8192usize, 32usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let (wq, scales) = quant(&w, n, k, group);
    let cpu_w = dequant(&wq, &scales, n, k, group);
    let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
    let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
    let weight = Weight::quant(wb, sb, group as u32);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut commands = Commands::begin(&mut enc);
        ctx.backend
            .gemm_acc(
                &mut commands,
                Binding::Full(&xb),
                &weight,
                Binding::Full(&y),
                m as u32,
                false,
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &cpu_w, m, n, k),
        2e-2,
        3e-2,
    );
}

#[test]
fn gemm_classic_segs_long_k() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let mut rng = Rng(43);
    let (m, n, k, group) = (64usize, 64usize, 8192usize, 32usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let (wq, scales) = quant(&w, n, k, group);
    let cpu_w = dequant(&wq, &scales, n, k, group);
    let wb = ctx.backend.tensor_i8(&wq, vec![n as u32, k as u32]);
    let sb = ctx.f32(&scales, &[n as u32, (k / group) as u32]);
    let weight = Weight::quant(wb, sb, group as u32);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut commands = Commands::begin(&mut enc);
        ctx.backend
            .gemm_acc(
                &mut commands,
                Binding::Full(&xb),
                &weight,
                Binding::Full(&y),
                m as u32,
                false,
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &cpu_w, m, n, k),
        1e-2,
        1e-2,
    );
}

#[test]
fn gemm_bf16_multi_tile_m() {
    let _g = gpu();
    gemm_case(WType::Bf16, 32, 32, 64, 29);
}

#[test]
fn gemm_i8_group128() {
    let _g = gpu();
    gemm_case(WType::I8(128), 16, 256, 256, 17);
}

#[test]
fn gemm_i8_group64() {
    let _g = gpu();
    gemm_case(WType::I8(64), 16, 64, 960, 19);
}

#[test]
fn gemm_i8_group32() {
    let _g = gpu();
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
        shader::GEMV,
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
            shader::MERGE_GEMV,
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
    let _g = gpu();
    gemv_case(WType::Bf16, 32, 128, 31, 1);
}

#[test]
fn gemv_i8_group128() {
    let _g = gpu();
    gemv_case(WType::I8(128), 64, 256, 37, 1);
}

#[test]
fn gemv_i8_partial_chunk() {
    let _g = gpu();
    gemv_case(WType::I8(32), 32, 192, 41, 1);
}

#[test]
fn gemv_i8_split4() {
    let _g = gpu();
    gemv_case(WType::I8(128), 64, 512, 43, 4);
}

#[test]
fn gemv_bf16_split8() {
    let _g = gpu();
    gemv_case(WType::Bf16, 32, 1024, 47, 8);
}

#[test]
fn gemv_bf16_split_segs_divide_k_blocks() {
    let _g = gpu();
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
        shader::EMBED,
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
    let _g = gpu();
    embed_case(4, 16, 1.0, 11);
}

#[test]
fn embed_gemma_scale() {
    let _g = gpu();
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
        shader::NORM,
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
        &cpu_ref::norm(
            mode,
            &x,
            &w,
            &z,
            cpu_ref::NormArgs {
                rows,
                dim,
                w_dim,
                eps: 1e-6,
            },
        ),
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
    let _g = gpu();
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
    let args = ctx.rows(pos, rows / heads);

    ctx.dispatch(
        shader::NORM,
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
    let _g = gpu();
    norm_case(NormMode::Layer, 4, 64, 64, 29);
}

#[test]
fn norm_offset() {
    let _g = gpu();
    norm_case(NormMode::Offset, 3, 64, 64, 21);
}

#[test]
fn norm_gated_weight_repeats_across_row() {
    let _g = gpu();
    norm_case(NormMode::Gated, 4, 64, 8, 31);
}

#[test]
fn norm_direct() {
    let _g = gpu();
    norm_case(NormMode::Direct, 2, 32, 32, 27);
}

#[test]
fn add() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(33);
    let a = rng.fill(n);
    let b = rng.fill(n);
    let ab = ctx.f32(&a, &[n as u32]);
    let bb = ctx.f32(&b, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        shader::ADD,
        &[("N_ELEM", n as f64)],
        &[Binding::Full(&ab), Binding::Full(&bb), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::add(&a, &b), 0.0, 0.0);
}

#[test]
fn bias() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let (rows, dim) = (3usize, 16usize);
    let mut rng = Rng(35);
    let x = rng.fill(rows * dim);
    let b = rng.fill(dim);
    let xb = ctx.f32(&x, &[rows as u32, dim as u32]);
    let bb = ctx.f32(&b, &[dim as u32]);

    ctx.dispatch(
        shader::BIAS,
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
    let _g = gpu();
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(41);
    let g = rng.fill(n);
    let u = rng.fill(n);
    let gb = ctx.f32(&g, &[n as u32]);
    let ub = ctx.f32(&u, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        shader::SWIGLU,
        &[("N_ELEM", n as f64), ("MODE", 0.0)],
        &[Binding::Full(&gb), Binding::Full(&ub), Binding::Full(&y)],
        [1, 1, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::swiglu(&g, &u, Act::Silu),
        1e-5,
        1e-6,
    );
}

#[test]
fn swiglu_gelu_tanh() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(47);
    let g = rng.fill(n);
    let u = rng.fill(n);
    let gb = ctx.f32(&g, &[n as u32]);
    let ub = ctx.f32(&u, &[n as u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        shader::SWIGLU,
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
    let _g = gpu();
    let mut ctx = Ctx::new();
    let n = 100usize;
    let mut rng = Rng(53);
    let mut x = rng.fill(n);
    for v in x.iter_mut() {
        *v *= 40.0;
    }
    let xb = ctx.f32(&x, &[n as u32]);
    ctx.dispatch(
        shader::SOFTCAP,
        &[("N_ELEM", n as f64), ("CAP", 30.0)],
        &[Binding::Full(&xb)],
        [1, 1, 1],
    );
    cpu_ref::softcap(&mut x, 30.0);
    agree(&ctx.read(&xb), &x, 1e-5, 1e-6);
}

#[test]
fn mul_broadcast() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let n = 160usize;
    let mut rng = Rng(59);
    let a = rng.fill(n);
    let b = rng.fill(4);
    let ab = ctx.f32(&a, &[n as u32]);
    let bb = ctx.f32(&b, &[4u32]);
    let y = ctx.zero(&[n as u32]);

    ctx.dispatch(
        shader::MUL,
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
fn concat() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let (rows, d) = (2usize, 8usize);
    let mut rng = Rng(37);
    let a = rng.fill(rows * d);
    let b = rng.fill(rows * d);
    let ab = ctx.f32(&a, &[rows as u32, d as u32]);
    let bb = ctx.f32(&b, &[rows as u32, d as u32]);
    let y = ctx.zero(&[rows as u32, 2 * d as u32]);

    ctx.dispatch(
        shader::CONCAT,
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
    let args = ctx.rows(pos, m);

    ctx.dispatch(
        shader::ROPE,
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
    cpu_ref::rope(
        &mut cpu,
        &cos,
        &sin,
        cpu_ref::RopeArgs {
            m,
            heads,
            hd,
            rot,
            pos,
        },
    );
    agree(&ctx.read(&xb), &cpu, 1e-5, 1e-6);
}

#[test]
fn rope_partial_rotation_multi_row() {
    let _g = gpu();
    rope_case(2, 2, 32, 16, 2, 51);
}

#[test]
fn rope_full_rotation() {
    let _g = gpu();
    rope_case(1, 1, 32, 32, 0, 53);
}

#[test]
fn kv_store_writes_both_caches() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    let (m, nkv, hd, max_seq, pos) = (3usize, 2usize, 4usize, 8usize, 3usize);
    let mut rng = Rng(111);
    let k_src = rng.fill(m * nkv * hd);
    let v_src = rng.fill(m * nkv * hd);
    let kb = ctx.f32(&k_src, &[m as u32, nkv as u32, hd as u32]);
    let vb = ctx.f32(&v_src, &[m as u32, nkv as u32, hd as u32]);
    let k_cache = ctx.zero_bf16(&[nkv as u32, max_seq as u32, hd as u32]);
    let v_cache = ctx.zero_bf16(&[nkv as u32, max_seq as u32, hd as u32]);
    let args = ctx.rows(pos, m);
    let bt = ctx.block_table(max_seq);

    ctx.dispatch(
        shader::KV_STORE,
        &[
            ("N_KV", nkv as f64),
            ("HEAD_DIM", hd as f64),
            ("POOL_LEN", max_seq as f64),
            (
                "MAX_PAGES",
                max_seq.div_ceil(flint_kernel::PAGE_LEN as usize) as f64,
            ),
        ],
        &[
            Binding::Full(&kb),
            Binding::Full(&vb),
            Binding::Full(&k_cache),
            Binding::Full(&v_cache),
            Binding::Full(&args),
            Binding::Full(&bt),
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
    let _g = gpu();
    assert_eq!(
        cpu_ref::gemm(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2),
        vec![17.0, 23.0, 39.0, 53.0]
    );
}

#[test]
fn anchor_embed() {
    let _g = gpu();
    let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(
        cpu_ref::embed(&[2, 0], &table, 2, 2.0),
        vec![10.0, 12.0, 2.0, 4.0]
    );
}

#[test]
fn anchor_norm_modes() {
    let _g = gpu();
    let inv = (2.5f32 + 1e-6).sqrt().recip();
    let x = [1.0f32, 2.0];

    let direct = cpu_ref::norm(
        NormMode::Direct,
        &x,
        &[2.0, 3.0],
        &[],
        cpu_ref::NormArgs {
            rows: 1,
            dim: 2,
            w_dim: 2,
            eps: 1e-6,
        },
    );
    assert_eq!(direct, vec![inv * 2.0, inv * 6.0]);

    let offset = cpu_ref::norm(
        NormMode::Offset,
        &x,
        &[0.5, -0.25],
        &[],
        cpu_ref::NormArgs {
            rows: 1,
            dim: 2,
            w_dim: 2,
            eps: 1e-6,
        },
    );
    assert_eq!(offset, vec![inv * 1.5, inv * 1.5]);

    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let gated = cpu_ref::norm(
        NormMode::Gated,
        &x,
        &[2.0, 3.0],
        &[0.0, 1.0],
        cpu_ref::NormArgs {
            rows: 1,
            dim: 2,
            w_dim: 2,
            eps: 1e-6,
        },
    );
    assert_eq!(gated[0], 0.0, "silu(0) gates the first element to zero");
    assert!((gated[1] - inv * 6.0 * silu1).abs() < 1e-6);
}

#[test]
fn anchor_elementwise() {
    let _g = gpu();
    assert_eq!(cpu_ref::add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);

    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    cpu_ref::bias(&mut x, &[10.0, 20.0], 2);
    assert_eq!(x, [11.0, 22.0, 13.0, 24.0]);

    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let swi = cpu_ref::swiglu(&[0.0, 1.0], &[2.0, 3.0], Act::Silu);
    assert_eq!(swi[0], 0.0);
    assert!((swi[1] - silu1 * 3.0).abs() < 1e-6);

}

#[test]
fn anchor_layout_ops() {
    let _g = gpu();
    assert_eq!(
        cpu_ref::concat(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );

    let mut cache = vec![0.0f32; 8];
    cpu_ref::kv_store(&[1.0, 2.0, 3.0, 4.0], &mut cache, 2, 1, 2, 4, 1);
    assert_eq!(cache, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
}

#[test]
fn anchor_rope() {
    let _g = gpu();
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
    let _g = gpu();
    let k = [1.0, 0.0, 0.0, 1.0];
    let v = [1.0, 2.0, 3.0, 4.0];
    let q = [1.0, 1.0];

    let full = cpu_ref::attn(
        &q,
        &k,
        &v,
        cpu_ref::AttnArgs {
            m: 1,
            nq: 1,
            nkv: 1,
            hd: 2,
            max_seq: 2,
            pos: 1,
            window: 0,
            causal: true,
        },
    );
    assert!((full[0] - 2.0).abs() < 1e-6);
    assert!((full[1] - 3.0).abs() < 1e-6);

    let win = cpu_ref::attn(
        &q,
        &k,
        &v,
        cpu_ref::AttnArgs {
            m: 1,
            nq: 1,
            nkv: 1,
            hd: 2,
            max_seq: 2,
            pos: 1,
            window: 1,
            causal: true,
        },
    );
    assert!((win[0] - 3.0).abs() < 1e-6);
    assert!((win[1] - 4.0).abs() < 1e-6);
}

#[test]
fn gemv_bf16_wide() {
    let _g = gpu();
    gemv_case(WType::Bf16, 256, 2560, 99, 1);
}

#[test]
fn gemm_i8_9728() {
    let _g = gpu();
    gemm_case(WType::I8(128), 16, 2560, 9728, 77);
}
