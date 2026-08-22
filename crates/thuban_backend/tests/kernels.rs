use std::sync::{Mutex, MutexGuard};

use thuban_backend::{Backend, Binding, Commands};
use thuban_kernel::{Act, NormMode, shader};

mod support;
use support::cpu_ref;
use thuban_tensor::{DType, Quant, Tensor, Weight};

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
        let pages = max_seq.div_ceil(thuban_kernel::PAGE_LEN as usize);
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
        &self,
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

const ALL_QUANTS: &[Quant] = &[
    Quant::Q4_0,
    Quant::Q4_1,
    Quant::Q5_0,
    Quant::Q5_1,
    Quant::Q8_0,
    Quant::Q2K,
    Quant::Q3K,
    Quant::Q4K,
    Quant::Q5K,
    Quant::Q6K,
    Quant::Q8K,
    Quant::Iq2Xxs,
    Quant::Iq2Xs,
    Quant::Iq3Xxs,
    Quant::Iq1S,
    Quant::Iq4Nl,
    Quant::Iq3S,
    Quant::Iq2S,
    Quant::Iq4Xs,
    Quant::Iq1M,
    Quant::Tq1_0,
    Quant::Tq2_0,
];

fn f16_offsets(quant: Quant) -> &'static [usize] {
    match quant {
        Quant::Q4_1 => &[0, 2],
        Quant::Q5_1 => &[0, 2],
        Quant::Q2K => &[80, 82],
        Quant::Q3K => &[108],
        Quant::Q4K => &[0, 2],
        Quant::Q5K => &[0, 2],
        Quant::Q6K => &[208],
        Quant::Tq1_0 => &[52],
        Quant::Tq2_0 => &[64],
        Quant::Q8K | Quant::Iq1M => &[],
        _ => &[0],
    }
}

fn synth_blocks(quant: Quant, numel: usize, seed: u64) -> Vec<u8> {
    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    let bb = quant.block_bytes();
    let blocks = numel.div_ceil(quant.block_len());
    let mut raw = vec![0u8; blocks * bb];
    let mut s = seed | 1;
    for b in 0..blocks {
        let off = b * bb;
        for i in 0..bb {
            raw[off + i] = (xorshift(&mut s) >> 33) as u8;
        }
        match quant {
            Quant::Q8K => {
                let d = 0x3b00_0000u32 | (xorshift(&mut s) as u32 & 0x007f_ffff);
                raw[off..off + 4].copy_from_slice(&d.to_le_bytes());
            }
            Quant::Iq1M => {
                let sc3 = (xorshift(&mut s) & 0x00ff) as u16 | 0x3000;
                raw[off + 54..off + 56].copy_from_slice(&sc3.to_le_bytes());
            }
            _ => {
                for &fo in f16_offsets(quant) {
                    let h = (xorshift(&mut s) & 0x9fff) % 0x1800;
                    raw[off + fo..off + fo + 2].copy_from_slice(&(h as u16).to_le_bytes());
                }
            }
        }
    }
    raw
}

fn quant_weight(ctx: &Ctx, quant: Quant, n: u32, k: u32, seed: u64) -> (Weight, Vec<f32>) {
    let numel = (n as usize) * (k as usize);
    let raw = synth_blocks(quant, numel, seed);
    let cpu = thuban_checkpoint::dequant::to_f32(quant, &raw, numel).unwrap();
    let padded = quant.pad_blocks(&raw, numel).unwrap();
    let wb = ctx.backend.tensor_quant(&padded, vec![n, k], quant);
    (Weight::quantized(wb), cpu)
}

fn qtype_of(w: &Weight) -> u32 {
    match w.tensor().dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Bf16 => 30,
        DType::Quant(q) => q.as_u32(),
        DType::U32 => unreachable!("weights are never index tensors"),
    }
}

fn gemm_case(quant: Quant, m: usize, n: usize, k: usize, seed: u64) {
    let ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(m * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let (weight, cpu_w) = quant_weight(&ctx, quant, n as u32, k as u32, seed ^ 0x9e37);

    ctx.dispatch(
        shader::GEMM,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("QTYPE", qtype_of(&weight) as f64),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(weight.tensor()),
            Binding::Full(ctx.backend.quant_lut()),
            Binding::Full(&y),
        ],
        [n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1],
    );
    agree(&ctx.read(&y), &cpu_ref::gemm(&x, &cpu_w, m, n, k), 1e-3, 1e-2);
}

#[test]
fn gemm_bf16() {
    let _g = gpu();
    let ctx = Ctx::new();
    let mut rng = Rng(7);
    let (m, n, k) = (16usize, 64usize, 128usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let weight = Weight::plain(wb);
    ctx.dispatch(
        shader::GEMM,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("QTYPE", qtype_of(&weight) as f64),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(weight.tensor()),
            Binding::Full(ctx.backend.quant_lut()),
            Binding::Full(&y),
        ],
        [n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &bf16_round(&w), m, n, k),
        2e-2,
        5e-2,
    );
}

#[test]
fn gemm_each_quant() {
    let _g = gpu();
    for (i, &q) in ALL_QUANTS.iter().enumerate() {
        eprintln!("quant {q:?}");
        gemm_case(q, 64, 128, 256, 1000 + i as u64);
    }
}

#[test]
fn gemm_bf16_multi_tile_m() {
    let _g = gpu();
    let ctx = Ctx::new();
    let mut rng = Rng(29);
    let (m, n, k) = (32usize, 32usize, 64usize);
    let x = rng.fill(m * k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.zero(&[m as u32, n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let weight = Weight::plain(wb);
    ctx.dispatch(
        shader::GEMM,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("QTYPE", qtype_of(&weight) as f64),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xb),
            Binding::Full(weight.tensor()),
            Binding::Full(ctx.backend.quant_lut()),
            Binding::Full(&y),
        ],
        [n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1],
    );
    agree(
        &ctx.read(&y),
        &cpu_ref::gemm(&x, &bf16_round(&w), m, n, k),
        2e-2,
        5e-2,
    );
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
    ctx.dispatch(
        shader::GEMM_COOP_M16,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("QTYPE", qtype_of(w) as f64),
            ("ACC", acc as u32 as f64),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
        &[
            Binding::Full(&xf),
            Binding::Full(w.tensor()),
            Binding::Full(ctx.backend.quant_lut()),
            Binding::Full(y),
        ],
        [n.div_ceil(128), m.div_ceil(128), 1],
    );
}

#[test]
fn gemm_coop_bf16() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    if ctx.backend.device().coop_gemm().is_none() {
        return;
    }
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
fn gemm_coop_each_quant() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    if ctx.backend.device().coop_gemm().is_none() {
        return;
    }
    for (i, &q) in ALL_QUANTS.iter().enumerate() {
        eprintln!("quant {q:?}");
        let mut rng = Rng(2000 + i as u64);
        let (m, n, k) = (128usize, 128usize, 256usize);
        let x = rng.fill(m * k);
        let xb = ctx.f32(&x, &[m as u32, k as u32]);
        let y = ctx.zero(&[m as u32, n as u32]);
        let (weight, cpu_w) = quant_weight(&ctx, q, n as u32, k as u32, 3000 + i as u64);
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
        agree(&ctx.read(&y), &cpu_ref::gemm(&x, &cpu_w, m, n, k), 1e-1, 2e-1);
    }
}

#[test]
fn gemm_coop_bf16_acc() {
    let _g = gpu();
    let mut ctx = Ctx::new();
    if ctx.backend.device().coop_gemm().is_none() {
        return;
    }
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
    if ctx.backend.device().coop_gemm().is_none() {
        return;
    }
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

fn gemm_acc_case(m: usize, n: usize, k: usize, acc: bool, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(m * k);
    let y0 = rng.fill(m * n);
    let xb = ctx.f32(&x, &[m as u32, k as u32]);
    let y = ctx.f32(&y0, &[m as u32, n as u32]);
    let (weight, cpu_w) = quant_weight(&ctx, Quant::Q8_0, n as u32, k as u32, seed ^ 0x1234);
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
                acc,
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    let mut expect = cpu_ref::gemm(&x, &cpu_w, m, n, k);
    if acc {
        for i in 0..expect.len() {
            expect[i] += y0[i];
        }
    }
    agree(&ctx.read(&y), &expect, 2e-2, 1e-1);
}

#[test]
fn gemm_acc_coop_full() {
    let _g = gpu();
    gemm_acc_case(128, 128, 256, false, 31);
}

#[test]
fn gemm_acc_coop_tail() {
    let _g = gpu();
    gemm_acc_case(144, 128, 256, false, 37);
}

#[test]
fn gemm_acc_coop_tail_accumulates() {
    let _g = gpu();
    gemm_acc_case(144, 128, 256, true, 39);
}

#[test]
fn gemm_acc_coop_long_k() {
    let _g = gpu();
    gemm_acc_case(128, 128, 8192, false, 41);
}

#[test]
fn gemm_classic_segs_long_k() {
    let _g = gpu();
    gemm_acc_case(64, 64, 8192, false, 43);
}

fn gemv_case(quant: Quant, n: usize, k: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(k);
    let xb = ctx.f32(&x, &[k as u32]);
    let y = ctx.zero(&[n as u32]);
    let (weight, cpu_w) = quant_weight(&ctx, quant, n as u32, k as u32, seed ^ 0xbeef);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut pass = Commands::begin(&mut enc);
        ctx.backend
            .gemv(
                &mut pass,
                Binding::Full(&xb),
                &[thuban_backend::GemvOp {
                    w: &weight,
                    y: Binding::Full(&y),
                    acc: false,
                }],
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    agree(&ctx.read(&y), &cpu_ref::gemv(&x, &cpu_w, n, k), 1e-3, 1e-2);
}

fn gemv_bf16_case(n: usize, k: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(k);
    let w = rng.fill(n * k);
    let xb = ctx.f32(&x, &[k as u32]);
    let y = ctx.zero(&[n as u32]);
    let wb = ctx.bf16(&w, &[n as u32, k as u32]);
    let weight = Weight::plain(wb);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut pass = Commands::begin(&mut enc);
        ctx.backend
            .gemv(
                &mut pass,
                Binding::Full(&xb),
                &[thuban_backend::GemvOp {
                    w: &weight,
                    y: Binding::Full(&y),
                    acc: false,
                }],
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    agree(
        &ctx.read(&y),
        &cpu_ref::gemv(&x, &bf16_round(&w), n, k),
        2e-2,
        5e-2,
    );
}

#[test]
fn gemv_bf16() {
    let _g = gpu();
    gemv_bf16_case(32, 128, 31);
}

#[test]
fn gemv_each_quant() {
    let _g = gpu();
    for (i, &q) in ALL_QUANTS.iter().enumerate() {
        gemv_case(q, 64, 256, 4000 + i as u64);
    }
}

#[test]
fn gemv_each_quant_partial_chunk() {
    let _g = gpu();
    for (i, &q) in ALL_QUANTS.iter().enumerate() {
        let k = if q.block_len() == 32 { 192 } else { 256 };
        gemv_case(q, 32, k, 5000 + i as u64);
    }
}

#[test]
fn gemv_each_quant_wide_k() {
    let _g = gpu();
    for (i, &q) in ALL_QUANTS.iter().enumerate() {
        gemv_case(q, 64, 1024, 6000 + i as u64);
    }
}

#[test]
fn gemv_bf16_wide_k() {
    let _g = gpu();
    gemv_bf16_case(32, 1024, 47);
}

#[test]
fn gemv_bf16_unaligned_k() {
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
            .gemv(
                &mut pass,
                Binding::Full(&xb),
                &[thuban_backend::GemvOp {
                    w: &wt,
                    y: Binding::Full(&y),
                    acc: false,
                }],
            )
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
    let ctx = Ctx::new();
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
        DType::U32,
    );
    ctx.backend.write_u32(&ib.buf, &ids);
    let y = ctx.zero(&[16u32, dim as u32]);
    let w = Weight::plain(tb);

    ctx.dispatch(
        shader::EMBED,
        &[
            ("M", rows as f64),
            ("DIM", dim as f64),
            ("SCALE", scale as f64),
            ("QTYPE", qtype_of(&w) as f64),
        ],
        &[
            Binding::Full(&ib),
            Binding::Full(w.tensor()),
            Binding::Full(ctx.backend.quant_lut()),
            Binding::Full(&y),
        ],
        [(rows * dim / 32).div_ceil(256) as u32, 1, 1],
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
fn embed_native_each_quant() {
    let _g = gpu();
    let ctx = Ctx::new();
    let vocab = 4usize;
    let dim = 256usize;
    let rows = 8usize;
    let mut rng = Rng(77);
    let _table = rng.fill(vocab * dim);
    let ids: Vec<u32> = (0..rows).map(|r| (r * 2 + 1) as u32 % vocab as u32).collect();
    let ib = Tensor::new(
        ctx.backend.storage(rows as u64 * 4),
        vec![rows as u32],
        DType::U32,
    );
    ctx.backend.write_u32(&ib.buf, &ids);
    let y = ctx.zero(&[16u32, dim as u32]);
    for (i, &q) in ALL_QUANTS.iter().enumerate() {
        let (w, cpu) = quant_weight(&ctx, q, vocab as u32, dim as u32, 7000 + i as u64);
        ctx.dispatch(
            shader::EMBED,
            &[
                ("M", rows as f64),
                ("DIM", dim as f64),
                ("SCALE", 1.0),
                ("QTYPE", qtype_of(&w) as f64),
            ],
            &[
                Binding::Full(&ib),
                Binding::Full(w.tensor()),
                Binding::Full(ctx.backend.quant_lut()),
                Binding::Full(&y),
            ],
            [(rows * dim / 32).div_ceil(256) as u32, 1, 1],
        );
        let got = ctx.read(&y);
        agree(
            &got[..rows * dim],
            &cpu_ref::embed(&ids, &cpu, dim, 1.0),
            1e-3,
            1e-2,
        );
    }
}

#[test]
fn embed_unit_scale() {
    let _g = gpu();
    embed_case(4, 32, 1.0, 11);
}

#[test]
fn embed_gemma_scale() {
    let _g = gpu();
    embed_case(3, 32, 4.0, 13);
}

fn norm_case(mode: NormMode, rows: usize, dim: usize, w_dim: usize, seed: u64) {
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
    let ctx = Ctx::new();
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
                max_seq.div_ceil(thuban_kernel::PAGE_LEN as usize) as f64,
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
    gemv_bf16_case(256, 2560, 99);
}

#[test]
fn gemm_q8_0_9728() {
    let _g = gpu();
    gemm_case(Quant::Q8_0, 16, 2560, 9728, 77);
}

#[test]
fn gemv_real_q2k_tensor_if_present() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../temp/q2k_gate2.bin");
    if !path.exists() {
        eprintln!("skipping: temp/q2k_gate.bin not present");
        return;
    }
    let _g = gpu();
    let mut ctx = Ctx::new();
    let raw = std::fs::read(path).unwrap();
    let (n, k) = (3584usize, 1024usize);
    let q = Quant::Q2K;
    let cpu_w = thuban_checkpoint::dequant::to_f32(q, &raw, n * k).unwrap();
    let padded = q.pad_blocks(&raw, n * k).unwrap();
    let wb = ctx.backend.tensor_quant(&padded, vec![n as u32, k as u32], q);
    let weight = Weight::quantized(wb);
    let mut rng = Rng(99);
    let x = rng.fill(k);
    let xb = ctx.f32(&x, &[k as u32]);
    let y = ctx.zero(&[n as u32]);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut pass = Commands::begin(&mut enc);
        ctx.backend
            .gemv(
                &mut pass,
                Binding::Full(&xb),
                &[thuban_backend::GemvOp {
                    w: &weight,
                    y: Binding::Full(&y),
                    acc: false,
                }],
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    let gpu = ctx.read(&y);
    let cpu = cpu_ref::gemv(&x, &cpu_w, n, k);
    let mut worst = 0.0f32;
    for j in 0..n {
        worst = worst.max((gpu[j] - cpu[j]).abs());
    }
    let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!("real q2k gemv: worst {worst}, max|y| {scale}");
    assert!(worst < 1e-2 * scale + 1e-2, "worst {worst} scale {scale}");
}

#[test]
fn embed_real_model_if_present() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../temp/q4km_embd2.bin");
    if !path.exists() {
        eprintln!("skipping: real embedding tensor not present");
        return;
    }
    let _g = gpu();
    let ctx = Ctx::new();
    let raw = std::fs::read(path).unwrap();
    let (vocab, dim) = (248320usize, 1024usize);
    let q = Quant::Q6K;
    let rows = 4usize;
    let ids: Vec<u32> = vec![1, 777, 12345, 248319];
    let cpu = thuban_checkpoint::dequant::to_f32(q, &raw, vocab * dim).unwrap();
    let padded = q.pad_blocks(&raw, vocab * dim).unwrap();
    let wb = ctx.backend.tensor_quant(&padded, vec![vocab as u32, dim as u32], q);
    let w = Weight::quantized(wb);
    let ib = Tensor::new(ctx.backend.storage(rows as u64 * 4), vec![rows as u32], DType::U32);
    ctx.backend.write_u32(&ib.buf, &ids);
    let y = ctx.zero(&[rows as u32, dim as u32]);
    ctx.dispatch(
        shader::EMBED,
        &[
            ("M", rows as f64),
            ("DIM", dim as f64),
            ("SCALE", 1.0),
            ("QTYPE", qtype_of(&w) as f64),
        ],
        &[
            Binding::Full(&ib),
            Binding::Full(w.tensor()),
            Binding::Full(ctx.backend.quant_lut()),
            Binding::Full(&y),
        ],
        [(rows * dim / 32).div_ceil(256) as u32, 1, 1],
    );
    let gpu = ctx.read(&y);
    for r in 0..rows {
        for d in 0..dim {
            let a = gpu[r * dim + d];
            let b = cpu[ids[r] as usize * dim + d];
            assert!(
                (a - b).abs() <= 1e-4 * b.abs() + 1e-6,
                "row {r} col {d}: gpu {a} vs cpu {b}"
            );
        }
    }
}

fn pack_views(backend: &Backend, tensors: Vec<Tensor>) -> Vec<Tensor> {
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let packed = backend.pack_weights(&refs);
    let mut off = 0u64;
    tensors
        .into_iter()
        .map(|t| {
            let v = Tensor::view(packed.buf.clone(), off, t.shape.clone(), t.dtype);
            off += t.byte_len();
            v
        })
        .collect()
}

fn gemv_ops_case(
    quants: &[Quant],
    ns: &[u32],
    k: u32,
    seed: u64,
    x: &[f32],
    rel: f32,
    abs: f32,
) {
    let mut ctx = Ctx::new();
    let xb = ctx.f32(x, &[k]);
    let mut weights = Vec::new();
    let mut cpus = Vec::new();
    for (i, &q) in quants.iter().enumerate() {
        let (w, cpu) = quant_weight(&ctx, q, ns[i], k, seed ^ (i as u64) * 0x9e37);
        weights.push(w.tensor().clone());
        cpus.push(cpu);
    }
    let mut views = pack_views(&ctx.backend, weights);
    let wrap = |t: Tensor| Weight::quantized(t);
    let mut ws: Vec<Weight> = (0..views.len()).map(|_| wrap(views.remove(0))).collect();
    let ys: Vec<Tensor> = ns.iter().map(|&n| ctx.zero(&[n])).collect();
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut pass = Commands::begin(&mut enc);
        let mut ops: Vec<thuban_backend::GemvOp<'_>> = Vec::new();
        for i in 0..ys.len() {
            ops.push(thuban_backend::GemvOp {
                w: &ws[i],
                y: Binding::Full(&ys[i]),
                acc: false,
            });
        }
        ctx.backend.gemv(&mut pass, Binding::Full(&xb), &ops).unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    for i in 0..ys.len() {
        let gpu = ctx.read(&ys[i]);
        agree(
            &gpu,
            &cpu_ref::gemv(x, &cpus[i], ns[i] as usize, k as usize),
            rel,
            abs,
        );
    }
}

#[test]
fn gemv_qkv() {
    let _g = gpu();
    let k = 256u32;
    let x = Rng(77).fill(k as usize);
    gemv_ops_case(&[Quant::Q8_0, Quant::Q8_0, Quant::Q8_0], &[64, 16, 16], k, 91, &x, 1e-3, 1e-2);
}

#[test]
fn gemv_gateup() {
    let _g = gpu();
    let k = 896u32;
    let x = Rng(78).fill(k as usize);
    gemv_ops_case(&[Quant::Q8_0, Quant::Q8_0], &[96, 96], k, 92, &x, 1e-3, 1e-2);
}

#[test]
fn gemv_mixed_quant() {
    let _g = gpu();
    let k = 256u32;
    let x = Rng(79).fill(k as usize);
    gemv_ops_case(&[Quant::Q4_0, Quant::Q8_0, Quant::F16], &[32, 64, 48], k, 93, &x, 2e-2, 8e-2);
}


fn gemv_plain_case(dtype: DType, n: usize, k: usize, seed: u64) {
    let mut ctx = Ctx::new();
    let mut rng = Rng(seed);
    let x = rng.fill(k);
    let w: Vec<f32> = rng.fill(n * k);
    let xb = ctx.f32(&x, &[k as u32]);
    let wb = match dtype {
        DType::F32 => ctx.f32(&w, &[n as u32, k as u32]),
        DType::F16 => {
            let bytes: Vec<u8> = w
                .iter()
                .flat_map(|v| thuban_num::f32_to_f16(*v).to_le_bytes())
                .collect();
            ctx.backend.tensor_f16(&bytes, vec![n as u32, k as u32]).unwrap()
        }
        DType::Bf16 => {
            let bytes: Vec<u8> = w
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            ctx.backend.tensor_bf16(&bytes, vec![n as u32, k as u32]).unwrap()
        }
        _ => unreachable!("plain dtype only"),
    };
    let weight = Weight::plain(wb);
    let y = ctx.zero(&[n as u32]);
    let mut enc = ctx.backend.encoder().unwrap();
    {
        let mut pass = Commands::begin(&mut enc);
        ctx.backend
            .gemv(
                &mut pass,
                Binding::Full(&xb),
                &[thuban_backend::GemvOp {
                    w: &weight,
                    y: Binding::Full(&y),
                    acc: false,
                }],
            )
            .unwrap();
    }
    ctx.backend.submit(&mut enc).unwrap();
    let cpu_w: Vec<f32> = match dtype {
        DType::F16 => w.iter().map(|v| thuban_num::f16_to_f32(thuban_num::f32_to_f16(*v))).collect(),
        DType::Bf16 => w.iter().map(|v| f32::from_bits((v.to_bits() >> 16) << 16)).collect(),
        _ => w.clone(),
    };
    agree(&ctx.read(&y), &cpu_ref::gemv(&x, &cpu_w, n, k), 1e-3, 1e-2);
}

#[test]
fn gemv_f32() {
    let _g = gpu();
    gemv_plain_case(DType::F32, 1792, 896, 7001);
}

#[test]
fn gemv_f16() {
    let _g = gpu();
    gemv_plain_case(DType::F16, 1792, 896, 7003);
}

#[test]
fn gemv_bf16_large() {
    let _g = gpu();
    gemv_plain_case(DType::Bf16, 1792, 896, 7004);
}
