//! CPU-side decoders for every ggml quantization block layout Flint reads.
//! GGUF tensors dequantize to f32 (or stay bf16) on the host, then the
//! role-based upload re-packs them for the GPU. Block byte orders follow the
//! ggml struct definitions exactly.

use flint_error::{Error, Result};
use flint_num::{bf16_to_f32, f16_to_f32};

/// The ggml_type enum values used in GGUF tensor info records. Explicit
/// discriminants match the GGUF spec so `ty as u32` serializes correctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Bf16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::Bf16,
            other => {
                return Err(Error::Model(format!(
                    "unsupported ggml tensor type {other} (IQ-quants and exotic types are not implemented)"
                )));
            }
        })
    }

    /// Elements per quantization block (1 for dense float types).
    pub fn block_len(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => {
                32
            }
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => 256,
        }
    }

    /// Bytes per quantization block.
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 | GgmlType::Bf16 => 2,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q2K => 84,
            GgmlType::Q3K => 110,
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
        }
    }
}

fn half(bytes: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

/// Decodes `numel` elements of `ty` from `bytes` into f32.
pub fn to_f32(ty: GgmlType, bytes: &[u8], numel: usize) -> Result<Vec<f32>> {
    let bl = ty.block_len();
    let bb = ty.block_bytes();
    let blocks = numel.div_ceil(bl);
    let need = blocks * bb;
    if bytes.len() < need {
        return Err(Error::Model(format!(
            "tensor truncated: {ty:?} needs {need} bytes for {numel} elems, have {}",
            bytes.len()
        )));
    }
    let mut out = vec![0f32; blocks * bl];
    for b in 0..blocks {
        let blk = &bytes[b * bb..(b + 1) * bb];
        let dst = &mut out[b * bl..(b + 1) * bl];
        decode_block(ty, blk, dst);
    }
    out.truncate(numel);
    Ok(out)
}

fn decode_block(ty: GgmlType, blk: &[u8], y: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (i, c) in blk.chunks_exact(4).enumerate().take(y.len()) {
                y[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgmlType::F16 => {
            for (i, c) in blk.chunks_exact(2).enumerate().take(y.len()) {
                y[i] = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlType::Bf16 => {
            for (i, c) in blk.chunks_exact(2).enumerate().take(y.len()) {
                y[i] = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlType::Q8_0 => q8_0(blk, y),
        GgmlType::Q4_0 => q4_0(blk, y),
        GgmlType::Q4_1 => q4_1(blk, y),
        GgmlType::Q5_0 => q5_0(blk, y),
        GgmlType::Q5_1 => q5_1(blk, y),
        GgmlType::Q2K => q2k(blk, y),
        GgmlType::Q3K => q3k(blk, y),
        GgmlType::Q4K => q4k(blk, y),
        GgmlType::Q5K => q5k(blk, y),
        GgmlType::Q6K => q6k(blk, y),
    }
}

// struct { half d; int8 qs[32] }
fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for i in 0..32 {
        y[i] = (b[2 + i] as i8) as f32 * d;
    }
}

// struct { half d; u8 qs[16] }  (nibble - 8) * d
fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for i in 0..16 {
        let q = b[2 + i];
        y[i] = ((q & 0xf) as i32 - 8) as f32 * d;
        y[i + 16] = ((q >> 4) as i32 - 8) as f32 * d;
    }
}

// struct { half d; half m; u8 qs[16] }  q*d + m
fn q4_1(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let m = half(b, 2);
    for i in 0..16 {
        let q = b[4 + i];
        y[i] = (q & 0xf) as f32 * d + m;
        y[i + 16] = (q >> 4) as f32 * d + m;
    }
}

// struct { half d; u8 qh[4]; u8 qs[16] }  (q5 - 16) * d
fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for i in 0..16 {
        let q = b[6 + i];
        let lo = (q & 0xf) as u32 | (((qh >> i) & 1) << 4);
        let hi = (q >> 4) as u32 | (((qh >> (i + 16)) & 1) << 4);
        y[i] = (lo as i32 - 16) as f32 * d;
        y[i + 16] = (hi as i32 - 16) as f32 * d;
    }
}

// struct { half d; half m; u8 qh[4]; u8 qs[16] }  q5*d + m
fn q5_1(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let m = half(b, 2);
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for i in 0..16 {
        let q = b[8 + i];
        let lo = (q & 0xf) as u32 | (((qh >> i) & 1) << 4);
        let hi = (q >> 4) as u32 | (((qh >> (i + 16)) & 1) << 4);
        y[i] = lo as f32 * d + m;
        y[i + 16] = hi as f32 * d + m;
    }
}

// struct { u8 scales[16]; u8 qs[64]; half d; half dmin }
// 256 values = 16 sub-blocks of 16; scales[is] low nibble = scale, high = min.
fn q2k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 68);
    let min = half(b, 70);
    let scales = &b[0..16];
    let q = &b[16..80];
    let mut o = 0usize;
    let mut is = 0usize;
    let mut qoff = 0usize;
    for _n in 0..2 {
        let mut shift = 0;
        for _j in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xf) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                y[o] = dl * ((q[qoff + l] >> shift) & 3) as f32 - ml;
                o += 1;
            }
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xf) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                y[o] = dl * ((q[qoff + l + 16] >> shift) & 3) as f32 - ml;
                o += 1;
            }
            shift += 2;
        }
        qoff += 32;
    }
}

// struct { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d }
// 12 scale bytes unpack to 16 signed 6-bit scales; hmask carries one high bit
// per value, read bit-plane by bit-plane as `m` walks 1,2,4,...,128.
fn q3k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 108);
    let hmask = &b[0..32];
    let q = &b[32..96];
    let raw = &b[96..108];
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = raw[i * 4] as u32
            | (raw[i * 4 + 1] as u32) << 8
            | (raw[i * 4 + 2] as u32) << 16
            | (raw[i * 4 + 3] as u32) << 24;
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0f) | ((tmp & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
    let scales: Vec<i8> = aux
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|byte| byte as i8)
        .collect();

    let mut o = 0usize;
    let mut is = 0usize;
    let mut qoff = 0usize;
    let mut m = 1u8;
    for _n in 0..2 {
        let mut shift = 0;
        for _j in 0..4 {
            let dl = d * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let v = ((q[qoff + l] >> shift) & 3) as i32 - if hmask[l] & m != 0 { 0 } else { 4 };
                y[o] = dl * v as f32;
                o += 1;
            }
            let dl = d * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let v = ((q[qoff + l + 16] >> shift) & 3) as i32
                    - if hmask[l + 16] & m != 0 { 0 } else { 4 };
                y[o] = dl * v as f32;
                o += 1;
            }
            shift += 2;
            m <<= 1;
        }
        qoff += 32;
    }
}

// struct { half d; half dmin; u8 scales[12]; u8 qs[128] }
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn q4k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let min = half(b, 2);
    let scales = &b[4..16];
    let q = &b[16..144];
    let mut o = 0usize;
    let mut is = 0usize;
    for j in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        let base = j * 32;
        for l in 0..32 {
            y[o] = d1 * (q[base + l] & 0xf) as f32 - m1;
            o += 1;
        }
        for l in 0..32 {
            y[o] = d2 * (q[base + l] >> 4) as f32 - m2;
            o += 1;
        }
        is += 2;
    }
}

// struct { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }
fn q5k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let min = half(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let ql = &b[48..176];
    let mut o = 0usize;
    let mut is = 0usize;
    let (mut u1, mut u2) = (1u8, 2u8);
    for j in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        let base = j * 32;
        for l in 0..32 {
            // qh holds one high bit per value: byte l, bit plane u1 (low
            // half) / u2 (high half), advancing two planes per 64-value group.
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            y[o] = d1 * ((ql[base + l] & 0xf) + hi) as f32 - m1;
            o += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[o] = d2 * ((ql[base + l] >> 4) + hi) as f32 - m2;
            o += 1;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// struct { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d }
fn q6k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 208);
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let mut base = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    for _n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0xf) | (((qh[qh_off + l]) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xf) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
            y[base + l] = d * (sc[sc_off + is] as i8) as f32 * q1 as f32;
            y[base + l + 32] = d * (sc[sc_off + is + 2] as i8) as f32 * q2 as f32;
            y[base + l + 64] = d * (sc[sc_off + is + 4] as i8) as f32 * q3 as f32;
            y[base + l + 96] = d * (sc[sc_off + is + 6] as i8) as f32 * q4 as f32;
        }
        base += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
}
