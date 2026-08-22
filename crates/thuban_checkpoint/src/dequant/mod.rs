use thuban_error::{Error, Result};
use thuban_num::{bf16_to_f32, f16_to_f32};
pub use thuban_tensor::Quant;
use thuban_tensor::quant::tables::*;

fn half(bytes: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

fn f32_le(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn u16_le(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

fn u32_le(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

pub fn to_f32(ty: Quant, bytes: &[u8], numel: usize) -> Result<Vec<f32>> {
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

fn decode_block(ty: Quant, blk: &[u8], y: &mut [f32]) {
    match ty {
        Quant::F32 => {
            for (i, c) in blk.chunks_exact(4).enumerate().take(y.len()) {
                y[i] = f32_le(c, 0);
            }
        }
        Quant::F16 => {
            for (i, c) in blk.chunks_exact(2).enumerate().take(y.len()) {
                y[i] = f16_to_f32(u16_le(c, 0));
            }
        }
        Quant::Bf16 => {
            for (i, c) in blk.chunks_exact(2).enumerate().take(y.len()) {
                y[i] = bf16_to_f32(u16_le(c, 0));
            }
        }
        Quant::Q8_0 => q8_0(blk, y),
        Quant::Q4_0 => q4_0(blk, y),
        Quant::Q4_1 => q4_1(blk, y),
        Quant::Q5_0 => q5_0(blk, y),
        Quant::Q5_1 => q5_1(blk, y),
        Quant::Q2K => q2k(blk, y),
        Quant::Q3K => q3k(blk, y),
        Quant::Q4K => q4k(blk, y),
        Quant::Q5K => q5k(blk, y),
        Quant::Q6K => q6k(blk, y),
        Quant::Q8K => q8k(blk, y),
        Quant::Iq2Xxs => iq2_xxs(blk, y),
        Quant::Iq2Xs => iq2_xs(blk, y),
        Quant::Iq3Xxs => iq3_xxs(blk, y),
        Quant::Iq1S => iq1_s(blk, y),
        Quant::Iq4Nl => iq4_nl(blk, y),
        Quant::Iq3S => iq3_s(blk, y),
        Quant::Iq2S => iq2_s(blk, y),
        Quant::Iq4Xs => iq4_xs(blk, y),
        Quant::Iq1M => iq1_m(blk, y),
        Quant::Tq1_0 => tq1_0(blk, y),
        Quant::Tq2_0 => tq2_0(blk, y),
        Quant::Mxfp4 => mxfp4(blk, y),
        Quant::Nvfp4 => nvfp4(blk, y),
        Quant::Q1_0 => q1_0(blk, y),
        Quant::Q2_0 => q2_0(blk, y),
    }
}

fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for i in 0..32 {
        y[i] = (b[2 + i] as i8) as f32 * d;
    }
}

fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for i in 0..16 {
        let q = b[2 + i];
        y[i] = ((q & 0xf) as i32 - 8) as f32 * d;
        y[i + 16] = ((q >> 4) as i32 - 8) as f32 * d;
    }
}

fn q4_1(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let m = half(b, 2);
    for i in 0..16 {
        let q = b[4 + i];
        y[i] = (q & 0xf) as f32 * d + m;
        y[i + 16] = (q >> 4) as f32 * d + m;
    }
}

fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let qh = u32_le(b, 2);
    for i in 0..16 {
        let q = b[6 + i];
        let lo = (q & 0xf) as u32 | (((qh >> i) & 1) << 4);
        let hi = (q >> 4) as u32 | (((qh >> (i + 16)) & 1) << 4);
        y[i] = (lo as i32 - 16) as f32 * d;
        y[i + 16] = (hi as i32 - 16) as f32 * d;
    }
}

fn q5_1(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let m = half(b, 2);
    let qh = u32_le(b, 4);
    for i in 0..16 {
        let q = b[8 + i];
        let lo = (q & 0xf) as u32 | (((qh >> i) & 1) << 4);
        let hi = (q >> 4) as u32 | (((qh >> (i + 16)) & 1) << 4);
        y[i] = lo as f32 * d + m;
        y[i + 16] = hi as f32 * d + m;
    }
}

fn q2k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 80);
    let min = half(b, 82);
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

fn q3k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 108);
    let hmask = &b[0..32];
    let q = &b[32..96];
    let raw = &b[96..108];
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32_le(raw, i * 4);
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

fn q8k(b: &[u8], y: &mut [f32]) {
    let d = f32_le(b, 0);
    for i in 0..256 {
        y[i] = d * (b[4 + i] as i8) as f32;
    }
}

fn q1_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for j in 0..128 {
        y[j] = if b[2 + j / 8] >> (j % 8) & 1 != 0 { d } else { -d };
    }
}

fn q2_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for j in 0..64 {
        let q = (b[2 + j / 4] >> (2 * (j % 4))) & 0x3;
        y[j] = (q as i32 - 1) as f32 * d;
    }
}

fn grid_u64(grid: &[u64], idx: usize) -> [u8; 8] {
    let w = grid[idx].to_le_bytes();
    let mut v = [0u8; 8];
    v.copy_from_slice(&w);
    v
}

fn grid_u32(grid: &[u32], idx: usize) -> [u8; 4] {
    let w = grid[idx].to_le_bytes();
    let mut v = [0u8; 4];
    v.copy_from_slice(&w);
    v
}

fn grid_i8(grid: &[u64], idx: usize) -> [i8; 8] {
    let w = grid[idx].to_le_bytes();
    let mut v = [0i8; 8];
    for (i, b) in w.iter().enumerate() {
        v[i] = *b as i8;
    }
    v
}

fn iq2_xxs(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib32 in 0..8 {
        let aux0 = u32_le(b, 2 + 8 * ib32);
        let aux1 = u32_le(b, 2 + 8 * ib32 + 4);
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let aux8: [u8; 4] = aux0.to_le_bytes();
            let grid = grid_u64(&IQ2XXS_GRID, aux8[l] as usize);
            let signs = KSIGNS_IQ2XS[((aux1 >> 7 * l) & 127) as usize];
            for j in 0..8 {
                y[ib32 * 32 + l * 8 + j] =
                    db * grid[j] as f32 * if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
            }
        }
    }
}

fn iq2_xs(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (b[66 + ib32] & 0xf) as f32) * 0.25,
            d * (0.5 + (b[66 + ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let q = u16_le(b, 2 + 2 * (4 * ib32 + l));
            let grid = grid_u64(&IQ2XS_GRID, (q & 511) as usize);
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                y[ib32 * 32 + l * 8 + j] =
                    db[l / 2] * grid[j] as f32 * if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
            }
        }
    }
}

fn iq3_xxs(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib32 in 0..8 {
        let aux = u32_le(b, 2 + 64 + 4 * ib32);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> 7 * l) & 127) as usize];
            let grid1 = grid_u32(&IQ3XXS_GRID, b[2 + 2 * (4 * ib32 + l)] as usize);
            let grid2 = grid_u32(&IQ3XXS_GRID, b[2 + 2 * (4 * ib32 + l) + 1] as usize);
            for j in 0..4 {
                y[ib32 * 32 + l * 8 + j] =
                    db * grid1[j] as f32 * if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                y[ib32 * 32 + l * 8 + j + 4] =
                    db * grid2[j] as f32 * if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
            }
        }
    }
}

fn iq3_s(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let qs = &b[2..66];
    let qh = &b[66..74];
    let signs = &b[74..106];
    for ib32 in (0..8).step_by(2) {
        let db1 = d * (1.0 + 2.0 * (b[106 + ib32 / 2] & 0xf) as f32);
        let db2 = d * (1.0 + 2.0 * (b[106 + ib32 / 2] >> 4) as f32);
        for l in 0..4 {
            let grid1 = grid_u32(
                &IQ3S_GRID,
                (qs[8 * ib32 + 2 * l] as usize)
                    | (((qh[ib32] as u16) << (8 - 2 * l)) & 256) as usize,
            );
            let grid2 = grid_u32(
                &IQ3S_GRID,
                (qs[8 * ib32 + 2 * l + 1] as usize)
                    | (((qh[ib32] as u16) << (7 - 2 * l)) & 256) as usize,
            );
            for j in 0..4 {
                y[ib32 * 32 + l * 8 + j] = db1
                    * grid1[j] as f32
                    * if signs[4 * ib32 + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                y[ib32 * 32 + l * 8 + j + 4] = db1
                    * grid2[j] as f32
                    * if signs[4 * ib32 + l] & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
            }
        }
        for l in 0..4 {
            let grid1 = grid_u32(
                &IQ3S_GRID,
                (qs[8 * ib32 + 8 + 2 * l] as usize)
                    | (((qh[ib32 + 1] as u16) << (8 - 2 * l)) & 256) as usize,
            );
            let grid2 = grid_u32(
                &IQ3S_GRID,
                (qs[8 * ib32 + 8 + 2 * l + 1] as usize)
                    | (((qh[ib32 + 1] as u16) << (7 - 2 * l)) & 256) as usize,
            );
            for j in 0..4 {
                y[ib32 * 32 + 32 + l * 8 + j] = db2
                    * grid1[j] as f32
                    * if signs[4 * ib32 + 4 + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                y[ib32 * 32 + 32 + l * 8 + j + 4] = db2
                    * grid2[j] as f32
                    * if signs[4 * ib32 + 4 + l] & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
            }
        }
    }
}

fn iq2_s(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let qs = &b[2..34];
    let qh = &b[66..74];
    let signs = &b[34..66];
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (b[74 + ib32] & 0xf) as f32) * 0.25,
            d * (0.5 + (b[74 + ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let dl = db[l / 2];
            let grid = grid_u64(
                &IQ2S_GRID,
                (qs[4 * ib32 + l] as usize) | (((qh[ib32] as u16) << (8 - 2 * l)) & 0x300) as usize,
            );
            for j in 0..8 {
                y[ib32 * 32 + l * 8 + j] = dl
                    * grid[j] as f32
                    * if signs[4 * ib32 + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
            }
        }
    }
}

fn iq1_s(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib in 0..8 {
        let qh = u16_le(b, 2 + 32 + 2 * ib);
        let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
        let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
        for l in 0..4 {
            let grid = grid_i8(
                &IQ1S_GRID,
                (b[2 + 4 * ib + l] as usize) | ((((qh >> 3 * l) & 7) << 8) as usize),
            );
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl * (grid[j] as f32 + delta);
            }
        }
    }
}

fn iq1_m(b: &[u8], y: &mut [f32]) {
    let sc0 = u16_le(b, 32 + 16);
    let sc1 = u16_le(b, 32 + 16 + 2);
    let sc2 = u16_le(b, 32 + 16 + 4);
    let sc3 = u16_le(b, 32 + 16 + 6);
    let sc = [sc0, sc1, sc2, sc3];
    let scale = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    let d = f16_to_f32(scale);
    for ib in 0..8 {
        let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32 + 1.0);
        let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
        let qh0 = b[32 + 2 * ib];
        let qh1 = b[32 + 2 * ib + 1];
        let idx = [
            b[4 * ib] as usize | (((qh0 as usize) << 8) & 0x700),
            b[4 * ib + 1] as usize | (((qh0 as usize) << 4) & 0x700),
            b[4 * ib + 2] as usize | (((qh1 as usize) << 8) & 0x700),
            b[4 * ib + 3] as usize | (((qh1 as usize) << 4) & 0x700),
        ];
        let delta = [
            if qh0 & 0x08 != 0 { -0.125 } else { 0.125 },
            if qh0 & 0x80 != 0 { -0.125 } else { 0.125 },
            if qh1 & 0x08 != 0 { -0.125 } else { 0.125 },
            if qh1 & 0x80 != 0 { -0.125 } else { 0.125 },
        ];
        for l in 0..2 {
            let grid = grid_i8(&IQ1S_GRID, idx[l]);
            let dl = dl1;
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl * (grid[j] as f32 + delta[l]);
            }
        }
        for l in 2..4 {
            let grid = grid_i8(&IQ1S_GRID, idx[l]);
            let dl = dl2;
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl * (grid[j] as f32 + delta[l]);
            }
        }
    }
}

fn iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xf) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
    }
}

fn iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let scales_h = u16_le(b, 2);
    for ib in 0..8 {
        let ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) as u16 | (((scales_h >> 2 * ib) & 3) << 4);
        let dl = d * (ls as f32 - 32.0);
        for j in 0..16 {
            y[ib * 32 + j] = dl * KVALUES_IQ4NL[(b[8 + ib * 16 + j] & 0xf) as usize] as f32;
            y[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(b[8 + ib * 16 + j] >> 4) as usize] as f32;
        }
    }
}

fn tq1_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 52);
    const POW3: [u16; 6] = [1, 3, 9, 27, 81, 243];
    let dec = |q: u8, n: usize| {
        let v = (q as u16 * POW3[n]) & 0xff;
        (((v * 3) >> 8) as i32 - 1) as f32 * d
    };
    for n in 0..5 {
        for m in 0..32 {
            y[n * 32 + m] = dec(b[m], n);
        }
    }
    for n in 0..5 {
        for m in 0..16 {
            y[160 + n * 16 + m] = dec(b[32 + m], n);
        }
    }
    for n in 0..4 {
        for m in 0..4 {
            y[240 + n * 4 + m] = dec(b[48 + m], n);
        }
    }
}

fn tq2_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 64);
    for j in 0..32 {
        for l in 0..4 {
            let q = (b[j] >> (l * 2)) & 3;
            y[l * 32 + j] = (q as i32 - 1) as f32 * d;
        }
    }
    for j in 32..64 {
        for l in 0..4 {
            let q = (b[j] >> (l * 2)) & 3;
            y[128 + l * 32 + j - 32] = (q as i32 - 1) as f32 * d;
        }
    }
}

fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits = if e < 2 {
        0x0020_0000u32 << e
    } else {
        ((e - 1) as u32) << 23
    };
    f32::from_bits(bits)
}

fn mxfp4(b: &[u8], y: &mut [f32]) {
    let d = e8m0_to_f32_half(b[0]);
    for j in 0..16 {
        y[j] = d * KVALUES_FP4[(b[1 + j] & 0xf) as usize] as f32;
        y[j + 16] = d * KVALUES_FP4[(b[1 + j] >> 4) as usize] as f32;
    }
}

fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xf) as i32;
    let man = (x & 0x7) as i32;
    let raw = if exp == 0 {
        (man as f32) * 2f32.powi(-9)
    } else {
        (1.0 + man as f32 / 8.0) * 2f32.powi(exp - 7)
    };
    raw * 0.5
}

fn nvfp4(b: &[u8], y: &mut [f32]) {
    for s in 0..4 {
        let d = ue4m3_to_f32(b[s]);
        for j in 0..8 {
            y[s * 16 + j] = d * KVALUES_FP4[(b[4 + s * 8 + j] & 0xf) as usize] as f32;
            y[s * 16 + j + 8] = d * KVALUES_FP4[(b[4 + s * 8 + j] >> 4) as usize] as f32;
        }
    }
}
