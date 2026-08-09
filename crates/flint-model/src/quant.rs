use flint_error::{Error, Result};
use saturn_core::num::f16_to_f32;

pub fn choose_group(k: u32) -> Result<u32> {
    for g in [128u32, 64, 32] {
        if k.is_multiple_of(g) {
            return Ok(g);
        }
    }
    Err(Error::Config(format!(
        "dimension {k} is not a multiple of 32; cannot quantize"
    )))
}

pub fn repack_q8(bytes: &[u8], rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    assert!(cols.is_multiple_of(32), "Q8_0 K must be a multiple of 32");
    let groups = cols / 32;
    let expect = rows * groups * 34;
    if bytes.len() < expect {
        return Err(Error::Model(format!(
            "Q8_0 tensor truncated: need {expect} bytes, have {}",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; rows * cols];
    let mut scales = vec![0f32; rows * groups];
    for n in 0..rows {
        let row = &bytes[n * groups * 34..];
        for g in 0..groups {
            let blk = &row[g * 34..g * 34 + 34];
            scales[g * rows + n] = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for half in 0..2 {
                let kb = g * 2 + half;
                let dst = &mut out[(kb * rows + n) * 16..(kb * rows + n + 1) * 16];
                dst.copy_from_slice(&blk[2 + half * 16..2 + half * 16 + 16]);
            }
        }
    }
    Ok((out, scales))
}

pub fn quantize(data: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(
        cols.is_multiple_of(group),
        "quantized K must be a multiple of the group size"
    );
    assert!(
        cols.is_multiple_of(16),
        "quantized K must be a multiple of 16 (vec4 blocks)"
    );
    let groups = cols / group;
    let mut bytes = Vec::with_capacity(rows * cols);
    let mut scales = vec![0f32; rows * groups];
    for r in 0..rows {
        for g in 0..groups {
            let block = &data[r * cols + g * group..r * cols + (g + 1) * group];
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));

            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales[g * rows + r] = scale;
            for v in block {
                let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                bytes.push(q as u8);
            }
        }
    }

    let mut out = vec![0u8; rows * cols];
    for kb in 0..cols / 16 {
        for r in 0..rows {
            for i in 0..16 {
                out[(kb * rows + r) * 16 + i] = bytes[r * cols + kb * 16 + i];
            }
        }
    }
    (out, scales)
}
