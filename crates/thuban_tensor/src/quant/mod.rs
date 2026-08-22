pub mod tables;

use thuban_error::{Error, Result};

pub const LUT_LEN: u32 = 33944;

pub const LUT_IQ2XXS: u32 = 0;
pub const LUT_IQ2XS: u32 = 2048;
pub const LUT_IQ3XXS: u32 = 6144;
pub const LUT_IQ1S: u32 = 7168;
pub const LUT_IQ3S: u32 = 23552;
pub const LUT_IQ2S: u32 = 25600;
pub const LUT_IQ4NL: u32 = 33792;
pub const LUT_KSIGNS: u32 = 33808;
pub const LUT_KMASK: u32 = 33936;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Quant {
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
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl Quant {
    pub fn from_ggml(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Quant::F32,
            1 => Quant::F16,
            2 => Quant::Q4_0,
            3 => Quant::Q4_1,
            6 => Quant::Q5_0,
            7 => Quant::Q5_1,
            8 => Quant::Q8_0,
            10 => Quant::Q2K,
            11 => Quant::Q3K,
            12 => Quant::Q4K,
            13 => Quant::Q5K,
            14 => Quant::Q6K,
            15 => Quant::Q8K,
            16 => Quant::Iq2Xxs,
            17 => Quant::Iq2Xs,
            18 => Quant::Iq3Xxs,
            19 => Quant::Iq1S,
            20 => Quant::Iq4Nl,
            21 => Quant::Iq3S,
            22 => Quant::Iq2S,
            23 => Quant::Iq4Xs,
            29 => Quant::Iq1M,
            30 => Quant::Bf16,
            34 => Quant::Tq1_0,
            35 => Quant::Tq2_0,
            39 => Quant::Mxfp4,
            40 => Quant::Nvfp4,
            41 => Quant::Q1_0,
            42 => Quant::Q2_0,
            other => {
                return Err(Error::Model(format!("unsupported ggml tensor type {other}")));
            }
        })
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn block_len(self) -> usize {
        match self {
            Quant::F32 | Quant::F16 | Quant::Bf16 => 1,
            Quant::Q4_0 | Quant::Q4_1 | Quant::Q5_0 | Quant::Q5_1 | Quant::Q8_0 => 32,
            Quant::Iq4Nl | Quant::Mxfp4 => 32,
            Quant::Nvfp4 => 64,
            Quant::Q2_0 => 64,
            Quant::Q1_0 => 128,
            _ => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Quant::F32 => 4,
            Quant::F16 | Quant::Bf16 => 2,
            Quant::Q4_0 => 18,
            Quant::Q4_1 => 20,
            Quant::Q5_0 => 22,
            Quant::Q5_1 => 24,
            Quant::Q8_0 => 34,
            Quant::Q2K => 84,
            Quant::Q3K => 110,
            Quant::Q4K => 144,
            Quant::Q5K => 176,
            Quant::Q6K => 210,
            Quant::Q8K => 292,
            Quant::Iq2Xxs => 66,
            Quant::Iq2Xs => 74,
            Quant::Iq3Xxs => 98,
            Quant::Iq1S => 50,
            Quant::Iq4Nl => 18,
            Quant::Iq3S => 110,
            Quant::Iq2S => 82,
            Quant::Iq4Xs => 136,
            Quant::Iq1M => 56,
            Quant::Tq1_0 => 54,
            Quant::Tq2_0 => 66,
            Quant::Mxfp4 => 17,
            Quant::Nvfp4 => 36,
            Quant::Q1_0 => 18,
            Quant::Q2_0 => 18,
        }
    }

    pub fn padded_bytes(self) -> usize {
        (self.block_bytes() + 3) / 4 * 4
    }

    pub fn row_bytes(self, k: u32) -> usize {
        let k = k as usize;
        match self {
            Quant::F32 => k * 4,
            Quant::F16 | Quant::Bf16 => k * 2,
            _ => (k / self.block_len()) * self.padded_bytes(),
        }
    }

    pub fn is_block(self) -> bool {
        !matches!(self, Quant::F32 | Quant::F16 | Quant::Bf16)
    }

    pub fn is_plain(self) -> bool {
        !self.is_block()
    }

    pub fn pad_blocks(self, raw: &[u8], numel: usize) -> Result<Vec<u8>> {
        let bl = self.block_len();
        let bb = self.block_bytes();
        let pb = self.padded_bytes();
        let blocks = numel.div_ceil(bl);
        let expect = blocks * bb;
        if raw.len() < expect {
            return Err(Error::Model(format!(
                "quantized tensor truncated: need {expect} bytes, have {}",
                raw.len()
            )));
        }
        let mut out = vec![0u8; blocks * pb];
        for b in 0..blocks {
            let (src, dst) = (&raw[b * bb..(b + 1) * bb], &mut out[b * pb..(b + 1) * pb]);
            dst[..bb].copy_from_slice(src);
        }
        Ok(out)
    }
}

pub fn lut_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(LUT_LEN as usize);
    let push_u64s = |out: &mut Vec<u8>, grid: &[u64]| {
        for v in grid {
            out.extend_from_slice(&v.to_le_bytes());
        }
    };
    let push_u32s = |out: &mut Vec<u8>, grid: &[u32]| {
        for v in grid {
            out.extend_from_slice(&v.to_le_bytes());
        }
    };
    push_u64s(&mut out, &tables::IQ2XXS_GRID);
    push_u64s(&mut out, &tables::IQ2XS_GRID);
    push_u32s(&mut out, &tables::IQ3XXS_GRID);
    push_u64s(&mut out, &tables::IQ1S_GRID);
    push_u32s(&mut out, &tables::IQ3S_GRID);
    push_u64s(&mut out, &tables::IQ2S_GRID);
    for v in tables::KVALUES_IQ4NL {
        out.push(v as u8);
    }
    out.extend_from_slice(&tables::KSIGNS_IQ2XS);
    out.extend_from_slice(&tables::KMASK_IQ2XS);
    assert_eq!(out.len(), LUT_LEN as usize);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_layout_matches_offsets() {
        let lut = lut_bytes();
        assert_eq!(lut.len(), LUT_LEN as usize);
        assert_eq!(lut[LUT_IQ2XXS as usize], 0x08);
        assert_eq!(lut[LUT_IQ2XS as usize], 0x08);
        assert_eq!(lut[LUT_IQ3XXS as usize], 0x04);
        assert_eq!(lut[LUT_IQ1S as usize], 0xff);
        assert_eq!(lut[LUT_IQ3S as usize], 0x01);
        assert_eq!(lut[LUT_IQ2S as usize], 0x08);
        assert_eq!(lut[LUT_IQ4NL as usize], tables::KVALUES_IQ4NL[0] as u8);
        assert_eq!(lut[LUT_KSIGNS as usize], tables::KSIGNS_IQ2XS[0]);
        assert_eq!(lut[LUT_KMASK as usize], tables::KMASK_IQ2XS[0]);
    }

    #[test]
    fn padded_blocks_align_every_block_start() {
        for q in [Quant::Q4_0, Quant::Q8_0, Quant::Q6K, Quant::Iq2Xxs, Quant::Tq1_0] {
            let numel: usize = 256;
            let raw = vec![0xabu8; numel.div_ceil(q.block_len()) * q.block_bytes()];
            let padded = q.pad_blocks(&raw, numel).unwrap();
            for b in 0..numel / q.block_len() {
                assert_eq!(b * q.padded_bytes() % 4, 0, "{q:?}");
            }
            assert_eq!(padded.len(), numel / q.block_len() * q.padded_bytes());
        }
    }
}
