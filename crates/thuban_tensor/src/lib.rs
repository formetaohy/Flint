pub mod quant;
mod weight;

pub use quant::Quant;
pub use weight::Weight;

use thuban_gpu::Buffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    U32,
    Bf16,
    F16,
    Quant(Quant),
}

#[derive(Clone)]
pub struct Tensor {
    pub buf: Buffer,
    pub offset: u64,
    pub shape: Vec<u32>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(buf: Buffer, shape: Vec<u32>, dtype: DType) -> Self {
        Self {
            buf,
            offset: 0,
            shape,
            dtype,
        }
    }

    pub fn view(buf: Buffer, offset: u64, shape: Vec<u32>, dtype: DType) -> Self {
        Self {
            buf,
            offset,
            shape,
            dtype,
        }
    }

    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|d| *d as u64).product()
    }

    pub fn byte_len(&self) -> u64 {
        match self.dtype {
            DType::F32 | DType::U32 => self.numel() * 4,
            DType::Bf16 | DType::F16 => self.numel() * 2,
            DType::Quant(q) => {
                (self.numel() as usize).div_ceil(q.block_len()) as u64 * q.padded_bytes() as u64
            }
        }
    }
}
