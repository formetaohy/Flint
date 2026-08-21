mod weight;

pub use weight::Weight;

use thuban_gpu::Buffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    U32,
    Bf16,
    F16,
    I8,
}

pub struct Tensor {
    pub buf: Buffer,
    pub shape: Vec<u32>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(buf: Buffer, shape: Vec<u32>, dtype: DType) -> Self {
        Self { buf, shape, dtype }
    }

    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|d| *d as u64).product()
    }

    pub fn byte_len(&self) -> u64 {
        match self.dtype {
            DType::F32 | DType::U32 => self.numel() * 4,
            DType::Bf16 | DType::F16 => self.numel() * 2,
            DType::I8 => self.numel(),
        }
    }
}
