mod weight;

pub use weight::Weight;

use std::sync::atomic::{AtomicU64, Ordering};

use wgpu::Buffer;

/// Monotonic identity for live tensors; bind group caching keys on it so a
/// stable buffer set reuses one bind group across every forward step.
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

/// How the bytes of a GPU buffer are interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    U32,
    /// Two bf16 values packed into one u32 (low half = even index).
    Bf16Packed,
    /// One i8 per element, stored as raw bytes (used by quantized weights).
    I8,
}

/// A GPU-resident tensor. Shapes are logical; kernels receive dimensions
/// through pipeline override constants.
pub struct Tensor {
    pub buf: Buffer,
    pub shape: Vec<u32>,
    pub dtype: DType,
    /// Process-unique identity, stable for the tensor's lifetime.
    pub id: u64,
}

impl Tensor {
    pub fn new(buf: Buffer, shape: Vec<u32>, dtype: DType) -> Self {
        Self {
            buf,
            shape,
            dtype,
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|d| *d as u64).product()
    }

    /// Byte length of the backing buffer content.
    pub fn byte_len(&self) -> u64 {
        match self.dtype {
            DType::F32 | DType::U32 => self.numel() * 4,
            DType::Bf16Packed => self.numel() * 2,
            DType::I8 => self.numel(),
        }
    }
}
