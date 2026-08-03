use crate::{DType, Tensor};

/// A model weight consumed by gemm/gemv: either a dense tensor (f32 or packed
/// bf16) or a group-quantized i8 tensor with its per-group scales. The variants
/// are disjoint by construction, so no weight can carry a meaningless scale or
/// group.
pub enum Weight {
    /// Dense f32 or packed-bf16 tensor.
    Plain(Tensor),
    /// Group-quantized i8 tensor plus its per-group f32 scales.
    Quantized {
        tensor: Tensor,
        scale: Tensor,
        group: u32,
    },
}

impl Weight {
    pub fn plain(t: Tensor) -> Self {
        assert!(
            t.dtype == DType::F32 || t.dtype == DType::Bf16Packed,
            "plain weight must be f32 or bf16"
        );
        Self::Plain(t)
    }

    pub fn quant(t: Tensor, scale: Tensor, group: u32) -> Self {
        assert!(t.dtype == DType::I8, "scaled weight must be i8");
        Self::Quantized {
            tensor: t,
            scale,
            group,
        }
    }

    pub fn tensor(&self) -> &Tensor {
        match self {
            Self::Plain(t) => t,
            Self::Quantized { tensor, .. } => tensor,
        }
    }

    pub fn scale(&self) -> Option<&Tensor> {
        match self {
            Self::Plain(_) => None,
            Self::Quantized { scale, .. } => Some(scale),
        }
    }

    pub fn group(&self) -> u32 {
        match self {
            Self::Plain(_) => 128,
            Self::Quantized { group, .. } => *group,
        }
    }
}
