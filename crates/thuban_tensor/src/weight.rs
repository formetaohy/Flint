use crate::{DType, Quant, Tensor};

pub enum Weight {
    Plain(Tensor),
    Quantized(Tensor),
}

impl Weight {
    pub fn plain(t: Tensor) -> Self {
        assert!(
            matches!(t.dtype, DType::F32 | DType::Bf16 | DType::F16),
            "plain weight must be f32, bf16 or f16"
        );
        Self::Plain(t)
    }

    pub fn quantized(t: Tensor) -> Self {
        assert!(
            matches!(t.dtype, DType::Quant(_)),
            "quantized weight must carry a block format"
        );
        Self::Quantized(t)
    }

    pub fn tensor(&self) -> &Tensor {
        match self {
            Self::Plain(t) | Self::Quantized(t) => t,
        }
    }

    pub fn quant(&self) -> Option<Quant> {
        match self.tensor().dtype {
            DType::Quant(q) => Some(q),
            _ => None,
        }
    }
}
