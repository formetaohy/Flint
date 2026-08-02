use crate::{DType, Tensor};

/// A model weight plus its dequantization scales (quantized weights only).
pub struct Weight {
    pub t: Tensor,
    pub scale: Option<Tensor>,
    /// Quantization group size (elements per scale); irrelevant for plain weights.
    pub group: u32,
}

impl Weight {
    pub fn plain(t: Tensor) -> Self {
        assert!(t.dtype != DType::I8, "i8 weight requires scales");
        Self {
            t,
            scale: None,
            group: 128,
        }
    }

    pub fn quant(t: Tensor, scale: Tensor, group: u32) -> Self {
        assert!(t.dtype == DType::I8, "scaled weight must be i8");
        Self {
            t,
            scale: Some(scale),
            group,
        }
    }
}
