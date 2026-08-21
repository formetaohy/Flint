use thuban_error::Result;
use thuban_tensor::{Tensor, Weight};

use crate::loader::WeightSet;

pub struct SwigluMlp {
    pub norm: Tensor,
    pub norm_bias: Option<Tensor>,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

pub fn take_mlp(w: &mut WeightSet, prefix: &str, layernorm: bool) -> Result<SwigluMlp> {
    let k = |n: &str| format!("{prefix}.{n}");
    Ok(SwigluMlp {
        norm: w.take_tensor(&k("post_attention_layernorm.weight"))?,
        norm_bias: if layernorm {
            Some(w.take_tensor(&k("post_attention_layernorm.bias"))?)
        } else {
            None
        },
        gate: w.take(&k("mlp.gate_proj.weight"))?,
        up: w.take(&k("mlp.up_proj.weight"))?,
        down: w.take(&k("mlp.down_proj.weight"))?,
    })
}
