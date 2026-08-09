use flint_error::Result;
use flint_tensor::{Tensor, Weight};

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

pub enum MlpBlock {
    Dense(Box<SwigluMlp>),
    Moe(Box<MoeMlp>),
}

impl MlpBlock {
    pub fn norm(&self) -> &Tensor {
        match self {
            MlpBlock::Dense(m) => &m.norm,
            MlpBlock::Moe(m) => &m.norm,
        }
    }

    pub fn norm_bias(&self) -> Option<&Tensor> {
        match self {
            MlpBlock::Dense(m) => m.norm_bias.as_ref(),
            MlpBlock::Moe(m) => m.norm_bias.as_ref(),
        }
    }
}

pub struct ExpertWeights {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

pub struct MoeMlp {
    pub norm: Tensor,
    pub norm_bias: Option<Tensor>,
    pub router: Weight,
    pub experts: Vec<ExpertWeights>,
    pub shared: Option<ExpertWeights>,
    pub top_k: u32,
    pub shared_scale: f32,
}

pub fn take_moe(
    w: &mut WeightSet,
    prefix: &str,
    experts: u32,
    top_k: u32,
    shared_scale: f32,
    layernorm: bool,
) -> Result<MoeMlp> {
    let k = |n: &str| format!("{prefix}.{n}");
    let mut exp = Vec::with_capacity(experts as usize);
    for e in 0..experts {
        let ek = |n: &str| format!("{prefix}.mlp.experts.{e}.{n}");
        exp.push(ExpertWeights {
            gate: w.take(&ek("gate_proj.weight"))?,
            up: w.take(&ek("up_proj.weight"))?,
            down: w.take(&ek("down_proj.weight"))?,
        });
    }
    let shared = if w.has(&k("mlp.shared_expert.gate_proj.weight")) {
        let sk = |n: &str| format!("{prefix}.mlp.shared_expert.{n}");
        Some(ExpertWeights {
            gate: w.take(&sk("gate_proj.weight"))?,
            up: w.take(&sk("up_proj.weight"))?,
            down: w.take(&sk("down_proj.weight"))?,
        })
    } else {
        None
    };
    Ok(MoeMlp {
        norm: w.take_tensor(&k("post_attention_layernorm.weight"))?,
        norm_bias: if layernorm {
            Some(w.take_tensor(&k("post_attention_layernorm.bias"))?)
        } else {
            None
        },
        router: w.take(&k("mlp.router.weight"))?,
        experts: exp,
        shared,
        top_k,
        shared_scale,
    })
}
