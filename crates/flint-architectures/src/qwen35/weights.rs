use flint_error::Result;
use flint_model::loader::WeightSet;
use flint_model::pool::KvPool;
use flint_model::weights::{SwigluMlp, take_mlp};
use flint_tensor::{Tensor, Weight};

use super::state::RecurrentPool;

pub(super) struct FullLayerW {
    pub(super) attn_norm: Tensor,
    pub(super) q: Weight,
    pub(super) k: Weight,
    pub(super) v: Weight,
    pub(super) o: Weight,
    pub(super) q_norm: Tensor,
    pub(super) k_norm: Tensor,
    pub(super) mlp: SwigluMlp,
}

pub(super) struct LinearLayerW {
    pub(super) attn_norm: Tensor,
    pub(super) in_proj_qkv: Weight,
    pub(super) in_proj_z: Weight,
    pub(super) in_proj_b: Weight,
    pub(super) in_proj_a: Weight,
    pub(super) conv1d: Tensor,
    pub(super) a_log: Tensor,
    pub(super) dt_bias: Tensor,
    pub(super) norm: Tensor,
    pub(super) out_proj: Weight,
    pub(super) mlp: SwigluMlp,
}

pub(super) enum Layer {
    Full {
        w: Box<FullLayerW>,
        kv: KvPool,
    },
    Linear {
        w: Box<LinearLayerW>,
        state: RecurrentPool,
    },
}

pub(super) struct Mtp {
    pub(super) pre_fc_norm_embedding: Tensor,
    pub(super) pre_fc_norm_hidden: Tensor,
    pub(super) fc: Weight,
    pub(super) layer: Box<FullLayerW>,
    pub(super) norm: Tensor,
    pub(super) kv: KvPool,
}

pub(super) fn take_full_layer(w: &mut WeightSet, p: &str) -> Result<Box<FullLayerW>> {
    let k = |n: &str| format!("{p}.{n}");
    Ok(Box::new(FullLayerW {
        attn_norm: w.take_tensor(&k("input_layernorm.weight"))?,
        q: w.take(&k("self_attn.q_proj.weight"))?,
        k: w.take(&k("self_attn.k_proj.weight"))?,
        v: w.take(&k("self_attn.v_proj.weight"))?,
        o: w.take(&k("self_attn.o_proj.weight"))?,
        q_norm: w.take_tensor(&k("self_attn.q_norm.weight"))?,
        k_norm: w.take_tensor(&k("self_attn.k_norm.weight"))?,
        mlp: take_mlp(w, p, false, false)?,
    }))
}

pub(super) fn take_linear_layer(w: &mut WeightSet, p: &str) -> Result<Box<LinearLayerW>> {
    let k = |n: &str| format!("{p}.linear_attn.{n}");
    Ok(Box::new(LinearLayerW {
        attn_norm: w.take_tensor(&format!("{p}.input_layernorm.weight"))?,
        in_proj_qkv: w.take(&k("in_proj_qkv.weight"))?,
        in_proj_z: w.take(&k("in_proj_z.weight"))?,
        in_proj_b: w.take(&k("in_proj_b.weight"))?,
        in_proj_a: w.take(&k("in_proj_a.weight"))?,
        conv1d: w.take_tensor(&k("conv1d.weight"))?,
        a_log: w.take_tensor(&k("A_log"))?,
        dt_bias: w.take_tensor(&k("dt_bias"))?,
        norm: w.take_tensor(&k("norm.weight"))?,
        out_proj: w.take(&k("out_proj.weight"))?,
        mlp: take_mlp(w, p, false, false)?,
    }))
}
