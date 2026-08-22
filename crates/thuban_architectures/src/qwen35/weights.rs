use thuban_backend::Backend;
use thuban_error::Result;
use thuban_model::loader::{Plan, Role, WeightSet};
use thuban_model::pool::KvPool;
use thuban_model::weights::SwigluMlp;
use thuban_tensor::{Tensor, Weight};

use super::state::RecurrentPool;

fn take_mlp(w: &mut WeightSet, p: &str, backend: &Backend) -> Result<SwigluMlp> {
    let k = |n: &str| format!("{p}.{n}");
    let mut packed = thuban_model::weights::pack_weights(
        backend,
        vec![w.take(&k("mlp.gate_proj.weight"))?, w.take(&k("mlp.up_proj.weight"))?],
    );
    Ok(SwigluMlp {
        norm: w.take_tensor(&k("post_attention_norm.weight"))?,
        norm_bias: None,
        gate: packed.remove(0),
        up: packed.remove(0),
        down: w.take(&k("mlp.down_proj.weight"))?,
    })
}

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

pub fn role(key: &str) -> Role {
    if key.contains("norm")
        || key.ends_with("dt_bias")
        || key.ends_with("a_log")
        || key.contains("conv1d")
    {
        Role::F32
    } else {
        Role::Quant
    }
}

pub fn gguf_key(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("token_embd.weight") {
        return Some(format!("embed_tokens.weight{rest}"));
    }
    if name == "output.weight" {
        return Some("lm_head.weight".into());
    }
    if name == "output_norm.weight" {
        return Some("norm.weight".into());
    }

    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let layer: u32 = idx.parse().ok()?;
    if tail == "ssm_a" {
        return Some(format!("layers.{layer}.linear_attn.a_log"));
    }
    let (stem, suffix) = tail.rsplit_once('.')?;
    let canon = match stem {
        "attn_norm" => return Some(format!("layers.{layer}.input_layernorm.{suffix}")),
        "post_attention_norm" => {
            return Some(format!("layers.{layer}.post_attention_norm.{suffix}"));
        }
        "attn_q" => "self_attn.qg_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "attn_qkv" => "linear_attn.in_proj_qkv",
        "attn_gate" => "linear_attn.in_proj_z",
        "ssm_conv1d" => "linear_attn.conv1d",
        "ssm_alpha" => "linear_attn.in_proj_a",
        "ssm_beta" => "linear_attn.in_proj_b",
        "ssm_dt" => return Some(format!("layers.{layer}.linear_attn.dt_bias")),
        "ssm_norm" => "linear_attn.norm",
        "ssm_out" => "linear_attn.out_proj",
        "ffn_norm" => return None,
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        _ => return None,
    };
    Some(format!("layers.{layer}.{canon}.{suffix}"))
}

pub(super) fn plan() -> Plan {
    Plan {
        key: gguf_key,
        role,
    }
}

pub(super) fn take_full_layer(
    w: &mut WeightSet,
    p: &str,
    backend: &Backend,
) -> Result<Box<FullLayerW>> {
    let k = |n: &str| format!("{p}.{n}");
    let mut packed = thuban_model::weights::pack_weights(
        backend,
        vec![
            w.take(&k("self_attn.qg_proj.weight"))?,
            w.take(&k("self_attn.k_proj.weight"))?,
            w.take(&k("self_attn.v_proj.weight"))?,
        ],
    );
    Ok(Box::new(FullLayerW {
        attn_norm: w.take_tensor(&k("input_layernorm.weight"))?,
        q: packed.remove(0),
        k: packed.remove(0),
        v: packed.remove(0),
        o: w.take(&k("self_attn.o_proj.weight"))?,
        q_norm: w.take_tensor(&k("self_attn.q_norm.weight"))?,
        k_norm: w.take_tensor(&k("self_attn.k_norm.weight"))?,
        mlp: take_mlp(w, p, backend)?,
    }))
}

pub(super) fn take_linear_layer(
    w: &mut WeightSet,
    p: &str,
    backend: &Backend,
) -> Result<Box<LinearLayerW>> {
    let k = |n: &str| format!("{p}.linear_attn.{n}");
    Ok(Box::new(LinearLayerW {
        attn_norm: w.take_tensor(&format!("{p}.input_layernorm.weight"))?,
        in_proj_qkv: w.take(&k("in_proj_qkv.weight"))?,
        in_proj_z: w.take(&k("in_proj_z.weight"))?,
        in_proj_b: w.take(&k("in_proj_b.weight"))?,
        in_proj_a: w.take(&k("in_proj_a.weight"))?,
        conv1d: w.take_tensor(&k("conv1d.weight"))?,
        a_log: w.take_tensor(&k("a_log"))?,
        dt_bias: w.take_tensor(&k("dt_bias"))?,
        norm: w.take_tensor(&k("norm.weight"))?,
        out_proj: w.take(&k("out_proj.weight"))?,
        mlp: take_mlp(w, p, backend)?,
    }))
}
