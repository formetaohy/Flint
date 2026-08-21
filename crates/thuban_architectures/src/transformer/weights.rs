use thuban_backend::Backend;
use thuban_error::Result;
use thuban_model::MAX_M;
use thuban_model::loader::{Plan, Role, WeightSet};
use thuban_model::ops::MlpTiles;
use thuban_model::rows;
use thuban_model::weights::{SwigluMlp, take_mlp};
use thuban_tensor::{DType, Tensor, Weight};

use crate::keymap::gguf_key;
use crate::transformer::config::Config;

pub fn role(key: &str) -> Role {
    if key.contains("norm") || key.ends_with(".bias") || key.ends_with("layer_scalar") {
        Role::F32
    } else if key == "embed_tokens.weight"
        || key == "embed_tokens_per_layer.weight"
        || key.contains("router")
    {
        Role::Bf16
    } else {
        Role::I8
    }
}

pub fn plan() -> Plan {
    Plan {
        key: gguf_key,
        role,
    }
}

pub(crate) struct LayerW {
    pub(crate) attn_norm: Tensor,
    pub(crate) attn_norm_bias: Option<Tensor>,
    pub(crate) q: Weight,
    pub(crate) k: Option<Weight>,
    pub(crate) v: Option<Weight>,
    pub(crate) o: Weight,
    pub(crate) o_bias: Option<Tensor>,
    pub(crate) q_bias: Option<Tensor>,
    pub(crate) k_bias: Option<Tensor>,
    pub(crate) v_bias: Option<Tensor>,
    pub(crate) q_norm: Option<Tensor>,
    pub(crate) k_norm: Option<Tensor>,
    pub(crate) post_attn_norm: Option<Tensor>,
    pub(crate) mlp: SwigluMlp,
    pub(crate) post_ffn_norm: Option<Tensor>,
    pub(crate) per_layer_gate: Option<Weight>,
    pub(crate) per_layer_proj: Option<Weight>,
    pub(crate) per_layer_norm: Option<Tensor>,
    pub(crate) out_scale: Option<Tensor>,
    pub(crate) q_out: Tensor,
    pub(crate) k_out: Tensor,
    pub(crate) v_out: Tensor,
    pub(crate) q_normed: Tensor,
    pub(crate) k_normed: Tensor,

    pub(crate) v_normed: Tensor,
    pub(crate) attn_out: Tensor,
}

fn take_optional(w: &mut WeightSet, on: bool, key: &str) -> Result<Option<Tensor>> {
    if on {
        Ok(Some(w.take_tensor(key)?))
    } else {
        Ok(None)
    }
}

pub(crate) fn take_layer(
    w: &mut WeightSet,
    cfg: &Config,
    l: u32,
    backend: &Backend,
) -> Result<LayerW> {
    let k = |n: &str| format!("layers.{l}.{n}");
    let hd = cfg.head_dim(l);
    let qw = cfg.q_heads * hd;
    let kvw = cfg.kv_heads * hd;
    let has_kv = cfg.has_kv(l);

    let (k_w, v_w) = if has_kv {
        (
            Some(w.take(&k("self_attn.k_proj.weight"))?),
            Some(w.take(&k("self_attn.v_proj.weight"))?),
        )
    } else {
        (None, None)
    };
    let (k_b, v_b) = if has_kv && cfg.qkv_bias {
        (
            Some(w.take_tensor(&k("self_attn.k_proj.bias"))?),
            Some(w.take_tensor(&k("self_attn.v_proj.bias"))?),
        )
    } else {
        (None, None)
    };
    let mlp = take_mlp(w, &format!("layers.{l}"), cfg.layernorm)?;
    let per_layer = cfg.has_ple();
    let post_attn_key = "post_attention_norm.weight";
    let post_ffn_key = "post_ffw_norm.weight";
    Ok(LayerW {
        attn_norm: w.take_tensor(&k("input_layernorm.weight"))?,
        attn_norm_bias: take_optional(w, cfg.layernorm, &k("input_layernorm.bias"))?,
        q: w.take(&k("self_attn.q_proj.weight"))?,
        k: k_w,
        v: v_w,
        o: w.take(&k("self_attn.o_proj.weight"))?,
        o_bias: w.take_if(&k("self_attn.o_proj.bias"))?,
        q_bias: take_optional(w, cfg.qkv_bias, &k("self_attn.q_proj.bias"))?,
        k_bias: k_b,
        v_bias: v_b,
        q_norm: take_optional(w, cfg.qk_norm, &k("self_attn.q_norm.weight"))?,
        k_norm: take_optional(w, cfg.qk_norm && has_kv, &k("self_attn.k_norm.weight"))?,
        post_attn_norm: take_optional(w, cfg.sandwich, &k(post_attn_key))?,
        mlp,
        post_ffn_norm: take_optional(w, cfg.sandwich, &k(post_ffn_key))?,
        per_layer_gate: if per_layer {
            Some(w.take(&k("per_layer_input_gate.weight"))?)
        } else {
            None
        },
        per_layer_proj: if per_layer {
            Some(w.take(&k("per_layer_projection.weight"))?)
        } else {
            None
        },
        per_layer_norm: take_optional(w, per_layer, &k("post_per_layer_input_norm.weight"))?,
        out_scale: take_optional(w, per_layer, &k("layer_scalar"))?,
        q_out: backend.zero_tensor(&[MAX_M, qw], DType::F32),
        k_out: backend.zero_tensor(&[MAX_M, kvw], DType::F32),
        v_out: backend.zero_tensor(&[MAX_M, kvw], DType::F32),
        q_normed: backend.zero_tensor(&[MAX_M, qw], DType::F32),
        k_normed: backend.zero_tensor(&[MAX_M, kvw], DType::F32),
        v_normed: backend.zero_tensor(&[MAX_M, kvw], DType::F32),
        attn_out: backend.zero_tensor(&[MAX_M, qw], DType::F32),
    })
}

pub(crate) struct Scratch {
    pub(crate) ids: Tensor,

    pub(crate) meta: Tensor,
    pub(crate) hidden: Tensor,
    pub(crate) hidden2: Tensor,
    pub(crate) normed: Tensor,

    pub(crate) mlp: MlpTiles,

    pub(crate) logits: Tensor,

    pub(crate) per_layer_tok: Option<Tensor>,
    pub(crate) per_layer_ctx: Option<Tensor>,
    pub(crate) per_layer_out: Option<Tensor>,
    pub(crate) per_layer_gate: Option<Tensor>,

    pub(crate) per_layer_ones: Option<Tensor>,
}

pub(crate) fn alloc_scratch(cfg: &Config, backend: &Backend) -> Scratch {
    let mlp_w = cfg.max_mlp_width();
    let per_layer_dim = cfg.per_layer.map(|p| p.dim * cfg.layers);
    let alloc = |shape: &[u32]| per_layer_dim.map(|_| backend.zero_tensor(shape, DType::F32));
    Scratch {
        ids: rows::token_ids(backend),
        meta: rows::row_meta(backend),
        hidden: backend.zero_tensor(&[MAX_M, cfg.hidden], DType::F32),
        hidden2: backend.zero_tensor(&[MAX_M, cfg.hidden], DType::F32),
        normed: backend.zero_tensor(&[MAX_M, cfg.hidden], DType::F32),
        mlp: MlpTiles {
            gate_out: backend.zero_tensor(&[MAX_M, mlp_w], DType::F32),
            up_out: backend.zero_tensor(&[MAX_M, mlp_w], DType::F32),
            act: backend.zero_tensor(&[MAX_M, mlp_w], DType::F32),
            down_out: backend.zero_tensor(&[MAX_M, cfg.hidden], DType::F32),
        },
        logits: backend.zero_tensor(&[MAX_M, cfg.vocab], DType::F32),
        per_layer_tok: alloc(&[MAX_M, per_layer_dim.unwrap_or(0)]),
        per_layer_ctx: alloc(&[MAX_M, per_layer_dim.unwrap_or(0)]),
        per_layer_out: alloc(&[MAX_M, per_layer_dim.unwrap_or(0)]),
        per_layer_gate: alloc(&[MAX_M, cfg.per_layer.map_or(0, |p| p.dim)]),
        per_layer_ones: cfg
            .per_layer
            .map(|p| backend.tensor_f32(&vec![1.0; (MAX_M * p.dim) as usize], vec![MAX_M, p.dim])),
    }
}
