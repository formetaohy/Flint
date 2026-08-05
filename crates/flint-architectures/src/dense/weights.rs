//! Weight roles, the loading plan and per-layer/per-forward GPU scratch.

use flint_backend::Backend;
use flint_error::Result;
use flint_model::loader::{self, MlpBlock, Plan, Role, WeightSet, take_moe};
use flint_model::ops::{self, MlpTiles, MoeTiles, M_MAX};
use flint_tensor::{Tensor, Weight};

use crate::dense::config::DenseConfig;
use crate::names::{gguf_key, hf_key};

/// GPU storage role per weight: norms/biases f32, embeddings bf16, projections i8.
pub fn dense_role(key: &str) -> Role {
    if key.contains("norm") || key.ends_with(".bias") || key.ends_with("layer_scalar") {
        Role::F32
    } else if key == "embed_tokens.weight"
        || key == "embed_tokens_per_layer.weight"
        || key.contains("router")
    {
        // Routers: no quantization error tolerated.
        Role::Bf16
    } else {
        Role::I8
    }
}

/// Loading policy: native names via the format's key mapper, roles via [`dense_role`].
pub fn dense_plan(gguf: bool) -> Plan {
    Plan {
        key: if gguf { gguf_key } else { hf_key },
        role: dense_role,
    }
}


/// One transformer layer's weights; K/V absent on Gemma 4's KV-shared layers.
pub(crate) struct LayerW {
    pub(crate) attn_norm: Tensor,
    pub(crate) attn_norm_bias: Option<Tensor>,
    pub(crate) q: Weight,
    pub(crate) k: Option<Weight>,
    pub(crate) v: Option<Weight>,
    pub(crate) o: Weight,
    pub(crate) q_bias: Option<Tensor>,
    pub(crate) k_bias: Option<Tensor>,
    pub(crate) v_bias: Option<Tensor>,
    pub(crate) q_norm: Option<Tensor>,
    pub(crate) k_norm: Option<Tensor>,
    pub(crate) post_attn_norm: Option<Tensor>,
    pub(crate) mlp: MlpBlock,
    pub(crate) post_ffn_norm: Option<Tensor>,
    pub(crate) per_layer_gate: Option<Weight>, // PLE block (Gemma 4)
    pub(crate) per_layer_proj: Option<Weight>,
    pub(crate) per_layer_norm: Option<Tensor>,
    pub(crate) out_scale: Option<Tensor>,
    pub(crate) q_t: Tensor, // activation tiles sized to the layer's head dim
    pub(crate) k_t: Tensor,
    pub(crate) v_t: Tensor,
    pub(crate) q_normed: Tensor,
    pub(crate) k_normed: Tensor,
    /// V-norm output (Gemma 4); the cache stores this instead of v_t.
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

pub(crate) fn take_layer(w: &mut WeightSet, cfg: &DenseConfig, l: u32, backend: &Backend) -> Result<LayerW> {
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
    let mlp = match cfg.moe {
        Some(moe) => MlpBlock::Moe(Box::new(take_moe(
            w,
            &format!("layers.{l}"),
            moe.experts,
            moe.top_k,
            moe.scale,
            moe.shared_scale,
            cfg.layernorm,
        )?)),
        None => MlpBlock::Dense(Box::new(loader::take_mlp(w, &format!("layers.{l}"), cfg.layernorm)?)),
    };
    let ple = cfg.has_ple();
    Ok(LayerW {
        attn_norm: w.take_tensor(&k("input_layernorm.weight"))?,
        attn_norm_bias: take_optional(w, cfg.layernorm, &k("input_layernorm.bias"))?,
        q: w.take(&k("self_attn.q_proj.weight"))?,
        k: k_w,
        v: v_w,
        o: w.take(&k("self_attn.o_proj.weight"))?,
        q_bias: take_optional(w, cfg.qkv_bias, &k("self_attn.q_proj.bias"))?,
        k_bias: k_b,
        v_bias: v_b,
        q_norm: take_optional(w, cfg.qk_norm, &k("self_attn.q_norm.weight"))?,
        k_norm: take_optional(w, cfg.qk_norm && has_kv, &k("self_attn.k_norm.weight"))?,
        post_attn_norm: take_optional(w, cfg.sandwich, &k("post_attention_norm.weight"))?,
        mlp,
        post_ffn_norm: take_optional(w, cfg.sandwich, &k("post_ffw_norm.weight"))?,
        per_layer_gate: if ple {
            Some(w.take(&k("per_layer_input_gate.weight"))?)
        } else {
            None
        },
        per_layer_proj: if ple {
            Some(w.take(&k("per_layer_projection.weight"))?)
        } else {
            None
        },
        per_layer_norm: take_optional(w, ple, &k("post_per_layer_input_norm.weight"))?,
        out_scale: take_optional(w, ple, &k("layer_scalar"))?,
        q_t: backend.zero_tensor(&[M_MAX, qw], &format!("l{l}.q")),
        k_t: backend.zero_tensor(&[M_MAX, kvw], &format!("l{l}.k")),
        v_t: backend.zero_tensor(&[M_MAX, kvw], &format!("l{l}.v")),
        q_normed: backend.zero_tensor(&[M_MAX, qw], &format!("l{l}.q_normed")),
        k_normed: backend.zero_tensor(&[M_MAX, kvw], &format!("l{l}.k_normed")),
        v_normed: backend.zero_tensor(&[M_MAX, kvw], &format!("l{l}.v_normed")),
        attn_out: backend.zero_tensor(&[M_MAX, qw], &format!("l{l}.attn_out")),
    })
}

/// Per-forward scratch shared across layers.
pub(crate) struct Scratch {
    pub(crate) ids: Tensor,
    /// Step args [pos, attn segments]; rope/kv_store/attn read it.
    pub(crate) args: Tensor,
    pub(crate) hidden: Tensor,
    pub(crate) hidden2: Tensor,
    pub(crate) normed: Tensor,
    /// Split-K attention partials [m, kv_heads, ATTN_SEGS, MAX_GQA, hd+2] f32.
    pub(crate) attn_scratch: Tensor,
    /// Partial slot width: largest layer head dim + 2.
    pub(crate) attn_stride: u32,
    pub(crate) mlp: MlpTiles,
    /// Present only with a MoE config.
    pub(crate) moe: Option<MoeTiles>,
    pub(crate) logits: Tensor,
    /// PLE tiles [M_MAX, layers * dim]; present with PLE.
    pub(crate) ple_tok: Option<Tensor>,
    pub(crate) ple_ctx: Option<Tensor>,
    pub(crate) ple_out: Option<Tensor>,
    pub(crate) ple_gate: Option<Tensor>,
    /// Ones tile for the PLE gate activation [M_MAX, dim].
    pub(crate) ple_ones: Option<Tensor>,
}

pub(crate) fn alloc_scratch(cfg: &DenseConfig, backend: &Backend) -> Scratch {
    let max_hd = *cfg.head_dims.iter().max().unwrap();
    let mlp_w = cfg.max_mlp_width();
    let moe = cfg.moe.map(|m| {
        ops::MoeTiles::new(
            &ops::MoeTilesConfig {
                experts: m.experts,
                rows: M_MAX,
                top_k: m.top_k,
                hidden: cfg.hidden,
                intermediate: cfg.intermediate,
            },
            backend,
        )
    });
    let ple_dim = cfg.per_layer.map(|p| p.dim * cfg.layers);
    let ple = |shape: &[u32], label: &str| ple_dim.map(|_| backend.zero_tensor(shape, label));
    Scratch {
        ids: ops::token_ids(backend),
        args: ops::step_args(backend),
        hidden: backend.zero_tensor(&[M_MAX, cfg.hidden], "hidden"),
        hidden2: backend.zero_tensor(&[M_MAX, cfg.hidden], "hidden2"),
        normed: backend.zero_tensor(&[M_MAX, cfg.hidden], "normed"),
        attn_scratch: backend.zero_tensor(
            &[
                M_MAX,
                cfg.kv_heads,
                ops::ATTN_SEGS,
                ops::MAX_GQA,
                max_hd + 2,
            ],
            "attn_scratch",
        ),
        attn_stride: max_hd + 2,
        mlp: MlpTiles {
            gate_out: backend.zero_tensor(&[M_MAX, mlp_w], "mlp_gate"),
            up_out: backend.zero_tensor(&[M_MAX, mlp_w], "mlp_up"),
            act: backend.zero_tensor(&[M_MAX, mlp_w], "mlp_act"),
            down_out: backend.zero_tensor(&[M_MAX, cfg.hidden], "mlp_down"),
        },
        moe,
        logits: backend.zero_tensor(&[M_MAX, cfg.vocab], "logits"),
        ple_tok: ple(&[M_MAX, ple_dim.unwrap_or(0)], "ple_tok"),
        ple_ctx: ple(&[M_MAX, ple_dim.unwrap_or(0)], "ple_ctx"),
        ple_out: ple(&[M_MAX, ple_dim.unwrap_or(0)], "ple_out"),
        ple_gate: ple(&[M_MAX, cfg.per_layer.map_or(0, |p| p.dim)], "ple_gate"),
        ple_ones: cfg.per_layer.map(|p| {
            backend.tensor_f32(
                &vec![1.0; (M_MAX * p.dim) as usize],
                vec![M_MAX, p.dim],
                "ple_ones",
            )
        }),
    }
}
