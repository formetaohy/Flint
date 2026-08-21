use thuban_backend::Backend;
use thuban_model::MAX_M;
use thuban_model::ops::MlpTiles;
use thuban_model::rows;
use thuban_tensor::{DType, Tensor};

use super::config::Qwen35Config;

pub(super) struct Scratch {
    pub(super) ids: Tensor,

    pub(super) meta: Tensor,
    pub(super) hidden: Tensor,
    pub(super) hidden2: Tensor,
    pub(super) normed: Tensor,
    pub(super) qkv_proj: Tensor,
    pub(super) conv_out: Tensor,

    pub(super) qk_expanded: Tensor,
    pub(super) z: Tensor,
    pub(super) b: Tensor,
    pub(super) a: Tensor,
    pub(super) beta: Tensor,
    pub(super) g: Tensor,
    pub(super) attn_out: Tensor,
    pub(super) attn_gated: Tensor,

    pub(super) qg: Tensor,
    pub(super) q: Tensor,
    pub(super) q_normed: Tensor,
    pub(super) gate: Tensor,
    pub(super) k_raw: Tensor,
    pub(super) k_normed: Tensor,
    pub(super) v_raw: Tensor,
    pub(super) mlp: MlpTiles,
    pub(super) logits: Tensor,
}

pub(super) fn alloc_scratch(cfg: &Qwen35Config, backend: &Backend) -> Scratch {
    let h = cfg.hidden;
    let i = cfg.intermediate;
    let z = |shape: &[u32]| backend.zero_tensor(shape, DType::F32);
    Scratch {
        ids: rows::token_ids(backend),
        meta: rows::row_meta(backend),
        hidden: z(&[MAX_M, h]),
        hidden2: z(&[MAX_M, h]),
        normed: z(&[MAX_M, h]),
        qkv_proj: z(&[MAX_M, cfg.conv_dim()]),
        conv_out: z(&[MAX_M, cfg.conv_dim()]),
        qk_expanded: z(&[MAX_M, cfg.qk_exp_dim()]),
        z: z(&[MAX_M, cfg.value_dim()]),
        b: z(&[MAX_M, cfg.lin_val_heads]),
        a: z(&[MAX_M, cfg.lin_val_heads]),
        beta: z(&[cfg.lin_val_heads]),
        g: z(&[cfg.lin_val_heads]),
        attn_out: z(&[MAX_M, cfg.value_dim()]),
        attn_gated: z(&[MAX_M, cfg.value_dim()]),
        qg: z(&[MAX_M, cfg.q_heads * cfg.head_dim * 2]),
        q: z(&[MAX_M, cfg.q_heads * cfg.head_dim]),
        q_normed: z(&[MAX_M, cfg.q_heads * cfg.head_dim]),
        gate: z(&[MAX_M, cfg.q_heads * cfg.head_dim]),
        k_raw: z(&[MAX_M, cfg.kv_heads * cfg.head_dim]),
        k_normed: z(&[MAX_M, cfg.kv_heads * cfg.head_dim]),
        v_raw: z(&[MAX_M, cfg.kv_heads * cfg.head_dim]),
        mlp: MlpTiles {
            gate_out: z(&[MAX_M, i]),
            up_out: z(&[MAX_M, i]),
            act: z(&[MAX_M, i]),
            down_out: z(&[MAX_M, h]),
        },
        logits: z(&[MAX_M, cfg.vocab]),
    }
}
