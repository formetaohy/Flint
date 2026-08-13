mod attn;
mod elementwise;
mod embed;
mod gemm;
mod mlp;
mod moe;
mod norm;
mod rope;
mod ssm;

pub use attn::{ATTN_PAD, ATTN_SEGS, MAX_GQA, AttnSpec, attn, check_head_dim, kv_store,
    repeat_qk, RepeatQkSpec};
pub use elementwise::{RowMulSpec, add, bias, concat, mul, row_mul, sigmoid_mul};
pub use embed::{EmbedSpec, embed, embed_split};
pub use gemm::{QkvSpec, check_gemm_dims, gemm, gemm_acc, gemm_qkv, gemv};
pub use mlp::{MlpSpec, MlpTiles, softcap, swiglu, swiglu_mlp};
pub use moe::{GatherSpec, MoeSpec, MoeTiles, MoeTilesConfig, expert_gather, expert_scatter,
    moe_apply, zero_rows};
pub use norm::{NormRopeSpec, NormSpec, norm, norm_rope};
pub use rope::{RopeArgs, RopeInputs, RopeScaling, rope, rope_tables};
pub use ssm::{ConvSpec, DeltaGate, DeltaRecur, SplitQgSpec, conv1d, delta_gate, delta_recur,
    split_qg};

pub use flint_backend::{Act, NormMode};
