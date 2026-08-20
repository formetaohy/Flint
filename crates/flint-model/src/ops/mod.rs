mod attn;
mod elementwise;
mod embed;
mod gemm;
mod mlp;
mod norm;
mod rope;

pub use attn::{AttnSpec, attn, check_head_dim, kv_store};
pub use elementwise::{RowMulSpec, add, bias, concat, mul, row_mul};
pub use embed::{EmbedSpec, embed, embed_split};
pub use gemm::{QkvSpec, check_gemm_dims, gemm, gemm_acc, gemm_qkv, gemv};
pub use mlp::{MlpSpec, MlpTiles, softcap, swiglu, swiglu_mlp};
pub use norm::{NormRopeSpec, NormSpec, norm, norm_rope};
pub use rope::{RopeArgs, RopeInputs, RopeScaling, rope, rope_tables};

pub use flint_kernel::{Act, NormMode};
