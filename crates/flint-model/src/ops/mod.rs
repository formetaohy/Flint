mod attn;
mod gemm;
mod mlp;
mod moe;

pub use attn::{
    RopeScaling, attn, conv1d, delta_gate, delta_recur, embed, kv_store, norm, norm_rope, repeat_qk,
    rope, rope_tables, split_qg,
};
pub use gemm::{gemm, gemm_acc, gemm_qkv, gemv};
pub use mlp::{MlpTiles, add, bias, concat, mul, sigmoid_mul, softcap, swiglu, swiglu_mlp};
pub use moe::{MoeTiles, MoeTilesConfig, expert_gather, expert_scatter, moe_apply, zero_rows};

use flint_backend::Backend;
use flint_error::Result;
use flint_tensor::{DType, Tensor};

pub const ATTN_SEGS: u32 = 32;

pub const MAX_GQA: u32 = 8;

pub const M_MAX: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Silu,
    GeluTanh,
}

#[derive(Clone, Copy)]
pub enum NormMode {

    Offset = 0,

    Gated = 1,

    Direct = 2,

    Layer = 3,
}

pub fn token_ids(backend: &Backend) -> Tensor {
    Tensor::new(
        backend.storage(M_MAX as u64 * 4),
        vec![M_MAX],
        DType::U32,
    )
}

pub fn step_args(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(8), vec![2], DType::U32)
}

pub fn write_step_args(backend: &Backend, args: &Tensor, pos: u32, kv_len: u32) {
    let segs = kv_len.div_ceil(256).clamp(1, ATTN_SEGS);
    backend.write_u32(args.buf.as_ref(), &[pos, segs]);
}

pub fn read_rows(
    backend: &Backend,
    t: &Tensor,
    rows: &[u32],
    m: u32,
    count: u32,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(rows.len());
    for &r in rows {
        assert!(r < m, "row {r} outside chunk");
        out.push(backend.read_f32(t.buf.as_ref(), r as u64 * count as u64 * 4, count as usize)?);
    }
    Ok(out)
}
