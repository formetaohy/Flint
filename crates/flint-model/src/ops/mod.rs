//! Kernel dispatch layer (see submodules).

//! Kernel dispatchers shared by every architecture. Activations are [M_MAX, dim]

mod attn;
mod gemm;
mod mlp;
mod moe;

pub use attn::{
    RopeScaling, attn, conv1d, delta_gate, delta_recur, embed, kv_store, norm, norm_rope, repeat_qk,
    rope, rope_tables, split_qg,
};
pub use gemm::{gemm, gemm_acc, gemm_qkv};
pub use mlp::{MlpTiles, add, bias, concat, mul, sigmoid_mul, softcap, swiglu, swiglu_mlp};
pub use moe::{MoeTiles, MoeTilesConfig, expert_gather, expert_scatter, moe_apply, zero_rows};


use flint_backend::Backend;
use flint_error::Result;
use flint_tensor::{DType, Tensor};

/// Activation tile row capacity: the max chunk size for prefill.
/// Split-K attention partials: [m, kv_heads, ATTN_SEGS, MAX_GQA, hd+2] f32.
/// `stride` is the scratch slot width (largest layer head dim + 2).
pub const ATTN_SEGS: u32 = 32;

/// Max query heads sharing one kv head (the attention shader's head slots).
pub const MAX_GQA: u32 = 8;

pub const M_MAX: u32 = 128;

/// Gated MLP activation functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Silu,
    GeluTanh,
}

/// RMSNorm / LayerNorm variants shared across architectures; the discriminant
/// is the MODE constant of the norm shader.
#[derive(Clone, Copy)]
pub enum NormMode {
    /// Weight is an offset applied as (1 + w).
    Offset = 0,
    /// Gated: weight * silu(gate).
    Gated = 1,
    /// Direct weight.
    Direct = 2,
    /// Layer norm: (x - mean) * inv_std * w + bias (gate slot holds bias).
    Layer = 3,
}

/// [M_MAX]-wide u32 id scratch fed to `embed`.
pub fn token_ids(backend: &Backend) -> Tensor {
    Tensor::new(
        backend.storage(M_MAX as u64 * 4, "ids"),
        vec![M_MAX],
        DType::U32,
    )
}

/// Step-args tensor: [position, effective attention segments]; rope, kv_store
/// and attn read it. Short prefixes use fewer split-K segments.
pub fn step_args(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(8, "args"), vec![2], DType::U32)
}

/// Writes the step-args tensor for a chunk: its start position and the
/// effective attention segment count for `pos + m` keys.
pub fn write_step_args(backend: &Backend, args: &Tensor, pos: u32, kv_len: u32) {
    let segs = kv_len.div_ceil(ATTN_SEGS).clamp(1, ATTN_SEGS);
    backend.write_u32(&args.buf, &[pos, segs]);
}

/// Reads `count`-wide rows out of a result tensor: one Vec per requested row.
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
        out.push(backend.read_f32(&t.buf, r as u64 * count as u64 * 4, count as usize)?);
    }
    Ok(out)
}

