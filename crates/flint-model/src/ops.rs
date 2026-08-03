//! Kernel dispatchers shared by every architecture. All activations are
//! [ROWS, dim] tiles; gemm always operates on the full tile.

use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_tensor::{DType, Tensor, Weight};

use crate::loader::SwigluMlp;

/// Activation tile row capacity (also the chunk size for prefill).
pub const ROWS: u32 = 16;

/// RMSNorm variants shared across architectures; the discriminant is the
/// MODE override constant of the norm shader.
#[derive(Clone, Copy)]
pub enum NormMode {
    /// Weight is an offset applied as (1 + w).
    Offset = 0,
    /// Gated: weight * silu(gate).
    Gated = 1,
    /// Direct weight.
    Direct = 2,
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// [ROWS]-wide u32 id scratch fed to `embed`.
pub fn token_ids(backend: &Backend) -> Tensor {
    Tensor::new(
        backend.storage(ROWS as u64 * 4, "ids"),
        vec![ROWS],
        DType::U32,
    )
}

/// One-u32 step-args tensor holding the current position; rope, kv_store and
/// attn read it so the position stays out of the pipeline constants.
pub fn step_args(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(4, "args"), vec![1], DType::U32)
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

/// Projects the activation tile through a weight. Decode (`rows == 1`) takes
/// the bandwidth-optimal gemv fast path; multi-row prefill takes the tiled
/// matmul over the full ROWS tile.
pub fn gemm(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
    rows: u32,
) -> Result<()> {
    if rows == 1 {
        backend.gemv(pass, x, w, y)
    } else {
        backend.gemm(pass, x, w, y, ROWS)
    }
}

/// Scratch tiles for one SwiGLU MLP: gate/up projections, SiLU activation
/// and the down-projection target. Owned by the architecture's per-forward
/// scratch set; `y` doubles as the residual addend after the call.
pub struct MlpTiles {
    pub gate_out: Tensor,
    pub up_out: Tensor,
    pub act: Tensor,
    pub y: Tensor,
}

/// SwiGLU MLP: y = down(silu(gate(x)) * up(x)) over one activation tile.
pub fn swiglu_mlp(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    mlp: &SwigluMlp,
    t: &MlpTiles,
    rows: u32,
    intermediate: u32,
) -> Result<()> {
    gemm(
        backend,
        pass,
        x,
        &mlp.gate,
        Binding::Full(&t.gate_out),
        rows,
    )?;
    gemm(backend, pass, x, &mlp.up, Binding::Full(&t.up_out), rows)?;
    swiglu(
        backend,
        pass,
        Binding::Full(&t.gate_out),
        Binding::Full(&t.up_out),
        Binding::Full(&t.act),
        ROWS * intermediate,
    )?;
    gemm(
        backend,
        pass,
        Binding::Full(&t.act),
        &mlp.down,
        Binding::Full(&t.y),
        rows,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn norm(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    mode: NormMode,
    x: Binding<'_>,
    weight: &Tensor,
    gate: Binding<'_>,
    y: Binding<'_>,
    rows: u32,
    dim: u32,
    w_dim: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "norm",
        &[
            ("MODE", mode as u32 as f64),
            ("DIM", dim as f64),
            ("W_DIM", w_dim as f64),
        ],
        &[x, Binding::Full(weight), gate, y],
        [rows, 1, 1],
    )
}

pub fn embed(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    ids: &Tensor,
    table: &Tensor,
    y: Binding<'_>,
    dim: u32,
    scale: f32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "embed",
        &[
            ("ROWS", ROWS as f64),
            ("DIM", dim as f64),
            ("SCALE", scale as f64),
        ],
        &[Binding::Full(ids), Binding::Full(table), y],
        [div_ceil(ROWS * dim, 256), 1, 1],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    cos: &Tensor,
    sin: &Tensor,
    x: Binding<'_>,
    heads: u32,
    head_dim: u32,
    rot: u32,
    m: u32,
    args: &Tensor,
) -> Result<()> {
    backend.dispatch(
        pass,
        "rope",
        &[
            ("HEADS", heads as f64),
            ("HEAD_DIM", head_dim as f64),
            ("ROT", rot as f64),
            ("COS_STRIDE", (rot / 2) as f64),
        ],
        &[
            Binding::Full(cos),
            Binding::Full(sin),
            x,
            Binding::Full(args),
        ],
        [m, heads, 1],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn kv_store(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    src: Binding<'_>,
    cache: &Tensor,
    kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    args: &Tensor,
    m: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "kv_store",
        &[
            ("N_KV", kv_heads as f64),
            ("HEAD_DIM", head_dim as f64),
            ("MAX_SEQ", max_seq as f64),
        ],
        &[src, Binding::Full(cache), Binding::Full(args)],
        [div_ceil(kv_heads * (head_dim / 2), 256), m, 1],
    )
}

/// KV segments per (row, kv head) in the split-K attention kernel.
pub const ATTN_SEGS: u32 = 32;

/// Max query heads sharing one kv head (the attention shader's head slots).
pub const MAX_GQA: u32 = 8;

#[allow(clippy::too_many_arguments)]
pub fn attn(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    q: Binding<'_>,
    k_cache: &Tensor,
    v_cache: &Tensor,
    scratch: &Tensor,
    y: Binding<'_>,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    args: &Tensor,
    m: u32,
    window: u32,
) -> Result<()> {
    assert!(
        q_heads / kv_heads <= MAX_GQA,
        "GQA ratio {} exceeds the attention shader's {MAX_GQA} head slots",
        q_heads / kv_heads
    );
    // The KV range of each (row, kv head) is split into ATTN_SEGS parallel
    // segments, and every query head of a kv head shares its staged K/V
    // tiles; merge_attn reassembles the segments afterwards.
    backend.dispatch(
        pass,
        "attn",
        &[
            ("N_HEADS", q_heads as f64),
            ("KV_HEADS", kv_heads as f64),
            ("HEAD_DIM", head_dim as f64),
            ("MAX_SEQ", max_seq as f64),
            ("SCALE", 1.0 / (head_dim as f64).sqrt()),
            ("WINDOW", window as f64),
            ("NQ_PER_KV", (q_heads / kv_heads) as f64),
        ],
        &[
            q,
            Binding::Full(k_cache),
            Binding::Full(v_cache),
            Binding::Full(scratch),
            Binding::Full(args),
        ],
        [m, kv_heads, ATTN_SEGS],
    )?;
    backend.dispatch(
        pass,
        "merge_attn",
        &[
            ("N_HEADS", q_heads as f64),
            ("KV_HEADS", kv_heads as f64),
            ("HEAD_DIM", head_dim as f64),
        ],
        &[Binding::Full(scratch), y],
        [m, kv_heads, 1],
    )
}

pub fn split_qg(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    q: Binding<'_>,
    gate: Binding<'_>,
    heads: u32,
    head_dim: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "split_qg",
        &[
            ("ROWS", ROWS as f64),
            ("HEADS", heads as f64),
            ("HD", head_dim as f64),
        ],
        &[x, q, gate],
        [div_ceil(ROWS * heads * head_dim, 256), 1, 1],
    )
}

pub fn add(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    n_elem: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "add",
        &[("N_ELEM", n_elem as f64)],
        &[a, b, y],
        [div_ceil(n_elem, 256), 1, 1],
    )
}

/// In-place row-broadcast bias over a [ROWS, dim] tile: x += bias per row.
pub fn bias(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    bias: &Tensor,
    dim: u32,
) -> Result<()> {
    let n_elem = ROWS * dim;
    backend.dispatch(
        pass,
        "bias",
        &[("N_ELEM", n_elem as f64), ("DIM", dim as f64)],
        &[x, Binding::Full(bias)],
        [div_ceil(n_elem, 256), 1, 1],
    )
}

pub fn swiglu(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    gate: Binding<'_>,
    up: Binding<'_>,
    y: Binding<'_>,
    n_elem: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "swiglu",
        &[("N_ELEM", n_elem as f64)],
        &[gate, up, y],
        [div_ceil(n_elem, 256), 1, 1],
    )
}

pub fn sigmoid_mul(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    n_elem: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "sigmoid_mul",
        &[("N_ELEM", n_elem as f64)],
        &[a, b, y],
        [div_ceil(n_elem, 256), 1, 1],
    )
}

pub fn concat(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    dim: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "concat",
        &[("ROWS", ROWS as f64), ("D", dim as f64)],
        &[a, b, y],
        [div_ceil(ROWS * 2 * dim, 256), 1, 1],
    )
}

/// One causal conv1d time step over a `dim`-wide row slice, updating the
/// channel ring state in place.
pub fn conv1d(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    weight: &Tensor,
    state: &Tensor,
    y: Binding<'_>,
    dim: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "conv1d",
        &[("DIM", dim as f64)],
        &[x, Binding::Full(weight), Binding::Full(state), y],
        [div_ceil(dim, 256), 1, 1],
    )
}

/// Expands a [rows, conv_dim] conv tile's q/k segments from `n_k` key heads
/// to `n_v` value heads (repeat_interleave); output is [rows, 2*n_v*kd].
#[allow(clippy::too_many_arguments)]
pub fn repeat_qk(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    y: Binding<'_>,
    rows: u32,
    n_k: u32,
    n_v: u32,
    kd: u32,
    vd: u32,
) -> Result<()> {
    let conv_dim = 2 * n_k * kd + n_v * vd;
    backend.dispatch(
        pass,
        "repeat_qk",
        &[
            ("ROWS", rows as f64),
            ("N_K", n_k as f64),
            ("N_V", n_v as f64),
            ("K_DIM", kd as f64),
            ("RATIO", (n_v / n_k) as f64),
            ("CONV_DIM", conv_dim as f64),
        ],
        &[x, y],
        [rows, 1, 1],
    )
}

/// Per-head delta-rule gates (beta, g) for one chunk row.
#[allow(clippy::too_many_arguments)]
pub fn delta_gate(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    b: Binding<'_>,
    a: Binding<'_>,
    a_log: &Tensor,
    dt_bias: &Tensor,
    beta: Binding<'_>,
    g: Binding<'_>,
    heads: u32,
    row: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "delta_gate",
        &[("HEADS", heads as f64), ("ROW_T", row as f64)],
        &[b, a, Binding::Full(a_log), Binding::Full(dt_bias), beta, g],
        [1, 1, 1],
    )
}

/// One Gated DeltaNet recurrence step per head over a row's conv output,
/// updating the recurrent state in place; output lands in `y`.
#[allow(clippy::too_many_arguments)]
pub fn delta_recur(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    q: Binding<'_>,
    k: Binding<'_>,
    v: Binding<'_>,
    beta: Binding<'_>,
    g: Binding<'_>,
    state: &Tensor,
    y: Binding<'_>,
    heads: u32,
    key_dim: u32,
    val_dim: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        "delta_recur",
        &[
            ("HEADS", heads as f64),
            ("K_DIM", key_dim as f64),
            ("V_DIM", val_dim as f64),
        ],
        &[q, k, v, beta, g, Binding::Full(state), y],
        [heads, 1, 1],
    )
}

/// Precomputed RoPE tables [max_seq, rotary_dim/2].
pub fn rope_tables(
    backend: &Backend,
    max_seq: u32,
    rotary_dim: u32,
    theta: f64,
) -> (Tensor, Tensor) {
    let half = rotary_dim / 2;
    let mut cos = Vec::with_capacity((max_seq * half) as usize);
    let mut sin = Vec::with_capacity((max_seq * half) as usize);
    for pos in 0..max_seq {
        for i in 0..half {
            let inv = 1.0 / theta.powf((2 * i) as f64 / rotary_dim as f64);
            let angle = pos as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (
        backend.tensor_f32(&cos, vec![max_seq, half], "cos"),
        backend.tensor_f32(&sin, vec![max_seq, half], "sin"),
    )
}
