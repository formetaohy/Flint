use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_kernel::name;
use flint_tensor::{Tensor, Weight};

use crate::ops::{ATTN_SEGS, MAX_GQA, NormMode};

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
    eps: f32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::NORM,
        &[
            ("MODE", mode as u32 as f64),
            ("DIM", dim as f64),
            ("W_DIM", w_dim as f64),
            ("EPS", eps as f64),
            ("HEADS", 1.0),
            ("ROT", 2.0),
            ("COS_STRIDE", 1.0),
        ],

        &[x, Binding::Full(weight), gate, y, gate, gate, gate],
        [rows, 1, 1],
    )
}
pub fn norm_rope(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    weight: &Tensor,
    y: Binding<'_>,
    rows: u32,
    dim: u32,
    eps: f32,
    heads: u32,
    rot: u32,
    cos: &Tensor,
    sin: &Tensor,
    args: &Tensor,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::NORM,
        &[
            ("MODE", 4.0),
            ("DIM", dim as f64),
            ("W_DIM", dim as f64),
            ("EPS", eps as f64),
            ("HEADS", heads as f64),
            ("ROT", rot as f64),
            ("COS_STRIDE", (rot / 2) as f64),
        ],
        &[
            x,
            Binding::Full(weight),
            x,
            y,
            Binding::Full(cos),
            Binding::Full(sin),
            Binding::Full(args),
        ],
        [rows, 1, 1],
    )
}
pub fn embed(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    ids: &Tensor,
    table: &Weight,
    y: Binding<'_>,
    rows: u32,
    dim: u32,
    scale: f32,
) -> Result<()> {

    let (wdt, wd, group, scales) = match table {
        Weight::Plain(t) => (0.0, Binding::Full(t), 128.0, Binding::Full(t)),
        Weight::Quantized {
            tensor: t,
            scale: s,
            group: g,
        } => (1.0, Binding::Full(t), *g as f64, Binding::Full(s)),
    };
    backend.dispatch(
        pass,
        name::EMBED,
        &[
            ("M", rows as f64),
            ("DIM", dim as f64),
            ("SCALE", scale as f64),
            ("WDTYPE", wdt),
            ("GROUP", group),
        ],
        &[Binding::Full(ids), wd, scales, y],
        [(rows * dim).div_ceil(256), 1, 1],
    )
}
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
        name::ROPE,
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
pub fn kv_store(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    k_src: Binding<'_>,
    v_src: Binding<'_>,
    k_cache: &Tensor,
    v_cache: &Tensor,
    kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    args: &Tensor,
    m: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::KV_STORE,
        &[
            ("N_KV", kv_heads as f64),
            ("HEAD_DIM", head_dim as f64),
            ("MAX_SEQ", max_seq as f64),
        ],
        &[
            k_src,
            v_src,
            Binding::Full(k_cache),
            Binding::Full(v_cache),
            Binding::Full(args),
        ],
        [(kv_heads * (head_dim / 2)).div_ceil(256), m, 1],
    )
}
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
    stride: u32,
) -> Result<()> {
    assert!(
        q_heads / kv_heads <= MAX_GQA,
        "GQA ratio {} exceeds the attention shader's {MAX_GQA} head slots",
        q_heads / kv_heads
    );

    backend.dispatch(
        pass,
        name::ATTN,
        &[
            ("N_HEADS", q_heads as f64),
            ("KV_HEADS", kv_heads as f64),
            ("HEAD_DIM", head_dim as f64),
            ("MAX_SEQ", max_seq as f64),
            ("SCALE", 1.0 / (head_dim as f64).sqrt()),
            ("WINDOW", window as f64),
            ("NQ_PER_KV", (q_heads / kv_heads) as f64),
            ("STRIDE", stride as f64),
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
        name::MERGE_ATTN,
        &[
            ("N_HEADS", q_heads as f64),
            ("KV_HEADS", kv_heads as f64),
            ("HEAD_DIM", head_dim as f64),
            ("STRIDE", stride as f64),
        ],
        &[Binding::Full(scratch), y, Binding::Full(args)],
        [m, kv_heads, 1],
    )
}
pub fn split_qg(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    q: Binding<'_>,
    gate: Binding<'_>,
    rows: u32,
    heads: u32,
    head_dim: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::SPLIT_QG,
        &[
            ("ROWS", rows as f64),
            ("HEADS", heads as f64),
            ("HD", head_dim as f64),
        ],
        &[x, q, gate],
        [(rows * heads * head_dim).div_ceil(256), 1, 1],
    )
}

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
        name::CONV1D,
        &[("DIM", dim as f64)],
        &[x, Binding::Full(weight), Binding::Full(state), y],
        [dim.div_ceil(256), 1, 1],
    )
}
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
        name::REPEAT_QK,
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
        name::DELTA_GATE,
        &[("HEADS", heads as f64), ("ROW_T", row as f64)],
        &[b, a, Binding::Full(a_log), Binding::Full(dt_bias), beta, g],
        [1, 1, 1],
    )
}
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
        name::DELTA_RECUR,
        &[
            ("HEADS", heads as f64),
            ("K_DIM", key_dim as f64),
            ("V_DIM", val_dim as f64),
        ],
        &[q, k, v, beta, g, Binding::Full(state), y],
        [heads, 1, 1],
    )
}
pub fn rope_tables(
    backend: &Backend,
    max_seq: u32,
    rotary_dim: u32,
    freq_dim: u32,
    theta: f64,
    scaling: Option<&RopeScaling>,
) -> (Tensor, Tensor) {
    let half = rotary_dim / 2;
    let mut cos = Vec::with_capacity((max_seq * half) as usize);
    let mut sin = Vec::with_capacity((max_seq * half) as usize);
    for pos in 0..max_seq {
        for i in 0..half {
            let inv = 1.0 / theta.powf((2 * i) as f64 / freq_dim as f64);
            let factor = scaling.and_then(|s| {
                (pos < s.original_max)
                    .then(|| s.short.get(i as usize))
                    .flatten()
                    .or_else(|| s.long.get(i as usize))
            });
            let inv = factor.map_or(inv, |f| inv / *f as f64);
            let angle = pos as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (
        backend.tensor_f32(&cos, vec![max_seq, half]),
        backend.tensor_f32(&sin, vec![max_seq, half]),
    )
}

#[derive(Clone, Debug)]
pub struct RopeScaling {
    pub short: Vec<f32>,
    pub long: Vec<f32>,
    pub original_max: u32,
}
