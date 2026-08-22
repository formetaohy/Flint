use thuban_backend::{Backend, Binding, Commands};
use thuban_error::Result;
use thuban_kernel::shader;
use thuban_tensor::Tensor;

pub struct ConvSpec {
    pub dim: u32,
}

pub fn conv1d(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    weight: &Tensor,
    state: Binding<'_>,
    y: Binding<'_>,
    spec: &ConvSpec,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::CONV1D,
        &[("DIM", spec.dim as f64)],
        &[x, Binding::Full(weight), state, y],
        [spec.dim.div_ceil(256), 1, 1],
    )
}

pub struct DeltaGate<'a> {
    pub b: Binding<'a>,
    pub a: Binding<'a>,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub beta: Binding<'a>,
    pub g: Binding<'a>,
    pub heads: u32,
    pub row: u32,
}

pub fn delta_gate(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    spec: &DeltaGate<'_>,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::DELTA_GATE,
        &[("HEADS", spec.heads as f64), ("ROW_T", spec.row as f64)],
        &[
            spec.b,
            spec.a,
            Binding::Full(spec.a_log),
            Binding::Full(spec.dt_bias),
            spec.beta,
            spec.g,
        ],
        [1, 1, 1],
    )
}

pub struct DeltaRecur<'a> {
    pub q: Binding<'a>,
    pub k: Binding<'a>,
    pub v: Binding<'a>,
    pub beta: Binding<'a>,
    pub g: Binding<'a>,
    pub state: Binding<'a>,
    pub y: Binding<'a>,
    pub heads: u32,
    pub key_dim: u32,
    pub val_dim: u32,
}

pub fn delta_recur(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    spec: &DeltaRecur<'_>,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::DELTA_RECUR,
        &[
            ("HEADS", spec.heads as f64),
            ("K_DIM", spec.key_dim as f64),
            ("V_DIM", spec.val_dim as f64),
        ],
        &[
            spec.q, spec.k, spec.v, spec.beta, spec.g, spec.state, spec.y,
        ],
        [spec.heads, 1, 1],
    )
}

pub struct SplitQGateSpec {
    pub rows: u32,
    pub heads: u32,
    pub head_dim: u32,
}

pub fn split_q_gate(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    q: Binding<'_>,
    gate: Binding<'_>,
    spec: &SplitQGateSpec,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::SPLIT_Q_GATE,
        &[
            ("ROWS", spec.rows as f64),
            ("HEADS", spec.heads as f64),
            ("HD", spec.head_dim as f64),
        ],
        &[x, q, gate],
        [(spec.rows * spec.heads * spec.head_dim).div_ceil(256), 1, 1],
    )
}

pub fn sigmoid_mul(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    n_elem: u32,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::SIGMOID_MUL,
        &[("N_ELEM", n_elem as f64)],
        &[a, b, y],
        [n_elem.div_ceil(256), 1, 1],
    )
}

pub struct RepeatQkSpec {
    pub rows: u32,
    pub key_heads: u32,
    pub val_heads: u32,
    pub key_dim: u32,
    pub conv_dim: u32,
}

pub fn repeat_qk(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    y: Binding<'_>,
    spec: &RepeatQkSpec,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::REPEAT_QK,
        &[
            ("ROWS", spec.rows as f64),
            ("N_K", spec.key_heads as f64),
            ("N_V", spec.val_heads as f64),
            ("K_DIM", spec.key_dim as f64),
            ("CONV_DIM", spec.conv_dim as f64),
        ],
        &[x, y],
        [spec.rows, 1, 1],
    )
}
