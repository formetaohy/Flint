use flint_backend::{Backend, Binding, Commands};
use flint_error::Result;
use flint_kernel::shader;
use flint_tensor::Tensor;

pub fn add(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    n_elem: u32,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::ADD,
        &[("N_ELEM", n_elem as f64)],
        &[a, b, y],
        [n_elem.div_ceil(256), 1, 1],
    )
}

pub fn bias(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    bias: &Tensor,
    rows: u32,
    dim: u32,
) -> Result<()> {
    let n_elem = rows * dim;
    backend.dispatch(
        commands,
        shader::BIAS,
        &[("N_ELEM", n_elem as f64), ("DIM", dim as f64)],
        &[x, Binding::Full(bias)],
        [n_elem.div_ceil(256), 1, 1],
    )
}

pub fn mul(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    n: u32,
    m: u32,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::MUL,
        &[
            ("N", n as f64),
            ("M", m as f64),
            ("MODE", 0.0),
            ("STRIDE", 0.0),
            ("OFFSET", 0.0),
        ],
        &[a, b, y],
        [n.div_ceil(256), 1, 1],
    )
}

pub struct RowMulSpec {
    pub rows: u32,
    pub cols: u32,
    pub stride: u32,
    pub offset: u32,
}

pub fn row_mul(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    spec: &RowMulSpec,
) -> Result<()> {
    let n = spec.rows * spec.cols;
    backend.dispatch(
        commands,
        shader::MUL,
        &[
            ("N", n as f64),
            ("M", spec.cols as f64),
            ("MODE", 1.0),
            ("STRIDE", spec.stride as f64),
            ("OFFSET", spec.offset as f64),
        ],
        &[a, b, y],
        [n.div_ceil(256), 1, 1],
    )
}

pub fn concat(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    rows: u32,
    dim: u32,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::CONCAT,
        &[("ROWS", rows as f64), ("D", dim as f64)],
        &[a, b, y],
        [(rows * 2 * dim).div_ceil(256), 1, 1],
    )
}
