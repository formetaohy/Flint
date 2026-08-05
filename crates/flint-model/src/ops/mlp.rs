//! MLP and elementwise dispatchers: SwiGLU blocks, activations, adds.

use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_kernel::name;
use flint_tensor::Tensor;

use crate::loader::SwigluMlp;
use crate::ops::{Act, gemm, gemm_acc};

/// Scratch tiles for one SwiGLU MLP: gate/up projections, SiLU activation
/// and the down-projection target. `down_out` doubles as the residual addend.
pub struct MlpTiles {
    pub gate_out: Tensor,
    pub up_out: Tensor,
    pub act: Tensor,
    pub down_out: Tensor,
}
pub fn swiglu_mlp(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    mlp: &SwigluMlp,
    t: &MlpTiles,
    rows: u32,
    intermediate: u32,
    act: Act,
    y_out: Binding<'_>,
    acc_y: bool,
) -> Result<()> {
    if rows == 1 {
        backend.gemv_gateup(
            pass,
            x,
            &mlp.gate,
            &mlp.up,
            Binding::Full(&t.gate_out),
            Binding::Full(&t.up_out),
            intermediate,
            mlp.gate.tensor().shape[1],
        )?;
    } else {
        gemm(
            backend,
            pass,
            x,
            &mlp.gate,
            Binding::Full(&t.gate_out),
            rows,
        )?;
        gemm(backend, pass, x, &mlp.up, Binding::Full(&t.up_out), rows)?;
    }
    swiglu(
        backend,
        pass,
        Binding::Full(&t.gate_out),
        Binding::Full(&t.up_out),
        Binding::Full(&t.act),
        rows * intermediate,
        act,
    )?;
    gemm_acc(
        backend,
        pass,
        Binding::Full(&t.act),
        &mlp.down,
        y_out,
        rows,
        acc_y,
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
        name::ADD,
        &[("N_ELEM", n_elem as f64)],
        &[a, b, y],
        [n_elem.div_ceil(256), 1, 1],
    )
}
/// In-place row-broadcast bias over a [rows, dim] tile: x += bias per row.
pub fn bias(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    bias: &Tensor,
    rows: u32,
    dim: u32,
) -> Result<()> {
    let n_elem = rows * dim;
    backend.dispatch(
        pass,
        name::BIAS,
        &[("N_ELEM", n_elem as f64), ("DIM", dim as f64)],
        &[x, Binding::Full(bias)],
        [n_elem.div_ceil(256), 1, 1],
    )
}
pub fn swiglu(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    gate: Binding<'_>,
    up: Binding<'_>,
    y: Binding<'_>,
    n_elem: u32,
    act: Act,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::SWIGLU,
        &[("N_ELEM", n_elem as f64), ("MODE", act as u32 as f64)],
        &[gate, up, y],
        [n_elem.div_ceil(256), 1, 1],
    )
}
/// In-place logit softcapping (Gemma 4): x = cap * tanh(x / cap).
pub fn softcap(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    n_elem: u32,
    cap: f32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::SOFTCAP,
        &[("N_ELEM", n_elem as f64), ("CAP", cap as f64)],
        &[x],
        [n_elem.div_ceil(256), 1, 1],
    )
}
/// Elementwise multiply with broadcast: y = a * b (b length divides a's).
pub fn mul(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    n: u32,
    m: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::MUL,
        &[("N", n as f64), ("M", m as f64)],
        &[a, b, y],
        [n.div_ceil(256), 1, 1],
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
        name::SIGMOID_MUL,
        &[("N_ELEM", n_elem as f64)],
        &[a, b, y],
        [n_elem.div_ceil(256), 1, 1],
    )
}
pub fn concat(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    a: Binding<'_>,
    b: Binding<'_>,
    y: Binding<'_>,
    rows: u32,
    dim: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::CONCAT,
        &[("ROWS", rows as f64), ("D", dim as f64)],
        &[a, b, y],
        [(rows * 2 * dim).div_ceil(256), 1, 1],
    )
}
