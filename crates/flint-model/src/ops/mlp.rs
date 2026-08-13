use flint_backend::{Backend, Binding, Commands, shader};
use flint_error::Result;
use flint_tensor::Tensor;

use crate::mlp_weights::SwigluMlp;
use crate::ops::{Act, gemm, gemm_acc, gemv};

pub struct MlpTiles {
    pub gate_out: Tensor,
    pub up_out: Tensor,
    pub act: Tensor,
    pub down_out: Tensor,
}

pub struct MlpSpec {
    pub rows: u32,
    pub intermediate: u32,
    pub act: Act,
    pub acc: bool,
}

pub fn swiglu_mlp(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    mlp: &SwigluMlp,
    t: &MlpTiles,
    y: Binding<'_>,
    spec: &MlpSpec,
) -> Result<()> {
    if spec.rows == 1 {
        gemv(backend, commands, x, &mlp.gate, Binding::Full(&t.gate_out))?;
        gemv(backend, commands, x, &mlp.up, Binding::Full(&t.up_out))?;
    } else {
        gemm(
            backend,
            commands,
            x,
            &mlp.gate,
            Binding::Full(&t.gate_out),
            spec.rows,
        )?;
        gemm(backend, commands, x, &mlp.up, Binding::Full(&t.up_out), spec.rows)?;
    }
    swiglu(
        backend,
        commands,
        Binding::Full(&t.gate_out),
        Binding::Full(&t.up_out),
        Binding::Full(&t.act),
        spec.rows * spec.intermediate,
        spec.act,
    )?;
    gemm_acc(
        backend,
        commands,
        Binding::Full(&t.act),
        &mlp.down,
        y,
        spec.rows,
        spec.acc,
    )
}

pub fn swiglu(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    gate: Binding<'_>,
    up: Binding<'_>,
    y: Binding<'_>,
    n_elem: u32,
    act: Act,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::SWIGLU,
        &[("N_ELEM", n_elem as f64), ("MODE", act as u32 as f64)],
        &[gate, up, y],
        [n_elem.div_ceil(256), 1, 1],
    )
}

pub fn softcap(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    n_elem: u32,
    cap: f32,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::SOFTCAP,
        &[("N_ELEM", n_elem as f64), ("CAP", cap as f64)],
        &[x],
        [n_elem.div_ceil(256), 1, 1],
    )
}
