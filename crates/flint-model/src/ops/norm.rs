use flint_backend::{Backend, Binding, Commands, shader};
use flint_error::Result;
use flint_tensor::Tensor;

pub struct NormSpec {
    pub mode: super::NormMode,
    pub rows: u32,
    pub dim: u32,
    pub w_dim: u32,
    pub eps: f32,
    pub stride: u32,
    pub ple: u32,
    pub ple_layers: u32,
    pub ple_stride: u32,
}

impl NormSpec {
    pub fn new(mode: super::NormMode, rows: u32, dim: u32, eps: f32) -> Self {
        Self {
            mode,
            rows,
            dim,
            w_dim: dim,
            eps,
            stride: dim,
            ple: 0,
            ple_layers: 0,
            ple_stride: 0,
        }
    }
}

pub fn norm(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    spec: &NormSpec,
    x: Binding<'_>,
    weight: &Tensor,
    gate: Binding<'_>,
    y: Binding<'_>,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::NORM,
        &[
            ("MODE", spec.mode as u32 as f64),
            ("DIM", spec.dim as f64),
            ("W_DIM", spec.w_dim as f64),
            ("EPS", spec.eps as f64),
            ("HEADS", 1.0),
            ("ROT", 2.0),
            ("COS_STRIDE", 1.0),
            ("STRIDE", spec.stride as f64),
            ("PLE", spec.ple as f64),
            ("PLE_LAYERS", spec.ple_layers as f64),
            ("PLE_STRIDE", spec.ple_stride as f64),
        ],
        &[x, Binding::Full(weight), gate, y, gate, gate, gate],
        [spec.rows, 1, 1],
    )
}

pub struct NormRopeSpec {
    pub rows: u32,
    pub dim: u32,
    pub eps: f32,
    pub heads: u32,
    pub rot: u32,
}

pub fn norm_rope(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    spec: &NormRopeSpec,
    x: Binding<'_>,
    weight: &Tensor,
    y: Binding<'_>,
    tables: &super::rope::RopeInputs<'_>,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::NORM,
        &[
            ("MODE", NORM_ROPE_MODE as f64),
            ("DIM", spec.dim as f64),
            ("W_DIM", spec.dim as f64),
            ("EPS", spec.eps as f64),
            ("HEADS", spec.heads as f64),
            ("ROT", spec.rot as f64),
            ("COS_STRIDE", (spec.rot / 2) as f64),
            ("STRIDE", spec.dim as f64),
            ("PLE", 0.0),
            ("PLE_LAYERS", 0.0),
            ("PLE_STRIDE", 0.0),
        ],
        &[
            x,
            Binding::Full(weight),
            x,
            y,
            Binding::Full(tables.cos),
            Binding::Full(tables.sin),
            Binding::Full(tables.args),
        ],
        [spec.rows, 1, 1],
    )
}

const NORM_ROPE_MODE: u32 = 4;
