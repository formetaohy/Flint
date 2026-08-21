use thuban_backend::{Backend, Binding, Commands};
use thuban_error::Result;
use thuban_kernel::shader;
use thuban_tensor::Tensor;

pub struct RopeArgs {
    pub heads: u32,
    pub head_dim: u32,
    pub rot: u32,
    pub m: u32,
}

pub struct RopeInputs<'a> {
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub args: &'a Tensor,
}

pub fn rope(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    tables: &RopeInputs<'_>,
    spec: &RopeArgs,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::ROPE,
        &[
            ("HEADS", spec.heads as f64),
            ("HEAD_DIM", spec.head_dim as f64),
            ("ROT", spec.rot as f64),
            ("COS_STRIDE", (spec.rot / 2) as f64),
        ],
        &[
            Binding::Full(tables.cos),
            Binding::Full(tables.sin),
            x,
            Binding::Full(tables.args),
        ],
        [spec.m, spec.heads, 1],
    )
}

#[derive(Clone, Debug)]
pub struct RopeScaling {
    pub short: Vec<f32>,
    pub long: Vec<f32>,
    pub original_max: u32,
}

pub fn rope_tables(
    backend: &Backend,
    max_seq: u32,
    rotary_dim: u32,
    freq_dim: u32,
    theta: f64,
    partial: Option<u32>,
    scaling: Option<&RopeScaling>,
) -> (Tensor, Tensor) {
    let half = rotary_dim / 2;
    let mut cos = Vec::with_capacity((max_seq * half) as usize);
    let mut sin = Vec::with_capacity((max_seq * half) as usize);
    for pos in 0..max_seq {
        for i in 0..half {
            let inv = if partial.is_some_and(|p| i >= p) {
                0.0
            } else {
                1.0 / theta.powf((2 * i) as f64 / freq_dim as f64)
            };
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
