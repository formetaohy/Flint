use thuban_backend::{Backend, Binding, Commands};
use thuban_error::Result;
use thuban_kernel::shader;
use thuban_tensor::Weight;

pub struct EmbedSpec {
    pub rows: u32,
    pub dim: u32,
    pub scale: f32,
}

pub fn embed(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ids: &thuban_tensor::Tensor,
    table: &Weight,
    y: Binding<'_>,
    spec: &EmbedSpec,
) -> Result<()> {
    let qtype = match table.tensor().dtype {
        thuban_tensor::DType::F32 => 0,
        thuban_tensor::DType::F16 => 1,
        thuban_tensor::DType::Bf16 => 30,
        thuban_tensor::DType::Quant(q) => q.as_u32(),
        thuban_tensor::DType::U32 => unreachable!("embed table is never an index tensor"),
    };
    let lut = Binding::Full(backend.quant_lut());
    backend.dispatch(
        commands,
        shader::EMBED,
        &[
            ("M", spec.rows as f64),
            ("DIM", spec.dim as f64),
            ("SCALE", spec.scale as f64),
            ("QTYPE", qtype as f64),
        ],
        &[
            Binding::Full(ids),
            Binding::Full(table.tensor()),
            lut,
            y,
        ],
        [(spec.rows * spec.dim / 32).div_ceil(256), 1, 1],
    )
}
