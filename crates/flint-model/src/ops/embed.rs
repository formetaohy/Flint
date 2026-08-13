use flint_backend::{Backend, Binding, Commands, shader};
use flint_error::Result;
use flint_tensor::{Tensor, Weight};

pub struct EmbedSpec {
    pub rows: u32,
    pub dim: u32,
    pub scale: f32,
    pub split: u32,
}

pub fn embed(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ids: &Tensor,
    table: &Weight,
    y: Binding<'_>,
    spec: &EmbedSpec,
) -> Result<()> {
    let (wdt, wd, group, scales) = match table {
        Weight::Plain(t) => (0.0, Binding::Full(t), 128.0, Binding::Full(t)),
        Weight::Quantized {
            tensor: t,
            scale: s,
            group: g,
        } => (1.0, Binding::Full(t), *g as f64, Binding::Full(s)),
    };
    embed_dispatch(
        backend,
        commands,
        wd,
        wd,
        scales,
        y,
        &EmbedConsts {
            ids,
            rows: spec.rows,
            dim: spec.dim,
            scale: spec.scale,
            split: u32::MAX as f64,
            wdt,
            group,
            table_rows: table.tensor().shape[0],
        },
    )
}

pub fn embed_split(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ids: &Tensor,
    table: &Weight,
    y: Binding<'_>,
    spec: &EmbedSpec,
) -> Result<()> {
    let t = match table {
        Weight::Plain(t) => t,
        Weight::Quantized { tensor: t, .. } => t,
    };
    let (wdt, group, scales) = match table {
        Weight::Plain(_) => (0.0, 128.0, Binding::Slice(t, 0, t.byte_len())),
        Weight::Quantized {
            scale: s, group: g, ..
        } => (1.0, *g as f64, Binding::Full(s)),
    };
    let half = spec.split as u64 * spec.dim as u64 * 2;
    let t0 = Binding::Slice(t, 0, half);
    let t1 = Binding::Slice(t, half, t.byte_len() - half);
    embed_dispatch(
        backend,
        commands,
        t0,
        t1,
        scales,
        y,
        &EmbedConsts {
            ids,
            rows: spec.rows,
            dim: spec.dim,
            scale: spec.scale,
            split: spec.split as f64,
            wdt,
            group,
            table_rows: t.shape[0],
        },
    )
}

struct EmbedConsts<'a> {
    ids: &'a Tensor,
    rows: u32,
    dim: u32,
    scale: f32,
    split: f64,
    wdt: f64,
    group: f64,
    table_rows: u32,
}

fn embed_dispatch(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    w0: Binding<'_>,
    w1: Binding<'_>,
    scales: Binding<'_>,
    y: Binding<'_>,
    spec: &EmbedConsts<'_>,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::EMBED,
        &[
            ("M", spec.rows as f64),
            ("DIM", spec.dim as f64),
            ("SCALE", spec.scale as f64),
            ("WDTYPE", spec.wdt),
            ("GROUP", spec.group),
            ("SPLIT", spec.split),
            ("ROWS", spec.table_rows as f64),
        ],
        &[
            Binding::Full(spec.ids),
            w0,
            w1,
            scales,
            y,
        ],
        [(spec.rows * spec.dim).div_ceil(256), 1, 1],
    )
}
