use thuban_backend::{Backend, Binding, Commands};
use thuban_error::{Error, Result};
use thuban_tensor::Weight;

pub fn gemm(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
    rows: u32,
) -> Result<()> {
    gemm_acc(backend, commands, x, w, y, rows, false)
}

pub struct QkvSpec<'a> {
    pub wq: &'a Weight,
    pub wk: &'a Weight,
    pub wv: &'a Weight,
    pub yq: Binding<'a>,
    pub yk: Binding<'a>,
    pub yv: Binding<'a>,
    pub rows: u32,
    pub kv_width: u32,
}

pub fn gemm_qkv(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    spec: &QkvSpec<'_>,
) -> Result<()> {
    if spec.rows == 1 {
        gemv_qkv(backend, commands, x, spec)
    } else {
        gemm(backend, commands, x, spec.wq, spec.yq, spec.rows)?;
        if spec.kv_width > 0 {
            gemm(backend, commands, x, spec.wk, spec.yk, spec.rows)?;
            gemm(backend, commands, x, spec.wv, spec.yv, spec.rows)?;
        }
        Ok(())
    }
}

pub fn gemv_qkv(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    spec: &QkvSpec<'_>,
) -> Result<()> {
    let mut ops = Vec::with_capacity(3);
    ops.push(thuban_backend::GemvOp {
        w: spec.wq,
        y: spec.yq,
        acc: false,
    });
    if spec.kv_width > 0 {
        ops.push(thuban_backend::GemvOp {
            w: spec.wk,
            y: spec.yk,
            acc: false,
        });
        ops.push(thuban_backend::GemvOp {
            w: spec.wv,
            y: spec.yv,
            acc: false,
        });
    }
    backend.gemv(commands, x, &ops)
}

pub fn gemv_gateup(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    gate: &Weight,
    up: &Weight,
    yg: Binding<'_>,
    yu: Binding<'_>,
) -> Result<()> {
    backend.gemv(
        commands,
        x,
        &[
            thuban_backend::GemvOp {
                w: gate,
                y: yg,
                acc: false,
            },
            thuban_backend::GemvOp {
                w: up,
                y: yu,
                acc: false,
            },
        ],
    )
}

pub fn gemv(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
) -> Result<()> {
    backend.gemv(
        commands,
        x,
        &[thuban_backend::GemvOp { w, y, acc: false }],
    )
}

pub fn gemm_acc(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
    rows: u32,
    acc: bool,
) -> Result<()> {
    if rows == 1 {
        backend.gemv(
            commands,
            x,
            &[thuban_backend::GemvOp { w, y, acc }],
        )
    } else {
        backend.gemm_acc(commands, x, w, y, rows, acc)
    }
}

pub fn check_gemm_dims(pairs: &[(u32, u32)]) -> Result<()> {
    for &(n, k) in pairs {
        if !n.is_multiple_of(16) || !k.is_multiple_of(64) {
            return Err(Error::Config(format!(
                "dimension pair (N={n}, K={k}) does not satisfy N%16 and K%64"
            )));
        }
    }
    Ok(())
}
