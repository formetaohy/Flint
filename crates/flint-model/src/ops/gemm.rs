use flint_backend::{Backend, Binding, Commands};
use flint_error::{Error, Result};
use flint_tensor::Weight;

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
        gemv(backend, commands, x, spec.wq, spec.yq)?;
        if spec.kv_width > 0 {
            gemv(backend, commands, x, spec.wk, spec.yk)?;
            gemv(backend, commands, x, spec.wv, spec.yv)?;
        }
        Ok(())
    } else {
        gemm(backend, commands, x, spec.wq, spec.yq, spec.rows)?;
        if spec.kv_width > 0 {
            gemm(backend, commands, x, spec.wk, spec.yk, spec.rows)?;
            gemm(backend, commands, x, spec.wv, spec.yv, spec.rows)?;
        }
        Ok(())
    }
}

pub fn gemv(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
) -> Result<()> {
    backend.gemv(commands, x, w, y)
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
        backend.gemv_acc(commands, x, w, y, acc)
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
