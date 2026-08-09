use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_tensor::Weight;

pub fn gemm(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
    rows: u32,
) -> Result<()> {
    gemm_acc(backend, pass, x, w, y, rows, false)
}
pub fn gemm_qkv(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    wq: &Weight,
    wk: &Weight,
    wv: &Weight,
    yq: Binding<'_>,
    yk: Binding<'_>,
    yv: Binding<'_>,
    rows: u32,
    nk: u32,
) -> Result<()> {
    if rows == 1 {
        gemv(backend, pass, x, wq, yq)?;
        if nk > 0 {
            gemv(backend, pass, x, wk, yk)?;
            gemv(backend, pass, x, wv, yv)?;
        }
        Ok(())
    } else {
        gemm(backend, pass, x, wq, yq, rows)?;
        if nk > 0 {
            gemm(backend, pass, x, wk, yk, rows)?;
            gemm(backend, pass, x, wv, yv, rows)?;
        }
        Ok(())
    }
}

pub fn gemv(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
) -> Result<()> {
    backend.gemv(pass, x, w, y)
}

pub fn gemm_acc(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    w: &Weight,
    y: Binding<'_>,
    rows: u32,
    acc: bool,
) -> Result<()> {
    if rows == 1 {
        backend.gemv_acc(pass, x, w, y, acc)
    } else {
        backend.gemm_acc(pass, x, w, y, rows, acc)
    }
}
