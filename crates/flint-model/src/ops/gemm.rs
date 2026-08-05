//! Linear projection dispatchers: gemm/gemv fast paths and fused QKV.

use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_tensor::Weight;

/// Projects the activation tile through a weight. Decode (`rows == 1`) takes
/// the gemv fast path; multi-row prefill takes the tiled matmul.
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
    nq: u32,
    nk: u32,
    nv: u32,
) -> Result<()> {
    if rows == 1 {
        let k = wq.tensor().shape[1];
        backend.gemv_qkv(pass, x, wq, wk, wv, yq, yk, yv, nq, nk, nv, k)
    } else {
        gemm(backend, pass, x, wq, yq, rows)?;
        if nk > 0 {
            gemm(backend, pass, x, wk, yk, rows)?;
            gemm(backend, pass, x, wv, yv, rows)?;
        }
        Ok(())
    }
}
/// [`gemm`] with residual accumulation into `y` (the residual-stream fusion:
/// output projections accumulate directly onto the layer input).
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
