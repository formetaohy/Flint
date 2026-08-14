use flint_backend::Backend;
use flint_error::Result;
use flint_tensor::{DType, Tensor};

use crate::model::MAX_M;

pub fn token_ids(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(MAX_M as u64 * 4), vec![MAX_M], DType::U32)
}

pub fn step_args(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(4), vec![1], DType::U32)
}

pub fn write_step_args(backend: &Backend, args: &Tensor, pos: u32) {
    backend.write_u32(&args.buf, &[pos]);
}

pub fn read_rows(
    backend: &Backend,
    t: &Tensor,
    rows: &[u32],
    m: u32,
    count: u32,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(rows.len());
    for &r in rows {
        assert!(r < m, "row {r} outside chunk");
        out.push(backend.read_f32(&t.buf, r as u64 * count as u64 * 4, count as usize)?);
    }
    Ok(out)
}
