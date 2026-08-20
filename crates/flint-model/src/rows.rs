use flint_backend::Backend;
use flint_error::Result;
use flint_tensor::{DType, Tensor};

use crate::traits::MAX_M;

pub fn token_ids(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(MAX_M as u64 * 4), vec![MAX_M], DType::U32)
}

pub fn row_meta(backend: &Backend) -> Tensor {
    Tensor::new(
        backend.storage(8 * MAX_M as u64 * 4),
        vec![8 * MAX_M],
        DType::U32,
    )
}

pub fn write_row_meta(backend: &Backend, meta: &Tensor, positions: &[u32], seqs: &[u32], m: u32) {
    assert!(
        positions.len() >= m as usize && seqs.len() >= m as usize,
        "row meta arrays must cover the chunk size"
    );
    let mut data = vec![0u32; 8 * MAX_M as usize];
    for i in 0..m as usize {
        data[8 * i] = positions[i];
        data[8 * i + 1] = seqs[i];
    }
    backend.write_u32(&meta.buf, &data);
}

pub fn read_rows(
    backend: &Backend,
    t: &Tensor,
    rows: &[u32],
    m: u32,
    count: u32,
    base: u32,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(rows.len());
    for &r in rows {
        assert!(r < m, "row {r} outside chunk");
        out.push(backend.read_f32(&t.buf, (base + r) as u64 * count as u64 * 4, count as usize)?);
    }
    Ok(out)
}
