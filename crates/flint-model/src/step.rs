use flint_backend::{Backend, Commands, Readback};
use flint_error::Result;
use flint_tensor::{DType, Tensor};

use crate::model::MAX_M;
use crate::ops::ATTN_SEGS;

pub fn token_ids(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(MAX_M as u64 * 4), vec![MAX_M], DType::U32)
}

pub fn step_args(backend: &Backend) -> Tensor {
    Tensor::new(backend.storage(8), vec![2], DType::U32)
}

pub fn write_step_args(backend: &Backend, args: &Tensor, pos: u32, kv_len: u32) {
    let segs = kv_len.div_ceil(256).clamp(1, ATTN_SEGS);
    backend.write_u32(&args.buf, &[pos, segs]);
}

pub fn queue_rows(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    t: &Tensor,
    rows: &[u32],
    m: u32,
    count: u32,
) -> Result<Vec<Readback>> {
    let mut out = Vec::with_capacity(rows.len());
    for &r in rows {
        assert!(r < m, "row {r} outside chunk");
        out.push(backend.queue_read(
            commands,
            &t.buf,
            r as u64 * count as u64 * 4,
            count as usize,
        )?);
    }
    Ok(out)
}

pub fn collect_rows(
    backend: &mut Backend,
    logit_reads: &[Readback],
    hidden_reads: &[Readback],
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut reads = Vec::with_capacity(logit_reads.len() + hidden_reads.len());
    reads.extend_from_slice(logit_reads);
    reads.extend_from_slice(hidden_reads);
    let all = backend.collect_reads(&reads)?;
    Ok((
        all[..logit_reads.len()].to_vec(),
        all[logit_reads.len()..].to_vec(),
    ))
}
