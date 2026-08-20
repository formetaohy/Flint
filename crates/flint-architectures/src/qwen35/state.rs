use flint_backend::{Backend, Binding};
use flint_error::Result;
use flint_tensor::{DType, Tensor};

pub(super) struct RecurrentPool {
    recur: Tensor,
    conv: Tensor,
    heads: u32,
    key_dim: u32,
    val_dim: u32,
    conv_dim: u32,
}

impl RecurrentPool {
    pub(super) fn new(backend: &Backend, seqs: u32, recur_shape: [u32; 3], conv_dim: u32) -> Self {
        let [heads, key_dim, val_dim] = recur_shape;
        Self {
            recur: backend.zero_tensor(&[seqs * heads, key_dim, val_dim], DType::F32),
            conv: backend.zero_tensor(&[seqs, conv_dim, 3], DType::F32),
            heads,
            key_dim,
            val_dim,
            conv_dim,
        }
    }

    pub(super) fn zero(&self, backend: &Backend, seq: u32) -> Result<()> {
        let recur_span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        let conv_span = self.conv_dim as u64 * 12;
        let mut enc = backend.encoder()?;
        enc.clear(&self.recur.buf, seq as u64 * recur_span, recur_span)?;
        enc.clear(&self.conv.buf, seq as u64 * conv_span, conv_span)?;
        enc.finish().wait()?;
        Ok(())
    }

    pub(super) fn copy_seq(&self, backend: &Backend, src: &RecurrentPool, seq: u32) -> Result<()> {
        let recur_span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        let conv_span = self.conv_dim as u64 * 12;
        let mut enc = backend.encoder()?;
        enc.copy(
            &src.recur.buf,
            seq as u64 * recur_span,
            &self.recur.buf,
            seq as u64 * recur_span,
            recur_span,
        )?;
        enc.copy(
            &src.conv.buf,
            seq as u64 * conv_span,
            &self.conv.buf,
            seq as u64 * conv_span,
            conv_span,
        )?;
        enc.finish().wait()?;
        Ok(())
    }

    pub(super) fn recur_slice(&self, seq: u32) -> Binding<'_> {
        let span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        Binding::Slice(&self.recur, seq as u64 * span, span)
    }

    pub(super) fn conv_slice(&self, seq: u32) -> Binding<'_> {
        let span = self.conv_dim as u64 * 12;
        Binding::Slice(&self.conv, seq as u64 * span, span)
    }
}
