use thuban_backend::{Backend, Binding};
use thuban_error::Result;
use thuban_tensor::{DType, Tensor};

pub(super) struct RecurrentPool {
    recur: Tensor,
    conv: Tensor,
    heads: u32,
    key_dim: u32,
    val_dim: u32,
    conv_dim: u32,
    conv_kernel: u32,
}

impl RecurrentPool {
    pub(super) fn new(
        backend: &Backend,
        seqs: u32,
        heads: u32,
        key_dim: u32,
        val_dim: u32,
        conv_dim: u32,
        conv_kernel: u32,
    ) -> Self {
        Self {
            recur: backend.zero_tensor(&[seqs * heads, key_dim, val_dim], DType::F32),
            conv: backend.zero_tensor(&[seqs, conv_dim, conv_kernel - 1], DType::F32),
            heads,
            key_dim,
            val_dim,
            conv_dim,
            conv_kernel,
        }
    }

    pub(super) fn zero(&self, backend: &Backend, seq: u32) -> Result<()> {
        let recur_span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        let conv_span = self.conv_dim as u64 * (self.conv_kernel - 1) as u64 * 4;
        let mut enc = backend.encoder()?;
        enc.clear(&self.recur.buf, seq as u64 * recur_span, recur_span)?;
        enc.clear(&self.conv.buf, seq as u64 * conv_span, conv_span)?;
        enc.finish().wait()?;
        Ok(())
    }

    pub(super) fn recur_slice(&self, seq: u32) -> Binding<'_> {
        let span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        Binding::Slice(&self.recur, seq as u64 * span, span)
    }

    pub(super) fn conv_slice(&self, seq: u32) -> Binding<'_> {
        let span = self.conv_dim as u64 * (self.conv_kernel - 1) as u64 * 4;
        Binding::Slice(&self.conv, seq as u64 * span, span)
    }
}
