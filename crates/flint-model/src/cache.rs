use flint_backend::Backend;
use flint_tensor::Tensor;

/// Full-attention KV cache: [kv_heads, max_seq, head_dim] planes, packed bf16
/// (two elements per u32) to halve the dominant runtime memory footprint.
pub struct KvCache {
    pub k: Tensor,
    pub v: Tensor,
    pub kv_heads: u32,
    pub max_seq: u32,
    pub head_dim: u32,
}

impl KvCache {
    pub fn new(backend: &Backend, kv_heads: u32, max_seq: u32, head_dim: u32, label: &str) -> Self {
        let shape = [kv_heads, max_seq, head_dim];
        Self {
            k: backend.zero_bf16_tensor(&shape, &format!("{label}.k")),
            v: backend.zero_bf16_tensor(&shape, &format!("{label}.v")),
            kv_heads,
            max_seq,
            head_dim,
        }
    }

    pub fn zero(&self, backend: &Backend) {
        backend.zero_fill(&self.k);
        backend.zero_fill(&self.v);
    }
}

/// Gated DeltaNet state: recurrent [heads, key_dim, value_dim] matrices plus
/// a [conv_dim, 3] conv ring.
pub struct RecurrentState {
    pub recur: Tensor,
    pub conv: Tensor,
}

impl RecurrentState {
    pub fn new(backend: &Backend, recur_shape: [u32; 3], conv_dim: u32, label: &str) -> Self {
        Self {
            recur: backend.zero_tensor(&recur_shape, &format!("{label}.recur")),
            conv: backend.zero_tensor(&[conv_dim, 3], &format!("{label}.conv")),
        }
    }

    pub fn zero(&self, backend: &Backend) {
        backend.zero_fill(&self.recur);
        backend.zero_fill(&self.conv);
    }

    pub fn copy_from(&self, backend: &Backend, src: &RecurrentState) {
        backend.copy(&src.recur, &self.recur);
        backend.copy(&src.conv, &self.conv);
    }
}
