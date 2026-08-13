use flint_backend::Backend;
use flint_tensor::Tensor;

pub struct KvCache {
    pub k: Tensor,
    pub v: Tensor,
    pub kv_heads: u32,
    pub max_seq: u32,
    pub head_dim: u32,
}

impl KvCache {
    pub fn new(backend: &Backend, kv_heads: u32, max_seq: u32, head_dim: u32) -> Self {
        let shape = [kv_heads, max_seq, head_dim];
        Self {
            k: backend.zero_bf16_tensor(&shape),
            v: backend.zero_bf16_tensor(&shape),
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
