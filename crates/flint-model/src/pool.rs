use flint_backend::Backend;
use flint_error::Result;
use flint_tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub base: u32,
    pub len: u32,
}

pub struct KvPool {
    pub k: Tensor,
    pub v: Tensor,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub capacity: u32,
    slots: Vec<Slot>,
}

impl KvPool {
    pub fn new(backend: &Backend, kv_heads: u32, slot_lens: &[u32], head_dim: u32) -> Self {
        let capacity = slot_lens.iter().sum();
        let mut slots = Vec::with_capacity(slot_lens.len());
        let mut base = 0u32;
        for &len in slot_lens {
            slots.push(Slot { base, len });
            base += len;
        }
        let shape = [kv_heads, capacity, head_dim];
        Self {
            k: backend.zero_bf16_tensor(&shape),
            v: backend.zero_bf16_tensor(&shape),
            kv_heads,
            head_dim,
            capacity,
            slots,
        }
    }

    pub fn slot(&self, id: u32) -> &Slot {
        &self.slots[id as usize]
    }

    pub fn reset(&self, backend: &Backend, id: u32) -> Result<()> {
        let s = self.slot(id);
        let plane = self.capacity as u64 * self.head_dim as u64 * 2;
        let offset = s.base as u64 * self.head_dim as u64 * 2;
        let size = s.len as u64 * self.head_dim as u64 * 2;
        let mut enc = backend.encoder()?;
        for h in 0..self.kv_heads {
            enc.clear(&self.k.buf, h as u64 * plane + offset, size)?;
            enc.clear(&self.v.buf, h as u64 * plane + offset, size)?;
        }
        enc.finish().wait()?;
        Ok(())
    }
}
