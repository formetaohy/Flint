use flint_backend::{Backend, PAGE_LEN};
use flint_error::{Error, Result};
use flint_tensor::{DType, Tensor};

pub struct ArenaSpec {
    pub seq_lens: Vec<u32>,
    pub pages: Option<u32>,
}

pub struct KvArena {
    seq_lens: Vec<u32>,
    max_pages: u32,
    pages: u32,
    tables: Vec<Vec<u32>>,
    free: Vec<u32>,
}

impl KvArena {
    pub fn new(spec: &ArenaSpec) -> Result<Self> {
        if spec.seq_lens.is_empty() || spec.seq_lens.contains(&0) {
            return Err(Error::Model(
                "sequence budgets must be non-empty and positive".into(),
            ));
        }
        let max_pages = spec
            .seq_lens
            .iter()
            .map(|&l| l.div_ceil(PAGE_LEN))
            .max()
            .expect("budgets are non-empty");
        let pages = spec.pages.unwrap_or_else(|| {
            spec.seq_lens.iter().map(|&l| l.div_ceil(PAGE_LEN)).sum()
        });
        if pages == 0 {
            return Err(Error::Model("KV arena needs at least one page".into()));
        }
        Ok(Self {
            seq_lens: spec.seq_lens.clone(),
            max_pages,
            pages,
            tables: vec![Vec::new(); spec.seq_lens.len()],
            free: (0..pages).rev().collect(),
        })
    }

    pub fn alloc(&mut self, seq: u32, pos: u32, tokens: u32) -> Result<()> {
        let limit = self.seq_lens[seq as usize];
        if pos + tokens > limit {
            return Err(Error::Model(format!("context limit {limit} reached")));
        }
        let want = (pos + tokens).div_ceil(PAGE_LEN) as usize;
        let table = &mut self.tables[seq as usize];
        let need = want.saturating_sub(table.len());
        if need > self.free.len() {
            return Err(Error::Model(format!(
                "KV arena exhausted: {need} pages requested, {} free",
                self.free.len()
            )));
        }
        for _ in 0..need {
            table.push(self.free.pop().expect("free pages checked"));
        }
        Ok(())
    }

    pub fn free_seq(&mut self, seq: u32) {
        for p in self.tables[seq as usize].drain(..) {
            self.free.push(p);
        }
    }

    pub fn truncate(&mut self, seq: u32, keep_tokens: u32) {
        let table = &mut self.tables[seq as usize];
        let keep = (keep_tokens.div_ceil(PAGE_LEN) as usize).min(table.len());
        for p in table.drain(keep..) {
            self.free.push(p);
        }
    }

    pub fn covers(&self, seq: u32, tokens: u32) -> bool {
        self.tables[seq as usize].len() as u32 * PAGE_LEN >= tokens
    }

    pub fn table(&self) -> Vec<u32> {
        let mut flat = vec![u32::MAX; self.tables.len() * self.max_pages as usize];
        for (s, t) in self.tables.iter().enumerate() {
            for (i, &p) in t.iter().enumerate() {
                flat[s * self.max_pages as usize + i] = p;
            }
        }
        flat
    }

    pub fn pages(&self) -> u32 {
        self.pages
    }

    pub fn used(&self) -> u32 {
        self.pages - self.free.len() as u32
    }

    pub fn max_pages(&self) -> u32 {
        self.max_pages
    }

    pub fn seqs(&self) -> u32 {
        self.tables.len() as u32
    }

    pub fn seq_len(&self, seq: u32) -> u32 {
        self.seq_lens[seq as usize]
    }

    pub fn table_of(&self, seq: u32) -> &[u32] {
        &self.tables[seq as usize]
    }
}

pub struct KvPool {
    pub k: Tensor,
    pub v: Tensor,
    pub block_table: Tensor,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub capacity: u32,
    pub max_pages: u32,
}

impl KvPool {
    pub fn new(
        backend: &Backend,
        kv_heads: u32,
        head_dim: u32,
        seqs: u32,
        max_pages: u32,
        pages: u32,
    ) -> Self {
        let capacity = pages * PAGE_LEN;
        Self {
            k: backend.zero_bf16_tensor(&[kv_heads, capacity, head_dim]),
            v: backend.zero_bf16_tensor(&[kv_heads, capacity, head_dim]),
            block_table: Tensor::new(
                backend.storage(seqs as u64 * max_pages as u64 * 4),
                vec![seqs * max_pages],
                DType::U32,
            ),
            kv_heads,
            head_dim,
            capacity,
            max_pages,
        }
    }

    pub fn upload(&self, backend: &Backend, table: &[u32]) {
        assert_eq!(
            table.len() as u32,
            self.block_table.shape[0],
            "block table must cover every sequence"
        );
        backend.write_u32(&self.block_table.buf, table);
    }
}
