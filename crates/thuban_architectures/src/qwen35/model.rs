use thuban_backend::Backend;
use thuban_checkpoint::Checkpoint;
use thuban_error::Result;
use thuban_model::loader::{self};
use thuban_model::ops;
use thuban_model::pool::{ArenaSpec, KvArena, KvPool};
use thuban_tensor::{Tensor, Weight};
use serde_json::Value;

use super::config::{LayerKind, Qwen35Config};
use super::scratch::{Scratch, alloc_scratch};
use super::state::RecurrentPool;
use super::weights::{Layer, plan, take_full_layer, take_linear_layer};

pub struct Qwen35 {
    pub(super) cfg: Qwen35Config,
    pub(super) arena: KvArena,
    pub(super) pos: Vec<u32>,
    pub(super) embed: Weight,

    pub(super) lm_head: Option<Weight>,
    pub(super) norm: Tensor,
    pub(super) layers: Vec<Layer>,
    pub(super) s: Scratch,

    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
}

impl Qwen35 {
    pub(super) fn logit_head(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    pub fn load(
        source: &dyn Checkpoint,
        v: &Value,
        arena_spec: &ArenaSpec,
        backend: &Backend,
    ) -> Result<Self> {
        let arena = KvArena::new(arena_spec)?;
        let cfg = Qwen35Config::parse(v)?;
        let mut w = loader::load_weights(backend, source, &plan())?;
        let embed = w.take("embed_tokens.weight")?;
        let lm_head = if cfg.tied {
            None
        } else {
            Some(w.take("lm_head.weight")?)
        };
        let norm = w.take_tensor("norm.weight")?;

        let seqs = arena.seqs();
        let mut layers = Vec::with_capacity(cfg.layers as usize);
        for (i, kind) in cfg.layer_types.iter().enumerate() {
            let p = format!("layers.{i}");
            match kind {
                LayerKind::Full => layers.push(Layer::Full {
                    w: take_full_layer(&mut w, &p, backend)?,
                    kv: KvPool::new(
                        backend,
                        cfg.kv_heads,
                        cfg.head_dim,
                        seqs,
                        arena.max_pages(),
                        arena.pages(),
                    ),
                }),
                LayerKind::Linear => layers.push(Layer::Linear {
                    w: take_linear_layer(&mut w, &p, backend)?,
                    state: RecurrentPool::new(
                        backend,
                        seqs,
                        cfg.lin_val_heads,
                        cfg.lin_key_dim,
                        cfg.lin_val_dim,
                        cfg.conv_dim(),
                        cfg.conv_kernel,
                    ),
                }),
            }
        }

        let s = alloc_scratch(&cfg, backend);
        let (cos, sin) = ops::rope_tables(
            backend,
            *arena_spec
                .seq_lens
                .iter()
                .max()
                .expect("budgets are non-empty"),
            cfg.rotary_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
            None,
            None,
        );
        Ok(Self {
            cfg,
            arena,
            pos: vec![0; seqs as usize],
            embed,
            lm_head,
            norm,
            layers,
            s,
            cos,
            sin,
        })
    }
}
