use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::{Error, Result};
use flint_model::loader::{self, Plan, Role};
use flint_model::ops;
use flint_model::pool::{ArenaSpec, KvArena, KvPool};
use flint_tensor::{Tensor, Weight};
use serde_json::Value;

use super::config::{LayerKind, Qwen35Config};
use super::scratch::{Scratch, alloc_scratch};
use super::state::RecurrentPool;
use super::weights::{Layer, Mtp, take_full_layer, take_linear_layer};
use crate::keymap::hf_key;

const PLAN: Plan = Plan { key: hf_key, role };

fn role(key: &str) -> Role {
    if key.contains("norm")
        || key.ends_with("dt_bias")
        || key.ends_with("A_log")
        || key.contains("conv1d")
    {
        Role::F32
    } else if key == "embed_tokens.weight" {
        Role::Bf16
    } else {
        Role::I8
    }
}

pub struct Qwen35 {
    pub(super) cfg: Qwen35Config,
    pub(super) arena: KvArena,
    pub(super) pos: Vec<u32>,
    pub(super) saved_pos: Vec<u32>,
    pub(super) mtp_pos: Vec<u32>,
    pub(super) embed: Weight,

    pub(super) lm_head: Option<Weight>,
    pub(super) norm: Tensor,
    pub(super) layers: Vec<Layer>,
    pub(super) mtp: Option<Mtp>,
    pub(super) s: Scratch,

    pub(super) snap: Vec<RecurrentPool>,
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
        if source.kind() == CheckpointKind::Gguf {
            return Err(Error::Model("Qwen3.5 has no GGUF representation".into()));
        }
        let arena = KvArena::new(arena_spec)?;
        let cfg = Qwen35Config::parse(v)?;
        let mut w = loader::load_weights(backend, source, &PLAN)?;
        if cfg.has_mtp != w.has("mtp.fc.weight") {
            return Err(Error::Model(
                "mtp_num_hidden_layers and checkpoint mtp weights disagree".into(),
            ));
        }
        let embed = w.take("embed_tokens.weight")?;
        let lm_head = if cfg.tied {
            None
        } else {
            Some(w.take("lm_head.weight")?)
        };
        let norm = w.take_tensor("norm.weight")?;

        let seqs = arena.seqs();
        let mut layers = Vec::with_capacity(cfg.layers as usize);
        let mut linear_layers = 0usize;
        for (i, kind) in cfg.layer_types.iter().enumerate() {
            let p = format!("layers.{i}");
            match kind {
                LayerKind::Full => layers.push(Layer::Full {
                    w: take_full_layer(&mut w, &p)?,
                    kv: KvPool::new(
                        backend,
                        cfg.kv_heads,
                        cfg.head_dim,
                        seqs,
                        arena.max_pages(),
                        arena.pages(),
                    ),
                }),
                LayerKind::Linear => {
                    linear_layers += 1;
                    layers.push(Layer::Linear {
                        w: take_linear_layer(&mut w, &p)?,
                        state: RecurrentPool::new(
                            backend,
                            seqs,
                            [cfg.lin_val_heads, cfg.lin_key_dim, cfg.lin_val_dim],
                            cfg.conv_dim(),
                        ),
                    })
                }
            }
        }

        let mtp = if cfg.has_mtp {
            Some(Mtp {
                pre_fc_norm_embedding: w.take_tensor("mtp.pre_fc_norm_embedding.weight")?,
                pre_fc_norm_hidden: w.take_tensor("mtp.pre_fc_norm_hidden.weight")?,
                fc: w.take("mtp.fc.weight")?,
                layer: take_full_layer(&mut w, "mtp.layers.0")?,
                norm: w.take_tensor("mtp.norm.weight")?,
                kv: KvPool::new(
                    backend,
                    cfg.kv_heads,
                    cfg.head_dim,
                    seqs,
                    arena.max_pages(),
                    arena.pages(),
                ),
            })
        } else {
            None
        };

        let snap = if cfg.has_mtp {
            (0..linear_layers)
                .map(|_| {
                    RecurrentPool::new(
                        backend,
                        seqs,
                        [cfg.lin_val_heads, cfg.lin_key_dim, cfg.lin_val_dim],
                        cfg.conv_dim(),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        let s = alloc_scratch(&cfg, backend);
        let (cos, sin) = ops::rope_tables(
            backend,
            *arena_spec
                .seq_lens
                .iter()
                .max()
                .expect("budgets are non-empty"),
            cfg.rotary_dim(),
            cfg.rotary_dim(),
            cfg.rope_theta,
            None,
            None,
        );
        let seqs = arena.seqs();
        Ok(Self {
            cfg,
            arena,
            pos: vec![0; seqs as usize],
            saved_pos: vec![0; seqs as usize],
            mtp_pos: vec![0; seqs as usize],
            embed,
            lm_head,
            norm,
            layers,
            mtp,
            s,
            snap,
            cos,
            sin,
        })
    }
}
