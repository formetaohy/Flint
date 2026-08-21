use thuban_backend::{Backend, Binding, Commands};
use thuban_checkpoint::Checkpoint;
use thuban_error::Result;
use thuban_model::loader::{self, Plan};
use thuban_model::ops::{self, NormMode, NormSpec};
use thuban_model::pool::{ArenaSpec, KvArena, KvPool};
use thuban_model::{MAX_M, SeqChunk};
use thuban_tensor::{DType, Tensor, Weight};

use crate::transformer::config::Config;
use crate::transformer::weights::{LayerW, Scratch, alloc_scratch, take_layer};

pub struct Model {
    pub(super) cfg: Config,
    pub(super) arena: KvArena,
    pub(super) pos: Vec<u32>,
    pub(super) saved_pos: Vec<u32>,
    pub(super) spec_depth: u32,
    pub(super) embed: Weight,

    pub(super) head: Option<Weight>,

    pub(super) lm_bias: Option<Tensor>,
    pub(super) norm: Tensor,
    pub(super) norm_bias: Option<Tensor>,
    pub(super) layers: Vec<LayerW>,

    pub(super) kv: Vec<KvPool>,
    pub(super) kv_src: Vec<usize>,

    pub(super) ones: Tensor,
    pub(super) s: Scratch,
    pub(super) capture: Tensor,

    pub(super) cos: Vec<Tensor>,
    pub(super) sin: Vec<Tensor>,

    pub(super) per_layer_emb: Option<Weight>,
    pub(super) per_layer_proj: Option<Weight>,
    pub(super) per_layer_norm: Option<Tensor>,

    pub(super) per_layer_proj_scale: Tensor,
    pub(super) per_layer_combine_scale: Tensor,
}

impl Model {
    pub fn load(
        source: &dyn Checkpoint,
        cfg: Config,
        plan: &Plan,
        arena: &ArenaSpec,
        spec_depth: Option<u32>,
        backend: &Backend,
    ) -> Result<Self> {
        Self::load_extra(source, cfg, plan, Vec::new(), arena, spec_depth, backend)
    }

    pub fn load_extra(
        source: &dyn Checkpoint,
        cfg: Config,
        plan: &Plan,
        extra: Vec<(String, Weight)>,
        arena_spec: &ArenaSpec,
        spec_depth: Option<u32>,
        backend: &Backend,
    ) -> Result<Self> {
        cfg.validate()?;
        let arena = KvArena::new(arena_spec)?;
        let mut w = loader::load_weights(backend, source, plan)?;
        for (key, weight) in extra {
            w.insert(key, weight);
        }
        let embed = w.take("embed_tokens.weight")?;
        let head = if cfg.tied {
            None
        } else {
            Some(w.take("lm_head.weight")?)
        };
        let norm = w.take_tensor("norm.weight")?;
        let norm_bias = if cfg.layernorm {
            Some(w.take_tensor("norm.bias")?)
        } else {
            None
        };
        let lm_bias = if cfg.lm_bias {
            Some(w.take_tensor("lm_head.bias")?)
        } else {
            None
        };
        let (per_layer_emb, per_layer_proj, per_layer_norm) = if cfg.has_ple() {
            (
                Some(w.take("embed_tokens_per_layer.weight")?),
                Some(w.take("per_layer_model_projection.weight")?),
                Some(w.take_tensor("per_layer_projection_norm.weight")?),
            )
        } else {
            (None, None, None)
        };
        let layers = (0..cfg.layers)
            .map(|l| take_layer(&mut w, &cfg, l, backend))
            .collect::<Result<Vec<_>>>()?;

        let first_shared = cfg.first_shared();
        let mut kv = Vec::new();
        let mut kv_src = vec![0usize; cfg.layers as usize];
        let mut last_by_class: [Option<usize>; 2] = [None, None];
        for l in 0..cfg.layers as usize {
            if l as u32 >= first_shared {
                let class = (cfg.window(l as u32) > 0) as usize;
                kv_src[l] = last_by_class[class].expect("KV-shared layer without a source");
            } else {
                let idx = kv.len();
                kv.push(KvPool::new(
                    backend,
                    cfg.kv_heads,
                    cfg.head_dim(l as u32),
                    arena.seqs(),
                    arena.max_pages(),
                    arena.pages(),
                ));
                kv_src[l] = idx;
                last_by_class[(cfg.window(l as u32) > 0) as usize] = Some(idx);
            }
        }

        let spec_depth = spec_depth.unwrap_or(cfg.layers / 2).clamp(1, cfg.layers);

        let max_hd = *cfg.head_dims.iter().max().unwrap();
        let ones = backend.tensor_f32(&vec![1.0; max_hd as usize], vec![max_hd]);
        let per_layer_proj_scale =
            backend.tensor_f32(&[(cfg.hidden as f32).sqrt().recip()], vec![1]);
        let per_layer_combine_scale =
            backend.tensor_f32(&[std::f32::consts::SQRT_2.recip()], vec![1]);
        let s = alloc_scratch(&cfg, backend);
        let capture = backend.zero_tensor(&[MAX_M, cfg.hidden], DType::F32);
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for r in &cfg.rope {
            let (c, s) = ops::rope_tables(
                backend,
                *arena_spec
                    .seq_lens
                    .iter()
                    .max()
                    .expect("budgets are non-empty"),
                r.dim,
                r.freq_dim,
                r.theta,
                r.partial,
                r.scaling.as_ref(),
            );
            cos.push(c);
            sin.push(s);
        }
        let seqs = arena.seqs();
        Ok(Self {
            cfg,
            arena,
            pos: vec![0; seqs as usize],
            saved_pos: vec![0; seqs as usize],
            spec_depth,
            embed,
            head,
            lm_bias,
            norm,
            norm_bias,
            layers,
            kv,
            kv_src,
            ones,
            s,
            capture,
            cos,
            sin,
            per_layer_emb,
            per_layer_proj,
            per_layer_norm,
            per_layer_proj_scale,
            per_layer_combine_scale,
        })
    }

    pub(super) fn head_weight(&self) -> &Weight {
        self.head.as_ref().unwrap_or(&self.embed)
    }

    pub(super) fn norm_mode(&self) -> NormMode {
        if self.cfg.layernorm {
            NormMode::Layer
        } else {
            NormMode::Direct
        }
    }

    pub(super) fn norm_bias<'a>(&'a self, b: Option<&'a Tensor>) -> Binding<'a> {
        b.map(Binding::Full).unwrap_or(Binding::Full(&self.ones))
    }

    pub(super) fn per_layer_embed(
        &self,
        backend: &mut Backend,
        commands: &mut Commands<'_>,
        s: &Scratch,
        m: u32,
    ) -> Result<()> {
        let Some(per_layer) = self.cfg.per_layer else {
            return Ok(());
        };
        let (pe, pp, pn) = (
            self.per_layer_emb.as_ref().unwrap(),
            self.per_layer_proj.as_ref().unwrap(),
            self.per_layer_norm.as_ref().unwrap(),
        );
        let (pt, pc, po) = (
            s.per_layer_tok.as_ref().unwrap(),
            s.per_layer_ctx.as_ref().unwrap(),
            s.per_layer_out.as_ref().unwrap(),
        );
        let pd = per_layer.dim * self.cfg.layers;
        let embed_scale = (per_layer.dim as f32).sqrt();
        ops::embed_split(
            backend,
            commands,
            &s.ids,
            pe,
            Binding::Full(pt),
            &ops::EmbedSpec {
                rows: m,
                dim: pd,
                scale: embed_scale,
                split: pe.tensor().shape[0] / 2,
            },
        )?;
        ops::gemm(
            backend,
            commands,
            Binding::Full(&s.hidden),
            pp,
            Binding::Full(pc),
            m,
        )?;
        ops::mul(
            backend,
            commands,
            Binding::Full(pc),
            Binding::Full(&self.per_layer_proj_scale),
            Binding::Full(pc),
            m * pd,
            1,
        )?;
        ops::norm(
            backend,
            commands,
            &NormSpec {
                ple: 1,
                ple_layers: self.cfg.layers,
                ple_stride: pd,
                ..NormSpec::new(
                    NormMode::Direct,
                    m * self.cfg.layers,
                    per_layer.dim,
                    self.cfg.norm_eps,
                )
            },
            Binding::Full(pc),
            pn,
            Binding::Full(pc),
            Binding::Full(pc),
        )?;
        ops::add(
            backend,
            commands,
            Binding::Full(pc),
            Binding::Full(pt),
            Binding::Full(po),
            m * pd,
        )?;
        ops::mul(
            backend,
            commands,
            Binding::Full(po),
            Binding::Full(&self.per_layer_combine_scale),
            Binding::Full(po),
            m * pd,
            1,
        )?;
        Ok(())
    }

    pub(super) fn per_layer_step(
        &self,
        backend: &mut Backend,
        commands: &mut Commands<'_>,
        s: &Scratch,
        lw: &LayerW,
        l: usize,
        m: u32,
    ) -> Result<()> {
        let Some(per_layer) = self.cfg.per_layer else {
            return Ok(());
        };
        let (Some(gate), Some(proj), Some(pn)) =
            (&lw.per_layer_gate, &lw.per_layer_proj, &lw.per_layer_norm)
        else {
            return Ok(());
        };
        let (po, pc, pg, pon) = (
            s.per_layer_out.as_ref().unwrap(),
            s.per_layer_ctx.as_ref().unwrap(),
            s.per_layer_gate.as_ref().unwrap(),
            s.per_layer_ones.as_ref().unwrap(),
        );
        let pd = per_layer.dim;
        ops::gemm(
            backend,
            commands,
            Binding::Full(&s.hidden),
            gate,
            Binding::Full(pg),
            m,
        )?;
        ops::swiglu(
            backend,
            commands,
            Binding::Full(pg),
            Binding::Full(pon),
            Binding::Full(pc),
            m * pd,
            self.cfg.act,
        )?;
        ops::row_mul(
            backend,
            commands,
            Binding::Full(pc),
            Binding::Full(po),
            Binding::Full(pg),
            &ops::RowMulSpec {
                rows: m,
                cols: pd,
                stride: pd * self.cfg.layers,
                offset: l as u32 * pd,
            },
        )?;
        ops::gemm(
            backend,
            commands,
            Binding::Full(pg),
            proj,
            Binding::Full(&s.mlp.down_out),
            m,
        )?;
        ops::norm(
            backend,
            commands,
            &NormSpec::new(NormMode::Direct, m, self.cfg.hidden, self.cfg.norm_eps),
            Binding::Full(&s.mlp.down_out),
            pn,
            Binding::Full(&s.mlp.down_out),
            Binding::Full(&s.normed),
        )?;
        if let Some(os) = &lw.out_scale {
            ops::add(
                backend,
                commands,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                Binding::Full(&s.hidden2),
                m * self.cfg.hidden,
            )?;
            ops::mul(
                backend,
                commands,
                Binding::Full(&s.hidden2),
                Binding::Full(os),
                Binding::Full(&s.hidden),
                m * self.cfg.hidden,
                1,
            )?;
        } else {
            ops::add(
                backend,
                commands,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                Binding::Full(&s.hidden),
                m * self.cfg.hidden,
            )?;
        }
        Ok(())
    }

    pub(super) fn capture_hidden(
        &self,
        commands: &mut Commands<'_>,
        batch: &[SeqChunk],
    ) -> Result<()> {
        let size = self.cfg.hidden as u64 * 4;
        let mut base = 0u32;
        for chunk in batch {
            for &r in chunk.hidden_rows {
                commands.raw().copy(
                    &self.s.hidden.buf,
                    (base + r) as u64 * size,
                    &self.capture.buf,
                    (base + r) as u64 * size,
                    size,
                )?;
            }
            base += chunk.len();
        }
        Ok(())
    }

    pub(super) fn upload_tables(&self, backend: &Backend) {
        let table = self.arena.table();
        for kv in &self.kv {
            kv.upload(backend, &table);
        }
    }

    pub fn used_pages(&self) -> u32 {
        self.arena.used()
    }
}
