use flint_backend::{Backend, Binding, Commands};
use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use flint_model::loader::{self, Plan};
use flint_model::mlp_weights::MlpBlock;
use flint_model::ops::{self, NormMode, NormSpec, RopeArgs, RopeInputs};
use flint_model::pool::KvPool;
use flint_model::routing::Routing;
use flint_model::step;
use flint_model::{ChunkOut, LanguageModel, MAX_M, SeqChunk, Speculator};
use flint_tensor::{Tensor, Weight};

use crate::transformer::config::Config;
use crate::transformer::weights::{LayerW, Scratch, alloc_scratch, take_layer};

pub struct Model {
    cfg: Config,
    slot_lens: Vec<u32>,
    slot_bases: Vec<u32>,
    pos: Vec<u32>,
    saved_pos: Vec<u32>,
    spec_depth: u32,
    embed: Weight,

    head: Option<Weight>,

    lm_bias: Option<Tensor>,
    norm: Tensor,
    norm_bias: Option<Tensor>,
    layers: Vec<LayerW>,

    kv: Vec<KvPool>,
    kv_src: Vec<usize>,

    ones: Tensor,
    s: Scratch,
    capture: Tensor,

    cos: Vec<Tensor>,
    sin: Vec<Tensor>,

    per_layer_emb: Option<Weight>,
    per_layer_proj: Option<Weight>,
    per_layer_norm: Option<Tensor>,

    per_layer_proj_scale: Tensor,
    per_layer_combine_scale: Tensor,
}

impl Model {
    pub fn load(
        source: &dyn Checkpoint,
        cfg: Config,
        plan: &Plan,
        slot_lens: &[u32],
        spec_depth: Option<u32>,
        backend: &Backend,
    ) -> Result<Self> {
        Self::load_extra(source, cfg, plan, Vec::new(), slot_lens, spec_depth, backend)
    }

    pub fn load_extra(
        source: &dyn Checkpoint,
        cfg: Config,
        plan: &Plan,
        extra: Vec<(String, Weight)>,
        slot_lens: &[u32],
        spec_depth: Option<u32>,
        backend: &Backend,
    ) -> Result<Self> {
        cfg.validate()?;
        if slot_lens.is_empty() || slot_lens.contains(&0) {
            return Err(Error::Model("slot lengths must be non-empty and positive".into()));
        }
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
                    slot_lens,
                    cfg.head_dim(l as u32),
                ));
                kv_src[l] = idx;
                last_by_class[(cfg.window(l as u32) > 0) as usize] = Some(idx);
            }
        }

        let mut slot_bases = Vec::with_capacity(slot_lens.len());
        let mut base = 0u32;
        for &len in slot_lens {
            slot_bases.push(base);
            base += len;
        }
        let spec_depth = spec_depth.unwrap_or(cfg.layers / 2).clamp(1, cfg.layers);

        let max_hd = *cfg.head_dims.iter().max().unwrap();
        let ones = backend.tensor_f32(&vec![1.0; max_hd as usize], vec![max_hd]);
        let per_layer_proj_scale =
            backend.tensor_f32(&[(cfg.hidden as f32).sqrt().recip()], vec![1]);
        let per_layer_combine_scale =
            backend.tensor_f32(&[std::f32::consts::SQRT_2.recip()], vec![1]);
        let s = alloc_scratch(&cfg, backend);
        let capture = backend.zero_tensor(&[MAX_M, cfg.hidden]);
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for r in &cfg.rope {
            let (c, s) = ops::rope_tables(
                backend,
                base,
                r.dim,
                r.freq_dim,
                r.theta,
                r.partial,
                r.scaling.as_ref(),
            );
            cos.push(c);
            sin.push(s);
        }
        Ok(Self {
            cfg,
            slot_lens: slot_lens.to_vec(),
            slot_bases,
            pos: vec![0; slot_lens.len()],
            saved_pos: vec![0; slot_lens.len()],
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

    fn head_weight(&self) -> &Weight {
        self.head.as_ref().unwrap_or(&self.embed)
    }

    fn norm_mode(&self) -> NormMode {
        if self.cfg.layernorm {
            NormMode::Layer
        } else {
            NormMode::Direct
        }
    }

    fn norm_bias<'a>(&'a self, b: Option<&'a Tensor>) -> Binding<'a> {
        b.map(Binding::Full).unwrap_or(Binding::Full(&self.ones))
    }

    fn per_layer_embed(
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

    fn per_layer_step(
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

    fn capture_hidden(&self, commands: &mut Commands<'_>, batch: &[SeqChunk]) -> Result<()> {
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
}

struct ResidualSpec<'a> {
    scratch: &'a Scratch,
    post_norm: Option<&'a Tensor>,
    m: u32,
    hidden: u32,
    eps: f32,
}

fn residual_add(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    y: Binding<'_>,
    src: Binding<'_>,
    out: Binding<'_>,
    spec: &ResidualSpec<'_>,
) -> Result<()> {
    match spec.post_norm {
        Some(pn) => {
            ops::norm(
                backend,
                commands,
                &NormSpec::new(NormMode::Direct, spec.m, spec.hidden, spec.eps),
                y,
                pn,
                y,
                Binding::Full(&spec.scratch.normed),
            )?;
            ops::add(backend, commands, src, Binding::Full(&spec.scratch.normed), out, spec.m * spec.hidden)
        }
        None => ops::add(backend, commands, src, y, out, spec.m * spec.hidden),
    }
}

impl LanguageModel for Model {
    fn forward(
        &mut self,
        backend: &mut Backend,
        batch: &[SeqChunk],
    ) -> Result<Vec<ChunkOut>> {
        let m: u32 = batch.iter().map(SeqChunk::len).sum();
        if m == 0 || m > MAX_M {
            return Err(Error::Model(format!("chunk size {m} outside [1, {MAX_M}]")));
        }
        let mut ids = vec![0u32; MAX_M as usize];
        let mut positions = vec![0u32; MAX_M as usize];
        let mut slots = vec![0u32; MAX_M as usize];
        let mut hidden_wanted = false;
        let mut row = 0usize;
        for chunk in batch {
            let s = chunk.slot as usize;
            let (slot_len, slot_base) = (self.slot_lens[s], self.slot_bases[s]);
            if self.pos[s] + chunk.len() > slot_len {
                return Err(Error::Model(format!("context limit {slot_len} reached")));
            }
            for i in 0..chunk.tokens.len() {
                ids[row + i] = chunk.tokens[i];
                positions[row + i] = self.pos[s] + i as u32;
                slots[row + i] = slot_base;
            }
            row += chunk.tokens.len();
            hidden_wanted |= !chunk.hidden_rows.is_empty();
        }
        backend.write_u32(&self.s.ids.buf, &ids);
        step::write_row_meta(backend, &self.s.meta, &positions, &slots, m);

        let cfg = &self.cfg;
        let mut enc = backend.encoder()?;
        {
            let mut commands = Commands::begin(&mut enc);
            let s = &self.s;
            ops::embed(
                backend,
                &mut commands,
                &s.ids,
                &self.embed,
                Binding::Full(&s.hidden),
                &ops::EmbedSpec {
                    rows: m,
                    dim: cfg.hidden,
                    scale: cfg.embed_scale,
                    split: 0,
                },
            )?;
            self.per_layer_embed(backend, &mut commands, s, m)?;

            for (l, lw) in self.layers.iter().enumerate() {
                let hd = cfg.head_dim(l as u32);
                let (nq, nkv) = (cfg.q_heads, cfg.kv_heads);
                ops::norm(
                    backend,
                    &mut commands,
                    &NormSpec::new(self.norm_mode(), m, cfg.hidden, cfg.norm_eps),
                    Binding::Full(&s.hidden),
                    &lw.attn_norm,
                    self.norm_bias(lw.attn_norm_bias.as_ref()),
                    Binding::Full(&s.normed),
                )?;
                let kv_width = match (&lw.k, &lw.v) {
                    (Some(_), Some(_)) => nkv * hd,
                    _ => 0,
                };
                let (yq, yk, yv) = if lw.k.is_some() {
                    (
                        Binding::Full(&lw.q_out),
                        Binding::Full(&lw.k_out),
                        Binding::Full(&lw.v_out),
                    )
                } else {
                    (
                        Binding::Full(&lw.q_out),
                        Binding::Full(&lw.q_out),
                        Binding::Full(&lw.q_out),
                    )
                };
                ops::gemm_qkv(
                    backend,
                    &mut commands,
                    Binding::Full(&s.normed),
                    &ops::QkvSpec {
                        wq: &lw.q,
                        wk: lw.k.as_ref().unwrap_or(&lw.q),
                        wv: lw.v.as_ref().unwrap_or(&lw.q),
                        yq,
                        yk,
                        yv,
                        rows: m,
                        kv_width,
                    },
                )?;

                if let (Some(qb), Some(kb), Some(vb)) = (&lw.q_bias, &lw.k_bias, &lw.v_bias) {
                    ops::bias(backend, &mut commands, Binding::Full(&lw.q_out), qb, m, nq * hd)?;
                    ops::bias(
                        backend,
                        &mut commands,
                        Binding::Full(&lw.k_out),
                        kb,
                        m,
                        nkv * hd,
                    )?;
                    ops::bias(
                        backend,
                        &mut commands,
                        Binding::Full(&lw.v_out),
                        vb,
                        m,
                        nkv * hd,
                    )?;
                }

                let ri = cfg.layer_rope[l];
                let (cos, sin) = (&self.cos[ri as usize], &self.sin[ri as usize]);
                let rot = cfg.rope[ri as usize].dim;
                let qk_has_kv = lw.k.is_some();
                let (q_src, k_src): (&Tensor, Option<&Tensor>) = match (&lw.q_norm, &lw.k_norm) {
                    (Some(qn), Some(kn)) if qk_has_kv => {
                        let rope = RopeInputs {
                            cos,
                            sin,
                            args: &s.meta,
                        };
                        ops::norm_rope(
                            backend,
                            &mut commands,
                            &ops::NormRopeSpec { rows: m * nq, dim: hd, eps: cfg.norm_eps, heads: nq, rot },
                            Binding::Full(&lw.q_out),
                            qn,
                            Binding::Full(&lw.q_normed),
                            &rope,
                        )?;
                        ops::norm_rope(
                            backend,
                            &mut commands,
                            &ops::NormRopeSpec { rows: m * nkv, dim: hd, eps: cfg.norm_eps, heads: nkv, rot },
                            Binding::Full(&lw.k_out),
                            kn,
                            Binding::Full(&lw.k_normed),
                            &rope,
                        )?;
                        (&lw.q_normed, Some(&lw.k_normed))
                    }
                    (Some(qn), _) => {
                        let rope = RopeInputs {
                            cos,
                            sin,
                            args: &s.meta,
                        };
                        ops::norm_rope(
                            backend,
                            &mut commands,
                            &ops::NormRopeSpec { rows: m * nq, dim: hd, eps: cfg.norm_eps, heads: nq, rot },
                            Binding::Full(&lw.q_out),
                            qn,
                            Binding::Full(&lw.q_normed),
                            &rope,
                        )?;
                        (&lw.q_normed, None)
                    }
                    _ => (&lw.q_out, lw.k.as_ref().map(|_| &lw.k_out)),
                };

                if cfg.v_norm && lw.k.is_some() {
                    ops::norm(
                        backend,
                        &mut commands,
                        &NormSpec::new(NormMode::Direct, m * nkv, hd, cfg.norm_eps),
                        Binding::Full(&lw.v_out),
                        &self.ones,
                        Binding::Full(&lw.v_out),
                        Binding::Full(&lw.v_normed),
                    )?;
                }

                let qk_fused = lw.q_norm.is_some();
                if !qk_fused {
                    let rope = RopeInputs {
                        cos,
                        sin,
                        args: &s.meta,
                    };
                    ops::rope(
                        backend,
                        &mut commands,
                        Binding::Full(q_src),
                        &rope,
                        &RopeArgs {
                            heads: nq,
                            head_dim: hd,
                            rot,
                            m,
                        },
                    )?;
                    if let Some(k_src) = k_src {
                        ops::rope(
                            backend,
                            &mut commands,
                            Binding::Full(k_src),
                            &rope,
                            &RopeArgs {
                                heads: nkv,
                                head_dim: hd,
                                rot,
                                m,
                            },
                        )?;
                    }
                }

                let kvs = self.kv_src[l];
                let kv = &self.kv[kvs];
                if let Some(k_src) = k_src {
                    ops::kv_store(
                        backend,
                        &mut commands,
                        Binding::Full(k_src),
                        Binding::Full(if cfg.v_norm { &lw.v_normed } else { &lw.v_out }),
                        kv,
                        m,
                        &s.meta,
                    )?;
                }

                let qw = nq * hd;
                let mut row_off = 0u32;
                for chunk in batch {
                    let m_s = chunk.len();
                    let span = m_s as u64 * qw as u64 * 4;
                    ops::attn(
                        backend,
                        &mut commands,
                        Binding::Slice(q_src, row_off as u64 * qw as u64 * 4, span),
                        kv,
                        Binding::Slice(&lw.attn_out, row_off as u64 * qw as u64 * 4, span),
                        &ops::AttnSpec {
                            q_heads: nq,
                            window: cfg.window(l as u32),
                            scale: cfg.attn_scale.unwrap_or_else(|| (hd as f32).sqrt().recip()),
                            m: m_s,
                            causal: true,
                            slot: self.slot_bases[chunk.slot as usize],
                            args: Binding::Slice(&s.meta, row_off as u64 * 8, m_s as u64 * 8),
                        },
                    )?;
                    row_off += m_s;
                }

                let attn_fused = lw.post_attn_norm.is_none();
                ops::gemm_acc(
                    backend,
                    &mut commands,
                    Binding::Full(&lw.attn_out),
                    &lw.o,
                    Binding::Full(if attn_fused {
                        &s.hidden
                    } else {
                        &s.mlp.down_out
                    }),
                    m,
                    attn_fused,
                )?;
                if let Some(ob) = &lw.o_bias {
                    ops::bias(
                        backend,
                        &mut commands,
                        Binding::Full(if attn_fused {
                            &s.hidden
                        } else {
                            &s.mlp.down_out
                        }),
                        ob,
                        m,
                        cfg.hidden,
                    )?;
                }

                residual_add(
                    backend,
                    &mut commands,
                    Binding::Full(&s.mlp.down_out),
                    Binding::Full(&s.hidden),
                    Binding::Full(&s.hidden2),
                    &ResidualSpec {
                        scratch: s,
                        post_norm: lw.post_attn_norm.as_ref(),
                        m,
                        hidden: cfg.hidden,
                        eps: cfg.norm_eps,
                    },
                )?;

                let mlp_src = if attn_fused {
                    Binding::Full(&s.hidden)
                } else {
                    Binding::Full(&s.hidden2)
                };
                ops::norm(
                    backend,
                    &mut commands,
                    &NormSpec::new(self.norm_mode(), m, cfg.hidden, cfg.norm_eps),
                    mlp_src,
                    lw.mlp.norm(),
                    self.norm_bias(lw.mlp.norm_bias()),
                    Binding::Full(&s.normed),
                )?;

                match &lw.mlp {
                    MlpBlock::Dense(mlp) => {
                        let ffn_fused = attn_fused && lw.post_ffn_norm.is_none();
                        let y = Binding::Full(if ffn_fused {
                            &s.hidden
                        } else {
                            &s.mlp.down_out
                        });
                        ops::swiglu_mlp(
                            backend,
                            &mut commands,
                            Binding::Full(&s.normed),
                            mlp,
                            &s.mlp,
                            y,
                            &ops::MlpSpec {
                                rows: m,
                                intermediate: cfg.mlp_width(l as u32),
                                act: cfg.act,
                                acc: ffn_fused,
                            },
                        )?;
                        if !ffn_fused {
                            residual_add(
                                backend,
                                &mut commands,
                                Binding::Full(&s.mlp.down_out),
                                Binding::Full(&s.hidden2),
                                Binding::Full(&s.hidden),
                                &ResidualSpec {
                                    scratch: s,
                                    post_norm: lw.post_ffn_norm.as_ref(),
                                    m,
                                    hidden: cfg.hidden,
                                    eps: cfg.norm_eps,
                                },
                            )?;
                        }
                    }
                    MlpBlock::Moe(moe) => {
                        let moe_cfg = cfg.moe.expect("MoE block without MoE config");
                        let mt = s.moe.as_ref().expect("MoE block without MoE scratch");

                        ops::gemm(
                            backend,
                            &mut commands,
                            Binding::Full(&s.normed),
                            &moe.router,
                            Binding::Full(&mt.logits),
                            m,
                        )?;
                        backend.submit(commands.raw())?;
                        let logits = backend.read_f32(
                            &mt.logits.buf,
                            0,
                            (m * moe_cfg.experts) as usize,
                        )?;
                        let r = Routing::new(
                            &logits,
                            m,
                            moe_cfg.experts,
                            moe.top_k,
                            moe_cfg.kind,
                            moe.shared_scale,
                        );
                        backend.write_u32(&mt.rows.buf, &r.rows);
                        backend.write_f32(&mt.weights.buf, &r.weights);
                        ops::zero_rows(
                            backend,
                            &mut commands,
                            Binding::Full(&mt.acc),
                            m * cfg.hidden,
                        )?;
                        ops::moe_apply(
                            backend,
                            &mut commands,
                            Binding::Full(&s.normed),
                            moe,
                            mt,
                            &r,
                            &ops::MoeSpec {
                                intermediate: cfg.intermediate,
                                act: cfg.act,
                                hidden: cfg.hidden,
                            },
                        )?;
                        residual_add(
                            backend,
                            &mut commands,
                            Binding::Full(&mt.acc),
                            Binding::Full(&s.hidden2),
                            Binding::Full(&s.hidden),
                            &ResidualSpec {
                                scratch: s,
                                post_norm: lw.post_ffn_norm.as_ref(),
                                m,
                                hidden: cfg.hidden,
                                eps: cfg.norm_eps,
                            },
                        )?;
                    }
                }

                self.per_layer_step(backend, &mut commands, s, lw, l, m)?;

                if l as u32 + 1 == self.spec_depth && hidden_wanted {
                    self.capture_hidden(&mut commands, batch)?;
                }
            }

            ops::norm(
                backend,
                &mut commands,
                &NormSpec::new(self.norm_mode(), m, cfg.hidden, cfg.norm_eps),
                Binding::Full(&s.hidden),
                &self.norm,
                self.norm_bias(self.norm_bias.as_ref()),
                Binding::Full(&s.normed),
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&s.normed),
                self.head_weight(),
                Binding::Full(&s.logits),
                m,
            )?;
            if let Some(lb) = &self.lm_bias {
                ops::bias(
                    backend,
                    &mut commands,
                    Binding::Full(&s.logits),
                    lb,
                    m,
                    cfg.vocab,
                )?;
            }
            if let Some(cap) = cfg.softcap {
                ops::softcap(
                    backend,
                    &mut commands,
                    Binding::Full(&s.logits),
                    m * cfg.vocab,
                    cap,
                )?;
            }
        }
        backend.submit(&mut enc)?;

        let mut outs = Vec::with_capacity(batch.len());
        let mut base = 0u32;
        for chunk in batch {
            let m_s = chunk.len();
            let s = chunk.slot as usize;
            outs.push(ChunkOut {
                logits: step::read_rows(backend, &self.s.logits, chunk.logit_rows, m_s, cfg.vocab, base)?,
                hidden: step::read_rows(backend, &self.capture, chunk.hidden_rows, m_s, cfg.hidden, base)?,
            });
            base += m_s;
            self.pos[s] += m_s;
        }
        Ok(outs)
    }

    fn reset(&mut self, backend: &Backend, slot: u32) -> Result<()> {
        for kv in &self.kv {
            kv.reset(backend, slot)?;
        }
        self.pos[slot as usize] = 0;
        self.saved_pos[slot as usize] = 0;
        Ok(())
    }

    fn pos(&self, slot: u32) -> u32 {
        self.pos[slot as usize]
    }
    fn slot_len(&self, slot: u32) -> u32 {
        self.slot_lens[slot as usize]
    }
    fn slot_count(&self) -> u32 {
        self.slot_lens.len() as u32
    }
    fn vocab(&self) -> u32 {
        self.cfg.vocab
    }
    fn eos(&self) -> &[u32] {
        &self.cfg.eos
    }

    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        (self.spec_depth < self.cfg.layers).then_some(self as &mut dyn Speculator)
    }
}

impl Speculator for Model {
    fn draft(
        &mut self,
        backend: &mut Backend,
        _slot: u32,
        _token: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        assert_eq!(
            hidden.len(),
            self.cfg.hidden as usize,
            "hidden size mismatch"
        );
        backend.write_f32(&self.s.hidden.buf, hidden);
        let cfg = &self.cfg;
        let mut enc = backend.encoder()?;
        {
            let mut commands = Commands::begin(&mut enc);
            ops::norm(
                backend,
                &mut commands,
                &NormSpec::new(self.norm_mode(), 1, cfg.hidden, cfg.norm_eps),
                Binding::Full(&self.s.hidden),
                &self.norm,
                self.norm_bias(self.norm_bias.as_ref()),
                Binding::Full(&self.s.normed),
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&self.s.normed),
                self.head_weight(),
                Binding::Full(&self.s.logits),
                1,
            )?;
            if let Some(lb) = &self.lm_bias {
                ops::bias(
                    backend,
                    &mut commands,
                    Binding::Full(&self.s.logits),
                    lb,
                    1,
                    cfg.vocab,
                )?;
            }
            if let Some(cap) = cfg.softcap {
                ops::softcap(
                    backend,
                    &mut commands,
                    Binding::Full(&self.s.logits),
                    cfg.vocab,
                    cap,
                )?;
            }
        }
        backend.submit(&mut enc)?;
        backend.read_f32(&self.s.logits.buf, 0, cfg.vocab as usize)
    }

    fn prime(&mut self, _slot: u32) {}

    fn snapshot(&mut self, _backend: &Backend, slot: u32) {
        self.saved_pos[slot as usize] = self.pos[slot as usize];
    }

    fn restore(&mut self, _backend: &Backend, slot: u32) {
        self.pos[slot as usize] = self.saved_pos[slot as usize] + 1;
    }
}
