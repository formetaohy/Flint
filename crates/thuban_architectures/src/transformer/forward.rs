use thuban_backend::{Backend, Binding, Commands};
use thuban_error::{Error, Result};
use thuban_model::ops::{self, NormMode, NormSpec, RopeArgs, RopeInputs};
use thuban_model::rows;
use thuban_model::{ChunkOut, LanguageModel, MAX_M, SeqChunk, Speculator};
use thuban_tensor::Tensor;

use super::model::Model;
use super::weights::Scratch;

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
            ops::add(
                backend,
                commands,
                src,
                Binding::Full(&spec.scratch.normed),
                out,
                spec.m * spec.hidden,
            )
        }
        None => ops::add(backend, commands, src, y, out, spec.m * spec.hidden),
    }
}

impl LanguageModel for Model {
    fn forward(&mut self, backend: &mut Backend, batch: &[SeqChunk]) -> Result<Vec<ChunkOut>> {
        let m: u32 = batch.iter().map(SeqChunk::len).sum();
        if m == 0 || m > MAX_M {
            return Err(Error::Model(format!("chunk size {m} outside [1, {MAX_M}]")));
        }
        let mut ids = vec![0u32; MAX_M as usize];
        let mut positions = vec![0u32; MAX_M as usize];
        let mut seqs = vec![0u32; MAX_M as usize];
        let mut hidden_wanted = false;
        let mut row = 0usize;
        for chunk in batch {
            let s = chunk.seq as usize;
            let limit = self.arena.seq_len(chunk.seq);
            if self.pos[s] + chunk.len() > limit {
                return Err(Error::Model(format!("context limit {limit} reached")));
            }
            assert!(
                self.arena.covers(chunk.seq, self.pos[s] + chunk.len()),
                "pages must cover the chunk"
            );
            for i in 0..chunk.tokens.len() {
                ids[row + i] = chunk.tokens[i];
                positions[row + i] = self.pos[s] + i as u32;
                seqs[row + i] = chunk.seq;
            }
            row += chunk.tokens.len();
            hidden_wanted |= !chunk.hidden_rows.is_empty();
        }
        backend.write_u32(&self.s.ids.buf, &ids);
        rows::write_row_meta(backend, &self.s.meta, &positions, &seqs, m);

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
                    ops::bias(
                        backend,
                        &mut commands,
                        Binding::Full(&lw.q_out),
                        qb,
                        m,
                        nq * hd,
                    )?;
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
                            &ops::NormRopeSpec {
                                rows: m * nq,
                                dim: hd,
                                eps: cfg.norm_eps,
                                heads: nq,
                                rot,
                            },
                            Binding::Full(&lw.q_out),
                            qn,
                            Binding::Full(&lw.q_normed),
                            &rope,
                        )?;
                        ops::norm_rope(
                            backend,
                            &mut commands,
                            &ops::NormRopeSpec {
                                rows: m * nkv,
                                dim: hd,
                                eps: cfg.norm_eps,
                                heads: nkv,
                                rot,
                            },
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
                            &ops::NormRopeSpec {
                                rows: m * nq,
                                dim: hd,
                                eps: cfg.norm_eps,
                                heads: nq,
                                rot,
                            },
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
                            seq: chunk.seq,
                            args: Binding::Slice(&s.meta, row_off as u64 * 32, m_s as u64 * 32),
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
                    &lw.mlp.norm,
                    self.norm_bias(lw.mlp.norm_bias.as_ref()),
                    Binding::Full(&s.normed),
                )?;

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
                    &lw.mlp,
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
            let s = chunk.seq as usize;
            outs.push(ChunkOut {
                logits: rows::read_rows(
                    backend,
                    &self.s.logits,
                    chunk.logit_rows,
                    m_s,
                    cfg.vocab,
                    base,
                )?,
                hidden: rows::read_rows(
                    backend,
                    &self.capture,
                    chunk.hidden_rows,
                    m_s,
                    cfg.hidden,
                    base,
                )?,
            });
            base += m_s;
            self.pos[s] += m_s;
        }
        Ok(outs)
    }

    fn reset(&mut self, backend: &Backend, seq: u32) -> Result<()> {
        self.arena.free_seq(seq);
        self.upload_tables(backend);
        self.pos[seq as usize] = 0;
        self.saved_pos[seq as usize] = 0;
        Ok(())
    }

    fn pos(&self, seq: u32) -> u32 {
        self.pos[seq as usize]
    }
    fn context_limit(&self, seq: u32) -> u32 {
        self.arena.seq_len(seq)
    }
    fn seq_count(&self) -> u32 {
        self.arena.seqs()
    }
    fn vocab(&self) -> u32 {
        self.cfg.vocab
    }
    fn eos(&self) -> &[u32] {
        &self.cfg.eos
    }

    fn alloc_pages(&mut self, backend: &Backend, seq: u32, tokens: u32) -> Result<()> {
        self.arena.alloc(seq, self.pos[seq as usize], tokens)?;
        self.upload_tables(backend);
        Ok(())
    }

    fn free_pages(&mut self, backend: &Backend, seq: u32) -> Result<()> {
        self.arena.free_seq(seq);
        self.upload_tables(backend);
        Ok(())
    }

    fn truncate_pages(&mut self, backend: &Backend, seq: u32, keep_tokens: u32) -> Result<()> {
        self.arena.truncate(seq, keep_tokens);
        self.upload_tables(backend);
        Ok(())
    }

    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        (self.spec_depth < self.cfg.layers).then_some(self as &mut dyn Speculator)
    }
}
