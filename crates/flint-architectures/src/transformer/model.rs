use flint_backend::{Backend, Binding, Pass};
use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use flint_model::blocks::MlpBlock;
use flint_model::cache::KvCache;
use flint_model::loader::{self, Plan};
use flint_model::ops::{self, NormMode};
use flint_model::routing::Routing;
use flint_model::step::{self, MAX_M};
use flint_model::{ChunkOut, LanguageModel};
use flint_tensor::{Tensor, Weight};

use crate::transformer::config::TransformerConfig;
use crate::transformer::weights::{LayerW, Scratch, alloc_scratch, take_layer};

pub struct TransformerModel {
    cfg: TransformerConfig,
    max_seq: u32,
    pos: u32,
    embed: Weight,

    head: Option<Weight>,

    lm_bias: Option<Tensor>,
    norm: Tensor,
    norm_bias: Option<Tensor>,
    layers: Vec<LayerW>,

    kv: Vec<KvCache>,
    kv_src: Vec<usize>,

    ones: Tensor,
    s: Scratch,

    cos: Vec<Tensor>,
    sin: Vec<Tensor>,

    per_layer_emb: Option<Weight>,
    per_layer_proj: Option<Weight>,
    per_layer_norm: Option<Tensor>,

    per_layer_proj_scale: Tensor,
    per_layer_combine_scale: Tensor,
}

impl TransformerModel {
    pub fn load(
        source: &dyn Checkpoint,
        cfg: TransformerConfig,
        plan: &Plan,
        max_seq: u32,
        backend: &Backend,
    ) -> Result<Self> {
        Self::load_extra(source, cfg, plan, Vec::new(), max_seq, backend)
    }

    pub fn load_extra(
        source: &dyn Checkpoint,
        cfg: TransformerConfig,
        plan: &Plan,
        extra: Vec<(String, Weight)>,
        max_seq: u32,
        backend: &Backend,
    ) -> Result<Self> {
        cfg.validate()?;
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
                kv.push(KvCache::new(
                    backend,
                    cfg.kv_heads,
                    max_seq,
                    cfg.head_dim(l as u32),
                ));
                kv_src[l] = idx;
                last_by_class[(cfg.window(l as u32) > 0) as usize] = Some(idx);
            }
        }

        let max_hd = *cfg.head_dims.iter().max().unwrap();
        let ones = backend.tensor_f32(&vec![1.0; max_hd as usize], vec![max_hd]);
        let per_layer_proj_scale =
            backend.tensor_f32(&[(cfg.hidden as f32).sqrt().recip()], vec![1]);
        let per_layer_combine_scale =
            backend.tensor_f32(&[std::f32::consts::SQRT_2.recip()], vec![1]);
        let s = alloc_scratch(&cfg, backend);
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for r in &cfg.rope {
            let (c, s) = ops::rope_tables(
                backend,
                max_seq,
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
            max_seq,
            pos: 0,
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
        pass: &mut Pass<'_>,
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
            pass,
            &s.ids,
            pe,
            pe.tensor().shape[0] / 2,
            Binding::Full(pt),
            m,
            pd,
            embed_scale,
        )?;
        ops::gemm(
            backend,
            pass,
            Binding::Full(&s.hidden),
            pp,
            Binding::Full(pc),
            m,
        )?;
        ops::mul(
            backend,
            pass,
            Binding::Full(pc),
            Binding::Full(&self.per_layer_proj_scale),
            Binding::Full(pc),
            m * pd,
            1,
        )?;
        ops::norm_per_layer(
            backend,
            pass,
            Binding::Full(pc),
            pn,
            Binding::Full(pc),
            m * self.cfg.layers,
            per_layer.dim,
            self.cfg.norm_eps,
            self.cfg.layers,
            pd,
        )?;
        ops::add(
            backend,
            pass,
            Binding::Full(pc),
            Binding::Full(pt),
            Binding::Full(po),
            m * pd,
        )?;
        ops::mul(
            backend,
            pass,
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
        pass: &mut Pass<'_>,
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
            pass,
            Binding::Full(&s.hidden),
            gate,
            Binding::Full(pg),
            m,
        )?;
        ops::swiglu(
            backend,
            pass,
            Binding::Full(pg),
            Binding::Full(pon),
            Binding::Full(pc),
            m * pd,
            self.cfg.act,
        )?;
        ops::row_mul(
            backend,
            pass,
            Binding::Full(pc),
            Binding::Full(po),
            Binding::Full(pg),
            m,
            pd,
            pd * self.cfg.layers,
            l as u32 * pd,
        )?;
        ops::gemm(
            backend,
            pass,
            Binding::Full(pg),
            proj,
            Binding::Full(&s.mlp.down_out),
            m,
        )?;
        ops::norm(
            backend,
            pass,
            NormMode::Direct,
            Binding::Full(&s.mlp.down_out),
            pn,
            Binding::Full(&s.mlp.down_out),
            Binding::Full(&s.normed),
            m,
            self.cfg.hidden,
            self.cfg.hidden,
            self.cfg.norm_eps,
        )?;
        if let Some(os) = &lw.out_scale {
            ops::add(
                backend,
                pass,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                Binding::Full(&s.hidden2),
                m * self.cfg.hidden,
            )?;
            ops::mul(
                backend,
                pass,
                Binding::Full(&s.hidden2),
                Binding::Full(os),
                Binding::Full(&s.hidden),
                m * self.cfg.hidden,
                1,
            )?;
        } else {
            ops::add(
                backend,
                pass,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                Binding::Full(&s.hidden),
                m * self.cfg.hidden,
            )?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn residual_add(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    s: &Scratch,
    post_norm: Option<&Tensor>,
    y: Binding<'_>,
    src: Binding<'_>,
    out: Binding<'_>,
    m: u32,
    hidden: u32,
    eps: f32,
) -> Result<()> {
    match post_norm {
        Some(pn) => {
            ops::norm(
                backend,
                pass,
                NormMode::Direct,
                y,
                pn,
                y,
                Binding::Full(&s.normed),
                m,
                hidden,
                hidden,
                eps,
            )?;
            ops::add(
                backend,
                pass,
                src,
                Binding::Full(&s.normed),
                out,
                m * hidden,
            )
        }
        None => ops::add(backend, pass, src, y, out, m * hidden),
    }
}

impl LanguageModel for TransformerModel {
    fn forward(
        &mut self,
        backend: &mut Backend,
        tokens: &[u32],
        logit_rows: &[u32],
        hidden_rows: &[u32],
    ) -> Result<ChunkOut> {
        let m = tokens.len() as u32;
        if m == 0 || m > MAX_M {
            return Err(Error::Model(format!("chunk size {m} outside [1, {MAX_M}]")));
        }
        if self.pos + m > self.max_seq {
            return Err(Error::Model(format!(
                "context limit {} reached",
                self.max_seq
            )));
        }
        let mut ids = vec![0u32; MAX_M as usize];
        ids[..tokens.len()].copy_from_slice(tokens);
        backend.write_u32(self.s.ids.buf.as_ref(), &ids);
        step::write_step_args(backend, &self.s.args, self.pos, self.pos + m);

        let cfg = &self.cfg;
        let mut enc = backend.encoder()?;
        {
            let mut pass = Pass::begin(enc.as_mut());
            let s = &self.s;
            ops::embed(
                backend,
                &mut pass,
                &s.ids,
                &self.embed,
                Binding::Full(&s.hidden),
                m,
                cfg.hidden,
                cfg.embed_scale,
            )?;
            self.per_layer_embed(backend, &mut pass, s, m)?;

            for (l, lw) in self.layers.iter().enumerate() {
                let hd = cfg.head_dim(l as u32);
                let (nq, nkv) = (cfg.q_heads, cfg.kv_heads);
                ops::norm(
                    backend,
                    &mut pass,
                    self.norm_mode(),
                    Binding::Full(&s.hidden),
                    &lw.attn_norm,
                    self.norm_bias(lw.attn_norm_bias.as_ref()),
                    Binding::Full(&s.normed),
                    m,
                    cfg.hidden,
                    cfg.hidden,
                    cfg.norm_eps,
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
                    &mut pass,
                    Binding::Full(&s.normed),
                    &lw.q,
                    lw.k.as_ref().unwrap_or(&lw.q),
                    lw.v.as_ref().unwrap_or(&lw.q),
                    yq,
                    yk,
                    yv,
                    m,
                    kv_width,
                )?;

                if let (Some(qb), Some(kb), Some(vb)) = (&lw.q_bias, &lw.k_bias, &lw.v_bias) {
                    ops::bias(backend, &mut pass, Binding::Full(&lw.q_out), qb, m, nq * hd)?;
                    ops::bias(
                        backend,
                        &mut pass,
                        Binding::Full(&lw.k_out),
                        kb,
                        m,
                        nkv * hd,
                    )?;
                    ops::bias(
                        backend,
                        &mut pass,
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
                        ops::norm_rope(
                            backend,
                            &mut pass,
                            Binding::Full(&lw.q_out),
                            qn,
                            Binding::Full(&lw.q_normed),
                            m * nq,
                            hd,
                            cfg.norm_eps,
                            nq,
                            rot,
                            cos,
                            sin,
                            &self.s.args,
                        )?;
                        ops::norm_rope(
                            backend,
                            &mut pass,
                            Binding::Full(&lw.k_out),
                            kn,
                            Binding::Full(&lw.k_normed),
                            m * nkv,
                            hd,
                            cfg.norm_eps,
                            nkv,
                            rot,
                            cos,
                            sin,
                            &self.s.args,
                        )?;
                        (&lw.q_normed, Some(&lw.k_normed))
                    }
                    (Some(qn), _) => {
                        ops::norm_rope(
                            backend,
                            &mut pass,
                            Binding::Full(&lw.q_out),
                            qn,
                            Binding::Full(&lw.q_normed),
                            m * nq,
                            hd,
                            cfg.norm_eps,
                            nq,
                            rot,
                            cos,
                            sin,
                            &self.s.args,
                        )?;
                        (&lw.q_normed, None)
                    }
                    _ => (&lw.q_out, lw.k.as_ref().map(|_| &lw.k_out)),
                };

                if cfg.v_norm && lw.k.is_some() {
                    ops::norm(
                        backend,
                        &mut pass,
                        NormMode::Direct,
                        Binding::Full(&lw.v_out),
                        &self.ones,
                        Binding::Full(&lw.v_out),
                        Binding::Full(&lw.v_normed),
                        m * nkv,
                        hd,
                        hd,
                        cfg.norm_eps,
                    )?;
                }

                let qk_fused = lw.q_norm.is_some();
                if !qk_fused {
                    ops::rope(
                        backend,
                        &mut pass,
                        cos,
                        sin,
                        Binding::Full(q_src),
                        nq,
                        hd,
                        rot,
                        m,
                        &self.s.args,
                    )?;
                    if let Some(k_src) = k_src {
                        ops::rope(
                            backend,
                            &mut pass,
                            cos,
                            sin,
                            Binding::Full(k_src),
                            nkv,
                            hd,
                            rot,
                            m,
                            &self.s.args,
                        )?;
                    }
                }

                let kvs = self.kv_src[l];
                let kv = &self.kv[kvs];
                if let Some(k_src) = k_src {
                    ops::kv_store(
                        backend,
                        &mut pass,
                        Binding::Full(k_src),
                        Binding::Full(if cfg.v_norm { &lw.v_normed } else { &lw.v_out }),
                        &kv.k,
                        &kv.v,
                        nkv,
                        hd,
                        kv.max_seq,
                        &self.s.args,
                        m,
                    )?;
                }

                ops::attn(
                    backend,
                    &mut pass,
                    Binding::Full(q_src),
                    &kv.k,
                    &kv.v,
                    &s.attn_scratch,
                    Binding::Full(&lw.attn_out),
                    nq,
                    nkv,
                    hd,
                    kv.max_seq,
                    &self.s.args,
                    m,
                    cfg.window(l as u32),
                    s.attn_stride,
                    cfg.attn_scale.unwrap_or_else(|| (hd as f32).sqrt().recip()),
                )?;

                let attn_fused = lw.post_attn_norm.is_none();
                ops::gemm_acc(
                    backend,
                    &mut pass,
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
                        &mut pass,
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
                    &mut pass,
                    s,
                    lw.post_attn_norm.as_ref(),
                    Binding::Full(&s.mlp.down_out),
                    Binding::Full(&s.hidden),
                    Binding::Full(&s.hidden2),
                    m,
                    cfg.hidden,
                    cfg.norm_eps,
                )?;

                let mlp_src = if attn_fused {
                    Binding::Full(&s.hidden)
                } else {
                    Binding::Full(&s.hidden2)
                };
                ops::norm(
                    backend,
                    &mut pass,
                    self.norm_mode(),
                    mlp_src,
                    lw.mlp.norm(),
                    self.norm_bias(lw.mlp.norm_bias()),
                    Binding::Full(&s.normed),
                    m,
                    cfg.hidden,
                    cfg.hidden,
                    cfg.norm_eps,
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
                            &mut pass,
                            Binding::Full(&s.normed),
                            mlp,
                            &s.mlp,
                            m,
                            cfg.mlp_width(l as u32),
                            cfg.act,
                            y,
                            ffn_fused,
                        )?;
                        if !ffn_fused {
                            residual_add(
                                backend,
                                &mut pass,
                                s,
                                lw.post_ffn_norm.as_ref(),
                                Binding::Full(&s.mlp.down_out),
                                Binding::Full(&s.hidden2),
                                Binding::Full(&s.hidden),
                                m,
                                cfg.hidden,
                                cfg.norm_eps,
                            )?;
                        }
                    }
                    MlpBlock::Moe(moe) => {
                        let moe_cfg = cfg.moe.expect("MoE block without MoE config");
                        let mt = s.moe.as_ref().expect("MoE block without MoE scratch");

                        ops::gemm(
                            backend,
                            &mut pass,
                            Binding::Full(&s.normed),
                            &moe.router,
                            Binding::Full(&mt.logits),
                            m,
                        )?;
                        drop(pass);
                        backend.submit(enc)?;
                        let logits = backend.read_f32(
                            mt.logits.buf.as_ref(),
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
                        backend.write_u32(mt.rows.buf.as_ref(), &r.rows);
                        backend.write_f32(mt.weights.buf.as_ref(), &r.weights);
                        enc = backend.encoder()?;
                        pass = Pass::begin(enc.as_mut());
                        ops::zero_rows(backend, &mut pass, Binding::Full(&mt.acc), m * cfg.hidden)?;
                        ops::moe_apply(
                            backend,
                            &mut pass,
                            Binding::Full(&s.normed),
                            moe,
                            mt,
                            &r,
                            cfg.intermediate,
                            cfg.act,
                            cfg.hidden,
                        )?;
                        residual_add(
                            backend,
                            &mut pass,
                            s,
                            lw.post_ffn_norm.as_ref(),
                            Binding::Full(&mt.acc),
                            Binding::Full(&s.hidden2),
                            Binding::Full(&s.hidden),
                            m,
                            cfg.hidden,
                            cfg.norm_eps,
                        )?;
                    }
                }

                self.per_layer_step(backend, &mut pass, s, lw, l, m)?;
            }

            ops::norm(
                backend,
                &mut pass,
                self.norm_mode(),
                Binding::Full(&s.hidden),
                &self.norm,
                self.norm_bias(self.norm_bias.as_ref()),
                Binding::Full(&s.normed),
                m,
                cfg.hidden,
                cfg.hidden,
                cfg.norm_eps,
            )?;
            ops::gemm(
                backend,
                &mut pass,
                Binding::Full(&s.normed),
                self.head_weight(),
                Binding::Full(&s.logits),
                m,
            )?;
            if let Some(lb) = &self.lm_bias {
                ops::bias(
                    backend,
                    &mut pass,
                    Binding::Full(&s.logits),
                    lb,
                    m,
                    cfg.vocab,
                )?;
            }
            if let Some(cap) = cfg.softcap {
                ops::softcap(
                    backend,
                    &mut pass,
                    Binding::Full(&s.logits),
                    m * cfg.vocab,
                    cap,
                )?;
            }
        }
        backend.submit(enc)?;

        let out = ChunkOut {
            logits: step::read_rows(backend, &self.s.logits, logit_rows, m, cfg.vocab)?,
            hidden: step::read_rows(backend, &self.s.hidden, hidden_rows, m, cfg.hidden)?,
        };
        self.pos += m;
        Ok(out)
    }

    fn reset(&mut self, backend: &Backend) {
        for kv in &self.kv {
            kv.zero(backend);
        }
        self.pos = 0;
    }

    fn pos(&self) -> u32 {
        self.pos
    }
    fn max_seq(&self) -> u32 {
        self.max_seq
    }
    fn vocab(&self) -> u32 {
        self.cfg.vocab
    }
    fn eos(&self) -> &[u32] {
        &self.cfg.eos
    }
}
