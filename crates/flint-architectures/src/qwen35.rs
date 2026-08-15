use flint_backend::{Backend, Binding, Commands};
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::{Error, Result};
use flint_model::config::{f64_field, req, u32_field, u32_list};
use flint_model::loader::{self, Plan, Role, WeightSet};
use flint_model::mlp_weights::{SwigluMlp, take_mlp};
use flint_model::ops::{self, Act, MlpTiles, NormMode};
use flint_model::pool::KvPool;
use flint_model::step;
use flint_model::{ChunkOut, LanguageModel, MAX_M, SeqChunk, Speculator};
use flint_tensor::{Tensor, Weight};
use serde_json::Value;

use crate::keys::hf_key;

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

#[derive(Clone, Copy, PartialEq)]
pub enum LayerKind {
    Linear,
    Full,
}

pub struct Qwen35Config {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub layer_types: Vec<LayerKind>,

    pub lin_key_heads: u32,

    pub lin_val_heads: u32,
    pub lin_key_dim: u32,
    pub lin_val_dim: u32,
    pub rope_theta: f64,
    pub partial_rotary: f64,
    pub eos: Vec<u32>,

    pub tied: bool,
    pub has_mtp: bool,
}

impl Qwen35Config {
    pub fn parse(v: &Value) -> Result<Self> {
        let t = req(v, "text_config")?;
        let tied = req(v, "tie_word_embeddings")?.as_bool().unwrap_or(false);
        let layer_types = req(t, "layer_types")?
            .as_array()
            .ok_or_else(|| Error::Config("layer_types is not an array".into()))?
            .iter()
            .map(|k| match k.as_str() {
                Some("linear_attention") => Ok(LayerKind::Linear),
                Some("full_attention") => Ok(LayerKind::Full),
                other => Err(Error::Config(format!("unknown layer type {other:?}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        let rope = req(t, "rope_parameters")?;
        let mtp = t
            .get("mtp_num_hidden_layers")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if mtp > 1 {
            return Err(Error::Config(format!(
                "mtp_num_hidden_layers {mtp} > 1 unsupported"
            )));
        }
        let cfg = Self {
            hidden: u32_field(t, "hidden_size")?,
            intermediate: u32_field(t, "intermediate_size")?,
            layers: u32_field(t, "num_hidden_layers")?,
            q_heads: u32_field(t, "num_attention_heads")?,
            kv_heads: u32_field(t, "num_key_value_heads")?,
            head_dim: u32_field(t, "head_dim")?,
            vocab: u32_field(t, "vocab_size")?,
            layer_types,
            lin_key_heads: u32_field(t, "linear_num_key_heads")?,
            lin_val_heads: u32_field(t, "linear_num_value_heads")?,
            lin_key_dim: u32_field(t, "linear_key_head_dim")?,
            lin_val_dim: u32_field(t, "linear_value_head_dim")?,
            rope_theta: f64_field(rope, "rope_theta")?,
            partial_rotary: f64_field(rope, "partial_rotary_factor")?,
            eos: u32_list(t, "eos_token_id")?,
            tied,
            has_mtp: mtp == 1,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let t = &self;
        if t.layer_types.len() != t.layers as usize {
            return Err(Error::Config("layer_types length mismatch".into()));
        }
        if !t.q_heads.is_multiple_of(t.kv_heads) {
            return Err(Error::Config("q heads not divisible by kv heads".into()));
        }
        ops::check_head_dim(t.head_dim)?;
        if !t.rotary_dim().is_multiple_of(2) {
            return Err(Error::Config("rotary_dim must be even".into()));
        }
        if t.lin_key_dim > 128 || t.lin_val_dim > 128 {
            return Err(Error::Config("linear head dims must be <= 128".into()));
        }
        if t.lin_key_heads > 256 || t.lin_val_heads > 256 {
            return Err(Error::Config("linear heads must be <= 256".into()));
        }
        if !t.lin_val_heads.is_multiple_of(t.lin_key_heads) {
            return Err(Error::Config(format!(
                "linear value heads {} not divisible by key heads {}",
                t.lin_val_heads, t.lin_key_heads
            )));
        }
        ops::check_gemm_dims(&[
            (t.vocab, t.hidden),
            (t.conv_dim(), t.hidden),
            (t.value_dim(), t.hidden),
            (t.lin_val_heads, t.hidden),
            (t.hidden, t.value_dim()),
            (t.q_heads * t.head_dim * 2, t.hidden),
            (t.kv_heads * t.head_dim, t.hidden),
            (t.hidden, t.q_heads * t.head_dim),
            (t.intermediate, t.hidden),
            (t.hidden, t.intermediate),
            (t.hidden, 2 * t.hidden),
        ])
    }

    pub fn key_dim(&self) -> u32 {
        self.lin_key_heads * self.lin_key_dim
    }

    pub fn value_dim(&self) -> u32 {
        self.lin_val_heads * self.lin_val_dim
    }
    pub fn conv_dim(&self) -> u32 {
        self.key_dim() * 2 + self.value_dim()
    }

    pub fn qk_exp_dim(&self) -> u32 {
        self.lin_val_heads * self.lin_key_dim * 2
    }
    pub fn rotary_dim(&self) -> u32 {
        (self.head_dim as f64 * self.partial_rotary) as u32
    }
}

struct RecurrentPool {
    recur: Tensor,
    conv: Tensor,
    heads: u32,
    key_dim: u32,
    val_dim: u32,
    conv_dim: u32,
}

impl RecurrentPool {
    fn new(backend: &Backend, slots: u32, recur_shape: [u32; 3], conv_dim: u32) -> Self {
        let [heads, key_dim, val_dim] = recur_shape;
        Self {
            recur: backend.zero_tensor(&[slots * heads, key_dim, val_dim]),
            conv: backend.zero_tensor(&[slots, conv_dim, 3]),
            heads,
            key_dim,
            val_dim,
            conv_dim,
        }
    }

    fn zero(&self, backend: &Backend, slot: u32) -> Result<()> {
        let recur_span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        let conv_span = self.conv_dim as u64 * 12;
        let mut enc = backend.encoder()?;
        enc.clear(&self.recur.buf, slot as u64 * recur_span, recur_span)?;
        enc.clear(&self.conv.buf, slot as u64 * conv_span, conv_span)?;
        enc.finish().wait()?;
        Ok(())
    }

    fn copy_slot(&self, backend: &Backend, src: &RecurrentPool, slot: u32) -> Result<()> {
        let recur_span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        let conv_span = self.conv_dim as u64 * 12;
        let mut enc = backend.encoder()?;
        enc.copy(
            &src.recur.buf,
            slot as u64 * recur_span,
            &self.recur.buf,
            slot as u64 * recur_span,
            recur_span,
        )?;
        enc.copy(
            &src.conv.buf,
            slot as u64 * conv_span,
            &self.conv.buf,
            slot as u64 * conv_span,
            conv_span,
        )?;
        enc.finish().wait()?;
        Ok(())
    }

    fn recur_slice(&self, slot: u32) -> Binding<'_> {
        let span = self.heads as u64 * self.key_dim as u64 * self.val_dim as u64 * 4;
        Binding::Slice(&self.recur, slot as u64 * span, span)
    }

    fn conv_slice(&self, slot: u32) -> Binding<'_> {
        let span = self.conv_dim as u64 * 12;
        Binding::Slice(&self.conv, slot as u64 * span, span)
    }
}

struct FullLayerW {
    attn_norm: Tensor,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Tensor,
    k_norm: Tensor,
    mlp: SwigluMlp,
}

struct LinearLayerW {
    attn_norm: Tensor,
    in_proj_qkv: Weight,
    in_proj_z: Weight,
    in_proj_b: Weight,
    in_proj_a: Weight,
    conv1d: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out_proj: Weight,
    mlp: SwigluMlp,
}

enum Layer {
    Full {
        w: Box<FullLayerW>,
        kv: KvPool,
    },
    Linear {
        w: Box<LinearLayerW>,
        state: RecurrentPool,
    },
}

struct Mtp {
    pre_fc_norm_embedding: Tensor,
    pre_fc_norm_hidden: Tensor,
    fc: Weight,
    layer: Box<FullLayerW>,
    norm: Tensor,
    kv: KvPool,
}

fn take_full_layer(w: &mut WeightSet, p: &str) -> Result<Box<FullLayerW>> {
    let k = |n: &str| format!("{p}.{n}");
    Ok(Box::new(FullLayerW {
        attn_norm: w.take_tensor(&k("input_layernorm.weight"))?,
        q: w.take(&k("self_attn.q_proj.weight"))?,
        k: w.take(&k("self_attn.k_proj.weight"))?,
        v: w.take(&k("self_attn.v_proj.weight"))?,
        o: w.take(&k("self_attn.o_proj.weight"))?,
        q_norm: w.take_tensor(&k("self_attn.q_norm.weight"))?,
        k_norm: w.take_tensor(&k("self_attn.k_norm.weight"))?,
        mlp: take_mlp(w, p, false, false)?,
    }))
}

fn take_linear_layer(w: &mut WeightSet, p: &str) -> Result<Box<LinearLayerW>> {
    let k = |n: &str| format!("{p}.linear_attn.{n}");
    Ok(Box::new(LinearLayerW {
        attn_norm: w.take_tensor(&format!("{p}.input_layernorm.weight"))?,
        in_proj_qkv: w.take(&k("in_proj_qkv.weight"))?,
        in_proj_z: w.take(&k("in_proj_z.weight"))?,
        in_proj_b: w.take(&k("in_proj_b.weight"))?,
        in_proj_a: w.take(&k("in_proj_a.weight"))?,
        conv1d: w.take_tensor(&k("conv1d.weight"))?,
        a_log: w.take_tensor(&k("A_log"))?,
        dt_bias: w.take_tensor(&k("dt_bias"))?,
        norm: w.take_tensor(&k("norm.weight"))?,
        out_proj: w.take(&k("out_proj.weight"))?,
        mlp: take_mlp(w, p, false, false)?,
    }))
}

struct Scratch {
    ids: Tensor,

    meta: Tensor,
    hidden: Tensor,
    hidden2: Tensor,
    normed: Tensor,
    qkv_proj: Tensor,
    conv_out: Tensor,

    qk_expanded: Tensor,
    z: Tensor,
    b: Tensor,
    a: Tensor,
    beta: Tensor,
    g: Tensor,
    attn_out: Tensor,
    attn_gated: Tensor,

    qg: Tensor,
    q: Tensor,
    q_normed: Tensor,
    gate: Tensor,
    k_raw: Tensor,
    k_normed: Tensor,
    v_raw: Tensor,
    mlp: MlpTiles,
    logits: Tensor,
    mtp_emb: Tensor,
    mtp_hidden: Tensor,
    mtp_concat: Tensor,
}

fn alloc_scratch(cfg: &Qwen35Config, backend: &Backend) -> Scratch {
    let h = cfg.hidden;
    let i = cfg.intermediate;
    let z = |shape: &[u32]| backend.zero_tensor(shape);
    Scratch {
        ids: step::token_ids(backend),
        meta: step::row_meta(backend),
        hidden: z(&[MAX_M, h]),
        hidden2: z(&[MAX_M, h]),
        normed: z(&[MAX_M, h]),
        qkv_proj: z(&[MAX_M, cfg.conv_dim()]),
        conv_out: z(&[MAX_M, cfg.conv_dim()]),
        qk_expanded: z(&[MAX_M, cfg.qk_exp_dim()]),
        z: z(&[MAX_M, cfg.value_dim()]),
        b: z(&[MAX_M, cfg.lin_val_heads]),
        a: z(&[MAX_M, cfg.lin_val_heads]),
        beta: z(&[cfg.lin_val_heads]),
        g: z(&[cfg.lin_val_heads]),
        attn_out: z(&[MAX_M, cfg.value_dim()]),
        attn_gated: z(&[MAX_M, cfg.value_dim()]),
        qg: z(&[MAX_M, cfg.q_heads * cfg.head_dim * 2]),
        q: z(&[MAX_M, cfg.q_heads * cfg.head_dim]),
        q_normed: z(&[MAX_M, cfg.q_heads * cfg.head_dim]),
        gate: z(&[MAX_M, cfg.q_heads * cfg.head_dim]),
        k_raw: z(&[MAX_M, cfg.kv_heads * cfg.head_dim]),
        k_normed: z(&[MAX_M, cfg.kv_heads * cfg.head_dim]),
        v_raw: z(&[MAX_M, cfg.kv_heads * cfg.head_dim]),
        mlp: MlpTiles {
            gate_out: z(&[MAX_M, i]),
            up_out: z(&[MAX_M, i]),
            act: z(&[MAX_M, i]),
            down_out: z(&[MAX_M, h]),
        },
        logits: z(&[MAX_M, cfg.vocab]),
        mtp_emb: z(&[MAX_M, h]),
        mtp_hidden: z(&[MAX_M, h]),
        mtp_concat: z(&[MAX_M, 2 * h]),
    }
}

pub struct Qwen35 {
    cfg: Qwen35Config,
    slot_lens: Vec<u32>,
    slot_bases: Vec<u32>,
    pos: Vec<u32>,
    saved_pos: Vec<u32>,
    mtp_pos: Vec<u32>,
    embed: Weight,

    lm_head: Option<Weight>,
    norm: Tensor,
    layers: Vec<Layer>,
    mtp: Option<Mtp>,
    s: Scratch,

    snap: Vec<RecurrentPool>,
    cos: Tensor,
    sin: Tensor,
}

impl Qwen35 {
    fn logit_head(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    pub fn load(
        source: &dyn Checkpoint,
        v: &Value,
        slot_lens: &[u32],
        backend: &Backend,
    ) -> Result<Self> {
        if source.kind() == CheckpointKind::Gguf {
            return Err(Error::Model("Qwen3.5 has no GGUF representation".into()));
        }
        if slot_lens.is_empty() || slot_lens.contains(&0) {
            return Err(Error::Model("slot lengths must be non-empty and positive".into()));
        }
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

        let slots = slot_lens.len() as u32;
        let mut layers = Vec::with_capacity(cfg.layers as usize);
        let mut linear_layers = 0usize;
        for (i, kind) in cfg.layer_types.iter().enumerate() {
            let p = format!("layers.{i}");
            match kind {
                LayerKind::Full => layers.push(Layer::Full {
                    w: take_full_layer(&mut w, &p)?,
                    kv: KvPool::new(backend, cfg.kv_heads, slot_lens, cfg.head_dim),
                }),
                LayerKind::Linear => {
                    linear_layers += 1;
                    layers.push(Layer::Linear {
                        w: take_linear_layer(&mut w, &p)?,
                        state: RecurrentPool::new(
                            backend,
                            slots,
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
                kv: KvPool::new(backend, cfg.kv_heads, slot_lens, cfg.head_dim),
            })
        } else {
            None
        };

        let snap = if cfg.has_mtp {
            (0..linear_layers)
                .map(|_| {
                    RecurrentPool::new(
                        backend,
                        slots,
                        [cfg.lin_val_heads, cfg.lin_key_dim, cfg.lin_val_dim],
                        cfg.conv_dim(),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut slot_bases = Vec::with_capacity(slot_lens.len());
        let mut base = 0u32;
        for &len in slot_lens {
            slot_bases.push(base);
            base += len;
        }

        let s = alloc_scratch(&cfg, backend);
        let (cos, sin) = ops::rope_tables(
            backend,
            base,
            cfg.rotary_dim(),
            cfg.rotary_dim(),
            cfg.rope_theta,
            None,
            None,
        );
        Ok(Self {
            cfg,
            slot_lens: slot_lens.to_vec(),
            slot_bases,
            pos: vec![0; slot_lens.len()],
            saved_pos: vec![0; slot_lens.len()],
            mtp_pos: vec![0; slot_lens.len()],
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

impl LanguageModel for Qwen35 {
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
        let mut row = 0usize;
        for chunk in batch {
            let s = chunk.slot as usize;
            if self.pos[s] + chunk.len() > self.slot_lens[s] {
                return Err(Error::Model(format!(
                    "context limit {} reached",
                    self.slot_lens[s]
                )));
            }
            for i in 0..chunk.tokens.len() {
                ids[row + i] = chunk.tokens[i];
                positions[row + i] = self.pos[s] + i as u32;
                slots[row + i] = self.slot_bases[s];
            }
            row += chunk.tokens.len();
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
                    scale: 1.0,
                    split: 0,
                },
            )?;

            let ctx = FullCtx {
                cfg,
                s,
                cos: &self.cos,
                sin: &self.sin,
                slot_bases: &self.slot_bases,
            };
            for layer in &self.layers {
                match layer {
                    Layer::Full { w, kv } => {
                        full_layer(backend, &mut commands, &ctx, w, kv, m, batch)?
                    }
                    Layer::Linear { w, state } => {
                        linear_layer(backend, &mut commands, &ctx, w, state, m, batch)?
                    }
                }
            }

            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, m, cfg.hidden, 1e-6),
                Binding::Full(&s.hidden),
                &self.norm,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&s.normed),
                self.logit_head(),
                Binding::Full(&s.logits),
                m,
            )?;
        }
        backend.submit(&mut enc)?;

        let mut outs = Vec::with_capacity(batch.len());
        let mut base = 0u32;
        for chunk in batch {
            let m_s = chunk.len();
            let s = chunk.slot as usize;
            outs.push(ChunkOut {
                logits: step::read_rows(backend, &self.s.logits, chunk.logit_rows, m_s, cfg.vocab, base)?,
                hidden: step::read_rows(backend, &self.s.hidden, chunk.hidden_rows, m_s, cfg.hidden, base)?,
            });
            base += m_s;
            self.pos[s] += m_s;
        }
        Ok(outs)
    }

    fn reset(&mut self, backend: &Backend, slot: u32) -> Result<()> {
        for layer in &self.layers {
            match layer {
                Layer::Full { kv, .. } => kv.reset(backend, slot)?,
                Layer::Linear { state, .. } => state.zero(backend, slot)?,
            }
        }
        if let Some(mtp) = &self.mtp {
            mtp.kv.reset(backend, slot)?;
        }
        self.pos[slot as usize] = 0;
        self.saved_pos[slot as usize] = 0;
        self.mtp_pos[slot as usize] = 0;
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
        self.mtp.is_some().then_some(self as &mut dyn Speculator)
    }
}

impl Speculator for Qwen35 {
    fn draft(
        &mut self,
        backend: &mut Backend,
        slot: u32,
        token: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let Some(mtp) = self.mtp.as_ref() else {
            unreachable!("the speculator exists only with an MTP head");
        };
        assert_eq!(
            hidden.len(),
            self.cfg.hidden as usize,
            "hidden size mismatch"
        );
        assert!(
            self.mtp_pos[slot as usize] <= self.pos[slot as usize],
            "draft head ran past the target"
        );
        if self.mtp_pos[slot as usize] >= self.slot_lens[slot as usize] {
            return Err(Error::Model(format!(
                "context limit {} reached",
                self.slot_lens[slot as usize]
            )));
        }

        let mut ids = vec![0u32; MAX_M as usize];
        ids[0] = token;
        backend.write_u32(&self.s.ids.buf, &ids);
        step::write_row_meta(
            backend,
            &self.s.meta,
            &[self.mtp_pos[slot as usize]],
            &[self.slot_bases[slot as usize]],
            1,
        );
        backend.write_f32(&self.s.mtp_hidden.buf, hidden);

        let cfg = &self.cfg;
        let chunk = SeqChunk {
            tokens: &ids[..1],
            slot,
            logit_rows: &[],
            hidden_rows: &[],
        };
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
                    rows: 1,
                    dim: cfg.hidden,
                    scale: 1.0,
                    split: 0,
                },
            )?;
            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, 1, cfg.hidden, 1e-6),
                Binding::Full(&s.hidden),
                &mtp.pre_fc_norm_embedding,
                Binding::Full(&s.hidden),
                Binding::Full(&s.mtp_emb),
            )?;
            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, 1, cfg.hidden, 1e-6),
                Binding::Full(&s.mtp_hidden),
                &mtp.pre_fc_norm_hidden,
                Binding::Full(&s.mtp_hidden),
                Binding::Full(&s.normed),
            )?;
            ops::concat(
                backend,
                &mut commands,
                Binding::Full(&s.mtp_emb),
                Binding::Full(&s.normed),
                Binding::Full(&s.mtp_concat),
                1,
                cfg.hidden,
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&s.mtp_concat),
                &mtp.fc,
                Binding::Full(&s.hidden),
                1,
            )?;

            let ctx = FullCtx {
                cfg,
                s,
                cos: &self.cos,
                sin: &self.sin,
                slot_bases: &self.slot_bases,
            };
            full_layer(
                backend, &mut commands, &ctx, &mtp.layer, &mtp.kv, 1, std::slice::from_ref(&chunk),
            )?;

            ops::norm(
                backend,
                &mut commands,
                &ops::NormSpec::new(NormMode::Offset, 1, cfg.hidden, 1e-6),
                Binding::Full(&s.hidden),
                &mtp.norm,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
            )?;
            ops::gemm(
                backend,
                &mut commands,
                Binding::Full(&s.normed),
                self.logit_head(),
                Binding::Full(&s.logits),
                1,
            )?;
        }
        backend.submit(&mut enc)?;

        let logits = backend.read_f32(&self.s.logits.buf, 0, cfg.vocab as usize)?;
        self.mtp_pos[slot as usize] += 1;
        Ok(logits)
    }

    fn advance(
        &mut self,
        backend: &mut Backend,
        slot: u32,
        token: u32,
        hidden: &[f32],
    ) -> Result<()> {
        self.draft(backend, slot, token, hidden).map(|_| ())
    }

    fn prime(&mut self, slot: u32) {
        self.mtp_pos[slot as usize] = self.pos[slot as usize];
    }

    fn snapshot(&mut self, backend: &Backend, slot: u32) {
        self.saved_pos[slot as usize] = self.pos[slot as usize];
        let mut si = 0;
        for layer in &self.layers {
            if let Layer::Linear { state, .. } = layer {
                self.snap[si]
                    .copy_slot(backend, state, slot)
                    .expect("snapshot copy");
                si += 1;
            }
        }
    }

    fn restore(&mut self, backend: &Backend, slot: u32) {
        self.pos[slot as usize] = self.saved_pos[slot as usize] + 1;
        let mut si = 0;
        for layer in &self.layers {
            if let Layer::Linear { state, .. } = layer {
                state
                    .copy_slot(backend, &self.snap[si], slot)
                    .expect("restore copy");
                si += 1;
            }
        }
    }
}

struct FullCtx<'a> {
    cfg: &'a Qwen35Config,
    s: &'a Scratch,
    cos: &'a Tensor,
    sin: &'a Tensor,
    slot_bases: &'a [u32],
}

fn full_layer(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &FullLayerW,
    kv: &KvPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Offset, m, cfg.hidden, 1e-6),
        Binding::Full(&s.hidden),
        &w.attn_norm,
        Binding::Full(&s.hidden),
        Binding::Full(&s.normed),
    )?;
    full_attn_block(backend, commands, ctx, w, kv, m, batch)?;
    ops::add(
        backend,
        commands,
        Binding::Full(&s.hidden),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.hidden2),
        m * cfg.hidden,
    )?;
    post_mlp(backend, commands, cfg, s, &w.mlp, m)
}

fn linear_layer(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &LinearLayerW,
    state: &RecurrentPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Offset, m, cfg.hidden, 1e-6),
        Binding::Full(&s.hidden),
        &w.attn_norm,
        Binding::Full(&s.hidden),
        Binding::Full(&s.normed),
    )?;
    linear_attn_block(backend, commands, ctx, w, state, m, batch)?;
    ops::add(
        backend,
        commands,
        Binding::Full(&s.hidden),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.hidden2),
        m * cfg.hidden,
    )?;
    post_mlp(backend, commands, cfg, s, &w.mlp, m)
}

fn full_attn_block(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &FullLayerW,
    kv: &KvPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    let (nq, nkv, hd) = (cfg.q_heads, cfg.kv_heads, cfg.head_dim);

    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.q,
        Binding::Full(&s.qg),
        m,
    )?;
    ops::split_qg(
        backend,
        commands,
        Binding::Full(&s.qg),
        Binding::Full(&s.q),
        Binding::Full(&s.gate),
        &ops::SplitQgSpec {
            rows: m,
            heads: nq,
            head_dim: hd,
        },
    )?;
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Offset, m * nq, hd, 1e-6),
        Binding::Full(&s.q),
        &w.q_norm,
        Binding::Full(&s.q),
        Binding::Full(&s.q_normed),
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.k,
        Binding::Full(&s.k_raw),
        m,
    )?;
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Offset, m * nkv, hd, 1e-6),
        Binding::Full(&s.k_raw),
        &w.k_norm,
        Binding::Full(&s.k_raw),
        Binding::Full(&s.k_normed),
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.v,
        Binding::Full(&s.v_raw),
        m,
    )?;

    let rope = ops::RopeInputs {
        cos: ctx.cos,
        sin: ctx.sin,
        args: &s.meta,
    };
    ops::rope(
        backend,
        commands,
        Binding::Full(&s.q_normed),
        &rope,
        &ops::RopeArgs {
            heads: nq,
            head_dim: hd,
            rot: cfg.rotary_dim(),
            m,
        },
    )?;
    ops::rope(
        backend,
        commands,
        Binding::Full(&s.k_normed),
        &rope,
        &ops::RopeArgs {
            heads: nkv,
            head_dim: hd,
            rot: cfg.rotary_dim(),
            m,
        },
    )?;
    ops::kv_store(
        backend,
        commands,
        Binding::Full(&s.k_normed),
        Binding::Full(&s.v_raw),
        kv,
        m,
        &s.meta,
    )?;

    let qw = nq * hd;
    let mut row_off = 0u32;
    for chunk in batch {
        let m_s = chunk.len();
        let span = m_s as u64 * qw as u64 * 4;
        ops::attn(
            backend,
            commands,
            Binding::Slice(&s.q_normed, row_off as u64 * qw as u64 * 4, span),
            kv,
            Binding::Slice(&s.attn_out, row_off as u64 * qw as u64 * 4, span),
            &ops::AttnSpec {
                q_heads: nq,
                window: 0,
                scale: (hd as f32).sqrt().recip(),
                m: m_s,
                causal: true,
                slot: ctx.slot_bases[chunk.slot as usize],
                args: Binding::Slice(&s.meta, row_off as u64 * 8, m_s as u64 * 8),
            },
        )?;
        row_off += m_s;
    }
    ops::sigmoid_mul(
        backend,
        commands,
        Binding::Full(&s.attn_out),
        Binding::Full(&s.gate),
        Binding::Full(&s.attn_gated),
        m * nq * hd,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.attn_gated),
        &w.o,
        Binding::Full(&s.mlp.down_out),
        m,
    )
}

fn linear_attn_block(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &LinearLayerW,
    state: &RecurrentPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    let heads = cfg.lin_val_heads;
    let (key_d, val_d, conv_d) = (cfg.key_dim(), cfg.value_dim(), cfg.conv_dim());

    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_qkv,
        Binding::Full(&s.qkv_proj),
        m,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_z,
        Binding::Full(&s.z),
        m,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_b,
        Binding::Full(&s.b),
        m,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_a,
        Binding::Full(&s.a),
        m,
    )?;

    let row = |t: u32, stride: u32| t as u64 * stride as u64 * 4;
    let mut t_off = 0u32;
    for chunk in batch {
        let slot = chunk.slot;
        for _ in 0..chunk.len() {
            let t = t_off;
            ops::conv1d(
                backend,
                commands,
                Binding::Slice(&s.qkv_proj, row(t, conv_d), conv_d as u64 * 4),
                &w.conv1d,
                state.conv_slice(slot),
                Binding::Slice(&s.conv_out, row(t, conv_d), conv_d as u64 * 4),
                &ops::ConvSpec { dim: conv_d },
            )?;
            t_off += 1;
        }
    }
    ops::repeat_qk(
        backend,
        commands,
        Binding::Full(&s.conv_out),
        Binding::Full(&s.qk_expanded),
        &ops::RepeatQkSpec {
            rows: m,
            n_k: cfg.lin_key_heads,
            n_v: heads,
            key_dim: cfg.lin_key_dim,
            val_dim: cfg.lin_val_dim,
        },
    )?;
    let exp_d = cfg.qk_exp_dim();

    let qkb = exp_d as u64 * 2;
    let mut t_off = 0u32;
    for chunk in batch {
        let slot = chunk.slot;
        for _ in 0..chunk.len() {
            let t = t_off;
            ops::delta_gate(
                backend,
                commands,
                &ops::DeltaGate {
                    b: Binding::Full(&s.b),
                    a: Binding::Full(&s.a),
                    a_log: &w.a_log,
                    dt_bias: &w.dt_bias,
                    beta: Binding::Full(&s.beta),
                    g: Binding::Full(&s.g),
                    heads,
                    row: t,
                },
            )?;
            let ed = row(t, exp_d);
            let (kb, vb) = (key_d as u64 * 4, val_d as u64 * 4);
            ops::delta_recur(
                backend,
                commands,
                &ops::DeltaRecur {
                    q: Binding::Slice(&s.qk_expanded, ed, qkb),
                    k: Binding::Slice(&s.qk_expanded, ed + qkb, qkb),
                    v: Binding::Slice(&s.conv_out, row(t, conv_d) + kb * 2, vb),
                    beta: Binding::Full(&s.beta),
                    g: Binding::Full(&s.g),
                    state: state.recur_slice(slot),
                    y: Binding::Slice(&s.attn_out, row(t, val_d), vb),
                    heads,
                    key_dim: cfg.lin_key_dim,
                    val_dim: cfg.lin_val_dim,
                },
            )?;
            t_off += 1;
        }
    }

    let span = m as u64 * val_d as u64 * 4;
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Gated, m * heads, cfg.lin_val_dim, 1e-6),
        Binding::Slice(&s.attn_out, 0, span),
        &w.norm,
        Binding::Slice(&s.z, 0, span),
        Binding::Slice(&s.attn_gated, 0, span),
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.attn_gated),
        &w.out_proj,
        Binding::Full(&s.mlp.down_out),
        m,
    )
}

fn post_mlp(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    mlp: &SwigluMlp,
    m: u32,
) -> Result<()> {
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Offset, m, cfg.hidden, 1e-6),
        Binding::Full(&s.hidden2),
        &mlp.norm,
        Binding::Full(&s.hidden2),
        Binding::Full(&s.normed),
    )?;
    mlp_block(backend, commands, cfg, s, mlp, m)?;
    ops::add(
        backend,
        commands,
        Binding::Full(&s.hidden2),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.hidden),
        m * cfg.hidden,
    )
}

fn mlp_block(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    mlp: &SwigluMlp,
    m: u32,
) -> Result<()> {
    ops::swiglu_mlp(
        backend,
        commands,
        Binding::Full(&s.normed),
        mlp,
        &s.mlp,
        Binding::Full(&s.mlp.down_out),
        &ops::MlpSpec {
            rows: m,
            intermediate: cfg.intermediate,
            act: Act::Silu,
            acc: false,
        },
    )
}
