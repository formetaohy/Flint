use flint_backend::{Backend, Binding, Pass};
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::{Error, Result};
use flint_model::cache::{KvCache, RecurrentState};
use flint_model::config::{check_gemm_dims, check_head_dim, f64_field, req, u32_field, u32_list};
use flint_model::loader::{self, Plan, Role, SwigluMlp, WeightSet, take_mlp};
use flint_model::ops::{self, Act, M_MAX, MlpTiles, NormMode};
use flint_model::{ChunkOut, LanguageModel, Speculator};
use flint_tensor::{Tensor, Weight};
use serde_json::Value;

use crate::names::hf_key;

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
        if t.q_heads / t.kv_heads > ops::MAX_GQA {
            return Err(Error::Config(format!(
                "GQA ratio {} exceeds the attention shader's {} head slots",
                t.q_heads / t.kv_heads,
                ops::MAX_GQA
            )));
        }
        check_head_dim(t.head_dim)?;
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
        check_gemm_dims(&[
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
        kv: KvCache,
    },
    Linear {
        w: Box<LinearLayerW>,
        state: RecurrentState,
    },
}

struct Mtp {
    pre_fc_norm_embedding: Tensor,
    pre_fc_norm_hidden: Tensor,
    fc: Weight,
    layer: Box<FullLayerW>,
    norm: Tensor,
    kv: KvCache,
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
        mlp: take_mlp(w, p, false)?,
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
        mlp: take_mlp(w, p, false)?,
    }))
}

struct Scratch {
    ids: Tensor,

    args: Tensor,
    hidden: Tensor,
    hidden2: Tensor,
    normed: Tensor,
    qkv_raw: Tensor,
    convd: Tensor,

    qk_exp: Tensor,
    z: Tensor,
    b: Tensor,
    a: Tensor,
    beta: Tensor,
    g: Tensor,
    attn_out: Tensor,
    attn_out2: Tensor,

    attn_scratch: Tensor,
    qg: Tensor,
    q: Tensor,
    q2: Tensor,
    gate: Tensor,
    k1: Tensor,
    k2: Tensor,
    v1: Tensor,
    mlp: MlpTiles,
    logits: Tensor,
    mtp_e: Tensor,
    mtp_h: Tensor,
    mtp_cat: Tensor,
}

fn alloc_scratch(cfg: &Qwen35Config, backend: &Backend) -> Scratch {
    let h = cfg.hidden;
    let i = cfg.intermediate;
    let z = |shape: &[u32]| backend.zero_tensor(shape);
    Scratch {
        ids: ops::token_ids(backend),
        args: ops::step_args(backend),
        hidden: z(&[M_MAX, h]),
        hidden2: z(&[M_MAX, h]),
        normed: z(&[M_MAX, h]),
        qkv_raw: z(&[M_MAX, cfg.conv_dim()]),
        convd: z(&[M_MAX, cfg.conv_dim()]),
        qk_exp: z(&[M_MAX, cfg.qk_exp_dim()]),
        z: z(&[M_MAX, cfg.value_dim()]),
        b: z(&[M_MAX, cfg.lin_val_heads]),
        a: z(&[M_MAX, cfg.lin_val_heads]),
        beta: z(&[cfg.lin_val_heads]),
        g: z(&[cfg.lin_val_heads]),
        attn_out: z(&[M_MAX, cfg.value_dim()]),
        attn_out2: z(&[M_MAX, cfg.value_dim()]),
        attn_scratch: z(&[
            M_MAX,
            cfg.kv_heads,
            ops::ATTN_SEGS,
            ops::MAX_GQA,
            cfg.head_dim + 2,
        ]),
        qg: z(&[M_MAX, cfg.q_heads * cfg.head_dim * 2]),
        q: z(&[M_MAX, cfg.q_heads * cfg.head_dim]),
        q2: z(&[M_MAX, cfg.q_heads * cfg.head_dim]),
        gate: z(&[M_MAX, cfg.q_heads * cfg.head_dim]),
        k1: z(&[M_MAX, cfg.kv_heads * cfg.head_dim]),
        k2: z(&[M_MAX, cfg.kv_heads * cfg.head_dim]),
        v1: z(&[M_MAX, cfg.kv_heads * cfg.head_dim]),
        mlp: MlpTiles {
            gate_out: z(&[M_MAX, i]),
            up_out: z(&[M_MAX, i]),
            act: z(&[M_MAX, i]),
            down_out: z(&[M_MAX, h]),
        },
        logits: z(&[M_MAX, cfg.vocab]),
        mtp_e: z(&[M_MAX, h]),
        mtp_h: z(&[M_MAX, h]),
        mtp_cat: z(&[M_MAX, 2 * h]),
    }
}

pub struct Qwen35 {
    cfg: Qwen35Config,
    max_seq: u32,
    pos: u32,
    embed: Weight,

    lm_head: Option<Weight>,
    norm: Tensor,
    layers: Vec<Layer>,
    mtp: Option<Mtp>,
    s: Scratch,

    snap: Vec<RecurrentState>,
    cos: Tensor,
    sin: Tensor,
    mtp_pos: u32,
    saved_pos: u32,
}

impl Qwen35 {

    fn logit_head(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    pub fn load(
        source: &dyn Checkpoint,
        v: &Value,
        max_seq: u32,
        backend: &Backend,
    ) -> Result<Self> {
        if source.kind() == CheckpointKind::Gguf {
            return Err(Error::Model("Qwen3.5 has no GGUF representation".into()));
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

        let mut layers = Vec::with_capacity(cfg.layers as usize);
        let mut linear_layers = 0usize;
        for (i, kind) in cfg.layer_types.iter().enumerate() {
            let p = format!("layers.{i}");
            match kind {
                LayerKind::Full => layers.push(Layer::Full {
                    w: take_full_layer(&mut w, &p)?,
                    kv: KvCache::new(backend, cfg.kv_heads, max_seq, cfg.head_dim),
                }),
                LayerKind::Linear => {
                    linear_layers += 1;
                    layers.push(Layer::Linear {
                        w: take_linear_layer(&mut w, &p)?,
                        state: RecurrentState::new(
                            backend,
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
                kv: KvCache::new(backend, cfg.kv_heads, max_seq, cfg.head_dim),
            })
        } else {
            None
        };

        let snap = if cfg.has_mtp {
            (0..linear_layers)
                .map(|_| {
                    RecurrentState::new(
                        backend,
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
            max_seq,
            cfg.rotary_dim(),
            cfg.rotary_dim(),
            cfg.rope_theta,
            None,
        );
        Ok(Self {
            cfg,
            max_seq,
            pos: 0,
            embed,
            lm_head,
            norm,
            layers,
            mtp,
            s,
            snap,
            cos,
            sin,
            mtp_pos: 0,
            saved_pos: 0,
        })
    }
}

impl LanguageModel for Qwen35 {
    fn forward(
        &mut self,
        backend: &mut Backend,
        tokens: &[u32],
        logit_rows: &[u32],
        hidden_rows: &[u32],
    ) -> Result<ChunkOut> {
        let m = tokens.len() as u32;
        if m == 0 || m > M_MAX {
            return Err(Error::Model(format!("chunk size {m} outside [1, {M_MAX}]")));
        }
        if self.pos + m > self.max_seq {
            return Err(Error::Model(format!(
                "context limit {} reached",
                self.max_seq
            )));
        }
        let mut ids = vec![0u32; M_MAX as usize];
        ids[..tokens.len()].copy_from_slice(tokens);
        backend.write_u32(self.s.ids.buf.as_ref(), &ids);
        ops::write_step_args(backend, &self.s.args, self.pos, self.pos + m);

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
                1.0,
            )?;

            for layer in &self.layers {
                match layer {
                    Layer::Full { w, kv } => full_layer(
                        backend, &mut pass, cfg, s, &self.cos, &self.sin, w, kv, m, &s.args,
                    )?,
                    Layer::Linear { w, state } => {
                        linear_layer(backend, &mut pass, cfg, s, w, state, m)?
                    }
                }
            }

            ops::norm(
                backend,
                &mut pass,
                NormMode::Offset,
                Binding::Full(&s.hidden),
                &self.norm,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                m,
                cfg.hidden,
                cfg.hidden,
                1e-6,
            )?;
            ops::gemm(
                backend,
                &mut pass,
                Binding::Full(&s.normed),
                self.logit_head(),
                Binding::Full(&s.logits),
                m,
            )?;
        }
        backend.submit(enc)?;

        let out = ChunkOut {
            logits: ops::read_rows(backend, &self.s.logits, logit_rows, m, cfg.vocab)?,
            hidden: ops::read_rows(backend, &self.s.hidden, hidden_rows, m, cfg.hidden)?,
        };
        self.pos += m;
        backend.flush_profile()?;
        Ok(out)
    }

    fn reset(&mut self, backend: &Backend) {
        for layer in &self.layers {
            match layer {
                Layer::Full { kv, .. } => kv.zero(backend),
                Layer::Linear { state, .. } => state.zero(backend),
            }
        }
        if let Some(mtp) = &self.mtp {
            mtp.kv.zero(backend);
        }
        self.pos = 0;
        self.mtp_pos = 0;
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

    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        self.mtp.is_some().then_some(self as &mut dyn Speculator)
    }
}

impl Speculator for Qwen35 {
    fn draft(&mut self, backend: &mut Backend, token: u32, hidden: &[f32]) -> Result<Vec<f32>> {
        let Some(mtp) = self.mtp.as_ref() else {
            unreachable!("the speculator exists only with an MTP head");
        };
        assert_eq!(
            hidden.len(),
            self.cfg.hidden as usize,
            "hidden size mismatch"
        );
        assert!(self.mtp_pos <= self.pos, "draft head ran past the target");
        if self.mtp_pos >= self.max_seq {
            return Err(Error::Model(format!(
                "context limit {} reached",
                self.max_seq
            )));
        }

        let mut ids = vec![0u32; M_MAX as usize];
        ids[0] = token;
        backend.write_u32(self.s.ids.buf.as_ref(), &ids);
        ops::write_step_args(backend, &self.s.args, self.mtp_pos, self.mtp_pos + 1);
        backend.write_f32(self.s.mtp_h.buf.as_ref(), hidden);

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
                1,
                cfg.hidden,
                1.0,
            )?;
            ops::norm(
                backend,
                &mut pass,
                NormMode::Offset,
                Binding::Full(&s.hidden),
                &mtp.pre_fc_norm_embedding,
                Binding::Full(&s.hidden),
                Binding::Full(&s.mtp_e),
                1,
                cfg.hidden,
                cfg.hidden,
                1e-6,
            )?;
            ops::norm(
                backend,
                &mut pass,
                NormMode::Offset,
                Binding::Full(&s.mtp_h),
                &mtp.pre_fc_norm_hidden,
                Binding::Full(&s.mtp_h),
                Binding::Full(&s.normed),
                1,
                cfg.hidden,
                cfg.hidden,
                1e-6,
            )?;
            ops::concat(
                backend,
                &mut pass,
                Binding::Full(&s.mtp_e),
                Binding::Full(&s.normed),
                Binding::Full(&s.mtp_cat),
                1,
                cfg.hidden,
            )?;
            ops::gemm(
                backend,
                &mut pass,
                Binding::Full(&s.mtp_cat),
                &mtp.fc,
                Binding::Full(&s.hidden),
                1,
            )?;

            full_layer(
                backend, &mut pass, cfg, s, &self.cos, &self.sin, &mtp.layer, &mtp.kv, 1, &s.args,
            )?;

            ops::norm(
                backend,
                &mut pass,
                NormMode::Offset,
                Binding::Full(&s.hidden),
                &mtp.norm,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                1,
                cfg.hidden,
                cfg.hidden,
                1e-6,
            )?;
            ops::gemm(
                backend,
                &mut pass,
                Binding::Full(&s.normed),
                self.logit_head(),
                Binding::Full(&s.logits),
                1,
            )?;
        }
        backend.submit(enc)?;

        let logits = backend.read_f32(self.s.logits.buf.as_ref(), 0, cfg.vocab as usize)?;
        self.mtp_pos += 1;
        backend.flush_profile()?;
        Ok(logits)
    }

    fn prime(&mut self) {
        self.mtp_pos = self.pos;
    }

    fn snapshot(&mut self, backend: &Backend) {
        self.saved_pos = self.pos;
        let mut si = 0;
        for layer in &self.layers {
            if let Layer::Linear { state, .. } = layer {
                self.snap[si].copy_from(backend, state);
                si += 1;
            }
        }
    }

    fn restore(&mut self, backend: &Backend) {
        self.pos = self.saved_pos;
        let mut si = 0;
        for layer in &self.layers {
            if let Layer::Linear { state, .. } = layer {
                state.copy_from(backend, &self.snap[si]);
                si += 1;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn full_layer(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    cos: &Tensor,
    sin: &Tensor,
    w: &FullLayerW,
    kv: &KvCache,
    m: u32,
    args: &Tensor,
) -> Result<()> {
    ops::norm(
        backend,
        pass,
        NormMode::Offset,
        Binding::Full(&s.hidden),
        &w.attn_norm,
        Binding::Full(&s.hidden),
        Binding::Full(&s.normed),
        m,
        cfg.hidden,
        cfg.hidden,
        1e-6,
    )?;
    full_attn_block(backend, pass, cfg, s, cos, sin, w, kv, m, args)?;
    ops::add(
        backend,
        pass,
        Binding::Full(&s.hidden),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.hidden2),
        m * cfg.hidden,
    )?;
    post_mlp(backend, pass, cfg, s, &w.mlp, m)
}

fn linear_layer(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    w: &LinearLayerW,
    state: &RecurrentState,
    m: u32,
) -> Result<()> {
    ops::norm(
        backend,
        pass,
        NormMode::Offset,
        Binding::Full(&s.hidden),
        &w.attn_norm,
        Binding::Full(&s.hidden),
        Binding::Full(&s.normed),
        m,
        cfg.hidden,
        cfg.hidden,
        1e-6,
    )?;
    linear_attn_block(backend, pass, cfg, s, w, state, m)?;
    ops::add(
        backend,
        pass,
        Binding::Full(&s.hidden),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.hidden2),
        m * cfg.hidden,
    )?;
    post_mlp(backend, pass, cfg, s, &w.mlp, m)
}

#[allow(clippy::too_many_arguments)]
fn full_attn_block(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    cos: &Tensor,
    sin: &Tensor,
    w: &FullLayerW,
    kv: &KvCache,
    m: u32,
    args: &Tensor,
) -> Result<()> {
    let (nq, nkv, hd) = (cfg.q_heads, cfg.kv_heads, cfg.head_dim);

    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.normed),
        &w.q,
        Binding::Full(&s.qg),
        m,
    )?;
    ops::split_qg(
        backend,
        pass,
        Binding::Full(&s.qg),
        Binding::Full(&s.q),
        Binding::Full(&s.gate),
        m,
        nq,
        hd,
    )?;
    ops::norm(
        backend,
        pass,
        NormMode::Offset,
        Binding::Full(&s.q),
        &w.q_norm,
        Binding::Full(&s.q),
        Binding::Full(&s.q2),
        m * nq,
        hd,
        hd,
        1e-6,
    )?;
    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.normed),
        &w.k,
        Binding::Full(&s.k1),
        m,
    )?;
    ops::norm(
        backend,
        pass,
        NormMode::Offset,
        Binding::Full(&s.k1),
        &w.k_norm,
        Binding::Full(&s.k1),
        Binding::Full(&s.k2),
        m * nkv,
        hd,
        hd,
        1e-6,
    )?;
    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.normed),
        &w.v,
        Binding::Full(&s.v1),
        m,
    )?;

    ops::rope(
        backend,
        pass,
        cos,
        sin,
        Binding::Full(&s.q2),
        nq,
        hd,
        cfg.rotary_dim(),
        m,
        args,
    )?;
    ops::rope(
        backend,
        pass,
        cos,
        sin,
        Binding::Full(&s.k2),
        nkv,
        hd,
        cfg.rotary_dim(),
        m,
        args,
    )?;
    ops::kv_store(
        backend,
        pass,
        Binding::Full(&s.k2),
        Binding::Full(&s.v1),
        &kv.k,
        &kv.v,
        nkv,
        hd,
        kv.max_seq,
        args,
        m,
    )?;

    ops::attn(
        backend,
        pass,
        Binding::Full(&s.q2),
        &kv.k,
        &kv.v,
        &s.attn_scratch,
        Binding::Full(&s.attn_out),
        nq,
        nkv,
        hd,
        kv.max_seq,
        args,
        m,
        0,
        hd + 2,
    )?;
    ops::sigmoid_mul(
        backend,
        pass,
        Binding::Full(&s.attn_out),
        Binding::Full(&s.gate),
        Binding::Full(&s.attn_out2),
        m * nq * hd,
    )?;
    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.attn_out2),
        &w.o,
        Binding::Full(&s.mlp.down_out),
        m,
    )
}

fn linear_attn_block(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    w: &LinearLayerW,
    state: &RecurrentState,
    m: u32,
) -> Result<()> {
    let heads = cfg.lin_val_heads;
    let (key_d, val_d, conv_d) = (cfg.key_dim(), cfg.value_dim(), cfg.conv_dim());

    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.normed),
        &w.in_proj_qkv,
        Binding::Full(&s.qkv_raw),
        m,
    )?;
    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.normed),
        &w.in_proj_z,
        Binding::Full(&s.z),
        m,
    )?;
    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.normed),
        &w.in_proj_b,
        Binding::Full(&s.b),
        m,
    )?;
    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.normed),
        &w.in_proj_a,
        Binding::Full(&s.a),
        m,
    )?;

    let row = |t: u32, stride: u32| t as u64 * stride as u64 * 4;
    for t in 0..m {
        ops::conv1d(
            backend,
            pass,
            Binding::Slice(&s.qkv_raw, row(t, conv_d), conv_d as u64 * 4),
            &w.conv1d,
            &state.conv,
            Binding::Slice(&s.convd, row(t, conv_d), conv_d as u64 * 4),
            conv_d,
        )?;
    }
    ops::repeat_qk(
        backend,
        pass,
        Binding::Full(&s.convd),
        Binding::Full(&s.qk_exp),
        m,
        cfg.lin_key_heads,
        heads,
        cfg.lin_key_dim,
        cfg.lin_val_dim,
    )?;
    let exp_d = cfg.qk_exp_dim();

    let qkb = exp_d as u64 * 2;
    for t in 0..m {

        ops::delta_gate(
            backend,
            pass,
            Binding::Full(&s.b),
            Binding::Full(&s.a),
            &w.a_log,
            &w.dt_bias,
            Binding::Full(&s.beta),
            Binding::Full(&s.g),
            heads,
            t,
        )?;
        let ed = row(t, exp_d);
        let (kb, vb) = (key_d as u64 * 4, val_d as u64 * 4);
        ops::delta_recur(
            backend,
            pass,
            Binding::Slice(&s.qk_exp, ed, qkb),
            Binding::Slice(&s.qk_exp, ed + qkb, qkb),
            Binding::Slice(&s.convd, row(t, conv_d) + kb * 2, vb),
            Binding::Full(&s.beta),
            Binding::Full(&s.g),
            &state.recur,
            Binding::Slice(&s.attn_out, row(t, val_d), vb),
            heads,
            cfg.lin_key_dim,
            cfg.lin_val_dim,
        )?;
    }

    let span = m as u64 * val_d as u64 * 4;
    ops::norm(
        backend,
        pass,
        NormMode::Gated,
        Binding::Slice(&s.attn_out, 0, span),
        &w.norm,
        Binding::Slice(&s.z, 0, span),
        Binding::Slice(&s.attn_out2, 0, span),
        m * heads,
        cfg.lin_val_dim,
        cfg.lin_val_dim,
        1e-6,
    )?;
    ops::gemm(
        backend,
        pass,
        Binding::Full(&s.attn_out2),
        &w.out_proj,
        Binding::Full(&s.mlp.down_out),
        m,
    )
}

fn post_mlp(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    mlp: &SwigluMlp,
    m: u32,
) -> Result<()> {
    ops::norm(
        backend,
        pass,
        NormMode::Offset,
        Binding::Full(&s.hidden2),
        &mlp.norm,
        Binding::Full(&s.hidden2),
        Binding::Full(&s.normed),
        m,
        cfg.hidden,
        cfg.hidden,
        1e-6,
    )?;
    mlp_block(backend, pass, cfg, s, mlp, m)?;
    ops::add(
        backend,
        pass,
        Binding::Full(&s.hidden2),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.hidden),
        m * cfg.hidden,
    )
}

fn mlp_block(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    mlp: &SwigluMlp,
    m: u32,
) -> Result<()> {
    ops::swiglu_mlp(
        backend,
        pass,
        Binding::Full(&s.normed),
        mlp,
        &s.mlp,
        m,
        cfg.intermediate,
        Act::Silu,
        Binding::Full(&s.mlp.down_out),
        false,
    )
}
