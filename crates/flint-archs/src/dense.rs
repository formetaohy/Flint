//! Dense GQA transformer shared by the LLaMA family (LLaMA / Qwen2 / Qwen3 /
//! Mistral) and Gemma 3. The families differ only in norm placement, attention
//! window, embedding scale and always-on QK-norm — expressed as configuration
//! over one forward graph, not as separate implementations.

use flint_backend::{Backend, Binding};
use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use flint_model::cache::KvCache;
use flint_model::config::{check_gemm_dims, check_head_dim, f64_field, u32_field, u32_list};
use flint_model::loader::Plan;
use flint_model::loader::{self, WeightSet};
use flint_model::ops::{self, NormMode, ROWS};
use flint_model::{ChunkOut, LanguageModel};
use flint_tensor::{DType, Tensor, Weight};
use serde_json::Value;

/// Gemma-style alternating windows: layers whose (l+1) is a multiple of
/// `pattern` attend globally, the rest see only the trailing `size` tokens.
#[derive(Clone, Copy, Debug)]
pub struct SlidingWindow {
    pub size: u32,
    pub pattern: u32,
}

/// Validated dense GQA config covering every supported family.
#[derive(Clone, Debug)]
pub struct DenseConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub rope_theta: f64,
    pub eos: Vec<u32>,
    pub tied: bool,
    /// Input embedding scale (Gemma: sqrt(hidden); everyone else 1).
    pub embed_scale: f32,
    /// Q/K/V projection biases (Qwen2 small models, Phi).
    pub qkv_bias: bool,
    /// Per-head RMSNorm on Q and K before RoPE (Qwen3; always on for Gemma).
    pub qk_norm: bool,
    /// Sandwich norms on attention and MLP outputs before the residual (Gemma 3).
    pub sandwich: bool,
    /// Alternating sliding-window attention; None means every layer is global.
    pub window: Option<SlidingWindow>,
}

impl DenseConfig {
    /// Parses the fields common to every dense family. Family-specific knobs
    /// stay at their LLaMA defaults; callers adjust them, then `validate`.
    pub fn parse(v: &Value, tied_default: bool) -> Result<Self> {
        let hidden = u32_field(v, "hidden_size")?;
        let q_heads = u32_field(v, "num_attention_heads")?;
        let head_dim = match v.get("head_dim").and_then(Value::as_u64) {
            Some(d) => u32::try_from(d).map_err(|_| Error::Config("head_dim overflow".into()))?,
            None => {
                if !hidden.is_multiple_of(q_heads) {
                    return Err(Error::Config(
                        "hidden_size not divisible by num_attention_heads".into(),
                    ));
                }
                hidden / q_heads
            }
        };
        Ok(Self {
            hidden,
            intermediate: u32_field(v, "intermediate_size")?,
            layers: u32_field(v, "num_hidden_layers")?,
            q_heads,
            kv_heads: u32_field(v, "num_key_value_heads")?,
            head_dim,
            vocab: u32_field(v, "vocab_size")?,
            rope_theta: f64_field(v, "rope_theta")?,
            eos: u32_list(v, "eos_token_id")?,
            tied: v
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(tied_default),
            embed_scale: 1.0,
            qkv_bias: false,
            qk_norm: false,
            sandwich: false,
            window: None,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let t = &self;
        if !t.q_heads.is_multiple_of(t.kv_heads) {
            return Err(Error::Config("q heads not divisible by kv heads".into()));
        }
        check_head_dim(t.head_dim)?;
        if let Some(w) = t.window
            && w.pattern == 0
        {
            return Err(Error::Config(
                "sliding_window_pattern must be non-zero".into(),
            ));
        }
        check_gemm_dims(&[
            (t.vocab, t.hidden),
            (t.q_heads * t.head_dim, t.hidden),
            (t.kv_heads * t.head_dim, t.hidden),
            (t.hidden, t.q_heads * t.head_dim),
            (t.intermediate, t.hidden),
            (t.hidden, t.intermediate),
        ])
    }

    /// Attention window for layer `l`: 0 attends to the full causal prefix.
    pub fn window(&self, l: u32) -> u32 {
        match self.window {
            None => 0,
            Some(w) if (l + 1).is_multiple_of(w.pattern) => 0,
            Some(w) => w.size,
        }
    }
}

/// SwiGLU MLP weights plus the norm that feeds it.
struct MlpW {
    norm: Tensor,
    gate: Weight,
    up: Weight,
    down: Weight,
}

struct LayerW {
    attn_norm: Tensor,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    /// Sandwich norm on the attention output (Gemma 3).
    post_attn_norm: Option<Tensor>,
    mlp: MlpW,
    /// Sandwich norm on the MLP output (Gemma 3).
    post_ffn_norm: Option<Tensor>,
}

fn take_mlp(w: &mut WeightSet, p: &str) -> Result<MlpW> {
    let k = |n: &str| format!("{p}.{n}");
    Ok(MlpW {
        norm: w.take_tensor(&k("post_attention_layernorm.weight"))?,
        gate: w.take(&k("mlp.gate_proj.weight"))?,
        up: w.take(&k("mlp.up_proj.weight"))?,
        down: w.take(&k("mlp.down_proj.weight"))?,
    })
}

fn take_optional(w: &mut WeightSet, on: bool, key: &str) -> Result<Option<Tensor>> {
    if on {
        Ok(Some(w.take_tensor(key)?))
    } else {
        Ok(None)
    }
}

fn take_layer(w: &mut WeightSet, cfg: &DenseConfig, l: u32) -> Result<LayerW> {
    let k = |n: &str| format!("layers.{l}.{n}");
    Ok(LayerW {
        attn_norm: w.take_tensor(&k("input_layernorm.weight"))?,
        q: w.take(&k("self_attn.q_proj.weight"))?,
        k: w.take(&k("self_attn.k_proj.weight"))?,
        v: w.take(&k("self_attn.v_proj.weight"))?,
        o: w.take(&k("self_attn.o_proj.weight"))?,
        q_bias: take_optional(w, cfg.qkv_bias, &k("self_attn.q_proj.bias"))?,
        k_bias: take_optional(w, cfg.qkv_bias, &k("self_attn.k_proj.bias"))?,
        v_bias: take_optional(w, cfg.qkv_bias, &k("self_attn.v_proj.bias"))?,
        q_norm: take_optional(w, cfg.qk_norm, &k("self_attn.q_norm.weight"))?,
        k_norm: take_optional(w, cfg.qk_norm, &k("self_attn.k_norm.weight"))?,
        post_attn_norm: take_optional(w, cfg.sandwich, &k("post_attention_norm.weight"))?,
        mlp: take_mlp(w, &format!("layers.{l}"))?,
        post_ffn_norm: take_optional(w, cfg.sandwich, &k("post_ffw_norm.weight"))?,
    })
}

/// Per-forward scratch tiles, all [ROWS, dim].
struct Scratch {
    ids: Tensor,
    /// One-u32 step args holding the current position; rope/kv_store/attn read
    /// it so the position stays out of the pipeline constants.
    args: Tensor,
    hidden: Tensor,
    hidden2: Tensor,
    normed: Tensor,
    q: Tensor,
    k1: Tensor,
    v1: Tensor,
    /// QK-norm outputs; unused buffers when the config has no QK-norm.
    q2: Tensor,
    k2: Tensor,
    attn_out: Tensor,
    up1: Tensor,
    up2: Tensor,
    m1: Tensor,
    m2: Tensor,
    logits: Tensor,
}

fn alloc_scratch(cfg: &DenseConfig, backend: &Backend) -> Scratch {
    let z = |shape: &[u32], label: &str| backend.zero_tensor(shape, label);
    Scratch {
        ids: Tensor::new(
            backend.storage(ROWS as u64 * 4, "ids"),
            vec![ROWS],
            DType::U32,
        ),
        args: Tensor::new(backend.storage(4, "args"), vec![1], DType::U32),
        hidden: z(&[ROWS, cfg.hidden], "hidden"),
        hidden2: z(&[ROWS, cfg.hidden], "hidden2"),
        normed: z(&[ROWS, cfg.hidden], "normed"),
        q: z(&[ROWS, cfg.q_heads * cfg.head_dim], "q"),
        k1: z(&[ROWS, cfg.kv_heads * cfg.head_dim], "k1"),
        v1: z(&[ROWS, cfg.kv_heads * cfg.head_dim], "v1"),
        q2: z(&[ROWS, cfg.q_heads * cfg.head_dim], "q2"),
        k2: z(&[ROWS, cfg.kv_heads * cfg.head_dim], "k2"),
        attn_out: z(&[ROWS, cfg.q_heads * cfg.head_dim], "attn_out"),
        up1: z(&[ROWS, cfg.intermediate], "up1"),
        up2: z(&[ROWS, cfg.intermediate], "up2"),
        m1: z(&[ROWS, cfg.intermediate], "m1"),
        m2: z(&[ROWS, cfg.hidden], "m2"),
        logits: z(&[ROWS, cfg.vocab], "logits"),
    }
}

/// Dense GQA transformer with SwiGLU MLP and direct-weight RMSNorm.
pub struct DenseModel {
    cfg: DenseConfig,
    max_seq: u32,
    pos: u32,
    embed: Weight,
    /// Untied logits projection; None reuses the embedding table.
    head: Option<Weight>,
    norm: Tensor,
    layers: Vec<LayerW>,
    kv: Vec<KvCache>,
    s: Scratch,
    cos: Tensor,
    sin: Tensor,
}

impl DenseModel {
    pub fn load(
        source: &dyn Checkpoint,
        cfg: DenseConfig,
        plan: &Plan,
        max_seq: u32,
        backend: &Backend,
    ) -> Result<Self> {
        cfg.validate()?;
        let mut w = loader::load_weights(backend, source, plan)?;
        let embed = w.take("embed_tokens.weight")?;
        let head = if cfg.tied {
            None
        } else {
            Some(w.take("lm_head.weight")?)
        };
        let norm = w.take_tensor("norm.weight")?;
        let layers = (0..cfg.layers)
            .map(|l| take_layer(&mut w, &cfg, l))
            .collect::<Result<Vec<_>>>()?;
        let kv = (0..cfg.layers)
            .map(|i| {
                KvCache::new(
                    backend,
                    cfg.kv_heads,
                    max_seq,
                    cfg.head_dim,
                    &format!("l{i}"),
                )
            })
            .collect();
        let s = alloc_scratch(&cfg, backend);
        let (cos, sin) = ops::rope_tables(backend, max_seq, cfg.head_dim, cfg.rope_theta);
        Ok(Self {
            cfg,
            max_seq,
            pos: 0,
            embed,
            head,
            norm,
            layers,
            kv,
            s,
            cos,
            sin,
        })
    }

    fn head_weight(&self) -> &Weight {
        self.head.as_ref().unwrap_or(&self.embed)
    }
}

impl LanguageModel for DenseModel {
    fn forward(
        &mut self,
        backend: &mut Backend,
        tokens: &[u32],
        logit_rows: &[u32],
        hidden_rows: &[u32],
    ) -> Result<ChunkOut> {
        let m = tokens.len() as u32;
        if m == 0 || m > ROWS {
            return Err(Error::Model(format!("chunk size {m} outside [1, {ROWS}]")));
        }
        if self.pos + m > self.max_seq {
            return Err(Error::Model(format!(
                "context limit {} reached",
                self.max_seq
            )));
        }
        let mut ids = vec![0u32; ROWS as usize];
        ids[..tokens.len()].copy_from_slice(tokens);
        backend.write_u32(&self.s.ids.buf, &ids);
        backend.write_u32(&self.s.args.buf, &[self.pos]);

        let cfg = &self.cfg;
        let mut enc = backend.encoder();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward"),
                ..Default::default()
            });
            let s = &self.s;
            ops::embed(
                backend,
                &mut pass,
                &s.ids,
                &self.embed.t,
                Binding::Full(&s.hidden),
                cfg.hidden,
                cfg.embed_scale,
            )?;

            for (l, (lw, kv)) in self.layers.iter().zip(&self.kv).enumerate() {
                let (nq, nkv, hd) = (cfg.q_heads, cfg.kv_heads, cfg.head_dim);

                ops::norm(
                    backend,
                    &mut pass,
                    NormMode::Direct,
                    Binding::Full(&s.hidden),
                    &lw.attn_norm,
                    Binding::Full(&s.hidden),
                    Binding::Full(&s.normed),
                    ROWS,
                    cfg.hidden,
                    cfg.hidden,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.normed),
                    &lw.q,
                    Binding::Full(&s.q),
                    m,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.normed),
                    &lw.k,
                    Binding::Full(&s.k1),
                    m,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.normed),
                    &lw.v,
                    Binding::Full(&s.v1),
                    m,
                )?;

                if let (Some(qb), Some(kb), Some(vb)) = (&lw.q_bias, &lw.k_bias, &lw.v_bias) {
                    ops::bias(backend, &mut pass, Binding::Full(&s.q), qb, nq * hd)?;
                    ops::bias(backend, &mut pass, Binding::Full(&s.k1), kb, nkv * hd)?;
                    ops::bias(backend, &mut pass, Binding::Full(&s.v1), vb, nkv * hd)?;
                }

                let (q_src, k_src): (&Tensor, &Tensor) = match (&lw.q_norm, &lw.k_norm) {
                    (Some(qn), Some(kn)) => {
                        ops::norm(
                            backend,
                            &mut pass,
                            NormMode::Direct,
                            Binding::Full(&s.q),
                            qn,
                            Binding::Full(&s.q),
                            Binding::Full(&s.q2),
                            ROWS * nq,
                            hd,
                            hd,
                        )?;
                        ops::norm(
                            backend,
                            &mut pass,
                            NormMode::Direct,
                            Binding::Full(&s.k1),
                            kn,
                            Binding::Full(&s.k1),
                            Binding::Full(&s.k2),
                            ROWS * nkv,
                            hd,
                            hd,
                        )?;
                        (&s.q2, &s.k2)
                    }
                    _ => (&s.q, &s.k1),
                };

                ops::rope(
                    backend,
                    &mut pass,
                    &self.cos,
                    &self.sin,
                    Binding::Full(q_src),
                    nq,
                    hd,
                    hd,
                    m,
                    &self.s.args,
                )?;
                ops::rope(
                    backend,
                    &mut pass,
                    &self.cos,
                    &self.sin,
                    Binding::Full(k_src),
                    nkv,
                    hd,
                    hd,
                    m,
                    &self.s.args,
                )?;
                ops::kv_store(
                    backend,
                    &mut pass,
                    Binding::Full(k_src),
                    &kv.k,
                    nkv,
                    hd,
                    kv.max_seq,
                    &self.s.args,
                    m,
                )?;
                ops::kv_store(
                    backend,
                    &mut pass,
                    Binding::Full(&s.v1),
                    &kv.v,
                    nkv,
                    hd,
                    kv.max_seq,
                    &self.s.args,
                    m,
                )?;

                ops::attn(
                    backend,
                    &mut pass,
                    Binding::Full(q_src),
                    &kv.k,
                    &kv.v,
                    Binding::Full(&s.attn_out),
                    nq,
                    nkv,
                    hd,
                    kv.max_seq,
                    &self.s.args,
                    m,
                    cfg.window(l as u32),
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.attn_out),
                    &lw.o,
                    Binding::Full(&s.m2),
                    m,
                )?;

                // Sandwich families norm the attention output before the
                // residual; everyone else adds it directly.
                match &lw.post_attn_norm {
                    Some(pn) => {
                        ops::norm(
                            backend,
                            &mut pass,
                            NormMode::Direct,
                            Binding::Full(&s.m2),
                            pn,
                            Binding::Full(&s.m2),
                            Binding::Full(&s.normed),
                            ROWS,
                            cfg.hidden,
                            cfg.hidden,
                        )?;
                        ops::add(
                            backend,
                            &mut pass,
                            Binding::Full(&s.hidden),
                            Binding::Full(&s.normed),
                            Binding::Full(&s.hidden2),
                            ROWS * cfg.hidden,
                        )?;
                    }
                    None => {
                        ops::add(
                            backend,
                            &mut pass,
                            Binding::Full(&s.hidden),
                            Binding::Full(&s.m2),
                            Binding::Full(&s.hidden2),
                            ROWS * cfg.hidden,
                        )?;
                    }
                }

                ops::norm(
                    backend,
                    &mut pass,
                    NormMode::Direct,
                    Binding::Full(&s.hidden2),
                    &lw.mlp.norm,
                    Binding::Full(&s.hidden2),
                    Binding::Full(&s.normed),
                    ROWS,
                    cfg.hidden,
                    cfg.hidden,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.normed),
                    &lw.mlp.gate,
                    Binding::Full(&s.up1),
                    m,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.normed),
                    &lw.mlp.up,
                    Binding::Full(&s.up2),
                    m,
                )?;
                ops::swiglu(
                    backend,
                    &mut pass,
                    Binding::Full(&s.up1),
                    Binding::Full(&s.up2),
                    Binding::Full(&s.m1),
                    ROWS * cfg.intermediate,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.m1),
                    &lw.mlp.down,
                    Binding::Full(&s.m2),
                    m,
                )?;

                match &lw.post_ffn_norm {
                    Some(pn) => {
                        ops::norm(
                            backend,
                            &mut pass,
                            NormMode::Direct,
                            Binding::Full(&s.m2),
                            pn,
                            Binding::Full(&s.m2),
                            Binding::Full(&s.normed),
                            ROWS,
                            cfg.hidden,
                            cfg.hidden,
                        )?;
                        ops::add(
                            backend,
                            &mut pass,
                            Binding::Full(&s.hidden2),
                            Binding::Full(&s.normed),
                            Binding::Full(&s.hidden),
                            ROWS * cfg.hidden,
                        )?;
                    }
                    None => {
                        ops::add(
                            backend,
                            &mut pass,
                            Binding::Full(&s.hidden2),
                            Binding::Full(&s.m2),
                            Binding::Full(&s.hidden),
                            ROWS * cfg.hidden,
                        )?;
                    }
                }
            }

            ops::norm(
                backend,
                &mut pass,
                NormMode::Direct,
                Binding::Full(&s.hidden),
                &self.norm,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                ROWS,
                cfg.hidden,
                cfg.hidden,
            )?;
            ops::gemm(
                backend,
                &mut pass,
                Binding::Full(&s.normed),
                self.head_weight(),
                Binding::Full(&s.logits),
                m,
            )?;
        }
        backend.submit(enc);

        let mut out = ChunkOut {
            logits: Vec::new(),
            hidden: Vec::new(),
        };
        for &r in logit_rows {
            assert!(r < m, "logit row {r} outside chunk");
            out.logits.push(backend.read_f32(
                &self.s.logits.buf,
                r as u64 * cfg.vocab as u64 * 4,
                cfg.vocab as usize,
            )?);
        }
        for &r in hidden_rows {
            assert!(r < m, "hidden row {r} outside chunk");
            out.hidden.push(backend.read_f32(
                &self.s.hidden.buf,
                r as u64 * cfg.hidden as u64 * 4,
                cfg.hidden as usize,
            )?);
        }
        self.pos += m;
        backend.flush_profile()?;
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
