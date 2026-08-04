//! Dense GQA transformer shared by the LLaMA family (LLaMA / Qwen2 / Qwen3 /
//! Mistral), Gemma 3, Phi-4-mini, Phi-MoE and Gemma 4. The families differ in
//! norm placement, attention window, embedding scale, QK-norm, activation,
//! per-layer head dims / rope, logit softcapping and MoE FFNs — expressed as
//! configuration over one forward graph, not as separate implementations.

use flint_backend::{Backend, Binding, Pass};
use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use flint_model::cache::KvCache;
use flint_model::config::{check_gemm_dims, check_head_dim, f64_field, u32_field, u32_list};
use flint_model::loader::{self, MlpBlock, Plan, Role, WeightSet, take_moe};
use flint_model::ops::{self, Act, MlpTiles, MoeTiles, NormMode, ROWS, RopeScaling};
use flint_model::routing::{RouteKind, Routing};
use flint_model::{ChunkOut, LanguageModel};
use flint_tensor::{Tensor, Weight};
use serde_json::Value;

use crate::gguf_config::gguf_key;

/// GPU storage role for every dense-family weight: norms, biases and scalars
/// are f32, embeddings are packed bf16, projections are group-quantized i8.
pub fn dense_role(key: &str) -> Role {
    if key.contains("norm")
        || key.ends_with(".bias")
        || key.ends_with("layer_scalar")
    {
        Role::F32
    } else if key == "embed_tokens.weight"
        || key == "embed_tokens_per_layer.weight"
        || key.contains("router")
    {
        // Routers stay bf16: routing decisions tolerate no quantization error.
        Role::Bf16
    } else {
        Role::I8
    }
}

/// Loading policy for a dense-family checkpoint: native names map through
/// `key` (HF or GGUF), every weight takes the shared [`dense_role`].
pub fn dense_plan(gguf: bool, key: fn(&str) -> Option<String>) -> Plan {
    Plan {
        key: if gguf { gguf_key } else { key },
        role: dense_role,
    }
}

/// One RoPE variant; layers reference a set by index (`layer_rope`).
#[derive(Clone, Debug)]
pub struct RopeSpec {
    /// Rotated dims (partial rotary leaves the rest untouched).
    pub dim: u32,
    /// Inverse-frequency denominator (the head dim for proportional rope).
    pub freq_dim: u32,
    pub theta: f64,
    /// LongRoPE per-dimension factors; None for plain rope.
    pub scaling: Option<RopeScaling>,
}

/// MoE FFN configuration.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub experts: u32,
    pub top_k: u32,
    /// Router-logit scale applied before softmax.
    pub scale: f32,
    /// Weight of the shared expert (0 disables it).
    pub shared_scale: f32,
    /// Routing scheme.
    pub kind: RouteKind,
}

/// Per-Layer Embeddings (PLE, Gemma 4): an auxiliary per-token signal of
/// `dim` floats per layer, injected into every layer's output.
#[derive(Clone, Copy, Debug)]
pub struct PerLayerConfig {
    pub dim: u32,
}

/// Validated dense GQA config covering every supported family.
#[derive(Clone, Debug)]
pub struct DenseConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// Per-layer attention head dims (Gemma 4: global layers widen).
    pub head_dims: Vec<u32>,
    pub vocab: u32,
    pub eos: Vec<u32>,
    pub tied: bool,
    /// Input embedding scale (Gemma: sqrt(hidden); everyone else 1).
    pub embed_scale: f32,
    /// Q/K/V projection biases (Qwen2 small models, Phi-MoE).
    pub qkv_bias: bool,
    /// Per-head RMSNorm on Q and K before RoPE (Qwen3; always on for Gemma).
    pub qk_norm: bool,
    /// Scale-less RMSNorm on V before the cache (Gemma 4).
    pub v_norm: bool,
    /// Sandwich norms on attention and MLP outputs before the residual (Gemma 3).
    pub sandwich: bool,
    /// Attention window per layer; 0 attends to the full causal prefix.
    pub windows: Vec<u32>,
    /// Norm epsilon (RMSNorm families vary; LayerNorm reads it too).
    pub norm_eps: f32,
    /// Mean-centered norm with bias (Phi-MoE's LayerNorm).
    pub layernorm: bool,
    /// Logits projection bias (Phi-MoE's lm_head_bias).
    pub lm_bias: bool,
    /// MLP activation (Phi-4-mini / Gemma 4 use GELU).
    pub act: Act,
    /// Double-wide MLP per layer (Gemma 4's KV-shared layers).
    pub double_wide: Vec<bool>,
    /// RoPE sets plus the per-layer index into them.
    pub rope: Vec<RopeSpec>,
    pub layer_rope: Vec<u32>,
    /// Layers from `layers - kv_shared` on reuse the last non-shared KV of
    /// their layer type instead of owning projections (Gemma 4).
    pub kv_shared: u32,
    /// Final-logit softcap (Gemma 4); None disables.
    pub softcap: Option<f32>,
    /// MoE FFN config; None means a dense SwiGLU MLP.
    pub moe: Option<MoeConfig>,
    /// Per-Layer Embeddings (Gemma 4); None disables.
    pub per_layer: Option<PerLayerConfig>,
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
        let layers = u32_field(v, "num_hidden_layers")?;
        let rope_theta = f64_field(v, "rope_theta")?;
        Ok(Self {
            hidden,
            intermediate: u32_field(v, "intermediate_size")?,
            layers,
            q_heads,
            kv_heads: u32_field(v, "num_key_value_heads")?,
            head_dims: vec![head_dim; layers as usize],
            vocab: u32_field(v, "vocab_size")?,
            eos: u32_list(v, "eos_token_id")?,
            tied: v
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(tied_default),
            embed_scale: 1.0,
            qkv_bias: false,
            qk_norm: false,
            v_norm: false,
            sandwich: false,
            windows: vec![0; layers as usize],
            norm_eps: 1e-6,
            layernorm: false,
            lm_bias: false,
            act: Act::Silu,
            double_wide: vec![false; layers as usize],
            rope: vec![RopeSpec {
                dim: head_dim,
                freq_dim: head_dim,
                theta: rope_theta,
                scaling: None,
            }],
            layer_rope: vec![0; layers as usize],
            kv_shared: 0,
            softcap: None,
            moe: None,
            per_layer: None,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let t = &self;
        if !t.q_heads.is_multiple_of(t.kv_heads) {
            return Err(Error::Config("q heads not divisible by kv heads".into()));
        }
        if t.q_heads / t.kv_heads > flint_model::ops::MAX_GQA {
            return Err(Error::Config(format!(
                "GQA ratio {} exceeds the attention shader's {} head slots",
                t.q_heads / t.kv_heads,
                flint_model::ops::MAX_GQA
            )));
        }
        if t.head_dims.len() != t.layers as usize {
            return Err(Error::Config("head_dims length mismatch".into()));
        }
        if t.windows.len() != t.layers as usize
            || t.double_wide.len() != t.layers as usize
            || t.layer_rope.len() != t.layers as usize
        {
            return Err(Error::Config("per-layer config length mismatch".into()));
        }
        for &hd in &t.head_dims {
            check_head_dim(hd)?;
        }
        if let Some(moe) = t.moe {
            if !moe.experts.is_multiple_of(16) {
                return Err(Error::Config(format!(
                    "num_experts {} must be a multiple of 16 (gemm tiles)",
                    moe.experts
                )));
            }
            if moe.top_k == 0 || moe.top_k > moe.experts {
                return Err(Error::Config("top_k outside [1, experts]".into()));
            }
        }
        for r in &t.rope {
            if !r.dim.is_multiple_of(2) || r.freq_dim == 0 {
                return Err(Error::Config(format!(
                    "invalid rope spec (dim {}, freq_dim {})",
                    r.dim, r.freq_dim
                )));
            }
            if let Some(s) = &r.scaling
                && s.short.len() != r.dim as usize / 2
            {
                return Err(Error::Config(
                    "LongRoPE factors must match rotary dim / 2".into(),
                ));
            }
        }
        if t.kv_shared >= t.layers && t.kv_shared != 0 {
            return Err(Error::Config("kv_shared must be < num_hidden_layers".into()));
        }
        if let Some(p) = t.per_layer {
            if p.dim == 0 || (t.layers * p.dim).is_multiple_of(16) == false {
                return Err(Error::Config(
                    "per-layer dim must be non-zero and layers*dim a gemm multiple of 16".into(),
                ));
            }
        }
        check_gemm_dims(&[
            (t.vocab, t.hidden),
            (t.q_heads * *t.head_dims.iter().max().unwrap(), t.hidden),
            (t.kv_heads * *t.head_dims.iter().max().unwrap(), t.hidden),
            (t.hidden, t.q_heads * *t.head_dims.iter().max().unwrap()),
            (t.max_mlp_width(), t.hidden),
            (t.hidden, t.max_mlp_width()),
        ])
    }

    /// MLP width of layer `l` (double-wide layers double the intermediate).
    pub fn mlp_width(&self, l: u32) -> u32 {
        if self.double_wide[l as usize] {
            2 * self.intermediate
        } else {
            self.intermediate
        }
    }

    /// Largest MLP width across layers (scratch sizing).
    fn max_mlp_width(&self) -> u32 {
        (0..self.layers).map(|l| self.mlp_width(l)).max().unwrap()
    }

    /// Layers at and beyond this index share the last same-type KV.
    pub fn first_shared(&self) -> u32 {
        self.layers - self.kv_shared
    }

    /// Attention window of layer `l`.
    pub fn window(&self, l: u32) -> u32 {
        self.windows[l as usize]
    }

    /// Head dim of layer `l`.
    pub fn head_dim(&self, l: u32) -> u32 {
        self.head_dims[l as usize]
    }

    /// Whether layer `l` owns its KV projections (Gemma 4 sharing).
    pub fn has_kv(&self, l: u32) -> bool {
        l < self.first_shared()
    }

    /// Whether the layer carries a Per-Layer Embedding block.
    pub fn has_ple(&self) -> bool {
        self.per_layer.is_some()
    }
}

// ---------------------------------------------------------------- weights

/// The weights of one transformer layer. K/V weights are absent on Gemma 4's
/// KV-shared layers; the per-layer scratch tiles are sized to the layer's
/// head dim so the attention bindings stay contiguous.
struct LayerW {
    attn_norm: Tensor,
    attn_norm_bias: Option<Tensor>,
    q: Weight,
    k: Option<Weight>,
    v: Option<Weight>,
    o: Weight,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    post_attn_norm: Option<Tensor>,
    mlp: MlpBlock,
    post_ffn_norm: Option<Tensor>,
    // Per-Layer Embedding block (Gemma 4).
    per_layer_gate: Option<Weight>,
    per_layer_proj: Option<Weight>,
    per_layer_norm: Option<Tensor>,
    out_scale: Option<Tensor>,
    // Per-layer activation tiles, sized to the layer's head dim.
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    q2_t: Tensor,
    k2_t: Tensor,
    /// V-norm output (Gemma 4); the cache stores this instead of v_t.
    v2_t: Tensor,
    attn_out: Tensor,
}

fn take_optional(w: &mut WeightSet, on: bool, key: &str) -> Result<Option<Tensor>> {
    if on {
        Ok(Some(w.take_tensor(key)?))
    } else {
        Ok(None)
    }
}

fn take_layer(w: &mut WeightSet, cfg: &DenseConfig, l: u32, backend: &Backend) -> Result<LayerW> {
    let k = |n: &str| format!("layers.{l}.{n}");
    let hd = cfg.head_dim(l);
    let qw = cfg.q_heads * hd;
    let kvw = cfg.kv_heads * hd;
    let has_kv = cfg.has_kv(l);
    let norm_bias = |w: &mut WeightSet, key: &str| take_optional(w, cfg.layernorm, key);
    let (k_w, v_w) = if has_kv {
        (
            Some(w.take(&k("self_attn.k_proj.weight"))?),
            Some(w.take(&k("self_attn.v_proj.weight"))?),
        )
    } else {
        (None, None)
    };
    let (k_b, v_b) = if has_kv && cfg.qkv_bias {
        (
            Some(w.take_tensor(&k("self_attn.k_proj.bias"))?),
            Some(w.take_tensor(&k("self_attn.v_proj.bias"))?),
        )
    } else {
        (None, None)
    };
    let mlp = match cfg.moe {
        Some(moe) => MlpBlock::Moe(take_moe(
            w,
            &format!("layers.{l}"),
            moe.experts,
            moe.top_k,
            moe.scale,
            moe.shared_scale,
            cfg.layernorm,
        )?),
        None => MlpBlock::Dense(loader::take_mlp(w, &format!("layers.{l}"))?),
    };
    let ple = cfg.has_ple();
    Ok(LayerW {
        attn_norm: w.take_tensor(&k("input_layernorm.weight"))?,
        attn_norm_bias: norm_bias(w, &k("input_layernorm.bias"))?,
        q: w.take(&k("self_attn.q_proj.weight"))?,
        k: k_w,
        v: v_w,
        o: w.take(&k("self_attn.o_proj.weight"))?,
        q_bias: take_optional(w, cfg.qkv_bias, &k("self_attn.q_proj.bias"))?,
        k_bias: k_b,
        v_bias: v_b,
        q_norm: take_optional(w, cfg.qk_norm, &k("self_attn.q_norm.weight"))?,
        k_norm: take_optional(w, cfg.qk_norm && has_kv, &k("self_attn.k_norm.weight"))?,
        post_attn_norm: take_optional(w, cfg.sandwich, &k("post_attention_norm.weight"))?,
        mlp,
        post_ffn_norm: take_optional(w, cfg.sandwich, &k("post_ffw_norm.weight"))?,
        per_layer_gate: if ple {
            Some(w.take(&k("per_layer_input_gate.weight"))?)
        } else {
            None
        },
        per_layer_proj: if ple {
            Some(w.take(&k("per_layer_projection.weight"))?)
        } else {
            None
        },
        per_layer_norm: take_optional(w, ple, &k("post_per_layer_input_norm.weight"))?,
        out_scale: take_optional(w, ple, &k("layer_scalar"))?,
        q_t: backend.zero_tensor(&[ROWS, qw], &format!("l{l}.q")),
        k_t: backend.zero_tensor(&[ROWS, kvw], &format!("l{l}.k")),
        v_t: backend.zero_tensor(&[ROWS, kvw], &format!("l{l}.v")),
        q2_t: backend.zero_tensor(&[ROWS, qw], &format!("l{l}.q2")),
        k2_t: backend.zero_tensor(&[ROWS, kvw], &format!("l{l}.k2")),
        v2_t: backend.zero_tensor(&[ROWS, kvw], &format!("l{l}.v2")),
        attn_out: backend.zero_tensor(&[ROWS, qw], &format!("l{l}.attn_out")),
    })
}

/// Per-forward scratch shared across layers.
struct Scratch {
    ids: Tensor,
    /// One-u32 step args holding the current position; rope/kv_store/attn read
    /// it so the position stays out of the pipeline constants.
    args: Tensor,
    hidden: Tensor,
    hidden2: Tensor,
    normed: Tensor,
    /// Split-K attention partials: [m, kv_heads, ATTN_SEGS, MAX_GQA, hd+2] f32,
    /// sized to the largest layer head dim.
    attn_scratch: Tensor,
    /// Scratch stride for the attention partials: the largest head dim + 2.
    attn_stride: u32,
    mlp: MlpTiles,
    /// MoE scratch; present only with a MoE config.
    moe: Option<MoeTiles>,
    logits: Tensor,
    /// Per-Layer Embedding tiles [ROWS, layers * dim]; present with PLE.
    ple_tok: Option<Tensor>,
    ple_ctx: Option<Tensor>,
    ple_out: Option<Tensor>,
    ple_gate: Option<Tensor>,
    /// Ones tile for the PLE gate activation [ROWS, dim].
    ple_ones: Option<Tensor>,
}

fn alloc_scratch(cfg: &DenseConfig, backend: &Backend) -> Scratch {
    let max_hd = *cfg.head_dims.iter().max().unwrap();
    let mlp_w = cfg.max_mlp_width();
    let moe = cfg.moe.map(|m| {
        ops::MoeTiles::new(
            &ops::MoeTilesCfg {
                experts: m.experts,
                rows: ROWS,
                top_k: m.top_k,
                hidden: cfg.hidden,
                intermediate: cfg.intermediate,
            },
            backend,
        )
    });
    let ple_dim = cfg.per_layer.map(|p| p.dim * cfg.layers);
    let ple = |shape: &[u32], label: &str| {
        ple_dim.map(|_| backend.zero_tensor(shape, label))
    };
    Scratch {
        ids: ops::token_ids(backend),
        args: ops::step_args(backend),
        hidden: backend.zero_tensor(&[ROWS, cfg.hidden], "hidden"),
        hidden2: backend.zero_tensor(&[ROWS, cfg.hidden], "hidden2"),
        normed: backend.zero_tensor(&[ROWS, cfg.hidden], "normed"),
        attn_scratch: backend.zero_tensor(
            &[
                ROWS,
                cfg.kv_heads,
                ops::ATTN_SEGS,
                ops::MAX_GQA,
                max_hd + 2,
            ],
            "attn_scratch",
        ),
        attn_stride: max_hd + 2,
        mlp: MlpTiles {
            gate_out: backend.zero_tensor(&[ROWS, mlp_w], "up1"),
            up_out: backend.zero_tensor(&[ROWS, mlp_w], "up2"),
            act: backend.zero_tensor(&[ROWS, mlp_w], "m1"),
            y: backend.zero_tensor(&[ROWS, cfg.hidden], "m2"),
        },
        moe,
        logits: backend.zero_tensor(&[ROWS, cfg.vocab], "logits"),
        ple_tok: ple(&[ROWS, ple_dim.unwrap_or(0)], "ple_tok"),
        ple_ctx: ple(&[ROWS, ple_dim.unwrap_or(0)], "ple_ctx"),
        ple_out: ple(&[ROWS, ple_dim.unwrap_or(0)], "ple_out"),
        ple_gate: ple(&[ROWS, cfg.per_layer.map_or(0, |p| p.dim)], "ple_gate"),
        ple_ones: cfg.per_layer.map(|p| {
            backend.tensor_f32(
                &vec![1.0; (ROWS * p.dim) as usize],
                vec![ROWS, p.dim],
                "ple_ones",
            )
        }),
    }
}

/// Dense GQA transformer with a gated MLP (dense or MoE) and direct-weight
/// normalization (RMSNorm or LayerNorm).
pub struct DenseModel {
    cfg: DenseConfig,
    max_seq: u32,
    pos: u32,
    embed: Weight,
    /// Untied logits projection; None reuses the embedding table.
    head: Option<Weight>,
    /// Logits projection bias (Phi-MoE's lm_head_bias).
    lm_bias: Option<Tensor>,
    norm: Tensor,
    norm_bias: Option<Tensor>,
    layers: Vec<LayerW>,
    /// KV cache per non-shared layer plus the per-layer source index (Gemma 4
    /// shared layers read the last same-type cache).
    kv: Vec<KvCache>,
    kv_src: Vec<usize>,
    /// Ones tensor for weightless (scale-less) norms.
    ones: Tensor,
    s: Scratch,
    /// RoPE tables, one pair per rope set.
    cos: Vec<Tensor>,
    sin: Vec<Tensor>,
    /// Per-Layer Embedding weights (Gemma 4).
    ple_emb: Option<Weight>,
    ple_proj: Option<Weight>,
    ple_norm: Option<Tensor>,
}

impl DenseModel {
    pub fn load(
        source: &dyn Checkpoint,
        cfg: DenseConfig,
        plan: &Plan,
        max_seq: u32,
        backend: &Backend,
    ) -> Result<Self> {
        Self::load_extra(source, cfg, plan, Vec::new(), max_seq, backend)
    }

    /// Loads with pre-uploaded extra weights (MoE expert sets) inserted into
    /// the weight set before the per-layer take.
    pub fn load_extra(
        source: &dyn Checkpoint,
        cfg: DenseConfig,
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
        let (ple_emb, ple_proj, ple_norm) = if cfg.has_ple() {
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

        // KV caches: one per non-shared layer; shared layers map to the last
        // non-shared layer of their window class (Gemma 4).
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
                    &format!("l{l}"),
                ));
                kv_src[l] = idx;
                last_by_class[(cfg.window(l as u32) > 0) as usize] = Some(idx);
            }
        }

        let ones = backend.tensor_f32(
            &vec![1.0; *cfg.head_dims.iter().max().unwrap() as usize],
            vec![*cfg.head_dims.iter().max().unwrap()],
            "ones",
        );
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
            ple_emb,
            ple_proj,
            ple_norm,
        })
    }

    fn head_weight(&self) -> &Weight {
        self.head.as_ref().unwrap_or(&self.embed)
    }

    /// Norm mode for the family's main norms: direct weights (RMSNorm
    /// families) or mean-centered with bias (LayerNorm families).
    fn norm_mode(&self) -> NormMode {
        if self.cfg.layernorm {
            NormMode::Layer
        } else {
            NormMode::Direct
        }
    }

    /// Bias binding for a norm (the norm's own bias, or the unused ones tile).
    fn norm_bias<'a>(&'a self, b: Option<&'a Tensor>) -> Binding<'a> {
        b.map(Binding::Full).unwrap_or(Binding::Full(&self.ones))
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
            let mut pass = Pass::begin(&mut enc, "forward");
            let s = &self.s;
            ops::embed(
                backend,
                &mut pass,
                &s.ids,
                &self.embed,
                &self.ones,
                Binding::Full(&s.hidden),
                cfg.hidden,
                cfg.embed_scale,
            )?;

            // Per-Layer Embeddings: token identity + context projection,
            // combined and normalized once per forward (RMSNorm absorbs the
            // combination's constant scales, so only the relative 1/sqrt(hidden)
            // vs sqrt(dim) weighting matters and folds into the embed scale).
            if let (Some(pe), Some(pp), Some(pn)) = (&self.ple_emb, &self.ple_proj, &self.ple_norm) {
                let (pt, pc, po) = (
                    s.ple_tok.as_ref().unwrap(),
                    s.ple_ctx.as_ref().unwrap(),
                    s.ple_out.as_ref().unwrap(),
                );
                let pd = cfg.per_layer.expect("PLE config").dim * cfg.layers;
                let scale = ((cfg.per_layer.expect("PLE config").dim * cfg.hidden) as f32).sqrt();
                ops::embed(
                    backend,
                    &mut pass,
                    &s.ids,
                    pe,
                    &self.ones,
                    Binding::Full(pt),
                    pd,
                    scale,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.hidden),
                    pp,
                    Binding::Full(pc),
                    m,
                )?;
                ops::add(
                    backend,
                    &mut pass,
                    Binding::Full(pc),
                    Binding::Full(pt),
                    Binding::Full(po),
                    ROWS * pd,
                )?;
                ops::norm(
                    backend,
                    &mut pass,
                    NormMode::Direct,
                    Binding::Full(po),
                    pn,
                    Binding::Full(po),
                    Binding::Full(pc),
                    ROWS,
                    pd,
                    pd,
                    cfg.norm_eps,
                )?;
            }

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
                    ROWS,
                    cfg.hidden,
                    cfg.hidden,
                    cfg.norm_eps,
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&s.normed),
                    &lw.q,
                    Binding::Full(&lw.q_t),
                    m,
                )?;
                if let (Some(kw), Some(vw)) = (&lw.k, &lw.v) {
                    ops::gemm(
                        backend,
                        &mut pass,
                        Binding::Full(&s.normed),
                        kw,
                        Binding::Full(&lw.k_t),
                        m,
                    )?;
                    ops::gemm(
                        backend,
                        &mut pass,
                        Binding::Full(&s.normed),
                        vw,
                        Binding::Full(&lw.v_t),
                        m,
                    )?;
                }

                if let (Some(qb), Some(kb), Some(vb)) = (&lw.q_bias, &lw.k_bias, &lw.v_bias) {
                    ops::bias(backend, &mut pass, Binding::Full(&lw.q_t), qb, nq * hd)?;
                    ops::bias(backend, &mut pass, Binding::Full(&lw.k_t), kb, nkv * hd)?;
                    ops::bias(backend, &mut pass, Binding::Full(&lw.v_t), vb, nkv * hd)?;
                }

                // QK-norm (per-head RMSNorm before RoPE); K/V projections
                // exist only on layers that own their cache.
                let (q_src, k_src): (&Tensor, Option<&Tensor>) = match (&lw.q_norm, &lw.k_norm) {
                    (Some(qn), Some(kn)) => {
                        ops::norm(
                            backend,
                            &mut pass,
                            NormMode::Direct,
                            Binding::Full(&lw.q_t),
                            qn,
                            Binding::Full(&lw.q_t),
                            Binding::Full(&lw.q2_t),
                            ROWS * nq,
                            hd,
                            hd,
                            cfg.norm_eps,
                        )?;
                        ops::norm(
                            backend,
                            &mut pass,
                            NormMode::Direct,
                            Binding::Full(&lw.k_t),
                            kn,
                            Binding::Full(&lw.k_t),
                            Binding::Full(&lw.k2_t),
                            ROWS * nkv,
                            hd,
                            hd,
                            cfg.norm_eps,
                        )?;
                        (&lw.q2_t, Some(&lw.k2_t))
                    }
                    _ => (&lw.q_t, lw.k.as_ref().map(|_| &lw.k_t)),
                };

                // V norm before the cache (Gemma 4's weightless v_norm).
                if cfg.v_norm && lw.k.is_some() {
                    ops::norm(
                        backend,
                        &mut pass,
                        NormMode::Direct,
                        Binding::Full(&lw.v_t),
                        &self.ones,
                        Binding::Full(&lw.v_t),
                        Binding::Full(&lw.v2_t),
                        ROWS * nkv,
                        hd,
                        hd,
                        cfg.norm_eps,
                    )?;
                }

                let ri = cfg.layer_rope[l];
                let (cos, sin) = (&self.cos[ri as usize], &self.sin[ri as usize]);
                let rot = cfg.rope[ri as usize].dim;
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

                let kvs = self.kv_src[l];
                let kv = &self.kv[kvs];
                if let Some(k_src) = k_src {
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
                        Binding::Full(if cfg.v_norm { &lw.v2_t } else { &lw.v_t }),
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
                )?;
                ops::gemm(
                    backend,
                    &mut pass,
                    Binding::Full(&lw.attn_out),
                    &lw.o,
                    Binding::Full(&s.mlp.y),
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
                            Binding::Full(&s.mlp.y),
                            pn,
                            Binding::Full(&s.mlp.y),
                            Binding::Full(&s.normed),
                            ROWS,
                            cfg.hidden,
                            cfg.hidden,
                            cfg.norm_eps,
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
                            Binding::Full(&s.mlp.y),
                            Binding::Full(&s.hidden2),
                            ROWS * cfg.hidden,
                        )?;
                    }
                }

                ops::norm(
                    backend,
                    &mut pass,
                    self.norm_mode(),
                    Binding::Full(&s.hidden2),
                    lw.mlp.norm(),
                    self.norm_bias(lw.mlp.norm_bias()),
                    Binding::Full(&s.normed),
                    ROWS,
                    cfg.hidden,
                    cfg.hidden,
                    cfg.norm_eps,
                )?;

                match &lw.mlp {
                    MlpBlock::Dense(mlp) => {
                        ops::swiglu_mlp(
                            backend,
                            &mut pass,
                            Binding::Full(&s.normed),
                            mlp,
                            &s.mlp,
                            m,
                            cfg.mlp_width(l as u32),
                            cfg.act,
                        )?;
                        let y = Binding::Full(&s.mlp.y);
                        let src = Binding::Full(&s.hidden2);
                        let out = Binding::Full(&s.hidden);
                        match &lw.post_ffn_norm {
                            Some(pn) => {
                                ops::norm(
                                    backend,
                                    &mut pass,
                                    NormMode::Direct,
                                    y,
                                    pn,
                                    y,
                                    Binding::Full(&s.normed),
                                    ROWS,
                                    cfg.hidden,
                                    cfg.hidden,
                                    cfg.norm_eps,
                                )?;
                                ops::add(
                                    backend,
                                    &mut pass,
                                    src,
                                    Binding::Full(&s.normed),
                                    out,
                                    ROWS * cfg.hidden,
                                )?;
                            }
                            None => {
                                ops::add(backend, &mut pass, src, y, out, ROWS * cfg.hidden)?;
                            }
                        }
                    }
                    MlpBlock::Moe(moe) => {
                        let moe_cfg = cfg.moe.expect("MoE block without MoE config");
                        let mt = s.moe.as_ref().expect("MoE block without MoE scratch");
                        // Router logits close this encoder: routing runs on
                        // the CPU (the logits tile is tiny), then the expert
                        // block resumes on a fresh encoder.
                        ops::gemm(
                            backend,
                            &mut pass,
                            Binding::Full(&s.normed),
                            &moe.router,
                            Binding::Full(&mt.logits),
                            m,
                        )?;
                        drop(pass);
                        backend.submit(enc);
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
                        enc = backend.encoder();
                        pass = Pass::begin(&mut enc, "forward");
                        ops::zero_rows(
                            backend,
                            &mut pass,
                            Binding::Full(&mt.acc),
                            m * cfg.hidden,
                        )?;
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
                        let out = Binding::Full(&s.hidden);
                        match &lw.post_ffn_norm {
                            Some(pn) => {
                                ops::norm(
                                    backend,
                                    &mut pass,
                                    NormMode::Direct,
                                    Binding::Full(&mt.acc),
                                    pn,
                                    Binding::Full(&mt.acc),
                                    Binding::Full(&s.normed),
                                    ROWS,
                                    cfg.hidden,
                                    cfg.hidden,
                                    cfg.norm_eps,
                                )?;
                                ops::add(
                                    backend,
                                    &mut pass,
                                    Binding::Full(&s.hidden2),
                                    Binding::Full(&s.normed),
                                    out,
                                    ROWS * cfg.hidden,
                                )?;
                            }
                            None => {
                                ops::add(
                                    backend,
                                    &mut pass,
                                    Binding::Full(&s.hidden2),
                                    Binding::Full(&mt.acc),
                                    out,
                                    ROWS * cfg.hidden,
                                )?;
                            }
                        }
                    }
                }

                // Per-Layer Embedding block: inject the layer's PLE slice
                // into the layer output, then scale the layer (Gemma 4).
                if let (Some(gate), Some(proj), Some(pn), Some(os)) = (
                    &lw.per_layer_gate,
                    &lw.per_layer_proj,
                    &lw.per_layer_norm,
                    &lw.out_scale,
                ) {
                    let (po, pc, pg, pon) = (
                        s.ple_out.as_ref().unwrap(),
                        s.ple_ctx.as_ref().unwrap(),
                        s.ple_gate.as_ref().unwrap(),
                        s.ple_ones.as_ref().unwrap(),
                    );
                    let pd = cfg.per_layer.expect("PLE config").dim;
                    ops::gemm(
                        backend,
                        &mut pass,
                        Binding::Full(&s.hidden),
                        gate,
                        Binding::Full(pg),
                        m,
                    )?;
                    ops::swiglu(
                        backend,
                        &mut pass,
                        Binding::Full(pg),
                        Binding::Full(pon),
                        Binding::Full(pc),
                        ROWS * pd,
                        cfg.act,
                    )?;
                    ops::mul(
                        backend,
                        &mut pass,
                        Binding::Full(pc),
                        Binding::Slice(po, l as u64 * pd as u64 * 4, m as u64 * pd as u64 * 4),
                        Binding::Full(pg),
                        ROWS * pd,
                        pd,
                    )?;
                    ops::gemm(
                        backend,
                        &mut pass,
                        Binding::Full(pg),
                        proj,
                        Binding::Full(&s.mlp.y),
                        m,
                    )?;
                    ops::norm(
                        backend,
                        &mut pass,
                        NormMode::Direct,
                        Binding::Full(&s.mlp.y),
                        pn,
                        Binding::Full(&s.mlp.y),
                        Binding::Full(&s.normed),
                        ROWS,
                        cfg.hidden,
                        cfg.hidden,
                        cfg.norm_eps,
                    )?;
                    ops::add(
                        backend,
                        &mut pass,
                        Binding::Full(&s.hidden),
                        Binding::Full(&s.normed),
                        Binding::Full(&s.hidden2),
                        ROWS * cfg.hidden,
                    )?;
                    ops::mul(
                        backend,
                        &mut pass,
                        Binding::Full(&s.hidden2),
                        Binding::Full(os),
                        Binding::Full(&s.hidden),
                        ROWS * cfg.hidden,
                        1,
                    )?;
                }
            }

            ops::norm(
                backend,
                &mut pass,
                self.norm_mode(),
                Binding::Full(&s.hidden),
                &self.norm,
                self.norm_bias(self.norm_bias.as_ref()),
                Binding::Full(&s.normed),
                ROWS,
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
                ops::bias(backend, &mut pass, Binding::Full(&s.logits), lb, cfg.vocab)?;
            }
            if let Some(cap) = cfg.softcap {
                ops::softcap(
                    backend,
                    &mut pass,
                    Binding::Full(&s.logits),
                    ROWS * cfg.vocab,
                    cap,
                )?;
            }
        }
        backend.submit(enc);

        let out = ChunkOut {
            logits: ops::read_rows(backend, &self.s.logits, logit_rows, m, cfg.vocab)?,
            hidden: ops::read_rows(backend, &self.s.hidden, hidden_rows, m, cfg.hidden)?,
        };
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
