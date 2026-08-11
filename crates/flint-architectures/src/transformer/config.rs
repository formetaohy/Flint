use flint_error::{Error, Result};
use flint_model::config::{check_gemm_dims, check_head_dim, f64_field, u32_field, u32_list};
use flint_model::ops::{Act, RopeScaling};
use flint_model::routing::RouteKind;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct RopeSpec {
    pub dim: u32,

    pub freq_dim: u32,
    pub theta: f64,

    pub partial: Option<u32>,
    pub scaling: Option<RopeScaling>,
}

#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub experts: u32,
    pub top_k: u32,
    pub shared_scale: f32,
    pub kind: RouteKind,
}

#[derive(Clone, Copy, Debug)]
pub struct PerLayerConfig {
    pub dim: u32,
}

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,

    pub head_dims: Vec<u32>,
    pub vocab: u32,
    pub eos: Vec<u32>,
    pub tied: bool,

    pub embed_scale: f32,

    pub qkv_bias: bool,

    pub qk_norm: bool,

    pub v_norm: bool,

    pub sandwich: bool,

    pub hf_names: bool,

    pub windows: Vec<u32>,

    pub norm_eps: f32,

    pub layernorm: bool,

    pub lm_bias: bool,

    pub act: Act,

    pub double_wide: Vec<bool>,

    pub rope: Vec<RopeSpec>,
    pub layer_rope: Vec<u32>,

    pub kv_shared: u32,

    pub attn_scale: Option<f32>,

    pub softcap: Option<f32>,

    pub moe: Option<MoeConfig>,

    pub per_layer: Option<PerLayerConfig>,
}

impl TransformerConfig {
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
        let rope_freqs = v
            .get("rope_freqs")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_f64)
                    .map(|f| f as f32)
                    .collect::<Vec<_>>()
            })
            .filter(|f| !f.is_empty());
        let scaling = rope_freqs.map(|f| RopeScaling {
            short: f.clone(),
            long: f,
            original_max: 0,
        });
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
            hf_names: false,
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
                partial: None,
                scaling: scaling.clone(),
            }],
            layer_rope: vec![0; layers as usize],
            kv_shared: 0,
            attn_scale: None,
            softcap: None,
            moe: None,
            per_layer: None,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let t = self;
        if !t.q_heads.is_multiple_of(t.kv_heads) {
            return Err(Error::Config("q heads not divisible by kv heads".into()));
        }
        if t.q_heads / t.kv_heads > flint_model::step::MAX_GQA {
            return Err(Error::Config(format!(
                "GQA ratio {} exceeds the attention shader's {} head slots",
                t.q_heads / t.kv_heads,
                flint_model::step::MAX_GQA
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
        let max_hd = *t.head_dims.iter().max().expect("non-empty head_dims");
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
            return Err(Error::Config(
                "kv_shared must be < num_hidden_layers".into(),
            ));
        }
        if let Some(p) = t.per_layer
            && (p.dim == 0 || !(t.layers * p.dim).is_multiple_of(16))
        {
            return Err(Error::Config(
                "per-layer dim must be non-zero and layers*dim a gemm multiple of 16".into(),
            ));
        }
        check_gemm_dims(&[
            (t.vocab, t.hidden),
            (t.q_heads * max_hd, t.hidden),
            (t.kv_heads * max_hd, t.hidden),
            (t.hidden, t.q_heads * max_hd),
            (t.max_mlp_width(), t.hidden),
            (t.hidden, t.max_mlp_width()),
        ])
    }

    pub fn mlp_width(&self, l: u32) -> u32 {
        if self.double_wide[l as usize] {
            2 * self.intermediate
        } else {
            self.intermediate
        }
    }

    pub(crate) fn max_mlp_width(&self) -> u32 {
        (0..self.layers).map(|l| self.mlp_width(l)).max().unwrap()
    }

    pub fn first_shared(&self) -> u32 {
        self.layers - self.kv_shared
    }

    pub fn window(&self, l: u32) -> u32 {
        self.windows[l as usize]
    }

    pub fn head_dim(&self, l: u32) -> u32 {
        self.head_dims[l as usize]
    }

    pub fn has_kv(&self, l: u32) -> bool {
        l < self.first_shared()
    }

    pub fn has_ple(&self) -> bool {
        self.per_layer.is_some()
    }
}
