use flint_error::{Error, Result};
use flint_model::config::{f64_field, u32_field, u32_list};
use flint_model::ops;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    pub rotary_dim: u32,
    pub rope_theta: f64,
    pub attn_scale: f32,
    pub norm_eps: f32,

    pub lin_key_heads: u32,
    pub lin_val_heads: u32,
    pub lin_key_dim: u32,
    pub lin_val_dim: u32,
    pub conv_kernel: u32,

    pub eos: Vec<u32>,
    pub tied: bool,
}

impl Qwen35Config {
    pub fn parse(v: &Value) -> Result<Self> {
        let layer_types = v
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Config("qwen35 missing layer_types".into()))?
            .iter()
            .map(|k| match k.as_str() {
                Some("linear_attention") => Ok(LayerKind::Linear),
                Some("full_attention") => Ok(LayerKind::Full),
                other => Err(Error::Config(format!("unknown layer type {other:?}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        let head_dim = u32_field(v, "head_dim")?;
        let cfg = Self {
            hidden: u32_field(v, "hidden_size")?,
            intermediate: u32_field(v, "intermediate_size")?,
            layers: u32_field(v, "num_hidden_layers")?,
            q_heads: u32_field(v, "num_attention_heads")?,
            kv_heads: u32_field(v, "num_key_value_heads")?,
            head_dim,
            vocab: u32_field(v, "vocab_size")?,
            layer_types,
            rotary_dim: u32_field(v, "rotary_dim")?,
            rope_theta: f64_field(v, "rope_theta")?,
            attn_scale: v
                .get("attention_scale")
                .and_then(Value::as_f64)
                .map(|s| s as f32)
                .unwrap_or_else(|| (head_dim as f32).sqrt().recip()),
            norm_eps: v
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .map(|e| e as f32)
                .unwrap_or(1e-6),
            lin_key_heads: u32_field(v, "linear_num_key_heads")?,
            lin_val_heads: u32_field(v, "linear_num_value_heads")?,
            lin_key_dim: u32_field(v, "linear_key_head_dim")?,
            lin_val_dim: u32_field(v, "linear_value_head_dim")?,
            conv_kernel: u32_field(v, "linear_conv_kernel_dim")?,
            eos: u32_list(v, "eos_token_id")?,
            tied: v
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let t = self;
        if t.layer_types.len() != t.layers as usize {
            return Err(Error::Config("layer_types length mismatch".into()));
        }
        if !t.q_heads.is_multiple_of(t.kv_heads) {
            return Err(Error::Config("q heads not divisible by kv heads".into()));
        }
        ops::check_head_dim(t.head_dim)?;
        if !t.rotary_dim.is_multiple_of(2) || t.rotary_dim > t.head_dim {
            return Err(Error::Config(format!(
                "rotary_dim {} invalid for head dim {}",
                t.rotary_dim, t.head_dim
            )));
        }
        if t.lin_key_dim > 128 || t.lin_val_dim > 128 {
            return Err(Error::Config("linear head dims must be <= 128".into()));
        }
        if !t.lin_val_heads.is_multiple_of(t.lin_key_heads) {
            return Err(Error::Config(format!(
                "linear value heads {} not divisible by key heads {}",
                t.lin_val_heads, t.lin_key_heads
            )));
        }
        if t.conv_kernel != 4 {
            return Err(Error::Config(format!(
                "linear conv kernel {} unsupported (only 4 taps)",
                t.conv_kernel
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
}
