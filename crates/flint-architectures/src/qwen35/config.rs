use flint_error::{Error, Result};
use flint_model::config::{f64_field, req, u32_field, u32_list};
use flint_model::ops;
use serde_json::Value;

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
