//! LLaMA family (LLaMA / Qwen2 / Qwen3 / Mistral): dense GQA with SwiGLU MLP,
//! optional QKV biases and per-head QK-norm. A configuration of the shared
//! `DenseModel`, no forward graph of its own.

use flint_backend::Backend;
use flint_checkpoint::Checkpoint;
use flint_error::Result;
use flint_model::loader::{Plan, Role};
use serde_json::Value;

use crate::dense::{DenseConfig, DenseModel};
use crate::gguf_config::gguf_key;

/// HF safetensors names -> canonical keys ("model." stripped, lm_head kept).
fn hf_key(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("model.") {
        Some(rest.to_string())
    } else if name.starts_with("lm_head.") {
        Some(name.to_string())
    } else {
        None
    }
}

fn role(key: &str) -> Role {
    if key.contains("norm") || key.ends_with(".bias") {
        Role::F32
    } else if key == "embed_tokens.weight" {
        Role::Bf16
    } else {
        Role::I8
    }
}

fn plan(gguf: bool) -> Plan {
    Plan {
        key: if gguf { gguf_key } else { hf_key },
        role,
    }
}

/// Parses and validates a LLaMA-family config.
pub fn parse_config(v: &Value) -> Result<DenseConfig> {
    let mut cfg = DenseConfig::parse(v, false)?;
    cfg.qkv_bias = v
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    cfg.qk_norm = v.get("qk_norm").and_then(Value::as_bool).unwrap_or(false);
    cfg.validate()?;
    Ok(cfg)
}

/// Loads a LLaMA-family checkpoint as a shared dense model.
pub fn load(
    source: &dyn Checkpoint,
    v: &Value,
    max_seq: u32,
    backend: &Backend,
) -> Result<DenseModel> {
    let cfg = parse_config(v)?;
    DenseModel::load(
        source,
        cfg,
        &plan(source.kind() == "gguf"),
        max_seq,
        backend,
    )
}
