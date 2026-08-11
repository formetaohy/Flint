use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::Result;
use serde_json::Value;

use crate::transformer::{TransformerConfig, TransformerModel, transformer_plan};

pub fn parse_config(v: &Value) -> Result<TransformerConfig> {
    let mut cfg = TransformerConfig::parse(v, false)?;
    cfg.qkv_bias = v
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    cfg.qk_norm = v.get("qk_norm").and_then(Value::as_bool).unwrap_or(false);
    cfg.validate()?;
    Ok(cfg)
}

pub fn load(
    source: &dyn Checkpoint,
    v: &Value,
    max_seq: u32,
    backend: &Backend,
) -> Result<TransformerModel> {
    let mut cfg = parse_config(v)?;
    if source
        .names()
        .iter()
        .any(|n| n.ends_with("self_attn.q_proj.bias"))
    {
        cfg.qkv_bias = true;
    }
    if source
        .names()
        .iter()
        .any(|n| n.ends_with("self_attn.q_norm.weight"))
    {
        cfg.qk_norm = true;
    }
    TransformerModel::load(
        source,
        cfg,
        &transformer_plan(source.kind() == CheckpointKind::Gguf),
        max_seq,
        backend,
    )
}
