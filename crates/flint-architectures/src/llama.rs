use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::Result;
use serde_json::Value;

use crate::transformer::{Config, Model, plan};
use flint_model::pool::ArenaSpec;

pub fn parse_config(v: &Value) -> Result<Config> {
    let mut cfg = Config::parse(v, false)?;
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
    arena: &ArenaSpec,
    spec_depth: Option<u32>,
    backend: &Backend,
) -> Result<Model> {
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
    Model::load(
        source,
        cfg,
        &plan(source.kind() == CheckpointKind::Gguf),
        arena,
        spec_depth,
        backend,
    )
}
