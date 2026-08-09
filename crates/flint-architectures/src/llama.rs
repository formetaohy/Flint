use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::Result;
use serde_json::Value;

use crate::dense::{DenseConfig, DenseModel, dense_plan};

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
        &dense_plan(source.kind() == CheckpointKind::Gguf),
        max_seq,
        backend,
    )
}
