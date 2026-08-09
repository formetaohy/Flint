use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::{Error, Result};
use serde_json::Value;

use crate::dense::{DenseConfig, DenseModel, dense_plan};

pub fn parse_config(v: &Value) -> Result<DenseConfig> {
    let mut cfg = DenseConfig::parse(v, true)?;
    cfg.embed_scale = (cfg.hidden as f32).sqrt();
    cfg.qk_norm = true;
    cfg.sandwich = true;
    let size = v.get("sliding_window").and_then(Value::as_u64).unwrap_or(0) as u32;
    if size > 0 {
        let pattern = v
            .get("sliding_window_pattern")
            .and_then(Value::as_u64)
            .unwrap_or(6) as u32;
        if pattern == 0 {
            return Err(Error::Config(
                "sliding_window_pattern must be non-zero".into(),
            ));
        }
        cfg.windows = (0..cfg.layers)
            .map(|l| {
                if (l + 1).is_multiple_of(pattern) {
                    0
                } else {
                    size
                }
            })
            .collect();
    }
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
