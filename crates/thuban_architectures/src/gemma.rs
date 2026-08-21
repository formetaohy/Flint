use thuban_backend::Backend;
use thuban_checkpoint::Checkpoint;
use thuban_error::{Error, Result};
use serde_json::Value;

use crate::transformer::{Config, Model, plan};
use thuban_model::pool::ArenaSpec;

pub fn parse_config(v: &Value) -> Result<Config> {
    let mut cfg = Config::parse(v, true)?;
    cfg.embed_scale = (cfg.hidden as f32).sqrt();
    cfg.qk_norm = v.get("qk_norm").and_then(Value::as_bool).unwrap_or(true);
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
    arena: &ArenaSpec,
    spec_depth: Option<u32>,
    backend: &Backend,
) -> Result<Model> {
    let cfg = parse_config(v)?;
    Model::load(
        source,
        cfg,
        &plan(),
        arena,
        spec_depth,
        backend,
    )
}
