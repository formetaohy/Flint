//! Gemma 3: the shared dense model configured for Gemma's sandwich RMSNorm,
//! always-on per-head QK-norm, input embedding scaling and alternating
//! sliding-window attention.
//!
//! GGUF stores every Gemma norm weight already folded to its effective value
//! (HF's offset `w` is saved as `1 + w`), so all norms apply the weight
//! directly, like the rest of the dense family.

use flint_backend::Backend;
use flint_checkpoint::Checkpoint;
use flint_error::Result;
use flint_model::loader::Plan;
use serde_json::Value;

use crate::dense::{DenseConfig, DenseModel, SlidingWindow, dense_plan};

/// HF safetensors names -> canonical keys (model / language_model prefixes stripped).
fn hf_key(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        Some(rest.to_string())
    } else if let Some(rest) = name.strip_prefix("model.") {
        Some(rest.to_string())
    } else if name.starts_with("lm_head.") {
        Some(name.to_string())
    } else {
        None
    }
}

fn plan(gguf: bool) -> Plan {
    dense_plan(gguf, hf_key)
}

/// Parses and validates a Gemma 3 text config.
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
        cfg.window = Some(SlidingWindow { size, pattern });
    }
    cfg.validate()?;
    Ok(cfg)
}

/// Loads a Gemma 3 checkpoint as a shared dense model.
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
