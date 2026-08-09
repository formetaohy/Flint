use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, CheckpointKind};
use flint_error::{Error, Result};
use flint_model::ops::Act;
use serde_json::Value;

use crate::transformer::{TransformerConfig, TransformerModel, PerLayerConfig, RopeSpec, transformer_plan};

pub fn parse_config(v: &Value) -> Result<TransformerConfig> {
    let t = v.get("text_config").unwrap_or(v);
    let mut cfg = TransformerConfig::parse(t, true)?;
    cfg.embed_scale = (cfg.hidden as f32).sqrt();
    cfg.qk_norm = true;
    cfg.v_norm = true;
    cfg.act = match t.get("hidden_activation").and_then(Value::as_str) {
        Some("gelu_pytorch_tanh") => Act::GeluTanh,
        other => {
            return Err(Error::Config(format!(
                "unsupported hidden_activation {other:?}"
            )));
        }
    };

    let layer_types = t
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("gemma4_text missing layer_types".into()))?;
    if layer_types.len() != cfg.layers as usize {
        return Err(Error::Config("layer_types length mismatch".into()));
    }
    let global_hd = t
        .get("global_head_dim")
        .and_then(Value::as_u64)
        .unwrap_or(512) as u32;
    let sliding_hd = cfg.head_dims[0];
    let window = t.get("sliding_window").and_then(Value::as_u64).unwrap_or(0) as u32;
    let mut head_dims = Vec::with_capacity(cfg.layers as usize);
    let mut windows = Vec::with_capacity(cfg.layers as usize);
    let mut layer_rope = Vec::with_capacity(cfg.layers as usize);
    for lt in layer_types {
        match lt.as_str() {
            Some("sliding_attention") => {
                head_dims.push(sliding_hd);
                windows.push(window);
                layer_rope.push(0);
            }
            Some("full_attention") => {
                head_dims.push(global_hd);
                windows.push(0);
                layer_rope.push(1);
            }
            other => {
                return Err(Error::Config(format!("unknown layer type {other:?}")));
            }
        }
    }
    cfg.head_dims = head_dims;
    cfg.windows = windows;
    cfg.layer_rope = layer_rope;

    let rp = t
        .get("rope_parameters")
        .ok_or_else(|| Error::Config("gemma4_text missing rope_parameters".into()))?;
    let theta = |k: &str| -> Result<f64> {
        rp.get(k)
            .and_then(|x| x.get("rope_theta"))
            .and_then(Value::as_f64)
            .ok_or_else(|| Error::Config(format!("rope_parameters.{k}.rope_theta missing")))
    };
    let rot = rp
        .get("full_attention")
        .and_then(|x| x.get("partial_rotary_factor"))
        .and_then(Value::as_f64)
        .unwrap_or(0.25);
    cfg.rope = vec![
        RopeSpec {
            dim: sliding_hd,
            freq_dim: sliding_hd,
            theta: theta("sliding_attention")?,
            scaling: None,
        },
        RopeSpec {
            dim: (global_hd as f64 * rot) as u32,
            freq_dim: global_hd,
            theta: theta("full_attention")?,
            scaling: None,
        },
    ];

    cfg.kv_shared = t
        .get("num_kv_shared_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;

    if t.get("use_double_wide_mlp")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        cfg.double_wide = (0..cfg.layers).map(|l| l >= cfg.first_shared()).collect();
    }
    cfg.softcap = t
        .get("final_logit_softcapping")
        .and_then(Value::as_f64)
        .map(|c| c as f32);
    if let Some(d) = t
        .get("hidden_size_per_layer_input")
        .and_then(Value::as_u64)
        .filter(|&d| d > 0)
    {
        cfg.per_layer = Some(PerLayerConfig { dim: d as u32 });
    }
    if t.get("enable_moe_block")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Error::Config(
            "Gemma 4 enable_moe_block (26B-A4B) is not supported".into(),
        ));
    }
    cfg.validate()?;
    Ok(cfg)
}

pub fn load(
    source: &dyn Checkpoint,
    v: &Value,
    max_seq: u32,
    backend: &Backend,
) -> Result<TransformerModel> {
    let cfg = parse_config(v)?;
    TransformerModel::load(
        source,
        cfg,
        &transformer_plan(source.kind() == CheckpointKind::Gguf),
        max_seq,
        backend,
    )
}
