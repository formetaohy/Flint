//! Phi family: Phi-4-mini (dense, partial rotary, LongRoPE) and Phi-MoE
//! (layer-normed MoE with the sparsemixer router). Both are configurations of
//! the shared dense model; Phi-4-mini differs from LLaMA in its rotary range
//! and LongRoPE factors, Phi-MoE in its LayerNorm, uniform sliding window and
//! MoE feed-forward.

use flint_backend::Backend;
use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use flint_model::loader::{MoEPart, MoEPlan, Plan, Role, load_moe as load_moe_weights};
use flint_model::ops::{Act, RopeScaling};
use flint_model::routing::RouteKind;
use flint_tensor::Weight;
use serde_json::Value;

use crate::dense::{DenseConfig, DenseModel, MoeConfig, RopeSpec, dense_role};
use crate::gguf_config::gguf_key;
use crate::llama::hf_key;

/// HF safetensors names for Phi-MoE's expert set. Two layouts exist:
/// `model.layers.{l}.mlp.{router,gate_up_proj,down_proj}` (fused 3D) and the
/// Mixtral-style `model.layers.{l}.block_sparse_moe.{gate,experts.E.w1/w2/w3}`
/// (per-expert 2D, w1 = gate, w2 = down, w3 = up).
fn moe_key(name: &str) -> Option<(String, MoEPart)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    let prefix = format!("layers.{idx}.mlp");
    if let Some(t) = tail.strip_prefix("mlp.") {
        return match t {
            "router" => Some((prefix, MoEPart::Router)),
            "gate_up_proj" => Some((prefix, MoEPart::GateUp)),
            "down_proj" => Some((prefix, MoEPart::Down)),
            _ => None,
        };
    }
    let t = tail.strip_prefix("block_sparse_moe.")?;
    if t == "gate.weight" {
        return Some((prefix, MoEPart::Router));
    }
    let rest = t.strip_prefix("experts.")?;
    let (e, w) = rest.split_once(".w")?;
    let part = match w.trim_end_matches(".weight") {
        "1" => MoEPart::Gate,
        "2" => MoEPart::Down,
        "3" => MoEPart::Up,
        _ => return None,
    };
    Some((format!("{prefix}.experts.{e}"), part))
}

/// Parses a Phi-3.x dense config (Phi-4-mini): partial rotary, LongRoPE,
/// family norm epsilon and activation.
pub fn parse_dense(v: &Value) -> Result<DenseConfig> {
    let mut cfg = DenseConfig::parse(v, true)?;
    let hd = cfg.head_dims[0];
    let factor = v
        .get("partial_rotary_factor")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let dim = (hd as f64 * factor) as u32;
    if dim == 0 || dim > hd || !dim.is_multiple_of(2) {
        return Err(Error::Config(format!(
            "invalid partial_rotary_factor {factor} for head dim {hd}"
        )));
    }
    let mut rope = RopeSpec {
        dim,
        // Phi's inverse frequencies use the rotary dim as their denominator.
        freq_dim: dim,
        theta: cfg.rope[0].theta,
        scaling: None,
    };
    if let Some(rs) = v.get("rope_scaling") {
        match rs.get("type").and_then(Value::as_str) {
            None | Some("longrope") => {
                rope.scaling = Some(RopeScaling {
                    short: f32_list(rs, "short_factor")?,
                    long: f32_list(rs, "long_factor")?,
                    original_max: rs
                        .get("original_max_position_embeddings")
                        .and_then(Value::as_u64)
                        .unwrap_or(4096) as u32,
                });
            }
            Some(other) => {
                return Err(Error::Config(format!(
                    "unsupported rope_scaling type {other:?}"
                )));
            }
        }
    }
    cfg.rope = vec![rope];
    // Phi-4-mini normalizes with RMSNorm; older Phi-3 checkpoints declare a
    // LayerNorm epsilon instead.
    match v.get("rms_norm_eps").and_then(Value::as_f64) {
        Some(eps) => cfg.norm_eps = eps as f32,
        None => {
            if v.get("layer_norm_epsilon").is_some() {
                cfg.layernorm = true;
                cfg.norm_eps = v["layer_norm_epsilon"].as_f64().unwrap_or(1e-5) as f32;
            }
        }
    }
    // GGUF phi3 checkpoints carry no activation metadata; silu is the default.
    cfg.act = match v.get("hidden_act").and_then(Value::as_str) {
        None | Some("silu") => Act::Silu,
        Some("gelu_pytorch_tanh") => Act::GeluTanh,
        other => {
            return Err(Error::Config(format!(
                "unsupported hidden_act {other:?} (Flint supports silu and gelu_pytorch_tanh)"
            )));
        }
    };
    cfg.qkv_bias = v
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    cfg.validate()?;
    Ok(cfg)
}

/// Parses a Phi-MoE config: LayerNorm, QKV/logits biases, a uniform sliding
/// window and the sparsemixer-routed MoE feed-forward.
pub fn parse_moe(v: &Value) -> Result<DenseConfig> {
    let mut cfg = DenseConfig::parse(v, false)?;
    cfg.layernorm = true;
    cfg.lm_bias = v
        .get("lm_head_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    cfg.qkv_bias = true;
    cfg.norm_eps = v
        .get("rms_norm_eps")
        .and_then(Value::as_f64)
        .unwrap_or(1e-5) as f32;
    let window = v
        .get("sliding_window")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    cfg.windows = vec![window; cfg.layers as usize];
    cfg.moe = Some(MoeConfig {
        experts: flint_model::config::u32_field(v, "num_local_experts")?,
        top_k: flint_model::config::u32_field(v, "num_experts_per_tok")?,
        scale: 1.0,
        shared_scale: 0.0,
        kind: RouteKind::SparseMixer { jitter: 0.01 },
    });
    cfg.validate()?;
    Ok(cfg)
}

/// Phi's storage roles: the embedding table is group-quantized i8 like the
/// projections — Phi-4-mini's table alone is 1.23 GiB as bf16, over budget on
/// 6 GiB adapters.
fn role(key: &str) -> Role {
    if key == "embed_tokens.weight" {
        Role::I8
    } else {
        dense_role(key)
    }
}

fn plan(gguf: bool) -> Plan {
    Plan {
        key: if gguf { gguf_key } else { hf_key },
        role,
    }
}

/// Loads a Phi dense checkpoint (Phi-4-mini / Phi-3.x) as a shared dense model.
pub fn load(
    source: &dyn Checkpoint,
    v: &Value,
    max_seq: u32,
    backend: &Backend,
) -> Result<DenseModel> {
    let cfg = parse_dense(v)?;
    let gguf = source.kind() == "gguf";
    let extra = if gguf {
        gguf_split_qkv(backend, source, &cfg, role)?
    } else {
        Vec::new()
    };
    DenseModel::load_extra(source, cfg, &plan(gguf), extra, max_seq, backend)
}

/// Dense-plan key for Phi-MoE: the expert-set tensors are excluded so the
/// split loader owns them.
fn hf_key_moe_dense(name: &str) -> Option<String> {
    if moe_key(name).is_some() {
        return None;
    }
    hf_key(name)
}

/// Loads a Phi-MoE checkpoint: the dense plan plus the split expert tensors.
pub fn load_moe(
    source: &dyn Checkpoint,
    v: &Value,
    max_seq: u32,
    backend: &Backend,
) -> Result<DenseModel> {
    let cfg = parse_moe(v)?;
    let experts = cfg.moe.expect("MoE config").experts;
    let gguf = source.kind() == "gguf";
    let plan = if gguf {
        Plan {
            key: gguf_key,
            role,
        }
    } else {
        Plan {
            key: hf_key_moe_dense,
            role,
        }
    };
    let moe_plan = MoEPlan {
        key: if gguf { crate::gguf_config::gguf_moe_key } else { moe_key },
        experts,
        shared: false,
    };
    let extra = load_moe_weights(backend, source, &moe_plan, dense_role)?;
    DenseModel::load_extra(source, cfg, &plan, extra, max_seq, backend)
}

/// LongRoPE factor lists (length must equal rotary dim / 2).
fn f32_list(v: &Value, key: &str) -> Result<Vec<f32>> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|x| {
                    x.as_f64()
                        .ok_or_else(|| Error::Config(format!("{key} entry is not a number")))
                        .map(|f| f as f32)
                })
                .collect()
        })
        .unwrap_or_else(|| Err(Error::Config(format!("missing {key:?}"))))
}

/// GGUF phi3 checkpoints fuse QKV into `attn_qkv.weight` and gate+up into
/// `ffn_up.weight`; splits both into the canonical keys.
pub fn gguf_split_qkv(
    backend: &Backend,
    source: &dyn Checkpoint,
    cfg: &DenseConfig,
    role: fn(&str) -> Role,
) -> Result<Vec<(String, Weight)>> {
    let hd = cfg.head_dims[0];
    let (qw, kvw) = (cfg.q_heads * hd, cfg.kv_heads * hd);
    let mut out = Vec::new();
    let names = source.names();
    for l in 0..cfg.layers {
        let name = format!("blk.{l}.attn_qkv.weight");
        if !names.iter().any(|n| *n == name) {
            continue;
        }
        let raw = source.read(&name)?;
        if raw.shape.len() != 2 || raw.shape[0] != qw + 2 * kvw {
            return Err(Error::Config(format!(
                "{name}: unexpected fused QKV shape {:?} (q {qw}, kv {kvw})",
                raw.shape
            )));
        }
        split_rows(
            backend,
            &raw,
            &mut out,
            role,
            |part| format!("layers.{l}.self_attn.{part}.weight"),
            &[
                ("q_proj", (0, qw)),
                ("k_proj", (qw, qw + kvw)),
                ("v_proj", (qw + kvw, qw + 2 * kvw)),
            ],
        )?;
    }
    // Fused gate+up in ffn_up: rows [N, 2*inter] split in half. Checkpoints
    // with a plain up projection carry the single width and skip the split.
    for l in 0..cfg.layers {
        let name = format!("blk.{l}.ffn_up.weight");
        if !names.iter().any(|n| *n == name) {
            continue;
        }
        let raw = source.read(&name)?;
        if raw.shape.len() != 2 || raw.shape[0] != 2 * cfg.intermediate {
            continue;
        }
        let half = raw.shape[0] / 2;
        split_rows(
            backend,
            &raw,
            &mut out,
            role,
            |part| format!("layers.{l}.mlp.{part}.weight"),
            &[("gate_proj", (0, half)), ("up_proj", (half, 2 * half))],
        )?;
    }
    Ok(out)
}

/// Splits a raw [N, K] tensor's leading axis into ranges and uploads each as
/// a canonical key (the fused QKV / gate+up paths).
#[allow(clippy::type_complexity)]
fn split_rows(
    backend: &Backend,
    raw: &flint_checkpoint::RawTensor,
    out: &mut Vec<(String, Weight)>,
    role: fn(&str) -> Role,
    key_for: impl Fn(&str) -> String,
    parts: &[(&str, (u32, u32))],
) -> Result<()> {
    use flint_checkpoint::{RawTensor, TensorData};
    use flint_model::loader::upload_key;
    let k = raw.shape[1];
    let _ = k;
    let mut data: Option<Vec<f32>> = None;
    for (part, (lo, hi)) in parts {
        let key = key_for(part);
        let data = match &data {
            Some(d) => d,
            None => {
                let d = match &raw.data {
                    TensorData::F32(v) => v.clone(),
                    TensorData::Bf16(b) => b
                        .chunks_exact(2)
                        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                        .collect(),
                };
                data = Some(d);
                data.as_ref().unwrap()
            }
        };
        let slice: Vec<f32> = data[(*lo as usize * k as usize)..(*hi as usize * k as usize)].to_vec();
        out.push((
            key.clone(),
            upload_key(
                backend,
                &key,
                RawTensor {
                    shape: vec![hi - lo, k],
                    data: TensorData::F32(slice),
                },
                role(&key),
            )?,
        ));
    }
    let _ = k;
    Ok(())
}
