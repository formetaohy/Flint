use thuban_backend::Backend;
use thuban_checkpoint::Checkpoint;
use thuban_error::{Error, Result};
use thuban_model::loader::{Plan, Role, upload};
use thuban_model::ops::{Act, RopeScaling};
use thuban_model::pool::ArenaSpec;
use thuban_tensor::Weight;
use serde_json::Value;

use crate::keymap::gguf_key;
use crate::transformer::{Config, Model, RopeSpec, role as dense_role};

fn parse_transformer(v: &Value) -> Result<Config> {
    let mut cfg = Config::parse(v, true)?;
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

        freq_dim: dim,
        theta: cfg.rope[0].theta,
        partial: None,
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

    match v.get("rms_norm_eps").and_then(Value::as_f64) {
        Some(eps) => cfg.norm_eps = eps as f32,
        None => {
            if v.get("layer_norm_epsilon").is_some() {
                cfg.layernorm = true;
                cfg.norm_eps = v["layer_norm_epsilon"].as_f64().unwrap_or(1e-5) as f32;
            }
        }
    }

    cfg.act = match v.get("hidden_act").and_then(Value::as_str) {
        None | Some("silu") => Act::Silu,
        Some("gelu_pytorch_tanh") => Act::GeluTanh,
        other => {
            return Err(Error::Config(format!(
                "unsupported hidden_act {other:?} (Thuban supports silu and gelu_pytorch_tanh)"
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

fn plan() -> Plan {
    Plan {
        key: gguf_key,
        role: dense_role,
    }
}

pub fn load(
    source: &dyn Checkpoint,
    v: &Value,
    arena: &ArenaSpec,
    spec_depth: Option<u32>,
    backend: &Backend,
) -> Result<Model> {
    let cfg = parse_transformer(v)?;
    let extra = split_fused(backend, source, &cfg, dense_role)?;
    Model::load_extra(
        source,
        cfg,
        &plan(),
        extra,
        arena,
        spec_depth,
        backend,
    )
}

fn split_fused(
    backend: &Backend,
    source: &dyn Checkpoint,
    cfg: &Config,
    role: fn(&str) -> Role,
) -> Result<Vec<(String, Weight)>> {
    let hd = cfg.head_dims[0];
    let (qw, kvw) = (cfg.q_heads * hd, cfg.kv_heads * hd);
    let mut out = Vec::new();
    let names = source.names();
    for l in 0..cfg.layers {
        let qkv = format!("blk.{l}.attn_qkv.weight");
        let fused_mlp = format!("blk.{l}.ffn_up.weight");
        if names.contains(&qkv) {
            let raw = source.read(&qkv)?;
            if raw.shape.len() != 2 || raw.shape[0] != qw + 2 * kvw {
                return Err(Error::Model(format!(
                    "{qkv}: unexpected fused QKV shape {:?} (q {qw}, kv {kvw})",
                    raw.shape
                )));
            }
            split_rows(
                backend,
                raw,
                &mut out,
                role,
                |part| format!("layers.{l}.self_attn.{part}.weight"),
                &[
                    ("q_proj", 0, qw),
                    ("k_proj", qw, qw + kvw),
                    ("v_proj", qw + kvw, qw + 2 * kvw),
                ],
            )?;
        }
        if names.contains(&fused_mlp) {
            let raw = source.read(&fused_mlp)?;
            if raw.shape.len() != 2 || raw.shape[0] != 2 * cfg.intermediate {
                continue;
            }
            let half = raw.shape[0] / 2;
            split_rows(
                backend,
                raw,
                &mut out,
                role,
                |part| format!("layers.{l}.mlp.{part}.weight"),
                &[("gate_proj", 0, half), ("up_proj", half, 2 * half)],
            )?;
        }
    }
    Ok(out)
}

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

fn split_rows(
    backend: &Backend,
    raw: thuban_checkpoint::RawTensor,
    out: &mut Vec<(String, Weight)>,
    role: fn(&str) -> Role,
    key_for: impl Fn(&str) -> String,
    parts: &[(&str, u32, u32)],
) -> Result<()> {
    let k = raw.shape[1];
    match raw.data {
        thuban_checkpoint::TensorData::Quant { quant, bytes, .. } => {
            let bl = quant.block_len() as u32;
            for &(part, lo, hi) in parts {
                if !lo.is_multiple_of(bl) || !hi.is_multiple_of(bl) {
                    return Err(Error::Model(format!(
                        "{part}: fused row range [{lo}, {hi}) is not a multiple of the {quant:?} block length"
                    )));
                }
                let row_bytes = (k as usize / quant.block_len()) * quant.block_bytes();
                let slice = &bytes[lo as usize * row_bytes..hi as usize * row_bytes];
                let padded = quant.pad_blocks(slice, (hi - lo) as usize * k as usize)?;
                out.push((
                    key_for(part),
                    Weight::quantized(backend.tensor_quant(&padded, vec![hi - lo, k], quant)),
                ));
            }
        }
        other => {
            let data = other.into_f32()?;
            for &(part, lo, hi) in parts {
                let key = key_for(part);
                let slice: Vec<f32> =
                    data[(lo as usize * k as usize)..(hi as usize * k as usize)].to_vec();
                out.push((
                    key.clone(),
                    upload(
                        backend,
                        &key,
                        thuban_checkpoint::RawTensor {
                            shape: vec![hi - lo, k],
                            data: thuban_checkpoint::TensorData::F32(slice),
                        },
                        role(&key),
                    )?,
                ));
            }
        }
    }
    Ok(())
}
