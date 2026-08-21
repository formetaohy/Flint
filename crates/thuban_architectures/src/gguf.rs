use thuban_checkpoint::{Checkpoint, Metadata};
use thuban_error::{Error, Result};
use serde_json::{Value, json};

use crate::Family;

pub fn synthesize_config(source: &dyn Checkpoint, family: Family) -> Result<Value> {
    match family {
        Family::Llama => transformer_config(source, false),
        Family::Gemma => transformer_config(source, true),
        Family::Gemma4 => gemma4_config(source),
        Family::Phi => phi_config(source),
        Family::Qwen35 => qwen35_config(source),
    }
}

fn transformer_config(source: &dyn Checkpoint, gemma: bool) -> Result<Value> {
    let m = source.metadata()?;
    let arch = m
        .str("general.architecture")
        .ok_or_else(|| Error::Config("GGUF missing general.architecture".into()))?;
    let key = |k: &str| format!("{arch}.{k}");
    let req = |k: &str| -> Result<u32> {
        m.u32(&key(k))
            .ok_or_else(|| Error::Config(format!("GGUF metadata missing {}", key(k))))
    };

    let hidden = req("embedding_length")?;
    let heads = req("attention.head_count")?;
    let kv = m.u32(&key("attention.head_count_kv")).unwrap_or(heads);
    let head_dim = m
        .u32(&key("attention.key_length"))
        .unwrap_or(hidden / heads);
    let intermediate = req("feed_forward_length")?;
    let layers = req("block_count")?;
    let rope_theta = m.f64(&key("rope.freq_base")).unwrap_or(10000.0);

    let vocab = vocab_size(m, arch)?;
    let eos = eos_ids(m);
    let tied = tied(source);

    let mut cfg = json!({
        "model_type": if gemma { "gemma3" } else { "llama" },
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "rope_theta": rope_theta,
        "eos_token_id": eos,
        "tie_word_embeddings": tied,
    });

    if let Some(freqs) = m.f64_array("rope_freqs") {
        cfg["rope_freqs"] = json!(freqs);
    }

    if gemma {
        let sliding_window = m.u32(&key("attention.sliding_window")).unwrap_or(0);
        let sliding_window_pattern = m.u32(&key("attention.sliding_window_pattern")).unwrap_or(6);
        cfg["sliding_window"] = json!(sliding_window);
        cfg["sliding_window_pattern"] = json!(sliding_window_pattern);
        cfg["qk_norm"] = json!(
            source
                .names()
                .iter()
                .any(|n| n == "blk.0.attn_q_norm.weight")
        );

        if let Some(id) = m.str_array("tokenizer.ggml.tokens").and_then(|t| {
            t.iter()
                .position(|s| *s == "<end_of_turn>")
                .map(|i| i as u32)
        }) && !eos.contains(&id)
        {
            cfg["eos_token_id"] = json!([eos.as_slice(), &[id]].concat());
        }
    } else {
        let bias = source.names().iter().any(|n| n == "blk.0.attn_q.bias");

        let qk_norm = source
            .names()
            .iter()
            .any(|n| n == "blk.0.attn_q_norm.weight");
        cfg["attention_bias"] = json!(bias);
        cfg["qk_norm"] = json!(qk_norm);
    }

    Ok(cfg)
}

fn phi_config(source: &dyn Checkpoint) -> Result<Value> {
    let m = source.metadata()?;
    let key = |k: &str| format!("phi3.{k}");
    let req = |k: &str| -> Result<u32> {
        m.u32(&key(k))
            .ok_or_else(|| Error::Config(format!("GGUF metadata missing {}", key(k))))
    };
    let hidden = req("embedding_length")?;
    let heads = req("attention.head_count")?;
    let head_dim = m
        .u32(&key("attention.key_length"))
        .unwrap_or(hidden / heads);
    let rot = req("rope.dimension_count")?;
    if rot > head_dim || !rot.is_multiple_of(2) {
        return Err(Error::Config(format!(
            "phi3 rope dimension_count {rot} invalid for head dim {head_dim}"
        )));
    }
    let mut cfg = json!({
        "model_type": "phi3",
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": req("attention.head_count_kv")?,
        "head_dim": head_dim,
        "intermediate_size": req("feed_forward_length")?,
        "num_hidden_layers": req("block_count")?,
        "vocab_size": vocab_size(m, "phi3")?,
        "rope_theta": m.f64(&key("rope.freq_base")).unwrap_or(10000.0),
        "partial_rotary_factor": rot as f64 / head_dim as f64,
        "eos_token_id": eos_ids(m),
        "tie_word_embeddings": tied(source),
    });

    if let (Ok(sf), Ok(lf)) = (
        source.read("rope_factors_short.weight"),
        source.read("rope_factors_long.weight"),
    ) {
        let short = sf.data.into_f32()?;
        let long = lf.data.into_f32()?;
        if !short.is_empty() && short.len() == long.len() {
            cfg["rope_scaling"] = json!({
                "type": "longrope",
                "short_factor": short,
                "long_factor": long,
                "original_max_position_embeddings": m
                    .u32(&key("rope.scaling.original_context_length"))
                    .unwrap_or(4096),
            });
        }
    }
    if let Some(eps) = m.f64(&key("attention.layer_norm_rms_epsilon")) {
        cfg["rms_norm_eps"] = json!(eps);
    }
    Ok(cfg)
}

fn gemma4_config(source: &dyn Checkpoint) -> Result<Value> {
    let m = source.metadata()?;
    let key = |k: &str| format!("gemma4.{k}");
    let req = |k: &str| -> Result<u32> {
        m.u32(&key(k))
            .ok_or_else(|| Error::Config(format!("GGUF metadata missing {}", key(k))))
    };
    let hidden = req("embedding_length")?;
    let heads = req("attention.head_count")?;
    let kv_heads = req("attention.head_count_kv")?;
    let hd_swa = req("attention.key_length_swa")?;
    let hd_full = req("attention.key_length")?;
    let layers = req("block_count")?;
    let sliding = m
        .u32_array(&key("attention.sliding_window_pattern"))
        .or_else(|| {
            m.u32(&key("attention.sliding_window_pattern")).map(|p| {
                (0..layers)
                    .map(|l| (l + 1) % p != 0)
                    .map(|b| b as u32)
                    .collect()
            })
        })
        .ok_or_else(|| Error::Config("gemma4 missing sliding_window_pattern".into()))?;
    if sliding.len() != layers as usize {
        return Err(Error::Config(
            "sliding_window_pattern length mismatch".into(),
        ));
    }
    let layer_types: Vec<Value> = sliding
        .iter()
        .map(|&s| {
            json!(if s != 0 {
                "sliding_attention"
            } else {
                "full_attention"
            })
        })
        .collect();

    let intermediate = m
        .u32_array(&key("feed_forward_length"))
        .map(|v| *v.iter().min().unwrap())
        .or_else(|| m.u32(&key("feed_forward_length")))
        .ok_or_else(|| Error::Config("gemma4 missing feed_forward_length".into()))?;
    let mut rot_full = hd_full;
    if let Ok(raw) = source.read("rope_freqs.weight") {
        let freqs = raw.data.into_f32()?;
        let real = freqs.iter().take_while(|v| v.abs() < 1e10).count();
        if real > 0 {
            rot_full = (real as u32 * 2).min(hd_full);
        }
    }
    let mut cfg = json!({
        "model_type": "gemma4_text",
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": hd_swa,
        "global_head_dim": hd_full,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "vocab_size": vocab_size(m, "gemma4")?,
        "layer_types": layer_types,
        "sliding_window": req("attention.sliding_window")?,
        "rope_theta": m.f64(&key("rope.freq_base_swa")).unwrap_or(10000.0),
        "rope_parameters": {
            "sliding_attention": {
                "rope_theta": m.f64(&key("rope.freq_base_swa")).unwrap_or(10000.0),
                "rope_type": "default",
            },
            "full_attention": {
                "rope_theta": m.f64(&key("rope.freq_base")).unwrap_or(1000000.0),
                "partial_rotary_factor": rot_full as f64 / hd_full as f64,
                "rope_type": "proportional",
            },
        },
        "num_kv_shared_layers": m.u32(&key("attention.shared_kv_layers")).unwrap_or(0),
        "use_double_wide_mlp": true,
        "hidden_activation": "gelu_pytorch_tanh",
        "eos_token_id": eos_ids(m),
        "tie_word_embeddings": tied(source),
    });
    if let Some(cap) = m.f64(&key("final_logit_softcapping")) {
        cfg["final_logit_softcapping"] = json!(cap);
    }
    if let Some(d) = m
        .u32(&key("embedding_length_per_layer_input"))
        .filter(|&d| d > 0)
    {
        cfg["hidden_size_per_layer_input"] = json!(d);
    }
    if let Some(eps) = m.f64(&key("attention.layer_norm_rms_epsilon")) {
        cfg["rms_norm_eps"] = json!(eps);
    }
    Ok(cfg)
}

fn vocab_size(m: &Metadata, arch: &str) -> Result<u32> {
    m.u32(&format!("{arch}.vocab_size"))
        .or_else(|| m.str_array("tokenizer.ggml.tokens").map(|t| t.len() as u32))
        .ok_or_else(|| Error::Config("GGUF has no vocab size".into()))
}

fn qwen35_config(source: &dyn Checkpoint) -> Result<Value> {
    let m = source.metadata()?;
    let key = |k: &str| format!("qwen35.{k}");
    let req = |k: &str| -> Result<u32> {
        m.u32(&key(k))
            .ok_or_else(|| Error::Config(format!("GGUF metadata missing {}", key(k))))
    };

    let hidden = req("embedding_length")?;
    let heads = req("attention.head_count")?;
    let kv = req("attention.head_count_kv")?;
    let head_dim = req("attention.key_length")?;
    if let Some(v) = m.u32(&key("attention.value_length"))
        && v != head_dim
    {
        return Err(Error::Config(format!(
            "asymmetric head dims (key {head_dim}, value {v}) unsupported"
        )));
    }
    let block_count = req("block_count")?;
    let nextn = m.u32(&key("nextn_predict_layers")).unwrap_or(0);
    let layers = block_count.checked_sub(nextn).ok_or_else(|| {
        Error::Config(format!(
            "nextn_predict_layers {nextn} exceeds block_count {block_count}"
        ))
    })?;
    if layers == 0 {
        return Err(Error::Config("qwen35 trunk has zero layers".into()));
    }

    let layer_types: Vec<Value> = if let Some(rec) = m.u32_array(&key("attention.recurrent_layers")) {
        if rec.len() < layers as usize {
            return Err(Error::Config("recurrent_layers shorter than the trunk".into()));
        }
        rec[..layers as usize]
            .iter()
            .map(|&r| {
                if r != 0 { "linear_attention" } else { "full_attention" }.into()
            })
            .collect()
    } else {
        let interval = m
            .u32(&key("full_attention_interval"))
            .ok_or_else(|| {
                Error::Config(
                    "qwen35 has neither attention.recurrent_layers nor full_attention_interval"
                        .into(),
                )
            })?;
        if interval == 0 {
            return Err(Error::Config("full_attention_interval must be non-zero".into()));
        }
        (0..layers)
            .map(|l| {
                if (l + 1) % interval != 0 {
                    "linear_attention"
                } else {
                    "full_attention"
                }
                .into()
            })
            .collect()
    };

    let lin_val_heads = req("ssm.time_step_rank")?;
    let lin_inner = req("ssm.inner_size")?;
    if !lin_inner.is_multiple_of(lin_val_heads) {
        return Err(Error::Config(format!(
            "ssm.inner_size {lin_inner} not divisible by time_step_rank {lin_val_heads}"
        )));
    }

    let mut cfg = json!({
        "model_type": "qwen35",
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "intermediate_size": req("feed_forward_length")?,
        "num_hidden_layers": layers,
        "vocab_size": vocab_size(m, "qwen35")?,
        "layer_types": layer_types,
        "rotary_dim": m.u32(&key("rope.dimension_count")).unwrap_or(head_dim),
        "rope_theta": m.f64(&key("rope.freq_base")).unwrap_or(10_000_000.0),
        "rms_norm_eps": m
            .f64(&key("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6),
        "linear_num_key_heads": req("ssm.group_count")?,
        "linear_num_value_heads": lin_val_heads,
        "linear_key_head_dim": req("ssm.state_size")?,
        "linear_value_head_dim": lin_inner / lin_val_heads,
        "linear_conv_kernel_dim": m.u32(&key("ssm.conv_kernel")).unwrap_or(4),
        "eos_token_id": eos_ids(m),
        "tie_word_embeddings": tied(source),
    });

    if let Some(scale) = m.f64(&key("attention.scale")) {
        cfg["attention_scale"] = json!(scale);
    }
    Ok(cfg)
}

fn eos_ids(m: &Metadata) -> Vec<u32> {
    let mut eos = Vec::new();
    if let Some(id) = m.u32("tokenizer.ggml.eos_token_id") {
        eos.push(id);
    }
    eos
}

fn tied(source: &dyn Checkpoint) -> bool {
    !source.names().iter().any(|n| n == "output.weight")
}
