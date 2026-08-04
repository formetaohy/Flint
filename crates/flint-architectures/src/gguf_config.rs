//! Synthesizes a HF-shaped config `Value` from GGUF metadata and maps GGUF
//! tensor names onto the canonical keys the forward graphs read. GGUF tensor
//! names are architecture-generic (`blk.N.attn_q.weight`), so one map serves the
//! whole dense-GQA family.

use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use serde_json::{Value, json};

use crate::Family;

/// Maps a GGUF tensor name to its canonical registry key, or None to skip.
pub fn gguf_key(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("token_embd.weight") {
        return Some(format!("embed_tokens.weight{rest}"));
    }
    if name == "output.weight" {
        return Some("lm_head.weight".into());
    }
    if name == "output_norm.weight" {
        return Some("norm.weight".into());
    }
    // Gemma 4's Per-Layer Embeddings (model level); rope_freqs feeds config
    // synthesis only.
    match name {
        "per_layer_token_embd.weight" => return Some("embed_tokens_per_layer.weight".into()),
        "per_layer_model_proj.weight" => return Some("per_layer_model_projection.weight".into()),
        "per_layer_proj_norm.weight" => return Some("per_layer_projection_norm.weight".into()),
        "rope_freqs.weight" => return None,
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let layer: u32 = idx.parse().ok()?;
    // The per-layer scalar buffer carries no suffix on the canonical side.
    if tail == "layer_output_scale.weight" {
        return Some(format!("layers.{layer}.layer_scalar"));
    }
    let (stem, suffix) = tail.rsplit_once('.')?;
    let canon = match stem {
        "attn_norm" => "input_layernorm",
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "ffn_norm" => "post_attention_layernorm",
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        // Gemma's sandwich norms, applied to the block outputs before residual.
        "post_attention_norm" => "post_attention_norm",
        "post_ffw_norm" => "post_ffw_norm",
        // Gemma 4's Per-Layer Embeddings and per-layer scalar.
        "inp_gate" => "per_layer_input_gate",
        "proj" => "per_layer_projection",
        "post_norm" => "post_per_layer_input_norm",
        "layer_output_scale" => "layer_scalar",
        _ => return None,
    };
    Some(format!("layers.{layer}.{canon}.{suffix}"))
}

/// Maps a GGUF MoE tensor name to its canonical block prefix plus part
/// (llama.cpp conventions: `ffn_gate_inp` router, `*_exps` experts and
/// `*_shexp` shared expert).
pub fn gguf_moe_key(name: &str) -> Option<(String, flint_model::loader::MoEPart)> {
    use flint_model::loader::MoEPart;
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let prefix = format!("layers.{idx}.mlp");
    match tail {
        "ffn_gate_inp.weight" => Some((prefix, MoEPart::Router)),
        "ffn_gate_up_exps.weight" => Some((prefix, MoEPart::GateUp)),
        "ffn_gate_exps.weight" => Some((prefix, MoEPart::Gate)),
        "ffn_up_exps.weight" => Some((prefix, MoEPart::Up)),
        "ffn_down_exps.weight" => Some((prefix, MoEPart::Down)),
        "ffn_gate_shexp.weight" => Some((prefix, MoEPart::SharedGate)),
        "ffn_up_shexp.weight" => Some((prefix, MoEPart::SharedUp)),
        "ffn_down_shexp.weight" => Some((prefix, MoEPart::SharedDown)),
        _ => None,
    }
}

/// Builds the config the target architecture's parser expects.
pub fn synthesize_config(source: &dyn Checkpoint, family: Family) -> Result<Value> {
    match family {
        Family::Llama => dense_config(source, false),
        Family::Gemma => dense_config(source, true),
        Family::Gemma4 => gemma4_config(source),
        Family::Phi => phi_config(source),
        Family::Qwen35 | Family::PhiMoe => Err(Error::Config(
            "this architecture ships no GGUF (no ggml representation)".into(),
        )),
    }
}

/// Synthesizes the config for a dense-GQA family (Llama or Gemma). Gemma adds
/// the sliding-window fields and terminates on <end_of_turn>; Llama detects
/// QKV biases and QK-norm from the layer-0 tensor names.
fn dense_config(source: &dyn Checkpoint, gemma: bool) -> Result<Value> {
    let m = source.metadata();
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

    // Vocab: explicit field, else the tokenizer token table length.
    let vocab = m
        .u32(&key("vocab_size"))
        .or_else(|| m.str_array("tokenizer.ggml.tokens").map(|t| t.len() as u32))
        .ok_or_else(|| Error::Config("GGUF has no vocab size".into()))?;

    let mut eos = Vec::new();
    if let Some(id) = m.u32("tokenizer.ggml.eos_token_id") {
        eos.push(id);
    }
    // Tied embeddings: checkpoints without a separate output projection reuse
    // the token embedding table as the logits head.
    let tied = !source.names().iter().any(|n| n == "output.weight");

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

    if gemma {
        let sliding_window = m.u32(&key("attention.sliding_window")).unwrap_or(0);
        let sliding_window_pattern = m.u32(&key("attention.sliding_window_pattern")).unwrap_or(6);
        cfg["sliding_window"] = json!(sliding_window);
        cfg["sliding_window_pattern"] = json!(sliding_window_pattern);
        // Gemma also terminates on the <end_of_turn> marker when chatting.
        if let Some(id) = m.str_array("tokenizer.ggml.tokens").and_then(|t| {
            t.iter()
                .position(|s| *s == "<end_of_turn>")
                .map(|i| i as u32)
        }) && !eos.contains(&id)
        {
            cfg["eos_token_id"] = json!([eos.as_slice(), &[id]].concat());
        }
    } else {
        // Q/K/V biases: present in Qwen2 small models and Phi.
        let bias = source.names().iter().any(|n| n == "blk.0.attn_q.bias");
        // QK-norm: present in Qwen3 (per-head RMSNorm weights on Q and K).
        let qk_norm = source
            .names()
            .iter()
            .any(|n| n == "blk.0.attn_q_norm.weight");
        cfg["attention_bias"] = json!(bias);
        cfg["qk_norm"] = json!(qk_norm);
    }

    Ok(cfg)
}

/// Synthesizes the config for Phi-3.x / Phi-4-mini GGUFs (arch `phi3`): the
/// llama-style fields plus the rotary range and norm epsilon.
fn phi_config(source: &dyn Checkpoint) -> Result<Value> {
    let m = source.metadata();
    let key = |k: &str| format!("phi3.{k}");
    let req = |k: &str| -> Result<u32> {
        m.u32(&key(k))
            .ok_or_else(|| Error::Config(format!("GGUF metadata missing {}", key(k))))
    };
    let hidden = req("embedding_length")?;
    let heads = req("attention.head_count")?;
    let head_dim = m.u32(&key("attention.key_length")).unwrap_or(hidden / heads);
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
    // LongRoPE factors travel as dedicated tensors in phi3 GGUFs.
    if let (Ok(sf), Ok(lf)) = (
        source.read("rope_factors_short.weight"),
        source.read("rope_factors_long.weight"),
    ) {
        let short = sf.data.into_f32();
        let long = lf.data.into_f32();
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

/// Synthesizes the config for Gemma 4 GGUFs: per-layer head dims and windows
/// (from the per-layer sliding flags), per-layer rope, KV sharing, double-wide
/// MLPs, softcapping, Per-Layer Embeddings and GELU activation.
fn gemma4_config(source: &dyn Checkpoint) -> Result<Value> {
    let m = source.metadata();
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
            // Boolean arrays read as u32 via u32_array are rejected; a plain
            // pattern integer alternates every `pattern` layers.
            m.u32(&key("attention.sliding_window_pattern"))
                .map(|p| (0..layers).map(|l| (l + 1) % p != 0).map(|b| b as u32).collect())
        })
        .ok_or_else(|| Error::Config("gemma4 missing sliding_window_pattern".into()))?;
    if sliding.len() != layers as usize {
        return Err(Error::Config("sliding_window_pattern length mismatch".into()));
    }
    let layer_types: Vec<Value> = sliding
        .iter()
        .map(|&s| json!(if s != 0 { "sliding_attention" } else { "full_attention" }))
        .collect();
    // The base FFN width: per-layer arrays report the double-wide widths too.
    let intermediate = m
        .u32_array(&key("feed_forward_length"))
        .map(|v| *v.iter().min().unwrap())
        .or_else(|| m.u32(&key("feed_forward_length")))
        .ok_or_else(|| Error::Config("gemma4 missing feed_forward_length".into()))?;
    let mut rot_full = hd_full;
    if let Ok(raw) = source.read("rope_freqs.weight") {
        let freqs = raw.data.into_f32();
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
    if let Some(d) = m.u32(&key("embedding_length_per_layer_input")) {
        if d > 0 {
            cfg["hidden_size_per_layer_input"] = json!(d);
        }
    }
    if let Some(eps) = m.f64(&key("attention.layer_norm_rms_epsilon")) {
        cfg["rms_norm_eps"] = json!(eps);
    }
    Ok(cfg)
}

/// Vocab size: explicit field, else the tokenizer token table length.
fn vocab_size(m: &flint_checkpoint::Metadata, arch: &str) -> Result<u32> {
    m.u32(&format!("{arch}.vocab_size"))
        .or_else(|| m.str_array("tokenizer.ggml.tokens").map(|t| t.len() as u32))
        .ok_or_else(|| Error::Config("GGUF has no vocab size".into()))
}

fn eos_ids(m: &flint_checkpoint::Metadata) -> Value {
    let mut eos = Vec::new();
    if let Some(id) = m.u32("tokenizer.ggml.eos_token_id") {
        eos.push(id);
    }
    json!(eos)
}

fn tied(source: &dyn Checkpoint) -> bool {
    !source.names().iter().any(|n| n == "output.weight")
}
