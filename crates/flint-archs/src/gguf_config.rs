//! Synthesizes a HF-shaped config `Value` from GGUF metadata and maps GGUF
//! tensor names onto the canonical keys the forward graphs read. GGUF tensor
//! names are architecture-generic (`blk.N.attn_q.weight`), so one map serves the
//! whole dense-GQA family.

use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use serde_json::{Value, json};

use crate::ArchKind;

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
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let layer: u32 = idx.parse().ok()?;
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
        _ => return None,
    };
    Some(format!("layers.{layer}.{canon}.{suffix}"))
}

/// Builds the config the target architecture's parser expects.
pub fn synthesize_config(source: &dyn Checkpoint, kind: ArchKind) -> Result<Value> {
    match kind {
        ArchKind::Llama => llama_config(source),
        ArchKind::Gemma => gemma_config(source),
        ArchKind::Qwen35 => Err(Error::Config(
            "Qwen3.5 ships no GGUF (hybrid Gated DeltaNet is not a ggml architecture)".into(),
        )),
    }
}

fn llama_config(source: &dyn Checkpoint) -> Result<Value> {
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
    // Q/K/V biases: present in Qwen2 small models and Phi.
    let bias = source.names().iter().any(|n| n == "blk.0.attn_q.bias");
    // QK-norm: present in Qwen3 (per-head RMSNorm weights on Q and K).
    let qk_norm = source
        .names()
        .iter()
        .any(|n| n == "blk.0.attn_q_norm.weight");

    Ok(json!({
        "model_type": "llama",
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
        "attention_bias": bias,
        "qk_norm": qk_norm,
    }))
}

/// Synthesizes a Gemma 3 text config. Gemma layers apply sandwich norms
/// (post_attention_norm / post_ffw_norm) and alternate local sliding-window
/// attention with global attention every `sliding_window_pattern` layers.
fn gemma_config(source: &dyn Checkpoint) -> Result<Value> {
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
    let sliding_window = m.u32(&key("attention.sliding_window")).unwrap_or(0);
    let sliding_window_pattern = m.u32(&key("attention.sliding_window_pattern")).unwrap_or(6);

    let vocab = m
        .u32(&key("vocab_size"))
        .or_else(|| m.str_array("tokenizer.ggml.tokens").map(|t| t.len() as u32))
        .ok_or_else(|| Error::Config("GGUF has no vocab size".into()))?;

    let mut eos = Vec::new();
    if let Some(id) = m.u32("tokenizer.ggml.eos_token_id") {
        eos.push(id);
    }
    // Gemma also terminates on the <end_of_turn> marker when chatting.
    if let Some(id) = m.str_array("tokenizer.ggml.tokens").and_then(|t| {
        t.iter()
            .position(|s| *s == "<end_of_turn>")
            .map(|i| i as u32)
    }) && !eos.contains(&id)
    {
        eos.push(id);
    }
    let tied = !source.names().iter().any(|n| n == "output.weight");

    Ok(json!({
        "model_type": "gemma3",
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "rope_theta": rope_theta,
        "sliding_window": sliding_window,
        "sliding_window_pattern": sliding_window_pattern,
        "eos_token_id": eos,
        "tie_word_embeddings": tied,
    }))
}
