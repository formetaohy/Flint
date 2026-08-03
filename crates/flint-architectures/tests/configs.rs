//! Architecture config validation: each parser accepts a well-formed config
//! and rejects every class of invalid input fail-fast.

use flint_architectures::{Qwen35Config, gemma, llama};
use serde_json::{Value, json};

// ---------------------------------------------------------------- Llama

fn llama_json() -> Value {
    json!({
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "vocab_size": 512,
        "rope_theta": 10000.0,
        "eos_token_id": 7,
    })
}

#[test]
fn llama_parses_with_derived_head_dim() {
    let cfg = llama::parse_config(&llama_json()).unwrap();
    assert_eq!(cfg.head_dim, 64, "derived as hidden / heads");
    assert_eq!(cfg.eos, vec![7]);
    assert!(
        !cfg.tied && !cfg.qkv_bias && !cfg.qk_norm,
        "flags default to false"
    );
    assert!(
        !cfg.sandwich && cfg.window.is_none(),
        "plain residual, global attention"
    );
    assert_eq!(cfg.embed_scale, 1.0);
}

#[test]
fn llama_parses_with_explicit_head_dim_and_flags() {
    let mut v = llama_json();
    v["head_dim"] = json!(128);
    v["tie_word_embeddings"] = json!(true);
    v["attention_bias"] = json!(true);
    v["qk_norm"] = json!(true);
    v["eos_token_id"] = json!([1, 2]);
    let cfg = llama::parse_config(&v).unwrap();
    assert_eq!(cfg.head_dim, 128);
    assert!(cfg.tied && cfg.qkv_bias && cfg.qk_norm);
    assert_eq!(cfg.eos, vec![1, 2]);
}

#[test]
fn llama_rejects_invalid_configs() {
    let mut v = llama_json();
    v["num_attention_heads"] = json!(3);
    assert!(
        llama::parse_config(&v).is_err(),
        "hidden not divisible by heads"
    );

    let mut v = llama_json();
    v["head_dim"] = json!(32);
    assert!(llama::parse_config(&v).is_err(), "head_dim below 64");

    let mut v = llama_json();
    v["num_key_value_heads"] = json!(3);
    assert!(
        llama::parse_config(&v).is_err(),
        "q heads not divisible by kv heads"
    );

    let mut v = llama_json();
    v["intermediate_size"] = json!(100);
    assert!(
        llama::parse_config(&v).is_err(),
        "gemm dim not a multiple of 16"
    );

    let mut v = llama_json();
    v.as_object_mut().unwrap().remove("vocab_size");
    assert!(llama::parse_config(&v).is_err(), "missing field");
}

// ---------------------------------------------------------------- Gemma

fn gemma_json() -> Value {
    json!({
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "intermediate_size": 512,
        "num_hidden_layers": 6,
        "vocab_size": 512,
        "rope_theta": 10000.0,
        "sliding_window": 64,
        "sliding_window_pattern": 2,
    })
}

#[test]
fn gemma_parses_and_alternates_windows() {
    let cfg = gemma::parse_config(&gemma_json()).unwrap();
    assert!(cfg.tied, "gemma defaults to tied embeddings");
    assert!(cfg.sandwich && cfg.qk_norm, "gemma norms are always on");
    assert_eq!(cfg.embed_scale, 16.0, "sqrt(hidden)");
    // Layers whose (l+1) is a multiple of the pattern attend globally.
    assert_ne!(cfg.window(0), 0, "local layer keeps the window");
    assert_eq!(cfg.window(1), 0, "global layer");
    assert_eq!(cfg.window(3), 0, "global layer");
    assert_eq!(cfg.window(2), 64);

    let mut v = gemma_json();
    v["sliding_window"] = json!(0);
    let all_global = gemma::parse_config(&v).unwrap();
    assert_eq!(
        all_global.window(0),
        0,
        "no window means every layer global"
    );
}

#[test]
fn gemma_rejects_zero_pattern() {
    let mut v = gemma_json();
    v["sliding_window_pattern"] = json!(0);
    assert!(gemma::parse_config(&v).is_err());
}

// ---------------------------------------------------------------- Qwen3.5

fn qwen35_json() -> Value {
    json!({
        "tie_word_embeddings": true,
        "text_config": {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.5 },
            "eos_token_id": 3,
        },
    })
}

#[test]
fn qwen35_parses_hybrid_layout() {
    let cfg = Qwen35Config::parse(&qwen35_json()).unwrap();
    assert_eq!(cfg.layer_types.len(), 2);
    assert_eq!(cfg.rotary_dim(), 32, "head_dim * partial factor");
    assert_eq!(cfg.key_dim(), 512);
    assert_eq!(cfg.value_dim(), 512);
    assert_eq!(cfg.conv_dim(), 1536, "2*key + value");
    assert!(cfg.tied, "tied embeddings default");
    assert!(!cfg.has_mtp, "no mtp field means no draft head");
}

#[test]
fn qwen35_parses_split_key_value_heads() {
    let mut v = qwen35_json();
    v["text_config"]["linear_num_key_heads"] = json!(8);
    let cfg = Qwen35Config::parse(&v).unwrap();
    assert_eq!(cfg.key_dim(), 256, "key_dim follows key heads");
    assert_eq!(cfg.value_dim(), 512, "value_dim follows value heads");
    assert_eq!(cfg.conv_dim(), 1024, "2*key + value");
}

#[test]
fn qwen35_parses_untied_embeddings() {
    let mut v = qwen35_json();
    v["tie_word_embeddings"] = json!(false);
    let cfg = Qwen35Config::parse(&v).unwrap();
    assert!(!cfg.tied, "untied embeddings accepted");
}

#[test]
fn qwen35_rejects_invalid_configs() {
    let mut v = qwen35_json();
    v["text_config"]["layer_types"] = json!(["linear_attention"]);
    assert!(
        Qwen35Config::parse(&v).is_err(),
        "layer_types length mismatch"
    );

    let mut v = qwen35_json();
    v["text_config"]["layer_types"] = json!(["conv", "full_attention"]);
    assert!(Qwen35Config::parse(&v).is_err(), "unknown layer type");

    let mut v = qwen35_json();
    v["text_config"]["linear_key_head_dim"] = json!(256);
    assert!(
        Qwen35Config::parse(&v).is_err(),
        "linear head dim above 128"
    );

    let mut v = qwen35_json();
    v["text_config"]["linear_num_value_heads"] = json!(24);
    assert!(
        Qwen35Config::parse(&v).is_err(),
        "value heads not divisible by key heads"
    );

    let mut v = qwen35_json();
    v["text_config"]["mtp_num_hidden_layers"] = json!(2);
    assert!(
        Qwen35Config::parse(&v).is_err(),
        "multi-layer MTP unsupported"
    );

    let mut v = qwen35_json();
    v["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(0.3);
    assert!(
        Qwen35Config::parse(&v).is_err(),
        "odd rotary dim (64*0.3 = 19)"
    );

    let mut v = qwen35_json();
    v.as_object_mut().unwrap().remove("text_config");
    assert!(Qwen35Config::parse(&v).is_err(), "missing text_config");
}
