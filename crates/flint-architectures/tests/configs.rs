use flint_architectures::{gemma, llama};
use serde_json::{Value, json};

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
    assert_eq!(cfg.head_dim(0), 64, "derived as hidden / heads");
    assert_eq!(cfg.eos, vec![7]);
    assert!(
        !cfg.tied && !cfg.qkv_bias && !cfg.qk_norm,
        "flags default to false"
    );
    assert!(
        !cfg.sandwich && cfg.windows.iter().all(|&w| w == 0),
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
    assert_eq!(cfg.head_dim(0), 128);
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

