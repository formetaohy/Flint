use std::collections::HashMap;

use flint_architectures::keys::gguf_key;
use flint_checkpoint::{Checkpoint, CheckpointKind, MetaVal, Metadata, RawTensor};
use serde_json::json;

#[test]
fn name_mapping_covers_the_dense_family() {
    let cases = &[
        ("token_embd.weight", Some("embed_tokens.weight")),
        ("output.weight", Some("lm_head.weight")),
        ("output_norm.weight", Some("norm.weight")),
        (
            "blk.0.attn_norm.weight",
            Some("layers.0.input_layernorm.weight"),
        ),
        (
            "blk.3.attn_q.weight",
            Some("layers.3.self_attn.q_proj.weight"),
        ),
        (
            "blk.3.attn_k.weight",
            Some("layers.3.self_attn.k_proj.weight"),
        ),
        (
            "blk.3.attn_v.weight",
            Some("layers.3.self_attn.v_proj.weight"),
        ),
        (
            "blk.3.attn_output.weight",
            Some("layers.3.self_attn.o_proj.weight"),
        ),
        (
            "blk.3.attn_q_norm.weight",
            Some("layers.3.self_attn.q_norm.weight"),
        ),
        (
            "blk.3.ffn_norm.weight",
            Some("layers.3.post_attention_layernorm.weight"),
        ),
        (
            "blk.3.ffn_gate.weight",
            Some("layers.3.mlp.gate_proj.weight"),
        ),
        ("blk.3.ffn_up.weight", Some("layers.3.mlp.up_proj.weight")),
        (
            "blk.3.ffn_down.weight",
            Some("layers.3.mlp.down_proj.weight"),
        ),
        (
            "blk.3.post_attention_norm.weight",
            Some("layers.3.post_attention_norm.weight"),
        ),
        (
            "blk.3.post_ffw_norm.weight",
            Some("layers.3.post_ffw_norm.weight"),
        ),
    ];
    for &(name, want) in cases {
        assert_eq!(gguf_key(name).as_deref(), want, "{name}");
    }
    assert_eq!(
        gguf_key("blk.0.attn_q.bias"),
        Some("layers.0.self_attn.q_proj.bias".into())
    );
    assert_eq!(
        gguf_key("tokenizer.ggml.tokens"),
        None,
        "non-tensor names skip"
    );
    assert_eq!(gguf_key("blk.notanumber.attn_q.weight"), None);
    assert_eq!(gguf_key("blk.0.unknown_stem.weight"), None);
}

struct FakeSource {
    meta: Metadata,
    tensor_names: Vec<String>,
}

impl FakeSource {
    fn new(arch: &str, kv: Vec<(&str, MetaVal)>, names: &[&str]) -> Self {
        let mut map = HashMap::from([(
            "general.architecture".to_string(),
            MetaVal::Str(arch.into()),
        )]);
        map.extend(kv.into_iter().map(|(k, v)| (k.to_string(), v)));
        Self {
            meta: Metadata::new(map),
            tensor_names: names.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Checkpoint for FakeSource {
    fn names(&self) -> Vec<String> {
        self.tensor_names.clone()
    }
    fn read(&self, name: &str) -> flint_error::Result<RawTensor> {
        unreachable!("synthesize never reads tensor bytes ({name})")
    }
    fn metadata(&self) -> &Metadata {
        &self.meta
    }
    fn config_json(&self) -> flint_error::Result<Option<serde_json::Value>> {
        Ok(None)
    }
    fn kind(&self) -> CheckpointKind {
        CheckpointKind::Gguf
    }
}

fn arr_str(items: &[&str]) -> MetaVal {
    MetaVal::Arr(items.iter().map(|s| MetaVal::Str(s.to_string())).collect())
}

#[test]
fn llama_synthesis_reads_metadata_and_tensor_names() {
    let src = FakeSource::new(
        "qwen2",
        vec![
            ("qwen2.embedding_length", MetaVal::UInt(960)),
            ("qwen2.attention.head_count", MetaVal::UInt(15)),
            ("qwen2.attention.head_count_kv", MetaVal::UInt(5)),
            ("qwen2.attention.key_length", MetaVal::UInt(64)),
            ("qwen2.feed_forward_length", MetaVal::UInt(2560)),
            ("qwen2.block_count", MetaVal::UInt(32)),
            ("qwen2.rope.freq_base", MetaVal::Float(10000.0)),
            ("tokenizer.ggml.tokens", arr_str(&["a", "b", "c"])),
            ("tokenizer.ggml.eos_token_id", MetaVal::UInt(2)),
        ],
        &[
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.bias",
            "blk.0.attn_q_norm.weight",
        ],
    );
    let cfg = flint_architectures::gguf_config::synthesize_config(
        &src,
        flint_architectures::Family::Llama,
    )
    .unwrap();
    assert_eq!(
        cfg,
        json!({
            "model_type": "llama",
            "hidden_size": 960,
            "num_attention_heads": 15,
            "num_key_value_heads": 5,
            "head_dim": 64,
            "intermediate_size": 2560,
            "num_hidden_layers": 32,
            "vocab_size": 3,
            "rope_theta": 10000.0,
            "eos_token_id": [2],
            "tie_word_embeddings": false,
            "attention_bias": true,
            "qk_norm": true,
        })
    );
}

#[test]
fn gemma_synthesis_adds_end_of_turn_to_eos() {
    let src = FakeSource::new(
        "gemma3",
        vec![
            ("gemma3.embedding_length", MetaVal::UInt(1152)),
            ("gemma3.attention.head_count", MetaVal::UInt(4)),
            ("gemma3.feed_forward_length", MetaVal::UInt(6912)),
            ("gemma3.block_count", MetaVal::UInt(26)),
            ("gemma3.attention.sliding_window", MetaVal::UInt(512)),
            ("gemma3.attention.sliding_window_pattern", MetaVal::UInt(6)),
            (
                "tokenizer.ggml.tokens",
                arr_str(&["<bos>", "<end_of_turn>", "x"]),
            ),
            ("tokenizer.ggml.eos_token_id", MetaVal::UInt(1)),
        ],
        &["token_embd.weight"],
    );
    let cfg = flint_architectures::gguf_config::synthesize_config(
        &src,
        flint_architectures::Family::Gemma,
    )
    .unwrap();
    assert_eq!(cfg["model_type"], json!("gemma3"));
    assert_eq!(cfg["hidden_size"], json!(1152));
    assert_eq!(cfg["head_dim"], json!(288), "derived as hidden / heads");
    assert_eq!(
        cfg["num_key_value_heads"],
        json!(4),
        "defaults to head_count"
    );
    assert_eq!(cfg["sliding_window"], json!(512));
    assert_eq!(cfg["tie_word_embeddings"], json!(true));
    assert_eq!(
        cfg["eos_token_id"],
        json!([1]),
        "<end_of_turn> duplicates the eos id once"
    );
}

#[test]
fn synthesis_fails_fast() {
    let src = FakeSource::new("llama", vec![], &[]);
    assert!(
        flint_architectures::gguf_config::synthesize_config(
            &src,
            flint_architectures::Family::Llama
        )
        .is_err(),
        "missing embedding_length"
    );
    assert!(
        flint_architectures::gguf_config::synthesize_config(
            &src,
            flint_architectures::Family::Qwen35
        )
        .is_err(),
        "Qwen3.5 has no GGUF form"
    );
}
