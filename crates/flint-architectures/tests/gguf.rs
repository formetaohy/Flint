use std::collections::HashMap;

use flint_architectures::keymap::gguf_key;
use flint_checkpoint::{Checkpoint, MetaVal, Metadata, RawTensor};
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
    fn metadata(&self) -> flint_error::Result<&Metadata> {
        Ok(&self.meta)
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
    let cfg =
        flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Llama)
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
    let cfg =
        flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Gemma)
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
fn qwen35_synthesis_reads_interval_layout() {
    let src = FakeSource::new(
        "qwen35",
        vec![
            ("qwen35.embedding_length", MetaVal::UInt(1024)),
            ("qwen35.feed_forward_length", MetaVal::UInt(3584)),
            ("qwen35.block_count", MetaVal::UInt(5)),
            ("qwen35.attention.head_count", MetaVal::UInt(8)),
            ("qwen35.attention.head_count_kv", MetaVal::UInt(2)),
            ("qwen35.attention.key_length", MetaVal::UInt(256)),
            ("qwen35.attention.value_length", MetaVal::UInt(256)),
            ("qwen35.rope.dimension_count", MetaVal::UInt(64)),
            ("qwen35.rope.freq_base", MetaVal::Float(10_000_000.0)),
            ("qwen35.full_attention_interval", MetaVal::UInt(4)),
            ("qwen35.ssm.conv_kernel", MetaVal::UInt(4)),
            ("qwen35.ssm.inner_size", MetaVal::UInt(2048)),
            ("qwen35.ssm.state_size", MetaVal::UInt(128)),
            ("qwen35.ssm.group_count", MetaVal::UInt(16)),
            ("qwen35.ssm.time_step_rank", MetaVal::UInt(16)),
            (
                "tokenizer.ggml.tokens",
                arr_str(&["<pad>", "<eos>", "a"]),
            ),
            ("tokenizer.ggml.eos_token_id", MetaVal::UInt(1)),
        ],
        &["token_embd.weight"],
    );
    let cfg = flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Qwen35)
        .unwrap();
    assert_eq!(cfg["hidden_size"], json!(1024));
    assert_eq!(cfg["head_dim"], json!(256));
    assert_eq!(cfg["num_hidden_layers"], json!(5));
    assert_eq!(cfg["rotary_dim"], json!(64));
    assert_eq!(cfg["rope_theta"], json!(10_000_000.0));
    assert_eq!(cfg["linear_key_head_dim"], json!(128));
    assert_eq!(cfg["linear_value_head_dim"], json!(128));
    assert_eq!(cfg["linear_conv_kernel_dim"], json!(4));
    assert_eq!(cfg["vocab_size"], json!(3));
    assert_eq!(cfg["eos_token_id"], json!([1]));
    assert_eq!(
        cfg["layer_types"],
        json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention"
        ])
    );
    assert_eq!(cfg["tie_word_embeddings"], json!(true));
    assert_eq!(cfg["rms_norm_eps"], json!(1e-6));
}

#[test]
fn qwen35_synthesis_reads_recurrent_layers_and_trims_nextn() {
    let src = FakeSource::new(
        "qwen35",
        vec![
            ("qwen35.embedding_length", MetaVal::UInt(64)),
            ("qwen35.feed_forward_length", MetaVal::UInt(128)),
            ("qwen35.block_count", MetaVal::UInt(4)),
            ("qwen35.attention.head_count", MetaVal::UInt(4)),
            ("qwen35.attention.head_count_kv", MetaVal::UInt(2)),
            ("qwen35.attention.key_length", MetaVal::UInt(64)),
            ("qwen35.rope.dimension_count", MetaVal::UInt(32)),
            ("qwen35.nextn_predict_layers", MetaVal::UInt(1)),
            (
                "qwen35.attention.recurrent_layers",
                MetaVal::Arr(vec![
                    MetaVal::Bool(true),
                    MetaVal::Bool(false),
                    MetaVal::Bool(true),
                    MetaVal::Bool(false),
                ]),
            ),
            ("qwen35.ssm.inner_size", MetaVal::UInt(512)),
            ("qwen35.ssm.state_size", MetaVal::UInt(32)),
            ("qwen35.ssm.group_count", MetaVal::UInt(8)),
            ("qwen35.ssm.time_step_rank", MetaVal::UInt(16)),
            (
                "tokenizer.ggml.tokens",
                arr_str(&["<eos>", "a", "b"]),
            ),
            ("tokenizer.ggml.eos_token_id", MetaVal::UInt(0)),
        ],
        &[],
    );
    let cfg = flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Qwen35)
        .unwrap();
    assert_eq!(
        cfg["layer_types"],
        json!(["linear_attention", "full_attention", "linear_attention"]),
        "the nextn layer is trimmed from the trunk"
    );
}

#[test]
fn qwen35_synthesis_fails_fast() {
    let base = vec![
        ("qwen35.embedding_length", MetaVal::UInt(64)),
        ("qwen35.feed_forward_length", MetaVal::UInt(128)),
        ("qwen35.block_count", MetaVal::UInt(2)),
        ("qwen35.attention.head_count", MetaVal::UInt(4)),
        ("qwen35.attention.head_count_kv", MetaVal::UInt(2)),
        ("qwen35.attention.key_length", MetaVal::UInt(64)),
        ("qwen35.rope.dimension_count", MetaVal::UInt(32)),
        ("qwen35.ssm.inner_size", MetaVal::UInt(512)),
        ("qwen35.ssm.state_size", MetaVal::UInt(32)),
        ("qwen35.ssm.group_count", MetaVal::UInt(8)),
        ("qwen35.ssm.time_step_rank", MetaVal::UInt(16)),
    ];

    let src = FakeSource::new("qwen35", base.clone(), &[]);
    assert!(
        flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Qwen35)
            .is_err(),
        "missing recurrent layout"
    );

    let mut with_nextn = base.clone();
    with_nextn.push(("qwen35.nextn_predict_layers", MetaVal::UInt(2)));
    with_nextn.push(("qwen35.full_attention_interval", MetaVal::UInt(4)));
    let src = FakeSource::new("qwen35", with_nextn, &[]);
    assert!(
        flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Qwen35)
            .is_err(),
        "nextn exceeds the block count"
    );

    let mut bad_inner = base.clone();
    bad_inner.push(("qwen35.full_attention_interval", MetaVal::UInt(4)));
    bad_inner.push(("qwen35.ssm.inner_size", MetaVal::UInt(511)));
    let src = FakeSource::new("qwen35", bad_inner, &[]);
    assert!(
        flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Qwen35)
            .is_err(),
        "inner_size not divisible by the value heads"
    );

    let mut asym = base.clone();
    asym.push(("qwen35.full_attention_interval", MetaVal::UInt(4)));
    asym.push(("qwen35.attention.value_length", MetaVal::UInt(128)));
    let src = FakeSource::new("qwen35", asym, &[]);
    assert!(
        flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Qwen35)
            .is_err(),
        "asymmetric head dims"
    );
}

#[test]
fn synthesis_fails_fast() {
    let src = FakeSource::new("llama", vec![], &[]);
    assert!(
        flint_architectures::gguf::synthesize_config(&src, flint_architectures::Family::Llama)
            .is_err(),
        "missing embedding_length"
    );
}
