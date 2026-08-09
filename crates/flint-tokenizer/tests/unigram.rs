use std::collections::HashMap;

use flint_checkpoint::{MetaVal, Metadata};
use flint_tokenizer::Tokenizer;

fn unigram_meta() -> Metadata {
    let tokens = [
        "<unk>",
        "▁",
        "▁hello",
        "▁world",
        "▁ok",
        "hello",
        "world",
        "ok",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
    ];
    let scores: Vec<f64> = (0..tokens.len()).map(|i| -(i as f64) - 1.0).collect();
    let types = [0u32, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3];

    let mut kv = HashMap::new();
    kv.insert("tokenizer.ggml.model".into(), MetaVal::Str("llama".into()));
    kv.insert(
        "tokenizer.ggml.tokens".into(),
        MetaVal::Arr(tokens.iter().map(|t| MetaVal::Str(t.to_string())).collect()),
    );
    kv.insert(
        "tokenizer.ggml.scores".into(),
        MetaVal::Arr(scores.iter().map(|s| MetaVal::Float(*s)).collect()),
    );
    kv.insert(
        "tokenizer.ggml.token_type".into(),
        MetaVal::Arr(types.iter().map(|t| MetaVal::UInt(*t as u64)).collect()),
    );
    kv.insert("tokenizer.ggml.unknown_token_id".into(), MetaVal::UInt(0));
    Metadata::new(kv)
}

#[test]
fn unigram_markers_encode_as_single_ids() {
    let tok = Tokenizer::from_gguf(&unigram_meta()).unwrap();

    let im_start = "<|im_start|>";
    assert_eq!(
        tok.encode(im_start).unwrap(),
        vec![tok.token_id(im_start).unwrap()],
        "control piece must encode to a single id"
    );
    assert_eq!(
        tok.encode("<|endoftext|>").unwrap(),
        vec![tok.token_id("<|endoftext|>").unwrap()]
    );
}

#[test]
fn unigram_streaming_roundtrip() {
    let tok = Tokenizer::from_gguf(&unigram_meta()).unwrap();

    let ids = tok.encode("hello world").unwrap();
    let mut state = tok.decoder();
    let mut text = String::new();
    for id in ids {
        text.push_str(&tok.step_decode(&mut state, id).unwrap().unwrap_or_default());
    }
    assert!(
        text.contains("hello") && text.contains("world"),
        "decoded {text:?}"
    );
}

fn mutate_meta(remove: &str, insert: Option<(&str, MetaVal)>) -> Metadata {
    let mut kv = HashMap::new();
    let src = unigram_meta();
    for key in [
        "tokenizer.ggml.model",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.unknown_token_id",
    ] {
        if key != remove
            && let Some(v) = src.get(key)
        {
            kv.insert(key.to_string(), v.clone());
        }
    }
    if let Some((k, v)) = insert {
        kv.insert(k.to_string(), v);
    }
    Metadata::new(kv)
}

#[test]
fn unigram_rejects_malformed_metadata() {
    assert!(
        Tokenizer::from_gguf(&mutate_meta("tokenizer.ggml.tokens", None)).is_err(),
        "missing tokens"
    );
    assert!(
        Tokenizer::from_gguf(&mutate_meta("tokenizer.ggml.scores", None)).is_err(),
        "missing scores"
    );

    let mismatch = mutate_meta(
        "",
        Some((
            "tokenizer.ggml.scores",
            MetaVal::Arr(vec![MetaVal::Float(-1.0)]),
        )),
    );
    let err = Tokenizer::from_gguf(&mismatch)
        .err()
        .expect("length mismatch fails");
    assert!(err.to_string().contains("mismatch"), "{err}");

    assert!(
        Tokenizer::from_gguf(&mutate_meta("tokenizer.ggml.unknown_token_id", None)).is_ok(),
        "unk id is optional"
    );
}
