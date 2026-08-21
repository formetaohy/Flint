use std::collections::HashMap;
use std::path::Path;

use thuban_checkpoint::{Checkpoint, GgufWriter, MetaVal, Metadata, RawTensor};

use thuban_tokenizer::{from_metadata, from_source, load};

fn bpe_meta() -> Metadata {
    let tokens = ["a", "b", "ab", "<|im_start|>", "<|im_end|>", "<unused>"];
    let types = [0u32, 0, 0, 3, 3, 5];

    let mut kv = HashMap::new();
    kv.insert("tokenizer.ggml.model".into(), MetaVal::Str("gpt2".into()));
    kv.insert(
        "tokenizer.ggml.tokens".into(),
        MetaVal::Arr(tokens.iter().map(|t| MetaVal::Str(t.to_string())).collect()),
    );
    kv.insert(
        "tokenizer.ggml.merges".into(),
        MetaVal::Arr(vec![MetaVal::Str("a b".into())]),
    );
    kv.insert(
        "tokenizer.ggml.token_type".into(),
        MetaVal::Arr(types.iter().map(|t| MetaVal::UInt(*t as u64)).collect()),
    );
    Metadata::new(kv)
}

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
fn bpe_pre_types_split_digits_per_regex() {
    let meta = |pre: &str| {
        let mut kv = HashMap::new();
        for (key, val) in [
            ("tokenizer.ggml.model", MetaVal::Str("gpt2".into())),
            ("tokenizer.ggml.pre", MetaVal::Str(pre.into())),
            (
                "tokenizer.ggml.tokens",
                MetaVal::Arr(
                    ["1", "2", "12", "a"]
                        .iter()
                        .map(|t| MetaVal::Str(t.to_string()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges",
                MetaVal::Arr(vec![MetaVal::Str("1 2".into())]),
            ),
        ] {
            kv.insert(key.into(), val);
        }
        Metadata::new(kv)
    };

    for pre in ["qwen2", "qwen3", "qwen35"] {
        let tok = from_metadata(&meta(pre)).unwrap();
        assert_eq!(
            tok.encode("12").unwrap(),
            vec![0, 1],
            "{pre}: \\p{{N}} splits digits one by one"
        );
    }
    let tok = from_metadata(&meta("llama3")).unwrap();
    assert_eq!(
        tok.encode("12").unwrap(),
        vec![2],
        "llama3: \\p{{N}}{{1,3}} keeps digit runs together"
    );
}

#[test]
fn bpe_rebuild_keeps_ids_and_honors_merges() {
    let tok = from_metadata(&bpe_meta()).unwrap();

    assert_eq!(tok.encode("a").unwrap(), vec![0]);
    assert_eq!(tok.encode("b").unwrap(), vec![1]);
    assert_eq!(tok.encode("ab").unwrap(), vec![2]);
    assert_eq!(tok.encode("aa").unwrap(), vec![0, 0]);
}

#[test]
fn bpe_control_pieces_are_single_specials() {
    let tok = from_metadata(&bpe_meta()).unwrap();

    for marker in ["<|im_start|>", "<|im_end|>"] {
        assert_eq!(
            tok.encode(marker).unwrap(),
            vec![tok.token_id(marker).unwrap()],
            "{marker} must be one special id"
        );
    }
    assert_eq!(tok.token_id("<unused>"), None, "unused piece is dropped");
}

#[test]
fn unigram_markers_encode_as_single_ids() {
    let tok = from_metadata(&unigram_meta()).unwrap();

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
    let tok = from_metadata(&unigram_meta()).unwrap();

    let ids = tok.encode("hello world").unwrap();
    let mut state = tok.stream_decoder();
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
        from_metadata(&mutate_meta("tokenizer.ggml.tokens", None)).is_err(),
        "missing tokens"
    );
    assert!(
        from_metadata(&mutate_meta("tokenizer.ggml.scores", None)).is_err(),
        "missing scores"
    );

    let mismatch = mutate_meta(
        "",
        Some((
            "tokenizer.ggml.scores",
            MetaVal::Arr(vec![MetaVal::Float(-1.0)]),
        )),
    );
    let err = from_metadata(&mismatch)
        .err()
        .expect("length mismatch fails");
    assert!(err.to_string().contains("mismatch"), "{err}");

    assert!(
        from_metadata(&mutate_meta("tokenizer.ggml.unknown_token_id", None)).is_ok(),
        "unk id is optional"
    );
}

struct NoTokenizer {
    meta: Metadata,
}

impl NoTokenizer {
    fn new() -> Self {
        Self {
            meta: Metadata::default(),
        }
    }
}

impl Checkpoint for NoTokenizer {
    fn names(&self) -> Vec<String> {
        Vec::new()
    }
    fn read(&self, name: &str) -> thuban_error::Result<RawTensor> {
        Err(thuban_error::Error::Checkpoint(format!(
            "NoTokenizer has no tensor {name}"
        )))
    }
    fn metadata(&self) -> thuban_error::Result<&Metadata> {
        Ok(&self.meta)
    }
}

#[test]
fn from_source_rejects_checkpoints_without_tokenizer_metadata() {
    let err = from_source(&NoTokenizer::new())
        .err()
        .expect("empty metadata carries no tokenizer");
    assert!(err.to_string().contains("tokenizer.ggml.tokens"), "{err}");
}

fn gguf_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "thuban-gguf-tok-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("t")
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", "llama");
    w.kv_str("tokenizer.ggml.model", "bpe");
    w.kv_str_array("tokenizer.ggml.tokens", &["a", "b", "ab"]);
    w.kv_str_array("tokenizer.ggml.merges", &["a b"]);
    std::fs::write(dir.join("model.gguf"), w.finish()).unwrap();
    dir
}

#[test]
fn load_prefers_tokenizer_json_over_gguf_metadata() {
    let dir = gguf_dir();
    std::fs::write(
        dir.join("tokenizer.json"),
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {"a": 0, "b": 1},
                "merges": []
            }
        }"#,
    )
    .unwrap();

    let source = thuban_checkpoint::open_checkpoint(&dir).unwrap();
    let tok = load(Path::new(&dir), &source).unwrap();
    assert_eq!(
        tok.encode("ab").unwrap(),
        vec![0, 1],
        "tokenizer.json wins over the GGUF-embedded tokenizer"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_falls_back_to_gguf_without_tokenizer_json() {
    let dir = gguf_dir();
    let source = thuban_checkpoint::open_checkpoint(&dir).unwrap();
    let tok = load(Path::new(&dir), &source).unwrap();
    assert_eq!(tok.encode("ab").unwrap(), vec![2], "GGUF metadata path");
    std::fs::remove_dir_all(&dir).ok();
}
