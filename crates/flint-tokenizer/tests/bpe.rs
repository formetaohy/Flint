//! Byte-level BPE tokenizer rebuilt from GGUF metadata (the non-"llama"
//! flavor): vocab keeps original ids, merges are honored, control pieces
//! become single-id specials, unused pieces are dropped. Also covers the
//! `Tokenizer::load` source dispatch and its fail-fast paths.

use std::collections::HashMap;
use std::path::Path;

use flint_checkpoint::{Checkpoint, GgufWriter, MetaVal, Metadata, RawTensor};
use flint_tokenizer::Tokenizer;

/// GGUF-style metadata for a tiny BPE tokenizer: three base pieces with one
/// merge, two control pieces and one unused piece.
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

#[test]
fn bpe_rebuild_keeps_ids_and_honors_merges() {
    let tok = Tokenizer::from_gguf(&bpe_meta()).unwrap();

    // Base pieces keep their original ids.
    assert_eq!(tok.encode("a").unwrap(), vec![0]);
    assert_eq!(tok.encode("b").unwrap(), vec![1]);
    // The "a b" merge collapses the pair into the pre-existing "ab" piece.
    assert_eq!(tok.encode("ab").unwrap(), vec![2]);
    // No merge for repeated a: two separate pieces.
    assert_eq!(tok.encode("aa").unwrap(), vec![0, 0]);
}

#[test]
fn bpe_control_pieces_are_single_specials() {
    let tok = Tokenizer::from_gguf(&bpe_meta()).unwrap();

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
fn load_prefers_tokenizer_json_over_gguf_metadata() {
    let dir = std::env::temp_dir().join(format!("flint-tok-load-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // A GGUF whose embedded tokenizer would map "ab" to a single id...
    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", "llama");
    w.kv_str("tokenizer.ggml.model", "bpe");
    w.kv_str_array("tokenizer.ggml.tokens", &["a", "b", "ab"]);
    w.kv_str_array("tokenizer.ggml.merges", &["a b"]);
    std::fs::write(dir.join("model.gguf"), w.finish()).unwrap();

    // ...and a tokenizer.json whose vocab splits "ab" into "a" + "b".
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

    let source = flint_checkpoint::open(&dir).unwrap();
    let tok = Tokenizer::load(&dir, source.as_ref()).unwrap();
    assert_eq!(
        tok.encode("ab").unwrap(),
        vec![0, 1],
        "tokenizer.json wins over the GGUF-embedded tokenizer"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A checkpoint source that carries no tokenizer metadata.
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
    fn read(&self, _name: &str) -> flint_error::Result<RawTensor> {
        unreachable!()
    }
    fn metadata(&self) -> &Metadata {
        &self.meta
    }
    fn config_json(&self) -> flint_error::Result<Option<serde_json::Value>> {
        Ok(None)
    }
    fn kind(&self) -> &'static str {
        "safetensors"
    }
}

#[test]
fn from_source_rejects_non_gguf_checkpoints() {
    let err = Tokenizer::from_source(&NoTokenizer::new())
        .err()
        .expect("safetensors carries no tokenizer metadata");
    assert!(err.to_string().contains("safetensors"), "{err}");
}

#[test]
fn load_falls_back_to_gguf_without_tokenizer_json() {
    let dir = std::env::temp_dir().join(format!("flint-tok-fallback-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", "llama");
    w.kv_str("tokenizer.ggml.model", "bpe");
    w.kv_str_array("tokenizer.ggml.tokens", &["a", "b", "ab"]);
    w.kv_str_array("tokenizer.ggml.merges", &["a b"]);
    std::fs::write(dir.join("model.gguf"), w.finish()).unwrap();

    let source = flint_checkpoint::open(&dir).unwrap();
    let tok = Tokenizer::load(Path::new(&dir), source.as_ref()).unwrap();
    assert_eq!(tok.encode("ab").unwrap(), vec![2], "GGUF metadata path");
    std::fs::remove_dir_all(&dir).ok();
}
