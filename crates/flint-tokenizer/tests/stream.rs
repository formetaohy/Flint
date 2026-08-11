use std::path::PathBuf;

use flint_tokenizer::Tokenizer;

fn tokenizer_path() -> PathBuf {
    let thread = std::thread::current().name().unwrap_or_default().replace([':', ' '], "_");
    let dir = std::env::temp_dir().join(format!("flint-tok-stream-{}-{thread}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer.json");
    std::fs::write(
        &path,
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [
                    ["<unk>", -10.0],
                    ["hello", -1.0],
                    ["world", -2.0],
                    ["<|endoftext|>", -3.0]
                ]
            }
        }"#,
    )
    .unwrap();
    path
}

#[test]
fn from_file_encodes_and_looks_up_ids() {
    let tok = Tokenizer::from_file(&tokenizer_path()).unwrap();
    assert_eq!(tok.encode("hello world").unwrap(), vec![1, 2]);
    assert_eq!(tok.token_id("world"), Some(2));
    assert_eq!(tok.token_id("nope"), None);
}

#[test]
fn stream_decoder_roundtrips_text() {
    let tok = Tokenizer::from_file(&tokenizer_path()).unwrap();
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
