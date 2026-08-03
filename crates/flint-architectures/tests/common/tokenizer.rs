//! Deterministic toy tokenizer for the test suite. One BPE vocab serves both
//! GGUF metadata and a HF `tokenizer.json`, so a model loads its tokenizer
//! identically from either source. Special tokens own the fixed trailing ids;
//! `<|endoftext|>` is always the first special.
//!
//! The vocab is plain tokens (ascii letters, digits, punctuation, short
//! words) plus the merge results, because the tokenizers BPE builder requires
//! every merge's output token to exist in the vocab.

use flint_error::{Error, Result};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
use tokenizers::tokenizer::{AddedToken, Tokenizer as HfTokenizer};

/// The BPE merge rules: `(left, right)` pairs over the plain single-char
/// tokens; every output token is added to the vocab by [`merge_results`].
const MERGES: &[(&str, &str)] = &[
    ("h", "e"),
    ("e", "l"),
    ("l", "l"),
    ("l", "o"),
    ("w", "o"),
    ("o", "r"),
    ("r", "l"),
    ("l", "d"),
    ("t", "h"),
    ("c", "o"),
    ("o", "u"),
    ("u", "n"),
    ("n", "t"),
    ("f", "r"),
    ("r", "o"),
    ("o", "m"),
];

/// The tokens produced by [`MERGES`], in the same order.
fn merge_results() -> Vec<String> {
    MERGES.iter().map(|(a, b)| format!("{a}{b}")).collect()
}

/// Plain tokens before the special block: 74 base tokens plus the 16 merge
/// outputs (90).
pub const PLAIN: usize = 90;
/// Total vocab size: plain + 6 specials (a multiple of 16 for gemm dims).
pub const VOCAB: usize = 96;
/// Id of `<|endoftext|>`, the first special.
pub const EOS_ID: u32 = PLAIN as u32;

/// Special tokens in id order (they occupy ids PLAIN..VOCAB).
fn specials() -> Vec<String> {
    [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<end_of_turn>",
    ]
    .map(String::from)
    .to_vec()
}

/// The base plain tokens: ascii letters, digits, a few punctuation marks and
/// short words, so round-trip tests can encode real text.
fn base_tokens() -> Vec<String> {
    let mut v = Vec::new();
    for c in 'a'..='z' {
        v.push(c.to_string());
    }
    for c in 'A'..='Z' {
        v.push(c.to_string());
    }
    for c in '0'..='9' {
        v.push(c.to_string());
    }
    v.extend([" ".into(), ",".into(), ".".into()]);
    for w in [
        "hello", "world", "ok", "the", "is", "count", "from", "to", "yes",
    ] {
        v.push(w.to_string());
    }
    debug_assert_eq!(v.len(), 74);
    v
}

/// All plain tokens: base block then merge outputs.
fn plain_tokens() -> Vec<String> {
    let mut v = base_tokens();
    v.extend(merge_results());
    debug_assert_eq!(v.len(), PLAIN);
    v
}

/// All tokens in id order: plain block then specials.
pub fn tokens() -> Vec<String> {
    let mut t = plain_tokens();
    t.extend(specials());
    t
}

/// GGUF token types: plain NORMAL (0), specials CONTROL (3) so the GGUF
/// rebuild path restores them as single-id special tokens.
pub fn token_types() -> Vec<u32> {
    let mut t = vec![0u32; PLAIN];
    t.extend(vec![3u32; VOCAB - PLAIN]);
    t
}

/// BPE merge pairs as `"a b"` strings, GGUF convention.
pub fn merges() -> Vec<String> {
    MERGES.iter().map(|(a, b)| format!("{a} {b}")).collect()
}

/// Serializes the toy BPE tokenizer as a HF `tokenizer.json`.
pub fn tokenizer_json() -> Result<Vec<u8>> {
    let vocab: ahash::AHashMap<String, u32> = tokens()
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i as u32))
        .collect();
    let merges: Vec<(String, String)> = MERGES
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| Error::Tokenizer(format!("toy BPE build: {e}")))?;
    let mut tok = HfTokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevelPre::default()));
    tok.with_decoder(Some(ByteLevelDecoder::default()));
    let added: Vec<AddedToken> = specials()
        .into_iter()
        .map(|t| AddedToken::from(t, true))
        .collect();
    tok.add_special_tokens(added)
        .map_err(|e| Error::Tokenizer(format!("toy specials: {e}")))?;
    serde_json::to_vec(&tok).map_err(|e| Error::Tokenizer(format!("serialize tokenizer: {e}")))
}
