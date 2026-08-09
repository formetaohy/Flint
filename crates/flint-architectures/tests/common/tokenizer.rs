use flint_error::{Error, Result};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
use tokenizers::tokenizer::{AddedToken, Tokenizer as HfTokenizer};

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

fn merge_results() -> Vec<String> {
    MERGES.iter().map(|(a, b)| format!("{a}{b}")).collect()
}

pub const PLAIN: usize = 90;

pub const VOCAB: usize = 96;

pub const EOS_ID: u32 = PLAIN as u32;

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

fn plain_tokens() -> Vec<String> {
    let mut v = base_tokens();
    v.extend(merge_results());
    debug_assert_eq!(v.len(), PLAIN);
    v
}

pub fn tokens() -> Vec<String> {
    let mut t = plain_tokens();
    t.extend(specials());
    t
}

pub fn token_types() -> Vec<u32> {
    let mut t = vec![0u32; PLAIN];
    t.extend(vec![3u32; VOCAB - PLAIN]);
    t
}

pub fn merges() -> Vec<String> {
    MERGES.iter().map(|(a, b)| format!("{a} {b}")).collect()
}

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
