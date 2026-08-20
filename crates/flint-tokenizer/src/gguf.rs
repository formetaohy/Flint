use std::path::Path;

use ahash::AHashMap;
use tokenizers::Tokenizer as HfTokenizer;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::normalizers::prepend::Prepend;
use tokenizers::normalizers::replace::Replace;
use tokenizers::normalizers::utils::Sequence as NormalizerSequence;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::sequence::Sequence as PreTokenizers;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::AddedToken;
use tokenizers::tokenizer::SplitDelimiterBehavior;

use flint_checkpoint::{Checkpoint, CheckpointKind, Metadata};
use flint_error::{Error, Result};

use crate::Tokenizer;

pub fn load(model_dir: &Path, source: &dyn Checkpoint) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    if path.exists() {
        return Tokenizer::from_file(&path);
    }
    from_source(source)
}

pub fn from_source(source: &dyn Checkpoint) -> Result<Tokenizer> {
    if source.kind() != CheckpointKind::Gguf {
        return Err(Error::Tokenizer(
            "checkpoint carries no tokenizer metadata".into(),
        ));
    }
    from_metadata(source.metadata()?)
}

pub fn from_metadata(meta: &Metadata) -> Result<Tokenizer> {
    match meta.str("tokenizer.ggml.model") {
        Some("llama") => from_gguf_unigram(meta),
        _ => from_gguf_bpe(meta),
    }
}

const LLAMA3_SPLIT: &str = r"(?:\'[sS]|\'[tT]|\'[rR][eE]|\'[vV][eE]|\'[mM]|\'[lL][lL]|\'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn from_gguf_bpe(meta: &Metadata) -> Result<Tokenizer> {
    let tokens = meta
        .str_array("tokenizer.ggml.tokens")
        .ok_or_else(|| Error::Tokenizer("GGUF missing tokenizer.ggml.tokens".into()))?;
    let types = meta
        .u32_array("tokenizer.ggml.token_type")
        .unwrap_or_default();
    let merges = meta.str_array("tokenizer.ggml.merges").unwrap_or_default();

    let is_added = |i: usize| matches!(types.get(i), Some(3) | Some(4));
    let is_unused = |i: usize| matches!(types.get(i), Some(5));

    let unk_id = meta
        .u32("tokenizer.ggml.unknown_token_id")
        .map(|i| i as usize);
    let mut vocab = AHashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        if (!is_added(i) && !is_unused(i)) || Some(i) == unk_id {
            vocab.insert(t.to_string(), i as u32);
        }
    }
    let merges: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .filter(|(a, b)| {
            let res = format!("{a}{b}");
            vocab.contains_key(a) && vocab.contains_key(b) && vocab.contains_key(&res)
        })
        .collect();

    let mut bpe = BPE::builder().vocab_and_merges(vocab, merges);
    if let Some(id) = unk_id {
        bpe = bpe.unk_token(tokens[id].to_string());
    }
    let llama3 = matches!(
        meta.str("tokenizer.ggml.pre"),
        Some("llama-bpe" | "llama3" | "llama-v3" | "falcon3" | "pixtral")
    );
    let bpe = bpe
        .build()
        .map_err(|e| Error::Tokenizer(format!("BPE build: {e}")))?;
    let mut inner = HfTokenizer::new(bpe);
    if llama3 {
        inner.with_pre_tokenizer(Some(PreTokenizers::new(vec![
            Split::new(
                SplitPattern::Regex(LLAMA3_SPLIT.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| Error::Tokenizer(format!("split pre-tokenizer: {e}")))?
            .into(),
            ByteLevelPre::default()
                .add_prefix_space(false)
                .use_regex(false)
                .into(),
        ])));
    } else {
        inner.with_pre_tokenizer(Some(ByteLevelPre::default().add_prefix_space(false)));
    }
    inner.with_decoder(Some(ByteLevelDecoder::default()));

    let added: Vec<AddedToken> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| is_added(*i))
        .map(|(_, t)| AddedToken::from(t.to_string(), false))
        .collect();
    inner
        .add_special_tokens(added)
        .map_err(|e| Error::Tokenizer(format!("add special tokens: {e}")))?;
    Ok(Tokenizer::from_hf(inner))
}

fn from_gguf_unigram(meta: &Metadata) -> Result<Tokenizer> {
    let tokens = meta
        .str_array("tokenizer.ggml.tokens")
        .ok_or_else(|| Error::Tokenizer("GGUF missing tokenizer.ggml.tokens".into()))?;
    let scores = meta
        .f64_array("tokenizer.ggml.scores")
        .ok_or_else(|| Error::Tokenizer("GGUF missing tokenizer.ggml.scores".into()))?;
    if tokens.len() != scores.len() {
        return Err(Error::Tokenizer(
            "GGUF tokens/scores length mismatch".into(),
        ));
    }

    let vocab: Vec<(String, f64)> = tokens
        .iter()
        .zip(&scores)
        .map(|(t, s)| (t.to_string(), *s))
        .collect();
    let unk_id = meta
        .u32("tokenizer.ggml.unknown_token_id")
        .map(|i| i as usize);
    let byte_fallback = tokens.contains(&"<0x00>");

    let unigram = Unigram::from(vocab, unk_id, byte_fallback)
        .map_err(|e| Error::Tokenizer(format!("Unigram build: {e}")))?;
    let mut inner = HfTokenizer::new(unigram);

    let prepend = NormalizerWrapper::Prepend(Prepend::new("▁".to_string()));
    let replace = NormalizerWrapper::Replace(
        Replace::new(" ", "▁").map_err(|e| Error::Tokenizer(format!("normalizer: {e}")))?,
    );
    inner
        .with_normalizer(Some(NormalizerWrapper::Sequence(NormalizerSequence::new(
            vec![prepend, replace],
        ))))
        .map_err(|e| Error::Tokenizer(format!("normalizer: {e}")))?;
    inner.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::First, false)));
    if byte_fallback {
        inner.with_decoder(Some(DecoderWrapper::Sequence(DecoderSequence::new(vec![
            DecoderWrapper::ByteFallback(ByteFallback::new()),
            DecoderWrapper::Metaspace(MetaspaceDecoder::new('▁', PrependScheme::First, false)),
        ]))));
    } else {
        inner.with_decoder(Some(DecoderWrapper::Metaspace(MetaspaceDecoder::new(
            '▁',
            PrependScheme::First,
            false,
        ))));
    }

    let types = meta
        .u32_array("tokenizer.ggml.token_type")
        .unwrap_or_default();
    let added: Vec<AddedToken> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| matches!(types.get(*i), Some(3) | Some(4)))
        .map(|(_, t)| AddedToken::from(t.to_string(), false))
        .collect();
    inner
        .add_special_tokens(added)
        .map_err(|e| Error::Tokenizer(format!("add special tokens: {e}")))?;
    Ok(Tokenizer::from_hf(inner))
}
