//! Format-agnostic tokenizer: text to token ids and back. Loads an HF
//! `tokenizer.json` when present, otherwise rebuilds the tokenizer embedded in
//! a GGUF checkpoint's metadata. Knows nothing about architectures or chat.

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
use tokenizers::tokenizer::{AddedToken, step_decode_stream};

use flint_checkpoint::{Checkpoint, Metadata};
use flint_error::{Error, Result};

/// Thin wrapper over the HF tokenizers crate.
pub struct Tokenizer {
    inner: HfTokenizer,
}

/// Incremental decode state: buffers tokens until they form valid text.
pub struct Decoder {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl Tokenizer {
    /// Loads the tokenizer for a model directory plus its already-opened
    /// checkpoint: `tokenizer.json` when present, otherwise the embedded GGUF
    /// tokenizer.
    pub fn load(model_dir: &Path, source: &dyn Checkpoint) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        if path.exists() {
            return Self::from_file(&path);
        }
        Self::from_source(source)
    }

    /// Loads an HF `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// Rebuilds the tokenizer embedded in a self-contained checkpoint.
    pub fn from_source(source: &dyn Checkpoint) -> Result<Self> {
        if source.kind() != "gguf" {
            return Err(Error::Tokenizer(format!(
                "{} checkpoint carries no tokenizer metadata",
                source.kind()
            )));
        }
        Self::from_gguf(source.metadata())
    }

    /// Rebuilds a tokenizer from GGUF metadata, dispatching on the
    /// SentencePiece model kind: `llama` is Unigram, anything else BPE.
    /// Special tokens keep their original ids in both paths.
    pub fn from_gguf(meta: &Metadata) -> Result<Self> {
        match meta.str("tokenizer.ggml.model") {
            Some("llama") => Self::from_gguf_unigram(meta),
            _ => Self::from_gguf_bpe(meta),
        }
    }

    /// Rebuilds a byte-level BPE tokenizer. Special tokens (ggml types
    /// CONTROL/USER_DEFINED) sit contiguously at the top of the vocab, so
    /// re-adding them in order reproduces their original ids.
    fn from_gguf_bpe(meta: &Metadata) -> Result<Self> {
        let tokens = meta
            .str_array("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::Tokenizer("GGUF missing tokenizer.ggml.tokens".into()))?;
        let types = meta
            .u32_array("tokenizer.ggml.token_type")
            .unwrap_or_default();
        let merges = meta.str_array("tokenizer.ggml.merges").unwrap_or_default();

        let is_added = |i: usize| matches!(types.get(i), Some(3) | Some(4));
        let is_unused = |i: usize| matches!(types.get(i), Some(5));

        // Base vocab keeps each non-special, non-unused token at its original
        // id; unused tokens are dropped so the added special tokens land at
        // the top with their true ids. The unknown token stays: the BPE
        // builder rejects an unk not in the vocab.
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
            // Some GGUFs ship merges whose parts or results are missing from
            // the vocab (a mojibake'd replacement char in phi4 GGUFs): the
            // BPE build is fail-fast, so drop them up front.
            .filter(|(a, b)| {
                let res = format!("{a}{b}");
                vocab.contains_key(a) && vocab.contains_key(b) && vocab.contains_key(&res)
            })
            .collect();

        let mut bpe = BPE::builder().vocab_and_merges(vocab, merges);
        if let Some(id) = unk_id {
            bpe = bpe.unk_token(tokens[id].to_string());
        }
        let bpe = bpe
            .build()
            .map_err(|e| Error::Tokenizer(format!("BPE build: {e}")))?;
        let mut inner = HfTokenizer::new(bpe);
        inner.with_pre_tokenizer(Some(ByteLevelPre::default()));
        inner.with_decoder(Some(ByteLevelDecoder::default()));

        let added: Vec<AddedToken> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| is_added(*i))
            .map(|(_, t)| AddedToken::from(t.to_string(), true))
            .collect();
        let _ = inner.add_special_tokens(added);
        Ok(Self { inner })
    }

    /// Rebuilds a SentencePiece Unigram tokenizer. The vocab keeps every piece at
    /// its original id; byte tokens drive byte-fallback decoding.
    fn from_gguf_unigram(meta: &Metadata) -> Result<Self> {
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

        // SentencePiece preprocessing: prepend ▁, fold spaces to ▁.
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

        // Restore control / user-defined pieces (types 3/4) as special tokens so
        // turn markers encode to single ids.
        let types = meta
            .u32_array("tokenizer.ggml.token_type")
            .unwrap_or_default();
        let added: Vec<AddedToken> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(types.get(*i), Some(3) | Some(4)))
            .map(|(_, t)| AddedToken::from(t.to_string(), true))
            .collect();
        let _ = inner.add_special_tokens(added);
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Resolves a special token literal (e.g. "im_end") to its id.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn decoder(&self) -> Decoder {
        Decoder {
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// Feeds one committed token; returns the newly printable text, if any.
    pub fn step_decode(&self, st: &mut Decoder, id: u32) -> Result<Option<String>> {
        step_decode_stream(
            &self.inner,
            vec![id],
            true,
            &mut st.ids,
            &mut st.prefix,
            &mut st.prefix_index,
        )
        .map_err(|e| Error::Tokenizer(e.to_string()))
    }
}
