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

use flint_checkpoint::{Checkpoint, CheckpointKind, Metadata};
use flint_error::{Error, Result};

pub struct Tokenizer {
    inner: HfTokenizer,
}

pub struct Decoder {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl Tokenizer {

    pub fn load(model_dir: &Path, source: &dyn Checkpoint) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        if path.exists() {
            return Self::from_file(&path);
        }
        Self::from_source(source)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    pub fn from_source(source: &dyn Checkpoint) -> Result<Self> {
        if source.kind() != CheckpointKind::Gguf {
            return Err(Error::Tokenizer(
                "checkpoint carries no tokenizer metadata".into(),
            ));
        }
        Self::from_gguf(source.metadata())
    }

    pub fn from_gguf(meta: &Metadata) -> Result<Self> {
        match meta.str("tokenizer.ggml.model") {
            Some("llama") => Self::from_gguf_unigram(meta),
            _ => Self::from_gguf_bpe(meta),
        }
    }

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
