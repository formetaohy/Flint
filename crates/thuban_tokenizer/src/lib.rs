use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;
use tokenizers::tokenizer::step_decode_stream;

use thuban_error::{Error, Result};

mod gguf;

pub use gguf::{from_metadata, from_source, load};
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

pub struct StreamDecoder {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    pub fn from_hf(inner: HfTokenizer) -> Self {
        Self { inner }
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

    pub fn vocab_size(&self) -> u32 {
        self.inner.get_vocab_size(true) as u32
    }

    pub fn decode_id(&self, id: u32) -> Option<Vec<u8>> {
        let text = self.inner.decode(&[id], true).ok()?;
        (!text.is_empty()).then(|| text.into_bytes())
    }

    pub fn stream_decoder(&self) -> StreamDecoder {
        StreamDecoder {
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    pub fn step_decode(&self, st: &mut StreamDecoder, id: u32) -> Result<Option<String>> {
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
