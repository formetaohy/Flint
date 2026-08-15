use flint_backend::Backend;
use flint_error::Result;

pub const MAX_M: u32 = 128;

#[derive(Debug)]
pub struct ChunkOut {
    pub logits: Vec<Vec<f32>>,
    pub hidden: Vec<Vec<f32>>,
}

pub struct SeqChunk<'a> {
    pub tokens: &'a [u32],
    pub slot: u32,
    pub logit_rows: &'a [u32],
    pub hidden_rows: &'a [u32],
}

impl SeqChunk<'_> {
    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub trait LanguageModel {
    fn forward(&mut self, backend: &mut Backend, batch: &[SeqChunk]) -> Result<Vec<ChunkOut>>;

    fn reset(&mut self, backend: &Backend, slot: u32) -> Result<()>;
    fn pos(&self, slot: u32) -> u32;
    fn slot_len(&self, slot: u32) -> u32;
    fn slot_count(&self) -> u32;
    fn vocab(&self) -> u32;
    fn eos(&self) -> &[u32];

    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        None
    }
}

pub trait Speculator {
    fn draft(
        &mut self,
        backend: &mut Backend,
        slot: u32,
        token: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>>;

    fn advance(
        &mut self,
        backend: &mut Backend,
        slot: u32,
        token: u32,
        hidden: &[f32],
    ) -> Result<()> {
        let _ = (backend, slot, token, hidden);
        Ok(())
    }

    fn prime(&mut self, slot: u32);

    fn snapshot(&mut self, backend: &Backend, slot: u32);

    fn restore(&mut self, backend: &Backend, slot: u32);
}

pub trait TextEmbedder {
    fn embed(&mut self, backend: &mut Backend, tokens: &[u32]) -> Result<Vec<f32>>;
}
