use flint_backend::Backend;
use flint_error::Result;

#[derive(Debug)]
pub struct ChunkOut {
    pub logits: Vec<Vec<f32>>,
    pub hidden: Vec<Vec<f32>>,
}

pub trait LanguageModel {
    fn forward(
        &mut self,
        backend: &mut Backend,
        tokens: &[u32],
        logit_rows: &[u32],
        hidden_rows: &[u32],
    ) -> Result<ChunkOut>;

    fn reset(&mut self, backend: &Backend);
    fn pos(&self) -> u32;
    fn max_seq(&self) -> u32;
    fn vocab(&self) -> u32;
    fn eos(&self) -> &[u32];

    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        None
    }
}

pub trait Speculator {
    fn draft(&mut self, backend: &mut Backend, token: u32, hidden: &[f32]) -> Result<Vec<f32>>;

    fn prime(&mut self);

    fn snapshot(&mut self, backend: &Backend);

    fn restore(&mut self, backend: &Backend);
}
