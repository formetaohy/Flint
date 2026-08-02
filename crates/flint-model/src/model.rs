use flint_backend::Backend;
use flint_error::Result;

/// Readouts of one forward chunk: logits and final hidden states for the
/// requested rows, in request order.
#[derive(Debug)]
pub struct ChunkOut {
    pub logits: Vec<Vec<f32>>,
    pub hidden: Vec<Vec<f32>>,
}

/// A runnable text model. Forward consumes 1..=ROWS tokens per call and
/// advances the internal position; decode is a one-token chunk.
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

    /// The speculative draft head, when the architecture ships one.
    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        None
    }
}

/// A draft head shadowing the target model one token ahead, plus the
/// rollback machinery speculative verification needs. Every method is
/// mandatory: an architecture that offers a speculator implements the full
/// protocol — there are no no-op defaults to silently corrupt state.
pub trait Speculator {
    /// Consumes the most recently committed token plus the target hidden
    /// state at its position, advances internal position and caches, and
    /// returns draft logits for the following position. Callers use the
    /// logits when they need a draft token and discard them when they only
    /// need to advance (the target already supplied the next token).
    fn draft(&mut self, backend: &mut Backend, token: u32, hidden: &[f32]) -> Result<Vec<f32>>;

    /// Aligns the draft head with the target position after prefill.
    fn prime(&mut self);

    /// Captures every piece of generation state a verification chunk will
    /// mutate, so a rejected chunk can be rolled back. Position-addressed
    /// caches need no snapshot: replaying overwrites the same slots.
    fn snapshot(&mut self, backend: &Backend);

    /// Restores the state captured by the last `snapshot`.
    fn restore(&mut self, backend: &Backend);
}
