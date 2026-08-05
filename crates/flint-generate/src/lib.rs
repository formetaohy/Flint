//! Architecture-agnostic generation runtime: the prefill/decode engine (with
//! optional speculative decoding) and the sampler. Generic over any
//! `LanguageModel` and `Tokenizer`; holds no model-family or chat-format
//! knowledge.

pub mod engine;
pub mod sampler;

pub use engine::{Engine, GenStats, Piece, Stream};
pub use sampler::{Dist, Sampler, SamplingParams, apply_repeat_penalty, softmax};
