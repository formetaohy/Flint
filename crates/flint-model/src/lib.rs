//! Architecture-agnostic model framework: the `LanguageModel` abstraction
//! plus the shared kernel dispatchers, weight loading, caches and config
//! helpers that concrete architectures (see the `flint-archs` crate) build
//! on. Checkpoint containers live in the `flint-checkpoint` crate below this
//! one. Operates on token ids and tensors only — no text, chat or tokenizer
//! knowledge.

pub mod cache;
pub mod config;
pub mod loader;
pub mod model;
pub mod ops;

pub use model::{ChunkOut, LanguageModel, Speculator};
pub use ops::ROWS;
