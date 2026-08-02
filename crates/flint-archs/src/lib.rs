//! Concrete model architectures, their chat prompt formats, and the
//! config/format-driven registry that assembles them into a ready-to-chat
//! bundle. Depends on the `flint-model` abstraction framework.

pub mod chat;
pub mod dense;
pub mod gemma;
pub mod gguf_config;
pub mod llama;
pub mod qwen3_5;

use std::path::Path;

use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, open};
use flint_error::{Error, Result};
use flint_model::LanguageModel;
use flint_tokenizer::Tokenizer;
use serde_json::Value;

pub use chat::{ChatFormat, ChatMl, ChatMlThink, GemmaChat};
pub use dense::{DenseConfig, DenseModel, SlidingWindow};
pub use gguf_config::{gguf_key, synthesize_config};
pub use qwen3_5::{Qwen35, Qwen35Config};

/// A fully assembled, chat-ready model: tensor engine, tokenizer, prompt format
/// and reply terminators.
pub struct ChatModel {
    pub model: Box<dyn LanguageModel>,
    pub tokenizer: Tokenizer,
    pub chat: Box<dyn ChatFormat>,
    pub stop: Vec<u32>,
}

/// Architecture families Flint can instantiate.
#[derive(Clone, Copy)]
pub enum ArchKind {
    Qwen35,
    Llama,
    Gemma,
}

impl ArchKind {
    /// Dispatches on a safetensors config's `model_type`.
    fn from_model_type(t: Option<&str>) -> Result<Self> {
        match t {
            Some("qwen3_5") => Ok(ArchKind::Qwen35),
            Some("llama" | "qwen2" | "qwen3" | "mistral") => Ok(ArchKind::Llama),
            Some("gemma" | "gemma2" | "gemma3" | "gemma3_text") => Ok(ArchKind::Gemma),
            other => Err(Error::Config(format!("unsupported model_type {other:?}"))),
        }
    }

    /// Dispatches on a GGUF `general.architecture`.
    fn from_gguf_arch(a: &str) -> Result<Self> {
        match a {
            "llama" | "qwen2" | "qwen3" | "mistral" => Ok(ArchKind::Llama),
            "gemma" | "gemma2" | "gemma3" => Ok(ArchKind::Gemma),
            other => Err(Error::Config(format!(
                "unsupported GGUF architecture {other:?}"
            ))),
        }
    }

    /// The family's chat prompt format.
    fn chat_format(self) -> Box<dyn ChatFormat> {
        match self {
            ArchKind::Qwen35 => Box::new(ChatMlThink),
            ArchKind::Llama => Box::new(ChatMl),
            ArchKind::Gemma => Box::new(GemmaChat),
        }
    }
}

/// Opens the checkpoint once, resolves its architecture and config,
/// instantiates the model and packages it with its tokenizer, chat format and
/// stop tokens. Format-agnostic: safetensors dirs and single-file GGUF both
/// work.
pub fn load(model_dir: &Path, max_seq: u32, backend: &Backend) -> Result<ChatModel> {
    let source = open(model_dir)?;
    let kind = arch_kind(source.as_ref())?;
    let config = config_for(source.as_ref(), kind)?;
    let model: Box<dyn LanguageModel> = match kind {
        ArchKind::Qwen35 => Box::new(Qwen35::load(source.as_ref(), &config, max_seq, backend)?),
        ArchKind::Llama => Box::new(llama::load(source.as_ref(), &config, max_seq, backend)?),
        ArchKind::Gemma => Box::new(gemma::load(source.as_ref(), &config, max_seq, backend)?),
    };
    let tokenizer = Tokenizer::load(model_dir, source.as_ref())?;
    let chat = kind.chat_format();
    let stop = stop_tokens(model.eos(), &tokenizer, chat.stop_literals());
    Ok(ChatModel {
        model,
        tokenizer,
        chat,
        stop,
    })
}

/// Model eos plus the family's reply terminators, resolved against the vocab
/// (literals absent from the vocab are skipped, duplicates dropped).
fn stop_tokens(eos: &[u32], tokenizer: &Tokenizer, literals: &[&str]) -> Vec<u32> {
    let mut stop = eos.to_vec();
    for lit in literals {
        if let Some(id) = tokenizer.token_id(lit)
            && !stop.contains(&id)
        {
            stop.push(id);
        }
    }
    stop
}

fn arch_kind(source: &dyn Checkpoint) -> Result<ArchKind> {
    match source.kind() {
        "safetensors" => {
            let v = source
                .config_json()?
                .ok_or_else(|| Error::Config("safetensors checkpoint has no config.json".into()))?;
            ArchKind::from_model_type(v["model_type"].as_str())
        }
        "gguf" => {
            let arch = source
                .metadata()
                .str("general.architecture")
                .ok_or_else(|| Error::Config("GGUF missing general.architecture".into()))?;
            ArchKind::from_gguf_arch(arch)
        }
        other => Err(Error::Config(format!(
            "unknown checkpoint format {other:?}"
        ))),
    }
}

fn config_for(source: &dyn Checkpoint, kind: ArchKind) -> Result<Value> {
    match source.kind() {
        "safetensors" => source
            .config_json()?
            .ok_or_else(|| Error::Config("safetensors checkpoint has no config.json".into())),
        "gguf" => gguf_config::synthesize_config(source, kind),
        other => Err(Error::Config(format!(
            "unknown checkpoint format {other:?}"
        ))),
    }
}
