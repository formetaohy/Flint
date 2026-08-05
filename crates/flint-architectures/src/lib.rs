//! Concrete model architectures, their chat prompt formats, and the
//! config/format-driven registry that assembles them into a ready-to-chat
//! bundle. Depends on the `flint-model` abstraction framework.

pub mod chat;
pub mod dense;
pub mod gemma;
pub mod gemma4;
pub mod gguf_config;
pub mod llama;
pub mod names;
pub mod phi;
pub mod qwen35;

use std::path::Path;

use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, open};
use flint_error::{Error, Result};
use flint_model::LanguageModel;
use flint_tokenizer::Tokenizer;
use serde_json::Value;

use crate::chat::{ChatFormat, ChatMl, ChatMlThink, Gemma4Chat, GemmaChat, Phi4Chat, PhiChat};
use crate::qwen35::Qwen35;

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
pub enum Family {
    Qwen35,
    Llama,
    Gemma,
    Phi,
    PhiMoe,
    Gemma4,
}

impl Family {
    /// Dispatches on a safetensors config's `model_type`.
    fn from_model_type(t: Option<&str>) -> Result<Self> {
        match t {
            Some("qwen3_5") => Ok(Family::Qwen35),
            Some("llama" | "qwen2" | "qwen3" | "mistral") => Ok(Family::Llama),
            Some("gemma" | "gemma2" | "gemma3" | "gemma3_text") => Ok(Family::Gemma),
            Some("gemma4" | "gemma4_text") => Ok(Family::Gemma4),
            Some("phi3") => Ok(Family::Phi),
            Some("phimoe") => Ok(Family::PhiMoe),
            other => Err(Error::Config(format!("unsupported model_type {other:?}"))),
        }
    }

    /// Dispatches on a GGUF `general.architecture`.
    fn from_gguf_arch(a: &str) -> Result<Self> {
        match a {
            "llama" | "qwen2" | "qwen3" | "mistral" => Ok(Family::Llama),
            "gemma" | "gemma2" | "gemma3" => Ok(Family::Gemma),
            "gemma4" => Ok(Family::Gemma4),
            "phi3" => Ok(Family::Phi),
            "phimoe" => Ok(Family::PhiMoe),
            other => Err(Error::Config(format!(
                "unsupported GGUF architecture {other:?}"
            ))),
        }
    }

    /// The family's chat prompt format.
    fn chat_format(self) -> Box<dyn ChatFormat> {
        match self {
            Family::Qwen35 => Box::new(ChatMlThink),
            Family::Llama => Box::new(ChatMl),
            Family::Gemma => Box::new(GemmaChat),
            Family::Phi => Box::new(Phi4Chat),
            Family::PhiMoe => Box::new(PhiChat),
            Family::Gemma4 => Box::new(Gemma4Chat),
        }
    }
}

/// Opens the checkpoint once, resolves its architecture and config,
/// instantiates the model and packages it with its tokenizer, chat format and
/// stop tokens. Format-agnostic: safetensors dirs and single-file GGUF both
/// work.
pub fn load(model_dir: &Path, max_seq: u32, backend: &Backend) -> Result<ChatModel> {
    let source = open(model_dir)?;
    let family = family_of(source.as_ref())?;
    let config = config_for(source.as_ref(), family)?;
    let model: Box<dyn LanguageModel> = match family {
        Family::Qwen35 => Box::new(Qwen35::load(source.as_ref(), &config, max_seq, backend)?),
        Family::Llama => Box::new(llama::load(source.as_ref(), &config, max_seq, backend)?),
        Family::Gemma => Box::new(gemma::load(source.as_ref(), &config, max_seq, backend)?),
        Family::Phi => Box::new(phi::load(source.as_ref(), &config, max_seq, backend)?),
        Family::PhiMoe => Box::new(phi::load_moe(source.as_ref(), &config, max_seq, backend)?),
        Family::Gemma4 => Box::new(gemma4::load(source.as_ref(), &config, max_seq, backend)?),
    };
    let tokenizer = Tokenizer::load(model_dir, source.as_ref())?;
    let chat = family.chat_format();
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

fn family_of(source: &dyn Checkpoint) -> Result<Family> {
    match source.kind() {
        "safetensors" => {
            let v = source
                .config_json()?
                .ok_or_else(|| Error::Config("safetensors checkpoint has no config.json".into()))?;
            Family::from_model_type(v["model_type"].as_str())
        }
        "gguf" => {
            let arch = source
                .metadata()
                .str("general.architecture")
                .ok_or_else(|| Error::Config("GGUF missing general.architecture".into()))?;
            Family::from_gguf_arch(arch)
        }
        other => Err(Error::Config(format!(
            "unknown checkpoint format {other:?}"
        ))),
    }
}

fn config_for(source: &dyn Checkpoint, family: Family) -> Result<Value> {
    match source.kind() {
        "safetensors" => source
            .config_json()?
            .ok_or_else(|| Error::Config("safetensors checkpoint has no config.json".into())),
        "gguf" => gguf_config::synthesize_config(source, family),
        other => Err(Error::Config(format!(
            "unknown checkpoint format {other:?}"
        ))),
    }
}
