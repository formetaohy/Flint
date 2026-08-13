pub mod chat;
pub mod gemma;
pub mod gemma4;
pub mod gguf_config;
pub mod keys;
pub mod llama;
pub mod phi;
pub mod qwen35;
pub mod transformer;

use std::path::Path;

use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, CheckpointKind, open_checkpoint};
use flint_error::{Error, Result};
use flint_model::LanguageModel;
use flint_tokenizer::Tokenizer;
use serde_json::Value;

use crate::chat::{
    ChatFormat, ChatMl, ChatMlThink, Gemma4Chat, GemmaChat, Llama2Chat, Llama3Chat, Phi3Chat,
    Phi4Chat,
};
use crate::qwen35::Qwen35;

pub struct ChatModel {
    pub model: Box<dyn LanguageModel>,
    pub tokenizer: Tokenizer,
    pub chat: Box<dyn ChatFormat>,
    pub stop: Vec<u32>,
}

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

    fn chat_format(self, config: &serde_json::Value) -> Box<dyn ChatFormat> {
        match self {
            Family::Qwen35 => Box::new(ChatMlThink),
            Family::Llama => llama_chat(config),
            Family::Gemma => Box::new(GemmaChat),
            Family::Phi => Box::new(Phi4Chat),
            Family::PhiMoe => Box::new(Phi3Chat),
            Family::Gemma4 => Box::new(Gemma4Chat),
        }
    }
}

fn llama_chat(config: &serde_json::Value) -> Box<dyn ChatFormat> {
    match config.get("vocab_size").and_then(serde_json::Value::as_u64) {
        Some(128256) => Box::new(Llama3Chat),
        Some(32000) => Box::new(Llama2Chat),
        _ => Box::new(ChatMl),
    }
}

pub fn load(model_dir: &Path, max_seq: u32, backend: &Backend) -> Result<ChatModel> {
    let source = open_checkpoint(model_dir)?;
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
    let tokenizer = flint_tokenizer::load_gguf(model_dir, source.as_ref())?;
    let chat = family.chat_format(&config);
    let stop = stop_tokens(model.eos(), &tokenizer, chat.stop_literals());
    Ok(ChatModel {
        model,
        tokenizer,
        chat,
        stop,
    })
}

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
        CheckpointKind::Safetensors => {
            Family::from_model_type(source.config_json()?["model_type"].as_str())
        }
        CheckpointKind::Gguf => {
            let arch = source
                .metadata()?
                .str("general.architecture")
                .ok_or_else(|| Error::Config("GGUF missing general.architecture".into()))?;
            Family::from_gguf_arch(arch)
        }
    }
}

fn config_for(source: &dyn Checkpoint, family: Family) -> Result<Value> {
    match source.kind() {
        CheckpointKind::Safetensors => source.config_json(),
        CheckpointKind::Gguf => gguf_config::synthesize_config(source, family),
    }
}
