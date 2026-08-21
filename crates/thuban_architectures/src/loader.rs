use std::path::Path;

use thuban_backend::Backend;
use thuban_checkpoint::{Checkpoint, open_checkpoint};
use thuban_error::{Error, Result};
use thuban_model::pool::ArenaSpec;
use thuban_model::LanguageModel;
use thuban_tokenizer::Tokenizer;
use serde_json::Value;

use crate::chat::{
    ChatFormat, ChatMl, ChatMlThink, Gemma4Chat, GemmaChat, Llama2Chat, Llama3Chat, Phi4Chat,
};
use crate::{gemma, gemma4, llama, phi, qwen35};

pub struct ChatModel {
    pub model: Box<dyn LanguageModel + Send>,
    pub tokenizer: Tokenizer,
    pub chat: Box<dyn ChatFormat + Send + Sync>,
    pub stop: Vec<u32>,
}

pub struct LoadOptions {
    pub seq_lens: Vec<u32>,
    pub pages: Option<u32>,
    pub spec_depth: Option<u32>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            seq_lens: vec![4096],
            pages: None,
            spec_depth: None,
        }
    }
}

#[derive(Clone, Copy)]
pub enum Family {
    Qwen35,
    Llama,
    Gemma,
    Phi,
    Gemma4,
}

impl Family {
    fn from_gguf_arch(a: &str) -> Result<Self> {
        match a {
            "qwen35" => Ok(Family::Qwen35),
            "llama" | "qwen2" | "qwen3" | "mistral" => Ok(Family::Llama),
            "gemma" | "gemma2" | "gemma3" => Ok(Family::Gemma),
            "gemma4" => Ok(Family::Gemma4),
            "phi3" => Ok(Family::Phi),
            other => Err(Error::Config(format!(
                "unsupported GGUF architecture {other:?}"
            ))),
        }
    }

    fn chat_format(self, config: &Value) -> Box<dyn ChatFormat + Send + Sync> {
        match self {
            Family::Qwen35 => Box::new(ChatMlThink),
            Family::Llama => llama_chat(config),
            Family::Gemma => Box::new(GemmaChat),
            Family::Phi => Box::new(Phi4Chat),
            Family::Gemma4 => Box::new(Gemma4Chat),
        }
    }
}

fn llama_chat(config: &Value) -> Box<dyn ChatFormat + Send + Sync> {
    match config.get("vocab_size").and_then(Value::as_u64) {
        Some(128256) => Box::new(Llama3Chat),
        Some(32000) => Box::new(Llama2Chat),
        _ => Box::new(ChatMl),
    }
}

pub fn load(model_dir: &Path, opts: &LoadOptions, backend: &Backend) -> Result<ChatModel> {
    let source = open_checkpoint(model_dir)?;
    let family = family_of(&source)?;
    let config = crate::gguf::synthesize_config(&source, family)?;
    let arena = ArenaSpec {
        seq_lens: opts.seq_lens.clone(),
        pages: opts.pages,
    };
    let model: Box<dyn LanguageModel + Send> = match family {
        Family::Qwen35 => Box::new(qwen35::Qwen35::load(
            &source,
            &config,
            &arena,
            backend,
        )?),
        Family::Llama => Box::new(llama::load(
            &source,
            &config,
            &arena,
            opts.spec_depth,
            backend,
        )?),
        Family::Gemma => Box::new(gemma::load(
            &source,
            &config,
            &arena,
            opts.spec_depth,
            backend,
        )?),
        Family::Phi => Box::new(phi::load(
            &source,
            &config,
            &arena,
            opts.spec_depth,
            backend,
        )?),
        Family::Gemma4 => Box::new(gemma4::load(
            &source,
            &config,
            &arena,
            opts.spec_depth,
            backend,
        )?),
    };
    let tokenizer = thuban_tokenizer::load(model_dir, &source)?;
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
    let arch = source
        .metadata()?
        .str("general.architecture")
        .ok_or_else(|| Error::Config("GGUF missing general.architecture".into()))?;
    Family::from_gguf_arch(arch)
}
