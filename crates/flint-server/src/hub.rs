use std::sync::Arc;

use flint_architectures::chat::{ChatFormat, ThinkMode};
use flint_error::{Error, Result};
use flint_generate::{Engine, Grammar, SamplingParams, SessionId};
use flint_tokenizer::Tokenizer;
use serde_json::{Value, json};

use crate::engine_hub::{Client, Command, EngineHandle, Event};
use crate::tools::{Tool, tool_instruction, wrapper_schema};
use tokio::sync::mpsc::unbounded_channel;

pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool(String),
}

pub struct GenerateRequest {
    pub system: String,
    pub history: Vec<(String, String)>,
    pub user: String,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub sampling: Option<SamplingParams>,
    pub schema: Option<Value>,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    pub thinking: bool,
}

pub struct Generation {
    pub client: Client,
    pub think: ThinkMode,
}

#[derive(Clone)]
pub struct RequestDefaults {
    pub model_id: String,
    pub max_tokens: usize,
}

#[derive(Clone)]
pub struct Hub {
    inner: Arc<Inner>,
}

struct Inner {
    engine: EngineHandle,
    chat: Box<dyn ChatFormat + Send + Sync>,
    tokenizer: Tokenizer,
    model_id: String,
    context_len: u32,
    default_max_tokens: usize,
}

impl Hub {
    pub fn new(
        engine: Engine,
        chat: Box<dyn ChatFormat + Send + Sync>,
        tokenizer: Tokenizer,
        model_id: String,
        context_len: u32,
        default_max_tokens: usize,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                engine: crate::engine_hub::spawn(engine),
                chat,
                tokenizer,
                model_id,
                context_len,
                default_max_tokens,
            }),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    pub fn context_len(&self) -> u32 {
        self.inner.context_len
    }

    pub fn default_max_tokens(&self) -> usize {
        self.inner.default_max_tokens
    }

    pub fn defaults(&self) -> RequestDefaults {
        RequestDefaults {
            model_id: self.inner.model_id.clone(),
            max_tokens: self.inner.default_max_tokens,
        }
    }

    pub async fn generate(&self, req: &GenerateRequest) -> Result<Generation> {
        let tools = self.active_tools(req)?;
        let grammar = self.grammar(req, &tools)?;
        let think = self.think(req);
        let prompt = self.render(req, &tools, think);
        let stop_extra: Vec<u32> = req
            .stop
            .iter()
            .filter_map(|s| self.inner.tokenizer.token_id(s))
            .collect();
        let (tx, mut rx) = unbounded_channel();
        self.inner
            .engine
            .send(Command::Generate {
                prompt,
                max_tokens: req.max_tokens,
                stop_extra,
                sampling: req.sampling,
                grammar,
                tx,
            });
        match rx.recv().await {
            Some(Event::Started(id)) => Ok(Generation {
                client: Client::new(id, rx, self.inner.engine.sender()),
                think,
            }),
            Some(Event::Failed(e)) => Err(Error::Model(e)),
            Some(_) => Err(Error::Model("engine sent an unexpected event".into())),
            None => Err(Error::Model("engine thread is gone".into())),
        }
    }

    pub fn count_tokens(&self, req: &GenerateRequest) -> Result<usize> {
        let tools = self.active_tools(req)?;
        let prompt = self.render(req, &tools, self.think(req));
        Ok(self.inner.tokenizer.encode(&prompt)?.len())
    }

    fn think(&self, req: &GenerateRequest) -> ThinkMode {
        if !req.thinking || req.grammar_active() {
            return ThinkMode::None;
        }
        self.inner.chat.think_mode()
    }

    fn active_tools(&self, req: &GenerateRequest) -> Result<Vec<Tool>> {
        let tools = match &req.tool_choice {
            ToolChoice::None => Vec::new(),
            ToolChoice::Tool(name) => req
                .tools
                .iter()
                .filter(|t| &t.name == name)
                .cloned()
                .collect(),
            _ => req.tools.clone(),
        };
        if matches!(req.tool_choice, ToolChoice::Tool(_) | ToolChoice::Required) && tools.is_empty()
        {
            return Err(Error::Model(
                "tool_choice requires at least one matching tool".into(),
            ));
        }
        Ok(tools)
    }

    fn grammar(&self, req: &GenerateRequest, tools: &[Tool]) -> Result<Option<Grammar>> {
        let text_allowed = matches!(req.tool_choice, ToolChoice::Auto);
        match (req.schema.as_ref(), tools.is_empty()) {
            (Some(schema), false) => Ok(Some(Grammar::from_schema(&json!({
                "anyOf": [schema, wrapper_schema(tools, text_allowed)]
            }))?)),
            (Some(schema), true) => Ok(Some(Grammar::from_schema(schema)?)),
            (None, false) => Ok(Some(Grammar::from_schema(&wrapper_schema(
                tools,
                text_allowed,
            ))?)),
            (None, true) => Ok(None),
        }
    }

    fn render(&self, req: &GenerateRequest, tools: &[Tool], think: ThinkMode) -> String {
        let mut system = req.system.clone();
        if !tools.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&tool_instruction(tools));
        }
        self.inner
            .chat
            .render(&system, &req.history, &req.user, think != ThinkMode::None)
    }

    pub fn close_now(&self, id: SessionId) {
        self.inner.engine.send(Command::Close(id));
    }
}

impl GenerateRequest {
    pub fn grammar_active(&self) -> bool {
        self.schema.is_some()
            || (!matches!(self.tool_choice, ToolChoice::None) && !self.tools.is_empty())
    }

    pub fn tool_wrapper(&self) -> bool {
        !matches!(self.tool_choice, ToolChoice::None) && !self.tools.is_empty()
    }
}
