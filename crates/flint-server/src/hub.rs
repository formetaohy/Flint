use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};

use flint_architectures::chat::ChatFormat;
use flint_error::{Error, Result};
use flint_generate::{Engine, Grammar, SamplingParams};
use flint_tokenizer::Tokenizer;
use serde_json::{Value, json};

use crate::engine_hub::{Client, CloseGuard, Command, Event, spawn};
use crate::tools::{Tool, tool_instruction, wrapper_schema};

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
}

#[derive(Clone)]
pub struct Hub {
    inner: Arc<Inner>,
}

struct Inner {
    cmd: Sender<Command>,
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
        let cmd = spawn(engine);
        Self {
            inner: Arc::new(Inner {
                cmd,
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

    pub fn generate(&self, req: &GenerateRequest) -> Result<Client> {
        let tools = self.active_tools(req)?;
        let grammar = self.grammar(req, &tools)?;
        let prompt = self.render(req, &tools);
        let stop_extra: Vec<u32> = req
            .stop
            .iter()
            .filter_map(|s| self.inner.tokenizer.token_id(s))
            .collect();
        let (tx, rx) = channel();
        self.inner
            .cmd
            .send(Command::Generate {
                prompt,
                max_tokens: req.max_tokens,
                stop_extra,
                sampling: req.sampling,
                grammar,
                tx,
            })
            .map_err(|_| Error::Model("engine thread is gone".into()))?;
        match rx.recv() {
            Ok(Event::Started(id)) => {
                let guard = CloseGuard {
                    id,
                    cmd: self.inner.cmd.clone(),
                };
                Ok(Client { id, rx, guard })
            }
            Ok(Event::Failed(e)) => Err(Error::Model(e)),
            _ => Err(Error::Model("engine thread is gone".into())),
        }
    }

    pub fn count_tokens(&self, req: &GenerateRequest) -> Result<usize> {
        let tools = self.active_tools(req)?;
        let prompt = self.render(req, &tools);
        Ok(self.inner.tokenizer.encode(&prompt)?.len())
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

    fn render(&self, req: &GenerateRequest, tools: &[Tool]) -> String {
        let mut system = req.system.clone();
        if !tools.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&tool_instruction(tools));
        }
        self.inner.chat.render(&system, &req.history, &req.user)
    }
}

pub fn grammar_active(schema: Option<&Value>, tools: &[Tool], choice: &ToolChoice) -> bool {
    let has_tools = !matches!(choice, ToolChoice::None) && !tools.is_empty();
    schema.is_some() || has_tools
}

impl GenerateRequest {
    pub fn grammar_active(&self) -> bool {
        grammar_active(self.schema.as_ref(), &self.tools, &self.tool_choice)
    }

    pub fn tool_wrapper(&self) -> bool {
        !matches!(self.tool_choice, ToolChoice::None) && !self.tools.is_empty()
    }
}

impl Hub {
    pub fn close_now(&self, id: flint_generate::SessionId) {
        let _ = self.inner.cmd.send(Command::Close(id));
    }
}
