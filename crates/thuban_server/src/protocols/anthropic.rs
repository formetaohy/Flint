use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thuban_architectures::chat::ThinkMode;
use thuban_error::{Error, Result};
use thuban_generate::{GenStats, Piece, SamplingParams};
use serde_json::{Value, json};

use crate::generator::{GenerateRequest, RequestDefaults, ToolChoice};
use crate::protocols::{
    Chat, DecisionSink, Part, SseFrame, StreamSink, collect, json_response, length_hit, next_id,
    split_reasoning, stream_response,
};
use crate::server::AppState;
use crate::tools::{Tool, render_tool_call};

pub struct Parsed {
    pub req: GenerateRequest,
    pub stream: bool,
    pub model: String,
}

pub async fn handle(State(state): State<AppState>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid JSON body: {e}"),
            );
        }
    };
    let parsed = match parse(&body, &state.generator.defaults()) {
        Ok(p) => p,
        Err(e) => {
            return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
        }
    };
    let generation = match state.generator.generate(&parsed.req).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e.to_string()),
    };
    let sink = MessageSink::new(
        parsed.model.clone(),
        parsed.req.max_tokens,
        generation.think,
        parsed.req.tool_wrapper(),
    );
    if parsed.stream {
        return stream_response(generation.client, sink).into_response();
    }
    let sink = match collect(generation.client, sink).await {
        Ok(s) => s,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e.to_string()),
    };
    json_response(sink.final_json()).into_response()
}

fn error(status: StatusCode, kind: &str, message: String) -> Response {
    (
        status,
        json_response(json!({"type": "error", "error": {"type": kind, "message": message}})),
    )
        .into_response()
}

pub async fn handle_count_tokens(State(state): State<AppState>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid JSON body: {e}"),
            );
        }
    };
    let parsed = match parse(&body, &state.generator.defaults()) {
        Ok(p) => p,
        Err(e) => {
            return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
        }
    };
    match state.generator.count_tokens(&parsed.req) {
        Ok(n) => json_response(json!({"input_tokens": n})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e.to_string()),
    }
}

pub fn parse(body: &Value, defaults: &RequestDefaults) -> Result<Parsed> {
    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_tokens"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(defaults.max_tokens);
    let stop = body
        .get("stop_sequences")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut sampling = SamplingParams::default();
    let mut any = false;
    if let Some(t) = body["temperature"].as_f64() {
        sampling.temperature = t as f32;
        any = true;
    }
    if let Some(t) = body["top_p"].as_f64() {
        sampling.top_p = t as f32;
        any = true;
    }
    if let Some(t) = body["top_k"].as_u64() {
        sampling.top_k = t as usize;
        any = true;
    }
    let thinking = !matches!(body["thinking"]["type"].as_str(), Some("disabled"));
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|t| Tool {
                    name: t["name"].as_str().unwrap_or_default().to_string(),
                    description: t["description"].as_str().unwrap_or_default().to_string(),
                    schema: t
                        .get("input_schema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object"})),
                })
                .collect()
        })
        .unwrap_or_default();
    let tool_choice = match body.get("tool_choice") {
        None | Some(Value::Null) => ToolChoice::Auto,
        Some(t) if t["type"] == "none" => ToolChoice::None,
        Some(t) if t["type"] == "auto" => ToolChoice::Auto,
        Some(t) if t["type"] == "any" => ToolChoice::Required,
        Some(t) if t["type"] == "tool" => t["name"]
            .as_str()
            .map(|n| ToolChoice::Tool(n.to_string()))
            .unwrap_or(ToolChoice::Required),
        _ => ToolChoice::Auto,
    };
    let (system, history, user) = extract_messages(body)?;
    let model = body["model"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| defaults.model_id.clone());
    Ok(Parsed {
        req: GenerateRequest {
            system,
            history,
            user,
            max_tokens,
            stop,
            sampling: any.then_some(sampling),
            schema: None,
            tools,
            tool_choice,
            thinking,
        },
        stream,
        model,
    })
}

fn extract_messages(body: &Value) -> Result<Chat> {
    let mut system = String::new();
    match body.get("system") {
        Some(Value::String(s)) => system.push_str(s),
        Some(Value::Array(parts)) => {
            for p in parts {
                if let Some(t) = p["text"].as_str() {
                    system.push_str(t);
                }
            }
        }
        _ => {}
    }
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("messages must be an array".into()))?;
    let mut tool_names: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for m in messages {
        if let Some(blocks) = m["content"].as_array()
            && m["role"] == "assistant"
        {
            for b in blocks {
                if b["type"] == "tool_use" {
                    let id = b["id"].as_str().unwrap_or_default().to_string();
                    let name = b["name"].as_str().unwrap_or_default().to_string();
                    tool_names.insert(id, name);
                }
            }
        }
    }
    let mut history: Vec<(String, String)> = Vec::new();
    let mut user = String::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or_default();
        let content = match &m["content"] {
            Value::String(s) => Some(split_reasoning(s)),
            Value::Array(blocks) => Some(render_blocks(blocks, &tool_names)),
            _ => None,
        };
        let Some(content) = content else { continue };
        match role {
            "user" => {
                if !user.is_empty() {
                    user.push('\n');
                }
                user.push_str(&content);
            }
            "assistant" => {
                let pending = std::mem::take(&mut user);
                history.push((pending, content));
            }
            other => eprintln!("[server] ignoring message role {other:?}"),
        }
    }
    Ok((system, history, user))
}

fn render_blocks(
    blocks: &[Value],
    tool_names: &std::collections::HashMap<String, String>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        match b["type"].as_str() {
            Some("text") => {
                if let Some(t) = b["text"].as_str() {
                    parts.push(split_reasoning(t));
                }
            }
            Some("thinking") | Some("redacted_thinking") => {}
            Some("tool_use") => {
                let name = b["name"].as_str().unwrap_or_default();
                let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                parts.push(render_tool_call(name, &input));
            }
            Some("tool_result") => {
                let id = b["tool_use_id"].as_str().unwrap_or_default();
                let name = tool_names.get(id).map(String::as_str).unwrap_or("");
                let label = if !name.is_empty() {
                    format!("{name} ({id})")
                } else {
                    id.to_string()
                };
                let inner = match b["content"].as_str() {
                    Some(s) => s.to_string(),
                    None => b["content"]
                        .as_array()
                        .map(|blocks| render_blocks(blocks, tool_names))
                        .unwrap_or_default(),
                };
                let err = b["is_error"].as_bool().unwrap_or(false);
                parts.push(format!(
                    "[tool result {label}]\n{}{}",
                    if err { "[error]\n" } else { "" },
                    inner
                ));
            }
            other => eprintln!("[server] ignoring content block type {other:?}"),
        }
    }
    parts.join("\n")
}

pub struct MessageSink {
    id: String,
    model: String,
    max_tokens: usize,
    decision: DecisionSink,
    started: bool,
    blocks: Vec<Block>,
    open_block: Option<usize>,
    stats: Option<GenStats>,
}

#[derive(Clone, Copy, PartialEq)]
enum Block {
    Thinking,
    Text,
    ToolUse,
}

impl MessageSink {
    pub fn new(model: String, max_tokens: usize, think: ThinkMode, constrained: bool) -> Self {
        Self {
            id: next_id("msg_"),
            model,
            max_tokens,
            decision: if constrained {
                DecisionSink::constrained()
            } else {
                DecisionSink::plain(think)
            },
            started: false,
            blocks: Vec::new(),
            open_block: None,
            stats: None,
        }
    }

    fn ensure_started(&mut self, frames: &mut Vec<SseFrame>) {
        if self.started {
            return;
        }
        self.started = true;
        frames.push(SseFrame {
            event: Some("message_start"),
            data: json!({"type": "message_start", "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 1}
            }})
            .to_string(),
        });
    }

    fn start_block(&mut self, block: Block, content_block: Value, frames: &mut Vec<SseFrame>) {
        self.ensure_started(frames);
        self.close_open_block(frames);
        let index = self.blocks.len();
        self.blocks.push(block);
        self.open_block = Some(index);
        frames.push(SseFrame {
            event: Some("content_block_start"),
            data: json!({"type": "content_block_start", "index": index, "content_block": content_block})
                .to_string(),
        });
    }

    fn close_open_block(&mut self, frames: &mut Vec<SseFrame>) {
        let Some(index) = self.open_block.take() else {
            return;
        };
        if self.blocks.get(index) == Some(&Block::Thinking) {
            frames.push(SseFrame {
                event: Some("content_block_delta"),
                data: json!({"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": ""}})
                    .to_string(),
            });
        }
        frames.push(SseFrame {
            event: Some("content_block_stop"),
            data: json!({"type": "content_block_stop", "index": index}).to_string(),
        });
    }

    fn stop_reason(&self, stats: &GenStats) -> &'static str {
        if self.decision.was_tool_branch() {
            "tool_use"
        } else if length_hit(stats, self.max_tokens) {
            "max_tokens"
        } else {
            "end_turn"
        }
    }

    fn usage(&self, stats: &GenStats) -> Value {
        json!({"input_tokens": stats.prefill_tokens, "output_tokens": stats.decode_tokens})
    }

    fn emit_part(&mut self, part: &Part, frames: &mut Vec<SseFrame>) {
        match part {
            Part::Text(chunk) => {
                if !self.blocks.contains(&Block::Text) {
                    self.start_block(Block::Text, json!({"type": "text", "text": ""}), frames);
                }
                let index = self
                    .blocks
                    .iter()
                    .position(|b| *b == Block::Text)
                    .expect("text block was started");
                frames.push(SseFrame {
                    event: Some("content_block_delta"),
                    data: json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": chunk}})
                        .to_string(),
                });
            }
            Part::Reasoning(chunk) => {
                if !self.blocks.contains(&Block::Thinking) {
                    self.start_block(
                        Block::Thinking,
                        json!({"type": "thinking", "thinking": "", "signature": ""}),
                        frames,
                    );
                }
                let index = self
                    .blocks
                    .iter()
                    .position(|b| *b == Block::Thinking)
                    .expect("thinking block was started");
                frames.push(SseFrame {
                    event: Some("content_block_delta"),
                    data: json!({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": chunk}})
                        .to_string(),
                });
            }
            Part::CallStart { name, .. } => {
                self.start_block(
                    Block::ToolUse,
                    json!({"type": "tool_use", "id": next_id("toolu_"), "name": name, "input": {}}),
                    frames,
                );
            }
            Part::CallArgs { chunk, .. } => {
                let index = self.open_block.expect("tool use block is open");
                frames.push(SseFrame {
                    event: Some("content_block_delta"),
                    data: json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": chunk}})
                        .to_string(),
                });
            }
        }
    }

    pub fn final_json(&self) -> Value {
        let stats = self.stats.expect("completion stats are recorded");
        let content: Vec<Value> = if self.decision.was_tool_branch() {
            self.decision
                .calls
                .iter()
                .map(|c| {
                    let input: Value = serde_json::from_str(&c.args).unwrap_or_else(|_| json!({}));
                    json!({"type": "tool_use", "id": next_id("toolu_"), "name": c.name.clone(), "input": input})
                })
                .collect()
        } else {
            let mut blocks = Vec::new();
            if self.decision.has_reasoning() {
                blocks.push(json!({
                    "type": "thinking",
                    "thinking": self.decision.reasoning_text,
                    "signature": "",
                }));
            }
            if !self.decision.text.is_empty() || blocks.is_empty() {
                blocks.push(json!({"type": "text", "text": self.decision.text}));
            }
            blocks
        };
        json!({
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": content,
            "stop_reason": self.stop_reason(&stats),
            "stop_sequence": null,
            "usage": self.usage(&stats),
        })
    }
}

impl StreamSink for MessageSink {
    fn on_delta(&mut self, piece: &Piece) -> Result<Vec<SseFrame>> {
        let parts = self.decision.push(piece)?;
        let mut frames = Vec::new();
        for part in &parts {
            self.emit_part(part, &mut frames);
        }
        self.decision.route(parts);
        Ok(frames)
    }

    fn on_done(&mut self, stats: &GenStats) -> Result<Vec<SseFrame>> {
        self.stats = Some(*stats);
        let parts = self.decision.finish()?;
        let mut frames = Vec::new();
        for part in &parts {
            self.emit_part(part, &mut frames);
        }
        self.decision.route(parts);
        self.close_open_block(&mut frames);
        let stop_reason = self.stop_reason(stats);
        frames.push(SseFrame {
            event: Some("message_delta"),
            data: json!({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": null}, "usage": {"output_tokens": stats.decode_tokens}})
                .to_string(),
        });
        frames.push(SseFrame {
            event: Some("message_stop"),
            data: json!({"type": "message_stop"}).to_string(),
        });
        Ok(frames)
    }

    fn on_failed(&mut self, msg: &str) -> Vec<SseFrame> {
        vec![SseFrame {
            event: Some("error"),
            data: json!({"type": "error", "error": {"type": "api_error", "message": msg}}).to_string(),
        }]
    }
}
