use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use flint_architectures::chat::ThinkMode;
use flint_error::{Error, Result};
use flint_generate::{GenStats, Piece, SamplingParams};
use serde_json::{Value, json};

use crate::generator::{GenerateRequest, ToolChoice};
use crate::protocols::{
    Chat, DecisionSink, Part, SseFrame, StreamSink, collect, json_response, length_hit, next_id,
    now_secs, split_reasoning, stream_response,
};
use crate::server::AppState;
use crate::tools::Tool;

pub struct Parsed {
    pub req: GenerateRequest,
    pub stream: bool,
    pub stream_options_usage: bool,
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
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    };
    let sink = ChatSink::new(
        parsed.model.clone(),
        parsed.req.max_tokens,
        parsed.stream_options_usage,
        generation.think,
        parsed.req.tool_wrapper(),
    );
    if parsed.stream {
        return stream_response(generation.client, sink).into_response();
    }
    let sink = match collect(generation.client, sink).await {
        Ok(s) => s,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    };
    json_response(sink.final_json()).into_response()
}

fn error(status: StatusCode, kind: &str, message: String) -> Response {
    (
        status,
        json_response(json!({"error": {"message": message, "type": kind}})),
    )
        .into_response()
}

pub fn parse(body: &Value, defaults: &crate::generator::RequestDefaults) -> Result<Parsed> {
    let stream = body["stream"].as_bool().unwrap_or(false);
    let stream_options_usage = body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let max_tokens = body["max_tokens"]
        .as_u64()
        .or_else(|| body["max_completion_tokens"].as_u64())
        .map(|v| v as usize)
        .unwrap_or(defaults.max_tokens);
    let stop = match &body["stop"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    let sampling = parse_sampling(body);
    let tools = parse_tools(body);
    let tool_choice = parse_tool_choice(body);
    let thinking = body["enable_thinking"]
        .as_bool()
        .or_else(|| body["chat_template_kwargs"]["enable_thinking"].as_bool())
        .unwrap_or(true);
    let schema = body
        .get("response_format")
        .and_then(|f| f.get("json_schema"))
        .and_then(|j| j.get("schema"))
        .cloned();
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
            sampling,
            schema,
            tools,
            tool_choice,
            thinking,
        },
        stream,
        stream_options_usage,
        model,
    })
}

pub fn parse_sampling(body: &Value) -> Option<SamplingParams> {
    let mut p = SamplingParams::default();
    let mut any = false;
    if let Some(t) = body["temperature"].as_f64() {
        p.temperature = t as f32;
        any = true;
    }
    if let Some(t) = body["top_p"].as_f64() {
        p.top_p = t as f32;
        any = true;
    }
    if let Some(t) = body["top_k"].as_u64() {
        p.top_k = t as usize;
        any = true;
    }
    if let Some(t) = body["min_p"].as_f64() {
        p.min_p = t as f32;
        any = true;
    }
    if let Some(t) = body["repetition_penalty"]
        .as_f64()
        .or_else(|| body["repeat_penalty"].as_f64())
    {
        p.repeat_penalty = t as f32;
        any = true;
    }
    any.then_some(p)
}

pub fn parse_tools(body: &Value) -> Vec<Tool> {
    let Some(tools) = body["tools"].as_array() else {
        return Vec::new();
    };
    tools
        .iter()
        .filter_map(|t| {
            let f = match t.get("function") {
                Some(f) => f,
                None => t,
            };
            Some(Tool {
                name: f["name"].as_str()?.to_string(),
                description: f["description"].as_str().unwrap_or_default().to_string(),
                schema: f
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"})),
            })
        })
        .collect()
}

pub fn parse_tool_choice(body: &Value) -> ToolChoice {
    match body.get("tool_choice") {
        None | Some(Value::Null) => ToolChoice::Auto,
        Some(Value::String(s)) if s == "none" => ToolChoice::None,
        Some(Value::String(s)) if s == "required" => ToolChoice::Required,
        Some(Value::String(s)) if s == "auto" => ToolChoice::Auto,
        Some(Value::Object(o)) => o
            .get("function")
            .and_then(|f| f["name"].as_str())
            .map(|n| ToolChoice::Tool(n.to_string()))
            .unwrap_or(ToolChoice::Auto),
        _ => ToolChoice::Auto,
    }
}

pub fn extract_messages(body: &Value) -> Result<Chat> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("messages must be an array".into()))?;
    let mut system = String::new();
    let mut history: Vec<(String, String)> = Vec::new();
    let mut user = String::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or_default();
        match role {
            "system" | "developer" => {
                if let Some(c) = content_text(&m["content"]) {
                    if !system.is_empty() {
                        system.push_str("\n\n");
                    }
                    system.push_str(&c);
                }
            }
            "user" => {
                if let Some(c) = content_text(&m["content"]) {
                    if !user.is_empty() {
                        user.push('\n');
                    }
                    user.push_str(&c);
                }
            }
            "assistant" => {
                let mut reply = String::new();
                if let Some(c) = content_text(&m["content"]) {
                    reply.push_str(&split_reasoning(&c));
                }
                if let Some(calls) = m["tool_calls"].as_array()
                    && !calls.is_empty()
                {
                    let calls: Vec<Value> = calls
                        .iter()
                        .filter_map(|c| {
                            let f = &c["function"];
                            Some(json!({
                                "name": f["name"].as_str()?,
                                "arguments": serde_json::from_str::<Value>(
                                    f["arguments"].as_str().unwrap_or("{}")
                                ).ok()?
                            }))
                        })
                        .collect();
                    if !reply.is_empty() {
                        reply.push('\n');
                    }
                    reply.push_str(&json!({"type": "tool_call", "calls": calls}).to_string());
                }
                let pending = std::mem::take(&mut user);
                history.push((pending, reply));
            }
            "tool" => {
                let id = m["tool_call_id"].as_str().unwrap_or_default();
                let c = content_text(&m["content"]).unwrap_or_default();
                user.push_str(&format!("[tool result {id}]\n{c}"));
            }
            other => eprintln!("[server] ignoring chat message role {other:?}"),
        }
    }
    if system.is_empty() && history.is_empty() && user.is_empty() {
        return Err(Error::Config("messages contain no content".into()));
    }
    Ok((system, history, user))
}

pub fn content_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let text: Vec<String> = parts
                .iter()
                .filter_map(|p| p["text"].as_str().map(str::to_string))
                .collect();
            (!text.is_empty()).then(|| text.join("\n"))
        }
        _ => None,
    }
}

pub struct ChatSink {
    id: String,
    created: u64,
    model: String,
    max_tokens: usize,
    usage_events: bool,
    decision: DecisionSink,
    started: bool,
    stats: Option<GenStats>,
}

impl ChatSink {
    pub fn new(
        model: String,
        max_tokens: usize,
        usage_events: bool,
        think: ThinkMode,
        constrained: bool,
    ) -> Self {
        Self {
            id: next_id("chatcmpl-"),
            created: now_secs(),
            model,
            max_tokens,
            usage_events,
            decision: if constrained {
                DecisionSink::constrained()
            } else {
                DecisionSink::plain(think)
            },
            started: false,
            stats: None,
        }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> SseFrame {
        SseFrame {
            event: None,
            data: json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }]
            })
            .to_string(),
        }
    }

    fn finish_reason(&self, stats: &GenStats) -> &'static str {
        if self.decision.was_tool_branch() {
            "tool_calls"
        } else if length_hit(stats, self.max_tokens) {
            "length"
        } else {
            "stop"
        }
    }

    fn usage(&self, stats: &GenStats) -> Value {
        let mut usage = json!({
            "prompt_tokens": stats.prefill_tokens,
            "completion_tokens": stats.decode_tokens,
            "total_tokens": stats.prefill_tokens + stats.decode_tokens,
        });
        if self.decision.reasoning_tokens > 0 {
            usage["completion_tokens_details"] = json!({
                "reasoning_tokens": self.decision.reasoning_tokens
            });
        }
        usage
    }

    fn emit_part(&mut self, part: &Part, frames: &mut Vec<SseFrame>) {
        match part {
            Part::Text(chunk) => {
                let delta = if !self.started {
                    self.started = true;
                    json!({"role": "assistant", "content": chunk})
                } else {
                    json!({"content": chunk})
                };
                frames.push(self.chunk(delta, None));
            }
            Part::Reasoning(chunk) => {
                let delta = if !self.started {
                    self.started = true;
                    json!({"role": "assistant", "reasoning_content": chunk})
                } else {
                    json!({"reasoning_content": chunk})
                };
                frames.push(self.chunk(delta, None));
            }
            Part::CallStart { index, name } => {
                self.started = true;
                frames.push(self.chunk(
                    json!({"tool_calls": [{"index": index, "id": next_id("call_"), "type": "function", "function": {"name": name, "arguments": ""}}]}),
                    None,
                ));
            }
            Part::CallArgs { index, chunk } => {
                frames.push(self.chunk(
                    json!({"tool_calls": [{"index": index, "function": {"arguments": chunk}}]}),
                    None,
                ));
            }
        }
    }

    pub fn final_json(&self) -> Value {
        let stats = self.stats.as_ref().expect("completion stats are recorded");
        let finish = self.finish_reason(stats);
        let mut message = if self.decision.was_tool_branch() {
            let tool_calls: Vec<Value> = self
                .decision
                .calls
                .iter()
                .map(|c| {
                    json!({
                        "id": next_id("call_"),
                        "type": "function",
                        "function": {"name": c.name.clone(), "arguments": c.args.clone()}
                    })
                })
                .collect();
            json!({"role": "assistant", "content": null, "tool_calls": tool_calls})
        } else {
            json!({"role": "assistant", "content": self.decision.text})
        };
        if self.decision.has_reasoning() {
            message["reasoning_content"] = json!(self.decision.reasoning_text);
        }
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": self.usage(stats),
        })
    }
}

impl StreamSink for ChatSink {
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
        if self.usage_events {
            frames.push(SseFrame {
                event: None,
                data: json!({
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": self.usage(stats),
                })
                .to_string(),
            });
        }
        let finish = self.finish_reason(stats);
        frames.push(self.chunk(json!({}), Some(finish)));
        frames.push(SseFrame {
            event: None,
            data: "[DONE]".to_string(),
        });
        Ok(frames)
    }

    fn on_failed(&mut self, msg: &str) -> Vec<SseFrame> {
        vec![SseFrame {
            event: None,
            data: json!({"error": {"message": msg, "type": "server_error"}}).to_string(),
        }]
    }
}
