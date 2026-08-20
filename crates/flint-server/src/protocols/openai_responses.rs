use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use flint_architectures::chat::ThinkMode;
use flint_error::{Error, Result};
use flint_generate::{GenStats, Piece};
use serde_json::{Value, json};

use crate::hub::{GenerateRequest, RequestDefaults};
use crate::protocols::{
    Chat, DecisionSink, Part, SseFrame, StreamSink, collect, json_response, length_hit, next_id,
    now_secs, split_reasoning, stream_response,
};
use crate::server::AppState;
use crate::tools::render_tool_call;

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
    let parsed = match parse(&body, &state.hub.defaults()) {
        Ok(p) => p,
        Err(e) => {
            return error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string());
        }
    };
    let generation = match state.hub.generate(&parsed.req).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    };
    let sink = ResponsesSink::new(
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

pub fn parse(body: &Value, defaults: &RequestDefaults) -> Result<Parsed> {
    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_output_tokens"]
        .as_u64()
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
    let sampling = crate::protocols::openai_chat::parse_sampling(body);
    let tools = crate::protocols::openai_chat::parse_tools(body);
    let tool_choice = crate::protocols::openai_chat::parse_tool_choice(body);
    let thinking = !matches!(body["reasoning"]["effort"].as_str(), Some("none"));
    let schema = body
        .get("text")
        .and_then(|t| t.get("format"))
        .and_then(|f| f.get("schema"))
        .cloned();
    let (system, history, user) = extract_input(body)?;
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
        model,
    })
}

fn extract_input(body: &Value) -> Result<Chat> {
    let mut system = String::new();
    if let Some(i) = body["instructions"].as_str() {
        system.push_str(i);
    }
    let mut history: Vec<(String, String)> = Vec::new();
    let mut user = String::new();
    let input = &body["input"];
    let items: Vec<&Value> = match input {
        Value::String(s) => return Ok((system, history, s.clone())),
        Value::Array(a) => a.iter().collect(),
        _ => return Err(Error::Config("input must be a string or an array".into())),
    };
    for item in items {
        let kind = item["type"].as_str().unwrap_or_default();
        match kind {
            "message" => {
                let role = item["role"].as_str().unwrap_or_default();
                let content = content_text(&item["content"]).unwrap_or_default();
                match role {
                    "system" | "developer" => {
                        if !system.is_empty() {
                            system.push_str("\n\n");
                        }
                        system.push_str(&content);
                    }
                    "user" => {
                        if !user.is_empty() {
                            user.push('\n');
                        }
                        user.push_str(&content);
                    }
                    "assistant" => {
                        let pending = std::mem::take(&mut user);
                        history.push((pending, split_reasoning(&content)));
                    }
                    other => eprintln!("[server] ignoring response message role {other:?}"),
                }
            }
            "reasoning" => {}
            "function_call" => {
                let name = item["name"].as_str().unwrap_or_default();
                let arguments: Value = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|a| serde_json::from_str(a).ok())
                    .unwrap_or_else(|| json!({}));
                let pending = std::mem::take(&mut user);
                history.push((pending, render_tool_call(name, &arguments)));
            }
            "function_call_output" => {
                let id = item["call_id"].as_str().unwrap_or_default();
                let output = item["output"].as_str().unwrap_or_default();
                user.push_str(&format!("[tool result {id}]\n{output}"));
            }
            other => eprintln!("[server] ignoring response input item type {other:?}"),
        }
    }
    Ok((system, history, user))
}

fn content_text(content: &Value) -> Option<String> {
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

pub struct ResponsesSink {
    id: String,
    message_id: String,
    reasoning_id: String,
    created: u64,
    model: String,
    max_tokens: usize,
    decision: DecisionSink,
    seq: u64,
    started: bool,
    reasoning_item_added: bool,
    message_item_added: bool,
    calls: Vec<CallIds>,
    stats: Option<GenStats>,
}

struct CallIds {
    item_id: String,
    call_id: String,
}

impl Clone for CallIds {
    fn clone(&self) -> Self {
        Self {
            item_id: self.item_id.clone(),
            call_id: self.call_id.clone(),
        }
    }
}

impl ResponsesSink {
    pub fn new(model: String, max_tokens: usize, think: ThinkMode, constrained: bool) -> Self {
        Self {
            id: next_id("resp_"),
            message_id: next_id("msg_"),
            reasoning_id: next_id("rs_"),
            created: now_secs(),
            model,
            max_tokens,
            decision: if constrained {
                DecisionSink::constrained()
            } else {
                DecisionSink::plain(think)
            },
            seq: 0,
            started: false,
            reasoning_item_added: false,
            message_item_added: false,
            calls: Vec::new(),
            stats: None,
        }
    }

    fn event(&mut self, frames: &mut Vec<SseFrame>, event: &'static str, mut payload: Value) {
        payload["type"] = json!(event);
        payload["sequence_number"] = json!(self.seq);
        self.seq += 1;
        frames.push(SseFrame {
            event: Some(event),
            data: payload.to_string(),
        });
    }

    fn response(&self, status: &str, output: Vec<Value>) -> Value {
        let stats = self.stats.unwrap_or(GenStats {
            prefill_tokens: 0,
            decode_tokens: 0,
            accepted: 0,
            prefill_secs: 0.0,
            decode_secs: 0.0,
        });
        let incomplete = status == "incomplete";
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "background": false,
            "error": null,
            "incomplete_details": if incomplete { json!({"reason": "max_output_tokens"}) } else { Value::Null },
            "instructions": null,
            "max_output_tokens": self.max_tokens,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "reasoning": {"effort": null, "summary": null},
            "store": true,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": stats.prefill_tokens,
                "output_tokens": stats.decode_tokens,
                "total_tokens": stats.prefill_tokens + stats.decode_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": self.decision.reasoning_tokens},
            },
            "user": null,
            "metadata": {},
        })
    }

    fn reasoning_item(&self, reasoning: Option<&str>, status: Option<&str>) -> Value {
        let content = match reasoning {
            Some(r) => json!([{"type": "reasoning_text", "text": r}]),
            None => json!([{"type": "reasoning_text", "text": ""}]),
        };
        let mut item = json!({
            "id": self.reasoning_id,
            "type": "reasoning",
            "summary": [],
            "content": content,
        });
        if let Some(s) = status {
            item["status"] = json!(s);
        }
        item
    }

    fn message_item(&self, status: &str) -> Value {
        let content: Vec<Value> = if self.decision.was_tool_branch() {
            self.decision
                .calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let ids = self
                        .calls
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| CallIds {
                            item_id: next_id("fc_"),
                            call_id: next_id("call_"),
                        });
                    json!({
                        "type": "function_call",
                        "id": ids.item_id,
                        "call_id": ids.call_id,
                        "name": c.name.clone(),
                        "arguments": c.args.clone(),
                        "status": "completed",
                    })
                })
                .collect()
        } else {
            vec![json!({
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": self.decision.text,
            })]
        };
        json!({
            "id": self.message_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": content,
        })
    }

    fn output_items(&self) -> Vec<Value> {
        let mut items = Vec::new();
        if self.decision.has_reasoning() {
            items.push(self.reasoning_item(Some(&self.decision.reasoning_text), Some("completed")));
        }
        items.push(self.message_item(if self.status() == "incomplete" { "incomplete" } else { "completed" }));
        items
    }

    fn message_index(&self) -> usize {
        usize::from(self.decision.has_reasoning())
    }

    fn status(&self) -> &'static str {
        match self.stats {
            Some(stats) if length_hit(&stats, self.max_tokens) => "incomplete",
            _ => "completed",
        }
    }

    fn ensure_started(&mut self, frames: &mut Vec<SseFrame>) {
        if self.started {
            return;
        }
        self.started = true;
        self.event(
            frames,
            "response.created",
            json!({"response": self.response("in_progress", vec![])}),
        );
        self.event(
            frames,
            "response.in_progress",
            json!({"response": self.response("in_progress", vec![])}),
        );
    }

    fn ensure_reasoning_item(&mut self, frames: &mut Vec<SseFrame>) {
        self.ensure_started(frames);
        if self.reasoning_item_added {
            return;
        }
        self.reasoning_item_added = true;
        self.event(
            frames,
            "response.output_item.added",
            json!({"output_index": 0, "item": self.reasoning_item(None, Some("in_progress"))}),
        );
    }

    fn ensure_message_item(&mut self, frames: &mut Vec<SseFrame>) {
        self.ensure_started(frames);
        if self.message_item_added {
            return;
        }
        if !self.reasoning_item_added && self.decision.has_reasoning() {
            self.ensure_reasoning_item(frames);
        }
        self.message_item_added = true;
        self.event(
            frames,
            "response.output_item.added",
            json!({"output_index": self.message_index(), "item": self.message_item("in_progress")}),
        );
        if !self.decision.was_tool_branch() {
            self.event(
                frames,
                "response.content_part.added",
                json!({"item_id": self.message_id, "output_index": self.message_index(), "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
            );
        }
    }

    pub fn final_json(&self) -> Value {
        self.response(self.status(), self.output_items())
    }
}

impl StreamSink for ResponsesSink {
    fn on_delta(&mut self, piece: &Piece) -> Result<Vec<SseFrame>> {
        let parts = self.decision.push(piece)?;
        let mut frames = Vec::new();
        for part in &parts {
            match part {
                Part::Text(chunk) => {
                    self.ensure_message_item(&mut frames);
                    self.event(
                        &mut frames,
                        "response.output_text.delta",
                        json!({"item_id": self.message_id, "output_index": self.message_index(), "content_index": 0, "delta": chunk, "logprobs": []}),
                    );
                }
                Part::Reasoning(chunk) => {
                    self.ensure_reasoning_item(&mut frames);
                    self.event(
                        &mut frames,
                        "response.reasoning_text.delta",
                        json!({"item_id": self.reasoning_id, "output_index": 0, "content_index": 0, "delta": chunk}),
                    );
                }
                Part::CallStart { name, .. } => {
                    self.ensure_message_item(&mut frames);
                    let ids = CallIds {
                        item_id: next_id("fc_"),
                        call_id: next_id("call_"),
                    };
                    let index = self.calls.len();
                    self.calls.push(ids.clone());
                    self.event(
                        &mut frames,
                        "response.content_part.added",
                        json!({"item_id": self.message_id, "output_index": self.message_index(), "content_index": index, "part": {"type": "function_call", "id": ids.item_id, "call_id": ids.call_id, "name": name, "arguments": "", "status": "in_progress"}}),
                    );
                }
                Part::CallArgs { chunk, .. } => {
                    self.event(
                        &mut frames,
                        "response.function_call_arguments.delta",
                        json!({"item_id": self.message_id, "output_index": self.message_index(), "delta": chunk}),
                    );
                }
            }
        }
        self.decision.route(parts);
        Ok(frames)
    }

    fn on_done(&mut self, stats: &GenStats) -> Result<Vec<SseFrame>> {
        self.stats = Some(*stats);
        let parts = self.decision.finish()?;
        self.decision.route(parts);
        let mut frames = Vec::new();
        self.ensure_started(&mut frames);
        if self.decision.has_reasoning() {
            self.ensure_reasoning_item(&mut frames);
            self.event(
                &mut frames,
                "response.reasoning_text.done",
                json!({"item_id": self.reasoning_id, "output_index": 0, "content_index": 0, "text": self.decision.reasoning_text}),
            );
            self.event(
                &mut frames,
                "response.output_item.done",
                json!({"output_index": 0, "item": self.reasoning_item(Some(&self.decision.reasoning_text), Some("completed"))}),
            );
        }
        self.ensure_message_item(&mut frames);
        if self.decision.was_tool_branch() {
            let calls: Vec<(String, String)> = self
                .decision
                .calls
                .iter()
                .map(|c| (c.name.clone(), c.args.clone()))
                .collect();
            for (i, (name, args)) in calls.into_iter().enumerate() {
                let ids = self
                    .calls
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| CallIds {
                        item_id: next_id("fc_"),
                        call_id: next_id("call_"),
                    });
                self.event(
                    &mut frames,
                    "response.function_call_arguments.done",
                    json!({"item_id": self.message_id, "output_index": self.message_index(), "arguments": args, "name": name, "call_id": ids.call_id}),
                );
                let part = json!({
                    "type": "function_call",
                    "id": ids.item_id,
                    "call_id": ids.call_id,
                    "name": name,
                    "arguments": args,
                    "status": "completed",
                });
                self.event(
                    &mut frames,
                    "response.content_part.done",
                    json!({"item_id": self.message_id, "output_index": self.message_index(), "content_index": i, "part": part}),
                );
            }
        } else {
            self.event(
                &mut frames,
                "response.output_text.done",
                json!({"item_id": self.message_id, "output_index": self.message_index(), "content_index": 0, "text": self.decision.text, "logprobs": []}),
            );
            self.event(
                &mut frames,
                "response.content_part.done",
                json!({"item_id": self.message_id, "output_index": self.message_index(), "content_index": 0, "part": self.message_item("completed")["content"][0]}),
            );
        }
        let completed = self.response(self.status(), self.output_items());
        self.event(&mut frames, "response.output_item.done", json!({
            "output_index": self.message_index(),
            "item": self.message_item(if self.status() == "incomplete" { "incomplete" } else { "completed" }),
        }));
        self.event(&mut frames, "response.completed", json!({"response": completed}));
        frames.push(SseFrame {
            event: Some("response.done"),
            data: "[DONE]".to_string(),
        });
        Ok(frames)
    }

    fn on_failed(&mut self, msg: &str) -> Vec<SseFrame> {
        vec![SseFrame {
            event: Some("error"),
            data: json!({"type": "error", "code": "server_error", "message": msg}).to_string(),
        }]
    }
}
