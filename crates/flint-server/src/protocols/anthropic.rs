use flint_error::{Error, Result};
use flint_generate::{GenStats, SamplingParams};
use serde_json::{Value, json};
use tiny_http::{Request, Response};

use crate::engine_hub::Event;
use crate::hub::{GenerateRequest, Hub, ToolChoice};
use crate::protocols::{
    Chat, DecisionSink, SseReader, StreamSink, json_response, length_hit, next_id, sse_event,
};
use crate::tools::{Tool, render_tool_call};

pub struct Parsed {
    pub req: GenerateRequest,
    pub stream: bool,
    pub model: String,
}

pub fn handle_count_tokens(mut request: Request, hub: &Hub) -> Result<()> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;
    let body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            let _ = request.respond(json_response(json!({"type": "error", "error": {"type": "invalid_request_error", "message": format!("invalid JSON body: {e}")}})));
            return Ok(());
        }
    };
    let parsed = match parse(&body, hub) {
        Ok(p) => p,
        Err(e) => {
            let _ = request.respond(json_response(json!({"type": "error", "error": {"type": "invalid_request_error", "message": e.to_string()}})));
            return Ok(());
        }
    };
    match hub.count_tokens(&parsed.req) {
        Ok(n) => {
            let _ = request.respond(json_response(json!({"input_tokens": n})));
        }
        Err(e) => {
            let _ = request.respond(json_response(
                json!({"type": "error", "error": {"type": "api_error", "message": e.to_string()}}),
            ));
        }
    }
    Ok(())
}

pub fn handle(mut request: Request, hub: &Hub) -> Result<()> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;
    let body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            let _ = request.respond(json_response(json!({"type": "error", "error": {"type": "invalid_request_error", "message": format!("invalid JSON body: {e}")}})));
            return Ok(());
        }
    };
    let parsed = match parse(&body, hub) {
        Ok(p) => p,
        Err(e) => {
            let _ = request.respond(json_response(json!({"type": "error", "error": {"type": "invalid_request_error", "message": e.to_string()}})));
            return Ok(());
        }
    };
    let client = match hub.generate(&parsed.req) {
        Ok(c) => c,
        Err(e) => {
            let _ = request.respond(json_response(
                json!({"type": "error", "error": {"type": "api_error", "message": e.to_string()}}),
            ));
            return Ok(());
        }
    };
    let sink = MessageSink::new(
        parsed.model.clone(),
        parsed.req.max_tokens,
        parsed.req.tool_wrapper(),
    );
    if parsed.stream {
        let reader = SseReader::new(client.rx, Box::new(sink));
        let response = Response::new(
            tiny_http::StatusCode(200),
            crate::server::sse_headers(),
            Box::new(reader),
            None,
            None,
        );
        request.respond(response)?;
        return Ok(());
    }
    let rx = client.rx;
    let mut sink = sink;
    let mut scratch = Vec::new();
    loop {
        match rx.recv() {
            Ok(Event::Piece(text)) => {
                if let Err(e) = sink.on_delta(&text, &mut scratch) {
                    let _ = request.respond(json_response(json!({"type": "error", "error": {"type": "api_error", "message": e.to_string()}})));
                    return Ok(());
                }
            }
            Ok(Event::Done(stats)) => {
                sink.on_done(&stats, &mut scratch);
                break;
            }
            Ok(Event::Failed(e)) => {
                let _ = request.respond(json_response(
                    json!({"type": "error", "error": {"type": "api_error", "message": e}}),
                ));
                return Ok(());
            }
            _ => break,
        }
    }
    let _ = request.respond(json_response(sink.final_json()));
    Ok(())
}

pub fn parse(body: &Value, hub: &Hub) -> Result<Parsed> {
    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_tokens"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or_else(|| hub.default_max_tokens());
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
        .unwrap_or_else(|| hub.model_id().to_string());
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
            Value::String(s) => Some(s.clone()),
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
                    parts.push(t.to_string());
                }
            }
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
    block: usize,
    stats: Option<GenStats>,
}

impl MessageSink {
    pub fn new(model: String, max_tokens: usize, constrained: bool) -> Self {
        Self {
            id: next_id("msg_"),
            model,
            max_tokens,
            decision: if constrained {
                DecisionSink::constrained()
            } else {
                DecisionSink::plain()
            },
            started: false,
            block: 0,
            stats: None,
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<u8>) {
        if self.started {
            return;
        }
        self.started = true;
        sse_event(
            out,
            "message_start",
            &json!({"type": "message_start", "message": {
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
        );
        if !self.decision.was_tool_branch() {
            sse_event(
                out,
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                    .to_string(),
            );
        }
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
            vec![json!({"type": "text", "text": self.decision.text})]
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
    fn on_delta(&mut self, text: &str, out: &mut Vec<u8>) -> Result<()> {
        let parts = self.decision.push(text)?;
        for part in &parts {
            match part {
                crate::tools::Part::Text(chunk) => {
                    self.ensure_started(out);
                    sse_event(
                        out,
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk}})
                            .to_string(),
                    );
                }
                crate::tools::Part::CallStart { index, name } => {
                    self.ensure_started(out);
                    if *index != 1 {
                        sse_event(
                            out,
                            "content_block_stop",
                            &json!({"type": "content_block_stop", "index": *index - 2}).to_string(),
                        );
                    }
                    sse_event(
                        out,
                        "content_block_start",
                        &json!({"type": "content_block_start", "index": index - 1, "content_block": {"type": "tool_use", "id": next_id("toolu_"), "name": name, "input": {}}})
                            .to_string(),
                    );
                    self.block = *index;
                }
                crate::tools::Part::CallArgs { index, chunk } => {
                    let _ = index;
                    sse_event(
                        out,
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": self.block - 1, "delta": {"type": "input_json_delta", "partial_json": chunk}})
                            .to_string(),
                    );
                }
            }
        }
        self.decision.route(parts);
        Ok(())
    }

    fn on_done(&mut self, stats: &GenStats, out: &mut Vec<u8>) {
        self.stats = Some(*stats);
        self.ensure_started(out);
        let last_block = if self.decision.was_tool_branch() {
            self.decision.calls.len()
        } else {
            1
        };
        sse_event(
            out,
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": last_block - 1}).to_string(),
        );
        let stop_reason = self.stop_reason(stats);
        sse_event(
            out,
            "message_delta",
            &json!({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": null}, "usage": {"output_tokens": stats.decode_tokens}})
                .to_string(),
        );
        sse_event(
            out,
            "message_stop",
            &json!({"type": "message_stop"}).to_string(),
        );
    }

    fn on_failed(&mut self, msg: &str, out: &mut Vec<u8>) {
        sse_event(
            out,
            "error",
            &json!({"type": "error", "error": {"type": "api_error", "message": msg}}).to_string(),
        );
    }
}
