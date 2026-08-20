use flint_error::{Error, Result};
use flint_generate::GenStats;
use serde_json::{Value, json};
use tiny_http::{Request, Response};

use crate::engine_hub::Event;
use crate::hub::{GenerateRequest, Hub};
use crate::protocols::{
    Chat, DecisionSink, SseReader, StreamSink, json_response, length_hit, next_id, now_secs,
    sse_event,
};
use crate::tools::render_tool_call;

pub struct Parsed {
    pub req: GenerateRequest,
    pub stream: bool,
    pub model: String,
}

pub fn handle(mut request: Request, hub: &Hub) -> Result<()> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;
    let body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            let _ = request.respond(
                json_response(json!({"error": {"message": format!("invalid JSON body: {e}"), "type": "invalid_request_error"}})),
            );
            return Ok(());
        }
    };
    let parsed = match parse(&body, hub) {
        Ok(p) => p,
        Err(e) => {
            let _ = request.respond(json_response(
                json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}}),
            ));
            return Ok(());
        }
    };
    let client = match hub.generate(&parsed.req) {
        Ok(c) => c,
        Err(e) => {
            let _ = request.respond(json_response(
                json!({"error": {"message": e.to_string(), "type": "server_error"}}),
            ));
            return Ok(());
        }
    };
    let sink = ResponsesSink::new(
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
                    let _ = request.respond(json_response(
                        json!({"error": {"message": e.to_string(), "type": "server_error"}}),
                    ));
                    return Ok(());
                }
            }
            Ok(Event::Done(stats)) => {
                sink.on_done(&stats, &mut scratch);
                break;
            }
            Ok(Event::Failed(e)) => {
                let _ = request.respond(json_response(
                    json!({"error": {"message": e, "type": "server_error"}}),
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
    let max_tokens = body["max_output_tokens"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or_else(|| hub.default_max_tokens());
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
    let schema = body
        .get("text")
        .and_then(|t| t.get("format"))
        .and_then(|f| f.get("schema"))
        .cloned();
    let (system, history, user) = extract_input(body)?;
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
            sampling,
            schema,
            tools,
            tool_choice,
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
                        history.push((pending, content));
                    }
                    other => eprintln!("[server] ignoring response message role {other:?}"),
                }
            }
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
    item_id: String,
    created: u64,
    model: String,
    max_tokens: usize,
    decision: DecisionSink,
    seq: u64,
    started: bool,
    call_ids: Vec<String>,
    stats: Option<GenStats>,
}

impl ResponsesSink {
    pub fn new(model: String, max_tokens: usize, constrained: bool) -> Self {
        Self {
            id: next_id("resp_"),
            item_id: next_id("msg_"),
            created: now_secs(),
            model,
            max_tokens,
            decision: if constrained {
                DecisionSink::constrained()
            } else {
                DecisionSink::plain()
            },
            seq: 0,
            started: false,
            call_ids: Vec::new(),
            stats: None,
        }
    }

    fn event(&mut self, out: &mut Vec<u8>, event: &str, payload: Value) {
        let mut data = payload;
        data["type"] = json!(event);
        data["sequence_number"] = json!(self.seq);
        self.seq += 1;
        sse_event(out, event, &data.to_string());
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
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "user": null,
            "metadata": {},
        })
    }

    fn output_items(&self) -> Vec<Value> {
        if self.decision.was_tool_branch() {
            let content: Vec<Value> = self
                .decision
                .calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    json!({
                        "type": "function_call",
                        "id": next_id("fc_"),
                        "call_id": self.call_ids.get(i).cloned().unwrap_or_else(|| next_id("call_")),
                        "name": c.name.clone(),
                        "arguments": c.args.clone(),
                        "status": "completed",
                    })
                })
                .collect();
            vec![json!({
                "id": self.item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": content,
            })]
        } else {
            vec![json!({
                "id": self.item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": self.decision.text,
                }],
            })]
        }
    }

    fn status(&self) -> &'static str {
        match self.stats {
            Some(stats) if length_hit(&stats, self.max_tokens) => "incomplete",
            _ => "completed",
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<u8>) {
        if self.started {
            return;
        }
        self.started = true;
        self.event(
            out,
            "response.created",
            json!({"response": self.response("in_progress", vec![])}),
        );
        self.event(
            out,
            "response.in_progress",
            json!({"response": self.response("in_progress", vec![])}),
        );
        self.event(
            out,
            "response.output_item.added",
            json!({"output_index": 0, "item": {"id": self.item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}}),
        );
        if self.decision.was_tool_branch() {
            return;
        }
        self.event(
            out,
            "response.content_part.added",
            json!({"item_id": self.item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
        );
    }

    pub fn final_json(&self) -> Value {
        self.response(self.status(), self.output_items())
    }
}

impl StreamSink for ResponsesSink {
    fn on_delta(&mut self, text: &str, out: &mut Vec<u8>) -> Result<()> {
        let parts = self.decision.push(text)?;
        for part in &parts {
            match part {
                crate::tools::Part::Text(chunk) => {
                    self.ensure_started(out);
                    self.event(
                        out,
                        "response.output_text.delta",
                        json!({"item_id": self.item_id, "output_index": 0, "content_index": 0, "delta": chunk, "logprobs": []}),
                    );
                }
                crate::tools::Part::CallStart { index, name } => {
                    self.ensure_started(out);
                    let call_id = next_id("call_");
                    while self.call_ids.len() < *index {
                        self.call_ids.push(String::new());
                    }
                    self.call_ids[*index - 1] = call_id.clone();
                    self.event(
                        out,
                        "response.content_part.added",
                        json!({"item_id": self.item_id, "output_index": 0, "content_index": index - 1, "part": {"type": "function_call", "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"}}),
                    );
                }
                crate::tools::Part::CallArgs { chunk, .. } => {
                    self.event(
                        out,
                        "response.function_call_arguments.delta",
                        json!({"item_id": self.item_id, "output_index": 0, "delta": chunk}),
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
        let items = self.output_items();
        let item = &items[0];
        if self.decision.was_tool_branch() {
            let done_parts: Vec<(usize, String, String, Value)> = self
                .decision
                .calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let call_id = self
                        .call_ids
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| next_id("call_"));
                    (i, call_id, c.name.clone(), item["content"][i].clone())
                })
                .collect();
            for (i, call_id, name, part) in done_parts {
                self.event(
                    out,
                    "response.function_call_arguments.done",
                    json!({"item_id": self.item_id, "output_index": 0, "arguments": part["arguments"], "name": name, "call_id": call_id}),
                );
                self.event(
                    out,
                    "response.content_part.done",
                    json!({"item_id": self.item_id, "output_index": 0, "content_index": i, "part": part}),
                );
            }
        } else {
            self.event(
                out,
                "response.output_text.done",
                json!({"item_id": self.item_id, "output_index": 0, "content_index": 0, "text": self.decision.text, "logprobs": []}),
            );
            self.event(
                out,
                "response.content_part.done",
                json!({"item_id": self.item_id, "output_index": 0, "content_index": 0, "part": item["content"][0]}),
            );
        }
        self.event(
            out,
            "response.output_item.done",
            json!({"output_index": 0, "item": item}),
        );
        let completed = self.response("completed", items);
        self.event(out, "response.completed", json!({"response": completed}));
        sse_event(out, "response.done", "[DONE]");
    }

    fn on_failed(&mut self, msg: &str, out: &mut Vec<u8>) {
        sse_event(
            out,
            "error",
            &json!({"type": "error", "code": "server_error", "message": msg}).to_string(),
        );
    }
}
