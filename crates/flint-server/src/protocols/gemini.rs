use flint_error::{Error, Result};
use flint_generate::{GenStats, SamplingParams};
use serde_json::{Value, json};
use tiny_http::{Request, Response};

use crate::engine_hub::Event;
use crate::hub::{GenerateRequest, Hub, ToolChoice};
use crate::protocols::{
    Chat, DecisionSink, SseReader, StreamSink, json_response, length_hit, sse_data,
};
use crate::tools::{Tool, render_tool_call};

pub fn handle(mut request: Request, hub: &Hub, stream: bool) -> Result<()> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;
    let body: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            let _ = request.respond(json_response(
                json!({"error": {"code": 400, "message": format!("invalid JSON body: {e}"), "status": "INVALID_ARGUMENT"}}),
            ));
            return Ok(());
        }
    };
    let parsed = match parse(&body, hub) {
        Ok(p) => p,
        Err(e) => {
            let _ = request.respond(json_response(
                json!({"error": {"code": 400, "message": e.to_string(), "status": "INVALID_ARGUMENT"}}),
            ));
            return Ok(());
        }
    };
    let client = match hub.generate(&parsed.req) {
        Ok(c) => c,
        Err(e) => {
            let _ = request.respond(json_response(
                json!({"error": {"code": 500, "message": e.to_string(), "status": "INTERNAL"}}),
            ));
            return Ok(());
        }
    };
    let sink = GeminiSink::new(
        parsed.model.clone(),
        parsed.req.max_tokens,
        parsed.req.tool_wrapper(),
    );
    if stream {
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
                        json!({"error": {"code": 500, "message": e.to_string(), "status": "INTERNAL"}}),
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
                    json!({"error": {"code": 500, "message": e, "status": "INTERNAL"}}),
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
    let config = &body["generationConfig"];
    let max_tokens = config["maxOutputTokens"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or_else(|| hub.default_max_tokens());
    let stop = config
        .get("stopSequences")
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
    if let Some(t) = config["temperature"].as_f64() {
        sampling.temperature = t as f32;
        any = true;
    }
    if let Some(t) = config["topP"].as_f64() {
        sampling.top_p = t as f32;
        any = true;
    }
    if let Some(t) = config["topK"].as_f64() {
        sampling.top_k = t as usize;
        any = true;
    }
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .flat_map(|t| {
                    t["functionDeclarations"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default()
                })
                .map(|t| Tool {
                    name: t["name"].as_str().unwrap_or_default().to_string(),
                    description: t["description"].as_str().unwrap_or_default().to_string(),
                    schema: t
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object"})),
                })
                .collect()
        })
        .unwrap_or_default();
    let tool_choice = match body["toolConfig"]["functionCallingConfig"]["mode"].as_str() {
        Some("NONE") => ToolChoice::None,
        Some("ANY") => ToolChoice::Required,
        Some("AUTO") => ToolChoice::Auto,
        _ => ToolChoice::Auto,
    };
    let (system, history, user) = extract_contents(body)?;
    let model = hub.model_id().to_string();
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
        model,
    })
}

fn extract_contents(body: &Value) -> Result<Chat> {
    let mut system = String::new();
    if let Some(parts) = body["systemInstruction"]["parts"].as_array() {
        for p in parts {
            if let Some(t) = p["text"].as_str() {
                system.push_str(t);
            }
        }
    }
    let contents = body
        .get("contents")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("contents must be an array".into()))?;
    let mut history: Vec<(String, String)> = Vec::new();
    let mut user = String::new();
    for c in contents {
        let role = c["role"].as_str().unwrap_or_default();
        let parts = c["parts"].as_array().cloned().unwrap_or_default();
        let text: Vec<String> = parts
            .iter()
            .filter_map(|p| p["text"].as_str().map(str::to_string))
            .collect();
        let calls: Vec<String> = parts
            .iter()
            .filter_map(|p| {
                let fc = &p["functionCall"];
                Some(render_tool_call(
                    fc["name"].as_str()?,
                    fc.get("args").unwrap_or(&json!({})),
                ))
            })
            .collect();
        let responses: Vec<String> = parts
            .iter()
            .map(|p| {
                let fr = &p["functionResponse"];
                format!(
                    "[tool result {}]\n{}",
                    fr["name"].as_str().unwrap_or_default(),
                    fr.get("response").unwrap_or(&json!({}))
                )
            })
            .collect();
        let content = [text, calls, responses].concat().join("\n");
        match role {
            "user" => {
                if !user.is_empty() {
                    user.push('\n');
                }
                user.push_str(&content);
            }
            "model" => {
                let pending = std::mem::take(&mut user);
                history.push((pending, content));
            }
            other => eprintln!("[server] ignoring gemini content role {other:?}"),
        }
    }
    if system.is_empty() && history.is_empty() && user.is_empty() {
        return Err(Error::Config("contents contain no prompt".into()));
    }
    Ok((system, history, user))
}

pub struct Parsed {
    pub req: GenerateRequest,
    pub model: String,
}

pub struct GeminiSink {
    model: String,
    max_tokens: usize,
    decision: DecisionSink,
    stats: Option<GenStats>,
}

impl GeminiSink {
    pub fn new(model: String, max_tokens: usize, constrained: bool) -> Self {
        Self {
            model,
            max_tokens,
            decision: if constrained {
                DecisionSink::constrained()
            } else {
                DecisionSink::plain()
            },
            stats: None,
        }
    }

    fn candidate(
        &self,
        text: Option<&str>,
        calls: Option<Vec<Value>>,
        finish: Option<&str>,
    ) -> Value {
        let parts: Vec<Value> = match calls {
            Some(calls) => calls
                .into_iter()
                .map(|c| {
                    let name = c["name"].as_str().unwrap_or_default().to_string();
                    let args: Value = serde_json::from_str(c["args"].as_str().unwrap_or("{}"))
                        .unwrap_or_else(|_| json!({}));
                    json!({"functionCall": {"name": name, "args": args}})
                })
                .collect(),
            None => vec![json!({"text": text.unwrap_or_default()})],
        };
        let mut cand = json!({
            "content": {"parts": parts, "role": "model"},
            "index": 0,
        });
        if let Some(f) = finish {
            cand["finishReason"] = json!(f);
        }
        cand
    }

    fn usage(&self, stats: &GenStats) -> Value {
        json!({
            "promptTokenCount": stats.prefill_tokens,
            "candidatesTokenCount": stats.decode_tokens,
            "totalTokenCount": stats.prefill_tokens + stats.decode_tokens,
        })
    }

    fn finish_reason(&self, stats: &GenStats) -> &'static str {
        if length_hit(stats, self.max_tokens) {
            "MAX_TOKENS"
        } else {
            "STOP"
        }
    }

    fn response_json(&self, text: Option<&str>, calls: Option<Vec<Value>>, finish: bool) -> Value {
        let stats = self.stats.unwrap_or(GenStats {
            prefill_tokens: 0,
            decode_tokens: 0,
            accepted: 0,
            prefill_secs: 0.0,
            decode_secs: 0.0,
        });
        json!({
            "candidates": [self.candidate(text, calls, finish.then(|| self.finish_reason(&stats)))],
            "usageMetadata": self.usage(&stats),
            "modelVersion": self.model,
        })
    }

    pub fn final_json(&self) -> Value {
        let calls = if self.decision.was_tool_branch() {
            Some(
                self.decision
                    .calls
                    .iter()
                    .map(|c| json!({"name": c.name.clone(), "args": c.args.clone()}))
                    .collect(),
            )
        } else {
            None
        };
        self.response_json(Some(self.decision.text.as_str()), calls, true)
    }
}

impl StreamSink for GeminiSink {
    fn on_delta(&mut self, text: &str, out: &mut Vec<u8>) -> Result<()> {
        let parts = self.decision.push(text)?;
        let mut text_delta = String::new();
        for part in &parts {
            if let crate::tools::Part::Text(chunk) = part {
                text_delta.push_str(chunk);
            }
        }
        if !text_delta.is_empty() {
            sse_data(
                out,
                &json!({
                    "candidates": [{
                        "content": {"parts": [{"text": text_delta}], "role": "model"},
                        "index": 0,
                    }],
                })
                .to_string(),
            );
        }
        self.decision.route(parts);
        Ok(())
    }

    fn on_done(&mut self, stats: &GenStats, out: &mut Vec<u8>) {
        self.stats = Some(*stats);
        let calls = if self.decision.was_tool_branch() {
            Some(
                self.decision
                    .calls
                    .iter()
                    .map(|c| json!({"name": c.name.clone(), "args": c.args.clone()}))
                    .collect(),
            )
        } else {
            None
        };
        sse_data(
            out,
            &self
                .response_json(Some(self.decision.text.as_str()), calls, true)
                .to_string(),
        );
    }

    fn on_failed(&mut self, msg: &str, out: &mut Vec<u8>) {
        sse_data(
            out,
            &json!({"error": {"code": 500, "message": msg, "status": "INTERNAL"}}).to_string(),
        );
    }
}
