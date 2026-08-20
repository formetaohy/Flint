use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use flint_architectures::chat::ThinkMode;
use flint_error::{Error, Result};
use flint_generate::{GenStats, Piece, SamplingParams};
use serde_json::{Value, json};

use crate::generator::{GenerateRequest, RequestDefaults, ToolChoice};
use crate::protocols::{
    Chat, DecisionSink, Part, SseFrame, StreamSink, collect, json_response, length_hit,
    stream_response,
};
use crate::server::AppState;
use crate::tools::{Tool, render_tool_call};

pub async fn handle(State(state): State<AppState>, body: Bytes, stream: bool) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                "INVALID_ARGUMENT",
                format!("invalid JSON body: {e}"),
            );
        }
    };
    let parsed = match parse(&body, &state.generator.defaults()) {
        Ok(p) => p,
        Err(e) => {
            return error(StatusCode::BAD_REQUEST, "INVALID_ARGUMENT", e.to_string());
        }
    };
    let generation = match state.generator.generate(&parsed.req).await {
        Ok(g) => g,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL", e.to_string()),
    };
    let sink = GeminiSink::new(
        parsed.model.clone(),
        parsed.req.max_tokens,
        generation.think,
        parsed.req.tool_wrapper(),
    );
    if stream {
        return stream_response(generation.client, sink).into_response();
    }
    let sink = match collect(generation.client, sink).await {
        Ok(s) => s,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL", e.to_string()),
    };
    json_response(sink.final_json()).into_response()
}

fn error(status: StatusCode, kind: &str, message: String) -> Response {
    (
        status,
        json_response(json!({"error": {"code": status.as_u16(), "message": message, "status": kind}})),
    )
        .into_response()
}

pub fn parse(body: &Value, defaults: &RequestDefaults) -> Result<Parsed> {
    let config = &body["generationConfig"];
    let max_tokens = config["maxOutputTokens"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(defaults.max_tokens);
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
    let thinking = !matches!(
        config["thinkingConfig"]["thinkingBudget"].as_u64(),
        Some(0)
    );
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
        model: defaults.model_id.clone(),
    })
}

fn extract_contents(body: &Value) -> Result<Chat> {
    let mut system = String::new();
    if let Some(parts) = body["systemInstruction"]["parts"].as_array() {
        for p in parts {
            if p.get("thought").and_then(Value::as_bool) == Some(true) {
                continue;
            }
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
            .filter(|p| p.get("thought").and_then(Value::as_bool) != Some(true))
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
            .filter(|p| p["functionResponse"].is_object())
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
    pub fn new(model: String, max_tokens: usize, think: ThinkMode, constrained: bool) -> Self {
        Self {
            model,
            max_tokens,
            decision: if constrained {
                DecisionSink::constrained()
            } else {
                DecisionSink::plain(think)
            },
            stats: None,
        }
    }

    fn parts(&self) -> Vec<Value> {
        let mut parts: Vec<Value> = Vec::new();
        if self.decision.has_reasoning() {
            parts.push(json!({"text": self.decision.reasoning_text, "thought": true}));
        }
        if self.decision.was_tool_branch() {
            for c in &self.decision.calls {
                let args: Value =
                    serde_json::from_str(&c.args).unwrap_or_else(|_| json!({}));
                parts.push(json!({"functionCall": {"name": c.name.clone(), "args": args}}));
            }
        } else if !self.decision.text.is_empty() || parts.is_empty() {
            parts.push(json!({"text": self.decision.text}));
        }
        parts
    }

    fn candidate(&self, finish: Option<&str>) -> Value {
        let mut cand = json!({
            "content": {"parts": self.parts(), "role": "model"},
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
            "thoughtsTokenCount": self.decision.reasoning_tokens,
        })
    }

    fn finish_reason(&self, stats: &GenStats) -> &'static str {
        if length_hit(stats, self.max_tokens) {
            "MAX_TOKENS"
        } else {
            "STOP"
        }
    }

    fn response_json(&self, finish: Option<&str>) -> Value {
        let stats = self.stats.unwrap_or(GenStats {
            prefill_tokens: 0,
            decode_tokens: 0,
            accepted: 0,
            prefill_secs: 0.0,
            decode_secs: 0.0,
        });
        json!({
            "candidates": [self.candidate(finish)],
            "usageMetadata": self.usage(&stats),
            "modelVersion": self.model,
        })
    }

    pub fn final_json(&self) -> Value {
        let stats = self.stats.expect("completion stats are recorded");
        self.response_json(Some(self.finish_reason(&stats)))
    }
}

impl StreamSink for GeminiSink {
    fn on_delta(&mut self, piece: &Piece) -> Result<Vec<SseFrame>> {
        let parts = self.decision.push(piece)?;
        let mut frames = Vec::new();
        for part in &parts {
            match part {
                Part::Text(chunk) => {
                    frames.push(SseFrame {
                        event: None,
                        data: json!({
                            "candidates": [{
                                "content": {"parts": [{"text": chunk}], "role": "model"},
                                "index": 0,
                            }],
                        })
                        .to_string(),
                    });
                }
                Part::Reasoning(chunk) => {
                    frames.push(SseFrame {
                        event: None,
                        data: json!({
                            "candidates": [{
                                "content": {"parts": [{"text": chunk, "thought": true}], "role": "model"},
                                "index": 0,
                            }],
                        })
                        .to_string(),
                    });
                }
                Part::CallStart { .. } | Part::CallArgs { .. } => {}
            }
        }
        self.decision.route(parts);
        Ok(frames)
    }

    fn on_done(&mut self, stats: &GenStats) -> Result<Vec<SseFrame>> {
        self.stats = Some(*stats);
        let parts = self.decision.finish()?;
        self.decision.route(parts);
        let finish = self.finish_reason(stats);
        Ok(vec![SseFrame {
            event: None,
            data: self.response_json(Some(finish)).to_string(),
        }])
    }

    fn on_failed(&mut self, msg: &str) -> Vec<SseFrame> {
        vec![SseFrame {
            event: None,
            data: json!({"error": {"code": 500, "message": msg, "status": "INTERNAL"}}).to_string(),
        }]
    }
}
