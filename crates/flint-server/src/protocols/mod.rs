pub mod anthropic;
pub mod decision;
pub mod gemini;
pub mod openai_chat;
pub mod openai_responses;
pub mod reasoning;

use std::future::ready;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event as SseEvent, Sse};
use flint_error::{Error, Result};
use flint_generate::{GenStats, Piece};
use futures_util::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::engine_worker::{Client, CloseGuard, Event};
use crate::protocols::decision::DecisionParser;
use crate::protocols::reasoning::ReasoningParser;
use flint_architectures::chat::ThinkMode;

pub type Chat = (String, Vec<(String, String)>, String);

#[derive(Debug)]
pub enum Part {
    Text(String),
    Reasoning(String),
    CallStart { index: usize, name: String },
    CallArgs { index: usize, chunk: String },
}

pub struct StreamCall {
    pub name: String,
    pub args: String,
}

pub struct DecisionSink {
    decision: Option<DecisionParser>,
    reasoning: Option<ReasoningParser>,
    pub text: String,
    pub reasoning_text: String,
    pub reasoning_tokens: usize,
    pub calls: Vec<StreamCall>,
}

impl DecisionSink {
    pub fn plain(think: ThinkMode) -> Self {
        Self {
            decision: None,
            reasoning: Some(ReasoningParser::new(think)),
            text: String::new(),
            reasoning_text: String::new(),
            reasoning_tokens: 0,
            calls: Vec::new(),
        }
    }

    pub fn constrained() -> Self {
        Self {
            decision: Some(DecisionParser::new()),
            reasoning: None,
            text: String::new(),
            reasoning_text: String::new(),
            reasoning_tokens: 0,
            calls: Vec::new(),
        }
    }

    pub fn push(&mut self, piece: &Piece) -> Result<Vec<Part>> {
        match &mut self.reasoning {
            Some(parser) => {
                let (parts, thinking) = parser.push(&piece.text);
                if thinking {
                    self.reasoning_tokens += 1;
                }
                Ok(parts)
            }
            None => match &mut self.decision {
                Some(parser) => parser.push(&piece.text),
                None => Ok(vec![Part::Text(piece.text.clone())]),
            },
        }
    }

    pub fn finish(&mut self) -> Result<Vec<Part>> {
        match &mut self.reasoning {
            Some(parser) => Ok(parser.finish()),
            None => Ok(Vec::new()),
        }
    }

    pub fn route(&mut self, parts: Vec<Part>) {
        for part in parts {
            match part {
                Part::Text(chunk) => self.text.push_str(&chunk),
                Part::Reasoning(chunk) => self.reasoning_text.push_str(&chunk),
                Part::CallStart { index, name } => {
                    while self.calls.len() < index {
                        self.calls.push(StreamCall {
                            name: String::new(),
                            args: String::new(),
                        });
                    }
                    self.calls[index - 1].name = name;
                }
                Part::CallArgs { index, chunk } => {
                    self.calls[index - 1].args.push_str(&chunk);
                }
            }
        }
    }

    pub fn was_tool_branch(&self) -> bool {
        self.decision
            .as_ref()
            .is_some_and(DecisionParser::was_tool_branch)
    }

    pub fn has_reasoning(&self) -> bool {
        !self.reasoning_text.is_empty() || self.reasoning_tokens > 0
    }
}

pub struct SseFrame {
    pub event: Option<&'static str>,
    pub data: String,
}

pub trait StreamSink: Send {
    fn on_delta(&mut self, piece: &Piece) -> Result<Vec<SseFrame>>;
    fn on_done(&mut self, stats: &GenStats) -> Result<Vec<SseFrame>>;
    fn on_failed(&mut self, msg: &str) -> Vec<SseFrame>;
}

struct StreamState<S> {
    sink: S,
    _guard: CloseGuard,
}

pub fn stream_response<S>(client: Client, sink: S) -> Sse<impl futures_core::Stream<Item = std::result::Result<SseEvent, std::convert::Infallible>>>
where
    S: StreamSink + 'static,
{
    let (rx, guard) = client.into_parts();
    let state = StreamState { sink, _guard: guard };
    let stream = UnboundedReceiverStream::new(rx)
        .scan(state, |state, event| {
            let frames = match event {
                Event::Started(_) => Vec::new(),
                Event::Piece(piece) => state
                    .sink
                    .on_delta(&piece)
                    .unwrap_or_else(|e| state.sink.on_failed(&e.to_string())),
                Event::Done(stats) => state
                    .sink
                    .on_done(&stats)
                    .unwrap_or_else(|e| state.sink.on_failed(&e.to_string())),
                Event::Failed(e) => state.sink.on_failed(&e),
            };
            ready(Some(frames))
        })
        .flat_map(futures_util::stream::iter)
        .map(|frame| Ok(match frame.event {
            Some(name) => SseEvent::default().event(name).data(frame.data),
            None => SseEvent::default().data(frame.data),
        }));
    Sse::new(stream)
}

pub async fn collect<S>(mut client: Client, mut sink: S) -> Result<S>
where
    S: StreamSink,
{
    while let Some(event) = client.rx.recv().await {
        match event {
            Event::Started(_) => {}
            Event::Piece(piece) => {
                sink.on_delta(&piece)?;
            }
            Event::Done(stats) => {
                sink.on_done(&stats)?;
                return Ok(sink);
            }
            Event::Failed(e) => return Err(Error::Model(e)),
        }
    }
    Err(Error::Model("engine closed the stream".into()))
}

pub fn split_reasoning(content: &str) -> String {
    match content.rfind(CLOSE_TAG) {
        Some(i) => content[i + CLOSE_TAG.len()..]
            .trim_start_matches(['\n'])
            .to_string(),
        None => content.to_string(),
    }
}

const CLOSE_TAG: &str = "</think>";

static SEQ: AtomicU64 = AtomicU64::new(0);

pub fn next_id(prefix: &str) -> String {
    format!("{prefix}{}", SEQ.fetch_add(1, Ordering::Relaxed))
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn length_hit(stats: &GenStats, max_tokens: usize) -> bool {
    stats.decode_tokens >= max_tokens
}

pub fn json_response(value: serde_json::Value) -> axum::Json<serde_json::Value> {
    axum::Json(value)
}
