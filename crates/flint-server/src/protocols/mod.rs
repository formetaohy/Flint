pub mod anthropic;
pub mod gemini;
pub mod openai_chat;
pub mod openai_responses;

use std::io::{self, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::time::{SystemTime, UNIX_EPOCH};

use flint_error::Result;
use flint_generate::GenStats;
use serde_json::Value;
use tiny_http::{Header, Response};

use crate::engine_hub::Event;
use crate::tools::{Call, DecisionParser, Part};

pub type Chat = (String, Vec<(String, String)>, String);

pub fn json_response(value: Value) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(value.to_string())
        .with_header(
            Header::from_bytes(b"Content-Type", b"application/json").expect("valid header"),
        )
        .with_header(
            Header::from_bytes(b"Access-Control-Allow-Origin", b"*").expect("valid header"),
        )
}

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

pub trait StreamSink: Send {
    fn on_delta(&mut self, text: &str, out: &mut Vec<u8>) -> Result<()>;
    fn on_done(&mut self, stats: &GenStats, out: &mut Vec<u8>);
    fn on_failed(&mut self, msg: &str, out: &mut Vec<u8>);
}

pub struct SseReader {
    rx: Receiver<Event>,
    sink: Box<dyn StreamSink>,
    buf: Vec<u8>,
    pos: usize,
    closed: bool,
}

impl SseReader {
    pub fn new(rx: Receiver<Event>, sink: Box<dyn StreamSink>) -> Self {
        Self {
            rx,
            sink,
            buf: Vec::new(),
            pos: 0,
            closed: false,
        }
    }
}

impl Read for SseReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.pos >= self.buf.len() && !self.closed {
            self.buf.clear();
            self.pos = 0;
            match self.rx.recv() {
                Ok(Event::Piece(text)) => {
                    if let Err(e) = self.sink.on_delta(&text, &mut self.buf) {
                        self.sink.on_failed(&e.to_string(), &mut self.buf);
                        self.closed = true;
                    }
                }
                Ok(Event::Done(stats)) => {
                    self.sink.on_done(&stats, &mut self.buf);
                    self.closed = true;
                }
                Ok(Event::Failed(e)) => {
                    self.sink.on_failed(&e, &mut self.buf);
                    self.closed = true;
                }
                Ok(Event::Started(_)) => {}
                Err(_) => self.closed = true,
            }
        }
        if self.pos < self.buf.len() {
            let n = (self.buf.len() - self.pos).min(out.len());
            out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
            self.pos += n;
            return Ok(n);
        }
        Ok(0)
    }
}

pub struct DecisionSink {
    parser: Option<DecisionParser>,
    pub text: String,
    pub calls: Vec<StreamCall>,
}

pub struct StreamCall {
    pub name: String,
    pub args: String,
}

impl DecisionSink {
    pub fn plain() -> Self {
        Self {
            parser: None,
            text: String::new(),
            calls: Vec::new(),
        }
    }

    pub fn constrained() -> Self {
        Self {
            parser: Some(DecisionParser::new()),
            text: String::new(),
            calls: Vec::new(),
        }
    }

    pub fn push(&mut self, piece: &str) -> Result<Vec<Part>> {
        match &mut self.parser {
            Some(p) => p.push(piece),
            None => Ok(vec![Part::Text(piece.to_string())]),
        }
    }

    pub fn route(&mut self, parts: Vec<Part>) {
        for part in parts {
            match part {
                Part::Text(chunk) => self.text.push_str(&chunk),
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
        self.parser
            .as_ref()
            .is_some_and(DecisionParser::was_tool_branch)
    }

    pub fn is_constrained(&self) -> bool {
        self.parser.is_some()
    }

    pub fn parsed_calls(&self) -> Option<Result<Vec<Call>>> {
        self.parser.as_ref().and_then(DecisionParser::tool_calls)
    }
}

pub fn sse_event(out: &mut Vec<u8>, event: &str, data: &str) {
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(event.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    out.extend_from_slice(data.as_bytes());
    out.extend_from_slice(b"\n\n");
}

pub fn sse_data(out: &mut Vec<u8>, data: &str) {
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(data.as_bytes());
    out.extend_from_slice(b"\n\n");
}
