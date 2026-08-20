use flint_architectures::chat::ThinkMode;

use crate::protocols::Part;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
const NL: [char; 2] = ['\n', '\r'];

enum State {
    Watch,
    PostOpen,
    Think,
    Post,
    Text,
}

pub struct ReasoningParser {
    state: State,
    buf: String,
    ws_hold: String,
}

impl ReasoningParser {
    pub fn new(mode: ThinkMode) -> Self {
        let state = match mode {
            ThinkMode::None => State::Text,
            ThinkMode::Emitted => State::Watch,
            ThinkMode::Preopened => State::Think,
        };
        Self {
            state,
            buf: String::new(),
            ws_hold: String::new(),
        }
    }

    pub fn push(&mut self, text: &str) -> (Vec<Part>, bool) {
        let mut parts = Vec::new();
        let mut thinking = false;
        match self.state {
            State::Text => {
                if !text.is_empty() {
                    parts.push(Part::Text(text.to_string()));
                }
            }
            State::Post => self.post_push(text, &mut parts),
            State::PostOpen => {
                thinking = true;
                self.post_open_push(text, &mut parts);
            }
            State::Think => {
                thinking = true;
                self.think_push(text, &mut parts);
            }
            State::Watch => self.watch_push(text, &mut parts, &mut thinking),
        }
        (parts, thinking)
    }

    pub fn finish(&mut self) -> Vec<Part> {
        let mut parts = Vec::new();
        match self.state {
            State::Think => {
                let mut rest = String::new();
                rest.push_str(&self.ws_hold);
                rest.push_str(&self.buf);
                let trimmed = rest.trim_end_matches(NL);
                if !trimmed.is_empty() {
                    parts.push(Part::Reasoning(trimmed.to_string()));
                }
                self.state = State::Text;
            }
            State::Watch | State::Post | State::PostOpen => {
                if !self.buf.is_empty() {
                    parts.push(Part::Text(std::mem::take(&mut self.buf)));
                }
                self.state = State::Text;
            }
            State::Text => {}
        }
        self.ws_hold.clear();
        parts
    }

    fn think_push(&mut self, text: &str, parts: &mut Vec<Part>) {
        self.buf.push_str(&std::mem::take(&mut self.ws_hold));
        self.buf.push_str(text);
        let mut scan = 0;
        loop {
            let Some(rel) = self.buf[scan..].find('<') else {
                let chunk = std::mem::take(&mut self.buf);
                self.emit_reasoning(&chunk, parts);
                return;
            };
            let i = scan + rel;
            let cand = &self.buf[i..];
            if let Some(rest) = cand.strip_prefix(CLOSE) {
                let chunk = self.buf[..i].to_string();
                let rest = rest.to_string();
                self.emit_reasoning(&chunk, parts);
                self.buf.clear();
                self.state = State::Post;
                if !rest.is_empty() {
                    self.post_push(&rest, parts);
                }
                return;
            }
            if CLOSE.starts_with(cand) {
                let chunk = self.buf[..i].to_string();
                let held = cand.to_string();
                self.emit_reasoning(&chunk, parts);
                self.buf = held;
                return;
            }
            scan = i + 1;
        }
    }

    fn emit_reasoning(&mut self, chunk: &str, parts: &mut Vec<Part>) {
        let trimmed = chunk.trim_end_matches(NL);
        if !trimmed.is_empty() {
            parts.push(Part::Reasoning(trimmed.to_string()));
        }
        self.ws_hold = chunk[trimmed.len()..].to_string();
    }

    fn post_open_push(&mut self, text: &str, parts: &mut Vec<Part>) {
        let start = text.len() - text.trim_start_matches(NL).len();
        self.state = State::Think;
        if start < text.len() {
            self.think_push(&text[start..], parts);
        }
    }

    fn post_push(&mut self, text: &str, parts: &mut Vec<Part>) {
        let start = text.len() - text.trim_start_matches(NL).len();
        if start < text.len() {
            parts.push(Part::Text(text[start..].to_string()));
            self.state = State::Text;
        }
    }

    fn watch_push(&mut self, text: &str, parts: &mut Vec<Part>, thinking: &mut bool) {
        self.buf.push_str(text);
        let mut scan = 0;
        loop {
            let Some(rel) = self.buf[scan..].find('<') else {
                let chunk = std::mem::take(&mut self.buf);
                if !chunk.is_empty() {
                    parts.push(Part::Text(chunk));
                }
                return;
            };
            let i = scan + rel;
            let cand = &self.buf[i..];
            if let Some(rest) = cand.strip_prefix(OPEN) {
                let prefix = self.buf[..i].to_string();
                let rest = rest.to_string();
                if !prefix.is_empty() {
                    parts.push(Part::Text(prefix));
                }
                *thinking = true;
                self.buf.clear();
                self.state = State::PostOpen;
                if !rest.is_empty() {
                    self.post_open_push(&rest, parts);
                }
                return;
            }
            if OPEN.starts_with(cand) {
                let prefix = self.buf[..i].to_string();
                let held = cand.to_string();
                if !prefix.is_empty() {
                    parts.push(Part::Text(prefix));
                }
                self.buf = held;
                return;
            }
            if let Some(rest) = cand.strip_prefix(CLOSE) {
                let chunk = self.buf[..i].to_string();
                let rest = rest.to_string();
                self.emit_reasoning(&chunk, parts);
                *thinking = true;
                self.buf.clear();
                self.state = State::Post;
                if !rest.is_empty() {
                    self.post_push(&rest, parts);
                }
                return;
            }
            if CLOSE.starts_with(cand) {
                let prefix = self.buf[..i].to_string();
                let held = cand.to_string();
                if !prefix.is_empty() {
                    parts.push(Part::Text(prefix));
                }
                self.buf = held;
                return;
            }
            scan = i + 1;
        }
    }
}
