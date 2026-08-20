use flint_error::{Error, Result};
use serde_json::Value;

use crate::protocols::Part;
use crate::tools::Call;

const TEXT_OPEN: &str = "{\"type\":\"text\",\"text\":\"";
const TOOL_OPEN: &str = "{\"type\":\"tool_call\",\"calls\":[";
const NAME_OPEN: &str = "\"name\":\"";
const ARGS_OPEN: &str = "\"arguments\":{";

#[derive(PartialEq)]
enum Stage {
    Prefix,
    Text,
    Calls,
    Done,
}

pub struct DecisionParser {
    buf: String,
    stage: Stage,
    pos: usize,
    call: usize,
    name_pending: bool,
    args_pending: bool,
    args_first: bool,
    depth: u32,
    in_string: bool,
}

impl Default for DecisionParser {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionParser {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            stage: Stage::Prefix,
            pos: 0,
            call: 0,
            name_pending: false,
            args_pending: false,
            args_first: false,
            depth: 0,
            in_string: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Result<Vec<Part>> {
        if self.stage == Stage::Done {
            return Ok(Vec::new());
        }
        self.buf.push_str(text);
        let mut parts = Vec::new();
        match self.stage {
            Stage::Prefix => self.scan_prefix(&mut parts)?,
            Stage::Text => self.scan_text(&mut parts)?,
            Stage::Calls => self.scan_calls(&mut parts)?,
            Stage::Done => {}
        }
        Ok(parts)
    }

    pub fn tool_calls(&self) -> Option<Result<Vec<Call>>> {
        if !matches!(self.stage, Stage::Calls) {
            return None;
        }
        Some(self.parse_calls())
    }

    pub fn was_tool_branch(&self) -> bool {
        matches!(self.stage, Stage::Calls)
    }

    fn parse_calls(&self) -> Result<Vec<Call>> {
        let wrapper: Value = serde_json::from_str(&self.buf)
            .map_err(|e| Error::Model(format!("tool call JSON is invalid: {e}")))?;
        let list = wrapper
            .get("calls")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Model("tool call wrapper has no calls".into()))?;
        list.iter()
            .map(|c| {
                Ok(Call {
                    name: c["name"]
                        .as_str()
                        .ok_or_else(|| Error::Model("tool call has no name".into()))?
                        .to_string(),
                    arguments: c
                        .get("arguments")
                        .cloned()
                        .ok_or_else(|| Error::Model("tool call has no arguments".into()))?,
                })
            })
            .collect()
    }

    fn scan_prefix(&mut self, parts: &mut Vec<Part>) -> Result<()> {
        let rest = &self.buf[self.pos..];
        if TEXT_OPEN.starts_with(rest) || TOOL_OPEN.starts_with(rest) {
            return Ok(());
        }
        if rest.starts_with(TEXT_OPEN) {
            self.stage = Stage::Text;
            self.pos += TEXT_OPEN.len();
            return self.scan_text(parts);
        }
        if rest.starts_with(TOOL_OPEN) {
            self.stage = Stage::Calls;
            self.pos += TOOL_OPEN.len();
            return self.scan_calls(parts);
        }
        Err(Error::Model(format!(
            "model output deviates from the decision schema: {rest:?}"
        )))
    }

    fn scan_text(&mut self, parts: &mut Vec<Part>) -> Result<()> {
        let rest = &self.buf[self.pos..];
        let Some(quote) = rest.find('"') else {
            return Ok(());
        };
        parts.push(Part::Text(rest[..quote].to_string()));
        self.pos += quote + 1;
        let tail = &self.buf[self.pos..];
        if tail.is_empty() {
            return Ok(());
        }
        if tail == "}" {
            self.stage = Stage::Done;
            Ok(())
        } else {
            Err(Error::Model(format!(
                "unexpected tail after the text field: {tail:?}"
            )))
        }
    }

    fn scan_calls(&mut self, parts: &mut Vec<Part>) -> Result<()> {
        loop {
            let rest = &self.buf[self.pos..];
            if self.args_pending {
                let (taken, depth, in_string) = consume_args(rest, self.depth, self.in_string);
                if taken > 0 {
                    let chunk = rest[..taken].to_string();
                    let chunk = if self.args_first {
                        self.args_first = false;
                        format!("{{{chunk}")
                    } else {
                        chunk
                    };
                    parts.push(Part::CallArgs {
                        index: self.call,
                        chunk,
                    });
                    self.pos += taken;
                    self.depth = depth;
                    self.in_string = in_string;
                }
                if self.depth > 0 {
                    return Ok(());
                }
                self.args_pending = false;
                self.name_pending = false;
                continue;
            }
            if self.name_pending {
                match find_marker(rest, ARGS_OPEN) {
                    Some(start) => {
                        self.pos += start + ARGS_OPEN.len();
                        self.depth = 1;
                        self.args_pending = true;
                        self.args_first = true;
                        continue;
                    }
                    None => return Ok(()),
                }
            }
            match find_marker(rest, NAME_OPEN) {
                Some(start) => {
                    let name_at = self.pos + start + NAME_OPEN.len();
                    let tail = &self.buf[name_at..];
                    match tail.find('"') {
                        Some(end) => {
                            self.call += 1;
                            self.pos = name_at + end + 1;
                            self.name_pending = true;
                            parts.push(Part::CallStart {
                                index: self.call,
                                name: tail[..end].to_string(),
                            });
                        }
                        None => {
                            self.pos += start;
                            return Ok(());
                        }
                    }
                }
                None => return Ok(()),
            }
        }
    }
}

fn consume_args(rest: &str, mut depth: u32, mut in_string: bool) -> (usize, u32, bool) {
    let mut taken = 0;
    for b in rest.bytes() {
        taken += 1;
        if in_string {
            if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (taken, 0, false);
                    }
                }
                _ => {}
            }
        }
    }
    (taken, depth, in_string)
}

fn find_marker(rest: &str, marker: &str) -> Option<usize> {
    rest.find(marker)
}
