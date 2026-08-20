use flint_error::{Error, Result};
use serde_json::{Value, json};

pub struct Tool {
    pub name: String,
    pub description: String,
    pub schema: Value,
}

impl Clone for Tool {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
        }
    }
}

pub struct Call {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug)]
pub enum Part {
    Text(String),
    CallStart { index: usize, name: String },
    CallArgs { index: usize, chunk: String },
}

const TEXT_OPEN: &str = "{\"type\":\"text\",\"text\":\"";
const TOOL_OPEN: &str = "{\"type\":\"tool_call\",\"calls\":[";
const NAME_OPEN: &str = "\"name\":\"";
const ARGS_OPEN: &str = "\"arguments\":{";

pub fn wrapper_schema(tools: &[Tool], text_allowed: bool) -> Value {
    let text = json!({
        "type": "object",
        "propertyOrder": ["type", "text"],
        "required": ["type", "text"],
        "properties": {
            "type": {"type": "string", "enum": ["text"]},
            "text": {"type": "string"}
        }
    });
    let calls = json!({
        "type": "object",
        "propertyOrder": ["type", "calls"],
        "required": ["type", "calls"],
        "properties": {
            "type": {"type": "string", "enum": ["tool_call"]},
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "propertyOrder": ["name", "arguments"],
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {"type": "string", "enum": tools.iter().map(|t| json!(t.name)).collect::<Vec<_>>()},
                        "arguments": {"anyOf": tools.iter().map(arguments_schema).collect::<Vec<_>>()}
                    }
                }
            }
        }
    });
    if text_allowed {
        json!({"anyOf": [text, calls]})
    } else {
        calls
    }
}

fn arguments_schema(tool: &Tool) -> Value {
    let schema = &tool.schema;
    let is_object = schema.get("type").and_then(Value::as_str) == Some("object")
        || schema.get("properties").is_some();
    if is_object {
        schema.clone()
    } else {
        json!({"type": "object"})
    }
}

pub fn render_tool_calls(calls: &[Call]) -> String {
    let calls: Vec<Value> = calls
        .iter()
        .map(|c| json!({"name": c.name, "arguments": c.arguments}))
        .collect();
    json!({"type": "tool_call", "calls": calls}).to_string()
}

pub fn render_tool_call(name: &str, arguments: &Value) -> String {
    render_tool_calls(&[Call {
        name: name.to_string(),
        arguments: arguments.clone(),
    }])
}

pub fn tool_instruction(tools: &[Tool]) -> String {
    let mut out = String::new();
    out.push_str("You are an agent with access to tools. Respond with exactly one JSON object, in one of these two forms:\n");
    out.push_str("1. To use tools: {\"type\":\"tool_call\",\"calls\":[{\"name\":\"<tool name>\",\"arguments\":{<arguments matching the tool schema>}}]}. List several calls to run tools in parallel.\n");
    out.push_str("2. To answer with plain text: {\"type\":\"text\",\"text\":\"<your answer>\"}.\n");
    out.push_str("Available tools:\n");
    for t in tools {
        out.push_str(&format!(
            "- {}: {}\n  arguments schema: {}\n",
            t.name, t.description, t.schema
        ));
    }
    out
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn text(parser: &mut DecisionParser, s: &str) -> Result<Vec<Part>> {
        parser.push(s)
    }

    #[test]
    fn text_branch_streams_inner_content() {
        let mut p = DecisionParser::new();
        let parts = text(&mut p, "{\"type\":\"te").unwrap();
        assert!(parts.is_empty());
        let parts = text(&mut p, "xt\",\"text\":\"hello world\"}").unwrap();
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            Part::Text(t) => assert_eq!(t, "hello world"),
            other => panic!("expected text, got {other:?}"),
        }
        assert!(!p.was_tool_branch());
    }

    #[test]
    fn text_branch_splits_quote_across_pieces() {
        let mut p = DecisionParser::new();
        assert!(
            text(&mut p, "{\"type\":\"text\",\"text\":\"a")
                .unwrap()
                .is_empty()
        );
        let parts = text(&mut p, "\"}").unwrap();
        match &parts[0] {
            Part::Text(t) => assert_eq!(t, "a"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn tool_branch_yields_start_and_args_and_calls() {
        let mut p = DecisionParser::new();
        assert!(
            text(&mut p, "{\"type\":\"tool_call\",\"calls\":[{\"name\":\"Ba")
                .unwrap()
                .is_empty()
        );
        let parts = text(&mut p, "sh\",\"arguments\":{\"cmd\":\"ls\"}}]}").unwrap();
        match &parts[0] {
            Part::CallStart { index, name } => {
                assert_eq!(*index, 1);
                assert_eq!(name, "Bash");
            }
            other => panic!("expected call start, got {other:?}"),
        }
        match &parts[1] {
            Part::CallArgs { index, chunk } => {
                assert_eq!(*index, 1);
                assert_eq!(chunk, "{\"cmd\":\"ls\"}");
            }
            other => panic!("expected args, got {other:?}"),
        }
        let calls = p.tool_calls().unwrap().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["cmd"], "ls");
    }

    #[test]
    fn braces_inside_argument_strings_do_not_split_args() {
        let mut p = DecisionParser::new();
        assert!(
            text(&mut p, "{\"type\":\"tool_call\",\"calls\":[{\"name\":\"E")
                .unwrap()
                .is_empty()
        );
        let parts = text(
            &mut p,
            "dit\",\"arguments\":{\"code\":\"if (x) { y(); }\"}}]}",
        )
        .unwrap();
        let args: String = parts
            .iter()
            .filter_map(|p| match p {
                Part::CallArgs { chunk, .. } => Some(chunk.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(args, "{\"code\":\"if (x) { y(); }\"}");
        let calls = p.tool_calls().unwrap().unwrap();
        assert_eq!(calls[0].arguments["code"], "if (x) { y(); }");
    }

    #[test]
    fn multiple_calls_stream_independently() {
        let mut p = DecisionParser::new();
        let parts = text(
            &mut p,
            "{\"type\":\"tool_call\",\"calls\":[{\"name\":\"A\",\"arguments\":{\"x\":1}},{\"name\":\"B\",\"arguments\":{}}]}",
        )
        .unwrap();
        let starts: Vec<String> = parts
            .iter()
            .filter_map(|p| match p {
                Part::CallStart { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(starts, ["A", "B"]);
        let calls = p.tool_calls().unwrap().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "A");
        assert_eq!(calls[1].name, "B");
        assert_eq!(calls[1].arguments, json!({}));
    }

    #[test]
    fn deviation_from_the_schema_is_rejected() {
        let mut p = DecisionParser::new();
        assert!(text(&mut p, "plain text").is_err());
        let mut p = DecisionParser::new();
        assert!(text(&mut p, "{\"type\":\"text\",\"text\":\"hi\"}xx").is_err());
    }

    #[test]
    fn empty_and_split_arguments_form_valid_json() {
        let mut p = DecisionParser::new();
        assert!(
            !text(
                &mut p,
                "{\"type\":\"tool_call\",\"calls\":[{\"name\":\"N\",\"arguments\":{}}]}"
            )
            .unwrap()
            .is_empty()
        );
        let mut p = DecisionParser::new();
        let mut all_args = String::new();
        for part in text(
            &mut p,
            "{\"type\":\"tool_call\",\"calls\":[{\"name\":\"S\",\"arguments\":{\"cmd\"",
        )
        .unwrap()
        {
            if let Part::CallArgs { chunk, .. } = part {
                all_args.push_str(&chunk);
            }
        }
        for part in text(&mut p, ":\"ls\"}}]}").unwrap() {
            if let Part::CallArgs { chunk, .. } = part {
                all_args.push_str(&chunk);
            }
        }
        assert_eq!(
            serde_json::from_str::<Value>(&all_args).unwrap()["cmd"],
            "ls"
        );
    }

    #[test]
    fn wrapper_schema_compiles_for_real_tools() {
        let tools = vec![
            Tool {
                name: "Bash".into(),
                description: "run a command".into(),
                schema: json!({"type": "object", "required": ["cmd"], "properties": {"cmd": {"type": "string"}}}),
            },
            Tool {
                name: "Noop".into(),
                description: "nothing".into(),
                schema: json!({}),
            },
        ];
        let _ = flint_generate::Grammar::from_schema(&wrapper_schema(&tools, true)).unwrap();
        let _ = flint_generate::Grammar::from_schema(&wrapper_schema(&tools, false)).unwrap();
    }
}
