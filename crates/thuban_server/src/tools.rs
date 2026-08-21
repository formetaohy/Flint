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
