use thuban_server::generator::RequestDefaults;
use thuban_server::protocols::anthropic;
use thuban_server::protocols::gemini;
use thuban_server::protocols::openai_chat;
use thuban_server::protocols::openai_responses;
use thuban_server::protocols::split_reasoning;
use serde_json::{Value, json};

fn defaults() -> RequestDefaults {
    RequestDefaults {
        model_id: "test-model".into(),
        max_tokens: 512,
    }
}

#[test]
fn split_reasoning_keeps_text_after_last_close_tag() {
    assert_eq!(split_reasoning("plain answer"), "plain answer");
    assert_eq!(split_reasoning("<think>hidden</think>\n\nanswer"), "answer");
    assert_eq!(
        split_reasoning("<think>a</think>\nb</think>final"),
        "final"
    );
}

#[test]
fn chat_parse_drops_reasoning_from_assistant_history() {
    let body = json!({
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>step by step</think>\n\ndone", "reasoning_content": "ignored"},
            {"role": "user", "content": "q2"}
        ]
    });
    let parsed = openai_chat::parse(&body, &defaults()).unwrap();
    assert_eq!(parsed.req.user, "q2");
    assert_eq!(parsed.req.history.len(), 1);
    assert_eq!(parsed.req.history[0].0, "q1");
    assert_eq!(parsed.req.history[0].1, "done");
    assert!(parsed.req.thinking);
}

#[test]
fn chat_parse_enable_thinking_off() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "enable_thinking": false
    });
    assert!(!openai_chat::parse(&body, &defaults()).unwrap().req.thinking);

    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": false}
    });
    assert!(!openai_chat::parse(&body, &defaults()).unwrap().req.thinking);

    let body = json!({"messages": [{"role": "user", "content": "hi"}]});
    assert!(openai_chat::parse(&body, &defaults()).unwrap().req.thinking);
}

#[test]
fn anthropic_parse_thinking_config() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"}
    });
    assert!(!anthropic::parse(&body, &defaults()).unwrap().req.thinking);

    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    });
    assert!(anthropic::parse(&body, &defaults()).unwrap().req.thinking);

    let body = json!({"messages": [{"role": "user", "content": "hi"}]});
    assert!(anthropic::parse(&body, &defaults()).unwrap().req.thinking);
}

#[test]
fn anthropic_parse_drops_thinking_blocks_from_history() {
    let body = json!({
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hidden", "signature": ""},
                {"type": "text", "text": "<think>embedded</think>\n\nvisible"}
            ]},
            {"role": "user", "content": "q2"}
        ]
    });
    let parsed = anthropic::parse(&body, &defaults()).unwrap();
    assert_eq!(parsed.req.history.len(), 1);
    assert_eq!(parsed.req.history[0].1, "visible");
}

#[test]
fn gemini_parse_thinking_budget() {
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}
    });
    assert!(!gemini::parse(&body, &defaults()).unwrap().req.thinking);

    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}
    });
    assert!(gemini::parse(&body, &defaults()).unwrap().req.thinking);

    let body = json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]});
    assert!(gemini::parse(&body, &defaults()).unwrap().req.thinking);
}

#[test]
fn gemini_parse_drops_thought_parts_from_history() {
    let body = json!({
        "contents": [
            {"role": "user", "parts": [{"text": "q1"}]},
            {"role": "model", "parts": [
                {"text": "hidden", "thought": true},
                {"text": "visible"}
            ]},
            {"role": "user", "parts": [{"text": "q2"}]}
        ]
    });
    let parsed = gemini::parse(&body, &defaults()).unwrap();
    assert_eq!(parsed.req.history.len(), 1);
    assert_eq!(parsed.req.history[0].1, "visible");
}

#[test]
fn responses_parse_reasoning_effort() {
    let body = json!({
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": {"effort": "none"}
    });
    assert!(!openai_responses::parse(&body, &defaults()).unwrap().req.thinking);

    let body = json!({
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "reasoning": {"effort": "high"}
    });
    assert!(openai_responses::parse(&body, &defaults()).unwrap().req.thinking);
}

#[test]
fn responses_parse_skips_reasoning_input_items() {
    let body = json!({
        "input": [
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "hidden"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
    });
    let parsed = openai_responses::parse(&body, &defaults()).unwrap();
    assert_eq!(parsed.req.user, "hi");
    assert!(parsed.req.history.is_empty());
}

#[test]
fn responses_parse_strips_reasoning_from_assistant_messages() {
    let body: Value = json!({
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q1"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "<think>hidden</think>\n\nvisible"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q2"}]}
        ]
    });
    let parsed = openai_responses::parse(&body, &defaults()).unwrap();
    assert_eq!(parsed.req.history.len(), 1);
    assert_eq!(parsed.req.history[0].1, "visible");
}
