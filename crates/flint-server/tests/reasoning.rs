use flint_architectures::chat::ThinkMode;
use flint_server::protocols::Part;
use flint_server::protocols::reasoning::ReasoningParser;

fn text(parts: &[Part]) -> Vec<String> {
    parts
        .iter()
        .filter_map(|p| match p {
            Part::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

fn reasoning(parts: &[Part]) -> Vec<String> {
    parts
        .iter()
        .filter_map(|p| match p {
            Part::Reasoning(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn preopened_streams_reasoning_until_close_tag() {
    let mut p = ReasoningParser::new(ThinkMode::Preopened);
    let (parts, thinking) = p.push("I need to");
    assert!(thinking);
    assert_eq!(text(&parts), Vec::<String>::new());
    assert_eq!(reasoning(&parts), ["I need to"]);

    let (parts, thinking) = p.push(" analyze\n");
    assert!(thinking);
    assert_eq!(reasoning(&parts), [" analyze"]);

    let (parts, _) = p.push("</think>");
    assert_eq!(text(&parts), Vec::<String>::new());
    assert_eq!(reasoning(&parts), Vec::<String>::new());

    let (parts, thinking) = p.push("\n\nHello world");
    assert!(!thinking);
    assert_eq!(text(&parts), ["Hello world"]);
}

#[test]
fn preopened_close_tag_split_across_pieces() {
    let mut p = ReasoningParser::new(ThinkMode::Preopened);
    let (parts, _) = p.push("ponder</th");
    assert_eq!(reasoning(&parts), ["ponder"]);
    let (parts, _) = p.push("ink>");
    assert!(parts.is_empty());
    let (parts, _) = p.push("\nAnswer");
    assert_eq!(text(&parts), ["Answer"]);
}

#[test]
fn preopened_unclosed_thinking_is_flushed_on_finish() {
    let mut p = ReasoningParser::new(ThinkMode::Preopened);
    let (parts, _) = p.push("unfinished");
    assert_eq!(reasoning(&parts), ["unfinished"]);
    let (parts, _) = p.push(" tail</th");
    assert_eq!(reasoning(&parts), [" tail"]);
    let parts = p.finish();
    assert_eq!(reasoning(&parts), ["</th"]);
}

#[test]
fn emitted_mode_detects_open_tag() {
    let mut p = ReasoningParser::new(ThinkMode::Emitted);
    let (parts, thinking) = p.push("<think>\n");
    assert!(thinking);
    assert!(parts.is_empty());
    let (parts, _) = p.push("reasoning");
    assert_eq!(reasoning(&parts), ["reasoning"]);
    let (parts, _) = p.push("</think>");
    assert!(parts.is_empty());
    let (parts, _) = p.push("answer");
    assert_eq!(text(&parts), ["answer"]);
}

#[test]
fn emitted_mode_open_tag_split_across_pieces() {
    let mut p = ReasoningParser::new(ThinkMode::Emitted);
    let (parts, _) = p.push("<th");
    assert!(parts.is_empty());
    let (parts, thinking) = p.push("ink>");
    assert!(thinking);
    assert!(parts.is_empty());
    let (parts, _) = p.push("deep thought");
    assert_eq!(reasoning(&parts), ["deep thought"]);
}

#[test]
fn emitted_mode_plain_text_falls_through() {
    let mut p = ReasoningParser::new(ThinkMode::Emitted);
    let (parts, thinking) = p.push("Hello");
    assert!(!thinking);
    assert_eq!(text(&parts), ["Hello"]);
    let (parts, _) = p.push(" world");
    assert_eq!(text(&parts), [" world"]);
    let (parts, _) = p.push("a < b");
    assert_eq!(text(&parts), ["a < b"]);
}

#[test]
fn emitted_mode_open_tag_inside_single_piece() {
    let mut p = ReasoningParser::new(ThinkMode::Emitted);
    let (parts, thinking) = p.push("<think>hidden</think>");
    assert!(thinking);
    assert_eq!(reasoning(&parts), ["hidden"]);
}

#[test]
fn literal_angle_bracket_inside_reasoning_is_kept() {
    let mut p = ReasoningParser::new(ThinkMode::Preopened);
    let (parts, _) = p.push("compare 3 < 5 then");
    assert_eq!(reasoning(&parts), ["compare 3 < 5 then"]);
    let (parts, _) = p.push("</think>");
    assert!(parts.is_empty());
}

#[test]
fn disabled_mode_never_splits() {
    let mut p = ReasoningParser::new(ThinkMode::None);
    let (parts, thinking) = p.push("<think>raw</think>");
    assert!(!thinking);
    assert_eq!(text(&parts), ["<think>raw</think>"]);
}
