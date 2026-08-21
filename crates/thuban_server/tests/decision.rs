use thuban_error::Result;
use thuban_server::protocols::Part;
use thuban_server::protocols::decision::DecisionParser;
use thuban_server::tools::Tool;
use serde_json::json;

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
        serde_json::from_str::<serde_json::Value>(&all_args).unwrap()["cmd"],
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
    let _ = thuban_generate::Grammar::from_schema(
        &thuban_server::tools::wrapper_schema(&tools, true),
    )
    .unwrap();
    let _ = thuban_generate::Grammar::from_schema(
        &thuban_server::tools::wrapper_schema(&tools, false),
    )
    .unwrap();
}
