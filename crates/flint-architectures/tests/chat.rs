use flint_architectures::chat::{ChatFormat, ChatMl, ChatMlThink, GemmaChat};

fn im(start: bool) -> String {
    format!("<|im_{}|>", if start { "start" } else { "end" })
}

fn turn(role: &str, content: &str) -> String {
    format!("{}{}\n{}{}\n", im(true), role, content, im(false))
}

#[test]
fn chatml_renders_system_user_and_assistant_prefix() {
    let got = ChatMl.render("sys", &[], "hi");
    let want = format!(
        "{}{}{}assistant\n",
        turn("system", "sys"),
        turn("user", "hi"),
        im(true)
    );
    assert_eq!(got, want);
    assert!(got.ends_with(&format!("{}assistant\n", im(true))));
}

#[test]
fn chatml_interleaves_history() {
    let got = ChatMl.render("", &[("u1".to_string(), "a1".to_string())], "u2");
    let want = format!(
        "{}{}{}{}{}assistant\n",
        turn("system", ""),
        turn("user", "u1"),
        turn("assistant", "a1"),
        turn("user", "u2"),
        im(true),
    );
    assert_eq!(got, want);
}

#[test]
fn chatml_think_opens_an_empty_think_block() {
    let plain = ChatMl.render("s", &[], "u");
    let think = ChatMlThink.render("s", &[], "u");
    let block = "<think>\n\n</think>\n\n".to_string();
    assert!(!plain.contains(&block), "plain ChatML has no think block");
    assert!(think.ends_with(&format!("{}assistant\n{block}", im(true))));
}

#[test]
fn gemma_folds_system_into_the_first_user_turn() {
    let got = GemmaChat.render("sys", &[], "hi");
    assert_eq!(
        got,
        "<bos><start_of_turn>user\nsys\n\nhi<end_of_turn>\n<start_of_turn>model\n"
    );

    let no_sys = GemmaChat.render("", &[], "hi");
    assert_eq!(
        no_sys,
        "<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"
    );
}

#[test]
fn gemma_appends_history_as_user_model_pairs() {
    let got = GemmaChat.render("", &[("q".to_string(), "a".to_string())], "q2");
    let want = format!(
        "<bos><start_of_turn>user\nq2<end_of_turn>\n{}{}<start_of_turn>model\n",
        "<start_of_turn>user\nq<end_of_turn>\n", "<start_of_turn>model\na<end_of_turn>\n",
    );
    assert_eq!(got, want);
}

#[test]
fn stop_literals_per_family() {
    assert_eq!(ChatMl.stop_literals(), &["im_end"]);
    assert_eq!(ChatMlThink.stop_literals(), &["im_end"]);
    assert_eq!(GemmaChat.stop_literals(), &["<end_of_turn>"]);
}
