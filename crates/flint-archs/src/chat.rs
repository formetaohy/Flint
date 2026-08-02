//! Per-family chat prompt formats. Each model family owns how its turns are
//! rendered into a prompt and which extra literals terminate a reply. Marker
//! tokens (im_start/im_end, think tags, Gemma turn tags) are plain
//! angle-bracket sequences registered as added/special tokens.

/// A chat prompt format plus the literals that end a reply.
pub trait ChatFormat {
    /// Renders system + history + the user turn, ending with the assistant
    /// generation prefix.
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String;
    /// Family-specific reply terminators, resolved against the tokenizer vocab.
    fn stop_literals(&self) -> &'static [&'static str];
}

/// Plain ChatML (im_start/im_end), no reasoning block. Qwen2/3, SmolLM and most
/// ChatML Instruct checkpoints.
pub struct ChatMl;

/// ChatML with Qwen3.5's empty non-thinking block opened up front.
pub struct ChatMlThink;

/// Gemma's turn format: `<start_of_turn>role\ncontent<end_of_turn>\n`.
pub struct GemmaChat;

impl ChatFormat for ChatMl {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        render_chatml(false, system, history, user)
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["im_end"]
    }
}

impl ChatFormat for ChatMlThink {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        render_chatml(true, system, history, user)
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["im_end"]
    }
}

impl ChatFormat for GemmaChat {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        render_gemma(system, history, user)
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["<end_of_turn>"]
    }
}

/// ChatML rendering. With `think`, opens an empty think block in the assistant
/// prefix (Qwen3.5 non-thinking mode).
fn render_chatml(think: bool, system: &str, history: &[(String, String)], user: &str) -> String {
    let mut out = String::new();
    push_turn(&mut out, "system", system);
    for (u, a) in history {
        push_turn(&mut out, "user", u);
        push_turn(&mut out, "assistant", a);
    }
    push_turn(&mut out, "user", user);
    // Assistant generation prefix.
    out.push_str(&im_marker(true));
    out.push_str("assistant");
    out.push('\n');
    if think {
        out.push_str(&think_marker(true));
        out.push('\n');
        out.push('\n');
        out.push_str(&think_marker(false));
        out.push('\n');
        out.push('\n');
    }
    out
}

/// Gemma turn format: `<bos><start_of_turn>role\ncontent<end_of_turn>\n`. Gemma
/// has no system role, so the system prompt is folded into the first user turn.
fn render_gemma(system: &str, history: &[(String, String)], user: &str) -> String {
    let mut out = String::new();
    out.push_str("<bos>");
    let first_user = if system.is_empty() {
        user.to_string()
    } else {
        format!("{system}\n\n{user}")
    };
    out.push_str("<start_of_turn>user\n");
    out.push_str(&first_user);
    out.push_str("<end_of_turn>\n");
    for (u, a) in history {
        out.push_str("<start_of_turn>user\n");
        out.push_str(u);
        out.push_str("<end_of_turn>\n");
        out.push_str("<start_of_turn>model\n");
        out.push_str(a);
        out.push_str("<end_of_turn>\n");
    }
    out.push_str("<start_of_turn>model\n");
    out
}

fn push_turn(out: &mut String, role: &str, content: &str) {
    out.push_str(&im_marker(true));
    out.push_str(role);
    out.push('\n');
    out.push_str(content);
    out.push_str(&im_marker(false));
    out.push('\n');
}

fn im_marker(start: bool) -> String {
    format!("<|{}|>", if start { "im_start" } else { "im_end" })
}

fn think_marker(open: bool) -> String {
    if open {
        "<think>".to_string()
    } else {
        "</think>".to_string()
    }
}
