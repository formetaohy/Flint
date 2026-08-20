pub trait ChatFormat: Send + Sync {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String;

    fn stop_literals(&self) -> &'static [&'static str];
}

pub struct ChatMl;

pub struct ChatMlThink;

pub struct GemmaChat;

pub struct Phi3Chat;

pub struct Phi4Chat;

pub struct Gemma4Chat;

impl ChatFormat for ChatMl {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        render_chatml(false, system, history, user)
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["im_end"]
    }
}

pub struct Llama3Chat;

pub struct Llama2Chat;

impl ChatFormat for Llama3Chat {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        let mut out = String::new();
        out.push_str(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

",
        );
        out.push_str(
            "Cutting Knowledge Date: December 2023
",
        );
        out.push_str(system);
        out.push_str("<|eot_id|>");
        for (u, a) in history {
            out.push_str(
                "<|start_header_id|>user<|end_header_id|>

",
            );
            out.push_str(u);
            out.push_str("<|eot_id|>");
            out.push_str(
                "<|start_header_id|>assistant<|end_header_id|>

",
            );
            out.push_str(a);
            out.push_str("<|eot_id|>");
        }
        out.push_str(
            "<|start_header_id|>user<|end_header_id|>

",
        );
        out.push_str(user);
        out.push_str("<|eot_id|>");
        out.push_str(
            "<|start_header_id|>assistant<|end_header_id|>

",
        );
        out
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["eot_id"]
    }
}

impl ChatFormat for Llama2Chat {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        let mut out = String::new();
        out.push_str("<s>");
        out.push_str("[INST] ");
        if !system.is_empty() {
            out.push_str(
                "<<SYS>>
",
            );
            out.push_str(system);
            out.push_str(
                "
<</SYS>>

",
            );
        }
        for (u, a) in history {
            out.push_str(u);
            out.push_str(" [/INST] ");
            out.push_str(a);
            out.push_str(" </s><s>[INST] ");
        }
        out.push_str(user);
        out.push_str(" [/INST]");
        out
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["[INST]"]
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

impl ChatFormat for Phi3Chat {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        let mut out = String::new();
        push_phi_turn(&mut out, "system", system);
        for (u, a) in history {
            push_phi_turn(&mut out, "user", u);
            push_phi_turn(&mut out, "assistant", a);
        }
        push_phi_turn(&mut out, "user", user);
        out.push_str("<|assistant|>\n");
        out
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["<|end|>"]
    }
}

impl ChatFormat for Phi4Chat {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        let mut out = String::new();
        push_phi4_turn(&mut out, "system", system);
        for (u, a) in history {
            push_phi4_turn(&mut out, "user", u);
            push_phi4_turn(&mut out, "assistant", a);
        }
        push_phi4_turn(&mut out, "user", user);
        out.push_str("<|assistant|>");
        out
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["<|end|>"]
    }
}

impl ChatFormat for Gemma4Chat {
    fn render(&self, system: &str, history: &[(String, String)], user: &str) -> String {
        let mut out = String::new();
        out.push_str("<bos>");
        if !system.is_empty() {
            out.push_str("<|turn>system\n");
            out.push_str(system.trim());
            out.push_str("<turn|>\n");
        }
        for (u, a) in history {
            push_gemma4_turn(&mut out, "user", u);
            push_gemma4_turn(&mut out, "model", a);
        }
        push_gemma4_turn(&mut out, "user", user);
        out.push_str("<|turn>model\n");
        out
    }
    fn stop_literals(&self) -> &'static [&'static str] {
        &["<turn|>"]
    }
}

fn push_phi_turn(out: &mut String, role: &str, content: &str) {
    if content.is_empty() {
        return;
    }
    out.push_str(&format!("<|{role}|>\n{content}<|end|>\n"));
}

fn push_phi4_turn(out: &mut String, role: &str, content: &str) {
    if content.is_empty() {
        return;
    }
    out.push_str(&format!("<|{role}|>{content}<|end|>"));
}

fn push_gemma4_turn(out: &mut String, role: &str, content: &str) {
    out.push_str(&format!("<|turn>{role}\n{content}<turn|>\n"));
}

fn render_chatml(think: bool, system: &str, history: &[(String, String)], user: &str) -> String {
    let mut out = String::new();
    push_turn(&mut out, "system", system);
    for (u, a) in history {
        push_turn(&mut out, "user", u);
        push_turn(&mut out, "assistant", a);
    }
    push_turn(&mut out, "user", user);

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
