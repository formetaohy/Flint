use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn dummy() -> Self {
        Self {
            start: u32::MAX,
            end: u32::MAX,
        }
    }

    pub fn to(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl Default for Span {
    fn default() -> Self {
        Self::dummy()
    }
}

#[derive(Debug, Clone)]
pub struct Source {
    name: String,
    text: String,
    line_starts: Vec<u32>,
}

impl Source {
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let mut line_starts = vec![0u32];
        for (index, byte) in text.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push(index as u32 + 1);
            }
        }
        Self {
            name: name.into(),
            text,
            line_starts,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        Ok(Self::new(path.display().to_string(), text))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let line = match self.line_starts.binary_search(&offset) {
            Ok(line) => line,
            Err(line) => line - 1,
        };
        let col = offset - self.line_starts[line] + 1;
        (line as u32 + 1, col)
    }

    pub fn render_span(&self, span: Span) -> String {
        if span == Span::dummy() {
            return format!("{}:<unknown>", self.name);
        }
        let (line, col) = self.line_col(span.start);
        format!("{}:{}:{}", self.name, line, col)
    }

    pub fn render(&self, diagnostic: &Diagnostic) -> String {
        let mut out = format!(
            "error: {}\n  --> {}",
            diagnostic.msg,
            self.render_span(diagnostic.span)
        );
        if diagnostic.span == Span::dummy() {
            return out;
        }
        let (line, _) = self.line_col(diagnostic.span.start);
        let start_line = self.line_starts[(line - 1) as usize];
        let line_text = self.text[start_line as usize..]
            .split('\n')
            .next()
            .unwrap_or_default();
        let (_, col) = self.line_col(diagnostic.span.start);
        let (_, end_col) = self.line_col(diagnostic.span.end);
        let caret_len = if diagnostic.span.end <= diagnostic.span.start {
            1
        } else {
            end_col.saturating_sub(col).max(1)
        };
        out.push_str(&format!(
            "\n   |\n {line:>3} | {line_text}\n   | {}{}",
            " ".repeat(col as usize - 1),
            "^".repeat(caret_len as usize),
        ));
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub span: Span,
    pub msg: String,
}

impl Diagnostic {
    pub fn new(span: Span, msg: impl Into<String>) -> Self {
        Self {
            span,
            msg: msg.into(),
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error: {}", self.msg)
    }
}

impl std::error::Error for Diagnostic {}

pub type Result<T> = std::result::Result<T, Diagnostic>;
