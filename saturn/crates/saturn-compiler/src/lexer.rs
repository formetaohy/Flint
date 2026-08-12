use crate::diag::{Diagnostic, Result, Span};
use crate::ir::Scalar;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Str(String),
    Int(u64, Option<Scalar>),
    Float(f64, Option<Scalar>),
    At,
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Dot,
    Range,
    Eq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Shl,
    Shr,
    Bang,
    AndAnd,
    OrOr,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Question,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn span(&self) -> Span {
        Span {
            start: self.pos as u32,
            end: self.pos as u32,
        }
    }

    fn span_to(&self, start: Span) -> Span {
        Span {
            start: start.start,
            end: self.pos as u32,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, ahead: usize) -> Option<u8> {
        self.src.get(self.pos + ahead).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn skip_ws_and_comments(&mut self) -> Result<()> {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => {
                    self.bump();
                }
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => {
                    let start = self.span();
                    self.bump();
                    self.bump();
                    loop {
                        match self.peek() {
                            Some(b'*') if self.peek_at(1) == Some(b'/') => {
                                self.bump();
                                self.bump();
                                break;
                            }
                            Some(_) => {
                                self.bump();
                            }
                            None => {
                                return Err(Diagnostic::new(
                                    self.span_to(start),
                                    "unterminated block comment",
                                ));
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn lex_number(&mut self) -> Result<Tok> {
        let start = self.span();
        let mut text = String::new();
        let mut is_hex = false;
        if self.peek() == Some(b'0') && matches!(self.peek_at(1), Some(b'x' | b'X')) {
            is_hex = true;
            self.bump();
            self.bump();
            while let Some(c) = self.peek() {
                if c.is_ascii_hexdigit() {
                    text.push(c as char);
                    self.bump();
                } else {
                    break;
                }
            }
            if text.is_empty() {
                return Err(Diagnostic::new(
                    self.span_to(start),
                    "expected hex digits after 0x",
                ));
            }
        } else {
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    text.push(c as char);
                    self.bump();
                } else {
                    break;
                }
            }
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') && !matches!(self.peek_at(1), Some(b'.')) {
            is_float = true;
            text.push('.');
            self.bump();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    text.push(c as char);
                    self.bump();
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            text.push(self.bump().unwrap() as char);
            if matches!(self.peek(), Some(b'+' | b'-')) {
                text.push(self.bump().unwrap() as char);
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    text.push(c as char);
                    self.bump();
                } else {
                    break;
                }
            }
        }
        let suffix_span = self.span();
        let ty = match self.peek() {
            Some(b'u') => {
                self.bump();
                if is_float {
                    return Err(Diagnostic::new(
                        suffix_span.to(self.span()),
                        "u suffix requires an integer literal",
                    ));
                }
                Some(Scalar::U32)
            }
            Some(b'i') => {
                self.bump();
                if is_float {
                    return Err(Diagnostic::new(
                        suffix_span.to(self.span()),
                        "i suffix requires an integer literal",
                    ));
                }
                Some(Scalar::I32)
            }
            Some(b'f') => {
                self.bump();
                if !is_float && !is_hex {
                    return Err(Diagnostic::new(
                        suffix_span.to(self.span()),
                        "f suffix requires a float literal",
                    ));
                }
                Some(Scalar::F32)
            }
            Some(b'h') => {
                self.bump();
                if !is_float {
                    return Err(Diagnostic::new(
                        suffix_span.to(self.span()),
                        "h suffix requires a float literal",
                    ));
                }
                Some(Scalar::F16)
            }
            Some(b'b') if self.peek_at(1) == Some(b'f') => {
                self.bump();
                self.bump();
                if !is_float {
                    return Err(Diagnostic::new(
                        suffix_span.to(self.span()),
                        "bf suffix requires a float literal",
                    ));
                }
                Some(Scalar::Bf16)
            }
            _ => None,
        };
        if is_float {
            let value: f64 = text
                .parse()
                .map_err(|_| Diagnostic::new(self.span_to(start), "invalid float literal"))?;
            return Ok(Tok::Float(value, ty));
        }
        let value = if is_hex {
            u64::from_str_radix(&text, 16)
                .map_err(|_| Diagnostic::new(self.span_to(start), "hex literal out of range"))?
        } else {
            text.parse()
                .map_err(|_| Diagnostic::new(self.span_to(start), "invalid integer literal"))?
        };
        Ok(Tok::Int(value, ty))
    }

    fn lex_ident(&mut self) -> Tok {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                name.push(c as char);
                self.bump();
            } else {
                break;
            }
        }
        Tok::Ident(name)
    }
}

pub fn lex(source: &crate::diag::Source) -> Result<Vec<Token>> {
    let mut lexer = Lexer::new(source.text());
    let mut tokens = Vec::new();
    loop {
        lexer.skip_ws_and_comments()?;
        let start = lexer.span();
        let Some(c) = lexer.peek() else {
            break;
        };
        let tok = match c {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => lexer.lex_ident(),
            b'0'..=b'9' => lexer.lex_number()?,
            b'"' => {
                lexer.bump();
                let mut text = String::new();
                loop {
                    match lexer.peek() {
                        Some(b'"') => {
                            lexer.bump();
                            break;
                        }
                        Some(b'\\') if lexer.peek_at(1) == Some(b'"') => {
                            lexer.bump();
                            lexer.bump();
                            text.push('"');
                        }
                        Some(c) => {
                            text.push(c as char);
                            lexer.bump();
                        }
                        None => {
                            return Err(Diagnostic::new(
                                lexer.span_to(start),
                                "unterminated string literal",
                            ));
                        }
                    }
                }
                Tok::Str(text)
            }
            b'@' => {
                lexer.bump();
                Tok::At
            }
            b'{' => {
                lexer.bump();
                Tok::LBrace
            }
            b'}' => {
                lexer.bump();
                Tok::RBrace
            }
            b'(' => {
                lexer.bump();
                Tok::LParen
            }
            b')' => {
                lexer.bump();
                Tok::RParen
            }
            b'[' => {
                lexer.bump();
                Tok::LBracket
            }
            b']' => {
                lexer.bump();
                Tok::RBracket
            }
            b',' => {
                lexer.bump();
                Tok::Comma
            }
            b':' => {
                lexer.bump();
                Tok::Colon
            }
            b';' => {
                lexer.bump();
                Tok::Semicolon
            }
            b'.' => {
                lexer.bump();
                if lexer.eat(b'.') {
                    Tok::Range
                } else {
                    Tok::Dot
                }
            }
            b'=' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::EqEq
                } else {
                    Tok::Eq
                }
            }
            b'+' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::PlusEq
                } else {
                    Tok::Plus
                }
            }
            b'-' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::MinusEq
                } else {
                    Tok::Minus
                }
            }
            b'*' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::StarEq
                } else {
                    Tok::Star
                }
            }
            b'/' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::SlashEq
                } else {
                    Tok::Slash
                }
            }
            b'%' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::PercentEq
                } else {
                    Tok::Percent
                }
            }
            b'&' => {
                lexer.bump();
                if lexer.eat(b'&') {
                    Tok::AndAnd
                } else if lexer.eat(b'=') {
                    Tok::AmpEq
                } else {
                    Tok::Amp
                }
            }
            b'|' => {
                lexer.bump();
                if lexer.eat(b'|') {
                    Tok::OrOr
                } else if lexer.eat(b'=') {
                    Tok::PipeEq
                } else {
                    Tok::Pipe
                }
            }
            b'^' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::CaretEq
                } else {
                    Tok::Caret
                }
            }
            b'<' => {
                lexer.bump();
                if lexer.eat(b'<') {
                    if lexer.eat(b'=') {
                        Tok::ShlEq
                    } else {
                        Tok::Shl
                    }
                } else if lexer.eat(b'=') {
                    Tok::Le
                } else {
                    Tok::Lt
                }
            }
            b'>' => {
                lexer.bump();
                if lexer.eat(b'>') {
                    if lexer.eat(b'=') {
                        Tok::ShrEq
                    } else {
                        Tok::Shr
                    }
                } else if lexer.eat(b'=') {
                    Tok::Ge
                } else {
                    Tok::Gt
                }
            }
            b'!' => {
                lexer.bump();
                if lexer.eat(b'=') {
                    Tok::Ne
                } else {
                    Tok::Bang
                }
            }
            b'?' => {
                lexer.bump();
                Tok::Question
            }
            _ => {
                return Err(Diagnostic::new(
                    Span {
                        start: start.start,
                        end: lexer.pos as u32 + 1,
                    },
                    format!("unexpected character '{}'", c as char),
                ));
            }
        };
        let end = lexer.span();
        tokens.push(Token {
            tok,
            span: Span {
                start: start.start,
                end: end.start,
            },
        });
    }
    Ok(tokens)
}
