use crate::ast::{BinOp, Expr, Kernel, Param, Scalar, Stmt, Type, UnOp};
use crate::diag::{Diagnostic, Result, Span};
use crate::lexer::{Tok, Token};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos).map(|t| &t.tok)
    }

    fn peek_at(&self, ahead: usize) -> Option<&Tok> {
        self.tokens.get(self.pos + ahead).map(|t| &t.tok)
    }

    fn span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|t| t.span)
            .unwrap_or_default()
    }

    fn bump(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn expect(&mut self, tok: Tok) -> Result<Token> {
        let span = self.span();
        match self.bump() {
            Some(token) if token.tok == tok => Ok(token),
            _ => Err(Diagnostic::new(
                span,
                format!("expected {}, found {}", tok_desc(&tok), self.tok_desc_at(0)),
            )),
        }
    }

    fn tok_desc_at(&self, ahead: usize) -> String {
        match self.peek_at(ahead) {
            Some(tok) => tok_desc(tok),
            None => "end of input".to_string(),
        }
    }

    fn is_ident(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(n)) if n == name)
    }

    fn eat_ident(&mut self, name: &str) -> Option<Token> {
        if self.is_ident(name) {
            self.bump()
        } else {
            None
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span)> {
        let span = self.span();
        match self.bump() {
            Some(Token {
                tok: Tok::Ident(name),
                span,
            }) => Ok((name, span)),
            _ => Err(Diagnostic::new(
                span,
                format!("expected identifier, found {}", self.tok_desc_at(0)),
            )),
        }
    }

    fn expect_int(&mut self) -> Result<(u64, Span)> {
        let span = self.span();
        match self.bump() {
            Some(Token {
                tok: Tok::Int(value),
                span,
            }) => Ok((value, span)),
            _ => Err(Diagnostic::new(
                span,
                format!("expected integer, found {}", self.tok_desc_at(0)),
            )),
        }
    }

    fn parse_scalar(&mut self) -> Result<Scalar> {
        let span = self.span();
        let name = match self.peek() {
            Some(Tok::Ident(name)) => name.clone(),
            _ => {
                return Err(Diagnostic::new(
                    span,
                    format!("expected scalar type, found {}", self.tok_desc_at(0)),
                ));
            }
        };
        let scalar = match name.as_str() {
            "f32" => Scalar::F32,
            "f16" => Scalar::F16,
            "bf16" => Scalar::Bf16,
            "i32" => Scalar::I32,
            "u32" => Scalar::U32,
            "i8" => Scalar::I8,
            "u8" => Scalar::U8,
            "bool" => Scalar::Bool,
            _ => {
                return Err(Diagnostic::new(
                    span,
                    format!("unknown scalar type '{name}'"),
                ));
            }
        };
        self.bump();
        Ok(scalar)
    }

    fn parse_type(&mut self) -> Result<Type> {
        if self.is_ident("buf") {
            self.bump();
            self.expect(Tok::Lt)?;
            let elem = self.parse_scalar()?;
            self.expect(Tok::Gt)?;
            return Ok(Type::Buf(elem));
        }
        if self.is_ident("vec2") || self.is_ident("vec3") || self.is_ident("vec4") {
            let (name, _) = self.expect_ident()?;
            let size = name.as_bytes()[3] - b'0';
            self.expect(Tok::Lt)?;
            let elem = self.parse_scalar()?;
            self.expect(Tok::Gt)?;
            return Ok(Type::Vec {
                size: size as u32,
                elem,
            });
        }
        if self.is_ident("matrix") {
            self.bump();
            self.expect(Tok::Lt)?;
            let elem = self.parse_scalar()?;
            self.expect(Tok::Comma)?;
            let role = match self.peek() {
                Some(Tok::Ident(name)) => name.as_str(),
                _ => {
                    return Err(Diagnostic::new(
                        self.span(),
                        "matrix role must be 'a', 'b' or 'acc'".to_string(),
                    ));
                }
            };
            let role = match role {
                "a" => crate::ir::MatrixRole::A,
                "b" => crate::ir::MatrixRole::B,
                "acc" => crate::ir::MatrixRole::Acc,
                _ => {
                    return Err(Diagnostic::new(
                        self.span(),
                        "matrix role must be 'a', 'b' or 'acc'".to_string(),
                    ));
                }
            };
            self.bump();
            self.expect(Tok::Gt)?;
            return Ok(Type::Matrix { elem, role });
        }
        Ok(Type::Scalar(self.parse_scalar()?))
    }

    fn parse_kernel(&mut self) -> Result<Kernel> {
        let head = self.span();
        self.expect_ident_span("kernel")?;
        let (name, _) = self.expect_ident()?;
        self.expect(Tok::LBracket)?;
        self.expect_ident_span("workgroup")?;
        self.expect(Tok::LParen)?;
        let mut workgroup_size = [0u32; 3];
        for (index, slot) in workgroup_size.iter_mut().enumerate() {
            let (value, _) = self.expect_int()?;
            *slot = value as u32;
            if index < 2 {
                self.expect(Tok::Comma)?;
            }
        }
        self.expect(Tok::RParen)?;
        self.expect(Tok::RBracket)?;
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            loop {
                let (pname, _) = self.expect_ident()?;
                self.expect(Tok::Colon)?;
                let ty = self.parse_type()?;
                match ty {
                    Type::Buf(_) | Type::Scalar(_) => params.push(Param { name: pname, ty }),
                    _ => {
                        return Err(Diagnostic::new(
                            self.span(),
                            format!("parameter '{pname}' must be buf<scalar> or scalar"),
                        ));
                    }
                }
                if matches!(self.peek(), Some(Tok::Comma)) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(Tok::RParen)?;
        let body = self.parse_block()?;
        let end = self.tokens.last().map(|t| t.span.end).unwrap_or(head.end);
        Ok(Kernel {
            name,
            workgroup_size,
            params,
            body,
            span: Span {
                start: head.start,
                end,
            },
        })
    }

    fn expect_ident_span(&mut self, name: &str) -> Result<()> {
        let span = self.span();
        match self.peek() {
            Some(Tok::Ident(n)) if n == name => {
                self.bump();
                Ok(())
            }
            _ => Err(Diagnostic::new(
                span,
                format!("expected '{name}', found {}", self.tok_desc_at(0)),
            )),
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>> {
        self.expect(Tok::LBrace)?;
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Some(Tok::RBrace) | None) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        let span = self.span();
        if self.is_ident("let") {
            self.bump();
            let (name, _) = self.expect_ident()?;
            let ty = if matches!(self.peek(), Some(Tok::Colon)) {
                self.bump();
                Some(self.parse_type()?)
            } else {
                None
            };
            self.expect(Tok::Eq)?;
            let init = self.parse_expr()?;
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Let {
                name,
                ty,
                init,
                span,
            });
        }
        if self.is_ident("var") {
            self.bump();
            let (name, _) = self.expect_ident()?;
            let ty = if matches!(self.peek(), Some(Tok::Colon)) {
                self.bump();
                Some(self.parse_type()?)
            } else {
                None
            };
            self.expect(Tok::Eq)?;
            let init = self.parse_expr()?;
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Var {
                name,
                ty,
                init,
                span,
            });
        }
        if self.is_ident("if") {
            self.bump();
            let cond = self.parse_expr()?;
            let then = self.parse_block()?;
            let mut els = Vec::new();
            if self.eat_ident("else").is_some() {
                if self.is_ident("if") {
                    els.push(self.parse_stmt()?);
                } else {
                    els = self.parse_block()?;
                }
            }
            return Ok(Stmt::If {
                cond,
                then,
                els,
                span,
            });
        }
        if self.is_ident("loop") {
            self.bump();
            let body = self.parse_block()?;
            return Ok(Stmt::Loop { body, span });
        }
        if self.is_ident("for") {
            self.bump();
            let (var, _) = self.expect_ident()?;
            self.expect_ident_span("in")?;
            let start = self.parse_expr()?;
            self.expect(Tok::Range)?;
            let end = self.parse_expr()?;
            let unroll = self.eat_ident("unroll").is_some();
            let body = self.parse_block()?;
            return Ok(Stmt::For {
                var,
                start,
                end,
                body,
                unroll,
                span,
            });
        }
        if self.is_ident("shared") {
            self.bump();
            let (name, _) = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            self.expect(Tok::LBracket)?;
            let elem = self.parse_scalar()?;
            self.expect(Tok::Semicolon)?;
            let len = self.parse_expr()?;
            self.expect(Tok::RBracket)?;
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Shared {
                name,
                elem,
                len,
                span,
            });
        }
        if self.is_ident("const") {
            self.bump();
            let (name, _) = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            let ty = self.parse_type()?;
            let Type::Scalar(scalar) = ty else {
                return Err(Diagnostic::new(
                    span,
                    "const type must be scalar".to_string(),
                ));
            };
            self.expect(Tok::Eq)?;
            let init = self.parse_expr()?;
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Const {
                name,
                ty: scalar,
                init,
                span,
            });
        }
        if self.is_ident("break") {
            self.bump();
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Break { span });
        }
        if self.is_ident("continue") {
            self.bump();
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Continue { span });
        }
        let expr = self.parse_expr()?;
        let compound = match self.peek() {
            Some(Tok::Eq) => {
                self.bump();
                None
            }
            Some(Tok::PlusEq) | Some(Tok::MinusEq) | Some(Tok::StarEq) | Some(Tok::SlashEq)
            | Some(Tok::PercentEq) | Some(Tok::AmpEq) | Some(Tok::PipeEq) | Some(Tok::CaretEq)
            | Some(Tok::ShlEq) | Some(Tok::ShrEq) => Some(self.bump().unwrap().tok),
            _ => {
                self.expect(Tok::Semicolon)?;
                return Ok(Stmt::ExprStmt { expr, span });
            }
        };
        let value = self.parse_expr()?;
        self.expect(Tok::Semicolon)?;
        let value = match compound {
            None => value,
            Some(op) => {
                let binop = match op {
                    Tok::PlusEq => BinOp::Add,
                    Tok::MinusEq => BinOp::Sub,
                    Tok::StarEq => BinOp::Mul,
                    Tok::SlashEq => BinOp::Div,
                    Tok::PercentEq => BinOp::Rem,
                    Tok::AmpEq => BinOp::And,
                    Tok::PipeEq => BinOp::Or,
                    Tok::CaretEq => BinOp::Xor,
                    Tok::ShlEq => BinOp::Shl,
                    Tok::ShrEq => BinOp::Shr,
                    _ => unreachable!(),
                };
                Expr::Binary {
                    op: binop,
                    lhs: Box::new(expr.clone()),
                    rhs: Box::new(value),
                    span,
                }
            }
        };
        Ok(Stmt::Assign {
            target: expr,
            value,
            span,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Result<Expr> {
        let cond = self.parse_or()?;
        if matches!(self.peek(), Some(Tok::Question)) {
            let span = self.span();
            self.bump();
            let then = self.parse_expr()?;
            self.expect(Tok::Colon)?;
            let els = self.parse_ternary()?;
            return Ok(Expr::Cond {
                cond: Box::new(cond),
                then: Box::new(then),
                els: Box::new(els),
                span,
            });
        }
        Ok(cond)
    }

    fn parse_binary(&mut self, next: fn(&mut Parser) -> Result<Expr>, ops: &[Tok]) -> Result<Expr> {
        let mut lhs = next(self)?;
        loop {
            let Some(op_tok) = self.peek().cloned() else {
                break;
            };
            let Some(op) = ops.iter().find(|op| **op == op_tok) else {
                break;
            };
            let span = self.span();
            self.bump();
            let rhs = next(self)?;
            let binop = match op {
                Tok::OrOr => BinOp::LOr,
                Tok::AndAnd => BinOp::LAnd,
                Tok::EqEq => BinOp::Eq,
                Tok::Ne => BinOp::Ne,
                Tok::Lt => BinOp::Lt,
                Tok::Le => BinOp::Le,
                Tok::Gt => BinOp::Gt,
                Tok::Ge => BinOp::Ge,
                Tok::Shl => BinOp::Shl,
                Tok::Shr => BinOp::Shr,
                Tok::Amp => BinOp::And,
                Tok::Caret => BinOp::Xor,
                Tok::Pipe => BinOp::Or,
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Rem,
                _ => unreachable!(),
            };
            lhs = Expr::Binary {
                op: binop,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_or(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_and, &[Tok::OrOr])
    }

    fn parse_and(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_eq, &[Tok::AndAnd])
    }

    fn parse_eq(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_rel, &[Tok::EqEq, Tok::Ne])
    }

    fn parse_rel(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_shift, &[Tok::Lt, Tok::Le, Tok::Gt, Tok::Ge])
    }

    fn parse_shift(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_band, &[Tok::Shl, Tok::Shr])
    }

    fn parse_band(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_bxor, &[Tok::Amp])
    }

    fn parse_bxor(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_bor, &[Tok::Caret])
    }

    fn parse_bor(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_add, &[Tok::Pipe])
    }

    fn parse_add(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_mul, &[Tok::Plus, Tok::Minus])
    }

    fn parse_mul(&mut self) -> Result<Expr> {
        self.parse_binary(Self::parse_unary, &[Tok::Star, Tok::Slash, Tok::Percent])
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        let span = self.span();
        if matches!(self.peek(), Some(Tok::Minus)) {
            self.bump();
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnOp::Neg,
                expr: Box::new(expr),
                span,
            });
        }
        if matches!(self.peek(), Some(Tok::Bang)) {
            self.bump();
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnOp::Not,
                expr: Box::new(expr),
                span,
            });
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            let span = self.span();
            match self.peek() {
                Some(Tok::LBracket) => {
                    self.bump();
                    let index = self.parse_expr()?;
                    self.expect(Tok::RBracket)?;
                    expr = Expr::Index {
                        base: Box::new(expr),
                        index: Box::new(index),
                        span,
                    };
                }
                Some(Tok::Dot) => {
                    let next = self.peek_at(1).cloned();
                    match next {
                        Some(Tok::Ident(component))
                            if component
                                .chars()
                                .all(|c| matches!(c, 'x' | 'y' | 'z' | 'w')) =>
                        {
                            self.bump();
                            self.bump();
                            let mask: Vec<u32> = component
                                .chars()
                                .map(|c| match c {
                                    'x' => 0,
                                    'y' => 1,
                                    'z' => 2,
                                    _ => 3,
                                })
                                .collect();
                            if mask.len() == 1 {
                                expr = Expr::Member {
                                    base: Box::new(expr),
                                    idx: mask[0],
                                    span,
                                };
                            } else {
                                expr = Expr::Swizzle {
                                    base: Box::new(expr),
                                    mask,
                                    span,
                                };
                            }
                        }
                        _ => break,
                    }
                }
                Some(Tok::Ident(_)) if self.is_ident("as") => {
                    self.bump();
                    let ty = self.parse_type()?;
                    expr = Expr::Convert {
                        ty,
                        expr: Box::new(expr),
                        span,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        let span = self.span();
        match self.peek().cloned() {
            Some(Tok::Int(value)) => {
                self.bump();
                Ok(Expr::IntLit(value, span))
            }
            Some(Tok::Float(value)) => {
                self.bump();
                Ok(Expr::FloatLit(value, span))
            }
            Some(Tok::Ident(name)) if name == "true" => {
                self.bump();
                Ok(Expr::BoolLit(true, span))
            }
            Some(Tok::Ident(name)) if name == "false" => {
                self.bump();
                Ok(Expr::BoolLit(false, span))
            }
            Some(Tok::Ident(name)) => {
                if matches!(self.peek_at(1), Some(Tok::Lt))
                    && matches!(name.as_str(), "vec2" | "vec3" | "vec4")
                {
                    let size = name.as_bytes()[3] - b'0';
                    self.bump();
                    self.bump();
                    let elem = self.parse_scalar()?;
                    self.expect(Tok::Gt)?;
                    self.expect(Tok::LParen)?;
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Some(Tok::Comma)) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Tok::RParen)?;
                    if args.len() != size as usize {
                        return Err(Diagnostic::new(
                            span,
                            format!("vec{size} constructor expects {size} arguments"),
                        ));
                    }
                    return Ok(Expr::Construct {
                        ty: Type::Vec {
                            size: size as u32,
                            elem,
                        },
                        args,
                        span,
                    });
                }
                if matches!(self.peek_at(1), Some(Tok::LParen)) {
                    self.bump();
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Some(Tok::Comma)) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Tok::RParen)?;
                    Ok(Expr::Call { name, args, span })
                } else {
                    self.bump();
                    Ok(Expr::Name(name, span))
                }
            }
            Some(Tok::LParen) => {
                self.bump();
                let expr = self.parse_expr()?;
                self.expect(Tok::RParen)?;
                Ok(expr)
            }
            _ => Err(Diagnostic::new(
                span,
                format!("expected expression, found {}", self.tok_desc_at(0)),
            )),
        }
    }
}

pub fn parse(tokens: &[Token]) -> Result<Kernel> {
    let mut parser = Parser {
        tokens: tokens.to_vec(),
        pos: 0,
    };
    let kernel = parser.parse_kernel()?;
    if parser.pos != parser.tokens.len() {
        return Err(Diagnostic::new(
            parser.span(),
            format!("unexpected trailing input {}", parser.tok_desc_at(0)),
        ));
    }
    Ok(kernel)
}

fn tok_desc(tok: &Tok) -> String {
    match tok {
        Tok::Ident(name) => format!("'{name}'"),
        Tok::Int(value) => value.to_string(),
        Tok::Float(value) => value.to_string(),
        Tok::LBrace => "'{'".to_string(),
        Tok::RBrace => "'}'".to_string(),
        Tok::LParen => "'('".to_string(),
        Tok::RParen => "')'".to_string(),
        Tok::LBracket => "'['".to_string(),
        Tok::RBracket => "']'".to_string(),
        Tok::Comma => "','".to_string(),
        Tok::Colon => "':'".to_string(),
        Tok::Semicolon => "';'".to_string(),
        Tok::Dot => "'.'".to_string(),
        Tok::Range => "'..'".to_string(),
        Tok::Eq => "'='".to_string(),
        Tok::PlusEq => "'+='".to_string(),
        Tok::MinusEq => "'-='".to_string(),
        Tok::StarEq => "'*='".to_string(),
        Tok::SlashEq => "'/='".to_string(),
        Tok::PercentEq => "'%='".to_string(),
        Tok::AmpEq => "'&='".to_string(),
        Tok::PipeEq => "'|='".to_string(),
        Tok::CaretEq => "'^='".to_string(),
        Tok::ShlEq => "'<<='".to_string(),
        Tok::ShrEq => "'>>='".to_string(),
        Tok::Plus => "'+'".to_string(),
        Tok::Minus => "'-'".to_string(),
        Tok::Star => "'*'".to_string(),
        Tok::Slash => "'/'".to_string(),
        Tok::Percent => "'%'".to_string(),
        Tok::Amp => "'&'".to_string(),
        Tok::Pipe => "'|'".to_string(),
        Tok::Caret => "'^'".to_string(),
        Tok::Shl => "'<<'".to_string(),
        Tok::Shr => "'>>'".to_string(),
        Tok::Bang => "'!'".to_string(),
        Tok::AndAnd => "'&&'".to_string(),
        Tok::OrOr => "'||'".to_string(),
        Tok::EqEq => "'=='".to_string(),
        Tok::Ne => "'!='".to_string(),
        Tok::Lt => "'<'".to_string(),
        Tok::Le => "'<='".to_string(),
        Tok::Gt => "'>'".to_string(),
        Tok::Ge => "'>='".to_string(),
        Tok::Question => "'?'".to_string(),
    }
}
