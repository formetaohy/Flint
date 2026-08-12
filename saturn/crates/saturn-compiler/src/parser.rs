use crate::ast::{BinOp, Expr, FnDecl, FnParam, Kernel, Param, Program, SpecDecl, Stmt, StructDecl, Type};
use crate::diag::{Diagnostic, Span};
use crate::ir::{Access, MemOrder, Scalar, UnOp};
use crate::lexer::{Tok, Token};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<Diagnostic>,
}

pub fn parse(tokens: &[Token]) -> std::result::Result<Program, Vec<Diagnostic>> {
    let mut parser = Parser {
        tokens: tokens.to_vec(),
        pos: 0,
        errors: Vec::new(),
    };
    let program = parser.parse_program();
    match program {
        Ok(program) if parser.errors.is_empty() => Ok(program),
        Ok(_) => Err(parser.errors),
        Err(diag) => {
            parser.errors.push(diag);
            Err(parser.errors)
        }
    }
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

    fn expect(&mut self, tok: Tok) -> Result<Token, Diagnostic> {
        let span = self.span();
        let desc = self.tok_desc_at(0);
        match self.bump() {
            Some(token) if token.tok == tok => Ok(token),
            _ => Err(Diagnostic::new(
                span,
                format!("expected {}, found {desc}", tok_desc(&tok)),
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

    fn expect_ident(&mut self) -> Result<(String, Span), Diagnostic> {
        let span = self.span();
        let desc = self.tok_desc_at(0);
        match self.bump() {
            Some(Token {
                tok: Tok::Ident(name),
                span,
            }) => Ok((name, span)),
            _ => Err(Diagnostic::new(
                span,
                format!("expected identifier, found {desc}"),
            )),
        }
    }

    fn expect_u32(&mut self) -> Result<(u32, Span), Diagnostic> {
        let span = self.span();
        let desc = self.tok_desc_at(0);
        match self.bump() {
            Some(Token {
                tok: Tok::Int(value, Some(Scalar::U32)),
                span,
            }) => Ok((value as u32, span)),
            Some(Token {
                tok: Tok::Int(value, None),
                span,
            }) => Ok((value as u32, span)),
            Some(Token {
                tok: Tok::Int(_, _),
                span,
            }) => Err(Diagnostic::new(span, "expected u32 literal")),
            _ => Err(Diagnostic::new(
                span,
                format!("expected integer, found {desc}"),
            )),
        }
    }

    fn expect_ident_span(&mut self, name: &str) -> Result<(), Diagnostic> {
        let span = self.span();
        let desc = self.tok_desc_at(0);
        match self.peek() {
            Some(Tok::Ident(n)) if n == name => {
                self.bump();
                Ok(())
            }
            _ => Err(Diagnostic::new(
                span,
                format!("expected '{name}', found {desc}"),
            )),
        }
    }

    fn recover(&mut self) {
        let mut depth = 0usize;
        while let Some(tok) = self.peek() {
            match tok {
                Tok::LBrace => depth += 1,
                Tok::RBrace => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                }
                Tok::Semicolon if depth == 0 => {
                    self.bump();
                    return;
                }
                _ => {}
            }
            self.bump();
        }
    }

    fn parse_scalar(&mut self) -> Result<Scalar, Diagnostic> {
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

    fn parse_type(&mut self) -> Result<Type, Diagnostic> {
        if self.is_ident("buf") {
            self.bump();
            self.expect(Tok::Lt)?;
            let elem = self.parse_type()?;
            self.expect(Tok::Gt)?;
            return Ok(Type::Buf(Box::new(elem)));
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
            self.expect(Tok::Gt)?;
            return Ok(Type::Matrix(elem));
        }
        if self.is_ident("threadgroup") {
            self.bump();
            self.expect(Tok::Lt)?;
            let elem = self.parse_type()?;
            self.expect(Tok::Gt)?;
            return Ok(Type::Threadgroup(Box::new(elem)));
        }
        if matches!(self.peek(), Some(Tok::LBracket)) {
            self.bump();
            let elem = self.parse_type()?;
            self.expect(Tok::Semicolon)?;
            let len = self.parse_expr()?;
            self.expect(Tok::RBracket)?;
            return Ok(Type::Array {
                elem: Box::new(elem),
                len: Box::new(len),
            });
        }
        let (name, _) = self.expect_ident()?;
        let scalar = match name.as_str() {
            "f32" => Scalar::F32,
            "f16" => Scalar::F16,
            "bf16" => Scalar::Bf16,
            "i32" => Scalar::I32,
            "u32" => Scalar::U32,
            "i8" => Scalar::I8,
            "u8" => Scalar::U8,
            "bool" => Scalar::Bool,
            _ => return Ok(Type::Struct(name)),
        };
        Ok(Type::Scalar(scalar))
    }

    fn parse_program(&mut self) -> Result<Program, Diagnostic> {
        let mut imports = Vec::new();
        let mut structs = Vec::new();
        let mut fns = Vec::new();
        let mut kernel = None;
        loop {
            if self.is_ident("import") {
                match self.parse_import() {
                    Ok(import) => imports.push(import),
                    Err(diag) => {
                        self.errors.push(diag);
                        self.recover();
                    }
                }
            } else if self.is_ident("struct") {
                match self.parse_struct() {
                    Ok(s) => structs.push(s),
                    Err(diag) => {
                        self.errors.push(diag);
                        self.recover();
                    }
                }
            } else if self.is_ident("fn") {
                match self.parse_fn() {
                    Ok(f) => fns.push(f),
                    Err(diag) => {
                        self.errors.push(diag);
                        self.recover();
                    }
                }
            } else if self.is_ident("kernel") || matches!(self.peek(), Some(Tok::At)) {
                if kernel.is_some() {
                    return Err(Diagnostic::new(
                        self.span(),
                        "multiple kernels in one file".to_string(),
                    ));
                }
                kernel = Some(self.parse_kernel().map_err(|diag| {
                    self.errors.push(diag);
                    Diagnostic::new(Span::dummy(), "aborting after syntax errors".to_string())
                })?);
                break;
            } else if self.peek().is_none() {
                break;
            } else {
                let span = self.span();
                self.errors.push(Diagnostic::new(
                    span,
                    format!(
                        "expected 'import', 'struct', 'fn' or 'kernel', found {}",
                        self.tok_desc_at(0)
                    ),
                ));
                self.recover();
                if self.pos >= self.tokens.len() {
                    break;
                }
            }
        }
        Ok(Program {
            imports,
            structs,
            fns,
            kernel,
        })
    }

    fn parse_import(&mut self) -> Result<(String, Span), Diagnostic> {
        let span = self.span();
        self.expect_ident_span("import")?;
        let name = match self.bump() {
            Some(Token {
                tok: Tok::Str(name),
                span,
            }) => {
                if name.contains('/') || name.contains('\\') || name.contains("..") {
                    return Err(Diagnostic::new(
                        span,
                        "import path must be a plain file name".to_string(),
                    ));
                }
                name
            }
            _ => {
                return Err(Diagnostic::new(
                    span,
                    format!("expected string literal, found {}", self.tok_desc_at(0)),
                ));
            }
        };
        self.expect(Tok::Semicolon)?;
        Ok((name, span))
    }

    fn parse_struct(&mut self) -> Result<StructDecl, Diagnostic> {
        let head = self.span();
        self.expect_ident_span("struct")?;
        let (name, _) = self.expect_ident()?;
        self.expect(Tok::LBrace)?;
        let mut fields = Vec::new();
        loop {
            if matches!(self.peek(), Some(Tok::RBrace)) {
                break;
            }
            let (field_name, _) = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            let ty = self.parse_type()?;
            fields.push((field_name, ty));
            if matches!(self.peek(), Some(Tok::Comma)) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(Tok::RBrace)?;
        let end = self.tokens.last().map(|t| t.span.end).unwrap_or(head.end);
        Ok(StructDecl {
            name,
            fields,
            span: Span {
                start: head.start,
                end,
            },
        })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, Diagnostic> {
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        loop {
            if matches!(self.peek(), Some(Tok::RParen)) {
                break;
            }
            {
                let span = self.span();
                let mut binding = None;
                let mut access = Access::ReadWrite;
                if matches!(self.peek(), Some(Tok::At)) {
                    self.bump();
                    self.expect_ident_span("binding")?;
                    self.expect(Tok::LParen)?;
                    let (value, _) = self.expect_u32()?;
                    binding = Some(value);
                    if matches!(self.peek(), Some(Tok::Comma)) {
                        self.bump();
                        let (mode, mode_span) = self.expect_ident()?;
                        access = match mode.as_str() {
                            "readonly" => Access::ReadOnly,
                            "writeonly" => Access::WriteOnly,
                            "readwrite" => Access::ReadWrite,
                            _ => {
                                return Err(Diagnostic::new(
                                    mode_span,
                                    "buffer access must be 'readonly', 'writeonly' or 'readwrite'"
                                        .to_string(),
                                ));
                            }
                        };
                    }
                    self.expect(Tok::RParen)?;
                }
                let (name, _) = self.expect_ident()?;
                self.expect(Tok::Colon)?;
                let ty = self.parse_type()?;
                params.push(Param {
                    name,
                    ty,
                    binding,
                    access,
                    span,
                });
                if matches!(self.peek(), Some(Tok::Comma)) {
                    self.bump();
                }
            }
        }
        self.expect(Tok::RParen)?;
        Ok(params)
    }

    fn parse_fn_params(&mut self) -> Result<Vec<FnParam>, Diagnostic> {
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        loop {
            if matches!(self.peek(), Some(Tok::RParen)) {
                break;
            }
            let (name, _) = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            let ty = self.parse_type()?;
            params.push(FnParam { name, ty });
            if matches!(self.peek(), Some(Tok::Comma)) {
                self.bump();
            }
        }
        self.expect(Tok::RParen)?;
        Ok(params)
    }

    fn parse_fn(&mut self) -> Result<FnDecl, Diagnostic> {
        let head = self.span();
        self.expect_ident_span("fn")?;
        let (name, _) = self.expect_ident()?;
        let params = self.parse_fn_params()?;
        let ret = if matches!(self.peek(), Some(Tok::Minus))
            && matches!(self.peek_at(1), Some(Tok::Gt))
        {
            self.bump();
            self.bump();
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        let end = self.tokens.last().map(|t| t.span.end).unwrap_or(head.end);
        Ok(FnDecl {
            name,
            params,
            ret,
            body,
            span: Span {
                start: head.start,
                end,
            },
        })
    }

    fn parse_kernel(&mut self) -> Result<Kernel, Diagnostic> {
        let head = self.span();
        let mut workgroup_size = [0u32; 3];
        if matches!(self.peek(), Some(Tok::At)) {
            self.bump();
            self.expect_ident_span("workgroup_size")?;
            self.expect(Tok::LParen)?;
            for (index, slot) in workgroup_size.iter_mut().enumerate() {
                let (value, _) = self.expect_u32()?;
                *slot = value;
                if index < 2 {
                    self.expect(Tok::Comma)?;
                }
            }
            self.expect(Tok::RParen)?;
        }
        self.expect_ident_span("kernel")?;
        let (name, _) = self.expect_ident()?;
        let mut params = Vec::new();
        if !matches!(self.peek(), Some(Tok::LParen) | Some(Tok::LBrace)) {
            return Err(Diagnostic::new(
                self.span(),
                format!("expected '(' or '{{', found {}", self.tok_desc_at(0)),
            ));
        }
        if matches!(self.peek(), Some(Tok::LParen)) {
            params = self.parse_params()?;
        }
        let body = self.parse_block()?;
        let (specs, body): (Vec<Stmt>, Vec<Stmt>) = body
            .into_iter()
            .partition(|stmt| matches!(stmt, Stmt::Spec(_)));
        let specs = specs
            .into_iter()
            .map(|stmt| match stmt {
                Stmt::Spec(spec) => spec,
                _ => unreachable!(),
            })
            .collect();
        let end = self.tokens.last().map(|t| t.span.end).unwrap_or(head.end);
        let kernel = Kernel {
            name,
            workgroup_size,
            params,
            specs,
            structs: Vec::new(),
            body,
            span: Span {
                start: head.start,
                end,
            },
        };
        Ok(kernel)
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, Diagnostic> {
        self.expect(Tok::LBrace)?;
        let mut stmts = Vec::new();
        loop {
            match self.peek() {
                None => {
                    return Err(Diagnostic::new(
                        self.span(),
                        "unterminated block".to_string(),
                    ));
                }
                Some(Tok::RBrace) => {
                    self.bump();
                    break;
                }
                _ => match self.parse_stmt() {
                    Ok(stmt) => stmts.push(stmt),
                    Err(diag) => {
                        self.errors.push(diag);
                        self.recover();
                        if self.pos >= self.tokens.len() {
                            return Err(Diagnostic::new(
                                self.span(),
                                "unterminated block".to_string(),
                            ));
                        }
                    }
                },
            }
        }
        Ok(stmts)
    }

    fn parse_spec(&mut self) -> Result<SpecDecl, Diagnostic> {
        let span = self.span();
        self.expect_ident_span("spec")?;
        let (name, _) = self.expect_ident()?;
        self.expect(Tok::Colon)?;
        let ty = self.parse_type()?;
        let Type::Scalar(scalar) = ty else {
            return Err(Diagnostic::new(
                span,
                "spec type must be scalar".to_string(),
            ));
        };
        self.expect(Tok::Eq)?;
        let init = self.parse_expr()?;
        self.expect(Tok::Semicolon)?;
        Ok(SpecDecl {
            name,
            ty: scalar,
            init,
            span,
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.span();
        if self.is_ident("spec") {
            return Ok(Stmt::Spec(self.parse_spec()?));
        }
        if self.is_ident("let") {
            self.bump();
            let mutable = self.eat_ident("mut").is_some();
            let (name, _) = self.expect_ident()?;
            let ty = if matches!(self.peek(), Some(Tok::Colon)) {
                self.bump();
                Some(self.parse_type()?)
            } else {
                None
            };
            let init = if matches!(self.peek(), Some(Tok::Eq)) {
                self.bump();
                Some(self.parse_expr()?)
            } else {
                None
            };
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Let {
                name,
                ty,
                init,
                mutable,
                span,
            });
        }
        if self.is_ident("var") {
            return Err(Diagnostic::new(
                span,
                "the 'var' keyword is removed: use 'let mut' instead".to_string(),
            ));
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
            let body = self.parse_block()?;
            return Ok(Stmt::For {
                var,
                start,
                end,
                body,
                unroll: false,
                span,
            });
        }
        if matches!(self.peek(), Some(Tok::At))
            && matches!(self.peek_at(1), Some(Tok::Ident(n)) if n == "unroll")
        {
            self.bump();
            self.bump();
            return match self.parse_stmt()? {
                Stmt::For {
                    var,
                    start,
                    end,
                    body,
                    unroll: _,
                    span,
                } => Ok(Stmt::For {
                    var,
                    start,
                    end,
                    body,
                    unroll: true,
                    span,
                }),
                other => Ok(other),
            };
        }
        if self.is_ident("return") {
            self.bump();
            if matches!(self.peek(), Some(Tok::Semicolon)) {
                self.bump();
                return Ok(Stmt::Return { value: None, span });
            }
            let value = self.parse_expr()?;
            self.expect(Tok::Semicolon)?;
            return Ok(Stmt::Return {
                value: Some(value),
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

    fn parse_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Result<Expr, Diagnostic> {
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

    fn parse_binary(
        &mut self,
        next: fn(&mut Parser) -> Result<Expr, Diagnostic>,
        ops: &[Tok],
    ) -> Result<Expr, Diagnostic> {
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

    fn parse_or(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_and, &[Tok::OrOr])
    }

    fn parse_and(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_eq, &[Tok::AndAnd])
    }

    fn parse_eq(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_rel, &[Tok::EqEq, Tok::Ne])
    }

    fn parse_rel(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_shift, &[Tok::Lt, Tok::Le, Tok::Gt, Tok::Ge])
    }

    fn parse_shift(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_band, &[Tok::Shl, Tok::Shr])
    }

    fn parse_band(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_bxor, &[Tok::Amp])
    }

    fn parse_bxor(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_bor, &[Tok::Caret])
    }

    fn parse_bor(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_add, &[Tok::Pipe])
    }

    fn parse_add(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_mul, &[Tok::Plus, Tok::Minus])
    }

    fn parse_mul(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_binary(Self::parse_unary, &[Tok::Star, Tok::Slash, Tok::Percent])
    }

    fn parse_unary(&mut self) -> Result<Expr, Diagnostic> {
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

    fn parse_postfix(&mut self) -> Result<Expr, Diagnostic> {
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
                        Some(Tok::Ident(name)) => {
                            self.bump();
                            self.bump();
                            expr = Expr::Field {
                                base: Box::new(expr),
                                name,
                                span,
                            };
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

    fn parse_primary(&mut self) -> Result<Expr, Diagnostic> {
        let span = self.span();
        match self.peek().cloned() {
            Some(Tok::Int(value, ty)) => {
                self.bump();
                Ok(Expr::IntLit { value, ty, span })
            }
            Some(Tok::Float(value, ty)) => {
                self.bump();
                Ok(Expr::FloatLit { value, ty, span })
            }
            Some(Tok::Dot) => {
                self.bump();
                let (name, name_span) = self.expect_ident()?;
                let order = match name.as_str() {
                    "relaxed" => MemOrder::Relaxed,
                    "acquire" => MemOrder::Acquire,
                    "release" => MemOrder::Release,
                    "acq_rel" => MemOrder::AcqRel,
                    "seq_cst" => MemOrder::SeqCst,
                    _ => {
                        return Err(Diagnostic::new(
                            name_span,
                            format!(
                                "unknown memory order '.{name}', expected relaxed, acquire, \
                                 release, acq_rel or seq_cst"
                            ),
                        ));
                    }
                };
                Ok(Expr::OrderLit(order, span))
            }
            Some(Tok::Ident(name)) => {
                if name == "true" {
                    self.bump();
                    return Ok(Expr::BoolLit { value: true, span });
                }
                if name == "false" {
                    self.bump();
                    return Ok(Expr::BoolLit { value: false, span });
                }
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
                if matches!(self.peek_at(1), Some(Tok::LBrace))
                    && matches!(self.peek_at(3), Some(Tok::Colon))
                {
                    self.bump();
                    self.bump();
                    let mut fields = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RBrace)) {
                        loop {
                            let (field_name, _) = self.expect_ident()?;
                            self.expect(Tok::Colon)?;
                            let value = self.parse_expr()?;
                            fields.push((field_name, value));
                            if matches!(self.peek(), Some(Tok::Comma)) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Tok::RBrace)?;
                    return Ok(Expr::ConstructStruct { name, fields, span });
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


fn tok_desc(tok: &Tok) -> String {
    match tok {
        Tok::Ident(name) => format!("'{name}'"),
        Tok::Str(_) => "string literal".to_string(),
        Tok::Int(value, _) => value.to_string(),
        Tok::Float(value, _) => value.to_string(),
        Tok::At => "'@'".to_string(),
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
