use std::collections::HashMap;

use crate::ast::{self, BinOp, UnOp};
use crate::consts::{self, CVal};
use crate::diag::{Diagnostic, Result, Span};
use crate::ir::{self, MatrixRole, Scalar, Type};

#[derive(Debug, Clone, PartialEq)]
enum Sym {
    BufParam(Scalar),
    ScalarParam(Scalar),
    Local { id: u32, ty: Type, mutable: bool },
    ForVar { id: u32 },
    Shared { elem: Scalar, len: u64 },
    InlineConst(CVal),
}

struct Checker {
    params: Vec<ir::Param>,
    scalars: Vec<ir::ScalarParam>,
    shareds: Vec<ir::Shared>,
    scopes: Vec<HashMap<String, Sym>>,
    consts: HashMap<String, (CVal, Scalar)>,
    loop_depth: u32,
    block_depth: u32,
    next_id: u32,
    coop_triples: Vec<(Scalar, Scalar, Scalar)>,
    coop_roles: Vec<(Scalar, ir::MatrixRole)>,
}

pub fn check(kernel: &ast::Kernel, specs: &[(&str, f64)]) -> Result<ir::Kernel> {
    for axis in kernel.workgroup_size {
        if axis == 0 || axis > 1024 {
            return Err(Diagnostic::new(
                kernel.span,
                format!(
                    "workgroup size {axis} out of range 1..=1024 for kernel {}",
                    kernel.name
                ),
            ));
        }
    }
    let mut consts = HashMap::new();
    for spec in &kernel.specs {
        if is_reserved(&spec.name) || spec.name.starts_with("__sat") {
            return Err(Diagnostic::new(
                spec.span,
                format!("'{}' is a reserved name", spec.name),
            ));
        }
        let value = consts::const_eval(&spec.init, &consts).ok_or_else(|| {
            Diagnostic::new(
                spec.span,
                format!("spec '{}' initializer must be a constant expression", spec.name),
            )
        })?;
        if !consts::validate(&value, spec.ty) {
            return Err(Diagnostic::new(
                spec.span,
                format!("spec '{}' initializer out of range for {:?}", spec.name, spec.ty),
            ));
        }
        consts.insert(spec.name.clone(), (value, spec.ty));
    }
    for (name, value) in specs {
        let Some((_, ty)) = consts.get(*name).copied() else {
            return Err(Diagnostic::new(
                kernel.span,
                format!("unknown spec '{name}'"),
            ));
        };
        let cval = match ty {
            Scalar::U32 | Scalar::I32 | Scalar::U8 | Scalar::I8 => {
                let v = *value as i64;
                let (min, max) = match ty {
                    Scalar::U32 => (0, u32::MAX as i64),
                    Scalar::I32 => (i32::MIN as i64, i32::MAX as i64),
                    Scalar::U8 => (0, u8::MAX as i64),
                    _ => (i8::MIN as i64, i8::MAX as i64),
                };
                if v < min || v > max {
                    return Err(Diagnostic::new(
                        kernel.span,
                        format!("spec '{name}' value {value} out of range for {ty:?}"),
                    ));
                }
                CVal::Int(v as u64)
            }
            Scalar::F32 | Scalar::F16 | Scalar::Bf16 => CVal::Float(*value),
            Scalar::Bool => CVal::Bool(*value != 0.0),
        };
        consts.insert(name.to_string(), (cval, ty));
    }
    let mut params = Vec::new();
    let mut scalars = Vec::new();
    let mut scope = HashMap::new();
    let mut scalar_offset = 0u32;
    for param in kernel.params.iter() {
        let name = param.name.clone();
        if is_reserved(&name) || name.starts_with("__sat") {
            return Err(Diagnostic::new(
                kernel.span,
                format!("'{name}' is a reserved name"),
            ));
        }
        match param.ty {
            ast::Type::Buf(elem) => {
                if elem == Scalar::Bool {
                    return Err(Diagnostic::new(
                        kernel.span,
                        format!(
                            "parameter '{name}' cannot be buf<bool>: Vulkan forbids bool in externally visible storage"
                        ),
                    ));
                }
                if scope.insert(name.clone(), Sym::BufParam(elem)).is_some() {
                    return Err(Diagnostic::new(
                        kernel.span,
                        format!("duplicate parameter '{name}'"),
                    ));
                }
                params.push(ir::Param {
                    name,
                    elem,
                    binding: params.len() as u32,
                });
            }
            ast::Type::Scalar(ty) => {
                if ty == Scalar::Bool {
                    return Err(Diagnostic::new(
                        kernel.span,
                        format!(
                            "parameter '{name}' cannot be bool: Vulkan forbids bool in externally visible storage"
                        ),
                    ));
                }
                if scope.insert(name.clone(), Sym::ScalarParam(ty)).is_some() {
                    return Err(Diagnostic::new(
                        kernel.span,
                        format!("duplicate parameter '{name}'"),
                    ));
                }
                scalar_offset = scalar_offset.div_ceil(4) * 4;
                scalars.push(ir::ScalarParam {
                    name,
                    ty,
                    offset: scalar_offset,
                });
                scalar_offset += ty.width();
            }
            _ => {
                return Err(Diagnostic::new(
                    kernel.span,
                    format!("parameter '{}' must be buf<scalar> or scalar", param.name),
                ));
            }
        }
    }
    let mut checker = Checker {
        params,
        scalars,
        shareds: Vec::new(),
        scopes: vec![scope],
        consts,
        loop_depth: 0,
        block_depth: 0,
        next_id: 1,
        coop_triples: Vec::new(),
        coop_roles: Vec::new(),
    };
    let body = checker.check_stmts(&kernel.body)?;
    Ok(ir::Kernel {
        name: kernel.name.clone(),
        workgroup_size: kernel.workgroup_size,
        params: checker.params,
        scalars: checker.scalars,
        shareds: checker.shareds,
        coop_triples: checker.coop_triples,
        coop_roles: checker.coop_roles,
        body,
    })
}

impl Checker {
    fn scope(&mut self) -> &mut HashMap<String, Sym> {
        self.scopes.last_mut().unwrap()
    }

    fn lookup(&self, name: &str) -> Option<Sym> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).cloned())
    }

    fn declare(&mut self, name: &str, sym: Sym, span: Span) -> Result<()> {
        if is_reserved(name) {
            return Err(Diagnostic::new(
                span,
                format!("'{name}' is a reserved name"),
            ));
        }
        if self.consts.contains_key(name) {
            return Err(Diagnostic::new(
                span,
                format!("'{name}' is already a const or spec"),
            ));
        }
        if self.scope().insert(name.to_string(), sym).is_some() {
            return Err(Diagnostic::new(
                span,
                format!("duplicate variable '{name}'"),
            ));
        }
        Ok(())
    }

    fn check_stmts(&mut self, stmts: &[ast::Stmt]) -> Result<Vec<ir::Stmt>> {
        let mut out = Vec::new();
        for stmt in stmts {
            out.extend(self.check_stmt(stmt)?);
        }
        Ok(out)
    }

    fn check_stmt(&mut self, stmt: &ast::Stmt) -> Result<Vec<ir::Stmt>> {
        match stmt {
            ast::Stmt::Return { span, .. } => Err(Diagnostic::new(
                *span,
                "return is only allowed inside functions".to_string(),
            )),
            ast::Stmt::Shared {
                name,
                elem,
                len,
                span,
            } => {
                if self.block_depth != 0 {
                    return Err(Diagnostic::new(
                        *span,
                        "shared must be declared at kernel top level".to_string(),
                    ));
                }
                let value = match consts::const_eval(len, &self.consts) {
                    Some(CVal::Int(value)) => value,
                    _ => {
                        return Err(Diagnostic::new(
                            *span,
                            "shared size must be a constant expression".to_string(),
                        ));
                    }
                };
                if value == 0 {
                    return Err(Diagnostic::new(
                        *span,
                        "shared size must be positive".to_string(),
                    ));
                }
                self.declare(
                    name,
                    Sym::Shared {
                        elem: *elem,
                        len: value,
                    },
                    *span,
                )?;
                self.shareds.push(ir::Shared {
                    name: name.clone(),
                    elem: *elem,
                    len: value,
                });
                Ok(Vec::new())
            }
            ast::Stmt::Const {
                name,
                ty,
                init,
                span,
            } => {
                if self.block_depth != 0 {
                    return Err(Diagnostic::new(
                        *span,
                        "const must be declared at kernel top level".to_string(),
                    ));
                }
                if self.consts.contains_key(name) || self.lookup(name).is_some() {
                    return Err(Diagnostic::new(*span, format!("duplicate name '{name}'")));
                }
                let value = consts::const_eval(init, &self.consts).ok_or_else(|| {
                    Diagnostic::new(
                        *span,
                        "const initializer must be a constant expression".to_string(),
                    )
                })?;
                if !consts::validate(&value, *ty) {
                    return Err(Diagnostic::new(
                        *span,
                        format!("const initializer out of range for {ty:?}"),
                    ));
                }
                self.consts.insert(name.clone(), (value, *ty));
                Ok(Vec::new())
            }
            ast::Stmt::Spec(spec) => Err(Diagnostic::new(
                spec.span,
                "spec must be declared at kernel top level".to_string(),
            )),
            ast::Stmt::Let {
                name,
                ty,
                init,
                span,
            } => {
                let (init, init_ty) = match ty {
                    Some(ast::Type::Scalar(scalar)) => {
                        let (expr, ty) = self.check_expr(init, Some(Type::Scalar(*scalar)))?;
                        (expr, ty)
                    }
                    Some(ast::Type::Matrix { elem, role }) => {
                        check_matrix_elem(*elem, *span)?;
                        let (expr, ty) = self.check_expr(
                            init,
                            Some(Type::Matrix {
                                elem: *elem,
                                role: *role,
                            }),
                        )?;
                        (expr, ty)
                    }
                    Some(ast::Type::Vec { size, elem }) => {
                        let (expr, ty) = self.check_expr(
                            init,
                            Some(Type::Vec {
                                size: *size,
                                elem: *elem,
                            }),
                        )?;
                        (expr, ty)
                    }
                    Some(_) => {
                        return Err(Diagnostic::new(
                            *span,
                            "local variables must be scalar, matrix or vector".to_string(),
                        ));
                    }
                    None if let Some(value) = consts::const_eval(init, &self.consts) => {
                        if consts::may_negate(init) {
                            let (init, init_ty) = self.check_expr(init, None)?;
                            (init, init_ty)
                        } else {
                            self.declare(name, Sym::InlineConst(value), *span)?;
                            return Ok(Vec::new());
                        }
                    }
                    None => self.check_expr(init, None)?,
                };
                if !matches!(
                    init_ty,
                    Type::Scalar(_) | Type::Matrix { .. } | Type::Vec { .. }
                ) {
                    return Err(Diagnostic::new(
                        *span,
                        "local variables must be scalar, matrix or vector".to_string(),
                    ));
                }
                let id = self.next_id;
                self.next_id += 1;
                self.declare(
                    name,
                    Sym::Local {
                        id,
                        ty: init_ty,
                        mutable: false,
                    },
                    *span,
                )?;
                Ok(vec![ir::Stmt::Let {
                    id,
                    name: name.clone(),
                    ty: init_ty,
                    init,
                    span: *span,
                }])
            }
            ast::Stmt::Var {
                name,
                ty,
                init,
                span,
            } => {
                let (init, init_ty) = match ty {
                    Some(ast::Type::Scalar(scalar)) => {
                        let (expr, ty) = self.check_expr(init, Some(Type::Scalar(*scalar)))?;
                        (expr, ty)
                    }
                    Some(ast::Type::Matrix { elem, role }) => {
                        check_matrix_elem(*elem, *span)?;
                        let (expr, ty) = self.check_expr(
                            init,
                            Some(Type::Matrix {
                                elem: *elem,
                                role: *role,
                            }),
                        )?;
                        (expr, ty)
                    }
                    Some(ast::Type::Vec { size, elem }) => {
                        let (expr, ty) = self.check_expr(
                            init,
                            Some(Type::Vec {
                                size: *size,
                                elem: *elem,
                            }),
                        )?;
                        (expr, ty)
                    }
                    Some(_) => {
                        return Err(Diagnostic::new(
                            *span,
                            "local variables must be scalar, matrix or vector".to_string(),
                        ));
                    }
                    None => {
                        let expect = if matches!(init, ast::Expr::IntLit(..)) {
                            Some(Type::Scalar(Scalar::U32))
                        } else {
                            None
                        };
                        self.check_expr(init, expect)?
                    }
                };
                if !matches!(
                    init_ty,
                    Type::Scalar(_) | Type::Matrix { .. } | Type::Vec { .. }
                ) {
                    return Err(Diagnostic::new(
                        *span,
                        "local variables must be scalar, matrix or vector".to_string(),
                    ));
                }
                let id = self.next_id;
                self.next_id += 1;
                self.declare(
                    name,
                    Sym::Local {
                        id,
                        ty: init_ty,
                        mutable: true,
                    },
                    *span,
                )?;
                Ok(vec![ir::Stmt::Var {
                    id,
                    name: name.clone(),
                    ty: init_ty,
                    init,
                    span: *span,
                }])
            }
            ast::Stmt::Assign {
                target,
                value,
                span,
            } => {
                let (target, target_ty) = self.check_target(target)?;
                let (value, _) = self.check_expr(value, Some(target_ty))?;
                Ok(vec![ir::Stmt::Assign {
                    target,
                    value,
                    span: *span,
                }])
            }
            ast::Stmt::If {
                cond,
                then,
                els,
                span,
            } => {
                let (cond, _) = self.check_expr(cond, Some(Type::Scalar(Scalar::Bool)))?;
                self.scopes.push(HashMap::new());
                self.block_depth += 1;
                let then = self.check_stmts(then)?;
                self.block_depth -= 1;
                self.scopes.pop();
                self.scopes.push(HashMap::new());
                self.block_depth += 1;
                let els = self.check_stmts(els)?;
                self.block_depth -= 1;
                self.scopes.pop();
                Ok(vec![ir::Stmt::If {
                    cond,
                    then,
                    els,
                    span: *span,
                }])
            }
            ast::Stmt::Loop { body, span } => {
                self.scopes.push(HashMap::new());
                self.loop_depth += 1;
                self.block_depth += 1;
                let body = self.check_stmts(body)?;
                self.block_depth -= 1;
                self.loop_depth -= 1;
                self.scopes.pop();
                Ok(vec![ir::Stmt::Loop { body, span: *span }])
            }
            ast::Stmt::For {
                var,
                start,
                end,
                body,
                unroll,
                span,
            } => {
                if *unroll {
                    let Some(CVal::Int(start_v)) = consts::const_eval(start, &self.consts) else {
                        return Err(Diagnostic::new(
                            *span,
                            "unrolled loop bounds must be constant".to_string(),
                        ));
                    };
                    let Some(CVal::Int(end_v)) = consts::const_eval(end, &self.consts) else {
                        return Err(Diagnostic::new(
                            *span,
                            "unrolled loop bounds must be constant".to_string(),
                        ));
                    };
                    if end_v < start_v || end_v - start_v > 256 {
                        return Err(Diagnostic::new(
                            *span,
                            "unrolled loop body count out of range 0..=256".to_string(),
                        ));
                    }
                    let mut expanded = Vec::new();
                    for value in start_v..end_v {
                        self.scopes.push(HashMap::new());
                        self.declare(
                            var,
                            Sym::Local {
                                id: self.next_id,
                                ty: Type::Scalar(Scalar::U32),
                                mutable: false,
                            },
                            *span,
                        )?;
                        self.next_id += 1;
                        self.block_depth += 1;
                        let mut iter_body = self.check_stmts(body)?;
                        self.block_depth -= 1;
                        self.scopes.pop();
                        for stmt in &mut iter_body {
                            substitute_loop_var(stmt, var, value);
                        }
                        expanded.extend(iter_body);
                    }
                    return Ok(expanded);
                }
                let (start, _) = self.check_expr(start, Some(Type::Scalar(Scalar::U32)))?;
                let (end, _) = self.check_expr(end, Some(Type::Scalar(Scalar::U32)))?;
                self.scopes.push(HashMap::new());
                let id = self.next_id;
                self.next_id += 1;
                self.declare(var, Sym::ForVar { id }, *span)?;
                self.loop_depth += 1;
                self.block_depth += 1;
                let body = self.check_stmts(body)?;
                self.block_depth -= 1;
                self.loop_depth -= 1;
                self.scopes.pop();
                Ok(vec![ir::Stmt::For {
                    id,
                    var: var.clone(),
                    start,
                    end,
                    body,
                    span: *span,
                }])
            }
            ast::Stmt::Break { span } => {
                if self.loop_depth == 0 {
                    return Err(Diagnostic::new(*span, "break outside loop".to_string()));
                }
                Ok(vec![ir::Stmt::Break { span: *span }])
            }
            ast::Stmt::Continue { span } => {
                if self.loop_depth == 0 {
                    return Err(Diagnostic::new(*span, "continue outside loop".to_string()));
                }
                Ok(vec![ir::Stmt::Continue { span: *span }])
            }
            ast::Stmt::ExprStmt { expr, span } => {
                let ast::Expr::Call {
                    name,
                    args,
                    span: call_span,
                } = expr
                else {
                    return Err(Diagnostic::new(
                        *span,
                        "expression statement must be a call".to_string(),
                    ));
                };
                if name == "barrier" && args.is_empty() {
                    return Ok(vec![ir::Stmt::Barrier { span: *span }]);
                }
                if name == "coop_store" {
                    if args.len() != 4 {
                        return Err(Diagnostic::new(
                            *span,
                            "coop_store expects 4 arguments".to_string(),
                        ));
                    }
                    let (dst, dst_ty) = self.check_expr(&args[0], None)?;
                    match dst_ty {
                        Type::Buf(_) | Type::SharedArray { .. } | Type::Scalar(_) => {}
                        _ => {
                            return Err(Diagnostic::new(
                                *span,
                                "coop_store destination must be a buffer or shared array"
                                    .to_string(),
                            ));
                        }
                    }
                    let (mat, mat_ty) = self.check_expr(&args[1], None)?;
                    let Type::Matrix { .. } = mat_ty else {
                        return Err(Diagnostic::new(
                            *span,
                            "coop_store matrix must be a matrix".to_string(),
                        ));
                    };
                    let (stride, _) = self.check_expr(&args[2], Some(Type::Scalar(Scalar::U32)))?;
                    let layout = match consts::const_eval(&args[3], &self.consts) {
                        Some(CVal::Bool(row_major)) => ir::Expr::IntLit {
                            value: if row_major { 0 } else { 1 },
                            ty: Scalar::U32,
                            span: args[3].span(),
                        },
                        _ => {
                            return Err(Diagnostic::new(
                                *span,
                                "coop store layout must be a constant bool".to_string(),
                            ));
                        }
                    };
                    return Ok(vec![ir::Stmt::ExprStmt {
                        expr: ir::Expr::Call {
                            name: "coop_store",
                            args: vec![dst, mat, stride, layout],
                            ty: Type::Scalar(Scalar::U32),
                            span: *span,
                        },
                        span: *span,
                    }]);
                }
                let (call, _) = self.check_call(name, args, None, *call_span)?;
                Ok(vec![ir::Stmt::ExprStmt {
                    expr: call,
                    span: *span,
                }])
            }
        }
    }

    fn check_target(&mut self, target: &ast::Expr) -> Result<(ir::Expr, Type)> {
        match target {
            ast::Expr::Name(name, span) => match self.lookup(name) {
                Some(Sym::BufParam(_)) | Some(Sym::ScalarParam(_)) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to parameter '{name}'"),
                )),
                Some(Sym::Shared { .. }) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to shared array '{name}'"),
                )),
                Some(Sym::ForVar { .. }) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to loop variable '{name}'"),
                )),
                Some(Sym::InlineConst(_)) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to constant '{name}'"),
                )),
                Some(Sym::Local { mutable: false, .. }) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to immutable '{name}'"),
                )),
                Some(Sym::Local {
                    id,
                    ty,
                    mutable: true,
                    ..
                }) => Ok((
                    ir::Expr::LocalRef {
                        id,
                        name: name.clone(),
                        ty,
                        span: *span,
                    },
                    ty,
                )),
                None => Err(Diagnostic::new(
                    *span,
                    format!("undefined variable '{name}'"),
                )),
            },
            ast::Expr::Index { base, index, span } => {
                let (base, base_ty) = self.check_expr(base, None)?;
                let elem = match base_ty {
                    Type::Buf(elem) => elem,
                    Type::SharedArray { elem, .. } => elem,
                    _ => {
                        return Err(Diagnostic::new(
                            *span,
                            "index target must be a buffer or shared array".to_string(),
                        ));
                    }
                };
                let (index, _) = self.check_expr(index, Some(Type::Scalar(Scalar::U32)))?;
                Ok((
                    ir::Expr::Index {
                        base: Box::new(base),
                        index: Box::new(index),
                        ty: elem,
                        span: *span,
                    },
                    Type::Scalar(elem),
                ))
            }
            _ => Err(Diagnostic::new(
                target.span(),
                "invalid assignment target".to_string(),
            )),
        }
    }

    fn check_expr(&mut self, expr: &ast::Expr, expect: Option<Type>) -> Result<(ir::Expr, Type)> {
        let (expr, ty) = self.check_expr_inner(expr, expect)?;
        if let Some(target) = expect {
            match (&ty, &target) {
                (Type::Scalar(actual), Type::Scalar(expected)) => {
                    if actual != expected {
                        return Err(Diagnostic::new(
                            expr_span(&expr),
                            format!(
                                "type mismatch: expected {}, found {}",
                                scalar_name(*expected),
                                scalar_name(*actual)
                            ),
                        ));
                    }
                }
                (Type::Matrix { .. }, Type::Matrix { .. }) => {
                    if ty != target {
                        return Err(Diagnostic::new(
                            expr_span(&expr),
                            "matrix type mismatch".to_string(),
                        ));
                    }
                }
                (Type::Vec { .. }, Type::Vec { .. }) => {
                    if ty != target {
                        return Err(Diagnostic::new(
                            expr_span(&expr),
                            "vector type mismatch".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(Diagnostic::new(
                        expr_span(&expr),
                        "type mismatch".to_string(),
                    ));
                }
            }
        }
        Ok((expr, ty))
    }

    fn check_expr_inner(
        &mut self,
        expr: &ast::Expr,
        expect: Option<Type>,
    ) -> Result<(ir::Expr, Type)> {
        match expr {
            ast::Expr::IntLit(value, span) => {
                let scalar = match expect {
                    Some(Type::Scalar(Scalar::U32)) => {
                        if *value > u32::MAX as u64 {
                            return Err(Diagnostic::new(
                                *span,
                                format!("integer literal {value} out of u32 range"),
                            ));
                        }
                        Scalar::U32
                    }
                    Some(Type::Scalar(Scalar::I32)) => {
                        if *value > i32::MAX as u64 {
                            return Err(Diagnostic::new(
                                *span,
                                format!("integer literal {value} out of i32 range"),
                            ));
                        }
                        Scalar::I32
                    }
                    Some(Type::Scalar(Scalar::I8)) => {
                        if *value > i8::MAX as u64 {
                            return Err(Diagnostic::new(
                                *span,
                                format!("integer literal {value} out of i8 range"),
                            ));
                        }
                        Scalar::I8
                    }
                    Some(Type::Scalar(Scalar::U8)) => {
                        if *value > u8::MAX as u64 {
                            return Err(Diagnostic::new(
                                *span,
                                format!("integer literal {value} out of u8 range"),
                            ));
                        }
                        Scalar::U8
                    }
                    Some(Type::Scalar(Scalar::F32)) => {
                        return Ok((
                            ir::Expr::FloatLit {
                                value: *value as f64,
                                ty: Scalar::F32,
                                span: *span,
                            },
                            Type::Scalar(Scalar::F32),
                        ));
                    }
                    Some(Type::Scalar(Scalar::F16)) => {
                        return Ok((
                            ir::Expr::FloatLit {
                                value: *value as f64,
                                ty: Scalar::F16,
                                span: *span,
                            },
                            Type::Scalar(Scalar::F16),
                        ));
                    }
                    Some(Type::Scalar(Scalar::Bf16 | Scalar::Bool)) => {
                        return Err(Diagnostic::new(
                            *span,
                            "integer literal in bf16/bool context".to_string(),
                        ));
                    }
                    Some(Type::Matrix { .. }) => Scalar::F32,
                    Some(Type::Vec { .. })
                    | Some(Type::Buf(_))
                    | Some(Type::SharedArray { .. }) => {
                        return Err(Diagnostic::new(
                            *span,
                            "integer literal in non-scalar context".to_string(),
                        ));
                    }
                    None => {
                        if *value > u32::MAX as u64 {
                            return Err(Diagnostic::new(
                                *span,
                                format!("integer literal {value} out of u32 range"),
                            ));
                        }
                        Scalar::U32
                    }
                };
                Ok((
                    ir::Expr::IntLit {
                        value: *value,
                        ty: scalar,
                        span: *span,
                    },
                    Type::Scalar(scalar),
                ))
            }
            ast::Expr::FloatLit(value, span) => {
                let scalar = match expect {
                    Some(Type::Scalar(Scalar::F32)) | None => Scalar::F32,
                    Some(Type::Scalar(Scalar::F16)) => Scalar::F16,
                    Some(Type::Scalar(Scalar::Bf16)) => {
                        return Err(Diagnostic::new(
                            *span,
                            "float literal in bf16 context".to_string(),
                        ));
                    }
                    Some(Type::Scalar(Scalar::I32 | Scalar::U32 | Scalar::I8 | Scalar::U8)) => {
                        return Err(Diagnostic::new(
                            *span,
                            "float literal in integer context".to_string(),
                        ));
                    }
                    Some(Type::Scalar(Scalar::Bool)) => {
                        return Err(Diagnostic::new(
                            *span,
                            "float literal in bool context".to_string(),
                        ));
                    }
                    Some(Type::Matrix { .. }) => Scalar::F32,
                    Some(Type::Vec { .. })
                    | Some(Type::Buf(_))
                    | Some(Type::SharedArray { .. }) => {
                        return Err(Diagnostic::new(
                            *span,
                            "float literal in non-scalar context".to_string(),
                        ));
                    }
                };
                Ok((
                    ir::Expr::FloatLit {
                        value: *value,
                        ty: scalar,
                        span: *span,
                    },
                    Type::Scalar(scalar),
                ))
            }
            ast::Expr::BoolLit(value, span) => {
                let scalar = match expect {
                    Some(Type::Scalar(Scalar::Bool)) | None => Scalar::Bool,
                    Some(_) => {
                        return Err(Diagnostic::new(
                            *span,
                            "bool literal in numeric context".to_string(),
                        ));
                    }
                };
                Ok((
                    ir::Expr::BoolLit {
                        value: *value,
                        span: *span,
                    },
                    Type::Scalar(scalar),
                ))
            }
            ast::Expr::Name(name, span) => {
                if let Some(builtin) = builtin_var(name) {
                    if matches!(builtin, "lane" | "subgroup_id" | "subgroup_size") {
                        return Ok((
                            ir::Expr::Builtin {
                                name: builtin,
                                size: 0,
                                span: *span,
                            },
                            Type::Scalar(Scalar::U32),
                        ));
                    }
                    return Ok((
                        ir::Expr::Builtin {
                            name: builtin,
                            size: 3,
                            span: *span,
                        },
                        Type::Vec {
                            size: 3,
                            elem: Scalar::U32,
                        },
                    ));
                }
                if let Some((value, ty)) = self.consts.get(name).copied() {
                    return Ok((const_literal(value, ty, *span), Type::Scalar(ty)));
                }
                match self.lookup(name) {
                    Some(Sym::InlineConst(value)) => {
                        let lit = match value {
                            CVal::Int(v) => ast::Expr::IntLit(v, *span),
                            CVal::Float(v) => ast::Expr::FloatLit(v, *span),
                            CVal::Bool(v) => ast::Expr::BoolLit(v, *span),
                        };
                        return self.check_expr_inner(&lit, expect);
                    }
                    Some(Sym::BufParam(elem)) => Ok((
                        ir::Expr::ParamRef {
                            name: name.clone(),
                            elem,
                            span: *span,
                        },
                        Type::Buf(elem),
                    )),
                    Some(Sym::ScalarParam(ty)) => Ok((
                        ir::Expr::ScalarRef {
                            name: name.clone(),
                            ty,
                            span: *span,
                        },
                        Type::Scalar(ty),
                    )),
                    Some(Sym::Shared { elem, len }) => Ok((
                        ir::Expr::SharedRef {
                            name: name.clone(),
                            elem,
                            len,
                            span: *span,
                        },
                        Type::SharedArray { elem, len },
                    )),
                    Some(Sym::Local { id, ty, .. }) => Ok((
                        ir::Expr::LocalRef {
                            id,
                            name: name.clone(),
                            ty,
                            span: *span,
                        },
                        ty,
                    )),
                    Some(Sym::ForVar { id }) => Ok((
                        ir::Expr::LocalRef {
                            id,
                            name: name.clone(),
                            ty: Type::Scalar(Scalar::U32),
                            span: *span,
                        },
                        Type::Scalar(Scalar::U32),
                    )),
                    None => Err(Diagnostic::new(
                        *span,
                        format!("undefined variable '{name}'"),
                    )),
                }
            }
            ast::Expr::Index { base, index, span } => {
                let (base, base_ty) = self.check_expr(base, None)?;
                let elem = match base_ty {
                    Type::Buf(elem) => elem,
                    Type::SharedArray { elem, .. } => elem,
                    _ => {
                        return Err(Diagnostic::new(
                            *span,
                            "index target must be a buffer or shared array".to_string(),
                        ));
                    }
                };
                let (index, _) = self.check_expr(index, Some(Type::Scalar(Scalar::U32)))?;
                Ok((
                    ir::Expr::Index {
                        base: Box::new(base),
                        index: Box::new(index),
                        ty: elem,
                        span: *span,
                    },
                    Type::Scalar(elem),
                ))
            }
            ast::Expr::Member { base, idx, span } => {
                let (base, base_ty) = self.check_expr(base, None)?;
                let Type::Vec { size, elem } = base_ty else {
                    return Err(Diagnostic::new(
                        *span,
                        "member access requires a vector".to_string(),
                    ));
                };
                if *idx >= size {
                    return Err(Diagnostic::new(
                        *span,
                        format!("member index {idx} out of range for vec{size}"),
                    ));
                }
                Ok((
                    ir::Expr::Member {
                        base: Box::new(base),
                        idx: *idx,
                        ty: elem,
                        span: *span,
                    },
                    Type::Scalar(elem),
                ))
            }
            ast::Expr::Unary { op, expr, span } => match op {
                UnOp::Neg => {
                    if let ast::Expr::IntLit(value, lit_span) = &**expr {
                        let scalar = match expect {
                            Some(Type::Scalar(Scalar::U32 | Scalar::U8)) => {
                                return Err(Diagnostic::new(
                                    *span,
                                    "cannot negate an unsigned integer literal".to_string(),
                                ));
                            }
                            Some(Type::Scalar(Scalar::I32)) => Scalar::I32,
                            Some(Type::Scalar(Scalar::I8)) => Scalar::I8,
                            Some(Type::Scalar(Scalar::F32 | Scalar::F16)) => {
                                let scalar = match expect {
                                    Some(Type::Scalar(s)) => s,
                                    _ => unreachable!(),
                                };
                                return Ok((
                                    ir::Expr::FloatLit {
                                        value: -(*value as f64),
                                        ty: scalar,
                                        span: *span,
                                    },
                                    Type::Scalar(scalar),
                                ));
                            }
                            Some(Type::Scalar(Scalar::Bf16 | Scalar::Bool)) => {
                                return Err(Diagnostic::new(
                                    *span,
                                    "cannot negate a bf16/bool literal".to_string(),
                                ));
                            }
                            Some(Type::Matrix { .. }) | Some(Type::Vec { .. }) => {
                                return Err(Diagnostic::new(
                                    *span,
                                    "cannot negate a literal in non-scalar context".to_string(),
                                ));
                            }
                            _ => Scalar::I32,
                        };
                        let neg = (*value as i128).wrapping_neg();
                        let (min, max) = match scalar {
                            Scalar::I32 => (i32::MIN as i128, i32::MAX as i128),
                            Scalar::I8 => (i8::MIN as i128, i8::MAX as i128),
                            _ => unreachable!(),
                        };
                        if neg < min || neg > max {
                            return Err(Diagnostic::new(
                                *lit_span,
                                format!("integer literal {} out of {scalar:?} range", *value),
                            ));
                        }
                        return Ok((
                            ir::Expr::IntLit {
                                value: neg as u64,
                                ty: scalar,
                                span: *span,
                            },
                            Type::Scalar(scalar),
                        ));
                    }
                    let (expr, ty) = self.check_expr(expr, expect)?;
                    let Type::Scalar(scalar) = ty else {
                        return Err(Diagnostic::new(
                            *span,
                            "negation requires a scalar".to_string(),
                        ));
                    };
                    match scalar {
                        Scalar::U32 => Err(Diagnostic::new(*span, "cannot negate u32".to_string())),
                        Scalar::Bool => {
                            Err(Diagnostic::new(*span, "cannot negate bool".to_string()))
                        }
                        _ => Ok((
                            ir::Expr::Unary {
                                op: UnOp::Neg,
                                expr: Box::new(expr),
                                ty: scalar,
                                span: *span,
                            },
                            Type::Scalar(scalar),
                        )),
                    }
                }
                UnOp::Not => {
                    let (expr, _) = self.check_expr(expr, Some(Type::Scalar(Scalar::Bool)))?;
                    Ok((
                        ir::Expr::Unary {
                            op: UnOp::Not,
                            expr: Box::new(expr),
                            ty: Scalar::Bool,
                            span: *span,
                        },
                        Type::Scalar(Scalar::Bool),
                    ))
                }
            },
            ast::Expr::Binary { op, lhs, rhs, span } => self.check_binary(*op, lhs, rhs, *span),
            ast::Expr::Cond {
                cond,
                then,
                els,
                span,
            } => {
                let (cond, _) = self.check_expr(cond, Some(Type::Scalar(Scalar::Bool)))?;
                let is_literal = |e: &ast::Expr| {
                    matches!(
                        e,
                        ast::Expr::IntLit(..) | ast::Expr::FloatLit(..) | ast::Expr::BoolLit(..)
                    )
                };
                let (then, els, branch_ty) = if is_literal(then) && !is_literal(els) {
                    let (els, els_ty) = self.check_expr(els, None)?;
                    let Type::Scalar(els_scalar) = els_ty else {
                        return Err(Diagnostic::new(
                            *span,
                            "ternary branches must be scalar".to_string(),
                        ));
                    };
                    let (then, _) = self.check_expr(then, Some(Type::Scalar(els_scalar)))?;
                    (then, els, els_scalar)
                } else {
                    let (then, then_ty) = self.check_expr(then, None)?;
                    let Type::Scalar(then_scalar) = then_ty else {
                        return Err(Diagnostic::new(
                            *span,
                            "ternary branches must be scalar".to_string(),
                        ));
                    };
                    let (els, _) = self.check_expr(els, Some(Type::Scalar(then_scalar)))?;
                    (then, els, then_scalar)
                };
                Ok((
                    ir::Expr::Cond {
                        cond: Box::new(cond),
                        then: Box::new(then),
                        els: Box::new(els),
                        ty: branch_ty,
                        span: *span,
                    },
                    Type::Scalar(branch_ty),
                ))
            }
            ast::Expr::Convert { ty, expr, span } => {
                let ast::Type::Scalar(target) = ty else {
                    return Err(Diagnostic::new(
                        *span,
                        "conversion target must be a scalar type".to_string(),
                    ));
                };
                if *target == Scalar::Bool {
                    return Err(Diagnostic::new(
                        *span,
                        "bool cannot be a conversion target".to_string(),
                    ));
                }
                let (expr, expr_ty) = self.check_expr(expr, None)?;
                let Type::Scalar(source) = expr_ty else {
                    return Err(Diagnostic::new(
                        *span,
                        "conversion source must be scalar".to_string(),
                    ));
                };
                if source == *target {
                    return Err(Diagnostic::new(
                        *span,
                        format!("redundant conversion to same type {}", scalar_name(source)),
                    ));
                }
                if matches!(*target, Scalar::Bf16 | Scalar::I8 | Scalar::U8)
                    && matches!(source, Scalar::Bf16 | Scalar::I8 | Scalar::U8)
                {
                    return Err(Diagnostic::new(
                        *span,
                        "direct narrow conversion not allowed".to_string(),
                    ));
                }
                Ok((
                    ir::Expr::Convert {
                        ty: *target,
                        expr: Box::new(expr),
                        span: *span,
                    },
                    Type::Scalar(*target),
                ))
            }
            ast::Expr::Call { name, args, span } => self.check_call(name, args, expect, *span),
            ast::Expr::Construct { ty, args, span } => {
                let ast::Type::Vec { size, elem } = ty else {
                    return Err(Diagnostic::new(
                        *span,
                        "constructor requires a vector type".to_string(),
                    ));
                };
                let mut checked = Vec::with_capacity(args.len());
                for arg in args {
                    let (expr, _) = self.check_expr(arg, Some(Type::Scalar(*elem)))?;
                    checked.push(expr);
                }
                Ok((
                    ir::Expr::Call {
                        name: "construct_vec",
                        args: checked,
                        ty: Type::Vec {
                            size: *size,
                            elem: *elem,
                        },
                        span: *span,
                    },
                    Type::Vec {
                        size: *size,
                        elem: *elem,
                    },
                ))
            }
            ast::Expr::Swizzle { base, mask, span } => {
                let (base, base_ty) = self.check_expr(base, None)?;
                let Type::Vec { size, elem } = base_ty else {
                    return Err(Diagnostic::new(
                        *span,
                        "swizzle requires a vector".to_string(),
                    ));
                };
                for &idx in mask {
                    if idx >= size {
                        return Err(Diagnostic::new(
                            *span,
                            format!("swizzle component out of range for vec{size}"),
                        ));
                    }
                }
                if mask.len() == 1 {
                    return Ok((
                        ir::Expr::Member {
                            base: Box::new(base),
                            idx: mask[0],
                            ty: elem,
                            span: *span,
                        },
                        Type::Scalar(elem),
                    ));
                }
                let size = mask.len() as u32;
                let mut args = vec![base];
                for &idx in mask {
                    args.push(ir::Expr::IntLit {
                        value: idx as u64,
                        ty: Scalar::U32,
                        span: *span,
                    });
                }
                Ok((
                    ir::Expr::Call {
                        name: "swizzle_vec",
                        args,
                        ty: Type::Vec { size, elem },
                        span: *span,
                    },
                    Type::Vec { size, elem },
                ))
            }
        }
    }

    fn check_call(
        &mut self,
        name: &str,
        args: &[ast::Expr],
        expect: Option<Type>,
        span: Span,
    ) -> Result<(ir::Expr, Type)> {
        if matches!(
            name,
            "atomic_add"
                | "atomic_max"
                | "atomic_min"
                | "atomic_exchange"
                | "atomic_and"
                | "atomic_or"
                | "atomic_xor"
        ) {
            if args.len() != 3 {
                return Err(Diagnostic::new(span, format!("{name} expects 3 arguments")));
            }
            let (buf, buf_ty) = self.check_expr(&args[0], None)?;
            let elem = match buf_ty {
                Type::Buf(elem) | Type::SharedArray { elem, .. } => elem,
                _ => {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} target must be a buffer or shared array"),
                    ));
                }
            };
            if !elem.is_int() {
                return Err(Diagnostic::new(
                    span,
                    format!("{name} requires an integer buffer"),
                ));
            }
            let (index, _) = self.check_expr(&args[1], Some(Type::Scalar(Scalar::U32)))?;
            let (value, _) = self.check_expr(&args[2], Some(Type::Scalar(elem)))?;
            return Ok((
                ir::Expr::Call {
                    name: name_static(name),
                    args: vec![buf, index, value],
                    ty: Type::Scalar(elem),
                    span,
                },
                Type::Scalar(elem),
            ));
        }
        if name == "coop_zero" {
            if !args.is_empty() {
                return Err(Diagnostic::new(
                    span,
                    "coop_zero takes no arguments".to_string(),
                ));
            }
            let Some(Type::Matrix { elem, role }) = expect else {
                return Err(Diagnostic::new(
                    span,
                    "coop_zero requires a matrix type annotation".to_string(),
                ));
            };
            check_matrix_elem(elem, span)?;
            self.coop_roles.push((elem, MatrixRole::Acc));
            return Ok((
                ir::Expr::Call {
                    name: "coop_zero",
                    args: Vec::new(),
                    ty: Type::Matrix { elem, role },
                    span,
                },
                Type::Matrix { elem, role },
            ));
        }
        if name == "coop_load_a" || name == "coop_load_b" {
            if args.len() != 3 {
                return Err(Diagnostic::new(span, format!("{name} expects 3 arguments")));
            }
            let (src, src_ty) = self.check_expr(&args[0], None)?;
            let elem = match src_ty {
                Type::Buf(elem) => elem,
                Type::SharedArray { elem, .. } => elem,
                Type::Scalar(elem) => elem,
                _ => {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} source must be a buffer or shared array"),
                    ));
                }
            };
            let (stride, _) = self.check_expr(&args[1], Some(Type::Scalar(Scalar::U32)))?;
            let layout = match consts::const_eval(&args[2], &self.consts) {
                Some(CVal::Bool(row_major)) => ir::Expr::IntLit {
                    value: if row_major { 0 } else { 1 },
                    ty: Scalar::U32,
                    span: args[2].span(),
                },
                _ => {
                    return Err(Diagnostic::new(
                        span,
                        "coop load layout must be a constant bool".to_string(),
                    ));
                }
            };
            check_matrix_elem(elem, span)?;
            let role = if name == "coop_load_a" {
                MatrixRole::A
            } else {
                MatrixRole::B
            };
            self.coop_roles.push((elem, role));
            return Ok((
                ir::Expr::Call {
                    name: name_static(name),
                    args: vec![src, stride, layout],
                    ty: Type::Matrix { elem, role },
                    span,
                },
                Type::Matrix { elem, role },
            ));
        }
        if name == "coop_mul_add" {
            if args.len() != 3 {
                return Err(Diagnostic::new(
                    span,
                    "coop_mul_add expects 3 arguments".to_string(),
                ));
            }
            let (a, a_ty) = self.check_expr(&args[0], None)?;
            let (b, b_ty) = self.check_expr(&args[1], None)?;
            let (c, c_ty) = self.check_expr(&args[2], None)?;
            let Type::Matrix {
                elem: a_elem,
                role: MatrixRole::A,
            } = a_ty
            else {
                return Err(Diagnostic::new(
                    span,
                    "coop_mul_add first argument must be an A matrix".to_string(),
                ));
            };
            let Type::Matrix {
                elem: b_elem,
                role: MatrixRole::B,
            } = b_ty
            else {
                return Err(Diagnostic::new(
                    span,
                    "coop_mul_add second argument must be a B matrix".to_string(),
                ));
            };
            let Type::Matrix {
                elem: c_elem,
                role: MatrixRole::Acc,
            } = c_ty
            else {
                return Err(Diagnostic::new(
                    span,
                    "coop_mul_add third argument must be an accumulator matrix".to_string(),
                ));
            };
            if a_elem != b_elem {
                return Err(Diagnostic::new(
                    span,
                    "coop_mul_add A and B element types must match".to_string(),
                ));
            }
            if !c_elem.is_float() || !a_elem.is_float() {
                return Err(Diagnostic::new(
                    span,
                    "coop_mul_add requires float element types".to_string(),
                ));
            }
            self.coop_triples.push((a_elem, b_elem, c_elem));
            let ret = Type::Matrix {
                elem: c_elem,
                role: MatrixRole::Acc,
            };
            return Ok((
                ir::Expr::Call {
                    name: "coop_mul_add",
                    args: vec![a, b, c],
                    ty: ret,
                    span,
                },
                ret,
            ));
        }
        if name == "bitcast_f32" || name == "bitcast_u32" {
            if args.len() != 1 {
                return Err(Diagnostic::new(
                    span,
                    format!("{name} expects 1 argument"),
                ));
            }
            let (arg, _) = self.check_expr(
                &args[0],
                Some(Type::Scalar(if name == "bitcast_f32" {
                    Scalar::U32
                } else {
                    Scalar::F32
                })),
            )?;
            let ret = if name == "bitcast_f32" {
                Scalar::F32
            } else {
                Scalar::U32
            };
            return Ok((
                ir::Expr::Call {
                    name: name_static(name),
                    args: vec![arg],
                    ty: Type::Scalar(ret),
                    span,
                },
                Type::Scalar(ret),
            ));
        }
        if matches!(name, "popcount" | "clz" | "ctz") {
            if args.len() != 1 {
                return Err(Diagnostic::new(
                    span,
                    format!("{name} expects 1 argument"),
                ));
            }
            let (arg, _) = self.check_expr(&args[0], Some(Type::Scalar(Scalar::U32)))?;
            return Ok((
                ir::Expr::Call {
                    name: name_static(name),
                    args: vec![arg],
                    ty: Type::Scalar(Scalar::U32),
                    span,
                },
                Type::Scalar(Scalar::U32),
            ));
        }
        if name == "dot" {
            if args.len() != 2 {
                return Err(Diagnostic::new(span, "dot expects 2 arguments"));
            }
            let (a, a_ty) = self.check_expr(&args[0], None)?;
            let Type::Vec { size, elem } = a_ty else {
                return Err(Diagnostic::new(span, "dot requires vector arguments"));
            };
            if !elem.is_float() {
                return Err(Diagnostic::new(
                    span,
                    "dot requires float vector arguments".to_string(),
                ));
            }
            let (b, _) = self.check_expr(&args[1], Some(Type::Vec { size, elem }))?;
            return Ok((
                ir::Expr::Call {
                    name: "dot",
                    args: vec![a, b],
                    ty: Type::Scalar(elem),
                    span,
                },
                Type::Scalar(elem),
            ));
        }
        let builtin = match name {
            "min"
            | "max"
            | "abs"
            | "floor"
            | "ceil"
            | "round"
            | "trunc"
            | "sign"
            | "fract"
            | "sqrt"
            | "rsqrt"
            | "exp"
            | "exp2"
            | "log"
            | "log2"
            | "tanh"
            | "pow"
            | "clamp"
            | "fma"
            | "select"
            | "subgroup_broadcast"
            | "subgroup_shuffle"
            | "subgroup_shuffle_down"
            | "subgroup_shuffle_up"
            | "subgroup_reduce_add"
            | "subgroup_reduce_max"
            | "subgroup_reduce_min"
            | "subgroup_inclusive_add"
            | "subgroup_all"
            | "subgroup_any" => name_static(name),
            _ => {
                return Err(Diagnostic::new(span, format!("unknown builtin '{name}'")));
            }
        };
        let arity = match builtin {
            "abs"
            | "floor"
            | "ceil"
            | "round"
            | "trunc"
            | "sign"
            | "fract"
            | "sqrt"
            | "rsqrt"
            | "exp"
            | "exp2"
            | "log"
            | "log2"
            | "tanh"
            | "subgroup_reduce_add"
            | "subgroup_reduce_max"
            | "subgroup_reduce_min"
            | "subgroup_inclusive_add"
            | "subgroup_all"
            | "subgroup_any" => 1,
            "min"
            | "max"
            | "pow"
            | "subgroup_broadcast"
            | "subgroup_shuffle"
            | "subgroup_shuffle_down"
            | "subgroup_shuffle_up" => 2,
            "clamp" | "fma" | "select" => 3,
            _ => unreachable!(),
        };
        if args.len() != arity {
            return Err(Diagnostic::new(
                span,
                format!(
                    "builtin '{name}' expects {arity} arguments, got {}",
                    args.len()
                ),
            ));
        }
        let mut checked = Vec::with_capacity(arity);
        let (first_expr, first_ty) = self.check_expr(&args[0], None)?;
        let Type::Scalar(first_scalar) = first_ty else {
            return Err(Diagnostic::new(
                span,
                format!("{name} requires scalar arguments"),
            ));
        };
        checked.push((first_expr, Type::Scalar(first_scalar)));
        for (index, arg) in args.iter().enumerate().skip(1) {
            let expect = match (builtin, index) {
                ("select", 2) => Some(Type::Scalar(Scalar::Bool)),
                (
                    "subgroup_broadcast"
                    | "subgroup_shuffle"
                    | "subgroup_shuffle_down"
                    | "subgroup_shuffle_up",
                    1,
                ) => Some(Type::Scalar(Scalar::U32)),
                ("subgroup_all" | "subgroup_any", 0) => Some(Type::Scalar(Scalar::Bool)),
                _ => Some(Type::Scalar(first_scalar)),
            };
            let (expr, ty) = self.check_expr(arg, expect)?;
            checked.push((expr, ty));
        }
        let (ret, unified) = match builtin {
            "subgroup_all" | "subgroup_any" => {
                let Type::Scalar(_) = checked[0].1 else {
                    return Err(Diagnostic::new(
                        span,
                        "subgroup_all/subgroup_any require a bool argument".to_string(),
                    ));
                };
                (Scalar::Bool, Scalar::Bool)
            }
            "subgroup_broadcast"
            | "subgroup_shuffle"
            | "subgroup_shuffle_down"
            | "subgroup_shuffle_up"
            | "subgroup_reduce_add"
            | "subgroup_reduce_max"
            | "subgroup_reduce_min"
            | "subgroup_inclusive_add" => {
                let Type::Scalar(first) = checked[0].1 else {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires scalar arguments"),
                    ));
                };
                if !first.is_float() && !first.is_int() {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires numeric arguments"),
                    ));
                }
                (first, first)
            }
            "select" => {
                let Type::Scalar(a) = checked[0].1 else {
                    return Err(Diagnostic::new(
                        span,
                        "select requires numeric arguments".to_string(),
                    ));
                };
                let Type::Scalar(b) = checked[1].1 else {
                    return Err(Diagnostic::new(
                        span,
                        "select requires numeric arguments".to_string(),
                    ));
                };
                let Type::Scalar(Scalar::Bool) = checked[2].1 else {
                    return Err(Diagnostic::new(
                        span,
                        "select condition must be bool".to_string(),
                    ));
                };
                if a != b {
                    return Err(Diagnostic::new(
                        span,
                        format!(
                            "select branches must match: {} vs {}",
                            scalar_name(a),
                            scalar_name(b)
                        ),
                    ));
                }
                if !a.is_float() && !a.is_int() {
                    return Err(Diagnostic::new(
                        span,
                        "select requires numeric arguments".to_string(),
                    ));
                }
                (a, a)
            }
            "min" | "max" | "clamp" | "fma" | "pow" | "abs" => {
                let Type::Scalar(first) = checked[0].1 else {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires scalar arguments"),
                    ));
                };
                if !first.is_float() && !first.is_int() {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires numeric arguments"),
                    ));
                }
                if first == Scalar::U32 && builtin == "abs" {
                    return Err(Diagnostic::new(
                        span,
                        "abs is not defined for u32".to_string(),
                    ));
                }
                for (_, ty) in &checked[1..] {
                    let Type::Scalar(other) = ty else {
                        return Err(Diagnostic::new(
                            span,
                            format!("{name} requires scalar arguments"),
                        ));
                    };
                    if *other != first {
                        return Err(Diagnostic::new(
                            span,
                            format!(
                                "{name} argument types must match: {} vs {}",
                                scalar_name(first),
                                scalar_name(*other)
                            ),
                        ));
                    }
                }
                (first, first)
            }
            _ => {
                let Type::Scalar(first) = checked[0].1 else {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires a float argument"),
                    ));
                };
                if !first.is_float() {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires a float argument"),
                    ));
                }
                (first, first)
            }
        };
        let args = checked.into_iter().map(|(expr, _)| expr).collect();
        Ok((
            ir::Expr::Call {
                name: builtin,
                args,
                ty: Type::Scalar(ret),
                span,
            },
            Type::Scalar(unified),
        ))
    }

    fn check_binary(
        &mut self,
        op: BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
        span: Span,
    ) -> Result<(ir::Expr, Type)> {
        match op {
            BinOp::LAnd | BinOp::LOr => {
                let (lhs, _) = self.check_expr(lhs, Some(Type::Scalar(Scalar::Bool)))?;
                let (rhs, _) = self.check_expr(rhs, Some(Type::Scalar(Scalar::Bool)))?;
                Ok((
                    ir::Expr::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                        ty: Type::Scalar(Scalar::Bool),
                        span,
                    },
                    Type::Scalar(Scalar::Bool),
                ))
            }
            BinOp::Eq | BinOp::Ne => {
                let (lhs, lhs_ty) = self.check_expr(lhs, None)?;
                let Type::Scalar(lhs_scalar) = lhs_ty else {
                    return Err(Diagnostic::new(
                        span,
                        "comparison requires scalars".to_string(),
                    ));
                };
                let (rhs, _) = self.check_expr(rhs, Some(Type::Scalar(lhs_scalar)))?;
                Ok((
                    ir::Expr::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                        ty: Type::Scalar(Scalar::Bool),
                        span,
                    },
                    Type::Scalar(Scalar::Bool),
                ))
            }
            BinOp::And | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::Shr => {
                let (lhs, lhs_ty) = self.check_expr(lhs, None)?;
                let Type::Scalar(lhs_scalar) = lhs_ty else {
                    return Err(Diagnostic::new(
                        span,
                        "bitwise operation requires scalars".to_string(),
                    ));
                };
                if !lhs_scalar.is_int() {
                    return Err(Diagnostic::new(
                        span,
                        format!(
                            "bitwise operation requires integer operands, found {}",
                            scalar_name(lhs_scalar)
                        ),
                    ));
                }
                let (rhs, _) = self.check_expr(rhs, Some(Type::Scalar(lhs_scalar)))?;
                Ok((
                    ir::Expr::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                        ty: Type::Scalar(lhs_scalar),
                        span,
                    },
                    Type::Scalar(lhs_scalar),
                ))
            }
            _ => {
                let (lhs, lhs_ty) = self.check_expr(lhs, None)?;
                match lhs_ty {
                    Type::Scalar(lhs_scalar) => {
                        if !lhs_scalar.is_float() && !lhs_scalar.is_int() {
                            return Err(Diagnostic::new(
                                span,
                                format!(
                                    "arithmetic requires numeric operands, found {}",
                                    scalar_name(lhs_scalar)
                                ),
                            ));
                        }
                        let (rhs, _) = self.check_expr(rhs, Some(Type::Scalar(lhs_scalar)))?;
                        let ty = if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
                            Type::Scalar(Scalar::Bool)
                        } else {
                            Type::Scalar(lhs_scalar)
                        };
                        Ok((
                            ir::Expr::Binary {
                                op,
                                lhs: Box::new(lhs),
                                rhs: Box::new(rhs),
                                ty,
                                span,
                            },
                            ty,
                        ))
                    }
                    Type::Vec { size, elem } => {
                        if matches!(
                            op,
                            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Rem
                        ) || !elem.is_float()
                        {
                            return Err(Diagnostic::new(
                                span,
                                "vector arithmetic supports + - * / on float vectors".to_string(),
                            ));
                        }
                        let (rhs, _) = self.check_expr(rhs, Some(Type::Vec { size, elem }))?;
                        let ty = Type::Vec { size, elem };
                        Ok((
                            ir::Expr::Binary {
                                op,
                                lhs: Box::new(lhs),
                                rhs: Box::new(rhs),
                                ty,
                                span,
                            },
                            ty,
                        ))
                    }
                    _ => Err(Diagnostic::new(
                        span,
                        "operation requires scalars".to_string(),
                    )),
                }
            }
        }
    }
}

fn const_literal(value: CVal, ty: Scalar, span: Span) -> ir::Expr {
    match (value, ty) {
        (CVal::Int(value), Scalar::U32 | Scalar::I32) => ir::Expr::IntLit { value, ty, span },
        (CVal::Float(value), Scalar::F32 | Scalar::F16) => ir::Expr::FloatLit { value, ty, span },
        (CVal::Bool(value), Scalar::Bool) => ir::Expr::BoolLit { value, span },
        _ => ir::Expr::IntLit {
            value: 0,
            ty: Scalar::U32,
            span,
        },
    }
}

fn substitute_loop_var(stmt: &mut ir::Stmt, var: &str, value: u64) {
    match stmt {
        ir::Stmt::Let { init, .. } | ir::Stmt::Var { init, .. } => {
            substitute_loop_expr(init, var, value);
        }
        ir::Stmt::Assign {
            target, value: v, ..
        } => {
            substitute_loop_expr(target, var, value);
            substitute_loop_expr(v, var, value);
        }
        ir::Stmt::If {
            cond, then, els, ..
        } => {
            substitute_loop_expr(cond, var, value);
            for stmt in then {
                substitute_loop_var(stmt, var, value);
            }
            for stmt in els {
                substitute_loop_var(stmt, var, value);
            }
        }
        ir::Stmt::Loop { body, .. } => {
            for stmt in body {
                substitute_loop_var(stmt, var, value);
            }
        }
        ir::Stmt::For {
            start, end, body, ..
        } => {
            substitute_loop_expr(start, var, value);
            substitute_loop_expr(end, var, value);
            for stmt in body {
                substitute_loop_var(stmt, var, value);
            }
        }
        ir::Stmt::ExprStmt { expr, .. } => substitute_loop_expr(expr, var, value),
        _ => {}
    }
}

fn substitute_loop_expr(expr: &mut ir::Expr, var: &str, value: u64) {
    match expr {
        ir::Expr::LocalRef { name, span, .. } if name == var => {
            *expr = ir::Expr::IntLit {
                value,
                ty: Scalar::U32,
                span: *span,
            };
        }
        ir::Expr::Index { base, index, .. } => {
            substitute_loop_expr(base, var, value);
            substitute_loop_expr(index, var, value);
        }
        ir::Expr::Member { base, .. } => substitute_loop_expr(base, var, value),
        ir::Expr::Unary { expr: e, .. } => substitute_loop_expr(e, var, value),
        ir::Expr::Binary { lhs, rhs, .. } => {
            substitute_loop_expr(lhs, var, value);
            substitute_loop_expr(rhs, var, value);
        }
        ir::Expr::Cond {
            cond, then, els, ..
        } => {
            substitute_loop_expr(cond, var, value);
            substitute_loop_expr(then, var, value);
            substitute_loop_expr(els, var, value);
        }
        ir::Expr::Convert { expr: e, .. } => substitute_loop_expr(e, var, value),
        ir::Expr::Call { args, .. } => {
            for arg in args {
                substitute_loop_expr(arg, var, value);
            }
        }
        _ => {}
    }
}

fn expr_span(expr: &ir::Expr) -> Span {
    match expr {
        ir::Expr::IntLit { span, .. }
        | ir::Expr::FloatLit { span, .. }
        | ir::Expr::BoolLit { span, .. }
        | ir::Expr::ParamRef { span, .. }
        | ir::Expr::LocalRef { span, .. }
        | ir::Expr::ScalarRef { span, .. }
        | ir::Expr::SharedRef { span, .. }
        | ir::Expr::Builtin { span, .. }
        | ir::Expr::Index { span, .. }
        | ir::Expr::Member { span, .. }
        | ir::Expr::Unary { span, .. }
        | ir::Expr::Binary { span, .. }
        | ir::Expr::Cond { span, .. }
        | ir::Expr::Convert { span, .. }
        | ir::Expr::Call { span, .. } => *span,
    }
}

pub(crate) fn is_reserved(name: &str) -> bool {
    matches!(
        name,
        "kernel"
            | "fn"
            | "return"
            | "spec"
            | "let"
            | "var"
            | "const"
            | "shared"
            | "if"
            | "else"
            | "loop"
            | "for"
            | "in"
            | "break"
            | "continue"
            | "as"
            | "unroll"
            | "workgroup"
            | "true"
            | "false"
            | "buf"
            | "vec2"
            | "vec3"
            | "vec4"
            | "matrix"
            | "gid"
            | "thread"
            | "block"
            | "block_dim"
            | "lane"
            | "subgroup_id"
            | "subgroup_size"
            | "barrier"
            | "bitcast_f32"
            | "bitcast_u32"
            | "atomic_add"
            | "atomic_max"
            | "atomic_min"
            | "atomic_exchange"
            | "atomic_and"
            | "atomic_or"
            | "atomic_xor"
            | "subgroup_broadcast"
            | "subgroup_shuffle"
            | "subgroup_shuffle_down"
            | "subgroup_shuffle_up"
            | "subgroup_reduce_add"
            | "subgroup_reduce_max"
            | "subgroup_reduce_min"
            | "subgroup_inclusive_add"
            | "subgroup_all"
            | "subgroup_any"
            | "coop_zero"
            | "coop_load_a"
            | "coop_load_b"
            | "coop_mul_add"
            | "coop_store"
    )
}

fn builtin_var(name: &str) -> Option<&'static str> {
    match name {
        "gid" => Some("gid"),
        "thread" => Some("thread"),
        "block" => Some("block"),
        "block_dim" => Some("block_dim"),
        "lane" => Some("lane"),
        "subgroup_id" => Some("subgroup_id"),
        "subgroup_size" => Some("subgroup_size"),
        _ => None,
    }
}

fn scalar_name(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::F32 => "f32",
        Scalar::F16 => "f16",
        Scalar::Bf16 => "bf16",
        Scalar::I32 => "i32",
        Scalar::U32 => "u32",
        Scalar::I8 => "i8",
        Scalar::U8 => "u8",
        Scalar::Bool => "bool",
    }
}

fn name_static(name: &str) -> &'static str {
    match name {
        "min" => "min",
        "max" => "max",
        "clamp" => "clamp",
        "fma" => "fma",
        "select" => "select",
        "abs" => "abs",
        "floor" => "floor",
        "ceil" => "ceil",
        "round" => "round",
        "trunc" => "trunc",
        "sign" => "sign",
        "fract" => "fract",
        "sqrt" => "sqrt",
        "rsqrt" => "rsqrt",
        "exp" => "exp",
        "exp2" => "exp2",
        "log" => "log",
        "log2" => "log2",
        "tanh" => "tanh",
        "dot" => "dot",
        "popcount" => "popcount",
        "clz" => "clz",
        "ctz" => "ctz",
        "bitcast_f32" => "bitcast_f32",
        "bitcast_u32" => "bitcast_u32",
        "pow" => "pow",
        "atomic_and" => "atomic_and",
        "atomic_or" => "atomic_or",
        "atomic_xor" => "atomic_xor",
        "subgroup_broadcast" => "subgroup_broadcast",
        "subgroup_shuffle" => "subgroup_shuffle",
        "subgroup_shuffle_down" => "subgroup_shuffle_down",
        "subgroup_shuffle_up" => "subgroup_shuffle_up",
        "subgroup_reduce_add" => "subgroup_reduce_add",
        "subgroup_reduce_max" => "subgroup_reduce_max",
        "subgroup_reduce_min" => "subgroup_reduce_min",
        "subgroup_inclusive_add" => "subgroup_inclusive_add",
        "subgroup_all" => "subgroup_all",
        "subgroup_any" => "subgroup_any",
        "coop_load_a" => "coop_load_a",
        "coop_load_b" => "coop_load_b",
        "atomic_add" => "atomic_add",
        "atomic_max" => "atomic_max",
        "atomic_min" => "atomic_min",
        "atomic_exchange" => "atomic_exchange",
        _ => unreachable!(),
    }
}

fn check_matrix_elem(elem: Scalar, span: Span) -> Result<()> {
    if !elem.is_float() {
        return Err(Diagnostic::new(
            span,
            format!("cooperative matrix element must be f32 or f16, got {elem:?}"),
        ));
    }
    Ok(())
}
