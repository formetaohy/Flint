use std::collections::{HashMap, HashSet};

use crate::ast::{self, BinOp, UnOp};
use crate::builtin::{self, Arg, Ret};
use crate::consts::{self, CVal};
use crate::diag::{Diagnostic, Result, Span};
use crate::ir::{self, Access, MatrixRole, Scalar, Type};

#[derive(Debug, Clone, PartialEq)]
enum Sym {
    BufParam { ty: Type, access: Access },
    ScalarParam(Scalar),
    Local { id: u32, ty: Type, mutable: bool },
    ForVar { id: u32 },
    InlineConst(CVal),
}

struct Checker {
    params: Vec<ir::Param>,
    scalars: Vec<ir::ScalarParam>,
    structs: HashMap<String, Vec<(String, Type)>>,
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
    let mut structs = HashMap::new();
    for decl in &kernel.structs {
        if builtin::is_reserved(&decl.name) || decl.name.starts_with("__scl") {
            return Err(Diagnostic::new(
                decl.span,
                format!("'{}' is a reserved name", decl.name),
            ));
        }
        let mut fields = Vec::new();
        let mut seen = HashSet::new();
        for (field_name, ty) in &decl.fields {
            if !seen.insert(field_name.clone()) {
                return Err(Diagnostic::new(
                    decl.span,
                    format!("duplicate field '{field_name}' in struct {}", decl.name),
                ));
            }
            let ty = resolve_struct_field_type(ty, decl.span)?;
            fields.push((field_name.clone(), ty));
        }
        if structs.insert(decl.name.clone(), fields).is_some() {
            return Err(Diagnostic::new(
                decl.span,
                format!("duplicate struct '{}'", decl.name),
            ));
        }
    }
    let mut consts = HashMap::new();
    for spec in &kernel.specs {
        if builtin::is_reserved(&spec.name) || spec.name.starts_with("__scl") {
            return Err(Diagnostic::new(
                spec.span,
                format!("'{}' is a reserved name", spec.name),
            ));
        }
        let value = consts::const_eval(&spec.init, &consts).ok_or_else(|| {
            Diagnostic::new(
                spec.span,
                format!(
                    "spec '{}' initializer must be a constant expression",
                    spec.name
                ),
            )
        })?;
        if !consts::validate(&value, spec.ty) {
            return Err(Diagnostic::new(
                spec.span,
                format!(
                    "spec '{}' initializer out of range for {:?}",
                    spec.name, spec.ty
                ),
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
    let mut bindings = HashSet::new();
    for param in &kernel.params {
        let name = param.name.clone();
        if builtin::is_reserved(&name) || name.starts_with("__scl") {
            return Err(Diagnostic::new(
                param.span,
                format!("'{name}' is a reserved name"),
            ));
        }
        match &param.ty {
            ast::Type::Buf(elem) => {
                let ty = resolve_buf_elem(elem, &structs, param.span)?;
                let binding = param.binding.ok_or_else(|| {
                    Diagnostic::new(
                        param.span,
                        format!("buffer parameter '{name}' must be annotated with @buffer(n)"),
                    )
                })?;
                if !bindings.insert(binding) {
                    return Err(Diagnostic::new(
                        param.span,
                        format!("duplicate @buffer({binding}) binding"),
                    ));
                }
                let buf_ty = Type::Buf(Box::new(ty));
                if scope
                    .insert(name.clone(), Sym::BufParam { ty: buf_ty.clone(), access: param.access })
                    .is_some()
                {
                    return Err(Diagnostic::new(
                        param.span,
                        format!("duplicate parameter '{name}'"),
                    ));
                }
                params.push(ir::Param {
                    name,
                    ty: buf_ty,
                    binding,
                    access: param.access,
                });
            }
            ast::Type::Scalar(ty) => {
                if param.binding.is_some() {
                    return Err(Diagnostic::new(
                        param.span,
                        format!("scalar parameter '{name}' cannot carry @buffer"),
                    ));
                }
                if *ty == Scalar::Bool {
                    return Err(Diagnostic::new(
                        param.span,
                        format!(
                            "parameter '{name}' cannot be bool: Vulkan forbids bool in externally visible storage"
                        ),
                    ));
                }
                if scope.insert(name.clone(), Sym::ScalarParam(*ty)).is_some() {
                    return Err(Diagnostic::new(
                        param.span,
                        format!("duplicate parameter '{name}'"),
                    ));
                }
                scalar_offset = scalar_offset.div_ceil(4) * 4;
                scalars.push(ir::ScalarParam {
                    name,
                    ty: *ty,
                    offset: scalar_offset,
                });
                scalar_offset += ty.width();
            }
            _ => {
                return Err(Diagnostic::new(
                    param.span,
                    format!(
                        "parameter '{}' must be buf<scalar|vec|struct> or scalar",
                        param.name
                    ),
                ));
            }
        }
    }
    let mut checker = Checker {
        params,
        scalars,
        structs,
        scopes: vec![scope],
        consts,
        loop_depth: 0,
        block_depth: 0,
        next_id: 1,
        coop_triples: Vec::new(),
        coop_roles: Vec::new(),
    };
    let body = checker.check_stmts(&kernel.body)?;
    let structs = checker
        .structs
        .iter()
        .map(|(name, fields)| ir::StructDecl {
            name: name.clone(),
            fields: fields.clone(),
            span: kernel.span,
        })
        .collect();
    Ok(ir::Kernel {
        name: kernel.name.clone(),
        workgroup_size: kernel.workgroup_size,
        params: checker.params,
        scalars: checker.scalars,
        structs,
        coop_triples: checker.coop_triples,
        coop_roles: checker.coop_roles,
        body,
    })
}

fn resolve_struct_field_type(ty: &ast::Type, span: Span) -> Result<Type> {
    match ty {
        ast::Type::Scalar(Scalar::F32 | Scalar::I32 | Scalar::U32) => {
            let ast::Type::Scalar(scalar) = ty else {
                unreachable!()
            };
            Ok(Type::Scalar(*scalar))
        }
        _ => Err(Diagnostic::new(
            span,
            "struct fields must be f32, i32 or u32 (4-byte types)".to_string(),
        )),
    }
}

fn resolve_buf_elem(
    ty: &ast::Type,
    structs: &HashMap<String, Vec<(String, Type)>>,
    span: Span,
) -> Result<Type> {
    match ty {
        ast::Type::Scalar(scalar) => {
            if *scalar == Scalar::Bool {
                return Err(Diagnostic::new(
                    span,
                    "buf<bool> is forbidden: Vulkan forbids bool in externally visible storage"
                        .to_string(),
                ));
            }
            Ok(Type::Scalar(*scalar))
        }
        ast::Type::Vec { size, elem } => Ok(Type::Vec {
            size: *size,
            elem: *elem,
        }),
        ast::Type::Struct(name) => {
            let fields = structs.get(name).cloned().ok_or_else(|| {
                Diagnostic::new(span, format!("unknown struct '{name}'"))
            })?;
            Ok(Type::Struct {
                name: name.clone(),
                fields,
            })
        }
        _ => Err(Diagnostic::new(
            span,
            "buffer element must be a scalar, vector or struct".to_string(),
        )),
    }
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
        if builtin::is_reserved(name) {
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

    fn resolve_type(&self, ty: &ast::Type, span: Span) -> Result<Type> {
        match ty {
            ast::Type::Scalar(scalar) => Ok(Type::Scalar(*scalar)),
            ast::Type::Vec { size, elem } => Ok(Type::Vec {
                size: *size,
                elem: *elem,
            }),
            ast::Type::Matrix(elem) => {
                check_matrix_elem(*elem, span)?;
                Ok(Type::Matrix {
                    elem: *elem,
                    role: MatrixRole::Acc,
                })
            }
            ast::Type::Struct(name) => {
                let fields = self.structs.get(name).cloned().ok_or_else(|| {
                    Diagnostic::new(span, format!("unknown struct '{name}'"))
                })?;
                Ok(Type::Struct {
                    name: name.clone(),
                    fields,
                })
            }
            ast::Type::Threadgroup(elem) => {
                let elem = self.resolve_type(elem, span)?;
                if !matches!(elem, Type::Array { .. }) {
                    return Err(Diagnostic::new(
                        span,
                        "threadgroup type requires an array element, e.g. threadgroup<[f32; 64]>"
                            .to_string(),
                    ));
                }
                Ok(Type::Threadgroup(Box::new(elem)))
            }
            ast::Type::Array { elem, len } => {
                let elem = self.resolve_type(elem, span)?;
                let value = consts::const_eval(len, &self.consts).ok_or_else(|| {
                    Diagnostic::new(span, "array length must be a constant expression".to_string())
                })?;
                let CVal::Int(value) = value else {
                    return Err(Diagnostic::new(
                        span,
                        "array length must be an integer constant".to_string(),
                    ));
                };
                if value == 0 {
                    return Err(Diagnostic::new(
                        span,
                        "array length must be positive".to_string(),
                    ));
                }
                Ok(Type::Array {
                    elem: Box::new(elem),
                    len: value,
                })
            }
            ast::Type::Buf(_) => Err(Diagnostic::new(
                span,
                "buffer type is not valid here".to_string(),
            )),
        }
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
                mutable,
                span,
            } => {
                let is_threadgroup = matches!(ty, Some(ast::Type::Threadgroup(_)));
                if is_threadgroup {
                    if !*mutable {
                        return Err(Diagnostic::new(
                            *span,
                            "threadgroup variable must be declared with 'let mut'".to_string(),
                        ));
                    }
                    if self.block_depth != 0 {
                        return Err(Diagnostic::new(
                            *span,
                            "threadgroup variable must be declared at kernel top level".to_string(),
                        ));
                    }
                    if init.is_some() {
                        return Err(Diagnostic::new(
                            *span,
                            "threadgroup variable cannot have an initializer".to_string(),
                        ));
                    }
                    let ty = self.resolve_type(ty.as_ref().expect("annotated"), *span)?;
                    let id = self.next_id;
                    self.next_id += 1;
                    self.declare(
                        name,
                        Sym::Local {
                            id,
                            ty: ty.clone(),
                            mutable: true,
                        },
                        *span,
                    )?;
                    return Ok(vec![ir::Stmt::Var {
                        id,
                        name: name.clone(),
                        ty,
                        init: None,
                        span: *span,
                    }]);
                }
                let (init, init_ty) = match (ty, init) {
                    (Some(ast::Type::Matrix(elem)), Some(init)) => {
                        let (expr, ty) = self.check_matrix_init(init, *elem, *span)?;
                        (Some(expr), ty)
                    }
                    (Some(annotated), Some(init)) => {
                        let target = self.resolve_type(annotated, *span)?;
                        let (expr, ty) = self.check_expr(init, Some(target.clone()))?;
                        if ty != target {
                            return Err(Diagnostic::new(
                                *span,
                                format!("type mismatch: expected {target:?}, found {ty:?}"),
                            ));
                        }
                        (Some(expr), ty)
                    }
                    (None, Some(init)) => {
                        if !mutable {
                            if let Some(value) = consts::const_eval(init, &self.consts)
                                && !consts::may_negate(init)
                            {
                                self.declare(name, Sym::InlineConst(value), *span)?;
                                return Ok(Vec::new());
                            }
                        }
                        let expect = if matches!(init, ast::Expr::IntLit { .. }) {
                            Some(Type::Scalar(Scalar::U32))
                        } else {
                            None
                        };
                        let (expr, ty) = self.check_expr(init, expect)?;
                        (Some(expr), ty)
                    }
                    (_, None) => {
                        let Some(annotated) = ty else {
                            return Err(Diagnostic::new(
                                *span,
                                "variable without initializer requires a type annotation"
                                    .to_string(),
                            ));
                        };
                        let ty = self.resolve_type(annotated, *span)?;
                        if !matches!(ty, Type::Array { .. }) {
                            return Err(Diagnostic::new(
                                *span,
                                "variable without initializer must be an array type".to_string(),
                            ));
                        }
                        if !mutable {
                            return Err(Diagnostic::new(
                                *span,
                                "array without initializer must be declared with 'let mut'"
                                    .to_string(),
                            ));
                        }
                        (None, ty)
                    }
                };
                if !is_local_type(&init_ty) {
                    return Err(Diagnostic::new(
                        *span,
                        format!("invalid local variable type {init_ty:?}"),
                    ));
                }
                let id = self.next_id;
                self.next_id += 1;
                self.declare(
                    name,
                    Sym::Local {
                        id,
                        ty: init_ty.clone(),
                        mutable: *mutable,
                    },
                    *span,
                )?;
                if *mutable {
                    Ok(vec![ir::Stmt::Var {
                        id,
                        name: name.clone(),
                        ty: init_ty,
                        init,
                        span: *span,
                    }])
                } else {
                    Ok(vec![ir::Stmt::Let {
                        id,
                        name: name.clone(),
                        ty: init_ty,
                        init: init.expect("immutable let has initializer"),
                        span: *span,
                    }])
                }
            }
            ast::Stmt::Assign {
                target,
                value,
                span,
            } => {
                let (target, target_ty) = self.check_target(target)?;
                let (value, _) = self.check_expr(value, Some(target_ty.clone()))?;
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
                if name == "barrier" {
                    if args.is_empty() {
                        return Ok(vec![ir::Stmt::Barrier { span: *span }]);
                    }
                    return Err(Diagnostic::new(
                        *span,
                        "barrier takes no arguments".to_string(),
                    ));
                }
                if name == "coop_store" {
                    if args.len() != 4 {
                        return Err(Diagnostic::new(
                            *span,
                            "coop_store expects 4 arguments".to_string(),
                        ));
                    }
                    let (dst, dst_elem) = self.check_addr(&args[0], *span)?;
                    if self.is_readonly_addr(&args[0])? {
                        return Err(Diagnostic::new(
                            *span,
                            "coop_store destination buffer is readonly".to_string(),
                        ));
                    }
                    check_matrix_elem(dst_elem, *span)?;
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

    fn is_readonly_addr(&self, expr: &ast::Expr) -> Result<bool> {
        let base = match expr {
            ast::Expr::Name(name, _) => name.clone(),
            ast::Expr::Index { base, .. } => match &**base {
                ast::Expr::Name(name, _) => name.clone(),
                _ => return Ok(false),
            },
            _ => return Ok(false),
        };
        match self.lookup(&base) {
            Some(Sym::BufParam { access, .. }) => Ok(access == Access::ReadOnly),
            _ => Ok(false),
        }
    }

    fn check_addr(&mut self, expr: &ast::Expr, span: Span) -> Result<(ir::Expr, Scalar)> {
        match expr {
            ast::Expr::Name(name, _) => match self.lookup(name) {
                Some(Sym::BufParam { ty, .. }) => {
                    let elem = ty.elem().ok_or_else(|| {
                        Diagnostic::new(span, "buffer element must be scalar".to_string())
                    })?;
                    Ok((
                        ir::Expr::ParamRef {
                            name: name.clone(),
                            ty: ty.clone(),
                            span,
                        },
                        elem,
                    ))
                }
                Some(Sym::Local {
                    ty: Type::Threadgroup(_),
                    id,
                    ..
                }) => {
                    let ty = match self.lookup(name) {
                        Some(Sym::Local { ty, .. }) => ty,
                        _ => unreachable!(),
                    };
                    let elem = ty.elem().ok_or_else(|| {
                        Diagnostic::new(span, "array element must be scalar".to_string())
                    })?;
                    Ok((
                        ir::Expr::LocalRef {
                            id,
                            name: name.clone(),
                            ty,
                            span,
                        },
                        elem,
                    ))
                }
                _ => Err(Diagnostic::new(
                    span,
                    "expected a buffer or threadgroup array element".to_string(),
                )),
            },
            ast::Expr::Index { base, index, span: _ } => {
                let (base_expr, elem) = self.check_addr(base, span)?;
                let (index, _) = self.check_expr(index, Some(Type::Scalar(Scalar::U32)))?;
                Ok((
                    ir::Expr::Index {
                        base: Box::new(base_expr),
                        index: Box::new(index),
                        ty: elem_type(elem),
                        span,
                    },
                    elem,
                ))
            }
            _ => Err(Diagnostic::new(
                span,
                "expected a buffer element or threadgroup array element".to_string(),
            )),
        }
    }

    fn check_target(&mut self, target: &ast::Expr) -> Result<(ir::Expr, Type)> {
        match target {
            ast::Expr::Name(name, span) => match self.lookup(name) {
                Some(Sym::BufParam { .. }) | Some(Sym::ScalarParam(_)) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to parameter '{name}'"),
                )),
                Some(Sym::ForVar { .. }) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to loop variable '{name}'"),
                )),
                Some(Sym::InlineConst(_)) => Err(Diagnostic::new(
                    *span,
                    format!("cannot assign to constant '{name}'"),
                )),
                Some(Sym::Local {
                    mutable: false, ..
                }) => Err(Diagnostic::new(
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
                        ty: ty.clone(),
                        span: *span,
                    },
                    ty,
                )),
                None => Err(Diagnostic::new(
                    *span,
                    format!("undefined variable '{name}'"),
                )),
            },
            ast::Expr::Field { base, name, span } => {
                let base_name = match &**base {
                    ast::Expr::Name(name, _) => name.clone(),
                    _ => {
                        return Err(Diagnostic::new(
                            *span,
                            "field assignment requires a local struct variable".to_string(),
                        ));
                    }
                };
                let sym = self.lookup(&base_name).ok_or_else(|| {
                    Diagnostic::new(*span, format!("undefined variable '{base_name}'"))
                })?;
                let Sym::Local {
                    id,
                    ty,
                    mutable: true,
                    ..
                } = sym
                else {
                    return Err(Diagnostic::new(
                        *span,
                        format!("cannot assign to immutable '{base_name}'"),
                    ));
                };
                let Type::Struct { fields, .. } = &ty else {
                    return Err(Diagnostic::new(
                        *span,
                        "field assignment requires a struct".to_string(),
                    ));
                };
                let field_ty = fields
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, ty)| ty.clone())
                    .ok_or_else(|| {
                        Diagnostic::new(*span, format!("unknown field '{name}'"))
                    })?;
                Ok((
                    ir::Expr::Field {
                        base: Box::new(ir::Expr::LocalRef {
                            id,
                            name: base_name.clone(),
                            ty: ty.clone(),
                            span: *span,
                        }),
                        name: name.clone(),
                        ty: field_ty.clone(),
                        span: *span,
                    },
                    field_ty,
                ))
            }
            ast::Expr::Index { base, index, span } => {
                if self.is_readonly_addr(&ast::Expr::Index {
                    base: base.clone(),
                    index: index.clone(),
                    span: *span,
                })? {
                    return Err(Diagnostic::new(
                        *span,
                        "cannot write to a readonly buffer".to_string(),
                    ));
                }
                let (base_expr, base_ty) = self.check_expr(base, None)?;
                let elem = match &base_ty {
                    Type::Buf(elem) => (**elem).clone(),
                    Type::Array { elem, .. } => (**elem).clone(),
                    _ => {
                        return Err(Diagnostic::new(
                            *span,
                            "index target must be a buffer or array".to_string(),
                        ));
                    }
                };
                let (index, _) = self.check_expr(index, Some(Type::Scalar(Scalar::U32)))?;
                Ok((
                    ir::Expr::Index {
                        base: Box::new(base_expr),
                        index: Box::new(index),
                        ty: elem.clone(),
                        span: *span,
                    },
                    elem,
                ))
            }
            _ => Err(Diagnostic::new(
                target.span(),
                "invalid assignment target".to_string(),
            )),
        }
    }

    fn check_expr(&mut self, expr: &ast::Expr, expect: Option<Type>) -> Result<(ir::Expr, Type)> {
        let (expr, ty) = self.check_expr_inner(expr, expect.clone())?;
        if let Some(target) = expect {
            if ty != target {
                return Err(Diagnostic::new(
                    expr_span(&expr),
                    format!("type mismatch: expected {target:?}, found {ty:?}"),
                ));
            }
        }
        Ok((expr, ty))
    }

    fn check_int_literal(
        &mut self,
        value: u64,
        suffix: Option<Scalar>,
        expect: Option<Type>,
        span: Span,
    ) -> Result<(ir::Expr, Scalar)> {
        let target = match expect {
            Some(Type::Scalar(scalar)) => Some(scalar),
            Some(Type::Matrix { elem, .. }) => {
                check_matrix_elem(elem, span)?;
                Some(elem)
            }
            _ => None,
        };
        if let Some(suffix) = suffix {
            if value > range_max(suffix) {
                return Err(Diagnostic::new(
                    span,
                    format!("integer literal {value} out of {suffix:?} range"),
                ));
            }
            return Ok((
                ir::Expr::IntLit {
                    value,
                    ty: suffix,
                    span,
                },
                suffix,
            ));
        }
        match target {
            Some(Scalar::F32 | Scalar::Bf16) => {
                if value > (1u64 << 24) {
                    return Err(Diagnostic::new(
                        span,
                        format!("integer literal {value} is not exactly representable as {target:?}"),
                    ));
                }
                Ok((
                    ir::Expr::FloatLit {
                        value: value as f64,
                        ty: target.expect("float target"),
                        span,
                    },
                    target.expect("float target"),
                ))
            }
            Some(Scalar::F16) => {
                if value > 2048 {
                    return Err(Diagnostic::new(
                        span,
                        format!("integer literal {value} is not exactly representable as f16"),
                    ));
                }
                Ok((
                    ir::Expr::FloatLit {
                        value: value as f64,
                        ty: Scalar::F16,
                        span,
                    },
                    Scalar::F16,
                ))
            }
            Some(Scalar::Bool) => Err(Diagnostic::new(
                span,
                "integer literal in bool context".to_string(),
            )),
            Some(integer @ (Scalar::U32 | Scalar::I32 | Scalar::U8 | Scalar::I8)) => {
                if value > range_max(integer) {
                    return Err(Diagnostic::new(
                        span,
                        format!("integer literal {value} out of {integer:?} range"),
                    ));
                }
                Ok((
                    ir::Expr::IntLit {
                        value,
                        ty: integer,
                        span,
                    },
                    integer,
                ))
            }
            None => {
                if value > u32::MAX as u64 {
                    return Err(Diagnostic::new(
                        span,
                        format!("integer literal {value} out of u32 range"),
                    ));
                }
                Ok((
                    ir::Expr::IntLit {
                        value,
                        ty: Scalar::U32,
                        span,
                    },
                    Scalar::U32,
                ))
            }
        }
    }

    fn check_float_literal(
        &mut self,
        value: f64,
        suffix: Option<Scalar>,
        expect: Option<Type>,
        span: Span,
    ) -> Result<(ir::Expr, Scalar)> {
        let target = match expect {
            Some(Type::Scalar(scalar)) => Some(scalar),
            Some(Type::Matrix { elem, .. }) => {
                check_matrix_elem(elem, span)?;
                Some(elem)
            }
            _ => None,
        };
        if let Some(suffix) = suffix {
            return Ok((
                ir::Expr::FloatLit {
                    value,
                    ty: suffix,
                    span,
                },
                suffix,
            ));
        }
        match target {
            Some(Scalar::F32 | Scalar::F16 | Scalar::Bf16) => Ok((
                ir::Expr::FloatLit {
                    value,
                    ty: target.expect("float target"),
                    span,
                },
                target.expect("float target"),
            )),
            Some(integer @ (Scalar::U32 | Scalar::I32 | Scalar::U8 | Scalar::I8)) => {
                if value.fract() != 0.0 {
                    return Err(Diagnostic::new(
                        span,
                        format!("float literal {value} is not an integer"),
                    ));
                }
                let iv = value as i128;
                let (min, max) = match integer {
                    Scalar::U32 => (0, u32::MAX as i128),
                    Scalar::I32 => (i32::MIN as i128, i32::MAX as i128),
                    Scalar::U8 => (0, u8::MAX as i128),
                    _ => (i8::MIN as i128, i8::MAX as i128),
                };
                if iv < min || iv > max {
                    return Err(Diagnostic::new(
                        span,
                        format!("float literal {value} out of {integer:?} range"),
                    ));
                }
                Ok((
                    ir::Expr::IntLit {
                        value: iv as u64,
                        ty: integer,
                        span,
                    },
                    integer,
                ))
            }
            Some(Scalar::Bool) => Err(Diagnostic::new(
                span,
                "float literal in bool context".to_string(),
            )),
            None => Ok((
                ir::Expr::FloatLit {
                    value,
                    ty: Scalar::F32,
                    span,
                },
                Scalar::F32,
            )),
        }
    }

    fn check_matrix_init(
        &mut self,
        init: &ast::Expr,
        elem: Scalar,
        span: Span,
    ) -> Result<(ir::Expr, Type)> {
        check_matrix_elem(elem, span)?;
        if let ast::Expr::Call {
            name, args, span: call_span,
        } = init
            && name == "coop_zero"
            && args.is_empty()
        {
            self.coop_roles.push((elem, MatrixRole::Acc));
            let ty = Type::Matrix {
                elem,
                role: MatrixRole::Acc,
            };
            return Ok((
                ir::Expr::Call {
                    name: "coop_zero",
                    args: Vec::new(),
                    ty: ty.clone(),
                    span: *call_span,
                },
                ty,
            ));
        }
        let (expr, ty) = self.check_expr(init, None)?;
        let Type::Matrix { elem: init_elem, .. } = ty else {
            return Err(Diagnostic::new(
                span,
                "matrix annotation requires a matrix initializer".to_string(),
            ));
        };
        if init_elem != elem {
            return Err(Diagnostic::new(
                span,
                format!("matrix element mismatch: expected {elem:?}, found {init_elem:?}"),
            ));
        }
        Ok((expr, ty))
    }

    fn check_expr_inner(
        &mut self,
        expr: &ast::Expr,
        expect: Option<Type>,
    ) -> Result<(ir::Expr, Type)> {
        match expr {
            ast::Expr::IntLit { value, ty, span } => {
                let (expr, scalar) = self.check_int_literal(*value, *ty, expect, *span)?;
                Ok((expr, Type::Scalar(scalar)))
            }
            ast::Expr::FloatLit { value, ty, span } => {
                let (expr, scalar) = self.check_float_literal(*value, *ty, expect, *span)?;
                Ok((expr, Type::Scalar(scalar)))
            }
            ast::Expr::BoolLit { value, span } => {
                let ty = match expect {
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
                    Type::Scalar(ty),
                ))
            }
            ast::Expr::Name(name, span) => {
                if let Some((value, ty)) = self.consts.get(name).copied() {
                    return Ok((const_literal(value, ty, *span), Type::Scalar(ty)));
                }
                match self.lookup(name) {
                    Some(Sym::InlineConst(value)) => {
                        let lit = match value {
                            CVal::Int(v) => ast::Expr::IntLit {
                                value: v,
                                ty: None,
                                span: *span,
                            },
                            CVal::Float(v) => ast::Expr::FloatLit {
                                value: v,
                                ty: None,
                                span: *span,
                            },
                            CVal::Bool(v) => ast::Expr::BoolLit {
                                value: v,
                                span: *span,
                            },
                        };
                        self.check_expr_inner(&lit, expect)
                    }
                    Some(Sym::BufParam { ty, .. }) => Ok((
                        ir::Expr::ParamRef {
                            name: name.clone(),
                            ty: ty.clone(),
                            span: *span,
                        },
                        ty.clone(),
                    )),
                    Some(Sym::ScalarParam(ty)) => Ok((
                        ir::Expr::ScalarRef {
                            name: name.clone(),
                            ty,
                            span: *span,
                        },
                        Type::Scalar(ty),
                    )),
                    Some(Sym::Local { id, ty, .. }) => {
                        let ty = match ty {
                            Type::Threadgroup(inner) => {
                                let Type::Array { elem, len } = &*inner else {
                                    unreachable!("threadgroup element is array");
                                };
                                let elem = elem.elem().expect("array elem");
                                Type::Array {
                                    elem: Box::new(elem_type(elem)),
                                    len: *len,
                                }
                            }
                            _ => ty.clone(),
                        };
                        Ok((
                            ir::Expr::LocalRef {
                                id,
                                name: name.clone(),
                                ty: ty.clone(),
                                span: *span,
                            },
                            ty,
                        ))
                    }
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
            ast::Expr::Index {
                base,
                index,
                span,
            } => {
                let (base_expr, base_ty) = self.check_expr(base, None)?;
                let elem = match &base_ty {
                    Type::Buf(elem) => (**elem).clone(),
                    Type::Array { elem, .. } => (**elem).clone(),
                    _ => {
                        return Err(Diagnostic::new(
                            *span,
                            "index target must be a buffer or array".to_string(),
                        ));
                    }
                };
                let (index, _) = self.check_expr(index, Some(Type::Scalar(Scalar::U32)))?;
                Ok((
                    ir::Expr::Index {
                        base: Box::new(base_expr),
                        index: Box::new(index),
                        ty: elem.clone(),
                        span: *span,
                    },
                    elem,
                ))
            }
            ast::Expr::Field { base, name, span } => {
                let (base_expr, base_ty) = self.check_expr(base, None)?;
                match &base_ty {
                    Type::Vec { size, elem } => {
                        if name.len() == 1 {
                            let idx = match name.as_bytes()[0] {
                                b'x' => 0,
                                b'y' => 1,
                                b'z' => 2,
                                b'w' => 3,
                                _ => {
                                    return Err(Diagnostic::new(
                                        *span,
                                        "vector components are x, y, z or w".to_string(),
                                    ));
                                }
                            };
                            if idx >= *size {
                                return Err(Diagnostic::new(
                                    *span,
                                    format!("component '{name}' out of range for vec{size}"),
                                ));
                            }
                            return Ok((
                                ir::Expr::Field {
                                    base: Box::new(base_expr),
                                    name: name.clone(),
                                    ty: Type::Scalar(*elem),
                                    span: *span,
                                },
                                Type::Scalar(*elem),
                            ));
                        }
                        let mut mask = Vec::new();
                        for c in name.chars() {
                            match c {
                                'x' => mask.push(0),
                                'y' => mask.push(1),
                                'z' => mask.push(2),
                                'w' => mask.push(3),
                                _ => {
                                    return Err(Diagnostic::new(
                                        *span,
                                        "vector swizzles use x, y, z or w".to_string(),
                                    ));
                                }
                            }
                        }
                        for &idx in &mask {
                            if idx >= *size {
                                return Err(Diagnostic::new(
                                    *span,
                                    format!("swizzle component out of range for vec{size}"),
                                ));
                            }
                        }
                        let out_size = mask.len() as u32;
                        let mut args = vec![base_expr];
                        for &idx in &mask {
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
                                ty: Type::Vec {
                                    size: out_size,
                                    elem: *elem,
                                },
                                span: *span,
                            },
                            Type::Vec {
                                size: out_size,
                                elem: *elem,
                            },
                        ))
                    }
                    Type::Struct { fields, .. } => {
                        let (_, field_ty) = fields
                            .iter()
                            .find(|(n, _)| n == name)
                            .ok_or_else(|| {
                                Diagnostic::new(*span, format!("unknown field '{name}'"))
                            })?;
                        Ok((
                            ir::Expr::Field {
                                base: Box::new(base_expr),
                                name: name.clone(),
                                ty: field_ty.clone(),
                                span: *span,
                            },
                            field_ty.clone(),
                        ))
                    }
                    _ => Err(Diagnostic::new(
                        *span,
                        "field access requires a vector or struct".to_string(),
                    )),
                }
            }
            ast::Expr::Unary { op, expr, span } => match op {
                UnOp::Neg => {
                    if let ast::Expr::IntLit { value, ty, .. } = &**expr {
                        let signed = match ty {
                            Some(Scalar::I32) | None => Scalar::I32,
                            Some(Scalar::U32) => {
                                return Err(Diagnostic::new(
                                    *span,
                                    "cannot negate a u32 literal".to_string(),
                                ));
                            }
                            _ => {
                                return Err(Diagnostic::new(
                                    *span,
                                    "negation requires a signed literal".to_string(),
                                ));
                            }
                        };
                        let neg = (*value as i128).wrapping_neg();
                        if neg < i32::MIN as i128 || neg > i32::MAX as i128 {
                            return Err(Diagnostic::new(
                                *span,
                                "integer literal out of i32 range".to_string(),
                            ));
                        }
                        return Ok((
                            ir::Expr::IntLit {
                                value: neg as u64,
                                ty: signed,
                                span: *span,
                            },
                            Type::Scalar(signed),
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
                let unsuffixed = |e: &ast::Expr| {
                    matches!(
                        e,
                        ast::Expr::IntLit { ty: None, .. } | ast::Expr::FloatLit { ty: None, .. }
                    )
                };
                let (then_expr, els_expr, branch_ty) =
                    if unsuffixed(then) && unsuffixed(els) {
                        if let Some(Type::Scalar(target)) = expect {
                            let (then_expr, _) =
                                self.check_expr(then, Some(Type::Scalar(target)))?;
                            let (els_expr, _) =
                                self.check_expr(els, Some(Type::Scalar(target)))?;
                            (then_expr, els_expr, Type::Scalar(target))
                        } else {
                            let (then_expr, then_ty) = self.check_expr(then, None)?;
                            let (els_expr, _) = self.check_expr(els, Some(then_ty.clone()))?;
                            (then_expr, els_expr, then_ty)
                        }
                    } else {
                        let (then_expr, then_ty) = self.check_expr(then, None)?;
                        let (els_expr, _) = self.check_expr(els, Some(then_ty.clone()))?;
                        (then_expr, els_expr, then_ty)
                    };
                Ok((
                    ir::Expr::Cond {
                        cond: Box::new(cond),
                        then: Box::new(then_expr),
                        els: Box::new(els_expr),
                        ty: branch_ty.clone(),
                        span: *span,
                    },
                    branch_ty,
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
                if let ast::Expr::IntLit { value, ty: None, .. } = &**expr {
                    if *target == Scalar::U32 {
                        return Err(Diagnostic::new(
                            *span,
                            format!(
                                "redundant conversion to same type {}",
                                scalar_name(*target)
                            ),
                        ));
                    }
                    let (expr, scalar) =
                        self.check_int_literal(*value, None, Some(Type::Scalar(*target)), *span)?;
                    return Ok((expr, Type::Scalar(scalar)));
                }
                if let ast::Expr::FloatLit { value, ty: None, .. } = &**expr {
                    if *target == Scalar::F32 {
                        return Err(Diagnostic::new(
                            *span,
                            format!(
                                "redundant conversion to same type {}",
                                scalar_name(*target)
                            ),
                        ));
                    }
                    let (expr, scalar) =
                        self.check_float_literal(*value, None, Some(Type::Scalar(*target)), *span)?;
                    return Ok((expr, Type::Scalar(scalar)));
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
            ast::Expr::OrderLit(_, span) => Err(Diagnostic::new(
                *span,
                "memory order literal is only valid as an atomic order argument".to_string(),
            )),
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
            ast::Expr::ConstructStruct {
                name,
                fields,
                span,
            } => {
                let struct_fields = self.structs.get(name).cloned().ok_or_else(|| {
                    Diagnostic::new(*span, format!("unknown struct '{name}'"))
                })?;
                let mut checked = Vec::new();
                let mut seen = HashSet::new();
                for (field_name, value) in fields {
                    if !seen.insert(field_name.clone()) {
                        return Err(Diagnostic::new(
                            *span,
                            format!("duplicate field '{field_name}'"),
                        ));
                    }
                    let (_, field_ty) = struct_fields
                        .iter()
                        .find(|(n, _)| n == field_name)
                        .ok_or_else(|| {
                            Diagnostic::new(
                                *span,
                                format!(
                                    "unknown field '{field_name}' for struct '{name}'"
                                ),
                            )
                        })?;
                    let (value, _) = self.check_expr(value, Some(field_ty.clone()))?;
                    checked.push((field_name.clone(), value));
                }
                if seen.len() != struct_fields.len() {
                    return Err(Diagnostic::new(
                        *span,
                        format!(
                            "struct '{name}' requires {} fields, got {}",
                            struct_fields.len(),
                            seen.len()
                        ),
                    ));
                }
                Ok((
                    ir::Expr::ConstructStruct {
                        name: name.clone(),
                        fields: checked,
                        ty: Type::Struct {
                            name: name.clone(),
                            fields: struct_fields.clone(),
                        },
                        span: *span,
                    },
                    Type::Struct {
                        name: name.clone(),
                        fields: struct_fields,
                    },
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
            let (target, elem) = self.check_addr(&args[0], span)?;
            let ir::Expr::Index { .. } = target else {
                return Err(Diagnostic::new(
                    span,
                    format!("{name} target must be a buffer or threadgroup array element"),
                ));
            };
            if !elem.is_int() {
                return Err(Diagnostic::new(
                    span,
                    format!("{name} requires an integer element type"),
                ));
            }
            if self.is_readonly_addr(&args[0])? {
                return Err(Diagnostic::new(
                    span,
                    format!("{name} target buffer is readonly"),
                ));
            }
            let (value, _) = self.check_expr(&args[1], Some(Type::Scalar(elem)))?;
            let order = match &args[2] {
                ast::Expr::OrderLit(order, order_span) => ir::Expr::OrderLit {
                    order: *order,
                    span: *order_span,
                },
                _ => {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} order must be a memory order literal"),
                    ));
                }
            };
            return Ok((
                ir::Expr::Call {
                    name: name_static(name),
                    args: vec![target, value, order],
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
            let Some(Type::Matrix { elem, .. }) = expect else {
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
                    ty: Type::Matrix {
                        elem,
                        role: MatrixRole::Acc,
                    },
                    span,
                },
                Type::Matrix {
                    elem,
                    role: MatrixRole::Acc,
                },
            ));
        }
        if name == "coop_load_a" || name == "coop_load_b" {
            if args.len() != 3 {
                return Err(Diagnostic::new(span, format!("{name} expects 3 arguments")));
            }
            let (src, elem) = self.check_addr(&args[0], span)?;
            check_matrix_elem(elem, span)?;
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
            let role = if name == "coop_load_a" {
                MatrixRole::A
            } else {
                MatrixRole::B
            };
            self.coop_roles.push((elem, role));
            let ty = Type::Matrix { elem, role };
            return Ok((
                ir::Expr::Call {
                    name: name_static(name),
                    args: vec![src, stride, layout],
                    ty: ty.clone(),
                    span,
                },
                ty,
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
                    ty: ret.clone(),
                    span,
                },
                ret,
            ));
        }
        let sig = builtin::lookup(name).ok_or_else(|| {
            Diagnostic::new(span, format!("unknown function or builtin '{name}'"))
        })?;
        if sig.ret == Ret::Void {
            return Err(Diagnostic::new(
                span,
                format!("'{name}' does not return a value and must be a statement"),
            ));
        }
        if args.len() != sig.args.len() {
            return Err(Diagnostic::new(
                span,
                format!(
                    "builtin '{name}' expects {} arguments, got {}",
                    sig.args.len(),
                    args.len()
                ),
            ));
        }
        let mut checked = Vec::with_capacity(args.len());
        let mut first: Option<(ir::Expr, Type)> = None;
        for (index, arg) in args.iter().enumerate() {
            let spec = sig.args[index];
            let (expr, ty) = match spec {
                Arg::Num => match &first {
                    Some((_, Type::Scalar(s))) => {
                        let (e, t) = self.check_expr(arg, Some(Type::Scalar(*s)))?;
                        (e, t)
                    }
                    Some(_) => {
                        return Err(Diagnostic::new(
                            span,
                            format!("{name} requires scalar arguments"),
                        ));
                    }
                    None => self.check_expr(arg, None)?,
                },
                Arg::Float => {
                    let (e, t) = self.check_expr(arg, None)?;
                    match &t {
                        Type::Scalar(s) if s.is_float() => {}
                        _ => {
                            return Err(Diagnostic::new(
                                span,
                                format!("{name} requires a float argument"),
                            ));
                        }
                    }
                    (e, t)
                }
                Arg::Bool => {
                    let (e, t) = self.check_expr(arg, Some(Type::Scalar(Scalar::Bool)))?;
                    (e, t)
                }
                Arg::U32 => {
                    let (e, t) = self.check_expr(arg, Some(Type::Scalar(Scalar::U32)))?;
                    (e, t)
                }
                Arg::Vec => {
                    let (e, t) = self.check_expr(arg, None)?;
                    match &t {
                        Type::Vec { .. } => {}
                        _ => {
                            return Err(Diagnostic::new(
                                span,
                                format!("{name} requires a vector argument"),
                            ));
                        }
                    }
                    (e, t)
                }
                Arg::Order => {
                    return Err(Diagnostic::new(
                        span,
                        format!("internal error: bare order argument for {name}"),
                    ));
                }
                Arg::ConstBool => {
                    return Err(Diagnostic::new(
                        span,
                        format!("internal error: bare const-bool argument for {name}"),
                    ));
                }
                Arg::Addr | Arg::MatA | Arg::MatB | Arg::MatAcc => {
                    return Err(Diagnostic::new(
                        span,
                        format!("internal error: bare specialized argument for {name}"),
                    ));
                }
            };
            if first.is_none() {
                first = Some((expr.clone(), ty.clone()));
            }
            checked.push(expr);
        }
        let first_ty = first.as_ref().map(|(_, ty)| ty.clone());
        let ret_ty = match sig.ret {
            Ret::SameAsFirst => first_ty.expect("first argument exists"),
            Ret::Bool => Type::Scalar(Scalar::Bool),
            Ret::U32 => Type::Scalar(Scalar::U32),
            Ret::F32 => Type::Scalar(Scalar::F32),
            Ret::Vec3U32 => Type::Vec {
                size: 3,
                elem: Scalar::U32,
            },
            Ret::ScalarOfVec => {
                let Type::Vec { elem, .. } = first_ty.expect("first argument exists") else {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires a vector argument"),
                    ));
                };
                if !elem.is_float() {
                    return Err(Diagnostic::new(
                        span,
                        format!("{name} requires a float vector argument"),
                    ));
                }
                Type::Scalar(elem)
            }
            _ => unreachable!(),
        };
        Ok((
            ir::Expr::Call {
                name: name_static(name),
                args: checked,
                ty: ret_ty.clone(),
                span,
            },
            ret_ty,
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
                let (rhs, _) = self.check_expr(rhs, Some(lhs_ty.clone()))?;
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
                let int_float_mix = matches!(
                    (lhs, rhs),
                    (
                        ast::Expr::IntLit { ty: None, .. },
                        ast::Expr::FloatLit { .. }
                    ) | (
                        ast::Expr::FloatLit { .. },
                        ast::Expr::IntLit { ty: None, .. }
                    )
                );
                if int_float_mix {
                    let (lhs, _) = self.check_expr(lhs, Some(Type::Scalar(Scalar::F32)))?;
                    let (rhs, _) = self.check_expr(rhs, Some(Type::Scalar(Scalar::F32)))?;
                    let ty = if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
                        Type::Scalar(Scalar::Bool)
                    } else {
                        Type::Scalar(Scalar::F32)
                    };
                    return Ok((
                        ir::Expr::Binary {
                            op,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                            ty: ty.clone(),
                            span,
                        },
                        ty,
                    ));
                }
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
                                ty: ty.clone(),
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
                                ty: ty.clone(),
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

fn is_local_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Scalar(_)
            | Type::Vec { .. }
            | Type::Matrix { .. }
            | Type::Array { .. }
            | Type::Threadgroup(_)
            | Type::Struct { .. }
    )
}

fn elem_type(elem: Scalar) -> Type {
    Type::Scalar(elem)
}

fn range_max(ty: Scalar) -> u64 {
    match ty {
        Scalar::U32 => u32::MAX as u64,
        Scalar::I32 => i32::MAX as u64,
        Scalar::U8 => u8::MAX as u64,
        Scalar::I8 => i8::MAX as u64,
        _ => u64::MAX,
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
        ir::Stmt::Let { init, .. } | ir::Stmt::Var { init: Some(init), .. } => {
            substitute_loop_expr(init, var, value);
        }
        ir::Stmt::Var { init: None, .. } => {}
        ir::Stmt::Assign {
            target,
            value: v,
            ..
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
        ir::Expr::Field { base, .. } => substitute_loop_expr(base, var, value),
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
        ir::Expr::ConstructStruct { fields, .. } => {
            for (_, field_expr) in fields {
                substitute_loop_expr(field_expr, var, value);
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
        | ir::Expr::Index { span, .. }
        | ir::Expr::Field { span, .. }
        | ir::Expr::Unary { span, .. }
        | ir::Expr::Binary { span, .. }
        | ir::Expr::Cond { span, .. }
        | ir::Expr::Convert { span, .. }
        | ir::Expr::OrderLit { span, .. }
        | ir::Expr::ConstructStruct { span, .. }
        | ir::Expr::Call { span, .. } => *span,
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
        "pow" => "pow",
        "dot" => "dot",
        "popcount" => "popcount",
        "clz" => "clz",
        "ctz" => "ctz",
        "bitcast_f32" => "bitcast_f32",
        "bitcast_u32" => "bitcast_u32",
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
        "global_id" => "global_id",
        "local_id" => "local_id",
        "group_id" => "group_id",
        "group_size" => "group_size",
        "subgroup_id" => "subgroup_id",
        "lane" => "lane",
        "subgroup_size" => "subgroup_size",
        _ => unreachable!("unknown builtin {name}"),
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
