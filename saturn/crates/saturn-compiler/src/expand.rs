use std::collections::{HashMap, HashSet};

use crate::ast::{Expr, FnDecl, Kernel, Program, Stmt, Type};
use crate::consts::{self, CVal};
use crate::diag::{Diagnostic, Result, Span};
use crate::ir::Scalar;
use crate::sema::is_reserved;

const MAX_DEPTH: usize = 32;

pub struct Expander<'a> {
    fns: HashMap<String, &'a FnDecl>,
    call_stack: Vec<String>,
    counter: u32,
    consts: HashMap<String, (CVal, Scalar)>,
}

pub fn expand(program: &Program) -> Result<Kernel> {
    let mut fns = HashMap::new();
    for f in &program.fns {
        validate_fn_decl(f)?;
        if fns.insert(f.name.clone(), f).is_some() {
            return Err(Diagnostic::new(
                f.span,
                format!("duplicate function '{}'", f.name),
            ));
        }
    }
    let mut expander = Expander {
        fns,
        call_stack: Vec::new(),
        counter: 0,
        consts: HashMap::new(),
    };
    for stmt in &program.kernel.body {
        if let Stmt::Const {
            name,
            ty,
            init,
            span,
        } = stmt
        {
            if let Some(value) = consts::const_eval(init, &expander.consts) {
                if !consts::validate(&value, *ty) {
                    return Err(Diagnostic::new(
                        *span,
                        format!("const initializer out of range for {ty:?}"),
                    ));
                }
                expander.consts.insert(name.clone(), (value, *ty));
            }
        }
    }
    let body = expander.expand_stmts(&program.kernel.body)?;
    Ok(Kernel {
        name: program.kernel.name.clone(),
        workgroup_size: program.kernel.workgroup_size,
        params: program.kernel.params.clone(),
        specs: program.kernel.specs.clone(),
        body,
        span: program.kernel.span,
    })
}

fn validate_fn_decl(f: &FnDecl) -> Result<()> {
    if is_reserved(&f.name) || f.name.starts_with("__scl") {
        return Err(Diagnostic::new(
            f.span,
            format!("'{}' is a reserved name", f.name),
        ));
    }
    for param in &f.params {
        if is_reserved(&param.name) || param.name.starts_with("__scl") {
            return Err(Diagnostic::new(
                f.span,
                format!("'{}' is a reserved name", param.name),
            ));
        }
        match &param.ty {
            Type::Buf(_) | Type::Scalar(_) | Type::Vec { .. } => {}
            Type::Matrix { .. } => {
                return Err(Diagnostic::new(
                    f.span,
                    format!("function '{}' cannot take matrix parameters", f.name),
                ));
            }
            Type::SharedArray { .. } => unreachable!("shared arrays are not parameter types"),
        }
        if param.is_const && !matches!(param.ty, Type::Scalar(_)) {
            return Err(Diagnostic::new(
                f.span,
                format!("const parameter '{}' must be scalar", param.name),
            ));
        }
    }
    if let Some(ret) = &f.ret {
        if matches!(ret, Type::Matrix { .. }) {
            return Err(Diagnostic::new(
                f.span,
                format!("function '{}' cannot return a matrix", f.name),
            ));
        }
    }
    Ok(())
}

impl<'a> Expander<'a> {
    fn fresh(&mut self, base: &str) -> String {
        self.counter += 1;
        format!("__scl_{base}_{}", self.counter)
    }

    fn expand_stmts(&mut self, stmts: &[Stmt]) -> Result<Vec<Stmt>> {
        let mut out = Vec::new();
        for stmt in stmts {
            self.expand_stmt(stmt, &mut out)?;
        }
        Ok(out)
    }

    fn expand_stmt(&mut self, stmt: &Stmt, out: &mut Vec<Stmt>) -> Result<()> {
        match stmt {
            Stmt::Let {
                name,
                ty,
                init,
                span,
            } => {
                let (prelude, init) = self.lift_expr(init)?;
                out.extend(prelude);
                out.push(Stmt::Let {
                    name: name.clone(),
                    ty: ty.clone(),
                    init,
                    span: *span,
                });
            }
            Stmt::Var {
                name,
                ty,
                init,
                span,
            } => {
                let (prelude, init) = self.lift_expr(init)?;
                out.extend(prelude);
                out.push(Stmt::Var {
                    name: name.clone(),
                    ty: ty.clone(),
                    init,
                    span: *span,
                });
            }
            Stmt::Assign {
                target,
                value,
                span,
            } => {
                let (prelude, value) = self.lift_expr(value)?;
                out.extend(prelude);
                out.push(Stmt::Assign {
                    target: target.clone(),
                    value,
                    span: *span,
                });
            }
            Stmt::If {
                cond,
                then,
                els,
                span,
            } => {
                let (prelude, cond) = self.lift_expr(cond)?;
                out.extend(prelude);
                out.push(Stmt::If {
                    cond,
                    then: self.expand_stmts(then)?,
                    els: self.expand_stmts(els)?,
                    span: *span,
                });
            }
            Stmt::Loop { body, span } => {
                out.push(Stmt::Loop {
                    body: self.expand_stmts(body)?,
                    span: *span,
                });
            }
            Stmt::For {
                var,
                start,
                end,
                body,
                unroll,
                span,
            } => {
                let (start_prelude, start) = self.lift_expr(start)?;
                let (end_prelude, end) = self.lift_expr(end)?;
                out.extend(start_prelude);
                out.extend(end_prelude);
                out.push(Stmt::For {
                    var: var.clone(),
                    start,
                    end,
                    body: self.expand_stmts(body)?,
                    unroll: *unroll,
                    span: *span,
                });
            }
            Stmt::ExprStmt { expr, span } => {
                if let Expr::Call { name, args, .. } = expr {
                    if let Some(callee) = self.fns.get(name.as_str()).copied() {
                        let mut prelude = Vec::new();
                        let args = self.lift_args(args, &mut prelude)?;
                        if callee.ret.is_some() {
                            return Err(Diagnostic::new(
                                *span,
                                format!(
                                    "call to non-void function '{name}' must be used as a value"
                                ),
                            ));
                        }
                        let (mut body, _) = self.expand_call(callee, &args, *span)?;
                        out.extend(prelude);
                        out.append(&mut body);
                        return Ok(());
                    }
                }
                let (prelude, expr) = self.lift_expr(expr)?;
                out.extend(prelude);
                out.push(Stmt::ExprStmt { expr, span: *span });
            }
            _ => out.push(stmt.clone()),
        }
        Ok(())
    }

    fn lift_args(&mut self, args: &[Expr], prelude: &mut Vec<Stmt>) -> Result<Vec<Expr>> {
        let mut lifted = Vec::new();
        for arg in args {
            lifted.push(self.lift_inner(arg, prelude)?);
        }
        Ok(lifted)
    }

    fn lift_expr(&mut self, expr: &Expr) -> Result<(Vec<Stmt>, Expr)> {
        let mut prelude = Vec::new();
        let expr = self.lift_inner(expr, &mut prelude)?;
        Ok((prelude, expr))
    }

    fn lift_inner(&mut self, expr: &Expr, prelude: &mut Vec<Stmt>) -> Result<Expr> {
        match expr {
            Expr::Call { name, args, span } => {
                let mut lifted_args = Vec::new();
                for arg in args {
                    lifted_args.push(self.lift_inner(arg, prelude)?);
                }
                if let Some(callee) = self.fns.get(name.as_str()).copied() {
                    if callee.ret.is_none() {
                        return Err(Diagnostic::new(
                            *span,
                            format!("void function '{name}' used as a value"),
                        ));
                    }
                    let (mut body, ret) = self.expand_call(callee, &lifted_args, *span)?;
                    let tmp = self.fresh("tmp");
                    body.push(Stmt::Let {
                        name: tmp.clone(),
                        ty: None,
                        init: ret.expect("non-void function returns a value"),
                        span: *span,
                    });
                    prelude.append(&mut body);
                    return Ok(Expr::Name(tmp, *span));
                }
                Ok(Expr::Call {
                    name: name.clone(),
                    args: lifted_args,
                    span: *span,
                })
            }
            Expr::Index { base, index, span } => {
                let base = self.lift_inner(base, prelude)?;
                let index = self.lift_inner(index, prelude)?;
                Ok(Expr::Index {
                    base: Box::new(base),
                    index: Box::new(index),
                    span: *span,
                })
            }
            Expr::Member { base, idx, span } => Ok(Expr::Member {
                base: Box::new(self.lift_inner(base, prelude)?),
                idx: *idx,
                span: *span,
            }),
            Expr::Unary { op, expr: e, span } => Ok(Expr::Unary {
                op: *op,
                expr: Box::new(self.lift_inner(e, prelude)?),
                span: *span,
            }),
            Expr::Binary {
                op,
                lhs,
                rhs,
                span,
            } => Ok(Expr::Binary {
                op: *op,
                lhs: Box::new(self.lift_inner(lhs, prelude)?),
                rhs: Box::new(self.lift_inner(rhs, prelude)?),
                span: *span,
            }),
            Expr::Cond {
                cond,
                then,
                els,
                span,
            } => Ok(Expr::Cond {
                cond: Box::new(self.lift_inner(cond, prelude)?),
                then: Box::new(self.lift_inner(then, prelude)?),
                els: Box::new(self.lift_inner(els, prelude)?),
                span: *span,
            }),
            Expr::Convert { ty, expr: e, span } => Ok(Expr::Convert {
                ty: ty.clone(),
                expr: Box::new(self.lift_inner(e, prelude)?),
                span: *span,
            }),
            Expr::Construct { ty, args, span } => {
                let mut lifted = Vec::new();
                for arg in args {
                    lifted.push(self.lift_inner(arg, prelude)?);
                }
                Ok(Expr::Construct {
                    ty: ty.clone(),
                    args: lifted,
                    span: *span,
                })
            }
            Expr::Swizzle { base, mask, span } => Ok(Expr::Swizzle {
                base: Box::new(self.lift_inner(base, prelude)?),
                mask: mask.clone(),
                span: *span,
            }),
            _ => Ok(expr.clone()),
        }
    }

    fn expand_call(
        &mut self,
        callee: &FnDecl,
        args: &[Expr],
        span: Span,
    ) -> Result<(Vec<Stmt>, Option<Expr>)> {
        if self.call_stack.len() >= MAX_DEPTH {
            return Err(Diagnostic::new(
                span,
                format!("function call depth exceeds {MAX_DEPTH}"),
            ));
        }
        if self.call_stack.iter().any(|f| f == &callee.name) {
            return Err(Diagnostic::new(
                span,
                format!("recursive call to '{}'", callee.name),
            ));
        }
        if args.len() != callee.params.len() {
            return Err(Diagnostic::new(
                span,
                format!(
                    "function '{}' expects {} arguments, got {}",
                    callee.name,
                    callee.params.len(),
                    args.len()
                ),
            ));
        }
        let mut bindings = HashMap::new();
        for (param, arg) in callee.params.iter().zip(args) {
            if param.is_const {
                let value = consts::const_eval(arg, &self.consts).ok_or_else(|| {
                    Diagnostic::new(
                        arg.span(),
                        format!(
                            "const argument for '{}' must be a constant expression",
                            param.name
                        ),
                    )
                })?;
                let Type::Scalar(pty) = param.ty else {
                    unreachable!("const parameter validated as scalar");
                };
                if !consts::validate(&value, pty) {
                    return Err(Diagnostic::new(
                        arg.span(),
                        format!("const argument for '{}' out of range", param.name),
                    ));
                }
                bindings.insert(
                    param.name.clone(),
                    consts::to_literal(&value, pty, arg.span()),
                );
            } else {
                bindings.insert(param.name.clone(), arg.clone());
            }
        }
        let mut body = callee.body.clone();
        let mut shadow = vec![HashSet::new()];
        subst_stmts(&mut body, &bindings, &mut shadow);
        check_function_body(&body, callee)?;
        if callee.ret.is_some() && !must_return(&body) {
            return Err(Diagnostic::new(
                span,
                format!("function '{}' must return on all paths", callee.name),
            ));
        }
        self.call_stack.push(callee.name.clone());
        let ret_name = callee.ret.as_ref().map(|_| self.fresh("ret"));
        let done_name = self.fresh("done");
        let ret_ref = ret_name
            .as_ref()
            .map(|name| Expr::Name(name.clone(), span));
        let mut wrapped = Vec::new();
        if let (Some(ret_name), Some(ret_ty)) = (&ret_name, &callee.ret) {
            wrapped.push(Stmt::Var {
                name: ret_name.clone(),
                ty: Some(ret_ty.clone()),
                init: zero_expr(ret_ty, span),
                span,
            });
        }
        wrapped.push(Stmt::Var {
            name: done_name.clone(),
            ty: Some(Type::Scalar(Scalar::Bool)),
            init: Expr::BoolLit(false, span),
            span,
        });
        transform_returns(&mut body, ret_name.as_deref(), &done_name);
        let mut loop_body = self.expand_stmts(&body)?;
        loop_body.push(Stmt::Break { span });
        wrapped.push(Stmt::Loop {
            body: loop_body,
            span,
        });
        self.call_stack.pop();
        Ok((wrapped, ret_ref))
    }
}

fn subst_stmts(
    stmts: &mut [Stmt],
    bindings: &HashMap<String, Expr>,
    shadow: &mut Vec<HashSet<String>>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Let { name, init, .. } | Stmt::Var { name, init, .. } => {
                subst_expr(init, bindings, shadow);
                shadow.last_mut().unwrap().insert(name.clone());
            }
            Stmt::Shared { name, len, .. } => {
                subst_expr(len, bindings, shadow);
                shadow.last_mut().unwrap().insert(name.clone());
            }
            Stmt::Const { name, init, .. } => {
                subst_expr(init, bindings, shadow);
                shadow.last_mut().unwrap().insert(name.clone());
            }
            Stmt::Assign { target, value, .. } => {
                subst_expr(target, bindings, shadow);
                subst_expr(value, bindings, shadow);
            }
            Stmt::If {
                cond, then, els, ..
            } => {
                subst_expr(cond, bindings, shadow);
                shadow.push(HashSet::new());
                subst_stmts(then, bindings, shadow);
                shadow.pop();
                shadow.push(HashSet::new());
                subst_stmts(els, bindings, shadow);
                shadow.pop();
            }
            Stmt::Loop { body, .. } => {
                shadow.push(HashSet::new());
                subst_stmts(body, bindings, shadow);
                shadow.pop();
            }
            Stmt::For {
                var,
                start,
                end,
                body,
                ..
            } => {
                subst_expr(start, bindings, shadow);
                subst_expr(end, bindings, shadow);
                shadow.push(HashSet::new());
                shadow.last_mut().unwrap().insert(var.clone());
                subst_stmts(body, bindings, shadow);
                shadow.pop();
            }
            Stmt::ExprStmt { expr, .. } => subst_expr(expr, bindings, shadow),
            Stmt::Return { value, .. } => {
                if let Some(value) = value {
                    subst_expr(value, bindings, shadow);
                }
            }
            _ => {}
        }
    }
}

fn subst_expr(expr: &mut Expr, bindings: &HashMap<String, Expr>, shadow: &[HashSet<String>]) {
    match expr {
        Expr::Name(name, _) => {
            if bindings.contains_key(name) && !shadow.iter().any(|s| s.contains(name)) {
                *expr = bindings[name].clone();
            }
        }
        Expr::Index { base, index, .. } => {
            subst_expr(base, bindings, shadow);
            subst_expr(index, bindings, shadow);
        }
        Expr::Member { base, .. } => subst_expr(base, bindings, shadow),
        Expr::Unary { expr: e, .. } => subst_expr(e, bindings, shadow),
        Expr::Binary { lhs, rhs, .. } => {
            subst_expr(lhs, bindings, shadow);
            subst_expr(rhs, bindings, shadow);
        }
        Expr::Cond {
            cond, then, els, ..
        } => {
            subst_expr(cond, bindings, shadow);
            subst_expr(then, bindings, shadow);
            subst_expr(els, bindings, shadow);
        }
        Expr::Convert { expr: e, .. } => subst_expr(e, bindings, shadow),
        Expr::Call { args, .. } => {
            for arg in args {
                subst_expr(arg, bindings, shadow);
            }
        }
        Expr::Construct { args, .. } => {
            for arg in args {
                subst_expr(arg, bindings, shadow);
            }
        }
        Expr::Swizzle { base, .. } => subst_expr(base, bindings, shadow),
        _ => {}
    }
}

fn check_function_body(stmts: &[Stmt], f: &FnDecl) -> Result<()> {
    let mut loop_depth = 0usize;
    check_stmts(stmts, f, &mut loop_depth)
}

fn check_stmts(stmts: &[Stmt], f: &FnDecl, loop_depth: &mut usize) -> Result<()> {
    for stmt in stmts {
        match stmt {
            Stmt::Return { span, .. } => {
                if f.ret.is_none() {
                    return Err(Diagnostic::new(
                        *span,
                        format!("'{}' does not return a value", f.name),
                    ));
                }
            }
            Stmt::Shared { span, .. } => {
                return Err(Diagnostic::new(
                    *span,
                    "shared arrays are not allowed inside functions".to_string(),
                ));
            }
            Stmt::Const { span, .. } => {
                return Err(Diagnostic::new(
                    *span,
                    "const declarations are not allowed inside functions".to_string(),
                ));
            }
            Stmt::Break { span } => {
                if *loop_depth == 0 {
                    return Err(Diagnostic::new(
                        *span,
                        "break inside function without an enclosing loop".to_string(),
                    ));
                }
            }
            Stmt::Continue { span } => {
                if *loop_depth == 0 {
                    return Err(Diagnostic::new(
                        *span,
                        "continue inside function without an enclosing loop".to_string(),
                    ));
                }
            }
            Stmt::If { then, els, .. } => {
                check_stmts(then, f, loop_depth)?;
                check_stmts(els, f, loop_depth)?;
            }
            Stmt::Loop { body, .. } => {
                *loop_depth += 1;
                check_stmts(body, f, loop_depth)?;
                *loop_depth -= 1;
            }
            Stmt::For { body, .. } => {
                *loop_depth += 1;
                check_stmts(body, f, loop_depth)?;
                *loop_depth -= 1;
            }
            _ => {}
        }
    }
    Ok(())
}

fn must_return(stmts: &[Stmt]) -> bool {
    let Some(last) = stmts.last() else {
        return false;
    };
    match last {
        Stmt::Return { .. } => true,
        Stmt::If {
            then, els, span, ..
        } if !els.is_empty() => {
            let _ = span;
            must_return(then) && must_return(els)
        }
        _ => false,
    }
}

fn transform_returns(stmts: &mut Vec<Stmt>, ret_name: Option<&str>, done_name: &str) {
    let mut checks = Vec::new();
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts.drain(..) {
        match stmt {
            Stmt::Return { value, span } => {
                if let (Some(ret_name), Some(value)) = (ret_name, value) {
                    out.push(Stmt::Assign {
                        target: Expr::Name(ret_name.to_string(), span),
                        value,
                        span,
                    });
                }
                out.push(Stmt::Assign {
                    target: Expr::Name(done_name.to_string(), span),
                    value: Expr::BoolLit(true, span),
                    span,
                });
                out.push(Stmt::Break { span });
            }
            Stmt::If {
                cond,
                mut then,
                mut els,
                span,
            } => {
                transform_returns(&mut then, ret_name, done_name);
                transform_returns(&mut els, ret_name, done_name);
                out.push(Stmt::If {
                    cond,
                    then,
                    els,
                    span,
                });
            }
            Stmt::Loop { mut body, span } => {
                let body_had_return = contains_return(&body);
                transform_returns(&mut body, ret_name, done_name);
                if body_had_return {
                    checks.push(span);
                }
                out.push(Stmt::Loop { body, span });
            }
            Stmt::For {
                var,
                start,
                end,
                mut body,
                unroll,
                span,
            } => {
                let body_had_return = contains_return(&body);
                transform_returns(&mut body, ret_name, done_name);
                if body_had_return {
                    checks.push(span);
                }
                out.push(Stmt::For {
                    var,
                    start,
                    end,
                    body,
                    unroll,
                    span,
                });
            }
            other => out.push(other),
        }
    }
    for span in checks {
        out.push(Stmt::If {
            cond: Expr::Name(done_name.to_string(), span),
            then: vec![Stmt::Break { span }],
            els: Vec::new(),
            span,
        });
    }
    *stmts = out;
}

fn contains_return(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Return { .. } => true,
        Stmt::If { then, els, .. } => contains_return(then) || contains_return(els),
        Stmt::Loop { body, .. } | Stmt::For { body, .. } => contains_return(body),
        _ => false,
    })
}

fn zero_expr(ty: &Type, span: Span) -> Expr {
    match ty {
        Type::Scalar(Scalar::F32 | Scalar::F16 | Scalar::Bf16) => Expr::FloatLit(0.0, span),
        Type::Scalar(Scalar::Bool) => Expr::BoolLit(false, span),
        Type::Scalar(_) => Expr::IntLit(0, span),
        Type::Vec { size, elem } => Expr::Construct {
            ty: Type::Vec {
                size: *size,
                elem: *elem,
            },
            args: vec![Expr::FloatLit(0.0, span); *size as usize],
            span,
        },
        _ => unreachable!("zero of non-scalar type"),
    }
}
