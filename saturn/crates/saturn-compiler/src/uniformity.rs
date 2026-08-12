use std::collections::HashMap;

use crate::builtin;
use crate::diag::{Diagnostic, Result, Span};
use crate::ir::{self, Expr, Stmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Fallthrough,
    Break,
    Continue,
}

pub fn check(kernel: &ir::Kernel) -> Result<()> {
    let mut ctx = Ctx {
        locals: HashMap::new(),
    };
    check_stmts(&kernel.body, &mut ctx, true)?;
    Ok(())
}

struct Ctx {
    locals: HashMap<u32, bool>,
}

fn uniform_expr(expr: &Expr, ctx: &Ctx) -> bool {
    match expr {
        Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. } => true,
        Expr::ParamRef { .. } => false,
        Expr::LocalRef { id, .. } => ctx.locals.get(id).copied().unwrap_or(false),
        Expr::ScalarRef { .. } => true,
        Expr::Index { index, .. } => uniform_expr(index, ctx),
        Expr::Field { base, .. } => uniform_expr(base, ctx),
        Expr::Unary { expr: e, .. } => uniform_expr(e, ctx),
        Expr::Binary { lhs, rhs, .. } => uniform_expr(lhs, ctx) && uniform_expr(rhs, ctx),
        Expr::Cond {
            cond, then, els, ..
        } => uniform_expr(cond, ctx) && uniform_expr(then, ctx) && uniform_expr(els, ctx),
        Expr::Convert { expr: e, .. } => uniform_expr(e, ctx),
        Expr::OrderLit { .. } => true,
        Expr::ConstructStruct { fields, .. } => {
            fields.iter().all(|(_, e)| uniform_expr(e, ctx))
        }
        Expr::Call { name, args, .. } => match builtin::lookup(name) {
            Some(sig) => {
                if args.is_empty() {
                    sig.uniform
                } else if sig.uniform {
                    true
                } else {
                    args.iter().all(|arg| uniform_expr(arg, ctx))
                }
            }
            None => args.iter().all(|arg| uniform_expr(arg, ctx)),
        },
    }
}

fn check_stmts(stmts: &[Stmt], ctx: &mut Ctx, uniform: bool) -> Result<Term> {
    for stmt in stmts {
        let term = check_stmt(stmt, ctx, uniform)?;
        if !matches!(term, Term::Fallthrough) {
            return Ok(term);
        }
    }
    Ok(Term::Fallthrough)
}

fn check_stmt(stmt: &Stmt, ctx: &mut Ctx, uniform: bool) -> Result<Term> {
    match stmt {
        Stmt::Let { id, init, .. } | Stmt::Var { id, init: Some(init), .. } => {
            let u = uniform && uniform_expr(init, ctx);
            ctx.locals.insert(*id, u);
            Ok(Term::Fallthrough)
        }
        Stmt::Var { id, init: None, .. } => {
            ctx.locals.insert(*id, uniform);
            Ok(Term::Fallthrough)
        }
        Stmt::Assign { target, value, .. } => {
            let u = uniform && uniform_expr(value, ctx);
            match target {
                Expr::LocalRef { id, .. } => {
                    ctx.locals.insert(*id, u);
                }
                Expr::Field { base, .. } => {
                    if let Expr::LocalRef { id, .. } = &**base {
                        ctx.locals.insert(*id, u);
                    }
                }
                _ => {}
            }
            Ok(Term::Fallthrough)
        }
        Stmt::If {
            cond, then, els, ..
        } => {
            let u = uniform && uniform_expr(cond, ctx);
            let before = ctx.locals.clone();
            let mut then_ctx = Ctx {
                locals: before.clone(),
            };
            let mut els_ctx = Ctx {
                locals: before.clone(),
            };
            let then_term = check_stmts(then, &mut then_ctx, u)?;
            let els_term = check_stmts(els, &mut els_ctx, u)?;
            ctx.locals = match (then_term, els_term) {
                (Term::Fallthrough, Term::Fallthrough) => {
                    merge_branches(&before, &then_ctx.locals, &els_ctx.locals)
                }
                (Term::Fallthrough, _) => then_ctx.locals,
                (_, Term::Fallthrough) => els_ctx.locals,
                _ => before,
            };
            Ok(Term::Fallthrough)
        }
        Stmt::Loop { body, .. } => {
            let before = ctx.locals.clone();
            let mut body_ctx = Ctx {
                locals: before.clone(),
            };
            check_stmts(body, &mut body_ctx, uniform)?;
            ctx.locals = merge_back(&before, &body_ctx.locals);
            Ok(Term::Fallthrough)
        }
        Stmt::For {
            id,
            start,
            end,
            body,
            ..
        } => {
            let u = uniform && uniform_expr(start, ctx) && uniform_expr(end, ctx);
            let before = ctx.locals.clone();
            let mut body_ctx = Ctx {
                locals: before.clone(),
            };
            body_ctx.locals.insert(*id, u);
            check_stmts(body, &mut body_ctx, u)?;
            ctx.locals = merge_back(&before, &body_ctx.locals);
            Ok(Term::Fallthrough)
        }
        Stmt::Barrier { span } => {
            if !uniform {
                return Err(barrier_error(*span));
            }
            Ok(Term::Fallthrough)
        }
        Stmt::Break { .. } => Ok(Term::Break),
        Stmt::Continue { .. } => Ok(Term::Continue),
        Stmt::ExprStmt { expr, .. } => {
            if let Expr::Call { name, .. } = expr
                && let Some(sig) = builtin::lookup(name)
                && sig.requires_uniform
                && !uniform
            {
                return Err(barrier_error(expr_span(expr)));
            }
            Ok(Term::Fallthrough)
        }
    }
}

fn merge_branches(
    before: &HashMap<u32, bool>,
    then: &HashMap<u32, bool>,
    els: &HashMap<u32, bool>,
) -> HashMap<u32, bool> {
    let mut merged = before.clone();
    for (id, u1) in then {
        match (merged.get(id), els.get(id)) {
            (Some(prev), Some(u2)) => {
                merged.insert(*id, *prev && *u1 && *u2);
            }
            (Some(prev), None) => {
                merged.insert(*id, *prev && *u1);
            }
            (None, Some(u2)) => {
                merged.insert(*id, *u1 && *u2);
            }
            (None, None) => {}
        }
    }
    for (id, u2) in els {
        if !then.contains_key(id)
            && let Some(prev) = merged.get(id)
        {
            merged.insert(*id, *prev && *u2);
        }
    }
    merged
}

fn merge_back(before: &HashMap<u32, bool>, body: &HashMap<u32, bool>) -> HashMap<u32, bool> {
    let mut merged = before.clone();
    for (id, u) in body {
        if let Some(prev) = merged.get(id) {
            merged.insert(*id, *prev && *u);
        }
    }
    merged
}

fn expr_span(expr: &Expr) -> Span {
    match expr {
        Expr::IntLit { span, .. }
        | Expr::FloatLit { span, .. }
        | Expr::BoolLit { span, .. }
        | Expr::ParamRef { span, .. }
        | Expr::LocalRef { span, .. }
        | Expr::ScalarRef { span, .. }
        | Expr::Index { span, .. }
        | Expr::Field { span, .. }
        | Expr::Unary { span, .. }
        | Expr::Binary { span, .. }
        | Expr::Cond { span, .. }
        | Expr::Convert { span, .. }
        | Expr::OrderLit { span, .. }
        | Expr::ConstructStruct { span, .. }
        | Expr::Call { span, .. } => *span,
    }
}



fn barrier_error(span: Span) -> Diagnostic {
    Diagnostic::new(
        span,
        "barrier() inside non-uniform control flow: the enclosing condition depends on \
         per-invocation values (@local_id, @lane, buffer or \
         threadgroup reads with non-uniform indices)",
    )
}
