use std::collections::HashMap;

use crate::diag::{Diagnostic, Result, Span};
use crate::ir::{self, Expr, Stmt};

pub fn check(kernel: &ir::Kernel) -> Result<()> {
    let mut ctx = Ctx {
        locals: HashMap::new(),
    };
    check_stmts(&kernel.body, &mut ctx, true)
}

struct Ctx {
    locals: HashMap<u32, bool>,
}

fn uniform_expr(expr: &Expr, ctx: &Ctx) -> bool {
    match expr {
        Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. } => true,
        Expr::ParamRef { .. } | Expr::SharedRef { .. } => false,
        Expr::ScalarRef { .. } => true,
        Expr::LocalRef { id, .. } => ctx.locals.get(id).copied().unwrap_or(false),
        Expr::Builtin { name, .. } => matches!(*name, "block" | "block_dim"),
        Expr::Index { index, .. } => uniform_expr(index, ctx),
        Expr::Member { base, .. } => uniform_expr(base, ctx),
        Expr::Unary { expr: e, .. } => uniform_expr(e, ctx),
        Expr::Binary { lhs, rhs, .. } => uniform_expr(lhs, ctx) && uniform_expr(rhs, ctx),
        Expr::Cond {
            cond, then, els, ..
        } => uniform_expr(cond, ctx) && uniform_expr(then, ctx) && uniform_expr(els, ctx),
        Expr::Convert { expr: e, .. } => uniform_expr(e, ctx),
        Expr::Call { name, args, .. } => match *name {
            "coop_zero" => true,
            "barrier" => true,
            "subgroup_broadcast"
            | "subgroup_shuffle"
            | "subgroup_shuffle_down"
            | "subgroup_shuffle_up"
            | "subgroup_reduce_add"
            | "subgroup_reduce_max"
            | "subgroup_reduce_min"
            | "subgroup_inclusive_add"
            | "subgroup_all"
            | "subgroup_any"
            | "coop_load_a"
            | "coop_load_b"
            | "coop_mul_add"
            | "coop_store" => false,
            "atomic_add" | "atomic_max" | "atomic_min" | "atomic_exchange" | "atomic_and"
            | "atomic_or" | "atomic_xor" => false,
            _ => args.iter().all(|arg| uniform_expr(arg, ctx)),
        },
    }
}

fn check_stmts(stmts: &[Stmt], ctx: &mut Ctx, uniform: bool) -> Result<()> {
    for stmt in stmts {
        match stmt {
            Stmt::Let { id, init, .. } | Stmt::Var { id, init, .. } => {
                let u = uniform && uniform_expr(init, ctx);
                ctx.locals.insert(*id, u);
            }
            Stmt::Assign { target, value, .. } => {
                if let Expr::LocalRef { id, .. } = target {
                    let u = uniform && uniform_expr(value, ctx);
                    ctx.locals.insert(*id, u);
                }
            }
            Stmt::If {
                cond,
                then,
                els,
                ..
            } => {
                let u = uniform && uniform_expr(cond, ctx);
                let before = ctx.locals.clone();
                let mut then_ctx = Ctx {
                    locals: before.clone(),
                };
                let mut els_ctx = Ctx {
                    locals: before.clone(),
                };
                check_stmts(then, &mut then_ctx, u)?;
                check_stmts(els, &mut els_ctx, u)?;
                ctx.locals = merge_branches(&before, &then_ctx.locals, &els_ctx.locals);
            }
            Stmt::Loop { body, .. } => {
                let before = ctx.locals.clone();
                let mut body_ctx = Ctx {
                    locals: before.clone(),
                };
                check_stmts(body, &mut body_ctx, uniform)?;
                ctx.locals = merge_back(&before, &body_ctx.locals);
            }
            Stmt::For {
                id, start, end, body, ..
            } => {
                let u = uniform && uniform_expr(start, ctx) && uniform_expr(end, ctx);
                let before = ctx.locals.clone();
                let mut body_ctx = Ctx {
                    locals: before.clone(),
                };
                body_ctx.locals.insert(*id, u);
                check_stmts(body, &mut body_ctx, u)?;
                ctx.locals = merge_back(&before, &body_ctx.locals);
            }
            Stmt::Barrier { span } => {
                if !uniform {
                    return Err(barrier_error(*span));
                }
            }
            Stmt::ExprStmt { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
        }
    }
    Ok(())
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
        if !then.contains_key(id) {
            if let Some(prev) = merged.get(id) {
                merged.insert(*id, *prev && *u2);
            }
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

fn barrier_error(span: Span) -> Diagnostic {
    Diagnostic::new(
        span,
        "barrier() inside non-uniform control flow: the enclosing condition \
         depends on per-invocation values (gid, thread, lane, subgroup state, \
         or buffer/shared reads with non-uniform indices)",
    )
}
