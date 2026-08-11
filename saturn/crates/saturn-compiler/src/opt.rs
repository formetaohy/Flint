use crate::ir::{self, BinOp, Scalar, UnOp};

pub trait Pass {
    fn name(&self) -> &'static str;
    fn run(&self, kernel: &mut ir::Kernel);
}

pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add(&mut self, pass: Box<dyn Pass>) -> &mut Self {
        self.passes.push(pass);
        self
    }

    pub fn run(&self, kernel: &mut ir::Kernel) {
        for pass in &self.passes {
            pass.run(kernel);
        }
    }
}

impl Default for PassManager {
    fn default() -> Self {
        let mut manager = Self::new();
        manager.add(Box::new(ConstFold));
        manager
    }
}

pub struct ConstFold;

impl Pass for ConstFold {
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(&self, kernel: &mut ir::Kernel) {
        kernel.body = fold_stmts(std::mem::take(&mut kernel.body));
    }
}

fn fold_stmts(stmts: Vec<ir::Stmt>) -> Vec<ir::Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        out.extend(fold_stmt(stmt));
    }
    out
}

fn fold_stmt(stmt: ir::Stmt) -> Vec<ir::Stmt> {
    match stmt {
        ir::Stmt::Let {
            id,
            name,
            ty,
            mut init,
            span,
        } => {
            fold_expr(&mut init);
            vec![ir::Stmt::Let {
                id,
                name,
                ty,
                init,
                span,
            }]
        }
        ir::Stmt::Var {
            id,
            name,
            ty,
            mut init,
            span,
        } => {
            fold_expr(&mut init);
            vec![ir::Stmt::Var {
                id,
                name,
                ty,
                init,
                span,
            }]
        }
        ir::Stmt::Assign {
            mut target,
            mut value,
            span,
        } => {
            fold_expr(&mut target);
            fold_expr(&mut value);
            vec![ir::Stmt::Assign {
                target,
                value,
                span,
            }]
        }
        ir::Stmt::If {
            mut cond,
            then,
            els,
            span,
        } => {
            fold_expr(&mut cond);
            if let ir::Expr::BoolLit { value, .. } = cond {
                let chosen = if value { then } else { els };
                fold_stmts(chosen)
            } else {
                vec![ir::Stmt::If {
                    cond,
                    then: fold_stmts(then),
                    els: fold_stmts(els),
                    span,
                }]
            }
        }
        ir::Stmt::Loop { body, span } => vec![ir::Stmt::Loop {
            body: fold_stmts(body),
            span,
        }],
        ir::Stmt::For {
            id,
            var,
            mut start,
            mut end,
            body,
            span,
        } => {
            fold_expr(&mut start);
            fold_expr(&mut end);
            let dead = match (&start, &end) {
                (ir::Expr::IntLit { value: a, .. }, ir::Expr::IntLit { value: b, .. }) => a >= b,
                _ => false,
            };
            if dead {
                Vec::new()
            } else {
                vec![ir::Stmt::For {
                    id,
                    var,
                    start,
                    end,
                    body: fold_stmts(body),
                    span,
                }]
            }
        }
        ir::Stmt::ExprStmt { mut expr, span } => {
            fold_expr(&mut expr);
            vec![ir::Stmt::ExprStmt { expr, span }]
        }
        other => vec![other],
    }
}

fn fold_expr(expr: &mut ir::Expr) {
    match expr {
        ir::Expr::Index { base, index, .. } => {
            fold_expr(base);
            fold_expr(index);
        }
        ir::Expr::Member { base, .. } => fold_expr(base),
        ir::Expr::Unary { expr: inner, .. } => {
            fold_expr(inner);
            fold_unary(expr);
        }
        ir::Expr::Binary { lhs, rhs, .. } => {
            fold_expr(lhs);
            fold_expr(rhs);
            fold_binary(expr);
        }
        ir::Expr::Cond {
            cond, then, els, ..
        } => {
            fold_expr(cond);
            fold_expr(then);
            fold_expr(els);
            fold_cond(expr);
        }
        ir::Expr::Convert { expr: inner, .. } => {
            fold_expr(inner);
            fold_convert(expr);
        }
        ir::Expr::Call { args, .. } => {
            for arg in args {
                fold_expr(arg);
            }
        }
        _ => {}
    }
}

fn fold_unary(expr: &mut ir::Expr) {
    let ir::Expr::Unary {
        op,
        expr: inner,
        ty: _,
        span,
    } = expr
    else {
        return;
    };
    let folded = match (op, &**inner) {
        (UnOp::Neg, ir::Expr::IntLit { value, ty, .. }) => Some(ir::Expr::IntLit {
            value: truncate(value.wrapping_neg(), *ty),
            ty: *ty,
            span: *span,
        }),
        (UnOp::Not, ir::Expr::BoolLit { value, .. }) => Some(ir::Expr::BoolLit {
            value: !*value,
            span: *span,
        }),
        _ => None,
    };
    if let Some(folded) = folded {
        *expr = folded;
    }
}

fn fold_binary(expr: &mut ir::Expr) {
    let ir::Expr::Binary {
        op,
        lhs,
        rhs,
        ty: _,
        span,
    } = expr
    else {
        return;
    };
    let folded = match (&**lhs, &**rhs) {
        (
            ir::Expr::IntLit {
                value: a, ty: a_ty, ..
            },
            ir::Expr::IntLit {
                value: b, ty: b_ty, ..
            },
        ) if a_ty == b_ty => fold_int_binary(*op, *a, *b, *a_ty).map(|value| match value {
            IntFold::Int(value) => ir::Expr::IntLit {
                value,
                ty: *a_ty,
                span: *span,
            },
            IntFold::Bool(value) => ir::Expr::BoolLit { value, span: *span },
        }),
        (ir::Expr::BoolLit { value: a, span: _ }, ir::Expr::BoolLit { value: b, span: _ }) => {
            fold_bool_binary(*op, *a, *b).map(|value| ir::Expr::BoolLit { value, span: *span })
        }
        _ => None,
    };
    if let Some(folded) = folded {
        *expr = folded;
    }
}

enum IntFold {
    Int(u64),
    Bool(bool),
}

fn fold_int_binary(op: BinOp, a: u64, b: u64, ty: Scalar) -> Option<IntFold> {
    if let Some(value) = fold_int_compare(op, a, b) {
        return Some(IntFold::Bool(value));
    }
    let value = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div if b != 0 => a / b,
        BinOp::Rem if b != 0 => a % b,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
        _ => return None,
    };
    Some(IntFold::Int(truncate(value, ty)))
}

fn fold_bool_binary(op: BinOp, a: bool, b: bool) -> Option<bool> {
    match op {
        BinOp::LAnd => Some(a && b),
        BinOp::LOr => Some(a || b),
        BinOp::Eq => Some(a == b),
        BinOp::Ne => Some(a != b),
        _ => None,
    }
}

fn fold_int_compare(op: BinOp, a: u64, b: u64) -> Option<bool> {
    match op {
        BinOp::Eq => Some(a == b),
        BinOp::Ne => Some(a != b),
        BinOp::Lt => Some(a < b),
        BinOp::Le => Some(a <= b),
        BinOp::Gt => Some(a > b),
        BinOp::Ge => Some(a >= b),
        _ => None,
    }
}

fn fold_cond(expr: &mut ir::Expr) {
    let ir::Expr::Cond {
        cond,
        then,
        els,
        span,
        ..
    } = expr
    else {
        return;
    };
    let chosen = match &**cond {
        ir::Expr::BoolLit { value: true, .. } => Some(then),
        ir::Expr::BoolLit { value: false, .. } => Some(els),
        _ => None,
    };
    if let Some(chosen) = chosen {
        let mut picked = chosen.clone();
        picked.set_span(*span);
        *expr = *picked;
    }
}

fn fold_convert(expr: &mut ir::Expr) {
    let ir::Expr::Convert {
        ty,
        expr: inner,
        span,
    } = expr
    else {
        return;
    };
    let folded = match (&**inner, *ty) {
        (ir::Expr::IntLit { value, .. }, Scalar::F32 | Scalar::F16) => Some(ir::Expr::FloatLit {
            value: (*value as f32) as f64,
            ty: *ty,
            span: *span,
        }),
        (ir::Expr::IntLit { value, .. }, Scalar::I32 | Scalar::U32) => Some(ir::Expr::IntLit {
            value: *value,
            ty: *ty,
            span: *span,
        }),
        (ir::Expr::FloatLit { value, .. }, Scalar::I32 | Scalar::U32)
            if value.fract() == 0.0 && *value >= i64::MIN as f64 && *value <= i64::MAX as f64 =>
        {
            Some(ir::Expr::IntLit {
                value: (*value as i64) as u64,
                ty: *ty,
                span: *span,
            })
        }
        _ => None,
    };
    if let Some(folded) = folded {
        *expr = folded;
    }
}

fn truncate(value: u64, ty: Scalar) -> u64 {
    match ty {
        Scalar::U32 | Scalar::I32 => value & 0xFFFF_FFFF,
        Scalar::U8 | Scalar::I8 => value & 0xFF,
        _ => value,
    }
}
