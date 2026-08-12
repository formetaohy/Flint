use std::collections::HashMap;

use crate::ast::{self, BinOp, UnOp};
use crate::diag::Span;
use crate::ir::Scalar;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CVal {
    Int(u64),
    Float(f64),
    Bool(bool),
}

pub fn const_eval(expr: &ast::Expr, consts: &HashMap<String, (CVal, Scalar)>) -> Option<CVal> {
    match expr {
        ast::Expr::IntLit { value, .. } => Some(CVal::Int(*value)),
        ast::Expr::FloatLit { value, .. } => Some(CVal::Float(*value)),
        ast::Expr::BoolLit { value, .. } => Some(CVal::Bool(*value)),
        ast::Expr::Name(name, _) => consts.get(name).map(|(value, _)| *value),
        ast::Expr::Unary { op, expr, .. } => {
            let value = const_eval(expr, consts)?;
            match (op, value) {
                (UnOp::Neg, CVal::Int(value)) => Some(CVal::Int(value.wrapping_neg())),
                (UnOp::Neg, CVal::Float(value)) => Some(CVal::Float(-value)),
                (UnOp::Not, CVal::Bool(value)) => Some(CVal::Bool(!value)),
                _ => None,
            }
        }
        ast::Expr::Binary { op, lhs, rhs, .. } => {
            let lhs = const_eval(lhs, consts)?;
            let rhs = const_eval(rhs, consts)?;
            match (op, lhs, rhs) {
                (BinOp::Add, CVal::Int(a), CVal::Int(b)) => Some(CVal::Int(a.wrapping_add(b))),
                (BinOp::Sub, CVal::Int(a), CVal::Int(b)) => Some(CVal::Int(a.wrapping_sub(b))),
                (BinOp::Mul, CVal::Int(a), CVal::Int(b)) => Some(CVal::Int(a.wrapping_mul(b))),
                (BinOp::Div, CVal::Int(a), CVal::Int(b)) if b != 0 => Some(CVal::Int(a / b)),
                (BinOp::Rem, CVal::Int(a), CVal::Int(b)) if b != 0 => Some(CVal::Int(a % b)),
                (BinOp::And, CVal::Int(a), CVal::Int(b)) => Some(CVal::Int(a & b)),
                (BinOp::Or, CVal::Int(a), CVal::Int(b)) => Some(CVal::Int(a | b)),
                (BinOp::Xor, CVal::Int(a), CVal::Int(b)) => Some(CVal::Int(a ^ b)),
                (BinOp::Shl, CVal::Int(a), CVal::Int(b)) => {
                    Some(CVal::Int(a.wrapping_shl(b as u32)))
                }
                (BinOp::Shr, CVal::Int(a), CVal::Int(b)) => {
                    Some(CVal::Int(a.wrapping_shr(b as u32)))
                }
                (BinOp::Add, CVal::Float(a), CVal::Float(b)) => Some(CVal::Float(a + b)),
                (BinOp::Sub, CVal::Float(a), CVal::Float(b)) => Some(CVal::Float(a - b)),
                (BinOp::Mul, CVal::Float(a), CVal::Float(b)) => Some(CVal::Float(a * b)),
                (BinOp::Div, CVal::Float(a), CVal::Float(b)) => Some(CVal::Float(a / b)),
                (BinOp::Eq, a, b) => Some(CVal::Bool(a == b)),
                (BinOp::Ne, a, b) => Some(CVal::Bool(a != b)),
                (BinOp::Lt, CVal::Int(a), CVal::Int(b)) => Some(CVal::Bool(a < b)),
                (BinOp::Le, CVal::Int(a), CVal::Int(b)) => Some(CVal::Bool(a <= b)),
                (BinOp::Gt, CVal::Int(a), CVal::Int(b)) => Some(CVal::Bool(a > b)),
                (BinOp::Ge, CVal::Int(a), CVal::Int(b)) => Some(CVal::Bool(a >= b)),
                (BinOp::Lt, CVal::Float(a), CVal::Float(b)) => Some(CVal::Bool(a < b)),
                (BinOp::Le, CVal::Float(a), CVal::Float(b)) => Some(CVal::Bool(a <= b)),
                (BinOp::Gt, CVal::Float(a), CVal::Float(b)) => Some(CVal::Bool(a > b)),
                (BinOp::Ge, CVal::Float(a), CVal::Float(b)) => Some(CVal::Bool(a >= b)),
                (BinOp::LAnd, CVal::Bool(a), CVal::Bool(b)) => Some(CVal::Bool(a && b)),
                (BinOp::LOr, CVal::Bool(a), CVal::Bool(b)) => Some(CVal::Bool(a || b)),
                _ => None,
            }
        }
        ast::Expr::Cond {
            cond, then, els, ..
        } => match const_eval(cond, consts)? {
            CVal::Bool(true) => const_eval(then, consts),
            CVal::Bool(false) => const_eval(els, consts),
            _ => None,
        },
        ast::Expr::Convert { ty, expr, .. } => {
            let value = const_eval(expr, consts)?;
            let ast::Type::Scalar(target) = ty else {
                return None;
            };
            match (value, target) {
                (CVal::Int(value), Scalar::F32 | Scalar::F16) => Some(CVal::Float(value as f64)),
                (CVal::Float(value), Scalar::U32 | Scalar::I32) => Some(CVal::Int(value as u64)),
                (CVal::Float(_), Scalar::F32 | Scalar::F16) => Some(value),
                (CVal::Int(_), Scalar::U32 | Scalar::I32) => Some(value),
                _ => None,
            }
        }
        ast::Expr::Call { name, args, .. } => match name.as_str() {
            "min" | "max" | "clamp" => {
                let mut values = Vec::new();
                for arg in args {
                    values.push(const_eval(arg, consts)?);
                }
                match values.as_slice() {
                    [CVal::Int(a), CVal::Int(b)] => Some(CVal::Int(match name.as_str() {
                        "min" => (*a).min(*b),
                        _ => (*a).max(*b),
                    })),
                    [CVal::Int(a), CVal::Int(b), CVal::Int(c)] => {
                        Some(CVal::Int(match name.as_str() {
                            "clamp" => (*a).max(*b).min(*c),
                            _ => unreachable!(),
                        }))
                    }
                    [CVal::Float(a), CVal::Float(b)] => Some(CVal::Float(match name.as_str() {
                        "min" => (*a).min(*b),
                        _ => (*a).max(*b),
                    })),
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    }
}

pub fn validate(value: &CVal, ty: Scalar) -> bool {
    match (value, ty) {
        (CVal::Int(value), Scalar::U32) => *value <= u32::MAX as u64,
        (CVal::Int(value), Scalar::I32) => *value <= i32::MAX as u64,
        (CVal::Int(value), Scalar::U8) => *value <= u8::MAX as u64,
        (CVal::Int(value), Scalar::I8) => *value <= i8::MAX as u64,
        (CVal::Int(_), Scalar::F32 | Scalar::F16 | Scalar::Bf16) => true,
        (CVal::Int(_), Scalar::Bool) => false,
        (CVal::Float(_), Scalar::F32 | Scalar::F16 | Scalar::Bf16) => true,
        (CVal::Float(_), Scalar::U32 | Scalar::I32 | Scalar::U8 | Scalar::I8) => true,
        (CVal::Float(_), Scalar::Bool) => false,
        (CVal::Bool(_), Scalar::Bool) => true,
        (CVal::Bool(_), _) => false,
    }
}

pub fn to_literal(value: &CVal, ty: Scalar, span: Span) -> ast::Expr {
    match (value, ty) {
        (CVal::Int(value), Scalar::F32 | Scalar::F16 | Scalar::Bf16) => ast::Expr::FloatLit {
            value: *value as f64,
            ty: Some(Scalar::F32),
            span,
        },
        (CVal::Int(value), _) => ast::Expr::IntLit {
            value: *value,
            ty: Some(Scalar::U32),
            span,
        },
        (CVal::Float(value), Scalar::U32 | Scalar::I32 | Scalar::U8 | Scalar::I8) => {
            ast::Expr::IntLit {
                value: *value as u64,
                ty: Some(Scalar::U32),
                span,
            }
        }
        (CVal::Float(value), _) => ast::Expr::FloatLit {
            value: *value,
            ty: Some(Scalar::F32),
            span,
        },
        (CVal::Bool(value), _) => ast::Expr::BoolLit {
            value: *value,
            span,
        },
    }
}

pub fn may_negate(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::Unary { op: UnOp::Neg, .. } => true,
        ast::Expr::Binary { op: BinOp::Sub, .. } => true,
        ast::Expr::Binary { lhs, rhs, .. }
        | ast::Expr::Cond {
            then: lhs,
            els: rhs,
            ..
        } => may_negate(lhs) || may_negate(rhs),
        ast::Expr::Unary { expr: e, .. }
        | ast::Expr::Index { base: e, .. }
        | ast::Expr::Convert { expr: e, .. } => may_negate(e),
        _ => false,
    }
}
