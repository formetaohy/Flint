pub use crate::ast::{BinOp, UnOp};
pub use saturn_core::{MatrixRole, Scalar};
use crate::diag::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(Scalar),
    Vec { size: u32, elem: Scalar },
    Buf(Scalar),
    SharedArray { elem: Scalar, len: u64 },
    Matrix { elem: Scalar, role: MatrixRole },
}

impl Type {
    pub fn elem(&self) -> Option<Scalar> {
        match self {
            Type::Scalar(scalar) => Some(*scalar),
            Type::Vec { elem, .. } => Some(*elem),
            Type::Buf(elem) => Some(*elem),
            Type::SharedArray { elem, .. } => Some(*elem),
            Type::Matrix { elem, .. } => Some(*elem),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub elem: Scalar,
    pub binding: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarParam {
    pub name: String,
    pub ty: Scalar,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shared {
    pub name: String,
    pub elem: Scalar,
    pub len: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub workgroup_size: [u32; 3],
    pub params: Vec<Param>,
    pub scalars: Vec<ScalarParam>,
    pub shareds: Vec<Shared>,
    pub coop_triples: Vec<(Scalar, Scalar, Scalar)>,
    pub coop_roles: Vec<(Scalar, MatrixRole)>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        id: u32,
        name: String,
        ty: Type,
        init: Expr,
        span: Span,
    },
    Var {
        id: u32,
        name: String,
        ty: Type,
        init: Expr,
        span: Span,
    },
    Assign {
        target: Expr,
        value: Expr,
        span: Span,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        els: Vec<Stmt>,
        span: Span,
    },
    Loop {
        body: Vec<Stmt>,
        span: Span,
    },
    For {
        id: u32,
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    Barrier {
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    ExprStmt {
        expr: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntLit {
        value: u64,
        ty: Scalar,
        span: Span,
    },
    FloatLit {
        value: f64,
        ty: Scalar,
        span: Span,
    },
    BoolLit {
        value: bool,
        span: Span,
    },
    ParamRef {
        name: String,
        elem: Scalar,
        span: Span,
    },
    ScalarRef {
        name: String,
        ty: Scalar,
        span: Span,
    },
    SharedRef {
        name: String,
        elem: Scalar,
        len: u64,
        span: Span,
    },
    LocalRef {
        id: u32,
        name: String,
        ty: Type,
        span: Span,
    },
    Builtin {
        name: &'static str,
        size: u32,
        span: Span,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        ty: Scalar,
        span: Span,
    },
    Member {
        base: Box<Expr>,
        idx: u32,
        ty: Scalar,
        span: Span,
    },
    Unary {
        op: UnOp,
        expr: Box<Expr>,
        ty: Scalar,
        span: Span,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        ty: Type,
        span: Span,
    },
    Cond {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
        ty: Scalar,
        span: Span,
    },
    Convert {
        ty: Scalar,
        expr: Box<Expr>,
        span: Span,
    },
    Call {
        name: &'static str,
        args: Vec<Expr>,
        ty: Type,
        span: Span,
    },
}

impl Expr {
    pub fn set_span(&mut self, span: Span) {
        match self {
            Expr::IntLit { span: s, .. }
            | Expr::FloatLit { span: s, .. }
            | Expr::BoolLit { span: s, .. }
            | Expr::ParamRef { span: s, .. }
            | Expr::ScalarRef { span: s, .. }
            | Expr::SharedRef { span: s, .. }
            | Expr::LocalRef { span: s, .. }
            | Expr::Builtin { span: s, .. }
            | Expr::Index { span: s, .. }
            | Expr::Member { span: s, .. }
            | Expr::Unary { span: s, .. }
            | Expr::Binary { span: s, .. }
            | Expr::Cond { span: s, .. }
            | Expr::Convert { span: s, .. }
            | Expr::Call { span: s, .. } => *s = span,
        }
    }
}
