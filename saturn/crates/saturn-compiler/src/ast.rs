use crate::diag::Span;

pub use crate::ir::{Scalar, Type};

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub fns: Vec<FnDecl>,
    pub kernel: Kernel,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub is_const: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub workgroup_size: [u32; 3],
    pub params: Vec<Param>,
    pub specs: Vec<SpecDecl>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpecDecl {
    pub name: String,
    pub ty: Scalar,
    pub init: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<Type>,
        init: Expr,
        span: Span,
    },
    Var {
        name: String,
        ty: Option<Type>,
        init: Expr,
        span: Span,
    },
    Shared {
        name: String,
        elem: Scalar,
        len: Expr,
        span: Span,
    },
    Const {
        name: String,
        ty: Scalar,
        init: Expr,
        span: Span,
    },
    Spec(SpecDecl),
    Return {
        value: Option<Expr>,
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
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
        unroll: bool,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LAnd,
    LOr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntLit(u64, Span),
    FloatLit(f64, Span),
    BoolLit(bool, Span),
    Name(String, Span),
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    Member {
        base: Box<Expr>,
        idx: u32,
        span: Span,
    },
    Unary {
        op: UnOp,
        expr: Box<Expr>,
        span: Span,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    Cond {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
        span: Span,
    },
    Convert {
        ty: Type,
        expr: Box<Expr>,
        span: Span,
    },
    Call {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    Construct {
        ty: Type,
        args: Vec<Expr>,
        span: Span,
    },
    Swizzle {
        base: Box<Expr>,
        mask: Vec<u32>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> crate::diag::Span {
        match self {
            Expr::IntLit(_, span)
            | Expr::FloatLit(_, span)
            | Expr::BoolLit(_, span)
            | Expr::Name(_, span)
            | Expr::Index { span, .. }
            | Expr::Member { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Cond { span, .. }
            | Expr::Convert { span, .. }
            | Expr::Call { span, .. }
            | Expr::Construct { span, .. }
            | Expr::Swizzle { span, .. } => *span,
        }
    }
}
