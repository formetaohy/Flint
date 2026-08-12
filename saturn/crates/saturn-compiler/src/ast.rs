use crate::diag::Span;
use crate::ir::{Access, MemOrder, Scalar};

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
pub enum Type {
    Scalar(Scalar),
    Vec { size: u32, elem: Scalar },
    Buf(Box<Type>),
    Array { elem: Box<Type>, len: Box<Expr> },
    Threadgroup(Box<Type>),
    Matrix(Scalar),
    Struct(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<(String, Type)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub imports: Vec<(String, Span)>,
    pub structs: Vec<StructDecl>,
    pub fns: Vec<FnDecl>,
    pub kernel: Option<Kernel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<FnParam>,
    pub ret: Option<Type>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnParam {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub workgroup_size: [u32; 3],
    pub params: Vec<Param>,
    pub specs: Vec<SpecDecl>,
    pub structs: Vec<StructDecl>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub binding: Option<u32>,
    pub access: Access,
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
        init: Option<Expr>,
        mutable: bool,
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

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntLit {
        value: u64,
        ty: Option<Scalar>,
        span: Span,
    },
    FloatLit {
        value: f64,
        ty: Option<Scalar>,
        span: Span,
    },
    BoolLit {
        value: bool,
        span: Span,
    },
    Name(String, Span),
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    Field {
        base: Box<Expr>,
        name: String,
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
    OrderLit(MemOrder, Span),
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
    ConstructStruct {
        name: String,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::IntLit { span, .. }
            | Expr::FloatLit { span, .. }
            | Expr::BoolLit { span, .. }
            | Expr::Name(_, span)
            | Expr::Index { span, .. }
            | Expr::Field { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Cond { span, .. }
            | Expr::Convert { span, .. }
            | Expr::OrderLit(_, span)
            | Expr::Call { span, .. }
            | Expr::Construct { span, .. }
            | Expr::ConstructStruct { span, .. } => *span,
        }
    }
}
