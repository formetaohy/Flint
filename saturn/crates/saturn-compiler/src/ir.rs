pub use crate::ast::{BinOp, UnOp};
use crate::diag::Span;
pub use saturn_core::{MatrixRole, Scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl MemOrder {
    pub fn name(&self) -> &'static str {
        match self {
            MemOrder::Relaxed => "relaxed",
            MemOrder::Acquire => "acquire",
            MemOrder::Release => "release",
            MemOrder::AcqRel => "acq_rel",
            MemOrder::SeqCst => "seq_cst",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(Scalar),
    Vec { size: u32, elem: Scalar },
    Buf(Box<Type>),
    Array { elem: Box<Type>, len: u64 },
    Threadgroup(Box<Type>),
    Matrix { elem: Scalar, role: MatrixRole },
    Struct { name: String, fields: Vec<(String, Type)> },
}

impl Type {
    pub fn elem(&self) -> Option<Scalar> {
        match self {
            Type::Scalar(scalar) => Some(*scalar),
            Type::Vec { elem, .. } => Some(*elem),
            Type::Buf(elem) => elem.elem(),
            Type::Array { elem, .. } => elem.elem(),
            Type::Threadgroup(elem) => elem.elem(),
            Type::Matrix { elem, .. } => Some(*elem),
            Type::Struct { fields, .. } => fields.first().map(|(_, ty)| ty.elem()).flatten(),
        }
    }

    pub fn scalar(&self) -> Option<Scalar> {
        match self {
            Type::Scalar(scalar) => Some(*scalar),
            Type::Vec { elem, .. } | Type::Matrix { elem, .. } => Some(*elem),
            _ => None,
        }
    }

    pub fn is_float(&self) -> bool {
        self.scalar().is_some_and(|s| s.is_float())
    }

    pub fn is_int(&self) -> bool {
        self.scalar().is_some_and(|s| s.is_int())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub binding: u32,
    pub access: Access,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarParam {
    pub name: String,
    pub ty: Scalar,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<(String, Type)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub workgroup_size: [u32; 3],
    pub params: Vec<Param>,
    pub scalars: Vec<ScalarParam>,
    pub structs: Vec<StructDecl>,
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
        init: Option<Expr>,
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
        ty: Type,
        span: Span,
    },
    ScalarRef {
        name: String,
        ty: Scalar,
        span: Span,
    },
    LocalRef {
        id: u32,
        name: String,
        ty: Type,
        span: Span,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        ty: Type,
        span: Span,
    },
    Field {
        base: Box<Expr>,
        name: String,
        ty: Type,
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
        ty: Type,
        span: Span,
    },
    Convert {
        ty: Scalar,
        expr: Box<Expr>,
        span: Span,
    },
    OrderLit {
        order: MemOrder,
        span: Span,
    },
    ConstructStruct {
        name: String,
        fields: Vec<(String, Expr)>,
        ty: Type,
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
            | Expr::LocalRef { span: s, .. }
            | Expr::Index { span: s, .. }
            | Expr::Field { span: s, .. }
            | Expr::Unary { span: s, .. }
            | Expr::Binary { span: s, .. }
            | Expr::Cond { span: s, .. }
            | Expr::Convert { span: s, .. }
            | Expr::OrderLit { span: s, .. }
            | Expr::ConstructStruct { span: s, .. }
            | Expr::Call { span: s, .. } => *s = span,
        }
    }
}
