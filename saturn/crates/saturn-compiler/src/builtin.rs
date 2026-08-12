use crate::ir::MemOrder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    Num,
    Float,
    Bool,
    U32,
    Addr,
    Order,
    ConstBool,
    MatA,
    MatB,
    MatAcc,
    Vec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ret {
    SameAsFirst,
    Bool,
    U32,
    F32,
    Vec3U32,
    Void,
    MatA,
    MatB,
    MatAcc,
    ScalarOfVec,
}

pub struct Builtin {
    pub name: &'static str,
    pub args: &'static [Arg],
    pub ret: Ret,
    pub uniform: bool,
    pub requires_uniform: bool,
}

pub const BUILTINS: &[Builtin] = &[
    Builtin {
        name: "global_id",
        args: &[],
        ret: Ret::Vec3U32,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "local_id",
        args: &[],
        ret: Ret::Vec3U32,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "group_id",
        args: &[],
        ret: Ret::Vec3U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "group_size",
        args: &[],
        ret: Ret::Vec3U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_id",
        args: &[],
        ret: Ret::U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "lane",
        args: &[],
        ret: Ret::U32,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_size",
        args: &[],
        ret: Ret::U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "barrier",
        args: &[],
        ret: Ret::Void,
        uniform: true,
        requires_uniform: true,
    },
    Builtin {
        name: "atomic_add",
        args: &[Arg::Addr, Arg::Num, Arg::Order],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "atomic_max",
        args: &[Arg::Addr, Arg::Num, Arg::Order],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "atomic_min",
        args: &[Arg::Addr, Arg::Num, Arg::Order],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "atomic_exchange",
        args: &[Arg::Addr, Arg::Num, Arg::Order],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "atomic_and",
        args: &[Arg::Addr, Arg::Num, Arg::Order],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "atomic_or",
        args: &[Arg::Addr, Arg::Num, Arg::Order],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "atomic_xor",
        args: &[Arg::Addr, Arg::Num, Arg::Order],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "coop_zero",
        args: &[],
        ret: Ret::MatAcc,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "coop_load_a",
        args: &[Arg::Addr, Arg::U32, Arg::ConstBool],
        ret: Ret::MatA,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "coop_load_b",
        args: &[Arg::Addr, Arg::U32, Arg::ConstBool],
        ret: Ret::MatB,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "coop_mul_add",
        args: &[Arg::MatA, Arg::MatB, Arg::MatAcc],
        ret: Ret::MatAcc,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "coop_store",
        args: &[Arg::Addr, Arg::MatAcc, Arg::U32, Arg::ConstBool],
        ret: Ret::Void,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "min",
        args: &[Arg::Num, Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "max",
        args: &[Arg::Num, Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "clamp",
        args: &[Arg::Num, Arg::Num, Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "fma",
        args: &[Arg::Num, Arg::Num, Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "abs",
        args: &[Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "floor",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "ceil",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "round",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "trunc",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "sign",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "fract",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "sqrt",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "rsqrt",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "exp",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "exp2",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "log",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "log2",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "tanh",
        args: &[Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "pow",
        args: &[Arg::Float, Arg::Float],
        ret: Ret::SameAsFirst,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "popcount",
        args: &[Arg::U32],
        ret: Ret::U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "clz",
        args: &[Arg::U32],
        ret: Ret::U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "ctz",
        args: &[Arg::U32],
        ret: Ret::U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "bitcast_f32",
        args: &[Arg::U32],
        ret: Ret::F32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "bitcast_u32",
        args: &[Arg::Float],
        ret: Ret::U32,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "dot",
        args: &[Arg::Vec, Arg::Vec],
        ret: Ret::ScalarOfVec,
        uniform: true,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_broadcast",
        args: &[Arg::Num, Arg::U32],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_shuffle",
        args: &[Arg::Num, Arg::U32],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_shuffle_down",
        args: &[Arg::Num, Arg::U32],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_shuffle_up",
        args: &[Arg::Num, Arg::U32],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_reduce_add",
        args: &[Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_reduce_max",
        args: &[Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_reduce_min",
        args: &[Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_inclusive_add",
        args: &[Arg::Num],
        ret: Ret::SameAsFirst,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_all",
        args: &[Arg::Bool],
        ret: Ret::Bool,
        uniform: false,
        requires_uniform: false,
    },
    Builtin {
        name: "subgroup_any",
        args: &[Arg::Bool],
        ret: Ret::Bool,
        uniform: false,
        requires_uniform: false,
    },
];

pub fn lookup(name: &str) -> Option<&'static Builtin> {
    BUILTINS.iter().find(|b| b.name == name)
}

pub fn is_reserved(name: &str) -> bool {
    lookup(name).is_some()
        || matches!(
            name,
            "kernel"
                | "fn"
                | "return"
                | "spec"
                | "let"
                | "var"
                | "mut"
                | "const"
                | "if"
                | "else"
                | "loop"
                | "for"
                | "in"
                | "break"
                | "continue"
                | "as"
                | "import"
                | "struct"
                | "unroll"
                | "workgroup_size"
                | "buffer"
                | "threadgroup"
                | "true"
                | "false"
                | "readonly"
                | "writeonly"
                | "readwrite"
                | "relaxed"
                | "acquire"
                | "release"
                | "acq_rel"
                | "seq_cst"
                | "buf"
                | "vec2"
                | "vec3"
                | "vec4"
                | "matrix"
        )
}

pub fn order_name(order: MemOrder) -> &'static str {
    order.name()
}
