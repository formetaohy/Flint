use crate::scalar::Scalar;
use thuban_gpu::BindingMode;

pub struct ShaderSpec {
    pub name: &'static str,
    pub wgsl: &'static str,
    pub bindings: &'static [BindingMode],
    pub scalars: &'static [(&'static str, Scalar)],
}

macro_rules! shader {
    ($name:expr, [$($file:literal),+ $(,)?], $bindings:expr, $scalars:expr) => {
        ShaderSpec {
            name: $name,
            wgsl: concat!($(include_str!(concat!("../wgsl/", $file))),+),
            bindings: $bindings,
            scalars: $scalars,
        }
    };
}

use crate::shader;

pub const SHADERS: &[ShaderSpec] = &[
    shader!(
        shader::GEMM,
        ["quant.wgsl", "gemm.wgsl"],
        &[
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadWrite
        ],
        &[
            ("N", Scalar::U32),
            ("K", Scalar::U32),
            ("M", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("QTYPE", Scalar::U32),
            ("ACC", Scalar::U32),
            ("Y_STRIDE", Scalar::U32),
            ("Y_OFF", Scalar::U32),
        ]
    ),
    shader!(
        shader::GEMM_COOP,
        ["gemm_coop_common.wgsl", "quant.wgsl", "gemm_coop.wgsl"],
        &[
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadWrite
        ],
        &[
            ("N", Scalar::U32),
            ("K", Scalar::U32),
            ("M", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("QTYPE", Scalar::U32),
            ("ACC", Scalar::U32),
            ("Y_STRIDE", Scalar::U32),
            ("Y_OFF", Scalar::U32),
        ]
    ),
    shader!(
        shader::GEMM_COOP8,
        ["gemm_coop_common.wgsl", "quant.wgsl", "gemm_coop8.wgsl"],
        &[
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadWrite
        ],
        &[
            ("N", Scalar::U32),
            ("K", Scalar::U32),
            ("M", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("QTYPE", Scalar::U32),
            ("ACC", Scalar::U32),
            ("Y_STRIDE", Scalar::U32),
            ("Y_OFF", Scalar::U32),
        ]
    ),
    shader!(
        shader::TO_F16,
        ["to_f16.wgsl"],
        &[BindingMode::ReadOnly, BindingMode::ReadWrite],
        &[("N_ELEM", Scalar::U32)]
    ),
    shader!(
        shader::MERGE_GEMM,
        ["merge_gemm.wgsl"],
        &[BindingMode::ReadWrite; 2],
        &[
            ("M", Scalar::U32),
            ("N", Scalar::U32),
            ("Y_STRIDE", Scalar::U32),
            ("Y_OFF", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("ACC", Scalar::U32),
        ]
    ),
    shader!(
        shader::GEMV,
        ["quant.wgsl", "gemv.wgsl"],
        &[
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadWrite
        ],
        &[
            ("N", Scalar::U32),
            ("K", Scalar::U32),
            ("QTYPE", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("ACC", Scalar::U32),
        ]
    ),
    shader!(
        shader::MERGE_GEMV,
        ["merge_gemv.wgsl"],
        &[BindingMode::ReadWrite; 2],
        &[
            ("N", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("ACC", Scalar::U32)
        ]
    ),
    shader!(
        shader::EMBED,
        ["quant.wgsl", "embed.wgsl"],
        &[
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadOnly,
            BindingMode::ReadWrite
        ],
        &[
            ("M", Scalar::U32),
            ("DIM", Scalar::U32),
            ("SCALE", Scalar::F32),
            ("QTYPE", Scalar::U32),
        ]
    ),
    shader!(
        shader::NORM,
        ["norm.wgsl"],
        &[BindingMode::ReadWrite; 7],
        &[
            ("MODE", Scalar::U32),
            ("DIM", Scalar::U32),
            ("W_DIM", Scalar::U32),
            ("EPS", Scalar::F32),
            ("HEADS", Scalar::U32),
            ("ROT", Scalar::U32),
            ("COS_STRIDE", Scalar::U32),
            ("STRIDE", Scalar::U32),
            ("PLE", Scalar::U32),
            ("PLE_LAYERS", Scalar::U32),
            ("PLE_STRIDE", Scalar::U32),
        ]
    ),
    shader!(
        shader::ADD,
        ["add.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("N_ELEM", Scalar::U32)]
    ),
    shader!(
        shader::BIAS,
        ["bias.wgsl"],
        &[BindingMode::ReadWrite; 2],
        &[("N_ELEM", Scalar::U32), ("DIM", Scalar::U32)]
    ),
    shader!(
        shader::CONCAT,
        ["concat.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("ROWS", Scalar::U32), ("D", Scalar::U32)]
    ),
    shader!(
        shader::SWIGLU,
        ["swiglu.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("N_ELEM", Scalar::U32), ("MODE", Scalar::U32)]
    ),
    shader!(
        shader::SOFTCAP,
        ["softcap.wgsl"],
        &[BindingMode::ReadWrite; 1],
        &[("N_ELEM", Scalar::U32), ("CAP", Scalar::F32)]
    ),
    shader!(
        shader::MUL,
        ["mul.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[
            ("N", Scalar::U32),
            ("M", Scalar::U32),
            ("MODE", Scalar::U32),
            ("STRIDE", Scalar::U32),
            ("OFFSET", Scalar::U32),
        ]
    ),
    shader!(
        shader::ROPE,
        ["rope.wgsl"],
        &[BindingMode::ReadWrite; 4],
        &[
            ("HEADS", Scalar::U32),
            ("HEAD_DIM", Scalar::U32),
            ("ROT", Scalar::U32),
            ("COS_STRIDE", Scalar::U32),
        ]
    ),
    shader!(
        shader::ATTN,
        ["attn.wgsl"],
        &[BindingMode::ReadWrite; 6],
        &[
            ("M", Scalar::U32),
            ("N_HEADS", Scalar::U32),
            ("HEAD_DIM", Scalar::U32),
            ("POOL_LEN", Scalar::U32),
            ("SCALE", Scalar::F32),
            ("WINDOW", Scalar::U32),
            ("NQ_PER_KV", Scalar::U32),
            ("SEQ", Scalar::U32),
            ("CAUSAL", Scalar::U32),
            ("MAX_PAGES", Scalar::U32),
        ]
    ),
    shader!(
        shader::KV_STORE,
        ["kv_store.wgsl"],
        &[BindingMode::ReadWrite; 6],
        &[
            ("N_KV", Scalar::U32),
            ("HEAD_DIM", Scalar::U32),
            ("POOL_LEN", Scalar::U32),
            ("MAX_PAGES", Scalar::U32),
        ]
    ),
    shader!(
        shader::SPLIT_QG,
        ["split_qg.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[
            ("ROWS", Scalar::U32),
            ("HEADS", Scalar::U32),
            ("HD", Scalar::U32),
        ]
    ),
    shader!(
        shader::SIGMOID_MUL,
        ["sigmoid_mul.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("N_ELEM", Scalar::U32)]
    ),
    shader!(
        shader::CONV1D,
        ["conv1d.wgsl"],
        &[BindingMode::ReadWrite; 4],
        &[("DIM", Scalar::U32)]
    ),
    shader!(
        shader::DELTA_GATE,
        ["delta_gate.wgsl"],
        &[BindingMode::ReadWrite; 6],
        &[("HEADS", Scalar::U32), ("ROW_T", Scalar::U32)]
    ),
    shader!(
        shader::DELTA_RECUR,
        ["delta_recur.wgsl"],
        &[BindingMode::ReadWrite; 7],
        &[
            ("HEADS", Scalar::U32),
            ("K_DIM", Scalar::U32),
            ("V_DIM", Scalar::U32),
        ]
    ),
    shader!(
        shader::REPEAT_QK,
        ["repeat_qk.wgsl"],
        &[BindingMode::ReadWrite; 2],
        &[
            ("ROWS", Scalar::U32),
            ("N_K", Scalar::U32),
            ("N_V", Scalar::U32),
            ("K_DIM", Scalar::U32),
            ("CONV_DIM", Scalar::U32),
        ]
    ),
];
