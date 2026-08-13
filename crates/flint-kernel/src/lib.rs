pub mod modes;
pub mod name {
    pub const GEMM: &str = "gemm";
    pub const MERGE_GEMM: &str = "merge_gemm";
    pub const GEMV: &str = "gemv";
    pub const MERGE_GEMV: &str = "merge_gemv";
    pub const EMBED: &str = "embed";
    pub const NORM: &str = "norm";
    pub const ADD: &str = "add";
    pub const BIAS: &str = "bias";
    pub const CONCAT: &str = "concat";
    pub const SWIGLU: &str = "swiglu";
    pub const SOFTCAP: &str = "softcap";
    pub const MUL: &str = "mul";
    pub const EXPERT_GATHER: &str = "expert_gather";
    pub const EXPERT_SCATTER: &str = "expert_scatter";
    pub const ZERO_ROWS: &str = "zero_rows";
    pub const SIGMOID_MUL: &str = "sigmoid_mul";
    pub const DELTA_GATE: &str = "delta_gate";
    pub const CONV1D: &str = "conv1d";
    pub const DELTA_RECUR: &str = "delta_recur";
    pub const REPEAT_QK: &str = "repeat_qk";
    pub const ROPE: &str = "rope";
    pub const ATTN: &str = "attn";
    pub const MERGE_ATTN: &str = "merge_attn";
    pub const KV_STORE: &str = "kv_store";
    pub const SPLIT_QG: &str = "split_qg";
}

use std::collections::HashMap;

use flint_error::{Error, Result};
use flint_gpu::{Device, Kernel};

mod scalar;

use scalar::{Scalar, ScalarField, ScalarLayout};

pub use modes::{Act, NormMode};

struct ShaderSpec {
    name: &'static str,
    wgsl: &'static str,
    bindings: u32,
    scalars: &'static [(&'static str, Scalar)],
}

macro_rules! shader {
    ($name:expr, $file:literal, $bindings:expr, $scalars:expr) => {
        ShaderSpec {
            name: $name,
            wgsl: include_str!(concat!("../wgsl/", $file)),
            bindings: $bindings,
            scalars: $scalars,
        }
    };
}

const SHADERS: &[ShaderSpec] = &[
    shader!(
        name::GEMM,
        "gemm.wgsl",
        4,
        &[
            ("N", Scalar::U32),
            ("K", Scalar::U32),
            ("M", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("WDTYPE", Scalar::U32),
            ("GROUP", Scalar::U32),
            ("ACC", Scalar::U32),
            ("Y_STRIDE", Scalar::U32),
            ("Y_OFF", Scalar::U32),
        ]
    ),
    shader!(
        name::MERGE_GEMM,
        "merge_gemm.wgsl",
        2,
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
        name::GEMV,
        "gemv.wgsl",
        4,
        &[
            ("N", Scalar::U32),
            ("K", Scalar::U32),
            ("WDTYPE", Scalar::U32),
            ("GROUP", Scalar::U32),
            ("SEGS", Scalar::U32),
            ("ACC", Scalar::U32),
        ]
    ),
    shader!(
        name::MERGE_GEMV,
        "merge_gemv.wgsl",
        2,
        &[("N", Scalar::U32), ("SEGS", Scalar::U32), ("ACC", Scalar::U32)]
    ),
    shader!(
        name::EMBED,
        "embed.wgsl",
        5,
        &[
            ("M", Scalar::U32),
            ("DIM", Scalar::U32),
            ("SCALE", Scalar::F32),
            ("WDTYPE", Scalar::U32),
            ("GROUP", Scalar::U32),
            ("SPLIT", Scalar::U32),
            ("ROWS", Scalar::U32),
        ]
    ),
    shader!(
        name::NORM,
        "norm.wgsl",
        7,
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
    shader!(name::ADD, "add.wgsl", 3, &[("N_ELEM", Scalar::U32)]),
    shader!(
        name::BIAS,
        "bias.wgsl",
        2,
        &[("N_ELEM", Scalar::U32), ("DIM", Scalar::U32)]
    ),
    shader!(
        name::CONCAT,
        "concat.wgsl",
        3,
        &[("ROWS", Scalar::U32), ("D", Scalar::U32)]
    ),
    shader!(
        name::SWIGLU,
        "swiglu.wgsl",
        3,
        &[("N_ELEM", Scalar::U32), ("MODE", Scalar::U32)]
    ),
    shader!(
        name::SOFTCAP,
        "softcap.wgsl",
        1,
        &[("N_ELEM", Scalar::U32), ("CAP", Scalar::F32)]
    ),
    shader!(
        name::MUL,
        "mul.wgsl",
        3,
        &[
            ("N", Scalar::U32),
            ("M", Scalar::U32),
            ("MODE", Scalar::U32),
            ("STRIDE", Scalar::U32),
            ("OFFSET", Scalar::U32),
        ]
    ),
    shader!(
        name::EXPERT_GATHER,
        "expert_gather.wgsl",
        3,
        &[("HIDDEN", Scalar::U32), ("COUNT", Scalar::U32)]
    ),
    shader!(
        name::EXPERT_SCATTER,
        "expert_scatter.wgsl",
        4,
        &[("HIDDEN", Scalar::U32), ("COUNT", Scalar::U32)]
    ),
    shader!(
        name::ZERO_ROWS,
        "zero_rows.wgsl",
        1,
        &[("N_ELEM", Scalar::U32)]
    ),
    shader!(
        name::SIGMOID_MUL,
        "sigmoid_mul.wgsl",
        3,
        &[("N_ELEM", Scalar::U32)]
    ),
    shader!(
        name::DELTA_GATE,
        "delta_gate.wgsl",
        6,
        &[("HEADS", Scalar::U32), ("ROW_T", Scalar::U32)]
    ),
    shader!(name::CONV1D, "conv1d.wgsl", 4, &[("DIM", Scalar::U32)]),
    shader!(
        name::DELTA_RECUR,
        "delta_recur.wgsl",
        7,
        &[
            ("HEADS", Scalar::U32),
            ("K_DIM", Scalar::U32),
            ("V_DIM", Scalar::U32),
        ]
    ),
    shader!(
        name::REPEAT_QK,
        "repeat_qk.wgsl",
        2,
        &[
            ("ROWS", Scalar::U32),
            ("N_K", Scalar::U32),
            ("N_V", Scalar::U32),
            ("K_DIM", Scalar::U32),
            ("RATIO", Scalar::U32),
            ("CONV_DIM", Scalar::U32),
        ]
    ),
    shader!(
        name::ROPE,
        "rope.wgsl",
        4,
        &[
            ("HEADS", Scalar::U32),
            ("HEAD_DIM", Scalar::U32),
            ("ROT", Scalar::U32),
            ("COS_STRIDE", Scalar::U32),
        ]
    ),
    shader!(
        name::ATTN,
        "attn.wgsl",
        5,
        &[
            ("N_HEADS", Scalar::U32),
            ("KV_HEADS", Scalar::U32),
            ("HEAD_DIM", Scalar::U32),
            ("MAX_SEQ", Scalar::U32),
            ("SCALE", Scalar::F32),
            ("WINDOW", Scalar::U32),
            ("NQ_PER_KV", Scalar::U32),
            ("STRIDE", Scalar::U32),
        ]
    ),
    shader!(
        name::MERGE_ATTN,
        "merge_attn.wgsl",
        3,
        &[
            ("N_HEADS", Scalar::U32),
            ("KV_HEADS", Scalar::U32),
            ("HEAD_DIM", Scalar::U32),
            ("STRIDE", Scalar::U32),
        ]
    ),
    shader!(
        name::KV_STORE,
        "kv_store.wgsl",
        5,
        &[
            ("N_KV", Scalar::U32),
            ("HEAD_DIM", Scalar::U32),
            ("MAX_SEQ", Scalar::U32),
        ]
    ),
    shader!(
        name::SPLIT_QG,
        "split_qg.wgsl",
        3,
        &[
            ("ROWS", Scalar::U32),
            ("HEADS", Scalar::U32),
            ("HD", Scalar::U32),
        ]
    ),
];

pub struct Kernels {
    kernels: HashMap<&'static str, Kernel>,
}

impl Kernels {
    pub fn new(device: &Device) -> Result<Self> {
        let mut kernels = HashMap::new();
        for spec in SHADERS {
            let layout = scalar_layout(spec.scalars)?;
            let kernel = device.create_kernel(&flint_gpu::KernelSpec {
                name: spec.name,
                wgsl: spec.wgsl,
                bindings: spec.bindings,
                immediate_size: layout.size,
            })?;
            kernels.insert(spec.name, kernel);
        }
        Ok(Self { kernels })
    }

    pub fn get(&self, name: &str) -> Result<&Kernel> {
        self.kernels
            .get(name)
            .ok_or_else(|| Error::Gpu(format!("unknown shader {name}")))
    }

    pub fn pack_scalars(&self, name: &str, consts: &[(&'static str, f64)]) -> Result<Vec<u8>> {
        let layout = SHADERS
            .iter()
            .find(|s| s.name == name)
            .map(|s| scalar_layout(s.scalars).expect("scalar layout is static"))
            .ok_or_else(|| Error::Gpu(format!("unknown shader {name}")))?;
        if consts.len() != layout.fields.len() {
            return Err(Error::Gpu(format!(
                "shader {name}: expected {} constants, got {}",
                layout.fields.len(),
                consts.len()
            )));
        }
        let mut bytes = vec![0u8; layout.size as usize];
        for field in &layout.fields {
            let value = consts
                .iter()
                .find(|(key, _)| *key == field.name)
                .ok_or_else(|| {
                    Error::Gpu(format!("shader {name}: missing constant {}", field.name))
                })?
                .1;
            let end = (field.offset + field.ty.width()) as usize;
            encode_scalar(&mut bytes[field.offset as usize..end], field.ty, value);
        }
        Ok(bytes)
    }
}

fn scalar_layout(scalars: &[(&'static str, Scalar)]) -> Result<ScalarLayout> {
    let mut fields = Vec::with_capacity(scalars.len());
    let mut offset = 0u32;
    for (name, ty) in scalars {
        fields.push(ScalarField {
            name: (*name).to_string(),
            offset,
            ty: *ty,
        });
        offset += ty.width();
    }
    Ok(ScalarLayout {
        size: offset,
        fields,
    })
}

fn encode_scalar(out: &mut [u8], ty: Scalar, value: f64) {
    out.copy_from_slice(&ty.encode(value)[..ty.width() as usize]);
}
