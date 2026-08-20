pub mod modes;
pub const ATTN_BR: u32 = 8;
pub const PAGE_LEN: u32 = 32;
pub mod name {
    pub const GEMM: &str = "gemm";
    pub const GEMM_COOP: &str = "gemm_coop";
    pub const GEMM_COOP8: &str = "gemm_coop8";
    pub const TO_F16: &str = "to_f16";
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
    pub const KV_STORE: &str = "kv_store";
    pub const SPLIT_QG: &str = "split_qg";
}

use std::collections::HashMap;

use flint_error::{Error, Result};
use flint_gpu::{BindingMode, Device, Kernel};

mod scalar;

use scalar::{Scalar, ScalarField, ScalarLayout};

pub use modes::{Act, NormMode};

struct ShaderSpec {
    name: &'static str,
    wgsl: &'static str,
    bindings: &'static [BindingMode],
    scalars: &'static [(&'static str, Scalar)],
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

const SHADERS: &[ShaderSpec] = &[
    shader!(
        name::GEMM,
        ["gemm.wgsl"],
        &[BindingMode::ReadWrite; 4],
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
        name::GEMM_COOP,
        ["gemm_coop_common.wgsl", "gemm_coop.wgsl"],
        &[BindingMode::ReadOnly, BindingMode::ReadOnly, BindingMode::ReadOnly, BindingMode::ReadWrite],
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
        name::GEMM_COOP8,
        ["gemm_coop_common.wgsl", "gemm_coop8.wgsl"],
        &[BindingMode::ReadOnly, BindingMode::ReadOnly, BindingMode::ReadOnly, BindingMode::ReadWrite],
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
    shader!(name::TO_F16, ["to_f16.wgsl"], &[BindingMode::ReadOnly, BindingMode::ReadWrite], &[("N_ELEM", Scalar::U32)]),
    shader!(
        name::MERGE_GEMM,
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
        name::GEMV,
        ["gemv.wgsl"],
        &[BindingMode::ReadWrite; 4],
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
        ["merge_gemv.wgsl"],
        &[BindingMode::ReadWrite; 2],
        &[("N", Scalar::U32), ("SEGS", Scalar::U32), ("ACC", Scalar::U32)]
    ),
    shader!(
        name::EMBED,
        ["embed.wgsl"],
        &[BindingMode::ReadWrite; 5],
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
    shader!(name::ADD, ["add.wgsl"], &[BindingMode::ReadWrite; 3], &[("N_ELEM", Scalar::U32)]),
    shader!(
        name::BIAS,
        ["bias.wgsl"],
        &[BindingMode::ReadWrite; 2],
        &[("N_ELEM", Scalar::U32), ("DIM", Scalar::U32)]
    ),
    shader!(
        name::CONCAT,
        ["concat.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("ROWS", Scalar::U32), ("D", Scalar::U32)]
    ),
    shader!(
        name::SWIGLU,
        ["swiglu.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("N_ELEM", Scalar::U32), ("MODE", Scalar::U32)]
    ),
    shader!(
        name::SOFTCAP,
        ["softcap.wgsl"],
        &[BindingMode::ReadWrite; 1],
        &[("N_ELEM", Scalar::U32), ("CAP", Scalar::F32)]
    ),
    shader!(
        name::MUL,
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
        name::EXPERT_GATHER,
        ["expert_gather.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("HIDDEN", Scalar::U32), ("COUNT", Scalar::U32)]
    ),
    shader!(
        name::EXPERT_SCATTER,
        ["expert_scatter.wgsl"],
        &[BindingMode::ReadWrite; 4],
        &[("HIDDEN", Scalar::U32), ("COUNT", Scalar::U32)]
    ),
    shader!(
        name::ZERO_ROWS,
        ["zero_rows.wgsl"],
        &[BindingMode::ReadWrite; 1],
        &[("N_ELEM", Scalar::U32)]
    ),
    shader!(
        name::SIGMOID_MUL,
        ["sigmoid_mul.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[("N_ELEM", Scalar::U32)]
    ),
    shader!(
        name::DELTA_GATE,
        ["delta_gate.wgsl"],
        &[BindingMode::ReadWrite; 6],
        &[("HEADS", Scalar::U32), ("ROW_T", Scalar::U32)]
    ),
    shader!(name::CONV1D, ["conv1d.wgsl"], &[BindingMode::ReadWrite; 4], &[("DIM", Scalar::U32)]),
    shader!(
        name::DELTA_RECUR,
        ["delta_recur.wgsl"],
        &[BindingMode::ReadWrite; 7],
        &[
            ("HEADS", Scalar::U32),
            ("K_DIM", Scalar::U32),
            ("V_DIM", Scalar::U32),
        ]
    ),
    shader!(
        name::REPEAT_QK,
        ["repeat_qk.wgsl"],
        &[BindingMode::ReadWrite; 2],
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
        name::ATTN,
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
        name::KV_STORE,
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
        name::SPLIT_QG,
        ["split_qg.wgsl"],
        &[BindingMode::ReadWrite; 3],
        &[
            ("ROWS", Scalar::U32),
            ("HEADS", Scalar::U32),
            ("HD", Scalar::U32),
        ]
    ),
];

type PackedScalars = HashMap<(String, Vec<u64>), Vec<u8>>;

fn coop_variant_of(name: &str) -> Option<flint_gpu::CoopVariant> {
    match name {
        name::GEMM_COOP => Some(flint_gpu::CoopVariant::M16),
        name::GEMM_COOP8 => Some(flint_gpu::CoopVariant::M8),
        _ => None,
    }
}

pub struct Kernels {
    kernels: HashMap<&'static str, Kernel>,
    layouts: HashMap<&'static str, ScalarLayout>,
    packed: std::cell::RefCell<PackedScalars>,
}

impl Kernels {
    pub fn new(device: &Device) -> Result<Self> {
        let mut kernels = HashMap::new();
        let mut layouts = HashMap::new();
        for spec in SHADERS {
            if coop_variant_of(spec.name).is_some_and(|v| device.coop_gemm() != Some(v)) {
                continue;
            }
            let layout = scalar_layout(spec.scalars)?;
            let kernel = device.create_kernel(&flint_gpu::KernelSpec {
                name: spec.name,
                wgsl: spec.wgsl,
                bindings: spec.bindings,
                immediate_size: layout.size,
            })?;
            kernels.insert(spec.name, kernel);
            layouts.insert(spec.name, layout);
        }
        Ok(Self {
            kernels,
            layouts,
            packed: std::cell::RefCell::new(HashMap::new()),
        })
    }

    pub fn get(&self, name: &str) -> Result<&Kernel> {
        self.kernels
            .get(name)
            .ok_or_else(|| Error::Gpu(format!("unknown shader {name}")))
    }

    pub fn pack_scalars(&self, name: &str, consts: &[(&'static str, f64)]) -> Result<Vec<u8>> {
        let layout = self
            .layouts
            .get(name)
            .ok_or_else(|| Error::Gpu(format!("unknown shader {name}")))?;
        if consts.len() != layout.fields.len() {
            return Err(Error::Gpu(format!(
                "shader {name}: expected {} constants, got {}",
                layout.fields.len(),
                consts.len()
            )));
        }
        let key = (name.to_string(), consts.iter().map(|(_, v)| v.to_bits()).collect());
        if let Some(bytes) = self.packed.borrow().get(&key) {
            return Ok(bytes.clone());
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
        self.packed.borrow_mut().insert(key, bytes.clone());
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

#[cfg(test)]
mod compile_tests {
    #[test]
    fn every_shader_compiles_to_spirv() {
        for spec in super::SHADERS {
            let module = match naga::front::wgsl::parse_str(spec.wgsl) {
                Ok(m) => m,
                Err(e) => panic!("shader {}: WGSL parse failed: {e}", spec.name),
            };
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            let info = match validator.validate(&module) {
                Ok(i) => i,
                Err(e) => panic!("shader {}: validation failed: {e}", spec.name),
            };
            naga::back::spv::write_vec(
                &module,
                &info,
                &naga::back::spv::Options::default(),
                None,
            )
            .unwrap_or_else(|e| panic!("shader {}: SPIR-V codegen failed: {e}", spec.name));
        }
    }
}
