pub mod cpu;
pub mod modes;

pub use modes::{Act, NormMode};

pub mod name {
    pub const GEMM: &str = "gemm";
    pub const MERGE_GEMM: &str = "merge_gemm";
    pub const GEMV: &str = "gemv";
    pub const MERGE_GEMV: &str = "merge_gemv";
    pub const GEMV_QKV: &str = "gemv_qkv";
    pub const MERGE_QKV: &str = "merge_qkv";
    pub const GEMV_GATEUP: &str = "gemv_gateup";
    pub const MERGE_GATEUP: &str = "merge_gateup";
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

use saturn_core::{Device, Kernel, KernelSpec, Scalar, ScalarLayout};
use saturn_scl::scl;

use flint_error::{Error, Result};

struct ShaderSpec {
    name: &'static str,
    kernel: &'static saturn_core::PrecompiledKernel<'static>,
}

const SHADERS: &[ShaderSpec] = &[
    ShaderSpec {
        name: name::GEMM,
        kernel: scl!("gemm.scl"),
    },
    ShaderSpec {
        name: name::MERGE_GEMM,
        kernel: scl!("merge_gemm.scl"),
    },
    ShaderSpec {
        name: name::GEMV,
        kernel: scl!("gemv.scl"),
    },
    ShaderSpec {
        name: name::MERGE_GEMV,
        kernel: scl!("merge_gemv.scl"),
    },
    ShaderSpec {
        name: name::GEMV_QKV,
        kernel: scl!("gemv_qkv.scl"),
    },
    ShaderSpec {
        name: name::MERGE_QKV,
        kernel: scl!("merge_qkv.scl"),
    },
    ShaderSpec {
        name: name::GEMV_GATEUP,
        kernel: scl!("gemv_gateup.scl"),
    },
    ShaderSpec {
        name: name::MERGE_GATEUP,
        kernel: scl!("merge_gateup.scl"),
    },
    ShaderSpec {
        name: name::EMBED,
        kernel: scl!("embed.scl"),
    },
    ShaderSpec {
        name: name::NORM,
        kernel: scl!("norm.scl"),
    },
    ShaderSpec {
        name: name::ADD,
        kernel: scl!("add.scl"),
    },
    ShaderSpec {
        name: name::BIAS,
        kernel: scl!("bias.scl"),
    },
    ShaderSpec {
        name: name::CONCAT,
        kernel: scl!("concat.scl"),
    },
    ShaderSpec {
        name: name::SWIGLU,
        kernel: scl!("swiglu.scl"),
    },
    ShaderSpec {
        name: name::SOFTCAP,
        kernel: scl!("softcap.scl"),
    },
    ShaderSpec {
        name: name::MUL,
        kernel: scl!("mul.scl"),
    },
    ShaderSpec {
        name: name::EXPERT_GATHER,
        kernel: scl!("expert_gather.scl"),
    },
    ShaderSpec {
        name: name::EXPERT_SCATTER,
        kernel: scl!("expert_scatter.scl"),
    },
    ShaderSpec {
        name: name::ZERO_ROWS,
        kernel: scl!("zero_rows.scl"),
    },
    ShaderSpec {
        name: name::SIGMOID_MUL,
        kernel: scl!("sigmoid_mul.scl"),
    },
    ShaderSpec {
        name: name::DELTA_GATE,
        kernel: scl!("delta_gate.scl"),
    },
    ShaderSpec {
        name: name::CONV1D,
        kernel: scl!("conv1d.scl"),
    },
    ShaderSpec {
        name: name::DELTA_RECUR,
        kernel: scl!("delta_recur.scl"),
    },
    ShaderSpec {
        name: name::REPEAT_QK,
        kernel: scl!("repeat_qk.scl"),
    },
    ShaderSpec {
        name: name::ROPE,
        kernel: scl!("rope.scl"),
    },
    ShaderSpec {
        name: name::ATTN,
        kernel: scl!("attn.scl"),
    },
    ShaderSpec {
        name: name::MERGE_ATTN,
        kernel: scl!("merge_attn.scl"),
    },
    ShaderSpec {
        name: name::KV_STORE,
        kernel: scl!("kv_store.scl"),
    },
    ShaderSpec {
        name: name::SPLIT_QG,
        kernel: scl!("split_qg.scl"),
    },
];

pub struct Kernels {
    kernels: HashMap<&'static str, Box<dyn Kernel>>,
}

impl Kernels {
    pub fn new(device: &dyn Device) -> Result<Self> {
        let mut kernels = HashMap::new();
        for spec in SHADERS {
            let kernel = device.create_kernel(&KernelSpec::precompiled(
                format!("scl/{}", spec.name),
                spec.kernel,
            ))?;
            assert_eq!(
                kernel.name(),
                spec.name,
                "kernel entry point mismatch"
            );
            kernels.insert(spec.name, kernel);
        }
        Ok(Self { kernels })
    }

    pub fn get(&self, name: &str) -> Result<&dyn Kernel> {
        Ok(self
            .kernels
            .get(name)
            .ok_or_else(|| Error::Gpu(format!("unknown shader {name}")))?
            .as_ref())
    }

    pub fn scalar_layout(&self, name: &str) -> Result<&ScalarLayout> {
        let kernel = self.get(name)?;
        kernel.scalar_layout().ok_or_else(|| {
            Error::Gpu(format!("shader {name} has no scalar parameters"))
        })
    }

    pub fn pack_scalars(
        &self,
        name: &str,
        consts: &[(&'static str, f64)],
    ) -> Result<Vec<u8>> {
        let layout = self.scalar_layout(name)?;
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

fn encode_scalar(out: &mut [u8], ty: Scalar, value: f64) {
    out.copy_from_slice(&ty.encode(value)[..ty.width() as usize]);
}
