pub mod cpu;

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
use saturn_sat::sat;

use flint_error::{Error, Result};

struct ShaderSpec {
    name: &'static str,
    source: &'static str,
}

const SHADERS: &[ShaderSpec] = &[
    ShaderSpec {
        name: name::GEMM,
        source: sat!("gemm.sat"),
    },
    ShaderSpec {
        name: name::MERGE_GEMM,
        source: sat!("merge_gemm.sat"),
    },
    ShaderSpec {
        name: name::GEMV,
        source: sat!("gemv.sat"),
    },
    ShaderSpec {
        name: name::MERGE_GEMV,
        source: sat!("merge_gemv.sat"),
    },
    ShaderSpec {
        name: name::GEMV_QKV,
        source: sat!("gemv_qkv.sat"),
    },
    ShaderSpec {
        name: name::MERGE_QKV,
        source: sat!("merge_qkv.sat"),
    },
    ShaderSpec {
        name: name::GEMV_GATEUP,
        source: sat!("gemv_gateup.sat"),
    },
    ShaderSpec {
        name: name::MERGE_GATEUP,
        source: sat!("merge_gateup.sat"),
    },
    ShaderSpec {
        name: name::EMBED,
        source: sat!("embed.sat"),
    },
    ShaderSpec {
        name: name::NORM,
        source: sat!("norm.sat"),
    },
    ShaderSpec {
        name: name::ADD,
        source: sat!("add.sat"),
    },
    ShaderSpec {
        name: name::BIAS,
        source: sat!("bias.sat"),
    },
    ShaderSpec {
        name: name::CONCAT,
        source: sat!("concat.sat"),
    },
    ShaderSpec {
        name: name::SWIGLU,
        source: sat!("swiglu.sat"),
    },
    ShaderSpec {
        name: name::SOFTCAP,
        source: sat!("softcap.sat"),
    },
    ShaderSpec {
        name: name::MUL,
        source: sat!("mul.sat"),
    },
    ShaderSpec {
        name: name::EXPERT_GATHER,
        source: sat!("expert_gather.sat"),
    },
    ShaderSpec {
        name: name::EXPERT_SCATTER,
        source: sat!("expert_scatter.sat"),
    },
    ShaderSpec {
        name: name::ZERO_ROWS,
        source: sat!("zero_rows.sat"),
    },
    ShaderSpec {
        name: name::SIGMOID_MUL,
        source: sat!("sigmoid_mul.sat"),
    },
    ShaderSpec {
        name: name::DELTA_GATE,
        source: sat!("delta_gate.sat"),
    },
    ShaderSpec {
        name: name::CONV1D,
        source: sat!("conv1d.sat"),
    },
    ShaderSpec {
        name: name::DELTA_RECUR,
        source: sat!("delta_recur.sat"),
    },
    ShaderSpec {
        name: name::REPEAT_QK,
        source: sat!("repeat_qk.sat"),
    },
    ShaderSpec {
        name: name::ROPE,
        source: sat!("rope.sat"),
    },
    ShaderSpec {
        name: name::ATTN,
        source: sat!("attn.sat"),
    },
    ShaderSpec {
        name: name::MERGE_ATTN,
        source: sat!("merge_attn.sat"),
    },
    ShaderSpec {
        name: name::KV_STORE,
        source: sat!("kv_store.sat"),
    },
    ShaderSpec {
        name: name::SPLIT_QG,
        source: sat!("split_qg.sat"),
    },
];

pub struct Kernels {
    kernels: HashMap<&'static str, Box<dyn Kernel>>,
}

impl Kernels {
    pub fn new(device: &dyn Device) -> Result<Self> {
        let mut kernels = HashMap::new();
        for spec in SHADERS {
            let kernel = device.create_kernel(&KernelSpec {
                name: format!("sat/{}", spec.name),
                source: spec.source,
            })?;
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
