pub mod modes;
pub mod registry;
pub mod shader;

pub use modes::{Act, NormMode};
pub use registry::{SHADERS, ShaderSpec};

pub const ATTN_BR: u32 = 8;
pub const PAGE_LEN: u32 = 32;

use std::collections::HashMap;

use thuban_error::{Error, Result};
use thuban_gpu::{Device, Kernel};

pub mod scalar;

use scalar::{Scalar, ScalarField, ScalarLayout};
use shader::{GEMM_COOP, GEMM_COOP8};

type PackedScalars = HashMap<(String, Vec<u64>), Vec<u8>>;

fn coop_variant_of(name: &str) -> Option<thuban_gpu::CoopVariant> {
    match name {
        GEMM_COOP => Some(thuban_gpu::CoopVariant::M16),
        GEMM_COOP8 => Some(thuban_gpu::CoopVariant::M8),
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
            let kernel = device.create_kernel(&thuban_gpu::KernelSpec {
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
        let key = (
            name.to_string(),
            consts.iter().map(|(_, v)| v.to_bits()).collect(),
        );
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
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap_or_else(|e| panic!("shader {}: SPIR-V codegen failed: {e}", spec.name));
        }
    }
}
