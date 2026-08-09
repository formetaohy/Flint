use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use saturn_core::error::{Error, Result};
use saturn_core::{Kernel, ScalarField, ScalarLayout};

use crate::device::{compile_kernel, MtlDevice};

pub struct MtlKernel {
    name: String,
    workgroup_size: [u32; 3],
    pub(crate) scalar_layout: Option<ScalarLayout>,
    pub(crate) buffer_count: usize,
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) max_threads: u64,
}

impl MtlKernel {
    pub fn create(device: &MtlDevice, spec: &saturn_core::KernelSpec) -> Result<Box<dyn Kernel>> {
        let source = saturn_compiler::Source::new(&spec.name, spec.source);
        let kernel = saturn_compiler::compile(&source).map_err(|diags| {
            Error::Shader(
                diags.iter()
                    .map(|d| source.render(d))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        })?;
        let (msl, entry) = saturn_shader::to_msl(&kernel).map_err(Error::Shader)?;
        let library = compile_kernel(&device.device, &msl)?;
        let function = library
            .newFunctionWithName(&NSString::from_str(&entry))
            .ok_or(Error::FunctionNotFound(entry.clone()))?;
        let pipeline = device
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Metal(e.to_string()))?;
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u64;
        let name = kernel.name.clone();
        let workgroup_size = kernel.workgroup_size;
        let scalar_layout = if kernel.scalars.is_empty() {
            None
        } else {
            let total = kernel
                .scalars
                .iter()
                .map(|p| p.offset + p.ty.width())
                .max()
                .unwrap_or(0);
            Some(ScalarLayout {
                size: total.div_ceil(4) * 4,
                fields: kernel
                    .scalars
                    .iter()
                    .map(|p| ScalarField {
                        name: p.name.clone(),
                        offset: p.offset,
                        ty: p.ty,
                    })
                    .collect(),
            })
        };
        let buffer_count = kernel.params.len();
        log::debug!("metal: compiled kernel {name}");
        Ok(Box::new(Self {
            name,
            workgroup_size,
            scalar_layout,
            buffer_count,
            pipeline,
            max_threads,
        }))
    }
}

impl Kernel for MtlKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    fn scalar_layout(&self) -> Option<&ScalarLayout> {
        self.scalar_layout.as_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
