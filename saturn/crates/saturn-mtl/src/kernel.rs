use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use saturn_core::error::{Error, Result};
use saturn_core::{Kernel, ScalarField, ScalarLayout};

use crate::device::{MtlDevice, compile_kernel};

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
        let (name, workgroup_size, scalars, buffer_count, msl, entry) = match spec.precompiled {
            Some(pc) => (
                pc.name.to_string(),
                pc.workgroup_size,
                pc.scalars
                    .iter()
                    .map(|f| ScalarField {
                        name: f.name.to_string(),
                        offset: f.offset,
                        ty: f.ty,
                    })
                    .collect(),
                pc.buffers,
                pc.msl.to_string(),
                pc.name.to_string(),
            ),
            None => {
                let source = saturn_compiler::Source::new(&spec.name, spec.source);
                let kernel = saturn_compiler::Driver::new()
                    .compile_with_specs(&source, spec.specs)
                    .map_err(|diags| {
                        Error::Shader(
                            diags
                                .iter()
                                .map(|d| source.render(d))
                                .collect::<Vec<_>>()
                                .join("\n"),
                        )
                    })?;
                let (msl, entry) = saturn_shader::to_msl(&kernel).map_err(Error::Shader)?;
                (
                    kernel.name.clone(),
                    kernel.workgroup_size,
                    kernel
                        .scalars
                        .iter()
                        .map(|p| ScalarField {
                            name: p.name.clone(),
                            offset: p.offset,
                            ty: p.ty,
                        })
                        .collect(),
                    kernel.params.iter().map(|p| p.binding).collect(),
                    kernel
                        .params
                        .iter()
                        .map(|p| p.binding)
                        .max()
                        .map_or(0, |max| max + 1),
                    msl,
                    entry,
                )
            }
        };
        let library = compile_kernel(&device.device, &msl)?;
        let function = library
            .newFunctionWithName(&NSString::from_str(&entry))
            .ok_or(Error::FunctionNotFound(entry.clone()))?;
        let pipeline = device
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Metal(e.to_string()))?;
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u64;
        let scalar_layout = if scalars.is_empty() {
            None
        } else {
            let total = scalars
                .iter()
                .map(|f| f.offset + f.ty.width())
                .max()
                .unwrap_or(0);
            Some(ScalarLayout {
                size: total.div_ceil(4) * 4,
                fields: scalars,
            })
        };
        log::debug!("metal: compiled kernel {name}");
        Ok(Box::new(Self {
            name,
            workgroup_size,
            scalar_layout,
            bindings,
            scalars_base,
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
