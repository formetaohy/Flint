use std::ffi::CString;
use std::sync::Arc;

use ash::vk;

use saturn_core::error::{Error, Result};
use saturn_core::{Kernel, ScalarField, ScalarLayout};

use crate::device::{VkDevice, VkDeviceInner};

fn check_coop_support(
    inner: std::sync::Arc<crate::device::VkDeviceInner>,
    kernel: &KernelMeta,
) -> Result<()> {
    use saturn_compiler::ir::MatrixRole;
    for (a, b, c) in &kernel.coop_triples {
        let shape = saturn_core::CoopShape {
            a: *a,
            b: *b,
            c: *c,
            m: 16,
            n: 16,
            k: 16,
        };
        if !inner.coop_shapes.contains(&shape) {
            return Err(Error::CoopUnsupported(format!(
                "A/B/C={:?}/{:?}/{:?} 16x16x16",
                shape.a, shape.b, shape.c
            )));
        }
    }
    for (elem, role) in &kernel.coop_roles {
        let ok = inner.coop_shapes.iter().any(|shape| match role {
            MatrixRole::A => shape.a == *elem,
            MatrixRole::B => shape.b == *elem,
            MatrixRole::Acc => shape.c == *elem,
        });
        if !ok {
            return Err(Error::CoopUnsupported(format!(
                "component {:?} for role {:?}",
                elem, role
            )));
        }
    }
    Ok(())
}

pub struct VkKernel {
    pub(crate) inner: Arc<VkDeviceInner>,
    name: String,
    workgroup_size: [u32; 3],
    pub(crate) scalar_layout: Option<ScalarLayout>,
    pub(crate) bindings: Vec<u32>,
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
}

struct KernelMeta {
    name: String,
    workgroup_size: [u32; 3],
    buffers: usize,
    scalars: Vec<saturn_compiler::ir::ScalarParam>,
    coop_triples: Vec<(saturn_core::Scalar, saturn_core::Scalar, saturn_core::Scalar)>,
    coop_roles: Vec<(saturn_core::Scalar, saturn_compiler::ir::MatrixRole)>,
    spirv: Vec<u8>,
}

fn meta_from_precompiled(pc: &saturn_core::PrecompiledKernel) -> KernelMeta {
    KernelMeta {
        name: pc.name.to_string(),
        workgroup_size: pc.workgroup_size,
        buffers: pc.buffers,
        scalars: pc
            .scalars
            .iter()
            .map(|f| saturn_compiler::ir::ScalarParam {
                name: f.name.to_string(),
                ty: f.ty,
                offset: f.offset,
            })
            .collect(),
        coop_triples: pc.coop_triples.to_vec(),
        coop_roles: pc
            .coop_roles
            .iter()
            .filter_map(|(elem, code)| {
                saturn_core::MatrixRole::decode(*code)
                    .map(|role| (*elem, role))
            })
            .collect(),
        spirv: pc.spirv.to_vec(),
    }
}

fn meta_from_compile(spec: &saturn_core::KernelSpec) -> Result<KernelMeta> {
    let source = saturn_compiler::Source::new(&spec.name, spec.source);
    let kernel = saturn_compiler::Driver::new()
        .compile_with_specs(&source, spec.specs)
        .map_err(|diags| {
            Error::Shader(
                diags.iter()
                    .map(|d| source.render(d))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        })?;
    let spirv = saturn_shader::to_spirv(&kernel).map_err(Error::Shader)?;
    Ok(KernelMeta {
        name: kernel.name.clone(),
        workgroup_size: kernel.workgroup_size,
        buffers: kernel.params.len(),
        scalars: kernel.scalars.clone(),
        coop_triples: kernel.coop_triples.clone(),
        coop_roles: kernel.coop_roles.clone(),
        spirv,
    })
}

impl VkKernel {
    pub fn create(device: &VkDevice, spec: &saturn_core::KernelSpec) -> Result<Box<dyn Kernel>> {
        let inner = device.inner.clone();
        let meta = match spec.precompiled {
            Some(pc) => meta_from_precompiled(pc),
            None => meta_from_compile(spec)?,
        };
        check_coop_support(inner.clone(), &meta)?;
        if meta.workgroup_size[0] > inner.max_workgroup_size[0]
            || meta.workgroup_size[1] > inner.max_workgroup_size[1]
            || meta.workgroup_size[2] > inner.max_workgroup_size[2]
        {
            return Err(Error::WorkgroupTooLarge {
                kernel: meta.name.clone(),
                size: meta
                    .workgroup_size
                    .iter()
                    .map(|&v| v as u64)
                    .product(),
                max: inner
                    .max_workgroup_size
                    .iter()
                    .map(|&v| v as u64)
                    .product(),
            });
        }
        let invocations: u64 = meta
            .workgroup_size
            .iter()
            .map(|&v| v as u64)
            .product();
        if invocations > inner.max_workgroup_invocations as u64 {
            return Err(Error::WorkgroupTooLarge {
                kernel: meta.name.clone(),
                size: invocations,
                max: inner.max_workgroup_invocations as u64,
            });
        }
        let scalar_layout = if meta.scalars.is_empty() {
            None
        } else {
            let total = meta
                .scalars
                .iter()
                .map(|p| p.offset + p.ty.width())
                .max()
                .unwrap_or(0);
            let size = total.div_ceil(4) * 4;
            if size > inner.max_push_constants_size {
                return Err(Error::Vulkan(format!(
                    "kernel {} push constants require {size} bytes, device max is {}",
                    meta.name, inner.max_push_constants_size
                )));
            }
            Some(ScalarLayout {
                size,
                fields: meta
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
        let code: Vec<u32> = meta
            .spirv
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let name = meta.name.clone();
        let workgroup_size = meta.workgroup_size;
        let bindings: Vec<u32> = (0..meta.buffers as u32).collect();
        let shader = unsafe {
            inner.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&code),
                None,
            )
        }
        .map_err(|e| Error::Vulkan(e.to_string()))?;

        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|&binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let layout = unsafe {
            inner.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings),
                None,
            )
        }
        .map_err(|e| Error::Vulkan(e.to_string()))?;

        let push_ranges = scalar_layout.as_ref().map(|layout| {
            vec![vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(layout.size)]
        });
        let layout_refs = [layout];
        let pipeline_layout = unsafe {
            let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layout_refs);
            if let Some(ranges) = &push_ranges {
                info = info.push_constant_ranges(ranges);
            }
            inner.device.create_pipeline_layout(&info, None)
        }
        .map_err(|e| Error::Vulkan(e.to_string()))?;

        let entry = CString::new(name.clone())
            .map_err(|_| Error::Vulkan("invalid kernel entry name".to_string()))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry);
        let pipeline_info =
            vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout);
        let pipeline = unsafe {
            inner
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, code)| Error::Vulkan(code.to_string()))?[0];

        log::debug!("vulkan: compiled kernel {name}");
        Ok(Box::new(Self {
            inner,
            name,
            workgroup_size,
            scalar_layout,
            bindings,
            layout,
            pipeline_layout,
            pipeline,
            shader,
        }))
    }
}

impl Kernel for VkKernel {
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

impl Drop for VkKernel {
    fn drop(&mut self) {
        unsafe {
            self.inner.device.destroy_pipeline(self.pipeline, None);
            self.inner
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.inner.device.destroy_shader_module(self.shader, None);
            self.inner
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
