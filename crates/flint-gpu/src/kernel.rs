use flint_error::Result;

use crate::device::KernelSpec;
use crate::DeviceRef;

pub struct Kernel {
    pub(crate) name: String,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) binding_count: u32,
}

impl Kernel {
    pub(crate) fn create(device: &DeviceRef, spec: &KernelSpec<'_>) -> Result<Self> {
        let module = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(spec.name),
                source: wgpu::ShaderSource::Wgsl(spec.wgsl.into()),
            });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..spec.bindings)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(spec.name),
                    entries: &entries,
                });
        let pipeline_layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(spec.name),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: spec.immediate_size,
            });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(spec.name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(spec.name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Ok(Self {
            name: spec.name.to_string(),
            pipeline,
            bind_group_layout,
            binding_count: spec.bindings,
        })
    }
}
