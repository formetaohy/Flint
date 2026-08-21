use std::collections::HashMap;
use std::sync::Mutex;

use thuban_error::Result;

use crate::DeviceRef;
use crate::device::{BindingMode, KernelSpec};
use crate::encoder::BindingRef;

const BIND_CACHE_CAP: usize = 8192;
const MAX_BINDINGS: usize = 8;

#[derive(PartialEq, Eq, Hash)]
struct BindKey {
    entries: [Option<(wgpu::Buffer, u64, u64)>; MAX_BINDINGS],
    len: u32,
}

impl BindKey {
    fn new(bindings: &[BindingRef<'_>]) -> Self {
        assert!(
            bindings.len() <= MAX_BINDINGS,
            "kernel binding count exceeds the cache key capacity"
        );
        let mut entries = std::array::from_fn(|_| None);
        for (i, b) in bindings.iter().enumerate() {
            entries[i] = Some((b.buffer.buffer.clone(), b.offset, b.size));
        }
        Self {
            entries,
            len: bindings.len() as u32,
        }
    }
}

pub struct Kernel {
    pub(crate) name: String,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) binding_count: u32,
    device: DeviceRef,
    bind_groups: Mutex<HashMap<BindKey, wgpu::BindGroup>>,
}

impl Kernel {
    pub(crate) fn create(device: &DeviceRef, spec: &KernelSpec<'_>) -> Result<Self> {
        let module = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(spec.name),
                source: wgpu::ShaderSource::Wgsl(spec.wgsl.into()),
            });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = spec
            .bindings
            .iter()
            .enumerate()
            .map(|(i, mode)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: *mode == BindingMode::ReadOnly,
                    },
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
        let pipeline_layout =
            device
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
            binding_count: spec.bindings.len() as u32,
            device: device.clone(),
            bind_groups: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn bind_group(&self, bindings: &[BindingRef<'_>]) -> wgpu::BindGroup {
        let key = BindKey::new(bindings);
        let mut cache = self
            .bind_groups
            .lock()
            .expect("bind group cache lock poisoned");
        if let Some(group) = cache.get(&key) {
            return group.clone();
        }
        if cache.len() >= BIND_CACHE_CAP {
            cache.clear();
        }
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .map(|b| wgpu::BindGroupEntry {
                binding: b.index,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &b.buffer.buffer,
                    offset: b.offset,
                    size: (b.size > 0).then(|| {
                        wgpu::BufferSize::new(b.size)
                            .expect("binding size is validated by the caller")
                    }),
                }),
            })
            .collect();
        let group = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &entries,
            });
        cache.insert(key, group.clone());
        group
    }
}
