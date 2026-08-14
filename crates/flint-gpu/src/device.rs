use flint_error::{Error, Result};

use crate::buffer::Buffer;
use crate::encoder::Encoder;
use crate::kernel::Kernel;
use crate::query::TimestampSet;
use crate::{DeviceInner, DeviceRef};

const MAX_IMMEDIATE_BYTES: u32 = 4096;
const MIN_IMMEDIATE_BYTES: u32 = 128;
const WORKGROUP_INVOCATIONS: u32 = 512;
const WORKGROUP_SIZE_X: u32 = 512;
const WORKGROUP_STORAGE_BYTES: u32 = 32 * 1024;
const STORAGE_BUFFERS_PER_STAGE: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAccess {
    None,
    Read,
    Write,
}

#[derive(Clone)]
pub struct Device {
    inner: DeviceRef,
    name: String,
    timestamps: bool,
    subgroup_min_size: u32,
    subgroup_max_size: u32,
    cooperative_matrix: Vec<wgpu::CooperativeMatrixProperties>,
}

impl Device {
    pub fn open() -> Result<Self> {
        let backends = if cfg!(any(target_os = "macos", target_os = "ios")) {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::VULKAN | wgpu::Backends::METAL
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            },
        ))
        .map_err(|e| {
            Error::Gpu(format!(
                "no wgpu adapter available for backends {backends:?}: {e}"
            ))
        })?;
        let info = adapter.get_info();
        let adapter_limits = adapter.limits();
        if adapter_limits.max_immediate_size < MIN_IMMEDIATE_BYTES {
            return Err(Error::Gpu(format!(
                "adapter immediate data size {} is below the required {MIN_IMMEDIATE_BYTES}",
                adapter_limits.max_immediate_size
            )));
        }
        let timestamps = adapter
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            && adapter
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let adapter_features = adapter.features();
        let mut features = wgpu::Features::IMMEDIATES;
        if timestamps {
            features |=
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        if adapter_features.contains(wgpu::Features::SUBGROUP) {
            features |= wgpu::Features::SUBGROUP;
        }
        let mut experimental = wgpu::ExperimentalFeatures::disabled();
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }
        if adapter_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
            experimental = unsafe { wgpu::ExperimentalFeatures::enabled() };
        }
        let limits = wgpu::Limits {
            max_immediate_size: adapter_limits
                .max_immediate_size
                .clamp(MIN_IMMEDIATE_BYTES, MAX_IMMEDIATE_BYTES),
            max_compute_invocations_per_workgroup: WORKGROUP_INVOCATIONS,
            max_compute_workgroup_size_x: WORKGROUP_SIZE_X,
            max_compute_workgroup_storage_size: WORKGROUP_STORAGE_BYTES,
            max_storage_buffers_per_shader_stage: STORAGE_BUFFERS_PER_STAGE,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            min_storage_buffer_offset_alignment: adapter_limits.min_storage_buffer_offset_alignment,
            ..wgpu::Limits::default()
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                experimental_features: experimental,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            }))
            .map_err(|e| Error::Gpu(format!("wgpu device request failed: {e}")))?;
        Ok(Self {
            inner: std::sync::Arc::new(DeviceInner { device, queue }),
            name: info.name,
            timestamps,
            subgroup_min_size: info.subgroup_min_size,
            subgroup_max_size: info.subgroup_max_size,
            cooperative_matrix: adapter.cooperative_matrix_properties(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn subgroup_min_size(&self) -> u32 {
        self.subgroup_min_size
    }

    pub fn subgroup_max_size(&self) -> u32 {
        self.subgroup_max_size
    }

    pub fn cooperative_matrix_properties(&self) -> &[wgpu::CooperativeMatrixProperties] {
        &self.cooperative_matrix
    }

    pub fn timestamp_period_ns(&self) -> f64 {
        self.inner.queue.get_timestamp_period() as f64
    }

    pub fn create_buffer(
        &self,
        size: u64,
        host_access: HostAccess,
        query_resolve: bool,
    ) -> Result<Buffer> {
        assert!(size > 0, "buffer size must be non-zero");
        let usage = match host_access {
            HostAccess::None => {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
            }
            HostAccess::Read => {
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
            }
            HostAccess::Write => {
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
            }
        } | if query_resolve {
            wgpu::BufferUsages::QUERY_RESOLVE
        } else {
            wgpu::BufferUsages::empty()
        };
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok(Buffer::new(
            buffer,
            self.inner.clone(),
            host_access != HostAccess::None,
        ))
    }

    pub fn create_kernel(&self, spec: &KernelSpec<'_>) -> Result<Kernel> {
        Kernel::create(&self.inner, spec)
    }

    pub fn create_timestamp_set(&self, capacity: u32) -> Result<TimestampSet> {
        if !self.timestamps {
            return Err(Error::Gpu(
                "timestamp queries not supported by this adapter".to_string(),
            ));
        }
        TimestampSet::create(&self.inner, capacity)
    }

    pub fn encoder(&self) -> Result<Encoder> {
        Encoder::new(self.inner.clone())
    }
}

pub struct KernelSpec<'a> {
    pub name: &'a str,
    pub wgsl: &'a str,
    pub bindings: u32,
    pub immediate_size: u32,
}
