//! Compute kernels: the WGSL implementations, their compilation and
//! per-constants pipeline cache, and the CPU reference that defines the
//! intended semantics of every kernel. The WGPU kernels are tested against
//! the reference so the two cannot silently disagree.

pub mod cpu;

use std::collections::HashMap;

use wgpu::{
    BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline, Device,
    PipelineCompilationOptions, PipelineLayout, ShaderModule, ShaderSource,
};

use flint_error::{Error, Result};

/// One shader source plus the read-only flag of each of its bindings.
struct ShaderSpec {
    name: &'static str,
    source: &'static str,
    read_only: &'static [bool],
}

const SHADERS: &[ShaderSpec] = &[
    ShaderSpec {
        name: "gemm",
        source: include_str!("wgsl/gemm.wgsl"),
        read_only: &[true, true, true, false],
    },
    ShaderSpec {
        name: "gemv",
        source: include_str!("wgsl/gemv.wgsl"),
        read_only: &[true, true, true, false],
    },
    ShaderSpec {
        name: "embed",
        source: include_str!("wgsl/embed.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: "norm",
        source: include_str!("wgsl/norm.wgsl"),
        read_only: &[true, true, true, false],
    },
    ShaderSpec {
        name: "add",
        source: include_str!("wgsl/add.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: "bias",
        source: include_str!("wgsl/bias.wgsl"),
        read_only: &[false, true],
    },
    ShaderSpec {
        name: "concat",
        source: include_str!("wgsl/concat.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: "swiglu",
        source: include_str!("wgsl/swiglu.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: "sigmoid_mul",
        source: include_str!("wgsl/sigmoid_mul.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: "delta_gate",
        source: include_str!("wgsl/delta_gate.wgsl"),
        read_only: &[true, true, true, true, false, false],
    },
    ShaderSpec {
        name: "conv1d",
        source: include_str!("wgsl/conv1d.wgsl"),
        read_only: &[true, true, false, false],
    },
    ShaderSpec {
        name: "delta_recur",
        source: include_str!("wgsl/delta_recur.wgsl"),
        read_only: &[true, true, true, true, true, false, false],
    },
    ShaderSpec {
        name: "rope",
        source: include_str!("wgsl/rope.wgsl"),
        read_only: &[true, true, false, true],
    },
    ShaderSpec {
        name: "attn",
        source: include_str!("wgsl/attn.wgsl"),
        read_only: &[true, true, true, false, true],
    },
    ShaderSpec {
        name: "kv_store",
        source: include_str!("wgsl/kv_store.wgsl"),
        read_only: &[true, false, true],
    },
    ShaderSpec {
        name: "split_qg",
        source: include_str!("wgsl/split_qg.wgsl"),
        read_only: &[true, false, false],
    },
];

struct CompiledShader {
    module: ShaderModule,
    layout: PipelineLayout,
    bind_group_layout: BindGroupLayout,
}

/// Most constants any shader takes (attn: 7).
const MAX_CONSTS: usize = 8;

/// Allocation-free pipeline cache key: the shader plus its override constants
/// (name interned as a static str, value as raw f64 bits).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    shader: &'static str,
    len: u8,
    pairs: [(&'static str, u64); MAX_CONSTS],
}

fn pipeline_key(shader: &'static str, consts: &[(&'static str, f64)]) -> PipelineKey {
    assert!(
        consts.len() <= MAX_CONSTS,
        "{shader}: too many pipeline constants"
    );
    let mut pairs = [("", 0u64); MAX_CONSTS];
    for (i, (k, v)) in consts.iter().enumerate() {
        pairs[i] = (*k, v.to_bits());
    }
    PipelineKey {
        shader,
        len: consts.len() as u8,
        pairs,
    }
}

/// Lazily compiles shaders and caches one pipeline per (shader, constants)
/// pair. Bind groups are owned by the backend, which caches them per buffer
/// set; this type only manages pipelines and bind group layouts.
pub struct Kernels {
    shaders: HashMap<&'static str, CompiledShader>,
    pipelines: HashMap<PipelineKey, ComputePipeline>,
}

impl Kernels {
    pub fn new(device: &Device) -> Result<Self> {
        let mut shaders = HashMap::new();
        for spec in SHADERS {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(spec.name),
                source: ShaderSource::Wgsl(spec.source.into()),
            });
            let entries: Vec<BindGroupLayoutEntry> = spec
                .read_only
                .iter()
                .enumerate()
                .map(|(i, &ro)| BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: ro },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(spec.name),
                    entries: &entries,
                });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(spec.name),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
            shaders.insert(
                spec.name,
                CompiledShader {
                    module,
                    layout,
                    bind_group_layout,
                },
            );
        }
        Ok(Self {
            shaders,
            pipelines: HashMap::new(),
        })
    }

    /// The bind group layout for a shader; the backend builds and caches bind
    /// groups against it.
    pub fn bind_group_layout(&self, name: &str) -> Result<&BindGroupLayout> {
        Ok(&self
            .shaders
            .get(name)
            .ok_or_else(|| Error::Gpu(format!("unknown shader {name}")))?
            .bind_group_layout)
    }

    /// The pipeline for a (shader, constants) pair, compiled once and cached.
    /// Allocation happens only on a cache miss.
    pub fn pipeline(
        &mut self,
        device: &Device,
        name: &'static str,
        consts: &[(&'static str, f64)],
    ) -> Result<&ComputePipeline> {
        let key = pipeline_key(name, consts);
        if !self.pipelines.contains_key(&key) {
            let shader = self
                .shaders
                .get(name)
                .ok_or_else(|| Error::Gpu(format!("unknown shader {name}")))?;
            let pairs: Vec<(&str, f64)> = consts.to_vec();
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&shader.layout),
                module: &shader.module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions {
                    constants: &pairs,
                    ..Default::default()
                },
                cache: None,
            });
            self.pipelines.insert(key, pipeline);
        }
        Ok(self.pipelines.get(&key).expect("just inserted"))
    }
}
