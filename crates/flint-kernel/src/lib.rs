//! Compute kernels: the WGSL implementations, their compilation and
//! per-constants pipeline cache, and the CPU reference that defines the
//! intended semantics of every kernel. The WGPU kernels are tested against
//! the reference so the two cannot silently disagree.

pub mod cpu;

/// Canonical kernel names: the single source of truth for the string
/// protocol shared by the shader table, the dispatch facade and the
/// profiler labels.
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
        name: name::GEMM,
        source: include_str!("wgsl/gemm.wgsl"),
        read_only: &[true, true, true, false],
    },
    ShaderSpec {
        name: name::MERGE_GEMM,
        source: include_str!("wgsl/merge_gemm.wgsl"),
        read_only: &[true, false],
    },
    ShaderSpec {
        name: name::GEMV,
        source: include_str!("wgsl/gemv.wgsl"),
        read_only: &[true, true, true, false],
    },
    ShaderSpec {
        name: name::MERGE_GEMV,
        source: include_str!("wgsl/merge_gemv.wgsl"),
        read_only: &[true, false],
    },
    ShaderSpec {
        name: name::GEMV_QKV,
        source: include_str!("wgsl/gemv_qkv.wgsl"),
        read_only: &[
            true, true, true, false, true, true, false, true, true, false, false,
        ],
    },
    ShaderSpec {
        name: name::MERGE_QKV,
        source: include_str!("wgsl/merge_qkv.wgsl"),
        read_only: &[true, false, false, false],
    },
    ShaderSpec {
        name: name::GEMV_GATEUP,
        source: include_str!("wgsl/gemv_gateup.wgsl"),
        read_only: &[true, true, true, false, true, true, false, false],
    },
    ShaderSpec {
        name: name::MERGE_GATEUP,
        source: include_str!("wgsl/merge_gateup.wgsl"),
        read_only: &[true, false, false],
    },
    ShaderSpec {
        name: name::EMBED,
        source: include_str!("wgsl/embed.wgsl"),
        read_only: &[true, true, true, false],
    },
    ShaderSpec {
        name: name::NORM,
        source: include_str!("wgsl/norm.wgsl"),
        read_only: &[
            true, true, true, false, true, true, true, false, true, true, true,
        ],
    },
    ShaderSpec {
        name: name::ADD,
        source: include_str!("wgsl/add.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: name::BIAS,
        source: include_str!("wgsl/bias.wgsl"),
        read_only: &[false, true],
    },
    ShaderSpec {
        name: name::CONCAT,
        source: include_str!("wgsl/concat.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: name::SWIGLU,
        source: include_str!("wgsl/swiglu.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: name::SOFTCAP,
        source: include_str!("wgsl/softcap.wgsl"),
        read_only: &[false],
    },
    ShaderSpec {
        name: name::MUL,
        source: include_str!("wgsl/mul.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: name::EXPERT_GATHER,
        source: include_str!("wgsl/expert_gather.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: name::EXPERT_SCATTER,
        source: include_str!("wgsl/expert_scatter.wgsl"),
        read_only: &[false, true, true, true],
    },
    ShaderSpec {
        name: name::ZERO_ROWS,
        source: include_str!("wgsl/zero_rows.wgsl"),
        read_only: &[false],
    },
    ShaderSpec {
        name: name::SIGMOID_MUL,
        source: include_str!("wgsl/sigmoid_mul.wgsl"),
        read_only: &[true, true, false],
    },
    ShaderSpec {
        name: name::DELTA_GATE,
        source: include_str!("wgsl/delta_gate.wgsl"),
        read_only: &[true, true, true, true, false, false],
    },
    ShaderSpec {
        name: name::CONV1D,
        source: include_str!("wgsl/conv1d.wgsl"),
        read_only: &[true, true, false, false],
    },
    ShaderSpec {
        name: name::DELTA_RECUR,
        source: include_str!("wgsl/delta_recur.wgsl"),
        read_only: &[true, true, true, true, true, false, false],
    },
    ShaderSpec {
        name: name::REPEAT_QK,
        source: include_str!("wgsl/repeat_qk.wgsl"),
        read_only: &[true, false],
    },
    ShaderSpec {
        name: name::ROPE,
        source: include_str!("wgsl/rope.wgsl"),
        read_only: &[true, true, false, true],
    },
    ShaderSpec {
        name: name::ATTN,
        source: include_str!("wgsl/attn.wgsl"),
        read_only: &[true, true, true, false, true],
    },
    ShaderSpec {
        name: name::MERGE_ATTN,
        source: include_str!("wgsl/merge_attn.wgsl"),
        read_only: &[true, false, true],
    },
    ShaderSpec {
        name: name::KV_STORE,
        source: include_str!("wgsl/kv_store.wgsl"),
        read_only: &[true, true, false, false, true],
    },
    ShaderSpec {
        name: name::SPLIT_QG,
        source: include_str!("wgsl/split_qg.wgsl"),
        read_only: &[true, false, false],
    },
];

struct CompiledShader {
    module: ShaderModule,
    layout: PipelineLayout,
    bind_group_layout: BindGroupLayout,
}

/// Most constants any shader takes (gemm: 9).
const MAX_CONSTS: usize = 12;

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
