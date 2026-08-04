//! Role-based weight loading shared by every format and architecture. The
//! checkpoint [`Checkpoint`] supplies decoded bytes; the [`Plan`] maps native
//! names to canonical keys and picks each key's GPU storage role; the upload
//! path re-packs each role (f32, packed bf16, group-quantized i8).
//! Architectures then resolve the uploaded set into typed weight structs — a
//! key missing from the checkpoint fails the load, never the forward.

use std::collections::HashMap;

use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, RawTensor, TensorData};
use flint_error::{Error, Result};
use flint_tensor::{Tensor, Weight};
/// Where a weight's bytes live and how kernels consume them.
pub enum Role {
    /// Decoded to f32 on the CPU (norms, biases, conv taps).
    F32,
    /// Kept as packed bf16 (embedding tables).
    Bf16,
    /// Quantized to per-group int8 on the CPU (projections).
    I8,
}

/// Architecture + format loading policy: how a checkpoint's native tensor names
/// map onto the canonical registry keys the forward graph reads, and how each
/// canonical key is stored on the GPU.
pub struct Plan {
    /// Maps a source-native tensor name to its canonical key, or None to skip.
    pub key: fn(&str) -> Option<String>,
    /// Storage role for a canonical key.
    pub role: fn(&str) -> Role,
}

/// Largest preferred quantization group that divides `k`. Group size trades
/// accuracy for scale storage; 128 is preferred, falling back for dimensions
/// (e.g. SmolLM's 960-wide hidden) that are not multiples of 128.
pub fn choose_group(k: u32) -> Result<usize> {
    for g in [128usize, 64, 32] {
        if (k as usize).is_multiple_of(g) {
            return Ok(g);
        }
    }
    Err(Error::Config(format!(
        "dimension {k} is not a multiple of 32; cannot quantize"
    )))
}

/// Row-wise group absmax quantization of f32 rows [rows, cols] with `group`
/// elements per scale.
pub fn quantize(data: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(
        cols.is_multiple_of(group),
        "quantized K must be a multiple of the group size"
    );
    let groups = cols / group;
    let mut bytes = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows * groups);
    for r in 0..rows {
        for g in 0..groups {
            let block = &data[r * cols + g * group..r * cols + (g + 1) * group];
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            // All-zero block: any scale dequantizes 0 to 0.
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales.push(scale);
            for v in block {
                let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                bytes.push(q as u8);
            }
        }
    }
    (bytes, scales)
}

/// Uploaded weights keyed by canonical name, consumed by typed take.
pub struct WeightSet {
    weights: HashMap<String, Weight>,
}

impl WeightSet {
    /// Takes a weight by canonical key; a missing key fails the load.
    pub fn take(&mut self, key: &str) -> Result<Weight> {
        self.weights
            .remove(key)
            .ok_or_else(|| Error::Model(format!("checkpoint is missing weight {key:?}")))
    }

    /// Takes an unquantized weight as a bare tensor.
    pub fn take_tensor(&mut self, key: &str) -> Result<Tensor> {
        match self.take(key)? {
            Weight::Plain(t) => Ok(t),
            Weight::Quantized { .. } => Err(Error::Model(format!(
                "{key:?} is quantized; expected a plain tensor"
            ))),
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.weights.contains_key(key)
    }

    /// Inserts a weight (MoE loading path).
    pub fn insert(&mut self, key: String, w: Weight) {
        self.weights.insert(key, w);
    }
}

/// SwiGLU MLP weights plus the norm that feeds it; the shared weight shape
/// used by every architecture's MLP block.
pub struct SwigluMlp {
    pub norm: Tensor,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

/// Takes an MLP's weights under `prefix` (e.g. `layers.0`).
pub fn take_mlp(w: &mut WeightSet, prefix: &str) -> Result<SwigluMlp> {
    let k = |n: &str| format!("{prefix}.{n}");
    Ok(SwigluMlp {
        norm: w.take_tensor(&k("post_attention_layernorm.weight"))?,
        gate: w.take(&k("mlp.gate_proj.weight"))?,
        up: w.take(&k("mlp.up_proj.weight"))?,
        down: w.take(&k("mlp.down_proj.weight"))?,
    })
}

/// A layer's feed-forward block: a dense gated MLP or a MoE block.
pub enum MlpBlock {
    Dense(SwigluMlp),
    Moe(MoeMlp),
}

impl MlpBlock {
    /// The norm feeding the block.
    pub fn norm(&self) -> &Tensor {
        match self {
            MlpBlock::Dense(m) => &m.norm,
            MlpBlock::Moe(m) => &m.norm,
        }
    }

    /// Layer-norm bias of the feeding norm, when the family uses one.
    pub fn norm_bias(&self) -> Option<&Tensor> {
        match self {
            MlpBlock::Dense(_) => None,
            MlpBlock::Moe(m) => m.norm_bias.as_ref(),
        }
    }
}

// ================================================================ MoE

/// One MoE weight's role in the expert set, mapped from a native tensor name
/// onto canonical keys under the block prefix (`layers.{l}.mlp`).
pub enum MoEPart {
    /// Router projection [E, hidden], uploaded as a dense weight.
    Router,
    /// Fused per-expert gate+up [E, 2N, K], split into gate/up halves.
    GateUp,
    /// Per-expert gate [E, N, K].
    Gate,
    /// Per-expert up [E, N, K].
    Up,
    /// Per-expert down [E, hidden, inter].
    Down,
    /// Dense shared expert (2D matrices, no expert axis).
    SharedGate,
    SharedUp,
    SharedDown,
}

/// MoE tensor classifier: maps each native name to its canonical block prefix
/// plus part, or None to skip. The expert count and shared-expert presence
/// come from the model config.
pub struct MoEPlan {
    pub key: fn(&str) -> Option<(String, MoEPart)>,
    pub experts: u32,
    pub shared: bool,
}

/// Loads every MoE weight the plan claims: the router as a dense weight, the
/// per-expert matrices split from their 3D tensors into individually
/// quantized weights, and the optional dense shared expert. Returns canonical
/// (key, weight) pairs for the caller's WeightSet.
pub fn load_moe(
    backend: &Backend,
    source: &dyn Checkpoint,
    plan: &MoEPlan,
    role: fn(&str) -> Role,
) -> Result<Vec<(String, Weight)>> {
    let mut names = source.names();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let Some((prefix, part)) = (plan.key)(&name) else {
            continue;
        };
        let raw = source.read(&name)?;
        match part {
            MoEPart::Router => {
                let key = format!("{prefix}.router.weight");
                out.push((key.clone(), upload(&key, raw, backend, role(&key))?));
            }
            MoEPart::GateUp => {
                out.extend(upload_experts(backend, &prefix, &["gate_proj", "up_proj"], raw, plan.experts, true, role)?);
            }
            MoEPart::Gate | MoEPart::Up | MoEPart::Down => {
                let part = match part {
                    MoEPart::Gate => "gate_proj",
                    MoEPart::Up => "up_proj",
                    _ => "down_proj",
                };
                if raw.shape.len() == 2 {
                    // Per-expert 2D tensors (Mixtral-style `block_sparse_moe`
                    // checkpoints): the prefix already carries the expert id.
                    let key = format!("{prefix}.{part}.weight");
                    out.push((key.clone(), upload(&key, raw, backend, role(&key))?));
                } else {
                    out.extend(upload_experts(backend, &prefix, &[part], raw, plan.experts, false, role)?);
                }
            }
            MoEPart::SharedGate | MoEPart::SharedUp | MoEPart::SharedDown => {
                let part = match part {
                    MoEPart::SharedGate => "shared_expert.gate_proj",
                    MoEPart::SharedUp => "shared_expert.up_proj",
                    _ => "shared_expert.down_proj",
                };
                let key = format!("{prefix}.{part}.weight");
                out.push((key.clone(), upload(&key, raw, backend, role(&key))?));
            }
        }
    }
    Ok(out)
}

/// Splits a 3D expert tensor along its leading axis and uploads each slice
/// as a quantized [rows, k] weight under `{prefix}.experts.{e}.{part}.weight`.
/// With `fused`, the middle axis is a concatenated gate+up [E, 2N, K] and
/// `parts` receives both halves per expert.
fn upload_experts(
    backend: &Backend,
    prefix: &str,
    parts: &[&str],
    raw: RawTensor,
    experts: u32,
    fused: bool,
    role: fn(&str) -> Role,
) -> Result<Vec<(String, Weight)>> {
    if raw.shape.len() != 3 || raw.shape[0] != experts {
        return Err(Error::Model(format!(
            "{prefix}.experts: expected a [{experts}, N, K] tensor, got {:?}",
            raw.shape
        )));
    }
    let (e_count, n, k) = (raw.shape[0], raw.shape[1], raw.shape[2]);
    if fused && !n.is_multiple_of(2) {
        return Err(Error::Model(format!(
            "{prefix}.experts: fused gate+up width {n} is odd"
        )));
    }
    let data = raw.data.into_f32();
    let mut out = Vec::with_capacity(e_count as usize * parts.len());
    for e in 0..e_count {
        for (i, part) in parts.iter().enumerate() {
            let (lo, hi) = if fused {
                (i as u32 * n / 2, (i as u32 + 1) * n / 2)
            } else {
                (0, n)
            };
            let rows = hi - lo;
            let key = format!("{prefix}.experts.{e}.{part}.weight");
            let slice: Vec<f32> = (lo..hi)
                .flat_map(|r| {
                    let base = ((e * n + r) * k) as usize;
                    data[base..base + k as usize].to_vec()
                })
                .collect();
            let raw = RawTensor {
                shape: vec![rows, k],
                data: TensorData::F32(slice),
            };
            out.push((key.clone(), upload(&key, raw, backend, role(&key))?));
        }
    }
    Ok(out)
}

/// One expert's gated MLP weights, per canonical key.
pub struct ExpertW {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

/// A MoE block's weights: the input norm, router and per-expert MLPs plus the
/// optional dense shared expert. Experts are individually quantized [N, K]
/// weights; the router stays a dense weight for routing precision.
pub struct MoeMlp {
    pub norm: Tensor,
    pub norm_bias: Option<Tensor>,
    pub router: Weight,
    pub experts: Vec<ExpertW>,
    pub shared: Option<ExpertW>,
    pub top_k: u32,
    pub scale: f32,
    pub shared_scale: f32,
}

/// Takes a MoE block's weights under `prefix` (e.g. `layers.0`): the input
/// norm (+ bias for layer-norm families), router, per-expert gate/up/down
/// and the optional shared expert.
pub fn take_moe(
    w: &mut WeightSet,
    prefix: &str,
    experts: u32,
    top_k: u32,
    scale: f32,
    shared_scale: f32,
    layernorm: bool,
) -> Result<MoeMlp> {
    let k = |n: &str| format!("{prefix}.{n}");
    let mut exp = Vec::with_capacity(experts as usize);
    for e in 0..experts {
        let ek = |n: &str| format!("{prefix}.mlp.experts.{e}.{n}");
        exp.push(ExpertW {
            gate: w.take(&ek("gate_proj.weight"))?,
            up: w.take(&ek("up_proj.weight"))?,
            down: w.take(&ek("down_proj.weight"))?,
        });
    }
    let shared = if w.has(&k("mlp.shared_expert.gate_proj.weight")) {
        let sk = |n: &str| format!("{prefix}.mlp.shared_expert.{n}");
        Some(ExpertW {
            gate: w.take(&sk("gate_proj.weight"))?,
            up: w.take(&sk("up_proj.weight"))?,
            down: w.take(&sk("down_proj.weight"))?,
        })
    } else {
        None
    };
    Ok(MoeMlp {
        norm: w.take_tensor(&k("post_attention_layernorm.weight"))?,
        norm_bias: if layernorm {
            Some(w.take_tensor(&k("post_attention_layernorm.bias"))?)
        } else {
            None
        },
        router: w.take(&k("mlp.router.weight"))?,
        experts: exp,
        shared,
        top_k,
        scale,
        shared_scale,
    })
}

/// Loads every checkpoint tensor the plan claims, onto the GPU by role.
pub fn load_weights(backend: &Backend, source: &dyn Checkpoint, plan: &Plan) -> Result<WeightSet> {
    let mut names = source.names();
    names.sort();
    let mut weights = HashMap::with_capacity(names.len());
    for name in names {
        let Some(key) = (plan.key)(&name) else {
            continue;
        };
        let raw = source.read(&name)?;
        let role = (plan.role)(&key);
        weights.insert(key.clone(), upload(&key, raw, backend, role)?);
    }
    Ok(WeightSet { weights })
}

/// Uploads a raw tensor under a canonical key with the plan's role (the
/// fused-QKV split path).
pub fn upload_key(
    backend: &Backend,
    key: &str,
    raw: RawTensor,
    role: Role,
) -> Result<Weight> {
    upload(key, raw, backend, role)
}

fn upload(key: &str, raw: RawTensor, backend: &Backend, role: Role) -> Result<Weight> {
    let label = format!("w:{key}");
    let shape = raw.shape;
    match role {
        Role::F32 => {
            let data = raw.data.into_f32();
            Ok(Weight::plain(backend.tensor_f32(&data, shape, &label)))
        }
        Role::Bf16 => {
            let bytes = match raw.data {
                TensorData::Bf16(b) => b,
                TensorData::F32(f) => f
                    .iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            };
            Ok(Weight::plain(backend.tensor_bf16(&bytes, shape, &label)?))
        }
        Role::I8 => {
            if shape.len() != 2 {
                return Err(Error::Model(format!(
                    "{key}: quantized weight must be a [N, K] matrix, got {shape:?}"
                )));
            }
            let group = choose_group(shape[1])?;
            let data = raw.data.into_f32();
            let (bytes, scales) = quantize(&data, shape[0] as usize, shape[1] as usize, group);
            let (n, groups) = (shape[0], shape[1] / group as u32);
            Ok(Weight::quant(
                backend.tensor_i8(&bytes, shape, &label),
                backend.tensor_f32(&scales, vec![n, groups], &format!("{label}.scale")),
                group as u32,
            ))
        }
    }
}
