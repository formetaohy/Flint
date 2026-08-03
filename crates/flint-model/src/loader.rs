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
