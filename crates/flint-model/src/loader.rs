use std::collections::HashMap;

use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, RawTensor, TensorData};
use flint_error::{Error, Result};
use saturn_core::num::{f16_to_f32, f32_to_bf16};
use flint_tensor::{Tensor, Weight};

pub enum Role {

    F32,

    Bf16,

    I8,
}

pub struct Plan {

    pub key: fn(&str) -> Option<String>,

    pub role: fn(&str) -> Role,
}

pub fn choose_group(k: u32) -> Result<u32> {
    for g in [128u32, 64, 32] {
        if k.is_multiple_of(g) {
            return Ok(g);
        }
    }
    Err(Error::Config(format!(
        "dimension {k} is not a multiple of 32; cannot quantize"
    )))
}

pub fn repack_q8(bytes: &[u8], rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    assert!(cols.is_multiple_of(32), "Q8_0 K must be a multiple of 32");
    let groups = cols / 32;
    let expect = rows * groups * 34;
    if bytes.len() < expect {
        return Err(Error::Model(format!(
            "Q8_0 tensor truncated: need {expect} bytes, have {}",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; rows * cols];
    let mut scales = vec![0f32; rows * groups];
    for n in 0..rows {
        let row = &bytes[n * groups * 34..];
        for g in 0..groups {
            let blk = &row[g * 34..g * 34 + 34];
            scales[g * rows + n] = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for half in 0..2 {
                let kb = g * 2 + half;
                let dst = &mut out[(kb * rows + n) * 16..(kb * rows + n + 1) * 16];
                dst.copy_from_slice(&blk[2 + half * 16..2 + half * 16 + 16]);
            }
        }
    }
    Ok((out, scales))
}

pub fn quantize(data: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(
        cols.is_multiple_of(group),
        "quantized K must be a multiple of the group size"
    );
    assert!(
        cols.is_multiple_of(16),
        "quantized K must be a multiple of 16 (vec4 blocks)"
    );
    let groups = cols / group;
    let mut bytes = Vec::with_capacity(rows * cols);
    let mut scales = vec![0f32; rows * groups];
    for r in 0..rows {
        for g in 0..groups {
            let block = &data[r * cols + g * group..r * cols + (g + 1) * group];
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));

            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales[g * rows + r] = scale;
            for v in block {
                let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                bytes.push(q as u8);
            }
        }
    }

    let mut out = vec![0u8; rows * cols];
    for kb in 0..cols / 16 {
        for r in 0..rows {
            for i in 0..16 {
                out[(kb * rows + r) * 16 + i] = bytes[r * cols + kb * 16 + i];
            }
        }
    }
    (out, scales)
}

pub struct WeightSet {
    weights: HashMap<String, Weight>,
}

impl WeightSet {

    pub fn take(&mut self, key: &str) -> Result<Weight> {
        self.weights
            .remove(key)
            .ok_or_else(|| Error::Model(format!("checkpoint is missing weight {key:?}")))
    }

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

    pub fn insert(&mut self, key: String, w: Weight) {
        self.weights.insert(key, w);
    }
}

pub struct SwigluMlp {
    pub norm: Tensor,

    pub norm_bias: Option<Tensor>,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

pub fn take_mlp(w: &mut WeightSet, prefix: &str, layernorm: bool) -> Result<SwigluMlp> {
    let k = |n: &str| format!("{prefix}.{n}");
    Ok(SwigluMlp {
        norm: w.take_tensor(&k("post_attention_layernorm.weight"))?,
        norm_bias: if layernorm {
            Some(w.take_tensor(&k("post_attention_layernorm.bias"))?)
        } else {
            None
        },
        gate: w.take(&k("mlp.gate_proj.weight"))?,
        up: w.take(&k("mlp.up_proj.weight"))?,
        down: w.take(&k("mlp.down_proj.weight"))?,
    })
}

pub enum MlpBlock {
    Dense(Box<SwigluMlp>),
    Moe(Box<MoeMlp>),
}

impl MlpBlock {

    pub fn norm(&self) -> &Tensor {
        match self {
            MlpBlock::Dense(m) => &m.norm,
            MlpBlock::Moe(m) => &m.norm,
        }
    }

    pub fn norm_bias(&self) -> Option<&Tensor> {
        match self {
            MlpBlock::Dense(m) => m.norm_bias.as_ref(),
            MlpBlock::Moe(m) => m.norm_bias.as_ref(),
        }
    }
}

pub enum MoEPart {

    Router,

    GateUp,

    Gate,

    Up,

    Down,

    SharedGate,
    SharedUp,
    SharedDown,
}

pub struct MoEPlan {
    pub key: fn(&str) -> Option<(String, MoEPart)>,
    pub experts: u32,
    pub shared: bool,
}

pub fn load_moe_experts(
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
                out.push((key.clone(), upload(backend, &key, raw, role(&key))?));
            }
            MoEPart::GateUp => {
                out.extend(upload_experts(
                    backend,
                    &prefix,
                    &["gate_proj", "up_proj"],
                    raw,
                    plan.experts,
                    true,
                    role,
                )?);
            }
            MoEPart::Gate | MoEPart::Up | MoEPart::Down => {
                let part = match part {
                    MoEPart::Gate => "gate_proj",
                    MoEPart::Up => "up_proj",
                    _ => "down_proj",
                };
                if raw.shape.len() == 2 {

                    let key = format!("{prefix}.{part}.weight");
                    out.push((key.clone(), upload(backend, &key, raw, role(&key))?));
                } else {
                    out.extend(upload_experts(
                        backend,
                        &prefix,
                        &[part],
                        raw,
                        plan.experts,
                        false,
                        role,
                    )?);
                }
            }
            MoEPart::SharedGate | MoEPart::SharedUp | MoEPart::SharedDown => {
                let part = match part {
                    MoEPart::SharedGate => "shared_expert.gate_proj",
                    MoEPart::SharedUp => "shared_expert.up_proj",
                    _ => "shared_expert.down_proj",
                };
                let key = format!("{prefix}.{part}.weight");
                out.push((key.clone(), upload(backend, &key, raw, role(&key))?));
            }
        }
    }
    Ok(out)
}

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
            out.push((key.clone(), upload(backend, &key, raw, role(&key))?));
        }
    }
    Ok(out)
}

pub struct ExpertWeights {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

pub struct MoeMlp {
    pub norm: Tensor,
    pub norm_bias: Option<Tensor>,
    pub router: Weight,
    pub experts: Vec<ExpertWeights>,
    pub shared: Option<ExpertWeights>,
    pub top_k: u32,
    pub scale: f32,
    pub shared_scale: f32,
}

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
        exp.push(ExpertWeights {
            gate: w.take(&ek("gate_proj.weight"))?,
            up: w.take(&ek("up_proj.weight"))?,
            down: w.take(&ek("down_proj.weight"))?,
        });
    }
    let shared = if w.has(&k("mlp.shared_expert.gate_proj.weight")) {
        let sk = |n: &str| format!("{prefix}.mlp.shared_expert.{n}");
        Some(ExpertWeights {
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
        weights.insert(key.clone(), upload(backend, &key, raw, role)?);
    }
    Ok(WeightSet { weights })
}

pub fn upload(backend: &Backend, key: &str, raw: RawTensor, role: Role) -> Result<Weight> {
    let shape = raw.shape;
    match role {
        Role::F32 => {
            let data = raw.data.into_f32();
            Ok(Weight::plain(backend.tensor_f32(&data, shape)))
        }
        Role::Bf16 => {
            let bytes = match raw.data {
                TensorData::Bf16(b) => b,
                TensorData::F32(f) => f
                    .iter()
                    .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
                    .collect(),

                TensorData::Q8 { .. } => raw
                    .data
                    .into_f32()
                    .iter()
                    .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
                    .collect(),
            };
            Ok(Weight::plain(backend.tensor_bf16(&bytes, shape)?))
        }
        Role::I8 => {
            if shape.len() != 2 {
                return Err(Error::Model(format!(
                    "{key}: quantized weight must be a [N, K] matrix, got {shape:?}"
                )));
            }
            let (n, k) = (shape[0], shape[1]);
            match raw.data {

                TensorData::Q8 { bytes, numel } => {
                    if numel != (n * k) as usize {
                        return Err(Error::Model(format!(
                            "{key}: Q8_0 numel {numel} does not match shape {shape:?}"
                        )));
                    }
                    let (bytes, scales) =
                        repack_q8(&bytes, n as usize, k as usize)?;
                    Ok(Weight::quant(
                        backend.tensor_i8(&bytes, shape),
                        backend.tensor_f32(&scales, vec![k / 32, n]),
                        32,
                    ))
                }
                other => {
                    let group = choose_group(k)?;
                    let data = other.into_f32();
                    let (bytes, scales) =
                        quantize(&data, n as usize, k as usize, group as usize);
                    let (n, groups) = (n, k / group);
                    Ok(Weight::quant(
                        backend.tensor_i8(&bytes, shape),
                        backend.tensor_f32(&scales, vec![groups, n]),
                        group,
                    ))
                }
            }
        }
    }
}
