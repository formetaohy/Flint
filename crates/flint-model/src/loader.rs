use std::collections::HashMap;

use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, RawTensor, TensorData};
use flint_error::{Error, Result};
use flint_tensor::{Tensor, Weight};
use saturn_core::num::f32_to_bf16;

use crate::quant::{choose_group, quantize, repack_q8};

pub enum Role {
    F32,
    Bf16,
    I8,
}

pub struct Plan {
    pub key: fn(&str) -> Option<String>,
    pub role: fn(&str) -> Role,
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
                    let (bytes, scales) = repack_q8(&bytes, n as usize, k as usize)?;
                    Ok(Weight::quant(
                        backend.tensor_i8(&bytes, shape),
                        backend.tensor_f32(&scales, vec![k / 32, n]),
                        32,
                    ))
                }
                other => {
                    let group = choose_group(k)?;
                    let data = other.into_f32();
                    let (bytes, scales) = quantize(&data, n as usize, k as usize, group as usize);
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
