use std::collections::HashMap;

use flint_backend::Backend;
use flint_checkpoint::{Checkpoint, RawTensor, TensorData};
use flint_error::{Error, Result};
use flint_num::f32_to_bf16;
use flint_tensor::{Tensor, Weight};

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

    pub fn take_if(&mut self, key: &str) -> Result<Option<Tensor>> {
        if self.weights.contains_key(key) {
            self.take_tensor(key).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.weights.contains_key(key)
    }

    pub fn insert(&mut self, key: String, w: Weight) {
        self.weights.insert(key, w);
    }
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
            let data = raw.data.into_f32()?;
            Ok(Weight::plain(backend.tensor_f32(&data, shape)))
        }
        Role::Bf16 => {
            let bytes = match raw.data {
                TensorData::Bf16Bytes(b) => b,
                TensorData::F32(f) => f
                    .iter()
                    .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
                    .collect(),
                TensorData::Q8_0 { .. } => raw
                    .data
                    .into_f32()?
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
                TensorData::Q8_0 { bytes, numel } => {
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
                    let data = other.into_f32()?;
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
