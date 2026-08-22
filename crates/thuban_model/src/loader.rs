use std::collections::HashMap;

use thuban_backend::Backend;
use thuban_checkpoint::{Checkpoint, RawTensor, TensorData};
use thuban_error::{Error, Result};
use thuban_tensor::{Tensor, Weight};

pub enum Role {
    F32,
    Quant,
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
            Weight::Quantized(_) => Err(Error::Model(format!(
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
        Role::Quant => {
            if shape.len() != 2 {
                return Err(Error::Model(format!(
                    "{key}: quantized weight must be a [N, K] matrix, got {shape:?}"
                )));
            }
            let (n, k) = (shape[0], shape[1]);
            let numel = (n as usize) * (k as usize);
            match raw.data {
                TensorData::Quant { quant, bytes, numel } => {
                    if numel != (n as usize) * (k as usize) {
                        return Err(Error::Model(format!(
                            "{key}: quantized numel {numel} does not match shape {shape:?}"
                        )));
                    }
                    if !k.is_multiple_of(quant.block_len() as u32) {
                        return Err(Error::Model(format!(
                            "{key}: K={k} is not a multiple of the {quant:?} block length {}",
                            quant.block_len()
                        )));
                    }
                    let padded = quant.pad_blocks(&bytes, numel)?;
                    Ok(Weight::quantized(backend.tensor_quant(&padded, shape, quant)))
                }
                TensorData::F32(f) => Ok(Weight::plain(backend.tensor_f32(&f, shape))),
                TensorData::F16Bytes(b) => {
                    if numel * 2 != b.len() {
                        return Err(Error::Model(format!(
                            "{key}: f16 numel mismatch for shape {shape:?}"
                        )));
                    }
                    Ok(Weight::plain(backend.tensor_f16(&b, shape)?))
                }
                TensorData::Bf16Bytes(b) => Ok(Weight::plain(backend.tensor_bf16(&b, shape)?)),
            }
        }
    }
}
