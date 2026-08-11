pub mod dequant;
pub mod gguf;
mod safetensors;

use std::collections::HashMap;
use std::path::Path;

use flint_error::{Error, Result};

pub use gguf::Gguf;
pub use gguf::GgufWriter;
pub use safetensors::{Safetensors, write_tensors};

#[derive(Clone, Debug, PartialEq)]
pub enum MetaVal {
    UInt(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<MetaVal>),
}

#[derive(Default)]
pub struct Metadata {
    kv: HashMap<String, MetaVal>,
}

impl Metadata {
    pub fn new(kv: HashMap<String, MetaVal>) -> Self {
        Self { kv }
    }

    pub fn insert(&mut self, key: String, value: MetaVal) {
        self.kv.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&MetaVal> {
        self.kv.get(key)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        match self.kv.get(key) {
            Some(MetaVal::Str(s)) => Some(s),
            _ => None,
        }
    }

    pub fn u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key) {
            Some(MetaVal::UInt(v)) => Some(*v),
            Some(MetaVal::Int(v)) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn u32(&self, key: &str) -> Option<u32> {
        self.u64(key).and_then(|v| u32::try_from(v).ok())
    }

    pub fn f64(&self, key: &str) -> Option<f64> {
        match self.kv.get(key) {
            Some(MetaVal::Float(v)) => Some(*v),
            Some(MetaVal::UInt(v)) => Some(*v as f64),
            Some(MetaVal::Int(v)) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn str_array(&self, key: &str) -> Option<Vec<&str>> {
        match self.kv.get(key) {
            Some(MetaVal::Arr(items)) => items
                .iter()
                .map(|v| match v {
                    MetaVal::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    pub fn u32_array(&self, key: &str) -> Option<Vec<u32>> {
        match self.kv.get(key) {
            Some(MetaVal::Arr(items)) => items
                .iter()
                .map(|v| match v {
                    MetaVal::UInt(n) => u32::try_from(*n).ok(),
                    MetaVal::Int(n) => u32::try_from(*n).ok(),
                    MetaVal::Bool(b) => Some(*b as u32),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    pub fn f64_array(&self, key: &str) -> Option<Vec<f64>> {
        match self.kv.get(key) {
            Some(MetaVal::Arr(items)) => items
                .iter()
                .map(|v| match v {
                    MetaVal::Float(f) => Some(*f),
                    MetaVal::UInt(n) => Some(*n as f64),
                    MetaVal::Int(n) => Some(*n as f64),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }
}

pub enum TensorData {
    F32(Vec<f32>),

    Bf16Bytes(Vec<u8>),

    Q8Blocks { bytes: Vec<u8>, numel: usize },
}

impl TensorData {
    pub fn into_f32(self) -> Result<Vec<f32>> {
        match self {
            TensorData::F32(v) => Ok(v),
            TensorData::Bf16Bytes(b) => Ok(b
                .chunks_exact(2)
                .map(|c| saturn_core::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            TensorData::Q8Blocks { bytes, numel } => {
                dequant::to_f32(dequant::GgmlType::Q8_0, &bytes, numel)
            }
        }
    }
}

pub struct RawTensor {
    pub shape: Vec<u32>,
    pub data: TensorData,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CheckpointKind {
    Gguf,
    Safetensors,
}

pub trait Checkpoint {
    fn names(&self) -> Vec<String>;
    fn read(&self, name: &str) -> Result<RawTensor>;
    fn metadata(&self) -> &Metadata;
    fn config_json(&self) -> Result<Option<serde_json::Value>>;
    fn kind(&self) -> CheckpointKind;
}

pub fn open(model_dir: &Path) -> Result<Box<dyn Checkpoint>> {
    if let Some(gguf) = find_gguf(model_dir)? {
        return Ok(Box::new(Gguf::open(&gguf)?));
    }
    Ok(Box::new(Safetensors::open(model_dir)?))
}

fn find_gguf(model_dir: &Path) -> Result<Option<std::path::PathBuf>> {
    let mut found: Vec<_> = std::fs::read_dir(model_dir)
        .map_err(|e| Error::Model(format!("cannot read {}: {e}", model_dir.display())))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .collect();
    match found.len() {
        0 => Ok(None),
        1 => Ok(Some(found.remove(0))),
        _ => Err(Error::Model(format!(
            "multiple .gguf shards in {} — merge into one file",
            model_dir.display()
        ))),
    }
}
