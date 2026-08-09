pub mod dequant;
pub mod gguf;
mod safetensors;

use std::path::Path;

use flint_error::{Error, Result};

pub use gguf::{Gguf, GgufWriter, MetaVal, Metadata};
pub use safetensors::{Safetensors, write_tensors};

pub enum TensorData {
    F32(Vec<f32>),

    Bf16(Vec<u8>),

    Q8 { bytes: Vec<u8>, numel: usize },
}

impl TensorData {

    pub fn into_f32(self) -> Vec<f32> {
        match self {
            TensorData::F32(v) => v,
            TensorData::Bf16(b) => b
                .chunks_exact(2)
                .map(|c| saturn_core::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            TensorData::Q8 { bytes, numel } => {

                dequant::to_f32(dequant::GgmlType::Q8_0, &bytes, numel)
                    .expect("Q8 block stream must be valid")
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
