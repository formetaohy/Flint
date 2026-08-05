//! Pluggable checkpoint formats. A [`Checkpoint`] exposes native tensor
//! names, CPU-decoded tensor bytes and (for self-describing formats)
//! metadata; [`open`] picks one from a model directory.
//!
//! The checkpoint layer owns only container mechanics; tensor naming and
//! config synthesis live in `flint-architectures`.

pub mod dequant;
pub mod gguf;
mod safetensors;

use std::path::Path;

use flint_error::{Error, Result};

pub use gguf::{Gguf, GgufWriter, MetaVal, Metadata};
pub use safetensors::{Safetensors, write_tensors};

/// CPU-decoded tensor bytes, ready for the role-based GPU upload.
pub enum TensorData {
    F32(Vec<f32>),
    /// Little-endian bf16, two bytes per element.
    Bf16(Vec<u8>),
    /// Raw ggml Q8_0 blocks: per block a 2-byte f16 scale plus 32 i8.
    Q8 { bytes: Vec<u8>, numel: usize },
}

impl TensorData {
    /// Materializes the bytes as f32 regardless of storage.
    pub fn into_f32(self) -> Vec<f32> {
        match self {
            TensorData::F32(v) => v,
            TensorData::Bf16(b) => b
                .chunks_exact(2)
                .map(|c| flint_num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            TensorData::Q8 { bytes, numel } => {
                // Bytes are length-checked at checkpoint read; a Q8 stream is
                // always decodable, so a block error here is a hard invariant
                // violation, not a recoverable condition.
                dequant::to_f32(dequant::GgmlType::Q8_0, &bytes, numel)
                    .expect("Q8 block stream must be valid")
            }
        }
    }
}

/// One tensor's decoded contents plus its logical shape.
pub struct RawTensor {
    pub shape: Vec<u32>,
    pub data: TensorData,
}

/// A readable weight checkpoint in some format.
pub trait Checkpoint {
    /// Every native tensor name in the checkpoint.
    fn names(&self) -> Vec<String>;
    /// Decodes one tensor by its native name.
    fn read(&self, name: &str) -> Result<RawTensor>;
    /// Embedded metadata (populated for GGUF, empty otherwise).
    fn metadata(&self) -> &Metadata;
    /// config.json contents for directory formats; None for self-contained ones.
    fn config_json(&self) -> Result<Option<serde_json::Value>>;
    /// Format label for diagnostics and dispatch.
    fn kind(&self) -> &'static str;
}

/// Detects the checkpoint format in `model_dir` and opens it.
pub fn open(model_dir: &Path) -> Result<Box<dyn Checkpoint>> {
    if let Some(gguf) = find_gguf(model_dir)? {
        return Ok(Box::new(Gguf::open(&gguf)?));
    }
    Ok(Box::new(Safetensors::open(model_dir)?))
}

/// Locates a single `.gguf` file in the directory, failing fast on splits.
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
