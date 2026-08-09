use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::serialize;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use flint_error::{Error, Result};

use super::gguf::Metadata;
use super::{Checkpoint, CheckpointKind, RawTensor, TensorData};

pub struct Safetensors {
    dir: PathBuf,
    weight_map: HashMap<String, String>,
    meta: Metadata,
}

impl Safetensors {
    pub fn open(dir: &Path) -> Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let index: Index = match std::fs::read_to_string(&index_path) {
            Ok(raw) => serde_json::from_str(&raw).map_err(|e| {
                Error::Model(format!("invalid index {}: {e}", index_path.display()))
            })?,

            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let shard = dir.join("model.safetensors");
                if !shard.exists() {
                    return Err(Error::Model(format!(
                        "no safetensors index or model.safetensors in {}",
                        dir.display()
                    )));
                }
                let names = shard_tensor_names(&shard)?;
                let mut weight_map = HashMap::new();
                for n in names {
                    weight_map.insert(n, "model.safetensors".to_string());
                }
                Index { weight_map }
            }
            Err(e) => {
                return Err(Error::Model(format!(
                    "cannot read {}: {e}",
                    index_path.display()
                )))
            }
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            weight_map: index.weight_map,
            meta: Metadata::default(),
        })
    }
}

impl Checkpoint for Safetensors {
    fn names(&self) -> Vec<String> {
        self.weight_map.keys().cloned().collect()
    }

    fn read(&self, name: &str) -> Result<RawTensor> {
        let file_name = self
            .weight_map
            .get(name)
            .ok_or_else(|| Error::Model(format!("tensor {name:?} not in checkpoint index")))?;
        let file = File::open(self.dir.join(file_name))
            .map_err(|e| Error::Model(format!("missing shard {file_name}: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::Model(format!("mmap {file_name}: {e}")))?;
        let tables = SafeTensors::deserialize(&mmap)
            .map_err(|e| Error::Model(format!("parse {file_name}: {e}")))?;
        let view = tables
            .tensor(name)
            .map_err(|e| Error::Model(format!("tensor {name}: {e}")))?;

        let shape: Vec<u32> = view.shape().iter().map(|d| *d as u32).collect();
        let data = match view.dtype() {
            Dtype::F32 => TensorData::F32(f32_bytes(view.data()).to_vec()),
            Dtype::BF16 => TensorData::Bf16(view.data().to_vec()),
            Dtype::F16 => TensorData::F32(view.data().chunks_exact(2).map(f16).collect()),
            other => {
                return Err(Error::Model(format!(
                    "{name}: unsupported safetensors dtype {other:?}"
                )));
            }
        };
        Ok(RawTensor { shape, data })
    }

    fn metadata(&self) -> &Metadata {
        &self.meta
    }

    fn config_json(&self) -> Result<Option<serde_json::Value>> {
        let path = self.dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;
        Ok(Some(serde_json::from_str(&raw)?))
    }

    fn kind(&self) -> CheckpointKind {
        CheckpointKind::Safetensors
    }
}

#[derive(serde::Deserialize)]
struct Index {
    #[serde(rename = "weight_map")]
    weight_map: HashMap<String, String>,
}

fn shard_tensor_names(path: &Path) -> Result<Vec<String>> {
    let file =
        File::open(path).map_err(|e| Error::Model(format!("open {}: {e}", path.display())))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::Model(format!("mmap: {e}")))?;
    let tables =
        SafeTensors::deserialize(&mmap).map_err(|e| Error::Model(format!("parse: {e}")))?;
    Ok(tables.tensors().into_iter().map(|(n, _)| n).collect())
}

pub fn write_tensors(path: &Path, tensors: &[(String, Vec<u32>, Vec<u8>, bool)]) -> Result<()> {
    let mut views = Vec::with_capacity(tensors.len());
    for (name, shape, data, bf16) in tensors {
        let dtype = if *bf16 { Dtype::BF16 } else { Dtype::F32 };
        let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let view = TensorView::new(dtype, shape, data)
            .map_err(|e| Error::Model(format!("{name}: {e}")))?;
        views.push((name.clone(), view));
    }
    let bytes =
        serialize(views, None).map_err(|e| Error::Model(format!("serialize safetensors: {e}")))?;
    std::fs::write(path, bytes)
        .map_err(|e| Error::Model(format!("write {}: {e}", path.display())))?;
    Ok(())
}

fn f32_bytes(data: &[u8]) -> &[f32] {
    assert!(
        (data.as_ptr() as usize).is_multiple_of(4),
        "safetensors data is not f32-aligned"
    );
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }
}

fn f16(c: &[u8]) -> f32 {
    saturn_core::num::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))
}
