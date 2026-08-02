//! GGUF container reader: header, metadata KV, tensor-info table and on-demand
//! tensor dequantization. Format mechanics only — architecture-specific name
//! mapping and config synthesis live in `flint-archs`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use flint_error::{Error, Result};

use super::dequant::{self, GgmlType};
use super::{Checkpoint, RawTensor, TensorData};

const MAGIC: u32 = 0x4655_4747; // "GGUF", little-endian

/// One GGUF metadata value, normalized to a small typed set.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaVal {
    UInt(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<MetaVal>),
}

/// Typed accessors over the GGUF metadata KV table.
#[derive(Default)]
pub struct Metadata {
    kv: HashMap<String, MetaVal>,
}

impl Metadata {
    /// Builds a metadata table from raw KV pairs.
    pub fn new(kv: HashMap<String, MetaVal>) -> Self {
        Self { kv }
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

    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }
}

struct TensorInfo {
    ty: GgmlType,
    shape: Vec<u32>,
    /// Absolute byte offset of the tensor's data within the mmap.
    offset: usize,
    numel: usize,
}

/// A memory-mapped GGUF file.
pub struct Gguf {
    mmap: Mmap,
    meta: Metadata,
    tensors: HashMap<String, TensorInfo>,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Model(format!("cannot open {}: {e}", path.display())))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::Model(format!("mmap {}: {e}", path.display())))?;
        let mut r = Reader(&mmap, 0);

        let magic = r.u32()?;
        if magic != MAGIC {
            return Err(Error::Model(format!(
                "{:?} is not a GGUF file",
                path.file_name()
            )));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Model(format!("unsupported GGUF version {version}")));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut meta = Metadata::default();
        for _ in 0..kv_count {
            let key = r.string()?;
            let val = r.value()?;
            meta.kv.insert(key, val);
        }

        let alignment = meta.u64("general.alignment").unwrap_or(32) as usize;

        let mut infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            let mut numel = 1usize;
            for _ in 0..n_dims {
                let d = r.u64()? as usize;
                numel *= d;
                dims.push(d as u32);
            }
            let ty = GgmlType::from_u32(r.u32()?)?;
            let rel = r.u64()? as usize;
            infos.push((name, ty, dims, numel, rel));
        }

        // Tensor data starts at the next alignment boundary after the header.
        let data_start = align_up(r.1, alignment);

        let mut tensors = HashMap::with_capacity(tensor_count);
        for (name, ty, dims, numel, rel) in infos {
            let offset = data_start + rel;
            tensors.insert(
                name,
                TensorInfo {
                    ty,
                    shape: dims,
                    offset,
                    numel,
                },
            );
        }
        Ok(Self {
            mmap,
            meta,
            tensors,
        })
    }
}

impl Checkpoint for Gguf {
    fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn read(&self, name: &str) -> Result<RawTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Model(format!("gguf tensor {name:?} not found")))?;
        let need = info.numel.div_ceil(info.ty.block_len()) * info.ty.block_bytes();
        let end = info.offset + need;
        if end > self.mmap.len() {
            return Err(Error::Model(format!(
                "gguf tensor {name:?} data out of bounds ({}..{} of {})",
                info.offset,
                end,
                self.mmap.len()
            )));
        }
        let bytes = &self.mmap[info.offset..end];
        // ggml lists dims fastest-first ([K, N] for a weight); its memory is
        // row-major over that, i.e. [N, K] with K contiguous — exactly Flint's
        // weight convention. Reverse the dim list to report [N, K].
        let shape: Vec<u32> = info.shape.iter().rev().cloned().collect();
        let data = TensorData::F32(dequant::to_f32(info.ty, bytes, info.numel)?);
        Ok(RawTensor { shape, data })
    }

    fn metadata(&self) -> &Metadata {
        &self.meta
    }

    fn config_json(&self) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    fn kind(&self) -> &'static str {
        "gguf"
    }
}

/// Cursor over the mmap with little-endian reads.
struct Reader<'a>(&'a [u8], usize);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.1 + n > self.0.len() {
            return Err(Error::Model("GGUF header truncated".into()));
        }
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f64(&mut self) -> Result<f64> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let b = self.take(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::Model("GGUF string is not UTF-8".into()))
    }
    fn value(&mut self) -> Result<MetaVal> {
        let ty = self.u32()?;
        self.value_typed(ty)
    }

    /// Reads a value whose type tag was already consumed (array elements carry
    /// no per-element tag; the array header gives it once).
    fn value_typed(&mut self, ty: u32) -> Result<MetaVal> {
        Ok(match ty {
            0 => MetaVal::UInt(self.u8()? as u64),
            1 => MetaVal::Int(self.u8()? as i8 as i64),
            2 => MetaVal::UInt(self.u16()? as u64),
            3 => MetaVal::Int(self.u16()? as i16 as i64),
            4 => MetaVal::UInt(self.u32()? as u64),
            5 => MetaVal::Int(self.u32()? as i32 as i64),
            6 => MetaVal::Float(self.f32()? as f64),
            7 => MetaVal::Bool(self.u8()? != 0),
            8 => MetaVal::Str(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.value_typed(elem_ty)?);
                }
                MetaVal::Arr(items)
            }
            10 => MetaVal::UInt(self.u64()?),
            11 => MetaVal::Int(self.u64()? as i64),
            12 => MetaVal::Float(self.f64()?),
            t => {
                return Err(Error::Model(format!(
                    "unknown GGUF metadata value type {t}"
                )));
            }
        })
    }
}

fn align_up(v: usize, a: usize) -> usize {
    v + (a - v % a) % a
}
