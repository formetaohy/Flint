use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use flint_error::{Error, Result};
use saturn_core::num::{f32_to_bf16, f32_to_f16};

use super::dequant::{self, GgmlType};
use super::{Checkpoint, CheckpointKind, RawTensor, TensorData};

const MAGIC: u32 = 0x4655_4747; 

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

struct TensorInfo {
    ty: GgmlType,
    shape: Vec<u32>,

    offset: usize,
    numel: usize,
}

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

        let shape: Vec<u32> = info.shape.iter().rev().cloned().collect();

        let data = match info.ty {
            GgmlType::Bf16 => TensorData::Bf16(bytes.to_vec()),
            GgmlType::Q8_0 => TensorData::Q8 {
                bytes: bytes.to_vec(),
                numel: info.numel,
            },
            _ => TensorData::F32(dequant::to_f32(info.ty, bytes, info.numel)?),
        };
        Ok(RawTensor { shape, data })
    }

    fn metadata(&self) -> &Metadata {
        &self.meta
    }

    fn config_json(&self) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    fn kind(&self) -> CheckpointKind {
        CheckpointKind::Gguf
    }
}

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

pub struct GgufWriter {
    kvs: Vec<(String, MetaVal)>,
    tensors: Vec<TensorSpec>,
    alignment: usize,
}

struct TensorSpec {
    name: String,
    ty: GgmlType,

    shape: Vec<u32>,
    data: Vec<u8>,
}

impl GgufWriter {
    pub fn new(alignment: u32) -> Self {
        Self {
            kvs: Vec::new(),
            tensors: Vec::new(),
            alignment: alignment as usize,
        }
    }

    pub fn kv_str(&mut self, key: &str, value: &str) {
        self.kvs
            .push((key.to_string(), MetaVal::Str(value.to_string())));
    }

    pub fn kv_u32(&mut self, key: &str, value: u32) {
        self.kvs
            .push((key.to_string(), MetaVal::UInt(value as u64)));
    }

    pub fn kv_f32(&mut self, key: &str, value: f32) {
        self.kvs
            .push((key.to_string(), MetaVal::Float(value as f64)));
    }

    pub fn kv_bool(&mut self, key: &str, value: bool) {
        self.kvs.push((key.to_string(), MetaVal::Bool(value)));
    }

    pub fn kv_str_array(&mut self, key: &str, value: &[impl AsRef<str>]) {
        self.kvs.push((
            key.to_string(),
            MetaVal::Arr(
                value
                    .iter()
                    .map(|s| MetaVal::Str(s.as_ref().to_string()))
                    .collect(),
            ),
        ));
    }

    pub fn kv_u32_array(&mut self, key: &str, value: &[u32]) {
        self.kvs.push((
            key.to_string(),
            MetaVal::Arr(value.iter().map(|v| MetaVal::UInt(*v as u64)).collect()),
        ));
    }

    pub fn kv_f64_array(&mut self, key: &str, value: &[f64]) {
        self.kvs.push((
            key.to_string(),
            MetaVal::Arr(value.iter().map(|v| MetaVal::Float(*v)).collect()),
        ));
    }

    pub fn tensor_f32(&mut self, name: &str, shape: &[u32], data: &[f32]) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.tensors.push(TensorSpec {
            name: name.to_string(),
            ty: GgmlType::F32,
            shape: shape.to_vec(),
            data: bytes,
        });
    }

    pub fn tensor_bf16(&mut self, name: &str, shape: &[u32], data: &[f32]) {
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for v in data {
            bytes.extend_from_slice(&f32_to_bf16(*v).to_le_bytes());
        }
        self.tensors.push(TensorSpec {
            name: name.to_string(),
            ty: GgmlType::Bf16,
            shape: shape.to_vec(),
            data: bytes,
        });
    }

    pub fn tensor_q8_0(&mut self, name: &str, shape: &[u32], data: &[f32]) {
        assert!(
            data.len().is_multiple_of(32),
            "Q8_0 requires a multiple of 32 elements, got {}",
            data.len()
        );
        let mut bytes = Vec::with_capacity(data.len() / 32 * 34);
        for block in data.chunks_exact(32) {
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            for v in block {
                bytes.push(((v / d).round().clamp(-127.0, 127.0) as i8) as u8);
            }
        }
        self.tensors.push(TensorSpec {
            name: name.to_string(),
            ty: GgmlType::Q8_0,
            shape: shape.to_vec(),
            data: bytes,
        });
    }

    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes()); 
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());

        for (key, val) in &self.kvs {
            write_str(&mut out, key);
            write_value(&mut out, val);
        }

        let header_end = out.len();
        let rec_len = |t: &TensorSpec| 8 + t.name.len() + 4 + t.shape.len() * 8 + 4 + 8;
        for t in &self.tensors {
            write_str(&mut out, &t.name);
            out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());

            for d in t.shape.iter().rev() {
                out.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            out.extend_from_slice(&(t.ty as u32).to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
        }

        let mut pos = align_up(out.len(), self.alignment);
        let mut cursor = 0usize;
        let mut rec_off = 0usize;
        for t in &self.tensors {
            cursor = align_up(cursor, self.alignment);
            let rel = cursor;
            cursor += t.data.len();
            let slot = header_end + rec_off + rec_len(t) - 8;
            out[slot..slot + 8].copy_from_slice(&(rel as u64).to_le_bytes());
            out.extend(std::iter::repeat_n(0u8, pos - out.len()));
            out.extend_from_slice(&t.data);
            pos = align_up(out.len(), self.alignment);
            rec_off += rec_len(t);
        }
        out
    }
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &MetaVal) {
    match v {
        MetaVal::UInt(n) => {
            assert!(
                *n <= u32::MAX as u64,
                "GGUF writer: value {n} does not fit u32"
            );
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&(*n as u32).to_le_bytes());
        }
        MetaVal::Int(i) => {
            assert!(
                i64::from(*i as i32) == *i,
                "GGUF writer: value {i} does not fit i32"
            );
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&(*i as i32).to_le_bytes());
        }
        MetaVal::Float(f) => {
            assert!(
                *f == (*f as f32) as f64,
                "GGUF writer: value {f} does not fit f32"
            );
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&(*f as f32).to_le_bytes());
        }
        MetaVal::Bool(b) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(*b as u8);
        }
        MetaVal::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_str(out, s);
        }
        MetaVal::Arr(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            let et = match items.first() {
                Some(MetaVal::UInt(_)) => 4u32,
                Some(MetaVal::Int(_)) => 5,
                Some(MetaVal::Float(_)) => 6,
                Some(MetaVal::Str(_)) => 8,
                other => panic!("cannot infer array element type from {other:?}"),
            };
            out.extend_from_slice(&et.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                write_value_typed(out, it);
            }
        }
    }
}

fn write_value_typed(out: &mut Vec<u8>, v: &MetaVal) {
    match v {
        MetaVal::UInt(n) => {
            assert!(
                *n <= u32::MAX as u64,
                "GGUF writer: value {n} does not fit u32"
            );
            out.extend_from_slice(&(*n as u32).to_le_bytes());
        }
        MetaVal::Int(i) => {
            assert!(
                i64::from(*i as i32) == *i,
                "GGUF writer: value {i} does not fit i32"
            );
            out.extend_from_slice(&(*i as i32).to_le_bytes());
        }
        MetaVal::Float(f) => {
            assert!(
                *f == (*f as f32) as f64,
                "GGUF writer: value {f} does not fit f32"
            );
            out.extend_from_slice(&(*f as f32).to_le_bytes());
        }
        MetaVal::Str(s) => write_str(out, s),
        other => panic!("unsupported array element {other:?}"),
    }
}
