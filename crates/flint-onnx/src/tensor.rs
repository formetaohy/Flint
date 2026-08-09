use std::path::Path;

use flint_error::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl Data {
    pub fn as_f32(&self) -> Vec<f32> {
        match self {
            Data::F32(v) => v.clone(),
            Data::I64(v) => v.iter().map(|&x| x as f32).collect(),
            Data::Bool(v) => v.iter().map(|&b| b as u8 as f32).collect(),
        }
    }

    pub fn as_i64(&self) -> Vec<i64> {
        match self {
            Data::I64(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x as i64).collect(),
            Data::Bool(v) => v.iter().map(|&b| b as i64).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Data,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: Data::F32(data),
            shape,
        }
    }

    pub fn i64(data: Vec<i64>, shape: Vec<usize>) -> Self {
        Self {
            data: Data::I64(data),
            shape,
        }
    }

    pub fn bool(data: Vec<bool>, shape: Vec<usize>) -> Self {
        Self {
            data: Data::Bool(data),
            shape,
        }
    }

    pub fn scalar_i64(v: i64) -> Self {
        Self::i64(vec![v], vec![])
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn broadcast_f32(&self, other: &Self) -> Result<(Vec<f32>, Vec<f32>)> {
        let shape = broadcast_shape(&self.shape, &other.shape)?;
        Ok((
            self.broadcast_to_f32(&shape)?,
            other.broadcast_to_f32(&shape)?,
        ))
    }

    pub fn broadcast_to_f32(&self, shape: &[usize]) -> Result<Vec<f32>> {
        let src = self.data.as_f32();
        Ok(broadcast_to(&src, &self.shape, shape)?)
    }

    pub fn from_proto(t: &crate::onnx::TensorProto, model_dir: &Path) -> Result<Tensor> {
        let shape: Vec<usize> = t.dims.iter().map(|&d| d.max(0) as usize).collect();
        let data = match t.data_type {
            1 => Data::F32(raw(t, model_dir)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()),
            10 => Data::F32(
                raw(t, model_dir)?
                    .chunks_exact(2)
                    .map(|c| saturn_core::num::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
            ),
            16 => Data::F32(
                raw(t, model_dir)?
                    .chunks_exact(2)
                    .map(|c| saturn_core::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
            ),
            11 => Data::F32(
                raw(t, model_dir)?
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
            ),
            2 | 3 | 4 | 5 | 6 | 7 | 9 | 12 | 13 => Data::I64(int_data(t, model_dir)?),
            other => {
                return Err(Error::Model(format!(
                    "unsupported tensor data_type {other}"
                )))
            }
        };
        let tensor = Tensor { data, shape };
        let n = tensor.numel();
        let got = tensor.len();
        if n != got {
            return Err(Error::Model(format!(
                "tensor {:?} shape {:?} expects {n} elements, has {got}",
                t.name, t.dims
            )));
        }
        Ok(tensor)
    }

    pub fn len(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::Bool(v) => v.len(),
        }
    }

    pub fn describe(&self, limit: usize) -> String {
        let dt = match self.data {
            Data::F32(_) => "f32",
            Data::I64(_) => "i64",
            Data::Bool(_) => "bool",
        };
        let head: Vec<String> = match &self.data {
            Data::F32(v) => v.iter().take(limit).map(|x| format!("{x:.6}")).collect(),
            Data::I64(v) => v.iter().take(limit).map(|x| x.to_string()).collect(),
            Data::Bool(v) => v.iter().take(limit).map(|b| b.to_string()).collect(),
        };
        let n = self.len();
        let more = if n > limit {
            format!(" … {} more", n - limit)
        } else {
            String::new()
        };
        format!("{dt} {:?} [{}]{}", self.shape, head.join(", "), more)
    }
}

pub fn broadcast_to<T: Copy>(data: &[T], src: &[usize], dst: &[usize]) -> Result<Vec<T>> {
    if src.is_empty() {
        return Ok(vec![data[0]; dst.iter().product()]);
    }
    if src == dst {
        return Ok(data.to_vec());
    }
    let n: usize = dst.iter().product();
    if data.is_empty() && n > 0 {
        return Err(Error::Model(format!(
            "cannot broadcast empty {src:?} to {dst:?}"
        )));
    }
    let off = dst.len() - src.len();
    for (i, d) in src.iter().enumerate() {
        if *d != 1 && *d != dst[off + i] {
            return Err(Error::Model(format!(
                "cannot broadcast {src:?} to {dst:?}"
            )));
        }
    }

    let mut src_strides = vec![0usize; src.len()];
    let mut acc = 1usize;
    for i in (0..src.len()).rev() {
        if src[i] != 1 {
            src_strides[i] = acc;
        }
        acc *= src[i];
    }
    let mut out = vec![data[0]; n];
    for (flat, o) in out.iter_mut().enumerate() {
        let mut idx = flat;
        let mut si = 0usize;
        for j in (0..dst.len()).rev() {
            let i = idx % dst[j];
            idx /= dst[j];
            if j >= off && src[j - off] != 1 {
                si += i * src_strides[j - off];
            }
        }
        *o = data[si];
    }
    Ok(out)
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let da = if i + a.len() >= n {
            a[i + a.len() - n]
        } else {
            1
        };
        let db = if i + b.len() >= n {
            b[i + b.len() - n]
        } else {
            1
        };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(Error::Model(format!(
                "broadcast mismatch: {a:?} vs {b:?}"
            )));
        };
    }
    Ok(out)
}

fn external_value(t: &crate::onnx::TensorProto, key: &str) -> Option<String> {
    t.external_data
        .iter()
        .find(|e| e.key == key)
        .map(|e| e.value.clone())
}

fn raw(t: &crate::onnx::TensorProto, model_dir: &Path) -> Result<Vec<u8>> {
    if !t.raw_data.is_empty() {
        return Ok(t.raw_data.clone());
    }
    if !t.float_data.is_empty() {
        return Ok(t
            .float_data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect());
    }
    external_bytes(t, model_dir)
}

fn external_bytes(t: &crate::onnx::TensorProto, model_dir: &Path) -> Result<Vec<u8>> {
    let Some(loc) = external_value(t, "location") else {
        return Err(Error::Model(format!(
            "tensor {:?} has no raw or typed data",
            t.name
        )));
    };
    let path = model_dir.join(loc);
    let offset: u64 = external_value(t, "offset")
        .map(|s| s.parse().unwrap_or(0))
        .unwrap_or(0);
    let length: Option<u64> = external_value(t, "length").map(|s| s.parse().ok()).flatten();
    let bytes = std::fs::read(&path)
        .map_err(|e| Error::Model(format!("external data {}: {e}", path.display())))?;
    let end = length
        .map(|l| offset + l)
        .unwrap_or(bytes.len() as u64);
    if end > bytes.len() as u64 {
        return Err(Error::Model(format!(
            "external data {}: requested range {offset}..{end} exceeds file",
            path.display()
        )));
    }
    Ok(bytes[offset as usize..end as usize].to_vec())
}

fn int_data(t: &crate::onnx::TensorProto, model_dir: &Path) -> Result<Vec<i64>> {
    if !t.raw_data.is_empty() {
        return Ok(match t.data_type {
            2 | 9 => t.raw_data.iter().map(|&b| b as i64).collect(),        
            3 => t.raw_data.iter().map(|&b| (b as i8) as i64).collect(),    
            4 => t
                .raw_data
                .chunks_exact(2)
                .map(|c| i64::from(u16::from_le_bytes([c[0], c[1]])))
                .collect(),                                                  
            5 => t
                .raw_data
                .chunks_exact(2)
                .map(|c| i64::from(i16::from_le_bytes([c[0], c[1]])))
                .collect(),                                                  
            6 => t
                .raw_data
                .chunks_exact(4)
                .map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap())))
                .collect(),                                                  
            12 => t
                .raw_data
                .chunks_exact(4)
                .map(|c| i64::from(u32::from_le_bytes(c.try_into().unwrap())))
                .collect(),                                                  
            7 => t
                .raw_data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),                                                  
            13 => t
                .raw_data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),                                                  
            _ => return Err(Error::Model("unsupported integer type".into())),
        });
    }
    if !t.int32_data.is_empty() {
        return Ok(t.int32_data.iter().map(|&x| x as i64).collect());
    }
    if !t.int64_data.is_empty() {
        return Ok(t.int64_data.clone());
    }
    external_bytes(t, model_dir).map(|b| {
        b.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    })
}
