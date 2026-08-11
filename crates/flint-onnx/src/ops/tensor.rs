use std::collections::HashMap;

use flint_error::{Error, Result};

use crate::graph::Node;
use crate::ops::{f32s, i64s, input, input_opt, norm_axis, output};
use crate::tensor::{Data, Tensor, broadcast_shape, broadcast_to};

pub(crate) fn shape(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let out: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
    output(env, node, 0, Tensor::i64(out, vec![x.shape.len()]))?;
    Ok(())
}

pub(crate) fn size(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    output(env, node, 0, Tensor::scalar_i64(x.numel() as i64))?;
    Ok(())
}

pub(crate) fn reshape(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let target = i64s(input(env, node, 1)?)?;
    let n = x.numel() as i64;
    let mut shape = vec![0i64; target.len()];
    let mut inferred: Option<usize> = None;
    let mut known = 1i64;
    for (i, &d) in target.iter().enumerate() {
        match d {
            0 => {
                shape[i] = x.shape.get(i).copied().unwrap_or(1) as i64;
                known *= shape[i];
            }
            -1 => {
                if inferred.is_some() {
                    return Err(Error::Model("Reshape: multiple -1 dims".into()));
                }
                inferred = Some(i);
            }
            d if d > 0 => {
                shape[i] = d;
                known *= d;
            }
            _ => return Err(Error::Model(format!("Reshape: invalid dim {d}"))),
        }
    }
    if let Some(i) = inferred {
        if known == 0 || n % known != 0 {
            return Err(Error::Model(format!(
                "Reshape: {n} elements cannot fill inferred dim (known {known})"
            )));
        }
        shape[i] = n / known;
    } else if known != n {
        return Err(Error::Model(format!(
            "Reshape: {n} elements do not match shape {target:?}"
        )));
    }
    let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let t = Tensor {
        data: x.data.clone(),
        shape,
    };
    output(env, node, 0, t)?;
    Ok(())
}

pub(crate) fn transpose(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let perm: Vec<usize> = match node.attr("perm") {
        Some(a) => a
            .ints()?
            .iter()
            .map(|&p| norm_axis(p, rank))
            .collect::<Result<_>>()?,
        None => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        return Err(Error::Model(format!(
            "Transpose: perm {:?} length != rank {rank}",
            perm
        )));
    }
    let shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    let data = match &x.data {
        Data::F32(v) => Data::F32(permute(v, &x.shape, &perm)),
        Data::I64(v) => Data::I64(permute(v, &x.shape, &perm)),
        Data::Bool(v) => Data::Bool(permute(v, &x.shape, &perm)),
    };
    output(env, node, 0, Tensor { data, shape })?;
    Ok(())
}

fn permute<T: Copy>(v: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
    let rank = shape.len();
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let n: usize = out_shape.iter().product();
    let mut out = vec![v[0]; n];

    let mut idx = vec![0usize; rank];
    let mut src_coords = vec![0usize; rank];
    for (flat, o) in out.iter_mut().enumerate() {
        let mut rem = flat;
        for k in (0..rank).rev() {
            idx[k] = rem % out_shape[k];
            rem /= out_shape[k];
        }
        for (k, &p) in perm.iter().enumerate() {
            src_coords[p] = idx[k];
        }
        let mut src = 0usize;
        for d in 0..rank {
            src = src * shape[d] + src_coords[d];
        }
        *o = v[src];
    }
    out
}

pub(crate) fn concat(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let first = input(env, node, 0)?;
    let rank = first.shape.len();
    let axis = norm_axis(node.need("axis")?.i()?, rank)?;
    let mut shape = first.shape.clone();
    let mut datas: Vec<(Data, Vec<usize>)> = Vec::with_capacity(node.inputs.len());
    for i in 0..node.inputs.len() {
        let t = input(env, node, i)?;
        if t.shape.len() != rank {
            return Err(Error::Model(format!(
                "Concat: input {i} rank {} != {rank}",
                t.shape.len()
            )));
        }
        for (d, dd) in t.shape.iter().enumerate() {
            if d != axis && *dd != shape[d] {
                return Err(Error::Model(format!(
                    "Concat: dim {d} mismatch {:?} vs {:?}",
                    shape, t.shape
                )));
            }
        }
        if i > 0 {
            shape[axis] += t.shape[axis];
        }
        datas.push((t.data.clone(), t.shape.clone()));
    }
    let kind = match datas[0].0 {
        Data::F32(_) => 0,
        Data::I64(_) => 1,
        Data::Bool(_) => 2,
    };
    let inner: usize = shape[axis + 1..].iter().product();
    let mut out_f = Vec::new();
    let mut out_i = Vec::new();
    let mut out_b = Vec::new();
    for (data, tshape) in &datas {
        let outer: usize = tshape[..axis].iter().product();
        let dim = tshape[axis];
        match (&data, kind) {
            (Data::F32(v), 0) => {
                for o in 0..outer {
                    out_f.extend_from_slice(&v[o * dim * inner..(o + 1) * dim * inner]);
                }
            }
            (Data::I64(v), 1) => {
                for o in 0..outer {
                    out_i.extend_from_slice(&v[o * dim * inner..(o + 1) * dim * inner]);
                }
            }
            (Data::Bool(v), 2) => {
                for o in 0..outer {
                    out_b.extend_from_slice(&v[o * dim * inner..(o + 1) * dim * inner]);
                }
            }
            _ => return Err(Error::Model("Concat: mixed dtypes".into())),
        }
    }
    let t = match kind {
        0 => Tensor::f32(out_f, shape),
        1 => Tensor::i64(out_i, shape),
        _ => Tensor::bool(out_b, shape),
    };
    output(env, node, 0, t)?;
    Ok(())
}

pub(crate) fn split(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;

    let data = x.data.clone();
    let shape = x.shape.clone();
    let rank = shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(0),
        rank,
    )?;
    let num_outputs = node.outputs.len();
    let splits: Vec<usize> = match node.attr("split") {
        Some(a) => a.ints()?.iter().map(|&s| s as usize).collect(),
        None => match input_opt(env, node, 1)? {
            Some(t) => i64s(t)?.iter().map(|&s| s as usize).collect(),
            None => {
                let n = shape[axis];
                let part = n / num_outputs;
                vec![part; num_outputs]
            }
        },
    };
    if splits.len() != num_outputs {
        return Err(Error::Model(format!(
            "Split: {} splits for {} outputs",
            splits.len(),
            num_outputs
        )));
    }
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut off = 0usize;
    for (i, &len) in splits.iter().enumerate() {
        let mut out_shape = shape.clone();
        out_shape[axis] = len;
        let mut v_f: Vec<f32> = vec![];
        let mut v_i: Vec<i64> = vec![];
        let mut v_b: Vec<bool> = vec![];
        for o in 0..outer {
            let base = (o * shape[axis] + off) * inner;
            match &data {
                Data::F32(v) => v_f.extend_from_slice(&v[base..base + len * inner]),
                Data::I64(v) => v_i.extend_from_slice(&v[base..base + len * inner]),
                Data::Bool(v) => v_b.extend_from_slice(&v[base..base + len * inner]),
            }
        }
        let t = match &data {
            Data::F32(_) => Tensor::f32(v_f, out_shape),
            Data::I64(_) => Tensor::i64(v_i, out_shape),
            Data::Bool(_) => Tensor::bool(v_b, out_shape),
        };
        output(env, node, i, t)?;
        off += len;
    }
    Ok(())
}

pub(crate) fn slice(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let starts = i64s(input(env, node, 1)?)?;
    let ends = i64s(input(env, node, 2)?)?;
    let axes: Vec<usize> = match input_opt(env, node, 3)? {
        Some(t) => i64s(t)?
            .iter()
            .map(|&a| norm_axis(a, rank))
            .collect::<Result<_>>()?,
        None => (0..starts.len()).collect(),
    };
    let steps: Vec<i64> = match input_opt(env, node, 4)? {
        Some(t) => i64s(t)?.to_vec(),
        None => vec![1; starts.len()],
    };
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(Error::Model(
            "Slice: starts/ends/axes/steps length mismatch".into(),
        ));
    }

    let mut slices: Vec<(usize, usize, i64)> = (0..rank).map(|d| (0, x.shape[d], 1)).collect();
    for (i, &ax) in axes.iter().enumerate() {
        let dim = x.shape[ax] as i64;
        let step = steps[i];
        if step == 0 {
            return Err(Error::Model("Slice: zero step".into()));
        }
        let norm = |v: i64| if v < 0 { v + dim } else { v };
        let (s, e) = if step > 0 {
            (norm(starts[i]).clamp(0, dim), norm(ends[i]).clamp(0, dim))
        } else {
            (
                norm(starts[i]).clamp(0, dim - 1),
                norm(ends[i]).clamp(-1, dim - 1),
            )
        };
        slices[ax] = (s as usize, e as usize, step);
    }

    let mut shape = x.shape.clone();
    for (d, (s, e, step)) in slices.iter().enumerate() {
        let (s, e) = (*s, *e);
        shape[d] = if *step > 0 {
            e.saturating_sub(s).div_ceil(*step as usize)
        } else {
            s.saturating_sub(e).div_ceil((-step) as usize)
        };
    }
    let data = match &x.data {
        Data::F32(v) => Data::F32(slice_data(v, &x.shape, &slices)),
        Data::I64(v) => Data::I64(slice_data(v, &x.shape, &slices)),
        Data::Bool(v) => Data::Bool(slice_data(v, &x.shape, &slices)),
    };
    output(env, node, 0, Tensor { data, shape })?;
    Ok(())
}

fn slice_data<T: Copy>(v: &[T], shape: &[usize], slices: &[(usize, usize, i64)]) -> Vec<T> {
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(d, _)| {
            let (s, e, step) = slices[d];
            if step > 0 {
                e.saturating_sub(s).div_ceil(step as usize)
            } else {
                s.saturating_sub(e).div_ceil((-step) as usize)
            }
        })
        .collect();
    let n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..n {
        let mut flat = 0usize;
        for d in 0..shape.len() {
            let (s, _, step) = slices[d];
            let i = if step > 0 {
                s + idx[d] * step as usize
            } else {
                s - idx[d] * (-step) as usize
            };
            flat = flat * shape[d] + i;
        }
        out.push(v[flat]);

        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

pub(crate) fn gather(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(0),
        rank,
    )?;
    let idx = input(env, node, 1)?;
    let indices: Vec<usize> = i64s(idx)?
        .iter()
        .map(|&i| {
            let i = if i < 0 { i + x.shape[axis] as i64 } else { i };
            if i < 0 || i >= x.shape[axis] as i64 {
                return Err(Error::Model(format!("Gather: index {i} out of range")));
            }
            Ok(i as usize)
        })
        .collect::<Result<_>>()?;
    let idx_shape = idx.shape.clone();

    let mut shape: Vec<usize> = x.shape[..axis].to_vec();
    shape.extend_from_slice(&idx_shape);
    shape.extend_from_slice(&x.shape[axis + 1..]);
    let outer: usize = x.shape[..axis].iter().product();
    let inner: usize = x.shape[axis + 1..].iter().product();
    let dim = x.shape[axis];
    let idx_numel = indices.len();
    let mut out_f = Vec::with_capacity(outer * idx_numel * inner);
    let mut out_i = Vec::with_capacity(outer * idx_numel * inner);
    let mut out_b = Vec::with_capacity(outer * idx_numel * inner);
    match &x.data {
        Data::F32(v) => {
            for o in 0..outer {
                for &i in &indices {
                    out_f.extend_from_slice(&v[(o * dim + i) * inner..(o * dim + i + 1) * inner]);
                }
            }
        }
        Data::I64(v) => {
            for o in 0..outer {
                for &i in &indices {
                    out_i.extend_from_slice(&v[(o * dim + i) * inner..(o * dim + i + 1) * inner]);
                }
            }
        }
        Data::Bool(v) => {
            for o in 0..outer {
                for &i in &indices {
                    out_b.extend_from_slice(&v[(o * dim + i) * inner..(o * dim + i + 1) * inner]);
                }
            }
        }
    }
    let t = match x.data {
        Data::F32(_) => Tensor::f32(out_f, shape),
        Data::I64(_) => Tensor::i64(out_i, shape),
        Data::Bool(_) => Tensor::bool(out_b, shape),
    };
    output(env, node, 0, t)?;
    Ok(())
}

pub(crate) fn gather_elements(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(0),
        rank,
    )?;
    let idx = input(env, node, 1)?;
    if idx.shape != x.shape {
        return Err(Error::Model(
            "GatherElements: index shape != data shape".into(),
        ));
    }
    let indices: Vec<usize> = i64s(idx)?
        .iter()
        .map(|&i| {
            let i = if i < 0 { i + x.shape[axis] as i64 } else { i };
            if i < 0 || i >= x.shape[axis] as i64 {
                return Err(Error::Model(format!(
                    "GatherElements: index {i} out of range"
                )));
            }
            Ok(i as usize)
        })
        .collect::<Result<_>>()?;
    let inner: usize = x.shape[axis + 1..].iter().product();
    let dim = x.shape[axis];
    let n = x.numel();
    let data = match &x.data {
        Data::F32(v) => {
            let mut out = vec![0f32; n];
            for (k, &i) in indices.iter().enumerate() {
                let o = k / (dim * inner);
                let r = k % (dim * inner);
                out[k] = v[(o * dim + i) * inner + r];
            }
            Data::F32(out)
        }
        Data::I64(v) => {
            let mut out = vec![0i64; n];
            for (k, &i) in indices.iter().enumerate() {
                let o = k / (dim * inner);
                let r = k % (dim * inner);
                out[k] = v[(o * dim + i) * inner + r];
            }
            Data::I64(out)
        }
        Data::Bool(v) => {
            let mut out = vec![false; n];
            for (k, &i) in indices.iter().enumerate() {
                let o = k / (dim * inner);
                let r = k % (dim * inner);
                out[k] = v[(o * dim + i) * inner + r];
            }
            Data::Bool(out)
        }
    };
    output(
        env,
        node,
        0,
        Tensor {
            data,
            shape: x.shape.clone(),
        },
    )?;
    Ok(())
}

pub(crate) fn gather_nd(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let idx = input(env, node, 1)?;
    let batch_dims = node
        .attr("batch_dims")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(0) as usize;
    let k = *idx.shape.last().unwrap_or(&0);
    if idx.shape.is_empty() || k < batch_dims {
        return Err(Error::Model("GatherND: bad index rank".into()));
    }
    if batch_dims > 0 {
        return Err(Error::Model(format!(
            "GatherND: batch_dims {batch_dims} is not supported"
        )));
    }
    let data_rank = x.shape.len();
    if k > data_rank {
        return Err(Error::Model("GatherND: k exceeds data rank".into()));
    }
    let indices: Vec<usize> = i64s(idx)?
        .iter()
        .map(|&i| {
            if i < 0 {
                return Err(Error::Model("GatherND: negative indices".into()));
            }
            Ok(i as usize)
        })
        .collect::<Result<_>>()?;
    let idx_rows = indices.len() / k;
    let slice_len: usize = x.shape[k..].iter().product();
    let mut out_f = Vec::with_capacity(idx_rows * slice_len);
    let mut out_i = Vec::with_capacity(idx_rows * slice_len);
    let mut out_b = Vec::with_capacity(idx_rows * slice_len);
    match &x.data {
        Data::F32(v) => {
            for r in 0..idx_rows {
                let off = gather_nd_offset(&x.shape, &indices[r * k..(r + 1) * k])?;
                out_f.extend_from_slice(&v[off..off + slice_len]);
            }
        }
        Data::I64(v) => {
            for r in 0..idx_rows {
                let off = gather_nd_offset(&x.shape, &indices[r * k..(r + 1) * k])?;
                out_i.extend_from_slice(&v[off..off + slice_len]);
            }
        }
        Data::Bool(v) => {
            for r in 0..idx_rows {
                let off = gather_nd_offset(&x.shape, &indices[r * k..(r + 1) * k])?;
                out_b.extend_from_slice(&v[off..off + slice_len]);
            }
        }
    }
    let mut shape: Vec<usize> = idx.shape[..idx.shape.len() - 1].to_vec();
    shape.extend_from_slice(&x.shape[k..]);
    let t = match x.data {
        Data::F32(_) => Tensor::f32(out_f, shape),
        Data::I64(_) => Tensor::i64(out_i, shape),
        Data::Bool(_) => Tensor::bool(out_b, shape),
    };
    output(env, node, 0, t)?;
    Ok(())
}

fn gather_nd_offset(shape: &[usize], indices: &[usize]) -> Result<usize> {
    let mut off = 0usize;
    for (d, &i) in indices.iter().enumerate() {
        if i >= shape[d] {
            return Err(Error::Model(format!("GatherND: index {i} out of range")));
        }
        off = off * shape[d] + i;
    }
    for d in indices.len()..shape.len() {
        off *= shape[d];
    }
    Ok(off)
}

pub(crate) fn squeeze(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axes: Vec<usize> = match node.attr("axes") {
        Some(a) => a
            .ints()?
            .iter()
            .map(|&a| norm_axis(a, rank))
            .collect::<Result<_>>()?,
        None => match input_opt(env, node, 1)? {
            Some(t) => i64s(t)?
                .iter()
                .map(|&a| norm_axis(a, rank))
                .collect::<Result<_>>()?,
            None => (0..rank).filter(|&d| x.shape[d] == 1).collect(),
        },
    };
    for &a in &axes {
        if x.shape[a] != 1 {
            return Err(Error::Model(format!(
                "Squeeze: axis {a} has size {}",
                x.shape[a]
            )));
        }
    }
    let shape: Vec<usize> = x
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, d)| *d)
        .collect();
    output(
        env,
        node,
        0,
        Tensor {
            data: x.data.clone(),
            shape,
        },
    )?;
    Ok(())
}

pub(crate) fn unsqueeze(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axes: Vec<i64> = match node.attr("axes") {
        Some(a) => a.ints()?.to_vec(),
        None => i64s(input(env, node, 1)?)?.to_vec(),
    };
    let out_rank = rank + axes.len();
    let mut axes_norm: Vec<usize> = axes
        .iter()
        .map(|&a| {
            let a = if a < 0 { a + out_rank as i64 } else { a };
            if a < 0 || a >= out_rank as i64 {
                return Err(Error::Model(format!("Unsqueeze: axis {a} out of range")));
            }
            Ok(a as usize)
        })
        .collect::<Result<_>>()?;
    axes_norm.sort_unstable();
    if axes_norm.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::Model("Unsqueeze: duplicate axes".into()));
    }
    let mut shape = vec![1usize; out_rank];
    let mut src = 0usize;
    for d in 0..out_rank {
        if !axes_norm.contains(&d) {
            shape[d] = x.shape[src];
            src += 1;
        }
    }
    output(
        env,
        node,
        0,
        Tensor {
            data: x.data.clone(),
            shape,
        },
    )?;
    Ok(())
}

pub(crate) fn flatten(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(1),
        rank,
    )?;
    let d0: usize = x.shape[..axis].iter().product();
    let d1: usize = x.shape[axis..].iter().product();
    let shape = vec![d0, d1];
    output(
        env,
        node,
        0,
        Tensor {
            data: x.data.clone(),
            shape,
        },
    )?;
    Ok(())
}

pub(crate) fn expand(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let target: Vec<usize> = i64s(input(env, node, 1)?)?
        .iter()
        .map(|&d| d as usize)
        .collect();
    let shape = broadcast_shape(&x.shape, &target)?;
    let data = match &x.data {
        Data::F32(v) => Data::F32(broadcast_to(v, &x.shape, &shape)?),
        Data::I64(v) => Data::I64(broadcast_to(v, &x.shape, &shape)?),
        Data::Bool(v) => Data::Bool(broadcast_to(v, &x.shape, &shape)?),
    };
    output(env, node, 0, Tensor { data, shape })?;
    Ok(())
}

pub(crate) fn tile(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let repeats: Vec<usize> = i64s(input(env, node, 1)?)?
        .iter()
        .map(|&r| r as usize)
        .collect();
    if repeats.len() != x.shape.len() {
        return Err(Error::Model("Tile: repeats length != rank".into()));
    }
    let shape: Vec<usize> = x.shape.iter().zip(&repeats).map(|(&d, &r)| d * r).collect();
    let n: usize = shape.iter().product();
    let data = match &x.data {
        Data::F32(v) => Data::F32(tile_data(v, &x.shape, &repeats, n)),
        Data::I64(v) => Data::I64(tile_data(v, &x.shape, &repeats, n)),
        Data::Bool(v) => Data::Bool(tile_data(v, &x.shape, &repeats, n)),
    };
    output(env, node, 0, Tensor { data, shape })?;
    Ok(())
}

fn tile_data<T: Copy>(v: &[T], shape: &[usize], repeats: &[usize], n: usize) -> Vec<T> {
    let rank = shape.len();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let mut src = 0usize;
        for d in 0..rank {
            src = src * shape[d] + idx[d] % shape[d];
        }
        out.push(v[src]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] * repeats[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

pub(crate) fn constant_of_shape(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let shape_in = i64s(input(env, node, 0)?)?;
    let shape: Vec<usize> = shape_in.iter().map(|&d| d as usize).collect();
    let t = match node.attr("value") {
        Some(crate::graph::Attr::T(t)) => match &t.data {
            Data::F32(v) => Tensor::f32(vec![v[0]; shape.iter().product()], shape),
            Data::I64(v) => Tensor::i64(vec![v[0]; shape.iter().product()], shape),
            Data::Bool(v) => Tensor::bool(vec![v[0]; shape.iter().product()], shape),
        },
        None => Tensor::f32(vec![0.0; shape.iter().product()], shape),
        Some(other) => {
            return Err(Error::Model(format!(
                "ConstantOfShape: value must be a tensor, got {other:?}"
            )));
        }
    };
    output(env, node, 0, t)?;
    Ok(())
}

pub(crate) fn range(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let start = input(env, node, 0)?;
    let limit = input(env, node, 1)?;
    let delta = input(env, node, 2)?;
    match (&start.data, &limit.data, &delta.data) {
        (Data::F32(s), Data::F32(l), Data::F32(d)) => {
            let mut out: Vec<f32> = vec![];
            let mut v = s[0];
            while if d[0] > 0.0 { v < l[0] } else { v > l[0] } {
                out.push(v);
                v += d[0];
            }
            let len = out.len();
            output(env, node, 0, Tensor::f32(out, vec![len]))?;
            Ok(())
        }
        (Data::I64(s), Data::I64(l), Data::I64(d)) => {
            if d[0] == 0 {
                return Err(Error::Model("Range: zero delta".into()));
            }
            let mut out: Vec<i64> = vec![];
            let mut v = s[0];
            while if d[0] > 0 { v < l[0] } else { v > l[0] } {
                out.push(v);
                v += d[0];
            }
            let len = out.len();
            output(env, node, 0, Tensor::i64(out, vec![len]))?;
            Ok(())
        }
        _ => Err(Error::Model("Range: mixed dtypes".into())),
    }?;
    Ok(())
}

pub(crate) fn identity(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    output(env, node, 0, x.clone())?;
    Ok(())
}

pub(crate) fn constant(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let t = match node.attr("value") {
        Some(crate::graph::Attr::T(t)) => t.clone(),
        Some(crate::graph::Attr::F(f)) => Tensor::f32(vec![*f], vec![]),
        Some(crate::graph::Attr::I(i)) => Tensor::i64(vec![*i], vec![]),
        Some(crate::graph::Attr::Floats(v)) => Tensor::f32(v.clone(), vec![v.len()]),
        Some(crate::graph::Attr::Ints(v)) => Tensor::i64(v.clone(), vec![v.len()]),
        Some(crate::graph::Attr::S(_)) => {
            return Err(Error::Model("Constant: string values unsupported".into()));
        }
        Some(other) => {
            return Err(Error::Model(format!(
                "Constant: unsupported attribute {other:?}"
            )));
        }
        None => return Err(Error::Model("Constant: no value attribute".into())),
    };
    output(env, node, 0, t)?;
    Ok(())
}

pub(crate) fn cast(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let to = node.need("to")?.i()? as i32;
    let data = match to {
        crate::dtype::F32 | crate::dtype::F16 | crate::dtype::F64 | crate::dtype::BF16 => {
            Data::F32(x.data.as_f32())
        }
        crate::dtype::U8
        | crate::dtype::I8
        | crate::dtype::U16
        | crate::dtype::I16
        | crate::dtype::I32
        | crate::dtype::I64
        | crate::dtype::U32
        | crate::dtype::U64 => Data::I64(x.data.as_i64()),
        crate::dtype::BOOL => Data::Bool(match &x.data {
            Data::Bool(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
            Data::I64(v) => v.iter().map(|&x| x != 0).collect(),
        }),
        other => {
            return Err(Error::Model(format!(
                "Cast: unsupported target type {other}"
            )));
        }
    };
    output(
        env,
        node,
        0,
        Tensor {
            data,
            shape: x.shape.clone(),
        },
    )?;
    Ok(())
}

pub(crate) fn cast_like(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let like = input(env, node, 1)?;
    let data = match &like.data {
        Data::F32(_) => Data::F32(x.data.as_f32()),
        Data::I64(_) => Data::I64(x.data.as_i64()),
        Data::Bool(_) => Data::Bool(match &x.data {
            Data::Bool(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
            Data::I64(v) => v.iter().map(|&x| x != 0).collect(),
        }),
    };
    output(
        env,
        node,
        0,
        Tensor {
            data,
            shape: x.shape.clone(),
        },
    )?;
    Ok(())
}

pub(crate) fn pad(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let pads = i64s(input(env, node, 1)?)?;
    if pads.len() != 2 * rank {
        return Err(Error::Model(format!(
            "Pad: pads length {} != 2*rank {rank}",
            pads.len()
        )));
    }
    let mode = node
        .attr("mode")
        .map(|a| a.s())
        .transpose()?
        .map(|s| String::from_utf8_lossy(s).to_string())
        .unwrap_or_else(|| "constant".into());
    let value = match input_opt(env, node, 2)? {
        Some(t) => t.data.as_f32()[0],
        None => 0.0,
    };
    let axes: Option<Vec<usize>> = match input_opt(env, node, 3)? {
        Some(t) => Some(
            i64s(t)?
                .iter()
                .map(|&a| norm_axis(a, rank))
                .collect::<Result<_>>()?,
        ),
        None => None,
    };
    let mut begins = vec![0i64; rank];
    let mut ends = vec![0i64; rank];
    for d in 0..rank {
        if axes.as_ref().is_none_or(|ax| ax.contains(&d)) {
            begins[d] = pads[d];
            ends[d] = pads[d + rank];
        }
    }
    let mut shape = Vec::with_capacity(rank);
    for d in 0..rank {
        shape.push(x.shape[d] + begins[d] as usize + ends[d] as usize);
    }
    let data = match &x.data {
        Data::F32(v) => Data::F32(pad_data(v, &x.shape, &shape, &begins, mode.as_str(), value)),
        Data::I64(v) => Data::I64(pad_data(
            v,
            &x.shape,
            &shape,
            &begins,
            mode.as_str(),
            value as i64,
        )),
        Data::Bool(v) => Data::Bool(pad_data(
            v,
            &x.shape,
            &shape,
            &begins,
            mode.as_str(),
            value != 0.0,
        )),
    };
    output(env, node, 0, Tensor { data, shape })?;
    Ok(())
}

fn pad_data<T: Copy>(
    v: &[T],
    shape: &[usize],
    out_shape: &[usize],
    begins: &[i64],
    mode: &str,
    constant: T,
) -> Vec<T> {
    let rank = shape.len();
    let n: usize = out_shape.iter().product();
    let mut out = vec![constant; n];
    let mut idx = vec![0usize; rank];
    for o in 0..n {
        let mut src_ok = true;
        let mut flat = 0usize;
        for d in 0..rank {
            let b = begins[d] as usize;
            let i = idx[d];
            if i < b || i >= b + shape[d] {
                src_ok = false;
                break;
            }
            flat = flat * shape[d] + (i - b);
        }
        if src_ok {
            match mode {
                "reflect" | "edge" => {
                    let mut flat = 0usize;
                    for d in 0..rank {
                        let b = begins[d] as usize;
                        let mut i = idx[d];
                        if i < b {
                            i = match mode {
                                "reflect" => b - i,
                                _ => 0,
                            };
                        } else if i >= b + shape[d] {
                            i = match mode {
                                "reflect" => 2 * b + shape[d] - 2 - i,
                                _ => shape[d] - 1,
                            };
                        }
                        flat = flat * shape[d] + (i - b);
                    }
                    out[o] = v[flat];
                }
                _ => out[o] = v[flat],
            }
        }

        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

pub(crate) fn topk(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(-1),
        rank,
    )?;
    let k = i64s(input(env, node, 1)?)?[0] as usize;
    let largest = node
        .attr("largest")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(1)
        != 0;
    let sorted = node.attr("sorted").map(|a| a.i()).transpose()?.unwrap_or(1) != 0;
    let v = f32s(x)?;
    let outer: usize = x.shape[..axis].iter().product();
    let dim = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product();
    let mut vals = vec![0f32; outer * k * inner];
    let mut idxs = vec![0i64; outer * k * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(f32, usize)> = (0..dim)
                .map(|d| (v[o * dim * inner + d * inner + i], d))
                .collect();
            pairs.sort_by(|a, b| {
                if largest {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                }
            });
            if !sorted {
                let mut selected = pairs[..k].to_vec();
                selected.sort_by_key(|p| p.1);
                pairs = selected;
            }
            for (j, (val, d)) in pairs.iter().take(k).enumerate() {
                vals[(o * k + j) * inner + i] = *val;
                idxs[(o * k + j) * inner + i] = *d as i64;
            }
        }
    }
    let mut shape = x.shape.clone();
    shape[axis] = k;
    output(env, node, 0, Tensor::f32(vals, shape.clone()))?;
    output(env, node, 1, Tensor::i64(idxs, shape))?;
    Ok(())
}

pub(crate) fn nonzero(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let n = x.numel();
    let mut coords: Vec<Vec<i64>> = vec![vec![]; rank];
    for flat in 0..n {
        let mut rem = flat;
        let mut idx = vec![0usize; rank];
        for d in (0..rank).rev() {
            idx[d] = rem % x.shape[d];
            rem /= x.shape[d];
        }
        let is_nonzero = match &x.data {
            Data::F32(v) => v[flat] != 0.0,
            Data::I64(v) => v[flat] != 0,
            Data::Bool(v) => v[flat],
        };
        if is_nonzero {
            for (d, &c) in idx.iter().enumerate() {
                coords[d].push(c as i64);
            }
        }
    }
    let count = coords[0].len();
    let mut out = vec![0i64; rank * count];
    for (d, c) in coords.iter().enumerate() {
        for (i, &v) in c.iter().enumerate() {
            out[i * rank + d] = v;
        }
    }
    output(env, node, 0, Tensor::i64(out, vec![rank, count]))?;
    Ok(())
}

pub(crate) fn scatter_nd(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let data = input(env, node, 0)?;
    let idx = input(env, node, 1)?;
    let updates = input(env, node, 2)?;
    let reduction = node
        .attr("reduction")
        .map(|a| a.s())
        .transpose()?
        .map(|s| String::from_utf8_lossy(s).to_string())
        .unwrap_or_else(|| "none".into());
    let k = *idx.shape.last().unwrap_or(&0);
    let indices: Vec<usize> = i64s(idx)?
        .iter()
        .map(|&i| {
            if i < 0 {
                return Err(Error::Model("ScatterND: negative index".into()));
            }
            Ok(i as usize)
        })
        .collect::<Result<_>>()?;
    let idx_rows = indices.len() / k;
    let slice_len: usize = data.shape[k..].iter().product();
    let mut out = match &data.data {
        Data::F32(v) => Data::F32(v.clone()),
        Data::I64(v) => Data::I64(v.clone()),
        Data::Bool(v) => Data::Bool(v.clone()),
    };
    for r in 0..idx_rows {
        let idx_vec = &indices[r * k..(r + 1) * k];

        let mut off = 0usize;
        for (d, &i) in idx_vec.iter().enumerate() {
            if i >= data.shape[d] {
                return Err(Error::Model(format!("ScatterND: index {i} out of range")));
            }
            off = off * data.shape[d] + i;
        }
        for d in k..data.shape.len() {
            off *= data.shape[d];
        }
        match (&mut out, &updates.data) {
            (Data::F32(o), Data::F32(u)) => {
                for j in 0..slice_len {
                    let u = u[r * slice_len + j];
                    match reduction.as_str() {
                        "add" => o[off + j] += u,
                        "mul" => o[off + j] *= u,
                        _ => o[off + j] = u,
                    }
                }
            }
            (Data::I64(o), Data::I64(u)) => {
                for j in 0..slice_len {
                    let u = u[r * slice_len + j];
                    match reduction.as_str() {
                        "add" => o[off + j] += u,
                        "mul" => o[off + j] *= u,
                        _ => o[off + j] = u,
                    }
                }
            }
            (Data::Bool(o), Data::Bool(u)) => {
                for j in 0..slice_len {
                    o[off + j] = u[r * slice_len + j];
                }
            }
            _ => return Err(Error::Model("ScatterND: dtype mismatch".into())),
        }
    }
    output(
        env,
        node,
        0,
        Tensor {
            data: out,
            shape: data.shape.clone(),
        },
    )?;
    Ok(())
}

pub(crate) fn scatter_elements(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let data = input(env, node, 0)?;
    let idx = input(env, node, 1)?;
    let updates = input(env, node, 2)?;
    let rank = data.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(0),
        rank,
    )?;
    let reduction = node
        .attr("reduction")
        .map(|a| a.s())
        .transpose()?
        .map(|s| String::from_utf8_lossy(s).to_string())
        .unwrap_or_else(|| "none".into());
    let indices: Vec<usize> = i64s(idx)?
        .iter()
        .map(|&i| {
            let i = if i < 0 {
                i + data.shape[axis] as i64
            } else {
                i
            };
            if i < 0 || i >= data.shape[axis] as i64 {
                return Err(Error::Model(format!(
                    "ScatterElements: index {i} out of range"
                )));
            }
            Ok(i as usize)
        })
        .collect::<Result<_>>()?;
    let inner: usize = data.shape[axis + 1..].iter().product();
    let dim = data.shape[axis];
    let mut out = match &data.data {
        Data::F32(v) => Data::F32(v.clone()),
        Data::I64(v) => Data::I64(v.clone()),
        Data::Bool(v) => Data::Bool(v.clone()),
    };
    for (k, &i) in indices.iter().enumerate() {
        let o = k / (dim * inner);
        let r = k % (dim * inner);
        let dst = (o * dim + i) * inner + r;
        match (&mut out, &updates.data) {
            (Data::F32(d), Data::F32(u)) => match reduction.as_str() {
                "add" => d[dst] += u[k],
                "mul" => d[dst] *= u[k],
                _ => d[dst] = u[k],
            },
            (Data::I64(d), Data::I64(u)) => match reduction.as_str() {
                "add" => d[dst] += u[k],
                "mul" => d[dst] *= u[k],
                _ => d[dst] = u[k],
            },
            (Data::Bool(d), Data::Bool(u)) => d[dst] = u[k],
            _ => return Err(Error::Model("ScatterElements: dtype mismatch".into())),
        }
    }
    output(
        env,
        node,
        0,
        Tensor {
            data: out,
            shape: data.shape.clone(),
        },
    )?;
    Ok(())
}
