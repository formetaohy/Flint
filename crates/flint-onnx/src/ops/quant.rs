use std::collections::HashMap;

use flint_error::{Error, Result};

use crate::graph::Node;
use crate::ops::{f32s, i64s, input, input_opt, norm_axis, output};
use crate::tensor::Tensor;

pub(crate) fn dynamic_quantize_linear(
    env: &mut HashMap<String, Tensor>,
    node: &Node,
) -> Result<()> {
    let x = input(env, node, 0)?;
    let v = f32s(x)?;
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &e in v {
        min = min.min(e);
        max = max.max(e);
    }
    let scale = (max - min) / 255.0;
    if scale == 0.0 {
        return Err(Error::Model("DynamicQuantizeLinear: constant input".into()));
    }
    let zp = if min >= 0.0 {
        0.0
    } else {
        (-min / scale).round().clamp(0.0, 255.0)
    };
    let y: Vec<i64> = v
        .iter()
        .map(|&e| ((e / scale).round() + zp).clamp(0.0, 255.0) as i64)
        .collect();
    output(env, node, 0, Tensor::i64(y, x.shape.clone()))?;
    output(env, node, 1, Tensor::f32(vec![scale], vec![]))?;
    output(env, node, 2, Tensor::i64(vec![zp as i64], vec![]))?;
    Ok(())
}

pub(crate) fn dequantize_linear(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let scale = input(env, node, 1)?;
    let zp = input_opt(env, node, 2)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(1),
        rank,
    )?;
    let xv = i64s(x)?;
    let sv = f32s(scale)?;
    let zv: Option<Vec<i64>> = zp.map(|t| i64s(t).map(|v| v.to_vec())).transpose()?;
    let n = x.numel();
    let mut out = vec![0f32; n];
    if sv.len() == 1 && zv.as_ref().is_none_or(|z| z.len() == 1) {
        let s = sv[0];
        let z = zv.as_ref().map_or(0.0, |z| z[0] as f32);
        for (o, &e) in out.iter_mut().zip(xv) {
            *o = (e as f32 - z) * s;
        }
        output(env, node, 0, Tensor::f32(out, x.shape.clone()))?;
        return Ok(());
    }

    let dim = x.shape[axis];
    if sv.len() != dim {
        return Err(Error::Model(format!(
            "DequantizeLinear: scale len {} != axis dim {dim}",
            sv.len()
        )));
    }
    if zv.as_ref().is_some_and(|z| z.len() != 1 && z.len() != dim) {
        return Err(Error::Model("DequantizeLinear: bad zero point shape".into()));
    }
    let inner: usize = x.shape[axis + 1..].iter().product();
    let outer: usize = x.shape[..axis].iter().product();
    for o in 0..outer {
        for d in 0..dim {
            let z = match &zv {
                Some(z) if z.len() == dim => z[o * dim + d] as f32,
                Some(z) => z[0] as f32,
                None => 0.0,
            };
            let base = (o * dim + d) * inner;
            let s = sv[d];
            for i in 0..inner {
                out[base + i] = (xv[base + i] as f32 - z) * s;
            }
        }
    }
    output(env, node, 0, Tensor::f32(out, x.shape.clone()))?;
    Ok(())
}

pub(crate) fn matmul_integer(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let a = input(env, node, 0)?;
    let b = input(env, node, 1)?;
    let a_zp = input_opt(env, node, 2)?;
    let b_zp = input_opt(env, node, 3)?;
    let av = i64s(a)?;
    let bv = i64s(b)?;
    let a_zpv: Option<Vec<i64>> = a_zp.map(|t| i64s(t).map(|v| v.to_vec())).transpose()?;
    let b_zpv: Option<Vec<i64>> = b_zp.map(|t| i64s(t).map(|v| v.to_vec())).transpose()?;
    let ra = a.shape.len();
    let rb = b.shape.len();
    let (m, k) = (a.shape[ra - 2], a.shape[ra - 1]);
    let (k2, n) = (b.shape[rb - 2], b.shape[rb - 1]);
    if k != k2 {
        return Err(Error::Model("MatMulInteger: inner dim mismatch".into()));
    }
    let batch = crate::tensor::broadcast_shape(&a.shape[..ra - 2], &b.shape[..rb - 2])?;
    let batch_n: usize = batch.iter().product();
    let mut shape = batch.clone();
    shape.extend_from_slice(&[m, n]);
    let mut out = vec![0i64; batch_n * m * n];

    let a_z = |r: usize| -> i64 {
        a_zpv.as_ref().map_or(0, |z| if z.len() == 1 { z[0] } else { z[r] })
    };
    let b_z = |c: usize| -> i64 {
        b_zpv.as_ref().map_or(0, |z| if z.len() == 1 { z[0] } else { z[c] })
    };
    for bt in 0..batch_n {
        let a_base = bt * m * k;
        let b_base = bt * k * n;
        let o_base = bt * m * n;
        for i in 0..m {
            let az = a_z(i);
            for j in 0..n {
                let bz = b_z(j);
                let mut acc = 0i64;
                for p in 0..k {
                    acc += (av[a_base + i * k + p] - az) * (bv[b_base + p * n + j] - bz);
                }
                out[o_base + i * n + j] = acc;
            }
        }
    }
    output(env, node, 0, Tensor::i64(out, shape))?;
    Ok(())
}
