use std::collections::HashMap;

use flint_error::{Error, Result};

use crate::graph::Node;
use crate::ops::{f32s, input, input_opt, norm_axis, output};
use crate::tensor::{Tensor, broadcast_shape, broadcast_to};

pub(crate) fn matmul(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let a = input(env, node, 0)?;
    let b = input(env, node, 1)?;
    let (av, bv) = (f32s(a)?, f32s(b)?);
    let ra = a.shape.len();
    let rb = b.shape.len();
    let (shape, out) = match (ra, rb) {
        (1, 1) => {
            if a.shape[0] != b.shape[0] {
                return Err(Error::Model("MatMul: inner dim mismatch".into()));
            }
            let s: f32 = av.iter().zip(bv).map(|(&x, &y)| x * y).sum();
            (vec![], vec![s])
        }
        (1, 2) => {
            let (k, n) = (a.shape[0], b.shape[1]);
            if k != b.shape[0] {
                return Err(Error::Model("MatMul: inner dim mismatch".into()));
            }
            let mut out = vec![0f32; n];
            for i in 0..n {
                out[i] = (0..k).map(|j| av[j] * bv[j * n + i]).sum();
            }
            (vec![n], out)
        }
        (2, 1) => {
            let (m, k) = (a.shape[0], a.shape[1]);
            if k != b.shape[0] {
                return Err(Error::Model("MatMul: inner dim mismatch".into()));
            }
            let mut out = vec![0f32; m];
            for i in 0..m {
                out[i] = (0..k).map(|j| av[i * k + j] * bv[j]).sum();
            }
            (vec![m], out)
        }
        _ => {
            let m = a.shape[ra - 2];
            let k = a.shape[ra - 1];
            let k2 = b.shape[rb - 2];
            let n = b.shape[rb - 1];
            if k != k2 {
                return Err(Error::Model("MatMul: inner dim mismatch".into()));
            }
            let batch = broadcast_shape(&a.shape[..ra - 2], &b.shape[..rb - 2])?;
            let batch_n: usize = batch.iter().product();
            let mut shape = batch.clone();
            shape.extend_from_slice(&[m, n]);
            let mut out = vec![0f32; batch_n * m * n];
            for bt in 0..batch_n {
                let a_idx = broadcast_index(&batch, &a.shape[..ra - 2], bt);
                let b_idx = broadcast_index(&batch, &b.shape[..rb - 2], bt);
                let a_base = a_idx * m * k;
                let b_base = b_idx * k * n;
                let o_base = bt * m * n;
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0f32;
                        for p in 0..k {
                            acc += av[a_base + i * k + p] * bv[b_base + p * n + j];
                        }
                        out[o_base + i * n + j] = acc;
                    }
                }
            }
            (shape, out)
        }
    };
    output(env, node, 0, Tensor::f32(out, shape))?;
    Ok(())
}

fn broadcast_index(batch: &[usize], src: &[usize], flat: usize) -> usize {
    if src.is_empty() || src.iter().all(|&d| d == 1) {
        return 0;
    }

    let mut strides = vec![0usize; src.len()];
    let mut acc = 1usize;
    for d in (0..src.len()).rev() {
        strides[d] = acc;
        acc *= src[d];
    }
    let mut idx = flat;
    let mut out = 0usize;
    let off = batch.len() - src.len();
    for d in (0..batch.len()).rev() {
        let i = idx % batch[d];
        idx /= batch[d];
        if d >= off && src[d - off] != 1 {
            out += i * strides[d - off];
        }
    }
    out
}

pub(crate) fn gemm(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let a = input(env, node, 0)?;
    let b = input(env, node, 1)?;
    let alpha = node
        .attr("alpha")
        .map(|x| x.f())
        .transpose()?
        .unwrap_or(1.0);
    let beta = node.attr("beta").map(|x| x.f()).transpose()?.unwrap_or(1.0);
    let trans_a = node.attr("transA").map(|x| x.i()).transpose()?.unwrap_or(0) != 0;
    let trans_b = node.attr("transB").map(|x| x.i()).transpose()?.unwrap_or(0) != 0;
    let (av, bv) = (f32s(a)?, f32s(b)?);
    let (m, k) = if trans_a {
        (a.shape[1], a.shape[0])
    } else {
        (a.shape[0], a.shape[1])
    };
    let (k2, n) = if trans_b {
        (b.shape[1], b.shape[0])
    } else {
        (b.shape[0], b.shape[1])
    };
    if k != k2 {
        return Err(Error::Model("Gemm: inner dim mismatch".into()));
    }
    let shape = vec![m, n];
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let a_v = if trans_a {
                    av[p * m + i]
                } else {
                    av[i * k + p]
                };
                let b_v = if trans_b {
                    bv[j * k + p]
                } else {
                    bv[p * n + j]
                };
                acc += a_v * b_v;
            }
            out[i * n + j] = alpha * acc;
        }
    }
    if let Some(c) = input_opt(env, node, 2)? {
        let cv = f32s(c)?;
        let c_shape = broadcast_shape(&shape, &c.shape)?;
        let c_b = broadcast_to(cv, &c.shape, &c_shape)?;
        if c_shape != shape {
            return Err(Error::Model("Gemm: C shape mismatch".into()));
        }
        for (o, c) in out.iter_mut().zip(c_b) {
            *o += beta * c;
        }
    }
    output(env, node, 0, Tensor::f32(out, shape))?;
    Ok(())
}

pub(crate) fn softmax(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(-1),
        rank,
    )?;
    let v = f32s(x)?;
    let outer: usize = x.shape[..axis].iter().product();
    let dim = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product();
    let mut out = vec![0f32; x.numel()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * dim * inner + i;
            let mut max = f32::NEG_INFINITY;
            for d in 0..dim {
                max = max.max(v[base + d * inner]);
            }
            let mut sum = 0f32;
            for d in 0..dim {
                let e = (v[base + d * inner] - max).exp();
                out[base + d * inner] = e;
                sum += e;
            }
            for d in 0..dim {
                out[base + d * inner] /= sum;
            }
        }
    }
    output(env, node, 0, Tensor::f32(out, x.shape.clone()))?;
    Ok(())
}

pub(crate) fn softplus(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let v = f32s(x)?;
    let out = v.iter().map(|&x| (1f32 + x.exp()).ln()).collect();
    output(env, node, 0, Tensor::f32(out, x.shape.clone()))?;
    Ok(())
}

pub(crate) fn layer_norm(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let scale = input(env, node, 1)?;
    let bias = input_opt(env, node, 2)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(-1),
        rank,
    )?;
    let epsilon = node
        .attr("epsilon")
        .map(|a| a.f())
        .transpose()?
        .unwrap_or(1e-5);
    let v = f32s(x)?;
    let sv = f32s(scale)?;
    let bv = bias.map(|b| f32s(b)).transpose()?;
    let norm_dims: usize = x.shape[axis..].iter().product();
    let outer: usize = x.shape[..axis].iter().product();
    if sv.len() != norm_dims {
        return Err(Error::Model(format!(
            "LayerNormalization: scale len {} != normalized {norm_dims}",
            sv.len()
        )));
    }
    let mut out = vec![0f32; x.numel()];
    let mut mean_out = Vec::with_capacity(outer);
    let mut inv_out = Vec::with_capacity(outer);
    for o in 0..outer {
        let base = o * norm_dims;

        let mut sum = 0f64;
        for i in 0..norm_dims {
            sum += v[base + i] as f64;
        }
        let mean = sum / norm_dims as f64;
        let mut var = 0f64;
        for i in 0..norm_dims {
            let d = v[base + i] as f64 - mean;
            var += d * d;
        }
        var /= norm_dims as f64;
        let inv = 1.0 / (var + epsilon as f64).sqrt();
        for i in 0..norm_dims {
            let y = ((v[base + i] as f64 - mean) * inv) as f32 * sv[i];
            out[base + i] = match &bv {
                Some(b) => y + b[i],
                None => y,
            };
        }
        mean_out.push(mean as f32);
        inv_out.push(inv as f32);
    }
    output(env, node, 0, Tensor::f32(out, x.shape.clone()))?;
    if node.outputs.len() > 1 {
        output(env, node, 1, Tensor::f32(mean_out, vec![outer]))?;
    }
    if node.outputs.len() > 2 {
        output(env, node, 2, Tensor::f32(inv_out, vec![outer]))?;
    }
    Ok(())
}

pub(crate) fn batch_norm(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let scale = input(env, node, 1)?;
    let bias = input(env, node, 2)?;
    let mean = input(env, node, 3)?;
    let var = input(env, node, 4)?;
    let epsilon = node
        .attr("epsilon")
        .map(|a| a.f())
        .transpose()?
        .unwrap_or(1e-5);
    let v = f32s(x)?;
    let (sv, bv, mv, vv) = (f32s(scale)?, f32s(bias)?, f32s(mean)?, f32s(var)?);
    let c = x.shape[1];
    if sv.len() != c || bv.len() != c || mv.len() != c || vv.len() != c {
        return Err(Error::Model(
            "BatchNormalization: channel count mismatch".into(),
        ));
    }
    let outer: usize = x.shape[..1].iter().product();
    let inner: usize = x.shape[2..].iter().product();
    let mut out = vec![0f32; x.numel()];
    for o in 0..outer {
        for ch in 0..c {
            let a = sv[ch] / (vv[ch] + epsilon).sqrt();
            let b = bv[ch] - mv[ch] * a;
            let base = (o * c + ch) * inner;
            for i in 0..inner {
                out[base + i] = v[base + i] * a + b;
            }
        }
    }
    output(env, node, 0, Tensor::f32(out, x.shape.clone()))?;
    Ok(())
}

pub(crate) fn conv(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let w = input(env, node, 1)?;
    let bias = input_opt(env, node, 2)?;
    let v = f32s(x)?;
    let wv = f32s(w)?;
    let spatial = x.shape.len() - 2;
    let (n, c_in) = (x.shape[0], x.shape[1]);
    let (c_out, c_per_group) = (w.shape[0], w.shape[1]);
    let group = node.attr("group").map(|a| a.i()).transpose()?.unwrap_or(1) as usize;
    if c_per_group * group != c_in || c_out % group != 0 {
        return Err(Error::Model("Conv: group/channel mismatch".into()));
    }
    let kernel: Vec<usize> = w.shape[2..].to_vec();
    if kernel.len() != spatial {
        return Err(Error::Model("Conv: kernel rank mismatch".into()));
    }
    let strides: Vec<usize> = attr_ints(node, "strides", vec![1; spatial])?;
    let dilations: Vec<usize> = attr_ints(node, "dilations", vec![1; spatial])?;
    let pads: Vec<usize> = attr_ints(node, "pads", vec![0; 2 * spatial])?;
    if strides.len() != spatial || dilations.len() != spatial || pads.len() != 2 * spatial {
        return Err(Error::Model("Conv: attr length mismatch".into()));
    }
    let bv = match bias {
        Some(b) => f32s(b)?.to_vec(),
        None => vec![0f32; c_out],
    };
    if bv.len() != c_out {
        return Err(Error::Model("Conv: bias length mismatch".into()));
    }

    let mut out_spatial = vec![0usize; spatial];
    for d in 0..spatial {
        let eff = dilations[d] * (kernel[d] - 1) + 1;
        out_spatial[d] = (x.shape[d + 2] + pads[d] + pads[d + spatial] - eff) / strides[d] + 1;
    }
    let mut shape = vec![n, c_out];
    shape.extend_from_slice(&out_spatial);
    let out_numel: usize = shape.iter().product();
    let mut out = vec![0f32; out_numel];
    let out_inner: usize = out_spatial.iter().product();
    let x_inner: usize = x.shape[2..].iter().product();
    let w_inner: usize = kernel.iter().product();
    for b in 0..n {
        for g in 0..group {
            for oc in 0..(c_out / group) {
                let o_ch = g * (c_out / group) + oc;
                for od in 0..out_inner {
                    let mut o_idx = vec![0usize; spatial];
                    let mut rem = od;
                    for d in (0..spatial).rev() {
                        o_idx[d] = rem % out_spatial[d];
                        rem /= out_spatial[d];
                    }
                    let mut acc = bv[o_ch];
                    for ic in 0..c_per_group {
                        let i_ch = g * c_per_group + ic;
                        for kd in 0..w_inner {
                            let mut k_idx = vec![0usize; spatial];
                            let mut rem = kd;
                            for d in (0..spatial).rev() {
                                k_idx[d] = rem % kernel[d];
                                rem /= kernel[d];
                            }
                            let mut in_ok = true;
                            for d in 0..spatial {
                                let pos = o_idx[d] * strides[d] + k_idx[d] * dilations[d];
                                let pad_l = pads[d];
                                if pos < pad_l || pos >= pad_l + x.shape[d + 2] {
                                    in_ok = false;
                                    break;
                                }
                            }
                            if !in_ok {
                                continue;
                            }

                            let mut x_off = (b * c_in + i_ch) * x_inner;
                            let mut inner_stride = x_inner;
                            for d in 0..spatial {
                                inner_stride /= x.shape[d + 2];
                                let pos = o_idx[d] * strides[d] + k_idx[d] * dilations[d];
                                x_off += (pos - pads[d]) * inner_stride;
                            }
                            let w_off = (o_ch * c_per_group + ic) * w_inner + kd;
                            acc += v[x_off] * wv[w_off];
                        }
                    }
                    out[(b * c_out + o_ch) * out_inner + od] = acc;
                }
            }
        }
    }
    output(env, node, 0, Tensor::f32(out, shape))?;
    Ok(())
}

fn attr_ints(node: &Node, name: &str, default: Vec<usize>) -> Result<Vec<usize>> {
    match node.attr(name) {
        Some(a) => Ok(a.ints()?.iter().map(|&x| x as usize).collect()),
        None => Ok(default),
    }
}

fn pool(env: &mut HashMap<String, Tensor>, node: &Node, avg: bool) -> Result<()> {
    let x = input(env, node, 0)?;
    let v = f32s(x)?;
    let spatial = x.shape.len() - 2;
    let kernel: Vec<usize> = attr_ints(node, "kernel_shape", vec![])?;
    if kernel.len() != spatial {
        return Err(Error::Model(format!(
            "{}: kernel_shape mismatch",
            node.op_type
        )));
    }
    let strides: Vec<usize> = attr_ints(node, "strides", kernel.clone())?;
    let pads: Vec<usize> = attr_ints(node, "pads", vec![0; 2 * spatial])?;
    let ceil_mode = node
        .attr("ceil_mode")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(0)
        != 0;
    let count_include_pad = node
        .attr("count_include_pad")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(0)
        != 0;
    let (n, c) = (x.shape[0], x.shape[1]);
    let mut out_spatial = vec![0usize; spatial];
    for d in 0..spatial {
        let dim = x.shape[d + 2] + pads[d] + pads[d + spatial];
        out_spatial[d] = if ceil_mode {
            (dim as f64 / strides[d] as f64).ceil() as usize
        } else {
            (dim - kernel[d]) / strides[d] + 1
        };
    }
    let mut shape = vec![n, c];
    shape.extend_from_slice(&out_spatial);
    let out_inner: usize = out_spatial.iter().product();
    let x_inner: usize = x.shape[2..].iter().product();
    let k_inner: usize = kernel.iter().product();
    let mut out = vec![0f32; n * c * out_inner];
    for b in 0..n {
        for ch in 0..c {
            for od in 0..out_inner {
                let mut o_idx = vec![0usize; spatial];
                let mut rem = od;
                for d in (0..spatial).rev() {
                    o_idx[d] = rem % out_spatial[d];
                    rem /= out_spatial[d];
                }
                let mut acc = if avg { 0f32 } else { f32::NEG_INFINITY };
                let mut count = 0usize;
                for kd in 0..k_inner {
                    let mut k_idx = vec![0usize; spatial];
                    let mut rem = kd;
                    for d in (0..spatial).rev() {
                        k_idx[d] = rem % kernel[d];
                        rem /= kernel[d];
                    }
                    let mut in_ok = true;
                    let mut x_off = (b * c + ch) * x_inner;
                    let mut inner_stride = x_inner;
                    for d in 0..spatial {
                        inner_stride /= x.shape[d + 2];
                        let pos = o_idx[d] * strides[d] + k_idx[d];
                        if pos < pads[d] || pos >= pads[d] + x.shape[d + 2] {
                            in_ok = false;
                            break;
                        }
                        x_off += (pos - pads[d]) * inner_stride;
                    }
                    if in_ok {
                        let val = v[x_off];
                        acc = if avg { acc + val } else { acc.max(val) };
                        count += 1;
                    }
                }
                let denom = if avg {
                    if count_include_pad {
                        k_inner
                    } else {
                        count.max(1)
                    }
                } else {
                    1
                };
                let val = if avg { acc / denom as f32 } else { acc };
                out[(b * c + ch) * out_inner + od] = val;
            }
        }
    }
    output(env, node, 0, Tensor::f32(out, shape))?;
    Ok(())
}

pub(crate) fn avg_pool(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    pool(env, node, true)
}

pub(crate) fn max_pool(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    pool(env, node, false)
}

pub(crate) fn global_avg_pool(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let v = f32s(x)?;
    let spatial = x.shape.len() - 2;
    let inner: usize = x.shape[2..].iter().product();
    let (n, c) = (x.shape[0], x.shape[1]);
    let mut out = vec![0f32; n * c];
    for (i, slot) in out.iter_mut().enumerate() {
        let base = i * inner;
        let sum: f32 = v[base..base + inner].iter().sum();
        *slot = sum / inner as f32;
    }
    let mut shape = vec![n, c];
    shape.extend(std::iter::repeat_n(1, spatial));
    output(env, node, 0, Tensor::f32(out, shape))?;
    Ok(())
}

pub(crate) fn gru(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let w = input(env, node, 1)?;
    let r = input(env, node, 2)?;
    let b = input_opt(env, node, 3)?;
    let h0 = input_opt(env, node, 5)?;
    let xv = f32s(x)?;
    let wv = f32s(w)?;
    let rv = f32s(r)?;
    let hidden_size = node
        .attr("hidden_size")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(0) as usize;
    let direction = node
        .attr("direction")
        .map(|a| a.s())
        .transpose()?
        .map(|s| String::from_utf8_lossy(s).to_string())
        .unwrap_or_else(|| "forward".into());
    let linear_before = node
        .attr("linear_before_reset")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(0)
        != 0;
    if direction != "forward" {
        return Err(Error::Model(format!(
            "GRU: unsupported direction {direction:?} (only forward)"
        )));
    }
    let layout = node.attr("layout").map(|a| a.i()).transpose()?.unwrap_or(0);
    if layout != 0 {
        return Err(Error::Model("GRU: only layout 0 supported".into()));
    }
    let seq_len = x.shape[0];
    let batch = x.shape[1];
    let input_size = x.shape[2];
    if wv.len() != 3 * hidden_size * input_size || rv.len() != 3 * hidden_size * hidden_size {
        return Err(Error::Model("GRU: weight shape mismatch".into()));
    }
    let bv: Vec<f32> = match b {
        Some(b) => f32s(b)?.to_vec(),
        None => vec![0f32; 6 * hidden_size],
    };
    if bv.len() != 6 * hidden_size {
        return Err(Error::Model("GRU: bias shape mismatch".into()));
    }
    let mut h = match h0 {
        Some(h) => {
            let hv = f32s(h)?;
            if hv.len() != batch * hidden_size {
                return Err(Error::Model("GRU: initial hidden shape mismatch".into()));
            }
            hv.to_vec()
        }
        None => vec![0f32; batch * hidden_size],
    };
    let mut y = vec![0f32; seq_len * batch * hidden_size];
    let act = |z: f32, f: &str| -> Result<f32> {
        Ok(match f {
            "relu" => z.max(0.0),
            "sigmoid" => 1.0 / (1.0 + (-z).exp()),
            "tanh" => z.tanh(),
            "affine" => z,
            "leakyrelu" => z.max(0.01 * z),
            "hardSigmoid" => (0.2 * z + 0.5).clamp(0.0, 1.0),
            "scaledtanh" => 1.7159 * (2.0 / 3.0 * z).tanh(),
            "elu" => {
                if z > 0.0 {
                    z
                } else {
                    z.exp() - 1.0
                }
            }
            "softsign" => z / (1.0 + z.abs()),
            "softplus" => (1.0 + z.exp()).ln(),
            other => {
                return Err(Error::Model(format!(
                    "GRU: unsupported activation {other:?}"
                )));
            }
        })
    };
    let act_f = node
        .attr("activations")
        .map(|a| a.strings())
        .transpose()?
        .map(|v| {
            v.iter()
                .map(|s| String::from_utf8_lossy(s).to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["sigmoid".into(), "tanh".into()]);
    if act_f.len() < 2 {
        return Err(Error::Model("GRU: needs two activations".into()));
    }
    let (act_gate, act_cell) = (act_f[0].as_str(), act_f[1].as_str());
    for t in 0..seq_len {
        let mut h_next = vec![0f32; batch * hidden_size];
        for b_i in 0..batch {
            for i in 0..hidden_size {
                let mut z = bv[i] + bv[3 * hidden_size + i];
                let mut r = bv[hidden_size + i] + bv[4 * hidden_size + i];
                for j in 0..input_size {
                    let xv_ij = xv[t * batch * input_size + b_i * input_size + j];
                    z += wv[i * input_size + j] * xv_ij;
                    r += wv[(hidden_size + i) * input_size + j] * xv_ij;
                }
                for j in 0..hidden_size {
                    let h_ij = h[b_i * hidden_size + j];
                    z += wv[(2 * hidden_size + i) * hidden_size + j] * h_ij;
                    r += wv[(3 * hidden_size + i) * hidden_size + j] * h_ij;
                }
                let zt = act(z, act_gate)?;
                let rt = act(r, act_gate)?;

                let mut c = bv[2 * hidden_size + i];
                for j in 0..input_size {
                    let xv_ij = xv[t * batch * input_size + b_i * input_size + j];
                    c += wv[(2 * hidden_size + i) * input_size + j] * xv_ij;
                }
                let mut rh = 0f32;
                for j in 0..hidden_size {
                    rh += rv[(2 * hidden_size + i) * hidden_size + j] * h[b_i * hidden_size + j];
                }
                if linear_before {
                    c += rh * rt + bv[5 * hidden_size + i];
                } else {
                    c += (rh + bv[5 * hidden_size + i]) * rt;
                }
                let ct = act(c, act_cell)?;
                let h_t = zt * h[b_i * hidden_size + i] + (1.0 - zt) * ct;
                h_next[b_i * hidden_size + i] = h_t;
                y[t * batch * hidden_size + b_i * hidden_size + i] = h_t;
            }
        }
        h = h_next;
    }
    let y_shape = vec![seq_len, batch, hidden_size];
    let h_shape = vec![1, batch, hidden_size];
    output(env, node, 0, Tensor::f32(y, y_shape))?;
    output(env, node, 1, Tensor::f32(h, h_shape))?;
    Ok(())
}
