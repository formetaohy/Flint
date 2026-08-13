use std::collections::HashMap;

use flint_error::{Error, Result};

use crate::graph::Node;
use crate::ops::{axes_attr, f32s, input, input_opt, norm_axis, output};
use crate::tensor::{Data, Tensor, broadcast_shape, broadcast_to};

pub(crate) fn arith<FF, IF>(
    env: &mut HashMap<String, Tensor>,
    node: &Node,
    ff: FF,
    fi: IF,
) -> Result<()>
where
    FF: Fn(f32, f32) -> f32,
    IF: Fn(i64, i64) -> i64,
{
    let a = input(env, node, 0)?;
    let b = input(env, node, 1)?;
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    match (&a.data, &b.data) {
        (Data::F32(x), Data::F32(y)) => {
            let x = broadcast_to(x, &a.shape, &shape)?;
            let y = broadcast_to(y, &b.shape, &shape)?;
            output(
                env,
                node,
                0,
                Tensor::f32(x.iter().zip(&y).map(|(&x, &y)| ff(x, y)).collect(), shape),
            )?;
        }
        (Data::I64(x), Data::I64(y)) => {
            let x = broadcast_to(x, &a.shape, &shape)?;
            let y = broadcast_to(y, &b.shape, &shape)?;
            output(
                env,
                node,
                0,
                Tensor::i64(x.iter().zip(&y).map(|(&x, &y)| fi(x, y)).collect(), shape),
            )?;
        }
        (x, y) => {
            return Err(Error::Model(format!(
                "{}: mixed or unsupported dtypes {:?} / {:?}",
                node.op_type, x, y
            )));
        }
    }
    Ok(())
}

pub(crate) fn variadic<FF, IF>(
    env: &mut HashMap<String, Tensor>,
    node: &Node,
    f0: f32,
    ff: FF,
    i0: i64,
    fi: IF,
) -> Result<()>
where
    FF: Fn(f32, f32) -> f32,
    IF: Fn(i64, i64) -> i64,
{
    let mut shape: Vec<usize> = vec![];
    let mut f32_mode = true;
    for i in 0..node.inputs.len() {
        let t = input(env, node, i)?;
        let is_f32 = matches!(t.data, Data::F32(_));
        if i == 0 {
            shape = t.shape.clone();
            f32_mode = is_f32;
        } else {
            shape = broadcast_shape(&shape, &t.shape)?;
            if f32_mode != is_f32 {
                return Err(Error::Model(format!("{}: mixed dtypes", node.op_type)));
            }
        }
    }
    let n = shape.iter().product::<usize>();
    let mut acc_f = vec![f0; n];
    let mut acc_i = vec![i0; n];
    for i in 0..node.inputs.len() {
        let t = input(env, node, i)?;
        match &t.data {
            Data::F32(v) => {
                let v = broadcast_to(v, &t.shape, &shape)?;
                for (a, b) in acc_f.iter_mut().zip(v) {
                    *a = ff(*a, b);
                }
            }
            Data::I64(v) => {
                let v = broadcast_to(v, &t.shape, &shape)?;
                for (a, b) in acc_i.iter_mut().zip(v) {
                    *a = fi(*a, b);
                }
            }
            _ => return Err(Error::Model(format!("{}: unsupported dtype", node.op_type))),
        }
    }
    if node.op_type == "Mean" {
        if !f32_mode {
            return Err(Error::Model("Mean on non-f32 inputs".into()));
        }
        let n = node.inputs.len() as f32;
        output(
            env,
            node,
            0,
            Tensor::f32(acc_f.into_iter().map(|x| x / n).collect(), shape),
        )?;
    } else if f32_mode {
        output(env, node, 0, Tensor::f32(acc_f, shape))?;
    } else {
        output(env, node, 0, Tensor::i64(acc_i, shape))?;
    }
    Ok(())
}

pub(crate) fn logic<BF>(env: &mut HashMap<String, Tensor>, node: &Node, f: BF) -> Result<()>
where
    BF: Fn(bool, bool) -> bool,
{
    let a = input(env, node, 0)?;
    let b = input(env, node, 1)?;
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let (Data::Bool(x), Data::Bool(y)) = (&a.data, &b.data) else {
        return Err(Error::Model(format!(
            "{}: expected bool inputs",
            node.op_type
        )));
    };
    let x = broadcast_to(x, &a.shape, &shape)?;
    let y = broadcast_to(y, &b.shape, &shape)?;
    output(
        env,
        node,
        0,
        Tensor::bool(x.iter().zip(&y).map(|(&x, &y)| f(x, y)).collect(), shape),
    )?;
    Ok(())
}

pub(crate) fn not(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let a = input(env, node, 0)?;
    let Data::Bool(x) = &a.data else {
        return Err(Error::Model("Not: expected bool input".into()));
    };
    output(
        env,
        node,
        0,
        Tensor::bool(x.iter().map(|&b| !b).collect(), a.shape.clone()),
    )?;
    Ok(())
}

pub(crate) fn cmp<FF, IF>(
    env: &mut HashMap<String, Tensor>,
    node: &Node,
    ff: FF,
    fi: IF,
) -> Result<()>
where
    FF: Fn(f32, f32) -> bool,
    IF: Fn(i64, i64) -> bool,
{
    let a = input(env, node, 0)?;
    let b = input(env, node, 1)?;
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let out = match (&a.data, &b.data) {
        (Data::F32(x), Data::F32(y)) => {
            let x = broadcast_to(x, &a.shape, &shape)?;
            let y = broadcast_to(y, &b.shape, &shape)?;
            x.iter().zip(&y).map(|(&x, &y)| ff(x, y)).collect()
        }
        (Data::I64(x), Data::I64(y)) => {
            let x = broadcast_to(x, &a.shape, &shape)?;
            let y = broadcast_to(y, &b.shape, &shape)?;
            x.iter().zip(&y).map(|(&x, &y)| fi(x, y)).collect()
        }
        (x, y) => {
            return Err(Error::Model(format!(
                "{}: mixed dtypes {:?} / {:?}",
                node.op_type, x, y
            )));
        }
    };
    output(env, node, 0, Tensor::bool(out, shape))?;
    Ok(())
}

pub(crate) fn where3(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let c = input(env, node, 0)?;
    let x = input(env, node, 1)?;
    let y = input(env, node, 2)?;
    let Data::Bool(cv) = &c.data else {
        return Err(Error::Model("Where: condition must be bool".into()));
    };
    let shape = broadcast_shape(&broadcast_shape(&c.shape, &x.shape)?, &y.shape)?;
    let c = broadcast_to(cv, &c.shape, &shape)?;
    let out = match (&x.data, &y.data) {
        (Data::F32(xv), Data::F32(yv)) => {
            let x = broadcast_to(xv, &x.shape, &shape)?;
            let y = broadcast_to(yv, &y.shape, &shape)?;
            Data::F32(
                c.iter()
                    .zip(x)
                    .zip(y)
                    .map(|((&c, x), y)| if c { x } else { y })
                    .collect(),
            )
        }
        (Data::I64(xv), Data::I64(yv)) => {
            let x = broadcast_to(xv, &x.shape, &shape)?;
            let y = broadcast_to(yv, &y.shape, &shape)?;
            Data::I64(
                c.iter()
                    .zip(x)
                    .zip(y)
                    .map(|((&c, x), y)| if c { x } else { y })
                    .collect(),
            )
        }
        (Data::Bool(xv), Data::Bool(yv)) => {
            let x = broadcast_to(xv, &x.shape, &shape)?;
            let y = broadcast_to(yv, &y.shape, &shape)?;
            Data::Bool(
                c.iter()
                    .zip(x)
                    .zip(y)
                    .map(|((&c, x), y)| if c { x } else { y })
                    .collect(),
            )
        }
        (x, y) => {
            return Err(Error::Model(format!(
                "Where: mixed dtypes {:?} / {:?}",
                x, y
            )));
        }
    };
    output(env, node, 0, Tensor { data: out, shape })?;
    Ok(())
}

pub(crate) fn unary_arith<FF, IF>(
    env: &mut HashMap<String, Tensor>,
    node: &Node,
    ff: FF,
    fi: IF,
) -> Result<()>
where
    FF: Fn(f32) -> f32,
    IF: Fn(i64) -> i64,
{
    let a = input(env, node, 0)?;
    match &a.data {
        Data::F32(v) => output(
            env,
            node,
            0,
            Tensor::f32(v.iter().map(|&x| ff(x)).collect(), a.shape.clone()),
        )?,
        Data::I64(v) => output(
            env,
            node,
            0,
            Tensor::i64(v.iter().map(|&x| fi(x)).collect(), a.shape.clone()),
        )?,
        other => {
            return Err(Error::Model(format!(
                "{}: unsupported dtype {other:?}",
                node.op_type
            )));
        }
    }
    Ok(())
}

pub(crate) fn unary_f32<F>(env: &mut HashMap<String, Tensor>, node: &Node, f: F) -> Result<()>
where
    F: Fn(f32) -> f32,
{
    let a = input(env, node, 0)?;
    let v = f32s(a)?;
    output(
        env,
        node,
        0,
        Tensor::f32(v.iter().map(|&x| f(x)).collect(), a.shape.clone()),
    )?;
    Ok(())
}

pub(crate) fn leaky_relu(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let alpha = node
        .attr("alpha")
        .map(|a| a.f())
        .transpose()?
        .unwrap_or(0.01);
    unary_f32(env, node, |x| if x >= 0.0 { x } else { alpha * x })
}

pub(crate) fn clip(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let min = match input_opt(env, node, 1)? {
        Some(t) => Some(f32s(t)?[0]),
        None => node.attr("min").map(|m| m.f()).transpose()?,
    };
    let max = match input_opt(env, node, 2)? {
        Some(t) => Some(f32s(t)?[0]),
        None => node.attr("max").map(|m| m.f()).transpose()?,
    };
    unary_f32(env, node, |x| {
        let x = match min {
            Some(m) => x.max(m),
            None => x,
        };
        match max {
            Some(m) => x.min(m),
            None => x,
        }
    })
}

pub(crate) fn hard_sigmoid(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let alpha = node
        .attr("alpha")
        .map(|a| a.f())
        .transpose()?
        .unwrap_or(0.2);
    let beta = node.attr("beta").map(|a| a.f()).transpose()?.unwrap_or(0.5);
    unary_f32(env, node, |x| (alpha * x + beta).clamp(0.0, 1.0))
}

pub(crate) fn prelu(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let x = input(env, node, 0)?;
    let slope = input(env, node, 1)?;
    let v = f32s(x)?;
    let shape = broadcast_shape(&x.shape, &slope.shape)?;
    let s = slope.broadcast_to_f32(&shape)?;
    let v = broadcast_to(v, &x.shape, &shape)?;
    output(
        env,
        node,
        0,
        Tensor::f32(
            v.iter()
                .zip(&s)
                .map(|(&x, &a)| if x >= 0.0 { x } else { a * x })
                .collect(),
            shape,
        ),
    )?;
    Ok(())
}

pub(crate) fn selu(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let alpha = node
        .attr("alpha")
        .map(|a| a.f())
        .transpose()?
        .unwrap_or(1.673_263_2);
    let gamma = node
        .attr("gamma")
        .map(|a| a.f())
        .transpose()?
        .unwrap_or(1.050_701);
    unary_f32(env, node, |x| {
        if x > 0.0 {
            gamma * x
        } else {
            gamma * alpha * (x.exp() - 1.0)
        }
    })
}

pub(crate) fn reduce<C, F>(
    env: &mut HashMap<String, Tensor>,
    node: &Node,
    seed: f32,
    combine: C,
    finish: F,
) -> Result<()>
where
    C: Fn(f32, f32) -> f32,
    F: Fn(usize, f32) -> f32,
{
    let x = input(env, node, 0)?;
    let v = f32s(x)?;
    let rank = x.shape.len();
    let keepdims = node
        .attr("keepdims")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(1)
        != 0;
    let noop = node
        .attr("noop_with_empty_axes")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(0)
        != 0;
    let axes = axes_attr(node, rank)?;
    if axes.is_none() && noop {
        output(env, node, 0, x.clone())?;
        return Ok(());
    }
    let axes = axes.unwrap_or_else(|| (0..rank).collect());
    let out = reduce_impl(v, &x.shape, &axes, seed, &combine, &finish);
    let shape = if keepdims {
        x.shape
            .iter()
            .enumerate()
            .map(|(i, d)| if axes.contains(&i) { 1 } else { *d })
            .collect()
    } else {
        x.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, d)| *d)
            .collect()
    };
    output(env, node, 0, Tensor::f32(out, shape))?;
    Ok(())
}

fn reduce_impl<C, F>(
    v: &[f32],
    shape: &[usize],
    axes: &[usize],
    seed: f32,
    combine: &C,
    finish: &F,
) -> Vec<f32>
where
    C: Fn(f32, f32) -> f32,
    F: Fn(usize, f32) -> f32,
{
    let mut order: Vec<usize> = (0..shape.len()).filter(|d| !axes.contains(d)).collect();
    for a in axes {
        order.push(*a);
    }
    let perm_shape: Vec<usize> = order.iter().map(|&i| shape[i]).collect();
    let kept: usize = perm_shape[..perm_shape.len() - axes.len()]
        .iter()
        .product::<usize>()
        .max(1);
    let reduced: usize = perm_shape[perm_shape.len() - axes.len()..]
        .iter()
        .product::<usize>()
        .max(1);
    let mut out = vec![seed; kept];
    for (o, slot) in out.iter_mut().enumerate() {
        for r in 0..reduced {
            let mut perm_idx = vec![0usize; shape.len()];
            let mut rem = o * reduced + r;
            for (k, d) in perm_shape.iter().enumerate().rev() {
                perm_idx[k] = rem % d;
                rem /= d;
            }
            let mut flat = 0usize;
            for (k, &src) in order.iter().enumerate() {
                flat = flat * shape[src] + perm_idx[k];
            }
            *slot = combine(*slot, v[flat]);
        }
    }
    for o in out.iter_mut() {
        *o = finish(reduced, *o);
    }
    out
}

pub(crate) fn argmax(env: &mut HashMap<String, Tensor>, node: &Node, largest: bool) -> Result<()> {
    let x = input(env, node, 0)?;
    let rank = x.shape.len();
    let axis = norm_axis(
        node.attr("axis").map(|a| a.i()).transpose()?.unwrap_or(0),
        rank,
    )?;
    let keepdims = node
        .attr("keepdims")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(1)
        != 0;
    let select_last = node
        .attr("select_last_index")
        .map(|a| a.i())
        .transpose()?
        .unwrap_or(0)
        != 0;

    let v: Vec<f64> = x.data.as_f32().into_iter().map(f64::from).collect();
    let outer: usize = x.shape[..axis].iter().product();
    let dim = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut best = 0usize;
            let mut best_v = if largest {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            for d in 0..dim {
                let val = v[o * dim * inner + d * inner + i];
                let better = if largest { val > best_v } else { val < best_v };
                if better || (select_last && val == best_v) {
                    best = d;
                    best_v = val;
                }
            }
            out.push(best as i64);
        }
    }
    let mut shape: Vec<usize> = x.shape.clone();
    shape.remove(axis);
    if keepdims {
        shape.insert(axis, 1);
    }
    output(env, node, 0, Tensor::i64(out, shape))?;
    Ok(())
}

pub(crate) fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}
