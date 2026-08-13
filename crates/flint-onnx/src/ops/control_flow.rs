use std::collections::HashMap;

use flint_error::{Error, Result};

use crate::graph::{Attr, Graph, Node};
use crate::ops::{input, input_opt, output};
use crate::tensor::{Data, Tensor};

fn run_subgraph(sub: &Graph, env: &mut HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
    for (name, t) in &sub.initializers {
        env.insert(name.clone(), t.clone());
    }
    for node in &sub.nodes {
        crate::ops::run(node, env)?;
    }
    let mut out = Vec::with_capacity(sub.outputs.len());
    for v in &sub.outputs {
        let t = env
            .get(&v.name)
            .ok_or_else(|| Error::Model(format!("subgraph output {:?} not produced", v.name)))?;
        out.push(t.clone());
    }
    Ok(out)
}

pub(crate) fn if_op(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let cond = input(env, node, 0)?;
    let cond_v = match &cond.data {
        Data::Bool(v) => v[0],
        Data::I64(v) => v[0] != 0,
        Data::F32(v) => v[0] != 0.0,
    };
    let sub = match node.attr("then_branch") {
        Some(Attr::G(g)) => g,
        _ => return Err(Error::Model("If: missing then_branch".into())),
    };
    let else_sub = match node.attr("else_branch") {
        Some(Attr::G(g)) => g,
        _ => return Err(Error::Model("If: missing else_branch".into())),
    };

    let mut sub_env = env.clone();
    let results = run_subgraph(if cond_v { sub } else { else_sub }, &mut sub_env)?;
    for (i, t) in results.into_iter().enumerate() {
        let name = node
            .outputs
            .get(i)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::Model(format!(
                    "If: missing output {i} (outputs {:?}, inputs {:?})",
                    node.outputs, node.inputs
                ))
            })?;
        env.insert(name.clone(), t);
    }
    Ok(())
}

pub(crate) fn loop_op(env: &mut HashMap<String, Tensor>, node: &Node) -> Result<()> {
    let body = match node.attr("body") {
        Some(Attr::G(g)) => g,
        _ => return Err(Error::Model("Loop: missing body".into())),
    };
    if body.inputs.len() < 2 {
        return Err(Error::Model("Loop: body needs iter/cond inputs".into()));
    }
    let num_carried = node.inputs.len().saturating_sub(2);
    let mut carried: Vec<Tensor> = (0..num_carried)
        .map(|i| input(env, node, 2 + i).cloned())
        .collect::<Result<Vec<_>>>()?;
    let num_scan = node.outputs.len().saturating_sub(1 + num_carried);
    let mut scans: Vec<Vec<Tensor>> = vec![vec![]; num_scan];

    let mut iter: i64 = 0;
    let mut cond: bool = match input_opt(env, node, 1)? {
        Some(c) => match &c.data {
            Data::Bool(v) => v[0],
            _ => return Err(Error::Model("Loop: cond must be bool".into())),
        },
        None => true,
    };
    let max_iter = match input_opt(env, node, 0)? {
        Some(m) => match &m.data {
            Data::I64(v) => v[0],
            _ => return Err(Error::Model("Loop: trip count must be i64".into())),
        },
        None => i64::MAX,
    };
    let iter_name = &body.inputs[0].name;
    let cond_name = &body.inputs[1].name;

    while iter < max_iter && cond {
        let mut sub_env = env.clone();
        sub_env.insert(iter_name.clone(), Tensor::scalar_i64(iter));
        sub_env.insert(cond_name.clone(), Tensor::bool(vec![cond], vec![]));
        for (i, t) in carried.iter().enumerate() {
            let name = body
                .inputs
                .get(2 + i)
                .map(|v| v.name.clone())
                .ok_or_else(|| Error::Model("Loop: body carried input missing".into()))?;
            sub_env.insert(name, t.clone());
        }
        let results = run_subgraph(body, &mut sub_env)?;
        if results.is_empty() {
            return Err(Error::Model("Loop: body produced no outputs".into()));
        }
        cond = match &results[0].data {
            Data::Bool(v) => v[0],
            _ => return Err(Error::Model("Loop: body cond must be bool".into())),
        };
        for (i, t) in results.iter().skip(1).take(num_carried).enumerate() {
            carried[i] = t.clone();
        }
        for (i, t) in results.iter().skip(1 + num_carried).enumerate() {
            if i < num_scan {
                scans[i].push(t.clone());
            }
        }
        iter += 1;
    }

    output(env, node, 0, Tensor::bool(vec![cond], vec![]))?;
    for (i, t) in carried.iter().enumerate() {
        output(env, node, 1 + i, t.clone())?;
    }
    for (i, scan) in scans.iter().enumerate() {
        let out_idx = 1 + num_carried + i;
        if scan.is_empty() {
            let t = Tensor::f32(vec![], vec![0]);
            output(env, node, out_idx, t)?;
            continue;
        }
        let first = &scan[0];
        let mut shape = vec![scan.len()];
        shape.extend_from_slice(&first.shape);
        let data = match &first.data {
            Data::F32(_) => {
                let mut v = Vec::with_capacity(scan.len() * first.len());
                for t in scan {
                    if let Data::F32(d) = &t.data {
                        v.extend_from_slice(d);
                    }
                }
                Data::F32(v)
            }
            Data::I64(_) => {
                let mut v = Vec::with_capacity(scan.len() * first.len());
                for t in scan {
                    if let Data::I64(d) = &t.data {
                        v.extend_from_slice(d);
                    }
                }
                Data::I64(v)
            }
            Data::Bool(_) => {
                let mut v = Vec::with_capacity(scan.len() * first.len());
                for t in scan {
                    if let Data::Bool(d) = &t.data {
                        v.extend_from_slice(d);
                    }
                }
                Data::Bool(v)
            }
        };
        output(env, node, out_idx, Tensor { data, shape })?;
    }
    Ok(())
}
