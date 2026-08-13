mod control_flow;
mod layout;
mod math;
mod nn;
mod quant;

use std::collections::HashMap;

use flint_error::{Error, Result};

use crate::graph::Node;
use crate::tensor::{Data, Tensor};

pub(crate) fn run(node: &Node, env: &mut HashMap<String, Tensor>) -> Result<()> {
    match node.op_type.as_str() {
        "Add" => math::arith(env, node, |a, b| a + b, |a, b| a + b),
        "Sub" => math::arith(env, node, |a, b| a - b, |a, b| a - b),
        "Mul" => math::arith(env, node, |a, b| a * b, |a, b| a * b),
        "Div" => math::arith(env, node, |a, b| a / b, |a, b| a / b),
        "Pow" => math::arith(
            env,
            node,
            |a, b| a.powf(b),
            |a, b| a.saturating_pow(b as u32),
        ),
        "Mod" => math::arith(env, node, |a, b| a % b, |a, b| a % b),
        "Min" => math::arith(env, node, f32::min, i64::min),
        "Max" => math::arith(env, node, f32::max, i64::max),
        "Sum" => math::variadic(env, node, 0.0, |a, b| a + b, 0i64, |a, b| a + b),
        "Mean" => math::variadic(env, node, 0.0, |a, b| a + b, 0i64, |a, b| a + b),
        "And" => math::logic(env, node, |a, b| a && b),
        "Or" => math::logic(env, node, |a, b| a || b),
        "Xor" => math::logic(env, node, |a, b| a ^ b),
        "Not" => math::not(env, node),
        "Equal" => math::cmp(env, node, |a, b| a == b, |a, b| a == b),
        "Greater" => math::cmp(env, node, |a, b| a > b, |a, b| a > b),
        "GreaterOrEqual" => math::cmp(env, node, |a, b| a >= b, |a, b| a >= b),
        "Less" => math::cmp(env, node, |a, b| a < b, |a, b| a < b),
        "LessOrEqual" => math::cmp(env, node, |a, b| a <= b, |a, b| a <= b),
        "Where" => math::where3(env, node),

        "Abs" => math::unary_arith(env, node, f32::abs, i64::abs),
        "Neg" => math::unary_arith(env, node, |x| -x, |x| -x),
        "Sign" => math::unary_arith(env, node, |x| x.signum(), |x| x.signum()),
        "Floor" => math::unary_arith(env, node, f32::floor, |x| x),
        "Ceil" => math::unary_arith(env, node, f32::ceil, |x| x),
        "Exp" => math::unary_f32(env, node, f32::exp),
        "Log" => math::unary_f32(env, node, f32::ln),
        "Sqrt" => math::unary_f32(env, node, f32::sqrt),
        "Erf" => math::unary_f32(env, node, math::erf),
        "Relu" => math::unary_f32(env, node, |x| x.max(0.0)),
        "LeakyRelu" => math::leaky_relu(env, node),
        "Sigmoid" => math::unary_f32(env, node, |x| 1.0 / (1.0 + (-x).exp())),
        "Tanh" => math::unary_f32(env, node, f32::tanh),
        "Sin" => math::unary_f32(env, node, f32::sin),
        "Cos" => math::unary_f32(env, node, f32::cos),
        "Reciprocal" => math::unary_f32(env, node, |x| 1.0 / x),
        "Clip" => math::clip(env, node),
        "HardSigmoid" => math::hard_sigmoid(env, node),
        "PRelu" => math::prelu(env, node),
        "Selu" => math::selu(env, node),

        "ReduceMean" => math::reduce(env, node, 0.0, |acc, x| acc + x, |n, s| s / n as f32),
        "ReduceSum" => math::reduce(env, node, 0.0, |acc, x| acc + x, |_, s| s),
        "ReduceMax" => math::reduce(env, node, f32::NEG_INFINITY, |a, b| a.max(b), |_, s| s),
        "ReduceMin" => math::reduce(env, node, f32::INFINITY, |a, b| a.min(b), |_, s| s),
        "ReduceProd" => math::reduce(env, node, 1.0, |acc, x| acc * x, |_, s| s),
        "ReduceL2" => math::reduce(env, node, 0.0, |acc, x| acc + x * x, |_, s| s.sqrt()),
        "ArgMax" => math::argmax(env, node, true),
        "ArgMin" => math::argmax(env, node, false),

        "Shape" => layout::shape(env, node),
        "Size" => layout::size(env, node),
        "Reshape" => layout::reshape(env, node),
        "Transpose" => layout::transpose(env, node),
        "Concat" => layout::concat(env, node),
        "Split" => layout::split(env, node),
        "Slice" => layout::slice(env, node),
        "Gather" => layout::gather(env, node),
        "GatherElements" => layout::gather_elements(env, node),
        "GatherND" => layout::gather_nd(env, node),
        "Squeeze" => layout::squeeze(env, node),
        "Unsqueeze" => layout::unsqueeze(env, node),
        "Flatten" => layout::flatten(env, node),
        "Expand" => layout::expand(env, node),
        "Tile" => layout::tile(env, node),
        "ConstantOfShape" => layout::constant_of_shape(env, node),
        "Range" => layout::range(env, node),
        "Identity" => layout::identity(env, node),
        "Constant" => layout::constant(env, node),
        "Cast" => layout::cast(env, node),
        "CastLike" => layout::cast_like(env, node),
        "Pad" => layout::pad(env, node),
        "TopK" => layout::topk(env, node),
        "NonZero" => layout::nonzero(env, node),
        "ScatterND" => layout::scatter_nd(env, node),
        "ScatterElements" => layout::scatter_elements(env, node),

        "MatMul" => nn::matmul(env, node),
        "Gemm" => nn::gemm(env, node),
        "Softmax" => nn::softmax(env, node),
        "Softplus" => nn::softplus(env, node),
        "LayerNormalization" => nn::layer_norm(env, node),
        "BatchNormalization" => nn::batch_norm(env, node),
        "Conv" => nn::conv(env, node),
        "AveragePool" => nn::avg_pool(env, node),
        "GlobalAveragePool" => nn::global_avg_pool(env, node),
        "MaxPool" => nn::max_pool(env, node),
        "GRU" => nn::gru(env, node),

        "DynamicQuantizeLinear" => quant::dynamic_quantize_linear(env, node),
        "DequantizeLinear" => quant::dequantize_linear(env, node),
        "MatMulInteger" => quant::matmul_integer(env, node),

        "If" => control_flow::if_op(env, node),
        "Loop" => control_flow::loop_op(env, node),

        other => Err(Error::Model(format!("unsupported operator {other:?}"))),
    }
}

pub(crate) fn input<'a>(
    env: &'a HashMap<String, Tensor>,
    node: &Node,
    i: usize,
) -> Result<&'a Tensor> {
    let name = node
        .inputs
        .get(i)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::Model(format!("{}: missing input {i}", node.op_type)))?;
    env.get(name)
        .ok_or_else(|| Error::Model(format!("{}: input {name:?} was not produced", node.op_type)))
}

pub(crate) fn input_opt<'a>(
    env: &'a HashMap<String, Tensor>,
    node: &Node,
    i: usize,
) -> Result<Option<&'a Tensor>> {
    match node.inputs.get(i).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(name) => env
            .get(name)
            .map(Some)
            .ok_or_else(|| Error::Model(format!("{}: input {name:?} missing", node.op_type))),
    }
}

pub(crate) fn output(
    env: &mut HashMap<String, Tensor>,
    node: &Node,
    i: usize,
    t: Tensor,
) -> Result<()> {
    let name = node
        .outputs
        .get(i)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::Model(format!("{}: missing output {i}", node.op_type)))?;
    env.insert(name.clone(), t);
    Ok(())
}

pub(crate) fn f32s(t: &Tensor) -> Result<&[f32]> {
    match &t.data {
        Data::F32(v) => Ok(v),
        other => Err(Error::Model(format!("expected f32 tensor, got {other:?}"))),
    }
}

pub(crate) fn i64s(t: &Tensor) -> Result<&[i64]> {
    match &t.data {
        Data::I64(v) => Ok(v),
        other => Err(Error::Model(format!("expected i64 tensor, got {other:?}"))),
    }
}

pub(crate) fn norm_axis(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a >= rank as i64 {
        return Err(Error::Model(format!(
            "axis {axis} out of range for rank {rank}"
        )));
    }
    Ok(a as usize)
}

pub(crate) fn axes_attr(node: &Node, rank: usize) -> Result<Option<Vec<usize>>> {
    let Some(attr) = node.attr("axes") else {
        return Ok(None);
    };
    let axes = attr.ints()?;
    Ok(Some(
        axes.iter()
            .map(|&a| norm_axis(a, rank))
            .collect::<Result<_>>()?,
    ))
}
