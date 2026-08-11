use std::sync::atomic::{AtomicU32, Ordering};

use flint_onnx::onnx;
use flint_onnx::tensor::Data;
use flint_onnx::{Session, Tensor};

static NEXT_FILE: AtomicU32 = AtomicU32::new(0);

fn one_op(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<onnx::AttributeProto>,
    inits: Vec<onnx::TensorProto>,
    graph_inputs: &[&str],
) -> Vec<u8> {
    let mut node = onnx::NodeProto::default();
    node.op_type = op_type.to_string();
    node.input = inputs.iter().map(|s| s.to_string()).collect();
    node.output = outputs.iter().map(|s| s.to_string()).collect();
    node.attribute = attrs;
    let mut g = onnx::GraphProto::default();
    g.node.push(node);
    g.initializer = inits;
    g.input = graph_inputs
        .iter()
        .map(|n| {
            let mut v = onnx::ValueInfoProto::default();
            v.name = n.to_string();
            v
        })
        .collect();
    g.output = outputs
        .iter()
        .map(|n| {
            let mut v = onnx::ValueInfoProto::default();
            v.name = n.to_string();
            v
        })
        .collect();
    let mut m = onnx::ModelProto::default();
    m.ir_version = 9;
    m.graph = Some(g);
    let mut opset = onnx::OperatorSetIdProto::default();
    opset.version = 17;
    m.opset_import.push(opset);
    let mut buf = Vec::new();
    prost::Message::encode(&m, &mut buf).unwrap();
    buf
}

fn attr_int(name: &str, v: i64) -> onnx::AttributeProto {
    let mut a = onnx::AttributeProto::default();
    a.name = name.to_string();
    a.r#type = onnx::attribute_proto::AttributeType::Int as i32;
    a.i = v;
    a
}

fn attr_ints(name: &str, v: &[i64]) -> onnx::AttributeProto {
    let mut a = onnx::AttributeProto::default();
    a.name = name.to_string();
    a.r#type = onnx::attribute_proto::AttributeType::Ints as i32;
    a.ints = v.to_vec();
    a
}

fn init_f32(name: &str, dims: &[i64], data: &[f32]) -> onnx::TensorProto {
    let mut t = onnx::TensorProto::default();
    t.name = name.to_string();
    t.dims = dims.to_vec();
    t.data_type = 1;
    t.float_data = data.to_vec();
    t
}

fn init_i64(name: &str, dims: &[i64], data: &[i64]) -> onnx::TensorProto {
    let mut t = onnx::TensorProto::default();
    t.name = name.to_string();
    t.dims = dims.to_vec();
    t.data_type = 7;
    t.int64_data = data.to_vec();
    t
}

fn run_bytes(
    bytes: &[u8],
    inputs: &[(&str, Tensor)],
) -> (Session, std::collections::HashMap<String, Tensor>) {
    let n = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("flint_onnx_test_{}_{}.onnx", std::process::id(), n));
    std::fs::write(&path, bytes).unwrap();
    let mut s = Session::load(&path).unwrap();
    for (n, t) in inputs {
        s.set_input(n, t.clone()).unwrap();
    }
    let out = s.run().unwrap();
    (s, out)
}

fn f32s(t: &Tensor) -> &[f32] {
    match &t.data {
        Data::F32(v) => v,
        other => panic!("expected f32, got {other:?}"),
    }
}

#[test]
fn add_broadcast() {
    let bytes = one_op("Add", &["a", "b"], &["y"], vec![], vec![], &["a", "b"]);
    let (_, out) = run_bytes(
        &bytes,
        &[
            ("a", Tensor::f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])),
            ("b", Tensor::f32(vec![10.0, 20.0], vec![1, 2])),
        ],
    );
    assert_eq!(f32s(&out["y"]), &[11.0, 22.0, 13.0, 24.0]);
}

#[test]
fn matmul_2d_and_batched() {
    let bytes = one_op("MatMul", &["a", "b"], &["y"], vec![], vec![], &["a", "b"]);
    let (_, out) = run_bytes(
        &bytes,
        &[
            (
                "a",
                Tensor::f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            ),
            (
                "b",
                Tensor::f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
            ),
        ],
    );

    assert_eq!(f32s(&out["y"]), &[58.0, 64.0, 139.0, 154.0]);

    let (_, out) = run_bytes(
        &bytes,
        &[
            (
                "a",
                Tensor::f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3]),
            ),
            (
                "b",
                Tensor::f32(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![2, 3, 1]),
            ),
        ],
    );
    assert_eq!(f32s(&out["y"]), &[6.0, 30.0]);
}

#[test]
fn transpose_4d() {
    let bytes = one_op(
        "Transpose",
        &["x"],
        &["y"],
        vec![attr_ints("perm", &[0, 2, 1, 3])],
        vec![],
        &["x"],
    );

    let mut x = Vec::with_capacity(24);
    for a in 0..1 {
        for b in 0..2 {
            for c in 0..3 {
                for d in 0..4 {
                    x.push((a * 1000 + b * 100 + c * 10 + d) as f32);
                }
            }
        }
    }
    let (_, out) = run_bytes(&bytes, &[("x", Tensor::f32(x, vec![1, 2, 3, 4]))]);
    let y = f32s(&out["y"]);
    assert_eq!(out["y"].shape, vec![1, 3, 2, 4]);

    for i0 in 0..1 {
        for i1 in 0..3 {
            for i2 in 0..2 {
                for i3 in 0..4 {
                    let want = (i0 * 1000 + i2 * 100 + i1 * 10 + i3) as f32;
                    assert_eq!(y[((i0 * 3 + i1) * 2 + i2) * 4 + i3], want);
                }
            }
        }
    }
}

#[test]
fn softmax_axis() {
    let bytes = one_op(
        "Softmax",
        &["x"],
        &["y"],
        vec![attr_int("axis", 1)],
        vec![],
        &["x"],
    );

    let (_, out) = run_bytes(
        &bytes,
        &[("x", Tensor::f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))],
    );
    let y = f32s(&out["y"]);
    let e = |v: f32| v.exp();
    let s0 = e(1.0) + e(2.0);
    let s1 = e(3.0) + e(4.0);
    assert!((y[0] - e(1.0) / s0).abs() < 1e-6);
    assert!((y[1] - e(2.0) / s0).abs() < 1e-6);
    assert!((y[2] - e(3.0) / s1).abs() < 1e-6);
    assert!((y[3] - e(4.0) / s1).abs() < 1e-6);
}

#[test]
fn layernorm_decomposition() {
    let inits = vec![
        init_f32("w", &[3], &[1.0, 2.0, 3.0]),
        init_f32("b", &[3], &[0.1, 0.2, 0.3]),
        init_f32("eps", &[], &[1e-5]),
        init_f32("two", &[], &[2.0]),
    ];
    let mut n1 = onnx::NodeProto::default();
    n1.op_type = "ReduceMean".into();
    n1.input = vec!["x".into()];
    n1.output = vec!["mean".into()];
    n1.attribute.push(attr_ints("axes", &[-1]));
    let mut n2 = onnx::NodeProto::default();
    n2.op_type = "Sub".into();
    n2.input = vec!["x".into(), "mean".into()];
    n2.output = vec!["centered".into()];
    let mut n3 = onnx::NodeProto::default();
    n3.op_type = "Pow".into();
    n3.input = vec!["centered".into(), "two".into()];
    n3.output = vec!["sq".into()];
    let mut n4 = onnx::NodeProto::default();
    n4.op_type = "ReduceMean".into();
    n4.input = vec!["sq".into()];
    n4.output = vec!["var".into()];
    n4.attribute.push(attr_ints("axes", &[-1]));
    let mut n5 = onnx::NodeProto::default();
    n5.op_type = "Add".into();
    n5.input = vec!["var".into(), "eps".into()];
    n5.output = vec!["var_eps".into()];
    let mut n6 = onnx::NodeProto::default();
    n6.op_type = "Sqrt".into();
    n6.input = vec!["var_eps".into()];
    n6.output = vec!["std".into()];
    let mut n7 = onnx::NodeProto::default();
    n7.op_type = "Div".into();
    n7.input = vec!["centered".into(), "std".into()];
    n7.output = vec!["norm".into()];
    let mut n8 = onnx::NodeProto::default();
    n8.op_type = "Mul".into();
    n8.input = vec!["norm".into(), "w".into()];
    n8.output = vec!["scaled".into()];
    let mut n9 = onnx::NodeProto::default();
    n9.op_type = "Add".into();
    n9.input = vec!["scaled".into(), "b".into()];
    n9.output = vec!["y".into()];

    let mut g = onnx::GraphProto::default();
    g.node = vec![n1, n2, n3, n4, n5, n6, n7, n8, n9];
    g.initializer = inits;
    {
        let name = "x";
        let mut v = onnx::ValueInfoProto::default();
        v.name = name.into();
        g.input.push(v);
    }
    {
        let name = "y";
        let mut v = onnx::ValueInfoProto::default();
        v.name = name.into();
        g.output.push(v);
    }
    let mut m = onnx::ModelProto::default();
    m.ir_version = 9;
    m.graph = Some(g);
    let mut opset = onnx::OperatorSetIdProto::default();
    opset.version = 17;
    m.opset_import.push(opset);
    let mut bytes = Vec::new();
    prost::Message::encode(&m, &mut bytes).unwrap();

    let x = Tensor::f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let (_, out) = run_bytes(&bytes, &[("x", x.clone())]);
    let y = f32s(&out["y"]);

    let xv: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (w, b) = ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]);
    for row in 0..2 {
        let m = (xv[row * 3] + xv[row * 3 + 1] + xv[row * 3 + 2]) / 3.0;
        let v = ((xv[row * 3] - m).powi(2)
            + (xv[row * 3 + 1] - m).powi(2)
            + (xv[row * 3 + 2] - m).powi(2))
            / 3.0;
        for j in 0..3 {
            let want = (xv[row * 3 + j] - m) / (v + 1e-5).sqrt() * w[j] + b[j];
            assert!((y[row * 3 + j] - want).abs() < 1e-5, "row {row} j {j}");
        }
    }
}

#[test]
fn gather_negative_and_reshape() {
    let mut g1 = onnx::NodeProto::default();
    g1.op_type = "Gather".into();
    g1.input = vec!["data".into(), "idx".into()];
    g1.output = vec!["g".into()];
    g1.attribute.push(attr_int("axis", 0));
    let mut r = onnx::NodeProto::default();
    r.op_type = "Reshape".into();
    r.input = vec!["g".into(), "shape".into()];
    r.output = vec!["y".into()];
    let mut g = onnx::GraphProto::default();
    g.node = vec![g1, r];
    g.initializer.push(init_i64("shape", &[2], &[2, -1]));
    for name in ["data", "idx"] {
        let mut v = onnx::ValueInfoProto::default();
        v.name = name.into();
        g.input.push(v);
    }
    {
        let name = "y";
        let mut v = onnx::ValueInfoProto::default();
        v.name = name.into();
        g.output.push(v);
    }
    let mut m = onnx::ModelProto::default();
    m.ir_version = 9;
    m.graph = Some(g);
    let mut opset = onnx::OperatorSetIdProto::default();
    opset.version = 17;
    m.opset_import.push(opset);
    let mut bytes = Vec::new();
    prost::Message::encode(&m, &mut bytes).unwrap();

    let (_, out) = run_bytes(
        &bytes,
        &[
            ("data", Tensor::f32(vec![10.0, 20.0, 30.0, 40.0], vec![4])),
            ("idx", Tensor::i64(vec![-1, 0], vec![2])),
        ],
    );
    assert_eq!(f32s(&out["y"]), &[40.0, 10.0]);
    assert_eq!(out["y"].shape, vec![2, 1]);
}

#[test]
fn slice_negative_step() {
    let bytes = one_op(
        "Slice",
        &["x", "starts", "ends", "axes", "steps"],
        &["y"],
        vec![],
        vec![],
        &["x", "starts", "ends", "axes", "steps"],
    );
    let (_, out) = run_bytes(
        &bytes,
        &[
            (
                "x",
                Tensor::f32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6]),
            ),
            ("starts", Tensor::i64(vec![5], vec![1])),
            ("ends", Tensor::i64(vec![0], vec![1])),
            ("axes", Tensor::i64(vec![0], vec![1])),
            ("steps", Tensor::i64(vec![-1], vec![1])),
        ],
    );
    assert_eq!(f32s(&out["y"]), &[5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn erf_accuracy() {
    let bytes = one_op("Erf", &["x"], &["y"], vec![], vec![], &["x"]);
    let (_, out) = run_bytes(
        &bytes,
        &[("x", Tensor::f32(vec![0.0, 0.5, 1.0, 2.0, -1.5], vec![5]))],
    );
    let y = f32s(&out["y"]);
    let want = [0.0, 0.5204999, 0.8427008, 0.9953223, -0.9661051];
    for (a, b) in y.iter().zip(want) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn if_control_flow() {
    let mut then_g = onnx::GraphProto::default();
    let mut add = onnx::NodeProto::default();
    add.op_type = "Add".into();
    add.input = vec!["x".into(), "one".into()];
    add.output = vec!["res".into()];
    then_g.node.push(add);
    then_g.initializer.push(init_f32("one", &[], &[1.0]));
    for name in ["x", "res"] {
        let mut v = onnx::ValueInfoProto::default();
        v.name = name.into();
        then_g.input.push(v);
    }
    {
        let mut v = onnx::ValueInfoProto::default();
        v.name = "res".into();
        then_g.output.push(v);
    }
    let mut else_g = onnx::GraphProto::default();
    let mut sub = onnx::NodeProto::default();
    sub.op_type = "Sub".into();
    sub.input = vec!["x".into(), "one".into()];
    sub.output = vec!["res".into()];
    else_g.node.push(sub);
    else_g.initializer.push(init_f32("one", &[], &[1.0]));
    for name in ["x", "res"] {
        let mut v = onnx::ValueInfoProto::default();
        v.name = name.into();
        else_g.input.push(v);
    }
    {
        let mut v = onnx::ValueInfoProto::default();
        v.name = "res".into();
        else_g.output.push(v);
    }
    let mut a_then = onnx::AttributeProto::default();
    a_then.name = "then_branch".into();
    a_then.r#type = onnx::attribute_proto::AttributeType::Graph as i32;
    a_then.g = Some(then_g);
    let mut a_else = onnx::AttributeProto::default();
    a_else.name = "else_branch".into();
    a_else.r#type = onnx::attribute_proto::AttributeType::Graph as i32;
    a_else.g = Some(else_g);
    let mut iff = onnx::NodeProto::default();
    iff.op_type = "If".into();
    iff.input = vec!["cond".into()];
    iff.output = vec!["y".into()];
    iff.attribute = vec![a_then, a_else];
    let mut g = onnx::GraphProto::default();
    g.node.push(iff);
    for name in ["cond", "x", "y"] {
        let mut v = onnx::ValueInfoProto::default();
        v.name = name.into();
        g.input.push(v);
        let mut v2 = onnx::ValueInfoProto::default();
        v2.name = name.into();
        g.output.push(v2);
    }
    let mut m = onnx::ModelProto::default();
    m.ir_version = 9;
    m.graph = Some(g);
    let mut opset = onnx::OperatorSetIdProto::default();
    opset.version = 17;
    m.opset_import.push(opset);
    let mut bytes = Vec::new();
    prost::Message::encode(&m, &mut bytes).unwrap();

    let (_, out) = run_bytes(
        &bytes,
        &[
            ("cond", Tensor::bool(vec![true], vec![])),
            ("x", Tensor::f32(vec![5.0], vec![])),
        ],
    );
    assert_eq!(f32s(&out["y"]), &[6.0]);
}
