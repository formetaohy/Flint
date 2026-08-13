use std::sync::atomic::{AtomicU32, Ordering};

use flint_onnx::onnx;
use flint_onnx::tensor::Data;
use flint_onnx::tensor::Tensor;
use flint_onnx::Session;

static NEXT_FILE: AtomicU32 = AtomicU32::new(0);

fn node(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<onnx::AttributeProto>,
) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op_type.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

fn value(name: &str) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

fn graph(
    nodes: Vec<onnx::NodeProto>,
    inits: Vec<onnx::TensorProto>,
    inputs: Vec<onnx::ValueInfoProto>,
    outputs: Vec<onnx::ValueInfoProto>,
) -> onnx::GraphProto {
    onnx::GraphProto {
        node: nodes,
        initializer: inits,
        input: inputs,
        output: outputs,
        ..Default::default()
    }
}

fn model(g: onnx::GraphProto) -> Vec<u8> {
    let m = onnx::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx::OperatorSetIdProto {
            version: 17,
            ..Default::default()
        }],
        graph: Some(g),
        ..Default::default()
    };
    let mut buf = Vec::new();
    prost::Message::encode(&m, &mut buf).unwrap();
    buf
}

fn one_op(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<onnx::AttributeProto>,
    inits: Vec<onnx::TensorProto>,
    graph_inputs: &[&str],
) -> Vec<u8> {
    model(graph(
        vec![node(op_type, inputs, outputs, attrs)],
        inits,
        graph_inputs.iter().map(|n| value(n)).collect(),
        outputs.iter().map(|n| value(n)).collect(),
    ))
}

fn attr_int(name: &str, v: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        i: v,
        ..Default::default()
    }
}

fn attr_ints(name: &str, v: &[i64]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Ints as i32,
        ints: v.to_vec(),
        ..Default::default()
    }
}

fn init_f32(name: &str, dims: &[i64], data: &[f32]) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn init_i64(name: &str, dims: &[i64], data: &[i64]) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: 7,
        int64_data: data.to_vec(),
        ..Default::default()
    }
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
    let n1 = node(
        "ReduceMean",
        &["x"],
        &["mean"],
        vec![attr_ints("axes", &[-1])],
    );
    let n2 = node("Sub", &["x", "mean"], &["centered"], vec![]);
    let n3 = node("Pow", &["centered", "two"], &["sq"], vec![]);
    let n4 = node(
        "ReduceMean",
        &["sq"],
        &["var"],
        vec![attr_ints("axes", &[-1])],
    );
    let n5 = node("Add", &["var", "eps"], &["var_eps"], vec![]);
    let n6 = node("Sqrt", &["var_eps"], &["std"], vec![]);
    let n7 = node("Div", &["centered", "std"], &["norm"], vec![]);
    let n8 = node("Mul", &["norm", "w"], &["scaled"], vec![]);
    let n9 = node("Add", &["scaled", "b"], &["y"], vec![]);
    let g = graph(
        vec![n1, n2, n3, n4, n5, n6, n7, n8, n9],
        inits,
        vec![value("x")],
        vec![value("y")],
    );
    let bytes = model(g);

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
    let g1 = node("Gather", &["data", "idx"], &["g"], vec![attr_int("axis", 0)]);
    let r = node("Reshape", &["g", "shape"], &["y"], vec![]);
    let g = graph(
        vec![g1, r],
        vec![init_i64("shape", &[2], &[2, -1])],
        vec![value("data"), value("idx")],
        vec![value("y")],
    );
    let bytes = model(g);

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
    let then_g = graph(
        vec![node("Add", &["x", "one"], &["res"], vec![])],
        vec![init_f32("one", &[], &[1.0])],
        vec![value("x"), value("res")],
        vec![value("res")],
    );
    let else_g = graph(
        vec![node("Sub", &["x", "one"], &["res"], vec![])],
        vec![init_f32("one", &[], &[1.0])],
        vec![value("x"), value("res")],
        vec![value("res")],
    );
    let a_then = onnx::AttributeProto {
        name: "then_branch".into(),
        r#type: onnx::attribute_proto::AttributeType::Graph as i32,
        g: Some(then_g),
        ..Default::default()
    };
    let a_else = onnx::AttributeProto {
        name: "else_branch".into(),
        r#type: onnx::attribute_proto::AttributeType::Graph as i32,
        g: Some(else_g),
        ..Default::default()
    };
    let iff = node("If", &["cond"], &["y"], vec![a_then, a_else]);
    let g = graph(
        vec![iff],
        vec![],
        vec![value("cond"), value("x"), value("y")],
        vec![value("cond"), value("x"), value("y")],
    );
    let bytes = model(g);

    let (_, out) = run_bytes(
        &bytes,
        &[
            ("cond", Tensor::bool(vec![true], vec![])),
            ("x", Tensor::f32(vec![5.0], vec![])),
        ],
    );
    assert_eq!(f32s(&out["y"]), &[6.0]);
}
