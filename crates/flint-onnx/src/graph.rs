use std::collections::HashMap;
use std::path::Path;

use flint_error::{Error, Result};

use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Node {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attr>,
}

impl Node {
    fn placeholder() -> Self {
        Node {
            op_type: String::new(),
            inputs: vec![],
            outputs: vec![],
            attrs: HashMap::new(),
        }
    }

    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.get(name)
    }

    pub fn need(&self, name: &str) -> Result<&Attr> {
        self.attr(name).ok_or_else(|| {
            Error::Model(format!("node {} missing attribute {name}", self.op_type))
        })
    }
}

#[derive(Debug)]
pub enum Attr {
    F(f32),
    I(i64),
    S(Vec<u8>),
    T(Tensor),
    G(Box<Graph>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<Vec<u8>>),
    Tensors(Vec<Tensor>),
    Graphs(Vec<Graph>),
}

impl Attr {
    pub fn i(&self) -> Result<i64> {
        match self {
            Attr::I(v) => Ok(*v),
            other => Err(Error::Model(format!("expected int attribute, got {other:?}"))),
        }
    }

    pub fn f(&self) -> Result<f32> {
        match self {
            Attr::F(v) => Ok(*v),
            other => Err(Error::Model(format!("expected float attribute, got {other:?}"))),
        }
    }

    pub fn s(&self) -> Result<&[u8]> {
        match self {
            Attr::S(v) => Ok(v),
            other => Err(Error::Model(format!("expected string attribute, got {other:?}"))),
        }
    }

    pub fn ints(&self) -> Result<&[i64]> {
        match self {
            Attr::Ints(v) => Ok(v),
            other => Err(Error::Model(format!("expected ints attribute, got {other:?}"))),
        }
    }

    pub fn strings(&self) -> Result<&[Vec<u8>]> {
        match self {
            Attr::Strings(v) => Ok(v),
            other => Err(Error::Model(format!("expected strings attribute, got {other:?}"))),
        }
    }
}

#[derive(Debug)]
pub struct ValueInfo {
    pub name: String,

    pub elem_type: i32,

    pub dims: Vec<i64>,
}

#[derive(Debug)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub initializers: HashMap<String, Tensor>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

impl Graph {

    pub fn load(path: &Path) -> Result<Graph> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Model(format!("cannot read {}: {e}", path.display())))?;
        let model: crate::onnx::ModelProto = prost::Message::decode(&bytes[..])
            .map_err(|e| Error::Model(format!("invalid ONNX protobuf: {e}")))?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| Error::Model("model has no graph".into()))?;
        Self::from_proto(graph, &dir)
    }

    pub fn from_proto(g: &crate::onnx::GraphProto, dir: &Path) -> Result<Graph> {
        let mut initializers = HashMap::new();
        for t in &g.initializer {
            let tensor = Tensor::from_proto(t, dir)?;
            initializers.insert(t.name.clone(), tensor);
        }
        if !g.sparse_initializer.is_empty() {
            return Err(Error::Model("sparse initializers are not supported".into()));
        }

        let mut nodes: Vec<Node> = g
            .node
            .iter()
            .map(|n| Self::node_from_proto(n, dir))
            .collect::<Result<_>>()?;
        topo_sort(&mut nodes)?;

        let inputs = g
            .input
            .iter()
            .map(|v| {
                let mut vi = ValueInfo {
                    name: v.name.clone(),
                    elem_type: 0,
                    dims: vec![],
                };
                if let Some(t) = v.r#type.as_ref().and_then(|t| t.value.as_ref()) {
                    if let crate::onnx::type_proto::Value::TensorType(tensor) = t {
                        vi.elem_type = tensor.elem_type;
                        if let Some(shape) = tensor.shape.as_ref() {
                            vi.dims = shape
                                .dim
                                .iter()
                                .map(|d| match d.value {
                                    Some(crate::onnx::tensor_shape_proto::dimension::Value::DimValue(x)) => x,
                                    _ => 0,
                                })
                                .collect();
                        }
                    }
                }
                vi
            })
            .collect();
        let outputs = g
            .output
            .iter()
            .map(|v| ValueInfo {
                name: v.name.clone(),
                elem_type: 0,
                dims: vec![],
            })
            .collect();

        Ok(Graph {
            name: g.name.clone(),
            nodes,
            initializers,
            inputs,
            outputs,
        })
    }

    fn node_from_proto(n: &crate::onnx::NodeProto, dir: &Path) -> Result<Node> {
        let mut attrs = HashMap::new();
        for a in &n.attribute {
            let attr = match a.r#type() {
                crate::onnx::attribute_proto::AttributeType::Float => Attr::F(a.f),
                crate::onnx::attribute_proto::AttributeType::Int => Attr::I(a.i),
                crate::onnx::attribute_proto::AttributeType::String => Attr::S(a.s.clone()),
                crate::onnx::attribute_proto::AttributeType::Tensor => Attr::T(
                    a.t
                        .as_ref()
                        .map(|t| Tensor::from_proto(t, dir))
                        .transpose()?
                        .ok_or_else(|| Error::Model("empty tensor attribute".into()))?,
                ),
                crate::onnx::attribute_proto::AttributeType::Graph => Attr::G(Box::new(
                    Self::from_proto(
                        a.g
                            .as_ref()
                            .ok_or_else(|| Error::Model("empty graph attribute".into()))?,
                        dir,
                    )?,
                )),
                crate::onnx::attribute_proto::AttributeType::Floats => Attr::Floats(a.floats.clone()),
                crate::onnx::attribute_proto::AttributeType::Ints => Attr::Ints(a.ints.clone()),
                crate::onnx::attribute_proto::AttributeType::Strings => Attr::Strings(a.strings.clone()),
                crate::onnx::attribute_proto::AttributeType::Tensors => Attr::Tensors(
                    a.tensors
                        .iter()
                        .map(|t| Tensor::from_proto(t, dir))
                        .collect::<Result<_>>()?,
                ),
                crate::onnx::attribute_proto::AttributeType::Graphs => Attr::Graphs(
                    a.graphs
                        .iter()
                        .map(|g| Self::from_proto(g, dir))
                        .collect::<Result<_>>()?,
                ),
                other => {
                    return Err(Error::Model(format!(
                        "unsupported attribute type {other:?} on {}",
                        n.op_type
                    )))
                }
            };
            attrs.insert(a.name.clone(), attr);
        }
        Ok(Node {
            op_type: n.op_type.clone(),
            inputs: n.input.clone(),
            outputs: n.output.clone(),
            attrs,
        })
    }
}

fn topo_sort(nodes: &mut Vec<Node>) -> Result<()> {
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for out in &n.outputs {
            if !out.is_empty() && producer.insert(out.as_str(), i).is_some() {
                return Err(Error::Model(format!("value {out:?} produced twice")));
            }
        }
    }
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        for inp in &n.inputs {
            if let Some(&p) = producer.get(inp.as_str())
                && p != i
            {
                consumers[p].push(i);
                indegree[i] += 1;
            }
        }
    }
    let mut ready: Vec<usize> = (0..nodes.len()).filter(|&i| indegree[i] == 0).collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(i) = ready.pop() {
        order.push(i);
        for &c in &consumers[i] {
            indegree[c] -= 1;
            if indegree[c] == 0 {
                ready.push(c);
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(Error::Model("graph contains a cycle".into()));
    }
    let mut sorted = Vec::with_capacity(nodes.len());
    for &i in &order {
        sorted.push(std::mem::replace(&mut nodes[i], Node::placeholder()));
    }
    *nodes = sorted;
    Ok(())
}
