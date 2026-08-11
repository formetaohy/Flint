pub mod dtype;
pub mod graph;
pub mod graph_ops;
pub mod ops;
pub mod session;
pub mod tensor;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use graph::Graph;
pub use session::Session;
pub use tensor::Tensor;
