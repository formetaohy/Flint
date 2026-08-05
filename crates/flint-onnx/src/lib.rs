//! ONNX model loading and graph execution for Flint: parses a serialized
//! `ModelProto`, runs its computation graph on CPU, and downloads models
//! from the Hugging Face Hub.

pub mod graph;
pub mod graph_ops;
pub mod hub;
pub mod ops;
pub mod session;
pub mod tensor;

/// Generated ONNX protobuf types (prost), used by graph parsing and tests.
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use graph::Graph;
pub use session::Session;
pub use tensor::Tensor;
