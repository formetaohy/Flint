pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod tensor;

pub use session::Session;

mod dtype;
mod graph;
mod ops;
mod session;
