//! GPU device services: wgpu adapter/device ownership, buffer and tensor
//! factories, readback, and the dispatch facade over the compute kernels.
//! Kernel semantics live in `flint-kernel`.

mod backend;

pub use backend::{Backend, Binding, Pass};
