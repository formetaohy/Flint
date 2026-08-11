use crate::Scalar;
use crate::buffer::Buffer;
use crate::command::{CommandEncoder, Submission};
use crate::error::Result;
use crate::kernel::Kernel;
use crate::query::TimestampSet;
use crate::spec::{BufferSpec, KernelSpec};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CoopShape {
    pub a: Scalar,
    pub b: Scalar,
    pub c: Scalar,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub trait Device {
    fn name(&self) -> &str;
    fn offset_alignment(&self) -> u64;
    fn create_buffer(&self, spec: &BufferSpec) -> Result<Box<dyn Buffer>>;
    fn create_kernel(&self, spec: &KernelSpec) -> Result<Box<dyn Kernel>>;
    fn encoder(&self) -> Result<Box<dyn CommandEncoder>>;
    fn submit(&self, encoder: Box<dyn CommandEncoder>) -> Result<Box<dyn Submission>>;
    fn coop_supported(&self, _shape: CoopShape) -> bool {
        false
    }
    fn create_timestamp_set(&self, _capacity: u32) -> Result<Box<dyn TimestampSet>> {
        Err(crate::Error::TimestampUnsupported)
    }
    fn timestamp_period_ns(&self) -> f64 {
        1.0
    }
}
