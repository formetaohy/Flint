use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no {0} backend available")]
    NoBackend(&'static str),
    #[error("shader: {0}")]
    Shader(String),
    #[error("timestamp queries not supported by this backend")]
    TimestampUnsupported,
    #[error("cooperative matrix {0} not supported by device")]
    CoopUnsupported(String),
    #[error("vulkan: {0}")]
    Vulkan(String),
    #[error("metal: {0}")]
    Metal(String),
    #[error("buffer is not host visible")]
    BufferNotHostVisible,
    #[error("buffer type mismatch: expected {expected}, got {actual}")]
    BufferTypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("kernel type mismatch: expected {expected}, got {actual}")]
    KernelTypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("encoder type mismatch: expected {expected}, got {actual}")]
    EncoderTypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("encoder was already submitted")]
    EncoderInactive,
    #[error("{kind} belongs to a different device")]
    DeviceMismatch { kind: &'static str },
    #[error("binding index {index} is not declared in kernel {kernel}")]
    UndeclaredBinding { index: u32, kernel: String },
    #[error("range {offset}..{end} exceeds buffer size {size}")]
    RangeOutOfBounds { offset: u64, end: u64, size: u64 },
    #[error("offset {offset} is not aligned to {alignment}")]
    MisalignedOffset { offset: u64, alignment: u64 },
    #[error("kernel {kernel} workgroup size {size} exceeds device maximum {max}")]
    WorkgroupTooLarge { kernel: String, size: u64, max: u64 },
    #[error("kernel function {0} not found in library")]
    FunctionNotFound(String),
    #[error("dispatch before bind")]
    UnboundDispatch,
    #[error("kernel {0} has scalar parameters but set_scalars was not called")]
    UnboundScalars(String),
    #[error("scalar data size {actual} does not match layout size {expected}")]
    ScalarSizeMismatch { expected: u32, actual: u32 },
}

pub type Result<T> = std::result::Result<T, Error>;
