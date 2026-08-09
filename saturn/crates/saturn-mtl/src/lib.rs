#![cfg(any(target_os = "macos", target_os = "ios"))]

pub mod buffer;
pub mod device;
pub mod encoder;
pub mod kernel;

pub use device::MtlDevice;
