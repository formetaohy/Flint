pub mod buffer;
pub mod device;
pub mod encoder;
pub mod kernel;
pub mod query;

pub use buffer::Buffer;
pub use device::{Device, HostAccess, KernelSpec};
pub use encoder::{BindingRef, Encoder, Submission};
pub use kernel::Kernel;
pub use query::TimestampSet;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) struct DeviceInner {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) bind_groups: Mutex<HashMap<u64, wgpu::BindGroup>>,
}

pub(crate) type DeviceRef = Arc<DeviceInner>;
