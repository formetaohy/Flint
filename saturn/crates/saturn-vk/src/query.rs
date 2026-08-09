use std::sync::Arc;

use ash::vk;

use saturn_core::error::{Error, Result};
use saturn_core::TimestampSet;

use crate::device::{VkDevice, VkDeviceInner};

pub struct VkTimestampSet {
    pub(crate) inner: Arc<VkDeviceInner>,
    pub(crate) pool: vk::QueryPool,
    capacity: u32,
}

impl VkTimestampSet {
    pub fn create(device: &VkDevice, capacity: u32) -> Result<Box<dyn TimestampSet>> {
        let inner = device.inner.clone();
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(capacity);
        let pool = unsafe { inner.device.create_query_pool(&info, None) }
            .map_err(|e| Error::Vulkan(e.to_string()))?;
        Ok(Box::new(Self {
            inner,
            pool,
            capacity,
        }))
    }
}

impl TimestampSet for VkTimestampSet {
    fn capacity(&self) -> u32 {
        self.capacity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for VkTimestampSet {
    fn drop(&mut self) {
        unsafe {
            self.inner.device.destroy_query_pool(self.pool, None);
        }
    }
}
