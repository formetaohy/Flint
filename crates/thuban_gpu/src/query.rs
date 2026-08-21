use thuban_error::Result;

use crate::DeviceRef;

pub struct TimestampSet {
    pub(crate) query_set: wgpu::QuerySet,
}

impl TimestampSet {
    pub(crate) fn create(device: &DeviceRef, capacity: u32) -> Result<Self> {
        let query_set = device.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });
        Ok(Self { query_set })
    }
}
