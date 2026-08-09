use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;

use saturn_core::error::{Error, Result};
use saturn_core::Buffer;

use crate::device::{check, VkDevice, VkDeviceInner};

pub struct VkBuffer {
    pub(crate) inner: Arc<VkDeviceInner>,
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) mapped: Option<NonNull<u8>>,
    pub(crate) size: u64,
}

impl VkBuffer {
    pub fn create(device: &VkDevice, spec: &saturn_core::BufferSpec) -> Result<Box<dyn Buffer>> {
        let inner = device.inner.clone();
        let info = vk::BufferCreateInfo::default()
            .size(spec.size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { inner.device.create_buffer(&info, None) }
            .map_err(|e| Error::Vulkan(e.to_string()))?;
        let requirements = unsafe { inner.device.get_buffer_memory_requirements(buffer) };
        let memory_type = if spec.host_visible {
            inner.host_memory
        } else {
            inner.device_memory
        };
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = unsafe { inner.device.allocate_memory(&allocate, None) }
            .map_err(|e| Error::Vulkan(e.to_string()))?;
        check(unsafe { inner.device.bind_buffer_memory(buffer, memory, 0) })?;
        let mapped = if spec.host_visible {
            let ptr = unsafe {
                inner
                    .device
                    .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
            }
            .map_err(|e| Error::Vulkan(e.to_string()))?;
            Some(NonNull::new(ptr.cast::<u8>()).expect("map_memory returned null"))
        } else {
            None
        };
        Ok(Box::new(Self {
            inner,
            buffer,
            memory,
            mapped,
            size: spec.size,
        }))
    }

    fn check_range(&self, offset: u64, len: usize) -> Result<()> {
        let end = offset
            .checked_add(len as u64)
            .ok_or(Error::RangeOutOfBounds {
                offset,
                end: u64::MAX,
                size: self.size,
            })?;
        if end > self.size {
            Err(Error::RangeOutOfBounds {
                offset,
                end,
                size: self.size,
            })
        } else {
            Ok(())
        }
    }
}

impl Buffer for VkBuffer {
    fn size(&self) -> u64 {
        self.size
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let ptr = self
            .mapped
            .ok_or(Error::BufferNotHostVisible)?;
        self.check_range(offset, data.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                ptr.as_ptr().add(offset as usize),
                data.len(),
            );
        }
        Ok(())
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let ptr = self.mapped.ok_or(Error::BufferNotHostVisible)?;
        self.check_range(offset, out.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr().add(offset as usize),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_some() {
                self.inner.device.unmap_memory(self.memory);
            }
            self.inner.device.destroy_buffer(self.buffer, None);
            self.inner.device.free_memory(self.memory, None);
        }
    }
}
