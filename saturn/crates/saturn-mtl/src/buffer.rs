use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use saturn_core::Buffer;
use saturn_core::error::{Error, Result};

use crate::device::MtlDevice;

pub struct MtlBuffer {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: u64,
    host_visible: bool,
}

impl MtlBuffer {
    pub fn create(device: &MtlDevice, spec: &saturn_core::BufferSpec) -> Result<Box<dyn Buffer>> {
        let options = if spec.host_visible {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        let raw = device
            .device
            .newBufferWithLength_options(spec.size as usize, options)
            .ok_or(Error::Metal("newBufferWithLength failed".to_string()))?;
        Ok(Box::new(Self {
            raw,
            size: spec.size,
            host_visible: spec.host_visible,
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

impl Buffer for MtlBuffer {
    fn size(&self) -> u64 {
        self.size
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if !self.host_visible {
            return Err(Error::BufferNotHostVisible);
        }
        self.check_range(offset, data.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.raw
                    .contents()
                    .as_ptr()
                    .add(offset as usize)
                    .cast::<u8>(),
                data.len(),
            );
        }
        Ok(())
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if !self.host_visible {
            return Err(Error::BufferNotHostVisible);
        }
        self.check_range(offset, out.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw
                    .contents()
                    .as_ptr()
                    .add(offset as usize)
                    .cast::<u8>(),
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
