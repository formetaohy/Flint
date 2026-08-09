use std::any::Any;

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::kernel::Kernel;
use crate::query::TimestampSet;

pub struct BindingRef<'a> {
    pub index: u32,
    pub buffer: &'a dyn Buffer,
    pub offset: u64,
    pub size: u64,
}

pub trait CommandEncoder: Any {
    fn bind(&mut self, kernel: &dyn Kernel, bindings: &[BindingRef]) -> Result<()>;
    fn set_scalars(&mut self, kernel: &dyn Kernel, bytes: &[u8]) -> Result<()>;
    fn dispatch(&mut self, groups: [u32; 3]) -> Result<()>;
    fn copy(
        &mut self,
        src: &dyn Buffer,
        src_offset: u64,
        dst: &dyn Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<()>;
    fn clear(&mut self, dst: &dyn Buffer, offset: u64, size: u64) -> Result<()>;
    fn barrier(&mut self) -> Result<()>;
    fn write_timestamp(&mut self, _set: &dyn TimestampSet, _index: u32) -> Result<()> {
        Err(Error::TimestampUnsupported)
    }
    fn resolve_timestamps(
        &mut self,
        _set: &dyn TimestampSet,
        _start: u32,
        _count: u32,
        _dst: &dyn Buffer,
        _dst_offset: u64,
    ) -> Result<()> {
        Err(Error::TimestampUnsupported)
    }
    fn as_any(&self) -> &dyn Any;
}

pub trait Submission {
    fn wait(&self) -> Result<()>;
}
