use std::any::Any;

use crate::error::Result;

pub trait Buffer: Any {
    fn size(&self) -> u64;
    fn write(&self, offset: u64, data: &[u8]) -> Result<()>;
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()>;
    fn as_any(&self) -> &dyn Any;
}
