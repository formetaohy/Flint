use std::any::Any;

pub trait TimestampSet: Any {
    fn capacity(&self) -> u32;
    fn as_any(&self) -> &dyn Any;
}
