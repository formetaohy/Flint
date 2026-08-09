use saturn_core::{Device, Result};

pub enum BackendKind {
    Vulkan,
    Metal,
}

pub fn open(kind: BackendKind) -> Result<Box<dyn Device>> {
    match kind {
        BackendKind::Vulkan => Ok(Box::new(saturn_vk::VkDevice::open(&saturn_vk::Options {
            validation: std::env::var("SATURN_VALIDATION").is_ok(),
        })?)),
        BackendKind::Metal => open_metal(),
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn open_metal() -> Result<Box<dyn Device>> {
    Ok(Box::new(saturn_mtl::MtlDevice::open()?))
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn open_metal() -> Result<Box<dyn Device>> {
    Err(saturn_core::Error::NoBackend("metal"))
}
