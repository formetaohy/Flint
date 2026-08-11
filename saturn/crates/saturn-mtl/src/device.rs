use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLanguageVersion, MTLLibrary,
};

use saturn_core::error::{Error, Result};
use saturn_core::{Buffer, CommandEncoder, Kernel, Submission};

use crate::buffer::MtlBuffer;
use crate::encoder::MtlEncoder;
use crate::kernel::MtlKernel;

pub struct MtlDevice {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    name: String,
}

impl MtlDevice {
    pub fn open() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoBackend("metal"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Metal("newCommandQueue failed".to_string()))?;
        let name = device.name().to_string();
        log::info!("metal: opened {name}");
        Ok(Self {
            device,
            queue,
            name,
        })
    }
}

impl saturn_core::Device for MtlDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn offset_alignment(&self) -> u64 {
        1
    }

    fn create_buffer(&self, spec: &saturn_core::BufferSpec) -> Result<Box<dyn Buffer>> {
        MtlBuffer::create(self, spec)
    }

    fn create_kernel(&self, spec: &saturn_core::KernelSpec) -> Result<Box<dyn Kernel>> {
        MtlKernel::create(self, spec)
    }

    fn encoder(&self) -> Result<Box<dyn CommandEncoder>> {
        MtlEncoder::create(self)
    }

    fn submit(&self, encoder: Box<dyn CommandEncoder>) -> Result<Box<dyn Submission>> {
        let actual = std::any::type_name_of_val(&*encoder);
        let any: Box<dyn std::any::Any> = encoder;
        let encoder = any
            .downcast::<MtlEncoder>()
            .map_err(|_| Error::EncoderTypeMismatch {
                expected: std::any::type_name::<MtlEncoder>(),
                actual,
            })?;
        Ok(Box::new(crate::encoder::MtlSubmission::new(*encoder)?))
    }
}

pub(crate) fn compile_kernel(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    msl: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version3_1);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(msl), Some(&options))
        .map_err(|e| Error::Metal(e.to_string()))
}
