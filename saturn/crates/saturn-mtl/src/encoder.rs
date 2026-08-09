use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use saturn_core::error::{Error, Result};
use saturn_core::{BindingRef, Buffer, CommandEncoder, Kernel, ScalarLayout, Submission};

use crate::buffer::MtlBuffer;
use crate::device::MtlDevice;
use crate::kernel::MtlKernel;

struct Bound {
    name: String,
    workgroup_size: [u32; 3],
    max_threads: u64,
    scalar_layout: Option<ScalarLayout>,
    scalars_set: bool,
}

pub struct MtlEncoder {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    compute: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    bound: Option<Bound>,
}

impl MtlEncoder {
    pub fn create(device: &MtlDevice) -> Result<Box<dyn CommandEncoder>> {
        let cmd = device
            .queue
            .commandBuffer()
            .ok_or(Error::Metal("commandBuffer failed".to_string()))?;
        Ok(Box::new(Self {
            cmd,
            compute: None,
            bound: None,
        }))
    }

    fn compute(&mut self) -> Result<&Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        if self.compute.is_none() {
            self.compute = self.cmd.computeCommandEncoder();
            if self.compute.is_none() {
                return Err(Error::Metal("computeCommandEncoder failed".to_string()));
            }
        }
        Ok(self.compute.as_ref().expect("compute encoder"))
    }

    fn end_compute(&mut self) {
        if let Some(encoder) = self.compute.take() {
            encoder.endEncoding();
        }
        self.bound = None;
    }
}

impl CommandEncoder for MtlEncoder {
    fn bind(&mut self, kernel: &dyn Kernel, bindings: &[BindingRef]) -> Result<()> {
        let kernel = kernel
            .as_any()
            .downcast_ref::<MtlKernel>()
            .ok_or(Error::KernelTypeMismatch {
                expected: std::any::type_name::<MtlKernel>(),
                actual: std::any::type_name_of_val(kernel),
            })?;
        let encoder = self.compute()?;
        encoder.setComputePipelineState(&kernel.pipeline);
        for binding in bindings {
            let buffer = binding
                .buffer
                .as_any()
                .downcast_ref::<MtlBuffer>()
                .ok_or(Error::BufferTypeMismatch {
                    expected: std::any::type_name::<MtlBuffer>(),
                    actual: std::any::type_name_of_val(binding.buffer),
                })?;
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&buffer.raw),
                    binding.offset as usize,
                    binding.index as usize,
                );
            }
        }
        self.bound = Some(Bound {
            name: kernel.name().to_string(),
            workgroup_size: kernel.workgroup_size(),
            max_threads: kernel.max_threads,
            scalar_layout: kernel.scalar_layout().cloned(),
            scalars_set: false,
        });
        Ok(())
    }

    fn set_scalars(&mut self, kernel: &dyn Kernel, bytes: &[u8]) -> Result<()> {
        let kernel = kernel
            .as_any()
            .downcast_ref::<MtlKernel>()
            .ok_or(Error::KernelTypeMismatch {
                expected: std::any::type_name::<MtlKernel>(),
                actual: std::any::type_name_of_val(kernel),
            })?;
        let Some(layout) = &kernel.scalar_layout else {
            return Err(Error::Metal(format!(
                "kernel {} has no scalar parameters",
                kernel.name()
            )));
        };
        if self.bound.as_ref().and_then(|b| b.scalar_layout.as_ref()) != Some(layout) {
            return Err(Error::Metal(format!(
                "scalars set for kernel {} while {} is bound",
                kernel.name(),
                self.bound
                    .as_ref()
                    .map(|b| b.name.as_str())
                    .unwrap_or("nothing")
            )));
        }
        if bytes.len() != layout.size as usize {
            return Err(Error::ScalarSizeMismatch {
                expected: layout.size,
                actual: bytes.len() as u32,
            });
        }
        let encoder = self.compute()?;
        let base = kernel.buffer_count;
        for (index, field) in layout.fields.iter().enumerate() {
            let width = field.ty.width() as usize;
            let ptr: *const std::ffi::c_void = bytes
                [field.offset as usize..field.offset as usize + width]
                .as_ptr()
                .cast();
            unsafe {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new(ptr as *mut std::ffi::c_void)
                        .expect("scalar bytes pointer"),
                    width,
                    (base + index) as usize,
                );
            }
        }
        if let Some(bound) = &mut self.bound {
            bound.scalars_set = true;
        }
        Ok(())
    }

    fn dispatch(&mut self, groups: [u32; 3]) -> Result<()> {
        if groups[0] == 0 || groups[1] == 0 || groups[2] == 0 {
            return Ok(());
        }
        let bound = self
            .bound
            .as_ref()
            .ok_or(Error::UnboundDispatch)?;
        if bound.scalar_layout.is_some() && !bound.scalars_set {
            return Err(Error::UnboundScalars(bound.name.clone()));
        }
        let threads = bound.workgroup_size;
        let size = threads[0] as u64 * threads[1] as u64 * threads[2] as u64;
        if size > bound.max_threads {
            return Err(Error::WorkgroupTooLarge {
                kernel: bound.name.clone(),
                size,
                max: bound.max_threads,
            });
        }
        let encoder = self.compute()?;
        let groups_size = MTLSize {
            width: groups[0] as usize,
            height: groups[1] as usize,
            depth: groups[2] as usize,
        };
        let threads_size = MTLSize {
            width: threads[0] as usize,
            height: threads[1] as usize,
            depth: threads[2] as usize,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups_size, threads_size);
        Ok(())
    }

    fn copy(
        &mut self,
        src: &dyn Buffer,
        src_offset: u64,
        dst: &dyn Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        let src = src
            .as_any()
            .downcast_ref::<MtlBuffer>()
            .ok_or(Error::BufferTypeMismatch {
                expected: std::any::type_name::<MtlBuffer>(),
                actual: std::any::type_name_of_val(src),
            })?;
        let dst = dst
            .as_any()
            .downcast_ref::<MtlBuffer>()
            .ok_or(Error::BufferTypeMismatch {
                expected: std::any::type_name::<MtlBuffer>(),
                actual: std::any::type_name_of_val(dst),
            })?;
        if src_offset
            .checked_add(size)
            .is_none_or(|end| end > src.size())
            || dst_offset
                .checked_add(size)
                .is_none_or(|end| end > dst.size())
        {
            return Err(Error::RangeOutOfBounds {
                offset: src_offset.min(dst_offset),
                end: src_offset
                    .saturating_add(size)
                    .max(dst_offset.saturating_add(size)),
                size: src.size().min(dst.size()),
            });
        }
        self.end_compute();
        let blit = self
            .cmd
            .blitCommandEncoder()
            .ok_or(Error::Metal("blitCommandEncoder failed".to_string()))?;
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src.raw,
                src_offset as usize,
                &dst.raw,
                dst_offset as usize,
                size as usize,
            );
        }
        blit.endEncoding();
        Ok(())
    }

    fn barrier(&mut self) -> Result<()> {
        self.compute()?.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        Ok(())
    }

    fn clear(&mut self, dst: &dyn Buffer, offset: u64, size: u64) -> Result<()> {
        let dst = dst
            .as_any()
            .downcast_ref::<MtlBuffer>()
            .ok_or(Error::BufferTypeMismatch {
                expected: std::any::type_name::<MtlBuffer>(),
                actual: std::any::type_name_of_val(dst),
            })?;
        if offset
            .checked_add(size)
            .is_none_or(|end| end > dst.size())
        {
            return Err(Error::RangeOutOfBounds {
                offset,
                end: offset.saturating_add(size),
                size: dst.size(),
            });
        }
        self.end_compute();
        let blit = self
            .cmd
            .blitCommandEncoder()
            .ok_or(Error::Metal("blitCommandEncoder failed".to_string()))?;
        unsafe {
            blit.fillBuffer_range_value(&dst.raw, offset as usize, size as usize, 0);
        }
        blit.endEncoding();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct MtlSubmission {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl MtlSubmission {
    pub fn new(mut encoder: MtlEncoder) -> Result<Self> {
        encoder.end_compute();
        encoder.cmd.commit();
        Ok(Self { cmd: encoder.cmd })
    }
}

impl Submission for MtlSubmission {
    fn wait(&self) -> Result<()> {
        self.cmd.waitUntilCompleted();
        Ok(())
    }
}
