use std::cell::Cell;
use std::sync::Arc;

use ash::vk;

use saturn_core::error::{Error, Result};
use saturn_core::{BindingRef, Buffer, CommandEncoder, Kernel, Submission, TimestampSet};

use crate::buffer::VkBuffer;
use crate::device::{VkDeviceInner, check};
use crate::kernel::VkKernel;
use crate::query::VkTimestampSet;

pub(crate) const MAX_SETS_PER_POOL: u32 = 4096;
pub(crate) const MAX_BINDINGS_PER_SET: u32 = 32;

pub(crate) struct Bound {
    pub(crate) name: String,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) scalar_layout: Option<saturn_core::ScalarLayout>,
    pub(crate) scalars_set: bool,
}

pub struct VkEncoder {
    pub(crate) inner: Arc<VkDeviceInner>,
    pub(crate) cmd: vk::CommandBuffer,
    pub(crate) active: bool,
    pub(crate) pools: Vec<vk::DescriptorPool>,
    pub(crate) current_pool: vk::DescriptorPool,
    pub(crate) sets_allocated: u32,
    pub(crate) bound: Option<Bound>,
}

impl VkEncoder {
    pub(crate) fn take_cmd(&mut self) -> vk::CommandBuffer {
        self.active = false;
        self.cmd
    }

    pub(crate) fn take_pools(&mut self) -> Vec<vk::DescriptorPool> {
        self.active = false;
        self.current_pool = vk::DescriptorPool::null();
        self.sets_allocated = 0;
        std::mem::take(&mut self.pools)
    }

    fn allocate_set(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        if std::env::var("TRACE_SET").is_ok() {
            eprintln!(
                "[set] pool={} sets={}",
                self.pools.len(),
                self.sets_allocated
            );
        }
        if self.sets_allocated >= MAX_SETS_PER_POOL {
            self.grow_pool()?;
        }
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.current_pool)
            .set_layouts(std::slice::from_ref(&layout));
        match unsafe { self.inner.device.allocate_descriptor_sets(&info) } {
            Ok(sets) => {
                self.sets_allocated += 1;
                Ok(sets[0])
            }
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                self.grow_pool()?;
                self.allocate_set(layout)
            }
            Err(e) => Err(Error::Vulkan(format!("descriptor allocation failed: {e}"))),
        }
    }

    fn grow_pool(&mut self) -> Result<()> {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_SETS_PER_POOL * MAX_BINDINGS_PER_SET);
        let pool = unsafe {
            self.inner.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_SETS_PER_POOL)
                    .pool_sizes(&[pool_size]),
                None,
            )
        }
        .map_err(|e| Error::Vulkan(e.to_string()))?;
        self.pools.push(pool);
        self.current_pool = pool;
        self.sets_allocated = 0;
        Ok(())
    }
}

impl CommandEncoder for VkEncoder {
    fn bind(&mut self, kernel: &dyn Kernel, bindings: &[BindingRef]) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        let kernel =
            kernel
                .as_any()
                .downcast_ref::<VkKernel>()
                .ok_or(Error::KernelTypeMismatch {
                    expected: std::any::type_name::<VkKernel>(),
                    actual: std::any::type_name_of_val(kernel),
                })?;
        if !Arc::ptr_eq(&self.inner, &kernel.inner) {
            return Err(Error::DeviceMismatch { kind: "kernel" });
        }
        if bindings.len() != kernel.bindings.len() {
            return Err(Error::Vulkan(format!(
                "kernel {} declares {} bindings, got {}",
                kernel.name(),
                kernel.bindings.len(),
                bindings.len()
            )));
        }
        for binding in bindings {
            if !kernel.bindings.contains(&binding.index) {
                return Err(Error::UndeclaredBinding {
                    index: binding.index,
                    kernel: kernel.name().to_string(),
                });
            }
        }
        for (i, binding) in bindings.iter().enumerate() {
            if bindings[..i]
                .iter()
                .any(|other| other.index == binding.index)
            {
                return Err(Error::Vulkan(format!(
                    "binding index {} bound twice",
                    binding.index
                )));
            }
            let buffer = binding.buffer.as_any().downcast_ref::<VkBuffer>().ok_or(
                Error::BufferTypeMismatch {
                    expected: std::any::type_name::<VkBuffer>(),
                    actual: std::any::type_name_of_val(binding.buffer),
                },
            )?;
            if !Arc::ptr_eq(&self.inner, &buffer.inner) {
                return Err(Error::DeviceMismatch { kind: "buffer" });
            }
            if binding.offset % self.inner.offset_alignment != 0 {
                return Err(Error::MisalignedOffset {
                    offset: binding.offset,
                    alignment: self.inner.offset_alignment,
                });
            }
        }
        let set = self.allocate_set(kernel.layout)?;
        let mut infos = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let buffer = binding
                .buffer
                .as_any()
                .downcast_ref::<VkBuffer>()
                .expect("buffer type verified");
            infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(binding.offset)
                    .range(if binding.size == 0 {
                        vk::WHOLE_SIZE
                    } else {
                        binding.size
                    }),
            );
        }
        let writes: Vec<vk::WriteDescriptorSet> = bindings
            .iter()
            .zip(infos.iter())
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding.index)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe {
            self.inner.device.update_descriptor_sets(&writes, &[]);
            self.inner.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            self.inner.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[set],
                &[],
            );
        }
        self.bound = Some(Bound {
            name: kernel.name().to_string(),
            pipeline_layout: kernel.pipeline_layout,
            scalar_layout: kernel.scalar_layout().cloned(),
            scalars_set: false,
        });
        Ok(())
    }

    fn set_scalars(&mut self, kernel: &dyn Kernel, bytes: &[u8]) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        let kernel =
            kernel
                .as_any()
                .downcast_ref::<VkKernel>()
                .ok_or(Error::KernelTypeMismatch {
                    expected: std::any::type_name::<VkKernel>(),
                    actual: std::any::type_name_of_val(kernel),
                })?;
        let bound = self.bound.as_ref().ok_or(Error::UnboundDispatch)?;
        let Some(layout) = &kernel.scalar_layout else {
            return Err(Error::Vulkan(format!(
                "kernel {} has no scalar parameters",
                kernel.name()
            )));
        };
        if bound.scalar_layout.as_ref() != Some(layout) {
            return Err(Error::Vulkan(format!(
                "scalars set for kernel {} while {} is bound",
                kernel.name(),
                bound.name
            )));
        }
        if bytes.len() != layout.size as usize {
            return Err(Error::ScalarSizeMismatch {
                expected: layout.size,
                actual: bytes.len() as u32,
            });
        }
        unsafe {
            self.inner.device.cmd_push_constants(
                self.cmd,
                bound.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
        if let Some(bound) = &mut self.bound {
            bound.scalars_set = true;
        }
        Ok(())
    }

    fn dispatch(&mut self, groups: [u32; 3]) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        if groups[0] == 0 || groups[1] == 0 || groups[2] == 0 {
            return Ok(());
        }
        let bound = self.bound.as_ref().ok_or(Error::UnboundDispatch)?;
        if bound.scalar_layout.is_some() && !bound.scalars_set {
            return Err(Error::UnboundScalars(bound.name.clone()));
        }
        unsafe {
            self.inner
                .device
                .cmd_dispatch(self.cmd, groups[0], groups[1], groups[2]);
        }
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
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        let src = src
            .as_any()
            .downcast_ref::<VkBuffer>()
            .ok_or(Error::BufferTypeMismatch {
                expected: std::any::type_name::<VkBuffer>(),
                actual: std::any::type_name_of_val(src),
            })?;
        let dst = dst
            .as_any()
            .downcast_ref::<VkBuffer>()
            .ok_or(Error::BufferTypeMismatch {
                expected: std::any::type_name::<VkBuffer>(),
                actual: std::any::type_name_of_val(dst),
            })?;
        if !Arc::ptr_eq(&self.inner, &src.inner) || !Arc::ptr_eq(&self.inner, &dst.inner) {
            return Err(Error::DeviceMismatch { kind: "buffer" });
        }
        let src_end = src_offset
            .checked_add(size)
            .ok_or(Error::RangeOutOfBounds {
                offset: src_offset,
                end: u64::MAX,
                size: src.size,
            })?;
        let dst_end = dst_offset
            .checked_add(size)
            .ok_or(Error::RangeOutOfBounds {
                offset: dst_offset,
                end: u64::MAX,
                size: dst.size,
            })?;
        if src_end > src.size || dst_end > dst.size {
            return Err(Error::RangeOutOfBounds {
                offset: src_offset.min(dst_offset),
                end: src_end.max(dst_end),
                size: src.size.min(dst.size),
            });
        }
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);
        unsafe {
            self.inner
                .device
                .cmd_copy_buffer(self.cmd, src.buffer, dst.buffer, &[region]);
        }
        Ok(())
    }

    fn barrier(&mut self) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    fn clear(&mut self, dst: &dyn Buffer, offset: u64, size: u64) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        if !size.is_multiple_of(4) {
            return Err(Error::Vulkan(format!(
                "clear size {size} is not a multiple of 4"
            )));
        }
        let dst = dst
            .as_any()
            .downcast_ref::<VkBuffer>()
            .ok_or(Error::BufferTypeMismatch {
                expected: std::any::type_name::<VkBuffer>(),
                actual: std::any::type_name_of_val(dst),
            })?;
        if !Arc::ptr_eq(&self.inner, &dst.inner) {
            return Err(Error::DeviceMismatch { kind: "buffer" });
        }
        let end = offset.checked_add(size).ok_or(Error::RangeOutOfBounds {
            offset,
            end: u64::MAX,
            size: dst.size,
        })?;
        if end > dst.size {
            return Err(Error::RangeOutOfBounds {
                offset,
                end,
                size: dst.size,
            });
        }
        unsafe {
            self.inner
                .device
                .cmd_fill_buffer(self.cmd, dst.buffer, offset, size, 0);
        }
        Ok(())
    }

    fn write_timestamp(&mut self, set: &dyn TimestampSet, index: u32) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        let set =
            set.as_any()
                .downcast_ref::<VkTimestampSet>()
                .ok_or(Error::KernelTypeMismatch {
                    expected: std::any::type_name::<VkTimestampSet>(),
                    actual: std::any::type_name_of_val(set),
                })?;
        if !Arc::ptr_eq(&self.inner, &set.inner) {
            return Err(Error::DeviceMismatch {
                kind: "timestamp set",
            });
        }
        if index >= set.capacity() {
            return Err(Error::Vulkan(format!(
                "timestamp index {index} out of range 0..{}",
                set.capacity()
            )));
        }
        unsafe {
            self.inner.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                set.pool,
                index,
            );
        }
        Ok(())
    }

    fn resolve_timestamps(
        &mut self,
        set: &dyn TimestampSet,
        start: u32,
        count: u32,
        dst: &dyn Buffer,
        dst_offset: u64,
    ) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        let set =
            set.as_any()
                .downcast_ref::<VkTimestampSet>()
                .ok_or(Error::KernelTypeMismatch {
                    expected: std::any::type_name::<VkTimestampSet>(),
                    actual: std::any::type_name_of_val(set),
                })?;
        let dst = dst
            .as_any()
            .downcast_ref::<VkBuffer>()
            .ok_or(Error::BufferTypeMismatch {
                expected: std::any::type_name::<VkBuffer>(),
                actual: std::any::type_name_of_val(dst),
            })?;
        if !Arc::ptr_eq(&self.inner, &set.inner) || !Arc::ptr_eq(&self.inner, &dst.inner) {
            return Err(Error::DeviceMismatch { kind: "resource" });
        }
        let end = start
            .checked_add(count)
            .ok_or(Error::Vulkan("timestamp range overflows".to_string()))?;
        if end > set.capacity() {
            return Err(Error::Vulkan(format!(
                "timestamp range {start}..{end} out of range 0..{}",
                set.capacity()
            )));
        }
        unsafe {
            self.inner.device.cmd_copy_query_pool_results(
                self.cmd,
                set.pool,
                start,
                count,
                dst.buffer,
                dst_offset,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
        Ok(())
    }

    fn reset_timestamps(&mut self, set: &dyn TimestampSet, start: u32, count: u32) -> Result<()> {
        if !self.active {
            return Err(Error::EncoderInactive);
        }
        let set =
            set.as_any()
                .downcast_ref::<VkTimestampSet>()
                .ok_or(Error::KernelTypeMismatch {
                    expected: std::any::type_name::<VkTimestampSet>(),
                    actual: std::any::type_name_of_val(set),
                })?;
        if !Arc::ptr_eq(&self.inner, &set.inner) {
            return Err(Error::DeviceMismatch {
                kind: "timestamp set",
            });
        }
        let end = start
            .checked_add(count)
            .ok_or(Error::Vulkan("timestamp range overflows".to_string()))?;
        if end > set.capacity() {
            return Err(Error::Vulkan(format!(
                "timestamp range {start}..{end} out of range 0..{}",
                set.capacity()
            )));
        }
        unsafe {
            self.inner
                .device
                .cmd_reset_query_pool(self.cmd, set.pool, start, count);
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for VkEncoder {
    fn drop(&mut self) {
        unsafe {
            if self.active {
                self.inner
                    .device
                    .free_command_buffers(self.inner.pool, &[self.cmd]);
            }
            for pool in self.pools.drain(..) {
                self.inner.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}

pub struct VkSubmission {
    pub(crate) inner: Arc<VkDeviceInner>,
    pub(crate) cmd: vk::CommandBuffer,
    pub(crate) fence: vk::Fence,
    pub(crate) pools: Vec<vk::DescriptorPool>,
    pub(crate) waited: Cell<bool>,
}

impl Submission for VkSubmission {
    fn wait(&self) -> Result<()> {
        if !self.waited.get() {
            check(unsafe {
                self.inner
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
            })?;
            self.waited.set(true);
        }
        Ok(())
    }
}

impl Drop for VkSubmission {
    fn drop(&mut self) {
        if !self.waited.get() {
            let _ = unsafe {
                self.inner
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
            };
        }
        unsafe {
            self.inner
                .device
                .free_command_buffers(self.inner.pool, &[self.cmd]);
            self.inner.device.destroy_fence(self.fence, None);
            for pool in self.pools.drain(..) {
                let _ = self
                    .inner
                    .device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty());
                self.inner.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}
