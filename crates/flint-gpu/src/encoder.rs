use std::hash::Hasher;

use flint_error::{Error, Result};

use crate::buffer::Buffer;
use crate::kernel::Kernel;
use crate::query::TimestampSet;
use crate::DeviceRef;

pub struct BindingRef<'a> {
    pub index: u32,
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: u64,
}

pub struct Encoder {
    inner: DeviceRef,
    enc: wgpu::CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
}

impl Encoder {
    pub(crate) fn new(inner: DeviceRef) -> Result<Self> {
        Ok(Self {
            enc: inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            inner,
            pass: None,
        })
    }

    fn ensure_pass(&mut self) -> &mut wgpu::ComputePass<'static> {
        if self.pass.is_none() {
            self.pass = Some(
                self.enc
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    })
                    .forget_lifetime(),
            );
        }
        self.pass.as_mut().expect("compute pass just opened")
    }

    fn close_pass(&mut self) {
        self.pass = None;
    }

    pub fn bind(&mut self, kernel: &Kernel, bindings: &[BindingRef<'_>]) -> Result<()> {
        if bindings.len() != kernel.binding_count as usize {
            return Err(Error::Gpu(format!(
                "kernel {} expects {} bindings, got {}",
                kernel.name, kernel.binding_count, bindings.len()
            )));
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&kernel.pipeline, &mut hasher);
        for b in bindings {
            std::hash::Hash::hash(&b.index, &mut hasher);
            std::hash::Hash::hash(&b.buffer.buffer, &mut hasher);
            std::hash::Hash::hash(&b.offset, &mut hasher);
            std::hash::Hash::hash(&b.size, &mut hasher);
        }
        let key = hasher.finish();
        let group = {
            let mut groups = self.inner.bind_groups.lock().expect("bind group cache lock");
            match groups.get(&key) {
                Some(group) => group.clone(),
                None => {
                    let entries: Vec<wgpu::BindGroupEntry> = bindings
                        .iter()
                        .map(|b| wgpu::BindGroupEntry {
                            binding: b.index,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &b.buffer.buffer,
                                offset: b.offset,
                                size: (b.size > 0).then(|| {
                                    wgpu::BufferSize::new(b.size)
                                        .expect("binding size is validated by the caller")
                                }),
                            }),
                        })
                        .collect();
                    let group = self
                        .inner
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &kernel.bind_group_layout,
                            entries: &entries,
                        });
                    if groups.len() >= 65536 {
                        groups.clear();
                    }
                    groups.insert(key, group.clone());
                    group
                }
            }
        };
        let pass = self.ensure_pass();
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        Ok(())
    }

    pub fn set_scalars(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        self.ensure_pass().set_immediates(0, bytes);
        Ok(())
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) -> Result<()> {
        self.ensure_pass()
            .dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }

    pub fn copy(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        self.close_pass();
        self.enc
            .copy_buffer_to_buffer(&src.buffer, src_offset, &dst.buffer, dst_offset, size);
        Ok(())
    }

    pub fn clear(&mut self, dst: &Buffer, offset: u64, size: u64) -> Result<()> {
        self.close_pass();
        self.enc
            .clear_buffer(&dst.buffer, offset, (size > 0).then_some(size));
        Ok(())
    }

    pub fn write_timestamp(&mut self, set: &TimestampSet, index: u32) -> Result<()> {
        self.close_pass();
        self.enc.write_timestamp(&set.query_set, index);
        Ok(())
    }

    pub fn resolve_timestamps(
        &mut self,
        set: &TimestampSet,
        start: u32,
        count: u32,
        dst: &Buffer,
        dst_offset: u64,
    ) -> Result<()> {
        self.close_pass();
        self.enc
            .resolve_query_set(&set.query_set, start..start + count, &dst.buffer, dst_offset);
        Ok(())
    }

    pub fn finish(mut self) -> Submission {
        self.pass = None;
        let cmd = self.enc.finish();
        let index = self.inner.queue.submit([cmd]);
        Submission {
            inner: self.inner.clone(),
            index,
        }
    }

    pub fn submit_and_reset(&mut self) -> Submission {
        self.pass = None;
        let enc = std::mem::replace(
            &mut self.enc,
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
        );
        let cmd = enc.finish();
        let index = self.inner.queue.submit([cmd]);
        Submission {
            inner: self.inner.clone(),
            index,
        }
    }
}

pub struct Submission {
    inner: DeviceRef,
    index: wgpu::SubmissionIndex,
}

impl Submission {
    pub fn wait(&self) -> Result<()> {
        self.inner
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(self.index.clone()),
                timeout: None,
            })
            .map_err(|e| Error::Gpu(format!("submission wait failed: {e}")))?;
        Ok(())
    }
}
