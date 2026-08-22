use std::sync::mpsc;

use thuban_error::{Error, Result};

use crate::DeviceRef;

#[derive(Clone)]
pub struct Buffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) device: DeviceRef,
    pub(crate) id: u64,
    host_visible: bool,
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Buffer {
    pub(crate) fn new(buffer: wgpu::Buffer, device: DeviceRef, host_visible: bool) -> Self {
        Self {
            buffer,
            device,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            host_visible,
        }
    }

    pub fn same(&self, other: &Self) -> bool {
        self.id == other.id
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        assert!(
            offset + data.len() as u64 <= self.buffer.size(),
            "write exceeds buffer size"
        );
        self.device.queue.write_buffer(&self.buffer, offset, data);
        Ok(())
    }

    pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if !self.host_visible {
            return Err(Error::Gpu("buffer is not host visible".to_string()));
        }
        if out.is_empty() {
            return Ok(());
        }
        assert!(
            offset + out.len() as u64 <= self.buffer.size(),
            "read exceeds buffer size"
        );
        let slice = self.buffer.slice(offset..offset + out.len() as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        self.device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("buffer map poll failed: {e}")))?;
        rx.recv()
            .map_err(|_| Error::Gpu("buffer map callback dropped".to_string()))?
            .map_err(|e| Error::Gpu(format!("buffer map failed: {e}")))?;
        let view = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapped range unavailable: {e}")))?;
        out.copy_from_slice(&view);
        drop(view);
        self.buffer.unmap();
        Ok(())
    }
}
