mod gemm;

use std::rc::Rc;

use saturn_api::{BackendKind, open as open_device};
use saturn_core::{BindingRef, Buffer, BufferSpec, CommandEncoder, Device};

use flint_error::{Error, Result};
use flint_kernel::{Kernels, name};
use flint_tensor::{DType, Tensor};

#[derive(Clone, Copy)]
pub enum Binding<'a> {
    Full(&'a Tensor),
    Slice(&'a Tensor, u64, u64),
}

impl<'a> Binding<'a> {
    fn resolve(&self, index: u32) -> BindingRef<'a> {
        match self {
            Binding::Full(t) => BindingRef {
                index,
                buffer: t.buf.as_ref(),
                offset: 0,
                size: 0,
            },
            Binding::Slice(t, off, size) => BindingRef {
                index,
                buffer: t.buf.as_ref(),
                offset: *off,
                size: *size,
            },
        }
    }
}

pub struct Pass<'a>(pub(crate) &'a mut dyn CommandEncoder);

impl<'a> Pass<'a> {
    pub fn begin(encoder: &'a mut dyn CommandEncoder) -> Self {
        Self(encoder)
    }

    pub fn raw(&mut self) -> &mut dyn CommandEncoder {
        self.0
    }
}

pub struct Backend {
    device: Rc<dyn Device>,
    kernels: Kernels,
    dummy_scale: Tensor,
    gemv_partial: Tensor,
    gemm_partial: Tensor,

    pending: Vec<Box<dyn saturn_core::Submission>>,
    retired: Vec<(Tensor, u32)>,
}

impl Backend {
    pub fn new() -> Result<Self> {
        let kind = if cfg!(target_os = "macos") {
            BackendKind::Metal
        } else {
            BackendKind::Vulkan
        };
        let device: Rc<dyn Device> = Rc::from(
            open_device(kind).map_err(|e| Error::Gpu(format!("no suitable backend: {e}")))?,
        );
        let kernels = Kernels::new(device.as_ref())?;
        Self::warmup(device.as_ref(), &kernels)?;
        let dummy_scale = Tensor::new(Self::zeroed_buf(device.as_ref(), 4), vec![1], DType::F32);
        let gemv_partial = Self::partial_buf(device.as_ref(), 8 * 65536)?;
        let gemm_partial = Self::partial_buf(device.as_ref(), 4 * 128 * 16384)?;
        Ok(Self {
            device,
            kernels,
            dummy_scale,
            gemv_partial,
            gemm_partial,
            pending: Vec::new(),
            retired: Vec::new(),
        })
    }

    pub fn device(&self) -> Rc<dyn Device> {
        self.device.clone()
    }

    pub fn adapter_name(&self) -> &str {
        self.device.name()
    }

    pub fn kernel(&self, name: &str) -> Result<&dyn saturn_core::Kernel> {
        self.kernels.get(name)
    }

    pub fn pack_scalars(&self, name: &str, consts: &[(&'static str, f64)]) -> Result<Vec<u8>> {
        self.kernels.pack_scalars(name, consts)
    }

    fn zeroed_buf(device: &dyn Device, size: u64) -> Box<dyn Buffer> {
        let buf = device
            .create_buffer(&BufferSpec {
                size,
                host_visible: false,
            })
            .expect("buffer allocation");
        let mut enc = device.encoder().expect("encoder");
        enc.clear(buf.as_ref(), 0, size).expect("clear");
        let sub = device.submit(enc).expect("submit");
        sub.wait().expect("wait");
        buf
    }

    fn partial_buf(device: &dyn Device, words: usize) -> Result<Tensor> {
        Ok(Tensor::new(
            device
                .create_buffer(&BufferSpec {
                    size: words as u64 * 4,
                    host_visible: false,
                })
                .map_err(|e| Error::Gpu(e.to_string()))?,
            vec![words as u32],
            DType::F32,
        ))
    }

    fn retire(&mut self, tensor: Tensor) {
        let refs = self.pending.len() as u32 + 1;
        self.retired.push((tensor, refs));
    }

    fn settle_retired(&mut self) {
        let mut i = 0;
        while i < self.retired.len() {
            self.retired[i].1 -= 1;
            if self.retired[i].1 == 0 {
                self.retired.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn storage(&self, size: u64) -> Box<dyn Buffer> {
        self.device
            .create_buffer(&BufferSpec {
                size,
                host_visible: false,
            })
            .expect("buffer allocation")
    }

    pub fn zero_tensor(&self, shape: &[u32]) -> Tensor {
        let numel: u64 = shape.iter().map(|d| *d as u64).product();
        Tensor::new(
            Self::zeroed_buf(self.device.as_ref(), numel * 4),
            shape.to_vec(),
            DType::F32,
        )
    }

    pub fn zero_bf16_tensor(&self, shape: &[u32]) -> Tensor {
        let numel: u64 = shape.iter().map(|d| *d as u64).product();
        Tensor::new(
            Self::zeroed_buf(self.device.as_ref(), numel * 2),
            shape.to_vec(),
            DType::Bf16,
        )
    }

    pub fn zero_fill(&self, t: &Tensor) {
        let mut enc = self.device.encoder().expect("encoder");
        enc.clear(t.buf.as_ref(), 0, t.byte_len()).expect("clear");
        let sub = self.device.submit(enc).expect("submit");
        sub.wait().expect("wait");
    }

    pub fn copy(&self, src: &Tensor, dst: &Tensor) {
        assert_eq!(src.byte_len(), dst.byte_len(), "copy size mismatch");
        let mut enc = self.device.encoder().expect("encoder");
        enc.copy(src.buf.as_ref(), 0, dst.buf.as_ref(), 0, src.byte_len())
            .expect("copy");
        let sub = self.device.submit(enc).expect("submit");
        sub.wait().expect("wait");
    }

    pub fn tensor_f32(&self, data: &[f32], shape: Vec<u32>) -> Tensor {
        let buf = self.upload(bytemuck::cast_slice(data));
        Tensor::new(buf, shape, DType::F32)
    }

    pub fn tensor_bf16(&self, bytes: &[u8], shape: Vec<u32>) -> Result<Tensor> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::Model("odd bf16 byte count".to_string()));
        }
        let packed: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let padded = if bytes.len() % 4 == 2 {
            let mut v = packed;
            let last = &bytes[bytes.len() - 2..];
            v.push(u32::from_le_bytes([last[0], last[1], 0, 0]));
            v
        } else {
            packed
        };
        let buf = self.upload(bytemuck::cast_slice(&padded));
        Ok(Tensor::new(buf, shape, DType::Bf16))
    }

    pub fn tensor_i8(&self, bytes: &[u8], shape: Vec<u32>) -> Tensor {
        assert!(
            bytes.len().is_multiple_of(4),
            "i8 count not a multiple of 4"
        );
        let buf = self.upload(bytes);
        Tensor::new(buf, shape, DType::I8)
    }

    fn upload(&self, bytes: &[u8]) -> Box<dyn Buffer> {
        Self::upload_buf(self.device.as_ref(), bytes)
    }

    fn upload_buf(device: &dyn Device, bytes: &[u8]) -> Box<dyn Buffer> {
        let dst = device
            .create_buffer(&BufferSpec {
                size: bytes.len() as u64,
                host_visible: false,
            })
            .expect("buffer allocation");
        Self::stage_copy(device, dst.as_ref(), bytes);
        dst
    }

    pub fn write_u32(&self, buf: &dyn Buffer, data: &[u32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::stage_copy(self.device.as_ref(), buf, &bytes);
    }

    pub fn write_f32(&self, buf: &dyn Buffer, data: &[f32]) {
        Self::stage_copy(self.device.as_ref(), buf, bytemuck::cast_slice(data));
    }

    fn stage_copy(device: &dyn Device, dst: &dyn Buffer, bytes: &[u8]) {
        let staging = device
            .create_buffer(&BufferSpec {
                size: bytes.len() as u64,
                host_visible: true,
            })
            .expect("staging allocation");
        staging.write(0, bytes).expect("staging write");
        let mut enc = device.encoder().expect("encoder");
        enc.copy(staging.as_ref(), 0, dst, 0, bytes.len() as u64)
            .expect("upload copy");
        let sub = device.submit(enc).expect("submit");
        sub.wait().expect("wait");
    }

    pub fn encoder(&self) -> Result<Box<dyn CommandEncoder>> {
        Ok(self.device.encoder()?)
    }

    pub fn dispatch(
        &mut self,
        pass: &mut Pass<'_>,
        name: &'static str,
        consts: &[(&'static str, f64)],
        bufs: &[Binding<'_>],
        groups: [u32; 3],
    ) -> Result<()> {
        Self::set(&self.kernels, pass, name, consts, bufs, groups)
    }

    fn set(
        kernels: &Kernels,
        pass: &mut Pass<'_>,
        name: &'static str,
        consts: &[(&'static str, f64)],
        bufs: &[Binding<'_>],
        groups: [u32; 3],
    ) -> Result<()> {
        let kernel = kernels.get(name)?;
        let bindings: Vec<BindingRef> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| b.resolve(i as u32))
            .collect();
        let scalars = kernels.pack_scalars(name, consts)?;
        pass.0.barrier()?;
        pass.0.bind(kernel, &bindings)?;
        if !scalars.is_empty() {
            pass.0.set_scalars(kernel, &scalars)?;
        }
        pass.0.dispatch(groups)?;
        Ok(())
    }

    pub fn dummy_scale(&self) -> &Tensor {
        &self.dummy_scale
    }

    pub fn submit(&mut self, encoder: Box<dyn CommandEncoder>) -> Result<()> {
        if self.pending.len() >= 2 {
            let done = self.pending.remove(0);
            done.wait()?;
            self.settle_retired();
        }
        self.pending.push(self.device.submit(encoder)?);
        Ok(())
    }

    pub fn read_f32(&self, src: &dyn Buffer, offset: u64, count: usize) -> Result<Vec<f32>> {
        let staging = self.device.create_buffer(&BufferSpec {
            size: count as u64 * 4,
            host_visible: true,
        })?;
        let mut enc = self.device.encoder()?;
        enc.copy(src, offset, staging.as_ref(), 0, count as u64 * 4)?;
        let sub = self.device.submit(enc)?;
        sub.wait()?;
        let mut bytes = vec![0u8; count * 4];
        staging.read(0, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn warmup(device: &dyn Device, kernels: &Kernels) -> Result<()> {
        let n = 1 << 20;
        let mk = |val: f32| {
            Tensor::new(
                Self::upload_buf(device, bytemuck::cast_slice(&vec![val; n])),
                vec![n as u32],
                DType::F32,
            )
        };
        let wxa = mk(1.0);
        let wxb = mk(1.0);
        let wy = Tensor::new(
            Self::zeroed_buf(device, n as u64 * 4),
            vec![n as u32],
            DType::F32,
        );
        for _ in 0..3 {
            let mut enc = device.encoder()?;
            {
                let mut pass = Pass::begin(enc.as_mut());
                Self::set(
                    kernels,
                    &mut pass,
                    name::ADD,
                    &[("N_ELEM", n as f64)],
                    &[Binding::Full(&wxa), Binding::Full(&wxb), Binding::Full(&wy)],
                    [(n / 256) as u32, 1, 1],
                )?;
            }
            let sub = device.submit(enc)?;
            sub.wait()?;
        }
        Ok(())
    }
}
