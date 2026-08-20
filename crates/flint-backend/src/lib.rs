mod matmul;

use std::rc::Rc;

use flint_error::{Error, Result};
use flint_gpu::{BindingRef, Buffer, Device, Encoder, HostAccess, Kernel, Submission};
use flint_kernel::Kernels;
use flint_tensor::{DType, Tensor};

pub use flint_kernel::name as shader;
pub use flint_kernel::{Act, ATTN_BR, NormMode, PAGE_LEN};

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
                buffer: &t.buf,
                offset: 0,
                size: 0,
            },
            Binding::Slice(t, off, size) => BindingRef {
                index,
                buffer: &t.buf,
                offset: *off,
                size: *size,
            },
        }
    }

    fn sub_slice(&self, off: u64, size: u64) -> Binding<'a> {
        match self {
            Binding::Full(t) => Binding::Slice(t, off, size),
            Binding::Slice(t, base, _) => Binding::Slice(t, base + off, size),
        }
    }
}

pub struct Commands<'a>(pub(crate) &'a mut Encoder);

impl<'a> Commands<'a> {
    pub fn begin(encoder: &'a mut Encoder) -> Self {
        Self(encoder)
    }

    pub fn raw(&mut self) -> &mut Encoder {
        self.0
    }
}

pub struct Backend {
    device: Rc<Device>,
    kernels: Kernels,
    unit_scale: Tensor,
    gemv_partial: Tensor,
    gemm_partial: Tensor,
    gemm_xf16: Tensor,
    read_staging: std::cell::RefCell<(Buffer, u64)>,
    profiler: Option<Rc<std::cell::RefCell<flint_profiler::GpuProfiler>>>,

    pending: Vec<Submission>,
    retired: Vec<(Tensor, u32)>,
}

impl Backend {
    pub fn new() -> Result<Self> {
        let device = Rc::new(
            Device::open()
                .map_err(|e| Error::Gpu(format!("no suitable backend: {e}")))?,
        );
        let kernels = Kernels::new(device.as_ref())?;
        Self::warmup(device.as_ref(), &kernels)?;
        let unit_scale = Tensor::new(Self::zeroed_buf(device.as_ref(), 4), vec![1], DType::F32);
        let gemv_partial = Self::partial_buf(device.as_ref(), 8 * 65536)?;
        let gemm_partial = Self::partial_buf(device.as_ref(), 4 * 128 * 16384)?;
        let gemm_xf16 = Self::partial_f16_buf(device.as_ref(), 128 * 8192)?;
        let read_staging = device
            .create_buffer(1 << 20, HostAccess::Read, false)
            .map_err(|e| Error::Gpu(e.to_string()))?;
        Ok(Self {
            device,
            kernels,
            unit_scale,
            gemv_partial,
            gemm_partial,
            gemm_xf16,
            read_staging: std::cell::RefCell::new((read_staging, 1 << 20)),
            profiler: None,
            pending: Vec::new(),
            retired: Vec::new(),
        })
    }

    pub fn device(&self) -> Rc<Device> {
        self.device.clone()
    }

    pub fn adapter_name(&self) -> &str {
        self.device.name()
    }

    pub fn kernel(&self, name: &str) -> Result<&Kernel> {
        self.kernels.get(name)
    }

    pub fn pack_scalars(&self, name: &str, consts: &[(&'static str, f64)]) -> Result<Vec<u8>> {
        self.kernels.pack_scalars(name, consts)
    }

    fn zeroed_buf(device: &Device, size: u64) -> Buffer {
        let buf = device
            .create_buffer(size, HostAccess::None, false)
            .expect("buffer allocation");
        let mut enc = device.encoder().expect("encoder");
        enc.clear(&buf, 0, size).expect("clear");
        enc.finish().wait().expect("wait");
        buf
    }

    fn partial_buf(device: &Device, words: usize) -> Result<Tensor> {
        Ok(Tensor::new(
            device
                .create_buffer(words as u64 * 4, HostAccess::None, false)
                .map_err(|e| Error::Gpu(e.to_string()))?,
            vec![words as u32],
            DType::F32,
        ))
    }

    fn partial_f16_buf(device: &Device, words: usize) -> Result<Tensor> {
        Ok(Tensor::new(
            device
                .create_buffer(words as u64 * 2, HostAccess::None, false)
                .map_err(|e| Error::Gpu(e.to_string()))?,
            vec![words as u32],
            DType::F16,
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

    pub fn storage(&self, size: u64) -> Buffer {
        self.device
            .create_buffer(size, HostAccess::None, false)
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

    pub fn zero_f16_tensor(&self, shape: &[u32]) -> Tensor {
        let numel: u64 = shape.iter().map(|d| *d as u64).product();
        Tensor::new(
            Self::zeroed_buf(self.device.as_ref(), numel * 2),
            shape.to_vec(),
            DType::F16,
        )
    }

    pub fn zero_fill(&self, t: &Tensor) {
        let mut enc = self.device.encoder().expect("encoder");
        enc.clear(&t.buf, 0, t.byte_len()).expect("clear");
        enc.finish().wait().expect("wait");
    }

    pub fn copy(&self, src: &Tensor, dst: &Tensor) {
        assert_eq!(src.byte_len(), dst.byte_len(), "copy size mismatch");
        let mut enc = self.device.encoder().expect("encoder");
        enc.copy(&src.buf, 0, &dst.buf, 0, src.byte_len())
            .expect("copy");
        enc.finish().wait().expect("wait");
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

    fn upload(&self, bytes: &[u8]) -> Buffer {
        Self::upload_buf(self.device.as_ref(), bytes)
    }

    fn upload_buf(device: &Device, bytes: &[u8]) -> Buffer {
        let dst = device
            .create_buffer(bytes.len() as u64, HostAccess::None, false)
            .expect("buffer allocation");
        dst.write(0, bytes).expect("buffer write");
        dst
    }

    pub fn write_u32(&self, buf: &Buffer, data: &[u32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        buf.write(0, &bytes).expect("buffer write");
    }

    pub fn write_f32(&self, buf: &Buffer, data: &[f32]) {
        buf.write(0, bytemuck::cast_slice(data))
            .expect("buffer write");
    }

    pub fn encoder(&self) -> Result<Encoder> {
        self.device.encoder()
    }

    pub fn attach_profiler(
        &mut self,
        profiler: Rc<std::cell::RefCell<flint_profiler::GpuProfiler>>,
    ) {
        self.profiler = Some(profiler);
    }

    pub fn dispatch(
        &mut self,
        commands: &mut Commands<'_>,
        name: &'static str,
        consts: &[(&'static str, f64)],
        bufs: &[Binding<'_>],
        groups: [u32; 3],
    ) -> Result<()> {
        match &self.profiler {
            Some(profiler) => {
                let span = profiler.borrow_mut().mark_begin(commands.raw())?;
                Self::set(&self.kernels, commands, name, consts, bufs, groups)?;
                profiler
                    .borrow_mut()
                    .mark_end(commands.raw(), name, span)
            }
            None => Self::set(&self.kernels, commands, name, consts, bufs, groups),
        }
    }

    fn set(
        kernels: &Kernels,
        commands: &mut Commands<'_>,
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
        commands.0.bind(kernel, &bindings)?;
        if !scalars.is_empty() {
            commands.0.set_scalars(&scalars)?;
        }
        commands.0.dispatch(groups)?;
        Ok(())
    }

    pub fn unit_scale(&self) -> &Tensor {
        &self.unit_scale
    }

    pub fn submit(&mut self, encoder: &mut Encoder) -> Result<()> {
        if self.pending.len() >= 2 {
            let done = self.pending.remove(0);
            done.wait()?;
            self.settle_retired();
        }
        self.pending.push(encoder.submit_and_reset());
        Ok(())
    }

    pub fn read_f32(&self, src: &Buffer, offset: u64, count: usize) -> Result<Vec<f32>> {
        let bytes = count as u64 * 4;
        let mut staging = self.read_staging.borrow_mut();
        if staging.1 < bytes {
            *staging = (
                self.device
                    .create_buffer(bytes, HostAccess::Read, false)
                    .map_err(|e| Error::Gpu(e.to_string()))?,
                bytes,
            );
        }
        let mut enc = self.device.encoder()?;
        enc.copy(src, offset, &staging.0, 0, bytes)?;
        enc.finish().wait()?;
        let mut out = vec![0u8; count * 4];
        staging.0.read(0, &mut out)?;
        Ok(out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn warmup(device: &Device, kernels: &Kernels) -> Result<()> {
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
                let mut commands = Commands::begin(&mut enc);
                Self::set(
                    kernels,
                    &mut commands,
                    shader::ADD,
                    &[("N_ELEM", n as f64)],
                    &[Binding::Full(&wxa), Binding::Full(&wxb), Binding::Full(&wy)],
                    [(n / 256) as u32, 1, 1],
                )?;
            }
            enc.finish().wait()?;
        }
        Ok(())
    }
}
