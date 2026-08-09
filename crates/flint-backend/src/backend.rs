use saturn_api::{BackendKind, open as open_device};
use saturn_core::{BindingRef, Buffer, BufferSpec, CommandEncoder, Device};

use flint_error::{Error, Result};
use flint_kernel::{Kernels, name};
use flint_profiler::{ProfileRow, Profiler};
use flint_tensor::{DType, Tensor, Weight};

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
    device: Box<dyn Device>,
    kernels: Kernels,
    dummy_scale: Tensor,
    gemv_partial: Tensor,
    gemm_partial: Tensor,

    profiler: Option<Profiler>,

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
        let device = open_device(kind)
            .map_err(|e| Error::Gpu(format!("no suitable backend: {e}")))?;
        let kernels = Kernels::new(device.as_ref())?;
        Self::warmup(device.as_ref(), &kernels)?;
        let dummy_scale = Tensor::new(Self::zeroed_buf(device.as_ref(), 4), vec![1], DType::F32);
        let profiler = if std::env::var("FLINT_PROFILE").is_ok() {
            Some(Profiler::new(device.as_ref(), 4096)?)
        } else {
            None
        };
        let gemv_partial = Self::partial_buf(device.as_ref(), 8 * 65536)?;
        let gemm_partial = Self::partial_buf(device.as_ref(), 4 * 128 * 16384)?;
        Ok(Self {
            device,
            kernels,
            dummy_scale,
            gemv_partial,
            gemm_partial,
            profiler,
            pending: Vec::new(),
            retired: Vec::new(),
        })
    }

    pub fn adapter_name(&self) -> &str {
        self.device.name()
    }

    pub fn kernel(&self, name: &str) -> Result<&dyn saturn_core::Kernel> {
        self.kernels.get(name)
    }

    pub fn pack_scalars(
        &self,
        name: &str,
        consts: &[(&'static str, f64)],
    ) -> Result<Vec<u8>> {
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

    pub fn dummy_scale(&self) -> &Tensor {
        &self.dummy_scale
    }

    fn scale_binding<'a>(dummy: &'a Tensor, w: &'a Weight) -> Binding<'a> {
        match w.scale() {
            Some(s) => Binding::Full(s),
            None => Binding::Full(dummy),
        }
    }

    fn ensure_gemm_partial(&mut self, words: u32) {
        if words > self.gemm_partial.numel() as u32 {
            let old = std::mem::replace(
                &mut self.gemm_partial,
                Self::partial_buf(self.device.as_ref(), words as usize)
                    .expect("gemm partial growth"),
            );
            self.retire(old);
        }
    }

    fn ensure_gemv_partial(&mut self, words: u32) {
        if words > self.gemv_partial.numel() as u32 {
            let old = std::mem::replace(
                &mut self.gemv_partial,
                Self::partial_buf(self.device.as_ref(), words as usize)
                    .expect("gemv partial growth"),
            );
            self.retire(old);
        }
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
            DType::Bf16Packed,
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
        Ok(Tensor::new(buf, shape, DType::Bf16Packed))
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
        let staging = device
            .create_buffer(&BufferSpec {
                size: bytes.len() as u64,
                host_visible: true,
            })
            .expect("staging allocation");
        staging.write(0, bytes).expect("staging write");
        let dst = device
            .create_buffer(&BufferSpec {
                size: bytes.len() as u64,
                host_visible: false,
            })
            .expect("buffer allocation");
        let mut enc = device.encoder().expect("encoder");
        enc.copy(staging.as_ref(), 0, dst.as_ref(), 0, bytes.len() as u64)
            .expect("upload copy");
        let sub = device.submit(enc).expect("submit");
        sub.wait().expect("wait");
        dst
    }

    pub fn write_u32(&self, buf: &dyn Buffer, data: &[u32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_to(buf, &bytes);
    }

    pub fn write_f32(&self, buf: &dyn Buffer, data: &[f32]) {
        self.upload_to(buf, bytemuck::cast_slice(data));
    }

    fn upload_to(&self, dst: &dyn Buffer, bytes: &[u8]) {
        Self::upload_to_buf(self.device.as_ref(), dst, bytes)
    }

    fn upload_to_buf(device: &dyn Device, dst: &dyn Buffer, bytes: &[u8]) {
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
        let span = self.prof_begin(pass)?;
        Self::set(&self.kernels, pass, name, consts, bufs, groups)?;
        self.prof_end(pass, name, span)?;
        Ok(())
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

    fn weight_io(w: &Weight) -> (u32, u32, Binding<'_>, DType) {
        assert_eq!(
            w.tensor().shape.len(),
            2,
            "gemm weight must be a [N, K] matrix"
        );
        let (n, k) = (w.tensor().shape[0], w.tensor().shape[1]);
        let dtype = match w.tensor().dtype {
            DType::Bf16Packed | DType::I8 => w.tensor().dtype,
            DType::F32 | DType::U32 => {
                unreachable!("gemm operands are weights, never index tensors")
            }
        };
        (n, k, Binding::Full(w.tensor()), dtype)
    }

    pub fn gemm(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
    ) -> Result<()> {
        let n = w.tensor().shape[0];
        self.gemm_strided(pass, x, w, y, rows, false, 0, n)
    }

    pub fn gemm_acc(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
        acc: bool,
    ) -> Result<()> {
        let n = w.tensor().shape[0];
        self.gemm_strided(pass, x, w, y, rows, acc, 0, n)
    }

    pub fn gemm_strided(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
        acc: bool,
        y_off: u32,
        y_stride: u32,
    ) -> Result<()> {
        let (n, k, wb, dtype) = Self::weight_io(w);
        assert!(
            k.is_multiple_of(32),
            "gemm K {k} is not a multiple of the BK=32 tile"
        );
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("M", rows as f64),
            ("WDTYPE", dtype_flag(dtype)),
            ("GROUP", group_const(w)),
            ("ACC", acc as u32 as f64),
            ("Y_STRIDE", y_stride as f64),
            ("Y_OFF", y_off as f64),
        ];
        let segs = if rows > 1 && k >= 8192 && k.is_multiple_of(128) {
            4
        } else {
            1
        };
        let consts = consts.into_iter().chain([("SEGS", segs as f64)]).collect::<Vec<_>>();
        let span = self.prof_begin(pass)?;
        let (yb, yslice) = if segs > 1 {
            self.ensure_gemm_partial(segs * rows * y_stride);
            (
                Binding::Slice(&self.gemm_partial, 0, (segs * rows * y_stride) as u64 * 4),
                y,
            )
        } else {
            (y, y)
        };
        let bufs = [x, wb, Self::scale_binding(&self.dummy_scale, w), yb];
        Self::set(
            &self.kernels,
            pass,
            name::GEMM,
            &consts,
            &bufs,
            [n.div_ceil(32), rows.div_ceil(32), segs],
        )?;
        if segs > 1 {
            let mconsts = [
                ("M", rows as f64),
                ("N", n as f64),
                ("Y_STRIDE", y_stride as f64),
                ("Y_OFF", y_off as f64),
                ("SEGS", segs as f64),
                ("ACC", acc as u32 as f64),
            ];
            let bufs = [
                Binding::Slice(&self.gemm_partial, 0, (segs * rows * y_stride) as u64 * 4),
                yslice,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_GEMM,
                &mconsts,
                &bufs,
                [n.div_ceil(256), 1, 1],
            )?;
        }
        self.prof_end(pass, name::GEMM, span)?;
        Ok(())
    }

    pub fn gemv(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
    ) -> Result<()> {
        self.gemv_acc(pass, x, w, y, false)
    }

    pub fn gemv_acc(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        acc: bool,
    ) -> Result<()> {
        let (n, k, wb, dtype) = Self::weight_io(w);
        let kb = k / 16;
        let base = n.div_ceil(256);
        let segs: u32 = [8u32, 4, 2, 1]
            .into_iter()
            .find(|s| kb % *s == 0 && base * *s >= 96)
            .unwrap_or(1);
        let span = self.prof_begin(pass)?;
        if segs > 1 {
            self.ensure_gemv_partial(n * segs);
        }
        let scale = Self::scale_binding(&self.dummy_scale, w);
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", dtype_flag(dtype)),
            ("GROUP", group_const(w)),
            ("SEGS", segs as f64),
            ("ACC", acc as u32 as f64),
        ];
        let out = if segs == 1 {
            y
        } else {

            Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64)
        };
        let bufs = [x, wb, scale, out];
        Self::set(
            &self.kernels,
            pass,
            name::GEMV,
            &consts,
            &bufs,
            [n.div_ceil(256), segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64),
                y,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_GEMV,
                &[
                    ("N", n as f64),
                    ("SEGS", segs as f64),
                    ("ACC", acc as u32 as f64),
                ],
                &bufs,
                [n.div_ceil(256), 1, 1],
            )?;
        }
        self.prof_end(pass, name::GEMV, span)?;
        Ok(())
    }

    pub fn gemv_qkv(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        wq: &Weight,
        wk: &Weight,
        wv: &Weight,
        yq: Binding<'_>,
        yk: Binding<'_>,
        yv: Binding<'_>,
        nq: u32,
        nk: u32,
        nv: u32,
        k: u32,
    ) -> Result<()> {
        assert!(
            wq.scale().is_some() && wk.scale().is_some() && wv.scale().is_some(),
            "gemv_qkv requires quantized weights"
        );
        let ntot = nq + nk + nv;
        let kb = k / 16;
        let base = ntot.div_ceil(256);
        let segs: u32 = [16u32, 8, 4, 2, 1]
            .into_iter()
            .find(|s| kb % *s == 0 && base * *s >= 256)
            .unwrap_or(1);
        let span = self.prof_begin(pass)?;
        if segs > 1 {
            self.ensure_gemv_partial(ntot * segs);
        }
        let (sq, sk, sv) = (
            Self::scale_binding(&self.dummy_scale, wq),
            Self::scale_binding(&self.dummy_scale, wk),
            Self::scale_binding(&self.dummy_scale, wv),
        );
        let out = if segs == 1 {
            yq
        } else {
            Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64)
        };
        let bufs = [
            x,
            Binding::Full(wq.tensor()),
            sq,
            yq,
            Binding::Full(wk.tensor()),
            sk,
            yk,
            Binding::Full(wv.tensor()),
            sv,
            yv,
            out,
        ];
        Self::set(
            &self.kernels,
            pass,
            name::GEMV_QKV,
            &[
                ("NQ", nq as f64),
                ("NK", nk as f64),
                ("NV", nv as f64),
                ("K", k as f64),
                ("GROUP", group_const(wq)),
                ("SEGS", segs as f64),
            ],
            &bufs,
            [ntot.div_ceil(256), segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64),
                yq,
                yk,
                yv,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_QKV,
                &[
                    ("NQ", nq as f64),
                    ("NK", nk as f64),
                    ("NV", nv as f64),
                    ("SEGS", segs as f64),
                ],
                &bufs,
                [ntot.div_ceil(256), 1, 1],
            )?;
        }
        self.prof_end(pass, name::GEMV_QKV, span)?;
        Ok(())
    }

    pub fn gemv_gateup(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        wg: &Weight,
        wu: &Weight,
        yg: Binding<'_>,
        yu: Binding<'_>,
        n: u32,
        k: u32,
    ) -> Result<()> {
        assert!(
            wg.scale().is_some() && wu.scale().is_some(),
            "gemv_gateup requires quantized weights"
        );
        let ntot = 2 * n;
        let kb = k / 16;
        let base = ntot.div_ceil(256);
        let segs: u32 = [4u32, 2, 1]
            .into_iter()
            .find(|s| kb % *s == 0 && base * *s >= 192)
            .unwrap_or(1);
        let span = self.prof_begin(pass)?;
        if segs > 1 {
            self.ensure_gemv_partial(ntot * segs);
        }
        let (sg, su) = (
            Self::scale_binding(&self.dummy_scale, wg),
            Self::scale_binding(&self.dummy_scale, wu),
        );
        let out = if segs == 1 {
            yg
        } else {
            Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64)
        };
        let bufs = [
            x,
            Binding::Full(wg.tensor()),
            sg,
            yg,
            Binding::Full(wu.tensor()),
            su,
            yu,
            out,
        ];
        Self::set(
            &self.kernels,
            pass,
            name::GEMV_GATEUP,
            &[
                ("NG", n as f64),
                ("K", k as f64),
                ("GROUP", group_const(wg)),
                ("SEGS", segs as f64),
            ],
            &bufs,
            [ntot.div_ceil(256), segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64),
                yg,
                yu,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_GATEUP,
                &[("NG", n as f64), ("SEGS", segs as f64)],
                &bufs,
                [ntot.div_ceil(256), 1, 1],
            )?;
        }
        self.prof_end(pass, name::GEMV_GATEUP, span)?;
        Ok(())
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

    fn prof_begin(&mut self, pass: &mut Pass<'_>) -> Result<Option<u32>> {
        match &mut self.profiler {
            Some(p) => p.begin(pass.0),
            None => Ok(None),
        }
    }

    fn prof_end(
        &mut self,
        pass: &mut Pass<'_>,
        label: &'static str,
        span: Option<u32>,
    ) -> Result<()> {
        if let Some(p) = &mut self.profiler {
            p.end(pass.0, label, span)?;
        }
        Ok(())
    }

    pub fn flush_profile(&mut self) -> Result<()> {
        let Some(prof) = self.profiler.as_mut() else {
            return Ok(());
        };
        if prof.pending() == 0 {
            return Ok(());
        }
        let mut enc = self.device.encoder()?;
        prof.resolve(enc.as_mut())?;
        let sub = self.device.submit(enc)?;
        sub.wait()?;
        prof.accumulate()
    }

    pub fn profiling(&self) -> bool {
        self.profiler.is_some()
    }

    pub fn profile_report(&self) -> Vec<ProfileRow> {
        self.profiler
            .as_ref()
            .map(|p| p.report())
            .unwrap_or_default()
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
        let wy = Tensor::new(Self::zeroed_buf(device, n as u64 * 4), vec![n as u32], DType::F32);
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

fn dtype_flag(dtype: DType) -> f64 {
    match dtype {
        DType::Bf16Packed => 0.0,
        DType::I8 => 1.0,
        DType::F32 | DType::U32 => unreachable!("gemm weights are bf16-packed or i8"),
    }
}

fn group_const(w: &Weight) -> f64 {
    match w.group() {
        Some(group) => group as f64,
        None => 0.0,
    }
}
