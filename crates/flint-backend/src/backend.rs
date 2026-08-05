use std::collections::HashMap;
use std::sync::mpsc;

use wgpu::{
    BindGroup, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferBinding,
    CommandEncoder, CommandEncoderDescriptor, Device, Instance, InstanceDescriptor, Limits,
    MapMode, MemoryHints, PollType, PowerPreference, Queue, RequestAdapterOptions, util::DeviceExt,
};

use flint_error::{Error, Result};
use flint_kernel::{Kernels, name};
use flint_profiler::{ProfileRow, Profiler};
use flint_tensor::{DType, Tensor, Weight};

/// A whole tensor or a byte-aligned sub-slice of it.
#[derive(Clone, Copy)]
pub enum Binding<'a> {
    Full(&'a Tensor),
    Slice(&'a Tensor, BufferAddress, BufferAddress),
}

impl<'a> Binding<'a> {
    fn resolve(&self) -> BufferBinding<'a> {
        match self {
            Binding::Full(t) => BufferBinding {
                buffer: &t.buf,
                offset: 0,
                size: None,
            },
            Binding::Slice(t, off, size) => BufferBinding {
                buffer: &t.buf,
                offset: *off,
                size: std::num::NonZeroU64::new(*size),
            },
        }
    }

    /// Cache signature: tensor identity plus bound byte range.
    fn sig(&self) -> (u64, u64, u64) {
        match self {
            Binding::Full(t) => (t.id, 0, 0),
            Binding::Slice(t, off, size) => (t.id, *off, *size),
        }
    }
}

/// Owns one GPU compute pass; the only way architectures interact with wgpu.
pub struct Pass<'a>(wgpu::ComputePass<'a>);

impl<'a> Pass<'a> {
    pub fn begin(encoder: &'a mut CommandEncoder, label: &str) -> Self {
        Self(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            ..Default::default()
        }))
    }
}

/// Most bindings any shader takes (gemv_qkv: 11).
const MAX_BINDINGS: usize = 12;

/// Bind group cache key: shader plus each binding's (tensor id, offset, size).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BgKey {
    shader: &'static str,
    len: u8,
    sig: [(u64, u64, u64); MAX_BINDINGS],
}

impl BgKey {
    fn new(shader: &'static str, bufs: &[Binding<'_>]) -> Self {
        assert!(bufs.len() <= MAX_BINDINGS, "{shader}: too many bindings");
        let mut sig = [(0u64, 0u64, 0u64); MAX_BINDINGS];
        for (i, b) in bufs.iter().enumerate() {
            sig[i] = b.sig();
        }
        Self {
            shader,
            len: bufs.len() as u8,
            sig,
        }
    }
}

/// Gets or creates the bind group for `key`, caching it for reuse.
fn cached_bind_group<'c>(
    cache: &'c mut HashMap<BgKey, BindGroup>,
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    key: BgKey,
    bufs: &[Binding<'_>],
) -> &'c BindGroup {
    cache.entry(key).or_insert_with(|| {
        let resolved: Vec<BufferBinding<'_>> = bufs.iter().map(|b| b.resolve()).collect();
        let entries: Vec<BindGroupEntry<'_>> = resolved
            .iter()
            .enumerate()
            .map(|(i, b)| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::Buffer(b.clone()),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key.shader),
            layout,
            entries: &entries,
        })
    })
}

/// Owns the wgpu device and every GPU-side resource factory.
pub struct Backend {
    pub device: Device,
    pub queue: Queue,
    pub kernels: Kernels,
    adapter_name: String,
    /// One-element f32 buffer bound as the gemm scale input of bf16 weights.
    dummy_scale: Tensor,
    /// Split-K gemv partials [SEGS, N] f32, grown on demand.
    gemv_partial: Tensor,
    /// Split-K gemm partials [SEGS, M, Y_STRIDE] f32, grown on demand.
    gemm_partial: Tensor,

    /// Bind groups keyed by their buffer signature, reused across steps.
    bg_cache: HashMap<BgKey, BindGroup>,
    /// GPU timestamp profiler; present only when FLINT_PROFILE is set and supported.
    profiler: Option<Profiler>,
}

impl Backend {
    pub fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| Error::Gpu(format!("no suitable adapter: {e}")))?;
        let adapter_name = adapter.get_info().name;

        let limits = Limits {
            max_storage_buffer_binding_size: (1u64 << 31) - 4,
            // 2 GiB: Phi-4-mini's bf16 embedding table alone is 1.23 GiB.
            max_buffer_size: 1u64 << 31,
            // The attention kernel runs 512 threads and its staged KV tiles
            // exceed the 16 KiB default workgroup storage.
            max_compute_workgroup_size_x: 1024,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_storage_size: 48 * 1024,
            // gemv_qkv binds three weights, their scales and three outputs.
            max_storage_buffers_per_shader_stage: 16,
            ..Limits::default()
        };
        // request_device rejects limits above the adapter's capability
        // (lavapipe caps workgroup storage at 32 KiB), so downlevel each.
        let supported = adapter.limits();
        let limits = Limits {
            max_storage_buffer_binding_size: limits
                .max_storage_buffer_binding_size
                .min(supported.max_storage_buffer_binding_size),
            max_buffer_size: limits.max_buffer_size.min(supported.max_buffer_size),
            max_compute_workgroup_size_x: limits
                .max_compute_workgroup_size_x
                .min(supported.max_compute_workgroup_size_x),
            max_compute_invocations_per_workgroup: limits
                .max_compute_invocations_per_workgroup
                .min(supported.max_compute_invocations_per_workgroup),
            max_compute_workgroup_storage_size: limits
                .max_compute_workgroup_storage_size
                .min(supported.max_compute_workgroup_storage_size),
            max_storage_buffers_per_shader_stage: limits
                .max_storage_buffers_per_shader_stage
                .min(supported.max_storage_buffers_per_shader_stage),
            ..limits
        };
        // Profiling is opt-in: request timestamp features only when asked.
        let want_profile = std::env::var("FLINT_PROFILE").is_ok();
        let profile_features = if want_profile {
            adapter.features()
                & (wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
        } else {
            wgpu::Features::empty()
        };
        // Always request in-pass timestamps: on this driver stack their
        // presence changes the DXIL path observably (see bench notes), so
        // profile and non-profile runs must use the same features.
        let ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let base_features = if ts {
            profile_features | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
        } else {
            profile_features
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flint"),
            required_features: base_features,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| Error::Gpu(format!("device creation failed: {e}")))?;

        let mut kernels = Kernels::new(&device)?;
        warmup(&device, &queue, &mut kernels)?;
        let dummy_scale = Tensor::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dummy_scale"),
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            vec![1],
            DType::F32,
        );
        let profiler = if profile_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
            Some(Profiler::new(&device, &queue, 4096))
        } else {
            if want_profile {
                eprintln!(
                    "[flint] FLINT_PROFILE set but adapter lacks in-pass timestamp queries; profiling disabled"
                );
            }
            None
        };
        let gemv_partial = Self::gemv_partial_buf(&device, 8 * 65536);
        let gemm_partial = Self::gemv_partial_buf(&device, 4 * 128 * 16384);
        Ok(Self {
            device,
            queue,
            kernels,
            adapter_name,
            dummy_scale,
            gemv_partial,
            gemm_partial,
            bg_cache: HashMap::new(),
            profiler,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Allocates a zeroed gemv-partial buffer of `words` f32 elements; the
    /// shape is a raw [words] view (bindings always slice it).
    fn gemv_partial_buf(device: &Device, words: usize) -> Tensor {
        Tensor::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gemv_partial"),
                contents: &vec![0u8; words * 4],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            vec![words as u32],
            DType::F32,
        )
    }

    /// Scale-slot binding for non-quantized gemm weights.
    pub fn dummy_scale(&self) -> &Tensor {
        &self.dummy_scale
    }

    /// Scale binding of a weight: its per-group scales, or the dummy slot.
    fn scale_binding<'a>(dummy: &'a Tensor, w: &'a Weight) -> Binding<'a> {
        match w.scale() {
            Some(s) => Binding::Full(s),
            None => Binding::Full(dummy),
        }
    }

    /// Grows the gemm-partial buffer to hold `words` f32 elements.
    fn ensure_gemm_partial(&mut self, words: u32) {
        if words > self.gemm_partial.numel() as u32 {
            self.gemm_partial = Self::gemv_partial_buf(&self.device, words as usize);
        }
    }

    /// Grows the gemv-partial buffer to hold `words` f32 elements.
    fn ensure_gemv_partial(&mut self, words: u32) {
        if words > self.gemv_partial.numel() as u32 {
            self.gemv_partial = Self::gemv_partial_buf(&self.device, words as usize);
        }
    }

    /// Zero-initialized storage buffer.
    pub fn storage(&self, size: u64, label: &str) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Zero-initialized f32 tensor.
    pub fn zero_tensor(&self, shape: &[u32], label: &str) -> Tensor {
        let numel: u64 = shape.iter().map(|d| *d as u64).product();
        Tensor::new(self.storage(numel * 4, label), shape.to_vec(), DType::F32)
    }

    /// Zero-initialized packed-bf16 tensor (two elements per u32).
    pub fn zero_bf16_tensor(&self, shape: &[u32], label: &str) -> Tensor {
        let numel: u64 = shape.iter().map(|d| *d as u64).product();
        Tensor::new(
            self.storage(numel * 2, label),
            shape.to_vec(),
            DType::Bf16Packed,
        )
    }

    /// Zeroes a tensor's backing buffer.
    pub fn zero_fill(&self, t: &Tensor) {
        let mut enc = self.encoder();
        enc.clear_buffer(&t.buf, 0, None);
        self.submit(enc);
    }

    /// Copies tensor contents byte-for-byte (recurrent state snapshots).
    pub fn copy(&self, src: &Tensor, dst: &Tensor) {
        assert_eq!(src.byte_len(), dst.byte_len(), "copy size mismatch");
        let mut enc = self.encoder();
        enc.copy_buffer_to_buffer(&src.buf, 0, &dst.buf, 0, src.byte_len());
        self.submit(enc);
    }

    pub fn tensor_f32(&self, data: &[f32], shape: Vec<u32>, label: &str) -> Tensor {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Tensor::new(buf, shape, DType::F32)
    }

    /// bf16 little-endian bytes, packed two-per-u32 for GPU-side unpacking.
    pub fn tensor_bf16(&self, bytes: &[u8], shape: Vec<u32>, label: &str) -> Result<Tensor> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::Model(format!("{label}: odd bf16 byte count")));
        }
        let packed: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Odd element count: pad the final pair.
        let padded = if bytes.len() % 4 == 2 {
            let mut v = packed;
            let last = &bytes[bytes.len() - 2..];
            v.push(u32::from_le_bytes([last[0], last[1], 0, 0]));
            v
        } else {
            packed
        };
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&padded),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Ok(Tensor::new(buf, shape, DType::Bf16Packed))
    }

    /// Raw i8 bytes, one element per byte; count must be a multiple of 4 so
    /// the buffer addresses as array<u32> in shaders.
    pub fn tensor_i8(&self, bytes: &[u8], shape: Vec<u32>, label: &str) -> Tensor {
        assert!(
            bytes.len().is_multiple_of(4),
            "{label}: i8 count not a multiple of 4"
        );
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Tensor::new(buf, shape, DType::I8)
    }

    pub fn write_u32(&self, buf: &Buffer, data: &[u32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(buf, 0, &bytes);
    }

    pub fn write_f32(&self, buf: &Buffer, data: &[f32]) {
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
    }

    pub fn encoder(&self) -> CommandEncoder {
        self.device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("step"),
            })
    }

    /// Encodes one kernel dispatch, reusing cached bind group and pipeline.
    pub fn dispatch(
        &mut self,
        pass: &mut Pass<'_>,
        name: &'static str,
        consts: &[(&'static str, f64)],
        bufs: &[Binding<'_>],
        groups: [u32; 3],
    ) -> Result<()> {
        let span = self.prof_begin(pass);
        Self::set(
            &mut self.kernels,
            &mut self.bg_cache,
            &self.device,
            pass,
            name,
            consts,
            bufs,
            groups,
        )?;
        self.prof_end(pass, name, span);
        Ok(())
    }

    /// Binds the cached bind group and pipeline for `name` and dispatches.
    /// Profiling spans are the caller's concern (fused kernels bracket both
    /// of their dispatches with one span).
    #[allow(clippy::too_many_arguments)]
    fn set(
        kernels: &mut Kernels,
        bg_cache: &mut HashMap<BgKey, BindGroup>,
        device: &Device,
        pass: &mut Pass<'_>,
        name: &'static str,
        consts: &[(&'static str, f64)],
        bufs: &[Binding<'_>],
        groups: [u32; 3],
    ) -> Result<()> {
        let key = BgKey::new(name, bufs);
        let layout = kernels.bind_group_layout(name)?.clone();
        let bind_group = cached_bind_group(bg_cache, device, &layout, key, bufs);
        let pipeline = kernels.pipeline(device, name, consts)?;
        pass.0.set_pipeline(pipeline);
        pass.0.set_bind_group(0, bind_group, &[]);
        pass.0.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }

    /// The gemm/gemv weight constants and scale binding for a weight.
    fn weight_io(w: &Weight) -> (u32, u32, Binding<'_>, f64) {
        assert_eq!(
            w.tensor().shape.len(),
            2,
            "gemm weight must be a [N, K] matrix"
        );
        let (n, k) = (w.tensor().shape[0], w.tensor().shape[1]);
        let wdtype = match w.tensor().dtype {
            DType::Bf16Packed => 0.0,
            DType::I8 => 1.0,
            DType::F32 | DType::U32 => {
                unreachable!("gemm operands are weights, never index tensors")
            }
        };
        (n, k, Binding::Full(w.tensor()), wdtype)
    }

    /// y = x @ dequant(w)^T over `rows` activation rows (multiple of 16).
    pub fn gemm(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
    ) -> Result<()> {
        self.gemm_strided(pass, x, w, y, rows, false, 0, 0)
    }

    /// [`Backend::gemm`] with residual accumulation into `y`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_acc(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
        acc: bool,
    ) -> Result<()> {
        self.gemm_strided(pass, x, w, y, rows, acc, 0, 0)
    }

    /// [`Backend::gemm`] writing column range [y_off, y_off + n) of a wider
    /// y tile whose row stride is `y_stride` (fused qkv projections).
    #[allow(clippy::too_many_arguments)]
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
        let (n, k, wb, wdtype) = Self::weight_io(w);
        // Tile size is baked into the kernel: TN=64 columns x TM=64 rows;
        // M_MAX chunks are exact multiples of TM.
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("M", rows as f64),
            ("WDTYPE", wdtype),
            ("GROUP", w.group() as f64),
            ("ACC", acc as u32 as f64),
            (
                "Y_STRIDE",
                if y_stride == 0 {
                    n as f64
                } else {
                    y_stride as f64
                },
            ),
            ("Y_OFF", y_off as f64),
        ];
        // Split long-K multi-row gemms across the dispatch z axis; short-K
        // shapes already saturate the SM count with column groups.
        let segs = if rows > 1 && k >= 8192 { 4 } else { 1 };
        let consts = consts.into_iter().chain([("SEGS", segs as f64)]).collect::<Vec<_>>();
        let span = self.prof_begin(pass);
        let (yb, yslice) = if segs > 1 {
            let stride = if y_stride == 0 { n } else { y_stride };
            self.ensure_gemm_partial(segs * rows * stride);
            (
                Binding::Slice(&self.gemm_partial, 0, (segs * rows * stride) as u64 * 4),
                y,
            )
        } else {
            (y, y)
        };
        let bufs = [x, wb, Self::scale_binding(&self.dummy_scale, w), yb];
        Self::set(
            &mut self.kernels,
            &mut self.bg_cache,
            &self.device,
            pass,
            name::GEMM,
            &consts,
            &bufs,
            [n.div_ceil(32), rows.div_ceil(32), segs],
        )?;
        self.prof_end(pass, name::GEMM, span);
        if segs > 1 {
            let stride = if y_stride == 0 { n } else { y_stride };
            let span = self.prof_begin(pass);
            let mconsts = [
                ("M", rows as f64),
                ("N", n as f64),
                ("Y_STRIDE", stride as f64),
                ("Y_OFF", y_off as f64),
                ("SEGS", segs as f64),
                ("ACC", acc as u32 as f64),
            ];
            let bufs = [
                Binding::Slice(&self.gemm_partial, 0, (segs * rows * stride) as u64 * 4),
                yslice,
            ];
            Self::set(
                &mut self.kernels,
                &mut self.bg_cache,
                &self.device,
                pass,
                name::MERGE_GEMM,
                &mconsts,
                &bufs,
                [n.div_ceil(256), 1, 1],
            )?;
            self.prof_end(pass, name::MERGE_GEMM, span);
        }
        Ok(())
    }

    /// y[n] = x[k] @ dequant(w)^T: the single-row (decode) fast path that
    /// streams the weight matrix; narrow outputs split K across segments.
    pub fn gemv(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
    ) -> Result<()> {
        self.gemv_acc(pass, x, w, y, false)
    }

    /// [`Backend::gemv`] with residual accumulation into `y` (acc = true).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_acc(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        acc: bool,
    ) -> Result<()> {
        let (n, k, wb, wdtype) = Self::weight_io(w);
        // K splits: SEGS=4 for wide outputs, SEGS=2 for mid ones (4096);
        // each must divide K into 16-block segments (SEGS <= K/16).
        let segs: u32 = if k.is_multiple_of(128) {
            4
        } else if k.is_multiple_of(32) {
            2
        } else {
            1
        };
        let span = self.prof_begin(pass);
        if segs > 1 {
            self.ensure_gemv_partial(n * segs);
        }
        let scale = Self::scale_binding(&self.dummy_scale, w);
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", wdtype),
            ("GROUP", w.group() as f64),
            ("SEGS", segs as f64),
            ("ACC", acc as u32 as f64),
        ];
        let out = if segs == 1 {
            y
        } else {
            // gemv writes one row per segment: [SEGS, N] f32.
            Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64)
        };
        let bufs = [x, wb, scale, out];
        Self::set(
            &mut self.kernels,
            &mut self.bg_cache,
            &self.device,
            pass,
            name::GEMV,
            &consts,
            &bufs,
            [n / 16, segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64),
                y,
            ];
            Self::set(
                &mut self.kernels,
                &mut self.bg_cache,
                &self.device,
                pass,
                name::MERGE_GEMV,
                &[
                    ("N", n as f64),
                    ("SEGS", segs as f64),
                    ("ACC", acc as u32 as f64),
                ],
                &bufs,
                [n / 16, 1, 1],
            )?;
        }
        self.prof_end(pass, name::GEMV, span);
        Ok(())
    }

    /// Fused q/k/v projection (decode path): one dispatch computes all three
    /// projections into separate outputs.
    #[allow(clippy::too_many_arguments)]
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
        let ntot = nq + nk + nv;
        // The K split must divide K into 16-block segments (SEGS <= K/16).
        let segs: u32 = if k.is_multiple_of(128) {
            8
        } else if k.is_multiple_of(32) {
            2
        } else {
            1
        };
        let consts = [
            ("NQ", nq as f64),
            ("NK", nk as f64),
            ("NV", nv as f64),
            ("K", k as f64),
            ("GROUP", wq.group() as f64),
            ("SEGS", segs as f64),
        ];
        let span = self.prof_begin(pass);
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
            &mut self.kernels,
            &mut self.bg_cache,
            &self.device,
            pass,
            name::GEMV_QKV,
            &consts,
            &bufs,
            [ntot / 16, segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64),
                yq,
                yk,
                yv,
            ];
            Self::set(
                &mut self.kernels,
                &mut self.bg_cache,
                &self.device,
                pass,
                name::MERGE_QKV,
                &[
                    ("NQ", nq as f64),
                    ("NK", nk as f64),
                    ("NV", nv as f64),
                    ("SEGS", segs as f64),
                ],
                &bufs,
                [ntot / 16, 1, 1],
            )?;
        }
        self.prof_end(pass, name::GEMV_QKV, span);
        Ok(())
    }

    /// Fused gate/up projection (decode path): both MLP input projections
    /// in one dispatch.
    #[allow(clippy::too_many_arguments)]
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
        // The K split must divide K into 16-block segments (SEGS <= K/16).
        let segs: u32 = if k.is_multiple_of(32) { 2 } else { 1 };
        let consts = [
            ("NG", n as f64),
            ("K", k as f64),
            ("GROUP", wg.group() as f64),
            ("SEGS", segs as f64),
        ];
        let span = self.prof_begin(pass);
        let ntot = 2 * n;
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
            &mut self.kernels,
            &mut self.bg_cache,
            &self.device,
            pass,
            name::GEMV_GATEUP,
            &consts,
            &bufs,
            [ntot / 16, segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64),
                yg,
                yu,
            ];
            Self::set(
                &mut self.kernels,
                &mut self.bg_cache,
                &self.device,
                pass,
                name::MERGE_GATEUP,
                &[("NG", n as f64), ("SEGS", segs as f64)],
                &bufs,
                [ntot / 16, 1, 1],
            )?;
        }
        self.prof_end(pass, name::GEMV_GATEUP, span);
        Ok(())
    }

    pub fn submit(&self, encoder: CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }

    /// Writes a start timestamp around a dispatch when profiling is active.
    fn prof_begin(&mut self, pass: &mut Pass<'_>) -> Option<u32> {
        self.profiler.as_mut().and_then(|p| p.begin(&mut pass.0))
    }

    /// Writes the matching end timestamp when profiling is active.
    fn prof_end(&mut self, pass: &mut Pass<'_>, label: &'static str, span: Option<u32>) {
        if let Some(p) = self.profiler.as_mut() {
            p.end(&mut pass.0, label, span);
        }
    }

    /// Resolves this frame's timestamps into the running totals; no-op when
    /// profiling is disabled.
    pub fn flush_profile(&mut self) -> Result<()> {
        let Some(prof) = self.profiler.as_mut() else {
            return Ok(());
        };
        let mut enc = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("step"),
            });
        prof.resolve(&mut enc);
        self.queue.submit(Some(enc.finish()));
        let pending = prof.pending();
        let timestamps = if pending > 0 {
            let bytes = map_read(&self.device, prof.read_buf())?;
            bytes
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        prof.accumulate(&timestamps)
    }

    /// Whether GPU profiling is active.
    pub fn profiling(&self) -> bool {
        self.profiler.is_some()
    }

    /// Per-shader GPU time totals, sorted slowest first.
    pub fn profile_report(&self) -> Vec<ProfileRow> {
        self.profiler
            .as_ref()
            .map(|p| p.report())
            .unwrap_or_default()
    }

    /// Copy `count` f32 values from `offset` bytes of `src` into a fresh vec.
    pub fn read_f32(&self, src: &Buffer, offset: BufferAddress, count: usize) -> Result<Vec<f32>> {
        let bytes = read_back(&self.device, &self.queue, src, offset, count * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// Maps `len` bytes of `src` at `offset` via a staging buffer and blocks
/// until ready (the wgpu map path is inherently asynchronous).
fn read_back(
    device: &Device,
    queue: &Queue,
    src: &Buffer,
    offset: BufferAddress,
    len: usize,
) -> Result<Vec<u8>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback"),
    });
    enc.copy_buffer_to_buffer(src, offset, &staging, 0, len as u64);
    queue.submit(Some(enc.finish()));
    map_read(device, &staging)
}

/// Maps a MAP_READ buffer directly and blocks for the result; the profiler's
/// readback buffer is already mapable, so it skips the staging copy.
fn map_read(device: &Device, buf: &Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    buf.slice(..).map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    loop {
        match rx.try_recv() {
            Ok(result) => {
                result.map_err(|e| Error::Gpu(format!("readback mapping failed: {e}")))?;
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {
                device
                    .poll(PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .map_err(|e| Error::Gpu(format!("device poll failed: {e}")))?;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(Error::Gpu("readback channel closed".into()));
            }
        }
    }
    let bytes = buf
        .slice(..)
        .get_mapped_range()
        .map_err(|e| Error::Gpu(format!("readback map failed: {e}")))?
        .to_vec();
    buf.unmap();
    Ok(bytes)
}

/// Warms up the GPU after device creation: model loading leaves the clocks
/// at idle levels on WDDM; a real compute dispatch + sync restores
/// steady-state boost (empty passes do nothing — the driver elides them).
/// Without it first-chunk timing swings 2-8x run to run.
fn warmup(device: &Device, queue: &Queue, kernels: &mut Kernels) -> Result<()> {
    let n = 1 << 20;
    let mk = |label: &str, val: f32| Tensor::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&vec![val; n]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }),
        vec![n as u32],
        DType::F32,
    );
    let wxa = mk("flint.warmup.a", 1.0);
    let wxb = mk("flint.warmup.b", 1.0);
    let wy = Tensor::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flint.warmup.y"),
            contents: &vec![0u8; n * 4],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }),
        vec![n as u32],
        DType::F32,
    );
    for _ in 0..3 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flint.warmup"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flint.warmup"),
                timestamp_writes: None,
            });
            let layout = kernels.bind_group_layout(name::ADD)?;
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flint.warmup.bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wxa.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wxb.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wy.buf.as_entire_binding() },
                ],
            });
            let pl = kernels.pipeline(device, name::ADD, &[("N_ELEM", n as f64)])?;
            pass.set_pipeline(pl);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n / 256) as u32, 1, 1);
            drop(pass);
        }
        queue.submit([enc.finish()]);
    }
    let (tx, rx) = std::sync::mpsc::channel();
    queue.on_submitted_work_done(move || {
        let _ = tx.send(());
    });
    let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
    Ok(())
}
