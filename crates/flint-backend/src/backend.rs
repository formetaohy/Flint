use std::collections::HashMap;
use std::sync::mpsc;

use wgpu::{
    BindGroup, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferBinding,
    CommandEncoder, CommandEncoderDescriptor, Device, Instance, InstanceDescriptor, Limits,
    MapMode, MemoryHints, PollType, PowerPreference, Queue, RequestAdapterOptions,
    util::DeviceExt,
};

use flint_error::{Error, Result};
use flint_kernel::Kernels;
use flint_profiler::{GpuProfiler, ProfileRow};
use flint_tensor::{DType, Tensor, Weight};

/// A reference to a whole tensor or a byte-aligned sub-slice of it.
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

    /// Cache signature: tensor identity plus the bound byte range. A whole
    /// buffer binds as (id, 0, 0); a slice carries its offset and size.
    fn sig(&self) -> (u64, u64, u64) {
        match self {
            Binding::Full(t) => (t.id, 0, 0),
            Binding::Slice(t, off, size) => (t.id, *off, *size),
        }
    }
}

/// Owns one GPU compute pass; the only way architectures and kernel
/// dispatchers interact with wgpu's pass API.
pub struct Pass<'a>(wgpu::ComputePass<'a>);

impl<'a> Pass<'a> {
    pub fn begin(encoder: &'a mut CommandEncoder, label: &str) -> Self {
        Self(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            ..Default::default()
        }))
    }
}

/// Most bindings any shader takes (delta_recur: 7).
const MAX_BINDINGS: usize = 8;

/// Allocation-free bind group cache key: the shader plus each binding's
/// (tensor id, offset, size) signature. The forward graph binds a fixed buffer
/// set every step, so each dispatch site keys one bind group for its lifetime.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BgKey {
    shader: &'static str,
    len: u8,
    sig: [(u64, u64, u64); MAX_BINDINGS],
}

impl BgKey {
    fn new(shader: &'static str, bufs: &[Binding<'_>]) -> Self {
        assert!(
            bufs.len() <= MAX_BINDINGS,
            "{shader}: too many bindings"
        );
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
    /// Split-K gemv partials [SEGS, N] f32, grown on demand: the slice an
    /// active gemv binds must always fit (wide vocab projections exceed any
    /// fixed size).
    gemv_partial: Tensor,
    /// Bind groups keyed by their buffer signature; the forward graph rebinds
    /// the same buffers every step, so each is created once and reused.
    bg_cache: HashMap<BgKey, BindGroup>,
    /// GPU timestamp profiler; present only when FLINT_PROFILE is set and the
    /// adapter supports in-pass timestamp queries.
    profiler: Option<GpuProfiler>,
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
            // The attention kernel runs 512 threads (8 head slots x 64 keys)
            // and its staged KV tiles exceed the 16 KiB default workgroup
            // storage.
            max_compute_workgroup_size_x: 1024,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_storage_size: 48 * 1024,
            ..Limits::default()
        };
        // request_device rejects any requested limit above the adapter's
        // capability (lavapipe on CI caps workgroup storage at 32 KiB), so
        // downlevel each raised field to what this adapter actually supports.
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
            ..limits
        };
        // Profiling is opt-in: request the timestamp features only when asked,
        // and only those the adapter actually exposes.
        let want_profile = std::env::var("FLINT_PROFILE").is_ok();
        let profile_features = if want_profile {
            adapter.features()
                & (wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flint"),
            required_features: profile_features,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| Error::Gpu(format!("device creation failed: {e}")))?;

        let kernels = Kernels::new(&device)?;
        let dummy_scale = Tensor::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dummy_scale"),
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            vec![1],
            DType::F32,
        );
        let profiler = if profile_features
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
        {
            Some(GpuProfiler::new(&device, &queue, 4096))
        } else {
            if want_profile {
                eprintln!(
                    "[flint] FLINT_PROFILE set but adapter lacks in-pass timestamp queries; profiling disabled"
                );
            }
            None
        };
        let gemv_partial = Self::gemv_partial_buf(&device, 8 * 65536);
        Ok(Self {
            device,
            queue,
            kernels,
            adapter_name,
            dummy_scale,
            gemv_partial,
            bg_cache: HashMap::new(),
            profiler,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Allocates a zeroed gemv-partial buffer of `words` f32 elements (the
    /// shape is a raw [words] view; bindings always slice it).
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
        Tensor::new(self.storage(numel * 2, label), shape.to_vec(), DType::Bf16Packed)
    }

    /// Zeroes a tensor's backing buffer.
    pub fn zero_fill(&self, t: &Tensor) {
        let mut enc = self.encoder();
        enc.clear_buffer(&t.buf, 0, None);
        self.submit(enc);
    }

    /// Copy tensor contents byte-for-byte (used to snapshot recurrent state).
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

    /// Raw i8 bytes, one element per byte; element count must be a multiple
    /// of 4 so the buffer addresses as array<u32> in shaders.
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

    /// Encodes one kernel dispatch, reusing a cached bind group for the buffer
    /// set and a cached pipeline for the constants.
    pub fn dispatch(
        &mut self,
        pass: &mut Pass<'_>,
        name: &'static str,
        consts: &[(&'static str, f64)],
        bufs: &[Binding<'_>],
        groups: [u32; 3],
    ) -> Result<()> {
        let span = self.prof_begin(pass);
        let key = BgKey::new(name, bufs);
        let layout = self.kernels.bind_group_layout(name)?.clone();
        let bind_group = cached_bind_group(&mut self.bg_cache, &self.device, &layout, key, bufs);
        let pipeline = self.kernels.pipeline(&self.device, name, consts)?;
        pass.0.set_pipeline(pipeline);
        pass.0.set_bind_group(0, bind_group, &[]);
        pass.0.dispatch_workgroups(groups[0], groups[1], groups[2]);
        self.prof_end(pass, name, span);
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
    /// N and K come from the weight shape; bf16 weights bind the dummy scale
    /// slot, i8 weights bind their scales.
    pub fn gemm(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
    ) -> Result<()> {
        let (n, k, wb, wdtype) = Self::weight_io(w);
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", wdtype),
            ("GROUP", w.group() as f64),
        ];
        let span = self.prof_begin(pass);
        let scale = match w.scale() {
            Some(s) => Binding::Full(s),
            None => Binding::Full(&self.dummy_scale),
        };
        let bufs = [x, wb, scale, y];
        let key = BgKey::new("gemm", &bufs);
        let layout = self.kernels.bind_group_layout("gemm")?.clone();
        let bind_group = cached_bind_group(&mut self.bg_cache, &self.device, &layout, key, &bufs);
        let pipeline = self.kernels.pipeline(&self.device, "gemm", &consts)?;
        pass.0.set_pipeline(pipeline);
        pass.0.set_bind_group(0, bind_group, &[]);
        // The streaming gemm covers 2 rows per workgroup (rows a multiple of 16).
        pass.0.dispatch_workgroups(n / 16, rows / 2, 1);
        self.prof_end(pass, "gemm", span);
        Ok(())
    }

    /// y[n] = x[k] @ dequant(w)^T: the single-row (decode) fast path. Streams
    /// the weight matrix at near-peak bandwidth instead of the tiled matmul.
    /// Narrow outputs (small N) split K across more workgroups and merge the
    /// segment partials, so every output width saturates the GPU.
    pub fn gemv(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
    ) -> Result<()> {
        let (n, k, wb, wdtype) = Self::weight_io(w);
        // K splits: measured bandwidth peaks at SEGS=4 for wide outputs and
        // narrow short-K projections, and SEGS=2 for mid outputs (4096);
        // every width gains workgroups, none loses more than the merge cost.
        let segs: u32 = if n >= 8192 {
            4
        } else if n >= 4096 {
            2
        } else {
            4
        };
        let span = self.prof_begin(pass);
        let scale = match w.scale() {
            Some(s) => Binding::Full(s),
            None => Binding::Full(&self.dummy_scale),
        };
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", wdtype),
            ("GROUP", w.group() as f64),
            ("SEGS", segs as f64),
        ];
        if segs > 1 && (n * segs) as usize > self.gemv_partial.numel() as usize {
            self.gemv_partial = Self::gemv_partial_buf(&self.device, (n * segs) as usize);
        }
        let out = if segs == 1 {
            y
        } else {
            // gemv writes one row per segment: [SEGS, N] f32.
            Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64)
        };
        let bufs = [x, wb, scale, out];
        let key = BgKey::new("gemv", &bufs);
        let layout = self.kernels.bind_group_layout("gemv")?.clone();
        let bind_group = cached_bind_group(&mut self.bg_cache, &self.device, &layout, key, &bufs);
        let pipeline = self.kernels.pipeline(&self.device, "gemv", &consts)?;
        pass.0.set_pipeline(pipeline);
        pass.0.set_bind_group(0, bind_group, &[]);
        pass.0.dispatch_workgroups(n / 16, segs, 1);
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64),
                y,
            ];
            let key = BgKey::new("merge_gemv", &bufs);
            let layout = self.kernels.bind_group_layout("merge_gemv")?.clone();
            let bind_group =
                cached_bind_group(&mut self.bg_cache, &self.device, &layout, key, &bufs);
            let pipeline = self.kernels.pipeline(
                &self.device,
                "merge_gemv",
                &[("N", n as f64), ("SEGS", segs as f64)],
            )?;
            pass.0.set_pipeline(pipeline);
            pass.0.set_bind_group(0, bind_group, &[]);
            pass.0.dispatch_workgroups(n / 16, 1, 1);
        }
        self.prof_end(pass, "gemv", span);
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

    /// Resolves this frame's timestamps and folds them into the running totals.
    /// A no-op when profiling is disabled.
    pub fn flush_profile(&mut self) -> Result<()> {
        let Some(prof) = self.profiler.as_mut() else {
            return Ok(());
        };
        let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
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
        self.profiler.as_ref().map(|p| p.report()).unwrap_or_default()
    }

    /// Copy `count` f32 values from `offset` bytes of `src` into a fresh vec (blocks).
    pub fn read_f32(&self, src: &Buffer, offset: BufferAddress, count: usize) -> Result<Vec<f32>> {
        let bytes = read_back(&self.device, &self.queue, src, offset, count * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// Maps `len` bytes of `src` at `offset` via a staging buffer and blocks
/// until they are ready (the wgpu map path is inherently asynchronous).
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

/// Maps a MAP_READ buffer directly and blocks for the result. The profiler's
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
