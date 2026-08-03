//! GPU timestamp profiling for Flint's compute kernels.
//!
//! Brackets every dispatch with two GPU timestamp queries and accumulates
//! per-shader totals across frames, so a handful of generated tokens yield a
//! stable breakdown of where GPU time actually goes. The profiler is only ever
//! constructed when profiling is requested; otherwise the dispatch fast path
//! is untouched. Timestamp readback is owned by the backend, which resolves,
//! maps and hands the raw values back to [`GpuProfiler::accumulate`].

use std::collections::HashMap;

use wgpu::{
    Buffer, BufferUsages, CommandEncoder, ComputePass, Device, QuerySet, QueryType, Queue,
};

use flint_error::Result;

/// One aggregated row: total GPU time spent in a shader across all frames.
pub struct ProfileRow {
    pub label: &'static str,
    pub total_ns: u64,
    pub count: u64,
}

/// A bracketed dispatch: timestamps at `start` and `end` query slots.
struct Span {
    label: &'static str,
    start: u32,
    end: u32,
}

/// Times dispatches on the GPU via timestamp queries and accumulates
/// per-shader totals. Frame state (query cursor + spans) resets on each
/// `accumulate`; totals persist for the session.
pub struct GpuProfiler {
    set: QuerySet,
    resolve_buf: Buffer,
    read_buf: Buffer,
    capacity: u32,
    next: u32,
    spans: Vec<Span>,
    totals: HashMap<&'static str, (u64, u64)>,
    period_ns: f64,
}

impl GpuProfiler {
    pub fn new(device: &Device, queue: &Queue, capacity: u32) -> Self {
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("flint.profile"),
            count: capacity,
            ty: QueryType::Timestamp,
        });
        let bytes = capacity as u64 * 8;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flint.profile.resolve"),
            size: bytes,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flint.profile.read"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            set,
            resolve_buf,
            read_buf,
            capacity,
            next: 0,
            spans: Vec::new(),
            totals: HashMap::new(),
            period_ns: queue.get_timestamp_period() as f64,
        }
    }

    /// Writes a start timestamp and returns its query slot, or `None` once the
    /// frame's query budget is exhausted (the dispatch still runs, untimed).
    pub fn begin(&mut self, pass: &mut ComputePass) -> Option<u32> {
        if self.next + 2 > self.capacity {
            return None;
        }
        let start = self.next;
        self.next += 1;
        pass.write_timestamp(&self.set, start);
        Some(start)
    }

    /// Writes the matching end timestamp and records the span.
    pub fn end(&mut self, pass: &mut ComputePass, label: &'static str, start: Option<u32>) {
        let Some(start) = start else { return };
        let end = self.next;
        self.next += 1;
        pass.write_timestamp(&self.set, end);
        self.spans.push(Span { label, start, end });
    }

    /// Resolves this frame's timestamps into the readback buffer. The caller
    /// submits the encoder, then calls `accumulate`.
    pub fn resolve(&self, encoder: &mut CommandEncoder) {
        if self.next == 0 {
            return;
        }
        encoder.resolve_query_set(&self.set, 0..self.next, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buf,
            0,
            &self.read_buf,
            0,
            self.next as u64 * 8,
        );
    }

    /// Resolved-timestamp readback target; the backend maps this after
    /// submitting the resolve encoder.
    pub fn read_buf(&self) -> &Buffer {
        &self.read_buf
    }

    /// Number of query slots this frame used (0 when nothing was timed).
    pub fn pending(&self) -> u32 {
        self.next
    }

    /// Folds the resolved timestamps of this frame's spans into the running
    /// totals, then resets the frame state.
    pub fn accumulate(&mut self, timestamps: &[u64]) -> Result<()> {
        let count = self.next as usize;
        if self.spans.is_empty() || count == 0 {
            self.next = 0;
            self.spans.clear();
            return Ok(());
        }
        if timestamps.len() < count {
            return Err(flint_error::Error::Gpu(format!(
                "profile readback returned {} timestamps, need {count}",
                timestamps.len()
            )));
        }
        for span in &self.spans {
            let start = timestamps[span.start as usize];
            let end = timestamps[span.end as usize];
            let ns = (end.wrapping_sub(start) as f64 * self.period_ns) as u64;
            let entry = self.totals.entry(span.label).or_insert((0, 0));
            entry.0 += ns;
            entry.1 += 1;
        }
        self.next = 0;
        self.spans.clear();
        Ok(())
    }

    /// Per-shader totals sorted by total time, descending.
    pub fn report(&self) -> Vec<ProfileRow> {
        let mut rows: Vec<ProfileRow> = self
            .totals
            .iter()
            .map(|(&label, &(total_ns, count))| ProfileRow {
                label,
                total_ns,
                count,
            })
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.total_ns));
        rows
    }
}

