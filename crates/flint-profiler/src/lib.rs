use std::collections::HashMap;
use std::sync::Arc;

use flint_error::{Error, Result};
use flint_gpu::{Buffer, Device, Encoder, HostAccess, TimestampSet};

const INITIAL_CAPACITY: u32 = 4096;

pub struct ProfileRow {
    pub label: &'static str,
    pub total_ns: u64,
    pub count: u64,
}

pub fn breakdown(rows: &[ProfileRow]) -> String {
    let total: u64 = rows.iter().map(|r| r.total_ns).sum();
    let mut out = String::new();
    for r in rows {
        let pct = if total > 0 {
            r.total_ns as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        out.push_str(&format!(
            "  {:<12} {:9.2} ms  {:8} calls  {:5.1}%\n",
            r.label,
            r.total_ns as f64 / 1e6,
            r.count,
            pct
        ));
    }
    out
}

struct Generation {
    set: TimestampSet,
    resolve_buf: Buffer,
    read_buf: Buffer,
    capacity: u32,
    next: u32,
}

impl Generation {
    fn new(device: &Device, capacity: u32) -> Result<Self> {
        let set = device.create_timestamp_set(capacity)?;
        let resolve_buf = device.create_buffer(capacity as u64 * 8, HostAccess::None, true)?;
        let read_buf = device.create_buffer(capacity as u64 * 8, HostAccess::Read, false)?;
        Ok(Self {
            set,
            resolve_buf,
            read_buf,
            capacity,
            next: 0,
        })
    }
}

struct Span {
    label: &'static str,
    generation: usize,
    start: u32,
    end: u32,
}

pub struct GpuProfiler {
    device: Arc<Device>,
    generations: Vec<Generation>,
    spans: Vec<Span>,
    totals: HashMap<&'static str, (u64, u64)>,
    period_ns: f64,
    open: usize,
}

impl GpuProfiler {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Self::with_initial_capacity(device, INITIAL_CAPACITY)
    }

    pub fn with_initial_capacity(device: Arc<Device>, capacity: u32) -> Result<Self> {
        let generation = Generation::new(device.as_ref(), capacity)?;
        Ok(Self {
            period_ns: device.timestamp_period_ns(),
            device,
            generations: vec![generation],
            spans: Vec::new(),
            totals: HashMap::new(),
            open: 0,
        })
    }

    fn grow(&mut self) -> Result<()> {
        self.generations
            .push(Generation::new(self.device.as_ref(), INITIAL_CAPACITY)?);
        Ok(())
    }

    pub fn begin_span(&mut self) -> Result<usize> {
        let mut enc = self.device.encoder()?;
        let span = self.begin(&mut enc)?;
        enc.submit_and_reset();
        Ok(span)
    }

    pub fn mark_begin(&mut self, encoder: &mut Encoder) -> Result<usize> {
        self.begin(encoder)
    }

    pub fn mark_end(
        &mut self,
        encoder: &mut Encoder,
        label: &'static str,
        span: usize,
    ) -> Result<()> {
        self.end(encoder, label, span)
    }

    pub fn end_span(&mut self, label: &'static str, span: usize) -> Result<()> {
        let mut enc = self.device.encoder()?;
        self.end(&mut enc, label, span)?;
        enc.submit_and_reset();
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.pending() == 0 {
            return Ok(());
        }
        let mut enc = self.device.encoder()?;
        self.resolve(&mut enc)?;
        let sub = enc.submit_and_reset();
        sub.wait()?;
        self.accumulate()
    }

    fn begin(&mut self, encoder: &mut Encoder) -> Result<usize> {
        let last = self
            .generations
            .last()
            .expect("profiler always has a generation");
        if last.next + self.open as u32 + 2 > last.capacity {
            self.grow()?;
        }
        let generation = self.generations.len() - 1;
        let current = self
            .generations
            .last_mut()
            .expect("profiler always has a generation");
        let start = current.next;
        current.next += 1;
        encoder.write_timestamp(&current.set, start)?;
        self.spans.push(Span {
            label: "",
            generation,
            start,
            end: 0,
        });
        self.open += 1;
        Ok(self.spans.len() - 1)
    }

    fn end(&mut self, encoder: &mut Encoder, label: &'static str, span: usize) -> Result<()> {
        let generation = self.spans[span].generation;
        let current = &mut self.generations[generation];
        let end = current.next;
        current.next += 1;
        encoder.write_timestamp(&current.set, end)?;
        self.spans[span].label = label;
        self.spans[span].end = end;
        self.open -= 1;
        Ok(())
    }

    fn resolve(&self, encoder: &mut Encoder) -> Result<()> {
        for generation in &self.generations {
            if generation.next > 0 {
                encoder.resolve_timestamps(
                    &generation.set,
                    0,
                    generation.next,
                    &generation.resolve_buf,
                    0,
                )?;
                encoder.copy(
                    &generation.resolve_buf,
                    0,
                    &generation.read_buf,
                    0,
                    generation.next as u64 * 8,
                )?;
            }
        }
        Ok(())
    }

    fn pending(&self) -> u32 {
        self.generations.iter().map(|g| g.next).sum()
    }

    fn accumulate(&mut self) -> Result<()> {
        if self.pending() == 0 {
            return Ok(());
        }
        let mut timestamps: Vec<Vec<u64>> = Vec::with_capacity(self.generations.len());
        for generation in &self.generations {
            let mut bytes = vec![0u8; generation.next as usize * 8];
            generation
                .read_buf
                .read(0, &mut bytes)
                .map_err(|e| Error::Profiler(e.to_string()))?;
            timestamps.push(
                bytes
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().expect("timestamp chunk is 8 bytes")))
                    .collect(),
            );
        }
        for span in &self.spans {
            if span.end == 0 {
                continue;
            }
            let ts = &timestamps[span.generation];
            let start = ts[span.start as usize];
            let end = ts[span.end as usize];
            let ns = (end.wrapping_sub(start) as f64 * self.period_ns) as u64;
            let entry = self.totals.entry(span.label).or_insert((0, 0));
            entry.0 += ns;
            entry.1 += 1;
        }
        self.generations.truncate(1);
        self.generations[0].next = 0;
        self.spans.clear();
        self.open = 0;
        Ok(())
    }

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
