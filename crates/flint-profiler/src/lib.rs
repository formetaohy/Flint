use std::collections::HashMap;
use std::rc::Rc;

use saturn_core::{Buffer, BufferSpec, CommandEncoder, Device, TimestampSet};

use flint_error::Result;

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
    set: Box<dyn TimestampSet>,
    read_buf: Box<dyn Buffer>,
    capacity: u32,
    next: u32,
}

impl Generation {
    fn new(device: &dyn Device, capacity: u32) -> Result<Self> {
        let set = device.create_timestamp_set(capacity)?;
        let read_buf = device.create_buffer(&BufferSpec {
            size: capacity as u64 * 8,
            host_visible: true,
        })?;
        Ok(Self {
            set,
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
    device: Rc<dyn Device>,
    generations: Vec<Generation>,
    spans: Vec<Span>,
    totals: HashMap<&'static str, (u64, u64)>,
    period_ns: f64,
    needs_reset: bool,
}

impl GpuProfiler {
    pub fn new(device: Rc<dyn Device>) -> Result<Self> {
        Self::with_initial_capacity(device, INITIAL_CAPACITY)
    }

    pub fn with_initial_capacity(device: Rc<dyn Device>, capacity: u32) -> Result<Self> {
        let generation = Generation::new(device.as_ref(), capacity)?;
        Ok(Self {
            period_ns: device.timestamp_period_ns(),
            device,
            generations: vec![generation],
            spans: Vec::new(),
            totals: HashMap::new(),
            needs_reset: false,
        })
    }

    fn grow(&mut self) -> Result<()> {
        let capacity = self
            .generations
            .iter()
            .map(|g| g.capacity)
            .sum::<u32>()
            .max(INITIAL_CAPACITY);
        self.generations
            .push(Generation::new(self.device.as_ref(), capacity)?);
        Ok(())
    }

    pub fn begin_span(&mut self) -> Result<u32> {
        let mut enc = self.device.encoder()?;
        let span = self.begin(enc.as_mut())?;
        self.device.submit(enc)?;
        Ok(span)
    }

    pub fn end_span(&mut self, label: &'static str, span: u32) -> Result<()> {
        let mut enc = self.device.encoder()?;
        self.end(enc.as_mut(), label, span)?;
        self.device.submit(enc)?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.pending() == 0 {
            return Ok(());
        }
        let mut enc = self.device.encoder()?;
        self.resolve(enc.as_mut())?;
        let sub = self.device.submit(enc)?;
        sub.wait()?;
        self.accumulate()
    }

    pub fn begin(&mut self, encoder: &mut dyn CommandEncoder) -> Result<u32> {
        if self.needs_reset {
            for generation in &self.generations {
                encoder.reset_timestamps(generation.set.as_ref(), 0, generation.capacity)?;
            }
            self.needs_reset = false;
        }
        let last = self
            .generations
            .last()
            .expect("profiler always has a generation");
        if last.next + 2 > last.capacity {
            self.grow()?;
        }
        let generation = self
            .generations
            .last_mut()
            .expect("profiler always has a generation");
        let start = generation.next;
        generation.next += 1;
        encoder.write_timestamp(generation.set.as_ref(), start)?;
        Ok(start)
    }

    pub fn end(
        &mut self,
        encoder: &mut dyn CommandEncoder,
        label: &'static str,
        span: u32,
    ) -> Result<()> {
        let generation = self
            .generations
            .last_mut()
            .expect("profiler always has a generation");
        let end = generation.next;
        generation.next += 1;
        encoder.write_timestamp(generation.set.as_ref(), end)?;
        self.spans.push(Span {
            label,
            generation: self.generations.len() - 1,
            start: span,
            end,
        });
        Ok(())
    }

    pub fn resolve(&self, encoder: &mut dyn CommandEncoder) -> Result<()> {
        for generation in &self.generations {
            if generation.next > 0 {
                encoder.resolve_timestamps(
                    generation.set.as_ref(),
                    0,
                    generation.next,
                    generation.read_buf.as_ref(),
                    0,
                )?;
            }
        }
        Ok(())
    }

    pub fn pending(&self) -> u32 {
        self.generations.iter().map(|g| g.next).sum()
    }

    pub fn accumulate(&mut self) -> Result<()> {
        if self.pending() == 0 {
            return Ok(());
        }
        let mut timestamps: Vec<Vec<u64>> = Vec::with_capacity(self.generations.len());
        for generation in &self.generations {
            let mut bytes = vec![0u8; generation.next as usize * 8];
            generation.read_buf.read(0, &mut bytes)?;
            timestamps.push(
                bytes
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().expect("timestamp chunk is 8 bytes")))
                    .collect(),
            );
        }
        for span in &self.spans {
            let ts = &timestamps[span.generation];
            let start = ts[span.start as usize];
            let end = ts[span.end as usize];
            let ns = (end.wrapping_sub(start) as f64 * self.period_ns) as u64;
            let entry = self.totals.entry(span.label).or_insert((0, 0));
            entry.0 += ns;
            entry.1 += 1;
        }
        self.needs_reset = true;
        self.generations.truncate(1);
        self.generations[0].next = 0;
        self.spans.clear();
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
