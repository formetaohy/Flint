use std::collections::HashMap;

use saturn_core::{Buffer, BufferSpec, CommandEncoder, Device, TimestampSet};

use flint_error::Result;

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

struct Span {
    label: &'static str,
    start: u32,
    end: u32,
}

pub struct Profiler {
    set: Box<dyn TimestampSet>,
    read_buf: Box<dyn Buffer>,
    capacity: u32,
    next: u32,
    spans: Vec<Span>,
    totals: HashMap<&'static str, (u64, u64)>,
    period_ns: f64,
}

impl Profiler {
    pub fn new(device: &dyn Device, capacity: u32) -> Result<Self> {
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
            spans: Vec::new(),
            totals: HashMap::new(),
            period_ns: device.timestamp_period_ns(),
        })
    }

    pub fn begin(&mut self, encoder: &mut dyn CommandEncoder) -> Result<Option<u32>> {
        if self.next + 2 > self.capacity {
            return Ok(None);
        }
        let start = self.next;
        self.next += 1;
        encoder.write_timestamp(self.set.as_ref(), start)?;
        Ok(Some(start))
    }

    pub fn end(
        &mut self,
        encoder: &mut dyn CommandEncoder,
        label: &'static str,
        start: Option<u32>,
    ) -> Result<()> {
        let Some(start) = start else {
            return Ok(());
        };
        let end = self.next;
        self.next += 1;
        encoder.write_timestamp(self.set.as_ref(), end)?;
        self.spans.push(Span { label, start, end });
        Ok(())
    }

    pub fn resolve(&self, encoder: &mut dyn CommandEncoder) -> Result<()> {
        if self.next == 0 {
            return Ok(());
        }
        Ok(encoder.resolve_timestamps(
            self.set.as_ref(),
            0,
            self.next,
            self.read_buf.as_ref(),
            0,
        )?)
    }

    pub fn pending(&self) -> u32 {
        self.next
    }

    pub fn accumulate(&mut self) -> Result<()> {
        let count = self.next as usize;
        if count == 0 {
            self.next = 0;
            self.spans.clear();
            return Ok(());
        }
        let mut bytes = vec![0u8; count * 8];
        self.read_buf.read(0, &mut bytes)?;
        let timestamps: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("timestamp chunk is 8 bytes")))
            .collect();
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
