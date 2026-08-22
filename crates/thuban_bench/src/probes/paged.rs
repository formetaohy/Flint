use thuban_backend::Backend;
use thuban_error::Result;

pub(super) fn paged_probe() -> Result<()> {
    use crate::synth::{BenchSpec, SynthCheckpoint};
    use thuban_architectures::transformer::{Config, Model, plan};
    use thuban_kernel::PAGE_LEN;
    use thuban_model::{LanguageModel, SeqChunk};

    const SEQS: u32 = 8;
    const BUDGET: u32 = 2048;
    const PREFILL: u32 = 128;
    const DECODE: u32 = 128;
    const PAGES: u32 = 128;

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let spec = BenchSpec {
        hidden: 256,
        intermediate: 1024,
        layers: 4,
        q_heads: 8,
        kv_heads: 8,
        head_dim: 64,
        vocab: 2048,
    };
    let cfg = Config::parse(&spec.config_json(), false).unwrap();
    let source = SynthCheckpoint::new(spec);
    let contiguous_pages = SEQS * BUDGET / PAGE_LEN;
    let bytes_per_layer = |pages: u32| {
        pages as u64 * PAGE_LEN as u64 * spec.kv_heads as u64 * spec.head_dim as u64 * 4
    };
    eprintln!(
        "[probe] contiguous budget: {SEQS} seqs x {BUDGET} tokens = {contiguous_pages} pages, \
         {:.1} MiB/layer KV",
        bytes_per_layer(contiguous_pages) as f64 / (1 << 20) as f64
    );
    eprintln!(
        "[probe] paged pool: {PAGES} pages ({:.1}x overcommit), {:.1} MiB/layer KV",
        contiguous_pages as f64 / PAGES as f64,
        bytes_per_layer(PAGES) as f64 / (1 << 20) as f64
    );

    let arena = thuban_model::pool::ArenaSpec {
        seq_lens: vec![BUDGET; SEQS as usize],
        pages: Some(PAGES),
    };
    let mut model = Model::load(&source, cfg, &plan(), &arena, None, &backend)?;
    for seq in 0..SEQS {
        model.alloc_pages(&backend, seq, PREFILL + DECODE + 32)?;
    }
    let prefills: Vec<Vec<u32>> = (0..SEQS)
        .map(|s| {
            (0..PREFILL)
                .map(|i| (s * PREFILL + i) % (spec.vocab - 1) + 1)
                .collect()
        })
        .collect();
    let mut offsets = vec![0u32; SEQS as usize];
    while offsets.iter().any(|&o| o < PREFILL) {
        let mut total = 0u32;
        let chunks: Vec<SeqChunk> = (0..SEQS)
            .map(|s| {
                let take = ((PREFILL - offsets[s as usize]).min(16)).min(128 - total);
                total += take;
                SeqChunk {
                    tokens: &prefills[s as usize]
                        [offsets[s as usize] as usize..(offsets[s as usize] + take) as usize],
                    seq: s,
                    logit_rows: &[],
                    hidden_rows: &[],
                }
            })
            .collect();
        let _ = model.forward(&mut backend, &chunks)?;
        for off in &mut offsets {
            *off += 16;
        }
    }
    let _ = backend.read_f32(&backend.quant_lut().buf, 0, 1)?;
    eprintln!(
        "[probe] prefill: {SEQS} x {PREFILL} tokens, {}/{} pages used",
        model.used_pages(),
        PAGES
    );

    let mut t = 0u32;
    let run = |model: &mut Model, backend: &mut Backend, t: &mut u32| -> Result<()> {
        let tok = [*t % (spec.vocab - 1) + 1];
        let chunks: Vec<SeqChunk> = (0..SEQS)
            .map(|s| SeqChunk {
                tokens: &tok,
                seq: s,
                logit_rows: &[0],
                hidden_rows: &[],
            })
            .collect();
        let _ = model.forward(backend, &chunks)?;
        *t += 1;
        Ok(())
    };
    for _ in 0..16 {
        run(&mut model, &mut backend, &mut t)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..DECODE {
        run(&mut model, &mut backend, &mut t)?;
    }
    let _ = backend.read_f32(&backend.quant_lut().buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let tokens = DECODE * SEQS;
    eprintln!(
        "[probe] decode: {SEQS} concurrent seqs, {tokens} tok in {secs:.2}s ({:.1} tok/s, \
         {:.2} ms/step)",
        tokens as f64 / secs,
        secs / DECODE as f64 * 1e3
    );
    eprintln!(
        "[probe] pages used: {}/{} ({:.0}% of contiguous)",
        model.used_pages(),
        PAGES,
        model.used_pages() as f64 / contiguous_pages as f64 * 100.0
    );
    Ok(())
}
