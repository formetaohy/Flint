//! flint-bench: synthetic-weight throughput benchmark for the Flint inference
//! engine. Builds a real-size dense model from deterministic random weights
//! (no disk, no downloads) and measures prefill/decode throughput, plus an
//! optional per-kernel GPU profile (FLINT_PROFILE=1).

use std::time::Instant;

use clap::Parser;

use flint_architectures::{DenseConfig, DenseModel};
use flint_backend::Backend;
use flint_error::Result;
use flint_model::LanguageModel;
use serde_json::json;

use flint_bench::synth;
use synth::{BenchSpec, SynthCheckpoint};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 4096)]
    hidden: u32,
    #[arg(long, default_value_t = 14336)]
    intermediate: u32,
    #[arg(long, default_value_t = 8)]
    layers: u32,
    #[arg(long, default_value_t = 32)]
    q_heads: u32,
    #[arg(long, default_value_t = 8)]
    kv_heads: u32,
    #[arg(long, default_value_t = 128)]
    head_dim: u32,
    #[arg(long, default_value_t = 32768)]
    vocab: u32,
    #[arg(long, default_value_t = 2048)]
    prefill_tokens: u32,
    #[arg(long, default_value_t = 64)]
    decode_tokens: u32,
    #[arg(long, default_value_t = 8192)]
    max_seq: u32,
    /// Run only the raw bandwidth probe (copy kernel) and exit.
    #[arg(long)]
    bandwidth: bool,
    /// Print per-step decode timings.
    #[arg(long)]
    verbose: bool,
}

fn config(s: &BenchSpec) -> DenseConfig {
    let v = json!({
        "hidden_size": s.hidden,
        "intermediate_size": s.intermediate,
        "num_hidden_layers": s.layers,
        "num_attention_heads": s.q_heads,
        "num_key_value_heads": s.kv_heads,
        "head_dim": s.head_dim,
        "vocab_size": s.vocab,
        "rope_theta": 500000.0,
        "eos_token_id": [0],
        "tie_word_embeddings": false,
    });
    DenseConfig::parse(&v, false).unwrap()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.bandwidth {
        return bandwidth_probe();
    }
    let spec = BenchSpec {
        hidden: args.hidden,
        intermediate: args.intermediate,
        layers: args.layers,
        q_heads: args.q_heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        vocab: args.vocab,
    };
    eprintln!(
        "[bench] model {}x{}x{} h={} i={} v={} ({} MiB weights)",
        spec.layers,
        spec.q_heads,
        spec.kv_heads,
        spec.hidden,
        spec.intermediate,
        spec.vocab,
        spec.weight_bytes() >> 20
    );

    eprintln!("[bench] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[bench] adapter: {}", backend.adapter_name());

    eprintln!("[bench] generating synthetic weights...");
    let t0 = Instant::now();
    let source = SynthCheckpoint::new(spec);
    let cfg = config(&spec);
    let plan = flint_architectures::dense::dense_plan(false);
    let mut backend = backend;
    let mut model = DenseModel::load(&source, cfg, &plan, args.max_seq, &backend)?;
    eprintln!(
        "[bench] weights loaded in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // ---- prefill: M_MAX-wide chunks ----
    let (chunks, rem) = (
        args.prefill_tokens / flint_model::M_MAX,
        args.prefill_tokens % flint_model::M_MAX,
    );
    let mut t = 0u32;
    // Warmup chunk: pipeline compilation happens once here, and a readback
    // forces the GPU to catch up so the timed run measures steady state.
    let warm_ids: Vec<u32> = (0..16).map(|i| i % (args.vocab - 1) + 1).collect();
    model.forward(&mut backend, &warm_ids, &[], &[])?;
    let _ = backend.read_f32(&backend.dummy_scale().buf, 0, 1)?;
    let t0 = Instant::now();
    for _ in 0..chunks {
        let ids: Vec<u32> = (t..t + flint_model::M_MAX)
            .map(|i| i % (args.vocab - 1) + 1)
            .collect();
        model.forward(&mut backend, &ids, &[], &[])?;
        t += flint_model::M_MAX;
    }
    if rem > 0 {
        let ids: Vec<u32> = (t..t + rem).map(|i| i % (args.vocab - 1) + 1).collect();
        model.forward(&mut backend, &ids, &[], &[])?;
    }
    // Drain the GPU queue: prefill issues no readbacks, so the elapsed time
    // above is pure submission rate. This sync makes it true GPU throughput.
    let _ = backend.read_f32(&backend.dummy_scale().buf, 0, 1)?;
    let prefill_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[bench] prefill: {} tok in {:.2}s ({:.1} tok/s)",
        args.prefill_tokens,
        prefill_secs,
        args.prefill_tokens as f64 / prefill_secs
    );

    // ---- decode: one token at a time, with logits readback (as in real use) ----
    let mut logits: Vec<f32> = Vec::new();
    let mut per_step: Vec<f64> = Vec::new();
    let t0 = Instant::now();
    for (i, tok) in (0..args.decode_tokens).enumerate() {
        let ids = [tok % (args.vocab - 1) + 1];
        let s0 = Instant::now();
        let out = model.forward(&mut backend, &ids, &[0], &[])?;
        per_step.push(s0.elapsed().as_secs_f64() * 1000.0);
        if args.verbose && i < 12 {
            eprintln!("[bench] step {i}: {:.2} ms", per_step.last().unwrap());
        }
        logits = out.logits[0].clone();
    }
    let decode_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[bench] decode: {} tok in {:.2}s ({:.1} tok/s)",
        args.decode_tokens,
        decode_secs,
        args.decode_tokens as f64 / decode_secs
    );
    eprintln!(
        "[bench] decode bandwidth: {:.1} GB/s (weights only)",
        spec.weight_bytes() as f64 / decode_secs / 1e9
    );
    eprintln!(
        "[bench] last logit[0..4]: {:?}",
        &logits[..4.min(logits.len())]
    );

    if backend.profiling() {
        let rows = backend.profile_report();
        let total: u64 = rows.iter().map(|r| r.total_ns).sum();
        eprintln!("[bench] GPU kernel time breakdown (cumulative over prefill+decode):");
        for r in rows {
            let pct = if total > 0 {
                r.total_ns as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  {:<12} {:9.2} ms  {:8} calls  {:5.1}%",
                r.label,
                r.total_ns as f64 / 1e6,
                r.count,
                pct
            );
        }
    }
    Ok(())
}

/// Raw device bandwidth probe: a trivial copy kernel over a large buffer.
fn bandwidth_probe() -> Result<()> {
    use flint_backend::{Binding, Pass};

    let mut backend = Backend::new()?;
    eprintln!("[bench] adapter: {}", backend.adapter_name());
    let n = 1 << 26; // 64M f32 = 256 MiB
    let x = backend.tensor_f32(&vec![1.0; n], vec![n as u32], "x");
    let y = backend.zero_tensor(&[n as u32], "y");

    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder();
        {
            let mut pass = Pass::begin(&mut enc, "copy");
            backend.dispatch(
                &mut pass,
                "add",
                &[("N_ELEM", n as f64)],
                &[Binding::Full(&x), Binding::Full(&x), Binding::Full(&y)],
                [1024, (n as u32).div_ceil(256 * 1024), 1],
            )?;
        }
        backend.submit(enc);
        Ok(())
    };

    for _ in 0..3 {
        run(&mut backend)?;
    }
    let t0 = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters {
        run(&mut backend)?;
    }
    // Force completion; the GPU runs ahead of the CPU otherwise.
    let _ = backend.read_f32(&y.buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let bytes = n as f64 * 4.0 * 2.0 * iters as f64; // read + write
    eprintln!("[bench] copy bandwidth: {:.1} GB/s", bytes / secs / 1e9);
    Ok(())
}
