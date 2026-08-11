use std::time::Instant;

use clap::Parser;

use flint_architectures::transformer::{TransformerConfig, TransformerModel, transformer_plan};
use flint_backend::Backend;
use flint_error::Result;
use flint_model::LanguageModel;
use serde_json::json;

use flint_bench::synth::{BenchSpec, SynthCheckpoint};

mod probes;

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

    #[arg(long)]
    bandwidth: bool,

    #[arg(long)]
    verbose: bool,

    #[arg(long)]
    gemv_probe: bool,

    #[arg(long)]
    cpu_probe: bool,

    #[arg(long)]
    gemm_probe: bool,

    #[arg(long)]
    attn_probe: bool,

    #[arg(long)]
    profile: bool,
}

fn config(s: &BenchSpec) -> TransformerConfig {
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
    TransformerConfig::parse(&v, false).unwrap()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.bandwidth {
        return probes::bandwidth_probe();
    }
    if args.gemv_probe {
        return probes::gemv_probe();
    }
    if args.cpu_probe {
        return probes::cpu_probe();
    }
    if args.gemm_probe {
        return probes::gemm_probe();
    }
    if args.attn_probe {
        return probes::attn_probe();
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
    let mut profiler = if args.profile {
        Some(flint_profiler::GpuProfiler::new(backend.device())?)
    } else {
        None
    };

    eprintln!("[bench] generating synthetic weights...");
    let t0 = Instant::now();
    let source = SynthCheckpoint::new(spec);
    let cfg = config(&spec);
    let plan = transformer_plan(false);
    let mut backend = backend;
    let mut model = TransformerModel::load(&source, cfg, &plan, args.max_seq, &backend)?;
    eprintln!(
        "[bench] weights loaded in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let (chunks, rem) = (
        args.prefill_tokens / flint_model::MAX_M,
        args.prefill_tokens % flint_model::MAX_M,
    );
    let mut t = 0u32;

    let warm_ids: Vec<u32> = (0..16).map(|i| i % (args.vocab - 1) + 1).collect();
    model.forward(&mut backend, &warm_ids, &[], &[])?;
    let _ = backend.read_f32(backend.dummy_scale().buf.as_ref(), 0, 1)?;
    let prefill_span = match &mut profiler {
        Some(p) => Some(p.begin_span()?),
        None => None,
    };
    let t0 = Instant::now();
    for _ in 0..chunks {
        let ids: Vec<u32> = (t..t + flint_model::MAX_M)
            .map(|i| i % (args.vocab - 1) + 1)
            .collect();
        model.forward(&mut backend, &ids, &[], &[])?;
        t += flint_model::MAX_M;
    }
    if rem > 0 {
        let ids: Vec<u32> = (t..t + rem).map(|i| i % (args.vocab - 1) + 1).collect();
        model.forward(&mut backend, &ids, &[], &[])?;
    }
    if let (Some(p), Some(span)) = (&mut profiler, prefill_span) {
        p.end_span("prefill", span)?;
    }

    let _ = backend.read_f32(backend.dummy_scale().buf.as_ref(), 0, 1)?;
    let prefill_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[bench] prefill: {} tok in {:.2}s ({:.1} tok/s)",
        args.prefill_tokens,
        prefill_secs,
        args.prefill_tokens as f64 / prefill_secs
    );

    let mut logits: Vec<f32> = Vec::new();
    let mut per_step: Vec<f64> = Vec::new();
    let decode_span = match &mut profiler {
        Some(p) => Some(p.begin_span()?),
        None => None,
    };
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
    if let (Some(p), Some(span)) = (&mut profiler, decode_span) {
        p.end_span("decode", span)?;
    }
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

    if let Some(p) = &mut profiler {
        p.flush()?;
        eprintln!("[bench] GPU time breakdown (cumulative over prefill+decode):");
        eprint!("{}", flint_profiler::breakdown(&p.report()));
    }
    Ok(())
}
