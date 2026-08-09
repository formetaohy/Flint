use std::time::Instant;

use clap::Parser;

use flint_architectures::transformer::{TransformerConfig, TransformerModel, transformer_plan};
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

    #[cfg(feature = "profile")]
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
        return bandwidth_probe();
    }
    if args.gemv_probe {
        return gemv_probe();
    }
    if args.cpu_probe {
        return cpu_probe();
    }
    if args.gemm_probe {
        return gemm_probe();
    }
    if args.attn_probe {
        return attn_probe();
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
    #[cfg(feature = "profile")]
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
        args.prefill_tokens / flint_model::M_MAX,
        args.prefill_tokens % flint_model::M_MAX,
    );
    let mut t = 0u32;

    let warm_ids: Vec<u32> = (0..16).map(|i| i % (args.vocab - 1) + 1).collect();
    model.forward(&mut backend, &warm_ids, &[], &[])?;
    let _ = backend.read_f32(backend.dummy_scale().buf.as_ref(), 0, 1)?;
    #[cfg(feature = "profile")]
    let prefill_span = match &mut profiler {
        Some(p) => Some(p.begin_span()?),
        None => None,
    };
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
    #[cfg(feature = "profile")]
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
    #[cfg(feature = "profile")]
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
    #[cfg(feature = "profile")]
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

    #[cfg(feature = "profile")]
    if let Some(p) = &mut profiler {
        p.flush()?;
        eprintln!("[bench] GPU time breakdown (cumulative over prefill+decode):");
        eprint!("{}", flint_profiler::breakdown(&p.report()));
    }
    Ok(())
}

fn bandwidth_probe() -> Result<()> {
    use flint_backend::{Binding, Pass};

    let mut backend = Backend::new()?;
    eprintln!("[bench] adapter: {}", backend.adapter_name());
    let n = 1 << 26; 
    let x = backend.tensor_f32(&vec![1.0; n], vec![n as u32]);
    let y = backend.zero_tensor(&[n as u32]);

    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut pass = Pass::begin(enc.as_mut());
            backend.dispatch(
                &mut pass,
                flint_kernel::name::ADD,
                &[("N_ELEM", n as f64)],
                &[Binding::Full(&x), Binding::Full(&x), Binding::Full(&y)],
                [1024, (n as u32).div_ceil(256 * 1024), 1],
            )?;
        }
        backend.submit(enc).unwrap();
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

    let _ = backend.read_f32(y.buf.as_ref(), 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let bytes = n as f64 * 4.0 * 2.0 * iters as f64; 
    eprintln!("[bench] copy bandwidth: {:.1} GB/s", bytes / secs / 1e9);
    Ok(())
}

fn gemv_probe() -> Result<()> {
    use flint_backend::{Binding, Pass};
    use flint_model::quant::{choose_group, quantize};

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let n = 14336u32;
    let k = 4096u32;
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 97) as f32) * 0.001 - 0.5).collect();
    let group = choose_group(k)?;
    let (bytes, scales) = quantize(&w, n as usize, k as usize, group as usize);
    let xb = backend.tensor_f32(&x, vec![k]);
    let wb = backend.tensor_i8(&bytes, vec![n, k]);
    let sb = backend.tensor_f32(&scales, vec![k / group, n]);
    let y = backend.zero_tensor(&[n]);
    let segs = std::env::var("PROBE_SEGS").map(|v| v.parse().unwrap()).unwrap_or(2u32);
    let kernel: &'static str =
        Box::leak(std::env::var("PROBE_KERNEL").unwrap_or("gemv".into()).into_boxed_str());
    let cols: u32 = if kernel.contains("c4") {
        1024
    } else if kernel.contains("c2") {
        512
    } else if kernel.contains("w512") {
        512
    } else if kernel.contains("w128") {
        128
    } else {
        256
    };
    let partial = backend.zero_tensor(&[segs * n]);
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut pass = Pass::begin(enc.as_mut());
            backend.dispatch(
                &mut pass,
                &kernel,
                &[
                    ("N", n as f64),
                    ("K", k as f64),
                    ("WDTYPE", 1.0),
                    ("GROUP", group as f64),
                    ("SEGS", segs as f64),
                    ("ACC", 0.0),
                ],
                &[
                    Binding::Full(&xb),
                    Binding::Full(&wb),
                    Binding::Full(&sb),
                    Binding::Full(&partial),
                ],
                [n.div_ceil(cols), segs, 1],
            )?;
            backend.dispatch(
                &mut pass,
                flint_kernel::name::MERGE_GEMV,
                &[("N", n as f64), ("SEGS", segs as f64), ("ACC", 0.0)],
                &[Binding::Full(&partial), Binding::Full(&y)],
                [n.div_ceil(256), 1, 1],
            )?;
        }
        backend.submit(enc).unwrap();
        Ok(())
    };

    for _ in 0..15 {
        run(&mut backend)?;
    }
    let t0 = std::time::Instant::now();
    let iters = 80;
    for _ in 0..iters {
        run(&mut backend)?;
    }
    let _ = backend.read_f32(y.buf.as_ref(), 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let bytes = n as f64 * k as f64 * iters as f64;
    eprintln!(
        "[probe] gemv bandwidth: {:.1} GB/s ({:.0} us/call)",
        bytes / secs / 1e9,
        secs / iters as f64 * 1e6
    );
    Ok(())
}

fn cpu_probe() -> Result<()> {
    use flint_backend::{Binding, Pass};
    use flint_model::quant::{choose_group, quantize};

    let mut backend = Backend::new()?;
    let n = 14336u32;
    let k = 4096u32;
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 97) as f32) * 0.001 - 0.5).collect();
    let group = choose_group(k)?;
    let (bytes, scales) = quantize(&w, n as usize, k as usize, group as usize);
    let xb = backend.tensor_f32(&x, vec![k]);
    let wb = backend.tensor_i8(&bytes, vec![n, k]);
    let sb = backend.tensor_f32(&scales, vec![k / group, n]);
    let y = backend.zero_tensor(&[n]);
    let partial = backend.zero_tensor(&[2 * n]);

    for _ in 0..3 {
        let mut enc = backend.encoder().unwrap();
        {
            let mut pass = Pass::begin(enc.as_mut());
            backend.dispatch(
                &mut pass,
                flint_kernel::name::GEMV,
                &[
                    ("N", n as f64),
                    ("K", k as f64),
                    ("WDTYPE", 1.0),
                    ("GROUP", group as f64),
                    ("SEGS", 2.0),
                    ("ACC", 0.0),
                ],
                &[
                    Binding::Full(&xb),
                    Binding::Full(&wb),
                    Binding::Full(&sb),
                    Binding::Full(&partial),
                ],
                [n.div_ceil(256), 2, 1],
            )?;
        }
        backend.submit(enc)?;
    }

    let kernel = backend.kernel("gemv")?;
    let mut enc = backend.encoder().unwrap();
    {
        let mut pass = Pass::begin(enc.as_mut());
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", 1.0),
            ("GROUP", group as f64),
            ("SEGS", 2.0),
            ("ACC", 0.0),
        ];
        let scalars = backend.pack_scalars("gemv", &consts)?;
        let binds = [
            saturn_core::BindingRef { index: 0, buffer: xb.buf.as_ref(), offset: 0, size: 0 },
            saturn_core::BindingRef { index: 1, buffer: wb.buf.as_ref(), offset: 0, size: 0 },
            saturn_core::BindingRef { index: 2, buffer: sb.buf.as_ref(), offset: 0, size: 0 },
            saturn_core::BindingRef { index: 3, buffer: partial.buf.as_ref(), offset: 0, size: 0 },
        ];

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            pass.raw().bind(kernel, &binds)?;
        }
        eprintln!("[probe] bind: {:.1} us", t0.elapsed().as_secs_f64() / 100.0 * 1e6);

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            pass.raw().set_scalars(kernel, &scalars)?;
        }
        eprintln!("[probe] set_scalars: {:.1} us", t0.elapsed().as_secs_f64() / 100.0 * 1e6);

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            pass.raw().dispatch([56, 2, 1])?;
        }
        eprintln!("[probe] dispatch: {:.1} us", t0.elapsed().as_secs_f64() / 100.0 * 1e6);

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            pass.raw().barrier()?;
        }
        eprintln!("[probe] barrier: {:.1} us", t0.elapsed().as_secs_f64() / 100.0 * 1e6);
    }
    let encoders = vec![enc];

    let t0 = std::time::Instant::now();
    for enc in encoders {
        backend.submit(enc)?;
    }
    let _ = backend.read_f32(y.buf.as_ref(), 0, 1)?;
    eprintln!(
        "[probe] submit+drain: {:.2} ms total ({} us/dispatch)",
        t0.elapsed().as_secs_f64() * 1e3,
        t0.elapsed().as_secs_f64() / 100.0 * 1e6
    );
    Ok(())
}

fn gemm_probe() -> Result<()> {
    use flint_backend::{Binding, Pass};
    use flint_model::quant::{choose_group, quantize};

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let m = 128u32;
    let n = 14336u32;
    let k = 4096u32;
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001 - 2.0) * 0.5).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 97) as f32) * 0.001 - 0.5).collect();
    let group = choose_group(k)?;
    let (bytes, scales) = quantize(&w, n as usize, k as usize, group as usize);
    let xb = backend.tensor_f32(&x, vec![m, k]);
    let wb = backend.tensor_i8(&bytes, vec![n, k]);
    let sb = backend.tensor_f32(&scales, vec![n, k / group]);
    let y = backend.zero_tensor(&[m, n]);
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut pass = Pass::begin(enc.as_mut());
            backend.dispatch(
                &mut pass,
                flint_kernel::name::GEMM,
                &[
                    ("N", n as f64),
                    ("K", k as f64),
                    ("M", m as f64),
                    ("SEGS", 1.0),
                    ("WDTYPE", 1.0),
                    ("GROUP", group as f64),
                    ("ACC", 0.0),
                    ("Y_STRIDE", n as f64),
                    ("Y_OFF", 0.0),
                ],
                &[Binding::Full(&xb), Binding::Full(&wb), Binding::Full(&sb), Binding::Full(&y)],
                [n.div_ceil(32), m.div_ceil(32), 1],
            )?;
        }
        backend.submit(enc).unwrap();
        Ok(())
    };
    for _ in 0..40 {
        run(&mut backend)?;
    }
    let t0 = std::time::Instant::now();
    let iters = 60;
    for _ in 0..iters {
        run(&mut backend)?;
    }
    let _ = backend.read_f32(y.buf.as_ref(), 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
    eprintln!(
        "[probe] gemm: {:.2} TFLOPS ({:.1} us/call)",
        flops / secs / 1e12,
        secs / iters as f64 * 1e6
    );
    Ok(())
}

fn attn_probe() -> Result<()> {
    use flint_backend::{Binding, Pass};
    use flint_model::ops::{M_MAX, step_args};
    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let (m, nq, nkv, hd, max_seq, pos) = (128u32, 32u32, 8u32, 128u32, 2048u32, 1024u32);
    let q: Vec<f32> = (0..m * nq * hd).map(|i| ((i as f32) * 0.001 - 0.5) * 0.1).collect();
    let kc: Vec<f32> = (0..nkv * max_seq * hd).map(|i| ((i as f32) * 0.0003 - 0.5) * 0.1).collect();
    let vc: Vec<f32> = (0..nkv * max_seq * hd).map(|i| ((i as f32) * 0.0005 - 0.5) * 0.1).collect();
    let qb = backend.tensor_f32(&q, vec![m, nq, hd]);
    let kbytes: Vec<u8> = kc.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let vbytes: Vec<u8> = vc.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let kb = backend.tensor_bf16(&kbytes, vec![nkv, max_seq, hd]).unwrap();
    let vb = backend.tensor_bf16(&vbytes, vec![nkv, max_seq, hd]).unwrap();
    let y = backend.zero_tensor(&[m, nq, hd]);
    let _ = M_MAX;
    let scratch = backend.zero_tensor(&[m, nkv, 32, 8, hd + 2]);
    let segs = std::env::var("PROBE_SEGS")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|_| (pos + m).div_ceil(256).clamp(1, 32));
    let args = step_args(&backend);
    backend.write_u32(args.buf.as_ref(), &[pos, segs]);
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut pass = Pass::begin(enc.as_mut());
            backend.dispatch(
                &mut pass,
                flint_kernel::name::ATTN,
                &[
                    ("N_HEADS", nq as f64),
                    ("KV_HEADS", nkv as f64),
                    ("HEAD_DIM", hd as f64),
                    ("MAX_SEQ", max_seq as f64),
                    ("SCALE", 1.0 / (hd as f64).sqrt()),
                    ("WINDOW", 0.0),
                    ("NQ_PER_KV", (nq / nkv) as f64),
                    ("STRIDE", (hd + 2) as f64),
                ],
                &[
                    Binding::Full(&qb),
                    Binding::Full(&kb),
                    Binding::Full(&vb),
                    Binding::Full(&scratch),
                    Binding::Full(&args),
                ],
                [m, nkv, segs],
            )?;
        }
        backend.submit(enc).unwrap();
        Ok(())
    };
    for _ in 0..10 {
        run(&mut backend)?;
    }
    let t0 = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters {
        run(&mut backend)?;
    }
    let _ = backend.read_f32(y.buf.as_ref(), 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[probe] attn M={m} kv={} segs={segs}: {:.2} ms/call",
        pos + m,
        secs / iters as f64 * 1e3
    );
    Ok(())
}
