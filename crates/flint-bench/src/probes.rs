use flint_backend::Backend;
use flint_error::Result;

fn caps_probe() -> Result<()> {
    let backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let device = backend.device();
    eprintln!(
        "[probe] subgroup size: {}-{}",
        device.subgroup_min_size(),
        device.subgroup_max_size()
    );
    let props = device.cooperative_matrix_properties();
    eprintln!("[probe] cooperative matrix configs: {}", props.len());
    for p in props {
        eprintln!(
            "[probe]   coop {}x{}x{} a/b={:?} c/r={:?} saturating={}",
            p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, p.saturating_accumulation
        );
    }
    Ok(())
}

fn bandwidth_probe() -> Result<()> {
    use flint_backend::{Binding, Commands};

    let mut backend = Backend::new()?;
    eprintln!("[bench] adapter: {}", backend.adapter_name());
    let n = 1 << 26;
    let x = backend.tensor_f32(&vec![1.0; n], vec![n as u32]);
    let y = backend.zero_tensor(&[n as u32]);

    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
                flint_kernel::name::ADD,
                &[("N_ELEM", n as f64)],
                &[Binding::Full(&x), Binding::Full(&x), Binding::Full(&y)],
                [1024, (n as u32).div_ceil(256 * 1024), 1],
            )?;
        }
        backend.submit(&mut enc).unwrap();
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

    let _ = backend.read_f32(&y.buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let bytes = n as f64 * 4.0 * 2.0 * iters as f64;
    eprintln!("[bench] copy bandwidth: {:.1} GB/s", bytes / secs / 1e9);
    Ok(())
}

fn gemv_probe() -> Result<()> {
    use flint_backend::{Binding, Commands};
    use flint_model::quant::{choose_group, quantize};

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let n = 14336u32;
    let k = 4096u32;
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 97) as f32) * 0.001 - 0.5)
        .collect();
    let group = choose_group(k)?;
    let (bytes, scales) = quantize(&w, n as usize, k as usize, group as usize);
    let xb = backend.tensor_f32(&x, vec![k]);
    let wb = backend.tensor_i8(&bytes, vec![n, k]);
    let sb = backend.tensor_f32(&scales, vec![k / group, n]);
    let y = backend.zero_tensor(&[n]);
    let segs = std::env::var("PROBE_SEGS")
        .map(|v| v.parse().unwrap())
        .unwrap_or(2u32);
    let kernel: &'static str = Box::leak(
        std::env::var("PROBE_KERNEL")
            .unwrap_or("gemv".into())
            .into_boxed_str(),
    );
    let cols: u32 = if kernel.contains("c4") {
        1024
    } else if kernel.contains("c2") || kernel.contains("w512") {
        512
    } else if kernel.contains("w128") {
        128
    } else if kernel.contains("v4") {
        512
    } else {
        256
    };
    let partial = backend.zero_tensor(&[segs * n]);
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
                kernel,
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
                &mut commands,
                flint_kernel::name::MERGE_GEMV,
                &[("N", n as f64), ("SEGS", segs as f64), ("ACC", 0.0)],
                &[Binding::Full(&partial), Binding::Full(&y)],
                [n.div_ceil(256), 1, 1],
            )?;
        }
        backend.submit(&mut enc).unwrap();
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
    let _ = backend.read_f32(&y.buf, 0, 1)?;
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
    use flint_backend::{Binding, Commands};
    use flint_model::quant::{choose_group, quantize};

    let mut backend = Backend::new()?;
    let n = 14336u32;
    let k = 4096u32;
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 97) as f32) * 0.001 - 0.5)
        .collect();
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
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
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
        backend.submit(&mut enc)?;
    }

    let mut enc = backend.encoder().unwrap();
    {
        let mut commands = Commands::begin(&mut enc);
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", 1.0),
            ("GROUP", group as f64),
            ("SEGS", 2.0),
            ("ACC", 0.0),
        ];
        let scalars = backend.pack_scalars("gemv", &consts)?;
        let kernel = backend.kernel("gemv")?;
        let binds = [
            flint_gpu::BindingRef {
                index: 0,
                buffer: &xb.buf,
                offset: 0,
                size: 0,
            },
            flint_gpu::BindingRef {
                index: 1,
                buffer: &wb.buf,
                offset: 0,
                size: 0,
            },
            flint_gpu::BindingRef {
                index: 2,
                buffer: &sb.buf,
                offset: 0,
                size: 0,
            },
            flint_gpu::BindingRef {
                index: 3,
                buffer: &partial.buf,
                offset: 0,
                size: 0,
            },
        ];

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            commands.raw().bind(kernel, &binds)?;
        }
        eprintln!(
            "[probe] bind: {:.1} us",
            t0.elapsed().as_secs_f64() / 100.0 * 1e6
        );

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            commands.raw().set_scalars(&scalars)?;
        }
        eprintln!(
            "[probe] set_scalars: {:.1} us",
            t0.elapsed().as_secs_f64() / 100.0 * 1e6
        );

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            commands.raw().dispatch([56, 2, 1])?;
        }
        eprintln!(
            "[probe] dispatch: {:.1} us",
            t0.elapsed().as_secs_f64() / 100.0 * 1e6
        );
    }

    let t0 = std::time::Instant::now();
    backend.submit(&mut enc)?;
    let _ = backend.read_f32(&y.buf, 0, 1)?;
    eprintln!(
        "[probe] submit+drain: {:.2} ms total ({} us/dispatch)",
        t0.elapsed().as_secs_f64() * 1e3,
        t0.elapsed().as_secs_f64() / 100.0 * 1e6
    );
    Ok(())
}

fn gemm_probe() -> Result<()> {
    use flint_backend::{Binding, Commands};
    use flint_model::quant::{choose_group, quantize};

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let m = 128u32;
    let n = 14336u32;
    let k = 4096u32;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.001 - 2.0) * 0.5)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 97) as f32) * 0.001 - 0.5)
        .collect();
    let group = choose_group(k)?;
    let (bytes, scales) = quantize(&w, n as usize, k as usize, group as usize);
    let xb = backend.tensor_f32(&x, vec![m, k]);
    let wdtype: f64 = std::env::var("PROBE_WDTYPE")
        .map(|v| v.parse().unwrap())
        .unwrap_or(1.0);
    let (wb, sb) = if wdtype == 0.0 {
        let wbytes: Vec<u8> = w
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let bf = backend.tensor_bf16(&wbytes, vec![n, k]).unwrap();
        let s = backend.tensor_f32(&[1.0], vec![1]);
        (bf, s)
    } else {
        let wi = backend.tensor_i8(&bytes, vec![n, k]);
        let si = backend.tensor_f32(&scales, vec![n, k / group]);
        (wi, si)
    };
    let kernel_name: &'static str = Box::leak(
        std::env::var("PROBE_KERNEL")
            .unwrap_or("gemm".into())
            .into_boxed_str(),
    );
    let tm: u32 = if kernel_name.contains("coop") { 64 } else { 32 };
    let y = backend.zero_tensor(&[m, n]);
    let scalars = backend.pack_scalars(kernel_name, &[
        ("N", n as f64),
        ("K", k as f64),
        ("M", m as f64),
        ("SEGS", 1.0),
        ("WDTYPE", wdtype),
        ("GROUP", group as f64),
        ("ACC", 0.0),
        ("Y_STRIDE", n as f64),
        ("Y_OFF", 0.0),
    ])?;
    let binds = [
        flint_gpu::BindingRef {
            index: 0,
            buffer: &xb.buf,
            offset: 0,
            size: 0,
        },
        flint_gpu::BindingRef {
            index: 1,
            buffer: &wb.buf,
            offset: 0,
            size: 0,
        },
        flint_gpu::BindingRef {
            index: 2,
            buffer: &sb.buf,
            offset: 0,
            size: 0,
        },
        flint_gpu::BindingRef {
            index: 3,
            buffer: &y.buf,
            offset: 0,
            size: 0,
        },
    ];
    let single = std::env::var("PROBE_SINGLE").is_ok();
    if single {
        for _ in 0..40 {
            let mut enc = backend.encoder().unwrap();
            {
                let mut commands = Commands::begin(&mut enc);
                let k = backend.kernel(kernel_name)?;
                commands.raw().bind(k, &binds)?;
                commands.raw().set_scalars(&scalars)?;
                commands.raw().dispatch([n.div_ceil(32), m.div_ceil(tm), 1])?;
            }
            backend.submit(&mut enc).unwrap();
        }
        let t0 = std::time::Instant::now();
        let iters = 60;
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            let k = backend.kernel(kernel_name)?;
            commands.raw().bind(k, &binds)?;
            commands.raw().set_scalars(&scalars)?;
            for _ in 0..iters {
                commands.raw().dispatch([n.div_ceil(32), m.div_ceil(tm), 1])?;
            }
        }
        backend.submit(&mut enc).unwrap();
        let _ = backend.read_f32(&y.buf, 0, 1)?;
        let secs = t0.elapsed().as_secs_f64();
        let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
        eprintln!(
            "[probe] gemm {kernel_name} single-enc wdtype={wdtype}: {:.2} TFLOPS ({:.1} us/call)",
            flops / secs / 1e12,
            secs / iters as f64 * 1e6
        );
        return Ok(());
    }
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
                kernel_name,
                &[
                    ("N", n as f64),
                    ("K", k as f64),
                    ("M", m as f64),
                    ("SEGS", 1.0),
                    ("WDTYPE", wdtype),
                    ("GROUP", group as f64),
                    ("ACC", 0.0),
                    ("Y_STRIDE", n as f64),
                    ("Y_OFF", 0.0),
                ],
                &[
                    Binding::Full(&xb),
                    Binding::Full(&wb),
                    Binding::Full(&sb),
                    Binding::Full(&y),
                ],
                [n.div_ceil(32), m.div_ceil(tm), 1],
            )?;
        }
        backend.submit(&mut enc).unwrap();
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
    let _ = backend.read_f32(&y.buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
    eprintln!(
        "[probe] gemm {kernel_name} wdtype={wdtype}: {:.2} TFLOPS ({:.1} us/call)",
        flops / secs / 1e12,
        secs / iters as f64 * 1e6
    );
    Ok(())
}

fn attn_probe() -> Result<()> {
    use flint_backend::{Binding, Commands};
    use flint_model::step::step_args;

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let m: u32 = std::env::var("PROBE_M").map(|v| v.parse().unwrap()).unwrap_or(128);
    let pos: u32 = std::env::var("PROBE_POS").map(|v| v.parse().unwrap()).unwrap_or(1024);
    let (nq, nkv, hd, max_seq) = (32u32, 8u32, 128u32, 2048u32);
    let q: Vec<f32> = (0..m * nq * hd)
        .map(|i| ((i as f32) * 0.001 - 0.5) * 0.1)
        .collect();
    let kc: Vec<f32> = (0..nkv * max_seq * hd)
        .map(|i| ((i as f32) * 0.0003 - 0.5) * 0.1)
        .collect();
    let vc: Vec<f32> = (0..nkv * max_seq * hd)
        .map(|i| ((i as f32) * 0.0005 - 0.5) * 0.1)
        .collect();
    let qb = backend.tensor_f32(&q, vec![m, nq, hd]);
    let kbytes: Vec<u8> = kc
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let vbytes: Vec<u8> = vc
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let kb = backend
        .tensor_bf16(&kbytes, vec![nkv, max_seq, hd])
        .unwrap();
    let vb = backend
        .tensor_bf16(&vbytes, vec![nkv, max_seq, hd])
        .unwrap();
    let y = backend.zero_tensor(&[m, nq, hd]);
    let scratch = backend.zero_tensor(&[m, nkv, 32, 8, hd + 2]);
    let segs = std::env::var("PROBE_SEGS")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|_| (pos + m).div_ceil(256).clamp(1, 32));
    let args = step_args(&backend);
    backend.write_u32(&args.buf, &[pos, segs]);
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
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
        backend.submit(&mut enc).unwrap();
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
    let _ = backend.read_f32(&y.buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[probe] attn M={} kv={pos} segs={segs}: {:.2} ms/call",
        pos + m,
        secs / iters as f64 * 1e3
    );
    Ok(())
}

pub fn run(name: &str) -> Result<()> {
    match name {
        "caps" => caps_probe(),
        "bandwidth" => bandwidth_probe(),
        "gemv" => gemv_probe(),
        "cpu" => cpu_probe(),
        "gemm" => gemm_probe(),
        "attn" => attn_probe(),
        other => Err(flint_error::Error::Model(format!(
            "unknown probe {other:?}"
        ))),
    }
}
