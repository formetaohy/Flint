use thuban_backend::Backend;
use thuban_error::Result;
use thuban_tensor::{DType, Quant};

pub(super) fn gemm_probe() -> Result<()> {
    use thuban_backend::{Binding, Commands};
    use crate::synth_blocks::synth_blocks;

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let env = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let m = env("PROBE_M", 128);
    let n = env("PROBE_N", 14336);
    let k = env("PROBE_K", 4096);
    let quant: Quant = std::env::var("PROBE_QTYPE")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Quant::from_ggml)
        .transpose()?
        .unwrap_or(Quant::Q8_0);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.001 - 2.0) * 0.5)
        .collect();
    let xb = backend.tensor_f32(&x, vec![m, k]);
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 97) as f32) * 0.001 - 0.5)
        .collect();
    let wb = match quant {
        Quant::Bf16 => {
            let wbytes: Vec<u8> = w
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            backend.tensor_bf16(&wbytes, vec![n, k]).unwrap()
        }
        Quant::F16 => {
            let wbytes: Vec<u8> = w
                .iter()
                .flat_map(|v| thuban_num::f32_to_f16(*v).to_le_bytes())
                .collect();
            backend.tensor_f16(&wbytes, vec![n, k]).unwrap()
        }
        Quant::F32 => backend.tensor_f32(&w, vec![n, k]),
        _ => backend.tensor_quant(&synth_blocks(quant, n, k), vec![n, k], quant),
    };
    let qtype = quant.as_u32();
    let kernel_name: &'static str = Box::leak(
        std::env::var("PROBE_KERNEL")
            .unwrap_or("gemm".into())
            .into_boxed_str(),
    );
    let coop = kernel_name.contains("coop");
    let tn: u32 = 128;
    let tm: u32 = if coop { 128 } else { 64 };
    let y = backend.zero_tensor(&[m, n], DType::F32);
    let xf16 = if coop {
        let xf = backend.zero_tensor(&[m * k], DType::F16);
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
                thuban_kernel::shader::TO_F16,
                &[("N_ELEM", (m * k) as f64)],
                &[Binding::Full(&xb), Binding::Full(&xf)],
                [(m * k / 4).div_ceil(256), 1, 1],
            )?;
        }
        backend.submit(&mut enc)?;
        Some(xf)
    } else {
        None
    };
    let scalars = backend.pack_scalars(
        kernel_name,
        &[
            ("N", n as f64),
            ("K", k as f64),
            ("M", m as f64),
            ("SEGS", 1.0),
            ("QTYPE", qtype as f64),
            ("ACC", 0.0),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
        ],
    )?;
    let single = std::env::var("PROBE_SINGLE").is_ok();
    if single {
        for _ in 0..40 {
            let mut enc = backend.encoder().unwrap();
            {
                let lut = thuban_gpu::BindingRef {
                    index: 2,
                    buffer: &backend.quant_lut().buf,
                    offset: 0,
                    size: 0,
                };
                let binds = [
                    thuban_gpu::BindingRef {
                        index: 0,
                        buffer: &xf16.as_ref().unwrap_or(&xb).buf,
                        offset: 0,
                        size: 0,
                    },
                    thuban_gpu::BindingRef {
                        index: 1,
                        buffer: &wb.buf,
                        offset: 0,
                        size: 0,
                    },
                    lut,
                    thuban_gpu::BindingRef {
                        index: 3,
                        buffer: &y.buf,
                        offset: 0,
                        size: 0,
                    },
                ];
                let mut commands = Commands::begin(&mut enc);
                let k = backend.kernel(kernel_name)?;
                commands.raw().bind(k, &binds)?;
                commands.raw().set_scalars(&scalars)?;
                commands
                    .raw()
                    .dispatch([n.div_ceil(tn), m.div_ceil(tm), 1])?;
            }
            backend.submit(&mut enc).unwrap();
        }
        let t0 = std::time::Instant::now();
        let iters = 60;
        let mut enc = backend.encoder().unwrap();
        {
            let lut = thuban_gpu::BindingRef {
                index: 2,
                buffer: &backend.quant_lut().buf,
                offset: 0,
                size: 0,
            };
            let binds = [
                thuban_gpu::BindingRef {
                    index: 0,
                    buffer: &xf16.as_ref().unwrap_or(&xb).buf,
                    offset: 0,
                    size: 0,
                },
                thuban_gpu::BindingRef {
                    index: 1,
                    buffer: &wb.buf,
                    offset: 0,
                    size: 0,
                },
                lut,
                thuban_gpu::BindingRef {
                    index: 3,
                    buffer: &y.buf,
                    offset: 0,
                    size: 0,
                },
            ];
            let mut commands = Commands::begin(&mut enc);
            let k = backend.kernel(kernel_name)?;
            commands.raw().bind(k, &binds)?;
            commands.raw().set_scalars(&scalars)?;
            for _ in 0..iters {
                commands
                    .raw()
                    .dispatch([n.div_ceil(tn), m.div_ceil(tm), 1])?;
            }
        }
        backend.submit(&mut enc).unwrap();
        let _ = backend.read_f32(&y.buf, 0, 1)?;
        let secs = t0.elapsed().as_secs_f64();
        let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
        eprintln!(
            "[probe] gemm {kernel_name} single-enc qtype={qtype}: {:.2} TFLOPS ({:.1} us/call)",
            flops / secs / 1e12,
            secs / iters as f64 * 1e6
        );
        return Ok(());
    }
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            if let Some(xf) = &xf16 {
                backend.dispatch(
                    &mut commands,
                    thuban_kernel::shader::TO_F16,
                    &[("N_ELEM", (m * k) as f64)],
                    &[Binding::Full(&xb), Binding::Full(xf)],
                    [(m * k / 4).div_ceil(256), 1, 1],
                )?;
            }
            backend.dispatch(
                &mut commands,
                kernel_name,
                &[
                    ("N", n as f64),
                    ("K", k as f64),
                    ("M", m as f64),
                    ("SEGS", 1.0),
                    ("QTYPE", qtype as f64),
                    ("ACC", 0.0),
                    ("Y_STRIDE", n as f64),
                    ("Y_OFF", 0.0),
                ],
                &[
                    Binding::Full(xf16.as_ref().unwrap_or(&xb)),
                    Binding::Full(&wb),
                    Binding::Full(backend.quant_lut()),
                    Binding::Full(&y),
                ],
                [n.div_ceil(tn), m.div_ceil(tm), 1],
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
        "[probe] gemm {kernel_name} qtype={qtype}: {:.2} TFLOPS ({:.1} us/call)",
        flops / secs / 1e12,
        secs / iters as f64 * 1e6
    );
    Ok(())
}
