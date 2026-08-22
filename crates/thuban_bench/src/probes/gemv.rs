use thuban_backend::Backend;
use thuban_error::Result;
use thuban_tensor::{DType, Quant};

pub(super) fn gemv_probe() -> Result<()> {
    use thuban_backend::{Binding, Commands};
    use crate::synth_blocks::synth_blocks;

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let n = 14336u32;
    let k = 4096u32;
    let quant: Quant = std::env::var("PROBE_QTYPE")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Quant::from_ggml)
        .transpose()?
        .unwrap_or(Quant::Q8_0);
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let blocks = synth_blocks(quant, n, k);
    let xb = backend.tensor_f32(&x, vec![k]);
    let wb = backend.tensor_quant(&blocks, vec![n, k], quant);
    let y = backend.zero_tensor(&[n], DType::F32);
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
    } else {
        64
    };
    let partial = backend.zero_tensor(&[segs * n], DType::F32);
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
                    ("QTYPE", quant.as_u32() as f64),
                    ("SEGS", segs as f64),
                    ("ACC", 0.0),
                ],
                &[
                    Binding::Full(&xb),
                    Binding::Full(&wb),
                    Binding::Full(backend.quant_lut()),
                    Binding::Full(&partial),
                ],
                [n.div_ceil(cols), segs, 1],
            )?;
            backend.dispatch(
                &mut commands,
                thuban_kernel::shader::MERGE_GEMV,
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
    let bytes = (n as u64 * quant.row_bytes(k) as u64) as f64 * iters as f64;
    eprintln!(
        "[probe] gemv {quant:?} bandwidth: {:.1} GB/s ({:.0} us/call)",
        bytes / secs / 1e9,
        secs / iters as f64 * 1e6
    );
    Ok(())
}
