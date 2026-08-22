use thuban_backend::Backend;
use thuban_error::Result;
use thuban_tensor::{DType, Quant, Weight};

pub(super) fn gemv_probe() -> Result<()> {
    use thuban_backend::{Binding, Commands};
    use crate::synth_blocks::synth_blocks;

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let n: u32 = std::env::var("PROBE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(14336);
    let k: u32 = std::env::var("PROBE_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let quant: Quant = std::env::var("PROBE_QTYPE")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Quant::from_ggml)
        .transpose()?
        .unwrap_or(Quant::Q8_0);
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let xb = backend.tensor_f32(&x, vec![k]);
    let wb = match quant {
        Quant::F32 => {
            let w: Vec<f32> = (0..n * k).map(|i| ((i % 97) as f32) * 0.001 - 0.5).collect();
            Weight::plain(backend.tensor_f32(&w, vec![n, k]))
        }
        Quant::F16 => {
            let bytes: Vec<u8> = (0..n * k)
                .flat_map(|i| {
                    thuban_num::f32_to_f16(((i % 97) as f32) * 0.001 - 0.5).to_le_bytes()
                })
                .collect();
            Weight::plain(backend.tensor_f16(&bytes, vec![n, k]).unwrap())
        }
        Quant::Bf16 => {
            let bytes: Vec<u8> = (0..n * k)
                .flat_map(|i| {
                    ((((i % 97) as f32) * 0.001 - 0.5).to_bits() >> 16).to_le_bytes()
                })
                .collect();
            Weight::plain(backend.tensor_bf16(&bytes, vec![n, k]).unwrap())
        }
        _ => Weight::quantized(backend.tensor_quant(
            &synth_blocks(quant, n, k),
            vec![n, k],
            quant,
        )),
    };
    let y = backend.zero_tensor(&[n], DType::F32);
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            let n_dispatch: u32 = std::env::var("PROBE_ND")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            for _ in 0..n_dispatch {
                backend.gemv(
                    &mut commands,
                    Binding::Full(&xb),
                    &[thuban_backend::GemvOp {
                        w: &wb,
                        y: Binding::Full(&y),
                        acc: false,
                    }],
                )?;
            }
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
    let n_dispatch: u32 = std::env::var("PROBE_ND")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let bytes = (n as u64 * quant.row_bytes(k) as u64) as f64 * iters as f64 * n_dispatch as f64;
    eprintln!(
        "[probe] gemv {quant:?} x{iters}x{n_dispatch} bandwidth: {:.1} GB/s ({:.0} us/call)",
        bytes / secs / 1e9,
        secs / iters as f64 * 1e6 / n_dispatch as f64
    );
    Ok(())
}
