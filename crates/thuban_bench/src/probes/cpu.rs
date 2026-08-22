use thuban_backend::Backend;
use thuban_error::Result;
use thuban_tensor::{DType, Quant};

pub(super) fn cpu_probe() -> Result<()> {
    use thuban_backend::{Binding, Commands};
    use crate::synth_blocks::synth_blocks;

    let backend = Backend::new()?;
    let n = 14336u32;
    let k = 4096u32;
    let quant = Quant::Q8_0;
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let blocks = synth_blocks(quant, n, k);
    let xb = backend.tensor_f32(&x, vec![k]);
    let wb = backend.tensor_quant(&blocks, vec![n, k], quant);
    let y = backend.zero_tensor(&[n], DType::F32);
    let partial = backend.zero_tensor(&[2 * n], DType::F32);

    for _ in 0..3 {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
                thuban_kernel::shader::GEMV,
                &[
                    ("N", n as f64),
                    ("K", k as f64),
                    ("QTYPE", quant.as_u32() as f64),
                    ("SEGS", 2.0),
                    ("ACC", 0.0),
                ],
                &[
                    Binding::Full(&xb),
                    Binding::Full(&wb),
                    Binding::Full(backend.quant_lut()),
                    Binding::Full(&partial),
                ],
                [n.div_ceil(64), 2, 1],
            )?;
        }
        backend.submit(&mut enc)?;
    }

    let kernel = backend.kernel("gemv")?;
    let mut enc = backend.encoder().unwrap();
    {
        let mut commands = Commands::begin(&mut enc);
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("QTYPE", quant.as_u32() as f64),
            ("SEGS", 2.0),
            ("ACC", 0.0),
        ];
        let scalars = backend.pack_scalars("gemv", &consts)?;
        let binds = [
            thuban_gpu::BindingRef {
                index: 0,
                buffer: &xb.buf,
                offset: 0,
                size: 0,
            },
            thuban_gpu::BindingRef {
                index: 1,
                buffer: &wb.buf,
                offset: 0,
                size: 0,
            },
            thuban_gpu::BindingRef {
                index: 2,
                buffer: &backend.quant_lut().buf,
                offset: 0,
                size: 0,
            },
            thuban_gpu::BindingRef {
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
