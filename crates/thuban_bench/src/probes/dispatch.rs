use thuban_backend::Backend;
use thuban_error::Result;
use thuban_tensor::DType;

pub(super) fn dispatch_probe() -> Result<()> {
    use thuban_backend::{Binding, Commands};

    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let groups: u32 = std::env::var("PROBE_WGS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let x = backend.tensor_f32(&vec![1.0; groups as usize * 256], vec![groups * 256]);
    let y = backend.zero_tensor(&[groups * 256], DType::F32);
    let run = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            let n_dispatch: u32 = std::env::var("PROBE_ND")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            for _ in 0..n_dispatch {
                backend.dispatch(
                    &mut commands,
                    thuban_kernel::shader::ADD,
                    &[("N_ELEM", (groups * 256) as f64)],
                    &[Binding::Full(&x), Binding::Full(&x), Binding::Full(&y)],
                    [groups, 1, 1],
                )?;
            }
        }
        backend.submit(&mut enc).unwrap();
        Ok(())
    };
    for _ in 0..20 {
        run(&mut backend)?;
    }
    let t0 = std::time::Instant::now();
    let iters = 200;
    for _ in 0..iters {
        run(&mut backend)?;
    }
    let _ = backend.read_f32(&y.buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[probe] add {groups} wg x200: {:.2} us/call",
        secs / iters as f64 * 1e6
    );
    Ok(())
}
