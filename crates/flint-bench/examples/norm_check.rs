use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_model::ops::{self, NormMode};

fn main() -> Result<()> {
    let mut backend = Backend::new()?;
    let dim = 2560u32;
    let x: Vec<f32> = (0..16 * dim)
        .map(|i| ((i as f32) * 0.13).sin() * 0.5)
        .collect();
    let w: Vec<f32> = (0..dim)
        .map(|i| 1.0 + ((i as f32) * 0.01).sin() * 0.1)
        .collect();
    let ones: Vec<f32> = vec![1.0; dim as usize];
    let xt = backend.tensor_f32(&x, vec![16, dim]);
    let wt = backend.tensor_f32(&w, vec![dim]);
    let ot = backend.tensor_f32(&ones, vec![dim]);
    let yt = backend.zero_tensor(&[16, dim]);
    let mut enc = backend.encoder().unwrap();
    {
        let mut pass = Pass::begin(enc.as_mut());
        ops::norm(
            &mut backend,
            &mut pass,
            NormMode::Direct,
            Binding::Full(&xt),
            &wt,
            Binding::Full(&ot),
            Binding::Full(&yt),
            16,
            dim,
            dim,
            1e-6,
        )?;
    }
    backend.submit(enc).unwrap();
    let y = backend.read_f32(yt.buf.as_ref(), 0, (16 * dim) as usize)?;
    let mut max_err = 0f32;
    for r in 0..16usize {
        let row = &x[r * dim as usize..(r + 1) * dim as usize];
        let s: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv = (s / dim as f64 + 1e-6).sqrt().recip();
        for i in 0..dim as usize {
            let want = row[i] * inv as f32 * w[i];
            max_err = max_err.max((y[r * dim as usize + i] - want).abs());
        }
    }
    eprintln!("norm MODE2 max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model("norm mismatch".into()));
    }
    Ok(())
}
