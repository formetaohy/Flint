//! Kernel correctness check: gemv/gemm against a CPU reference, exercising
//! both the block-major i8 layout and the row-major bf16 path.
//! Run: cargo run --release --example gemv_check

use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_model::loader::quantize;
use flint_tensor::Weight;

/// Block-major [K/16, N, 16] i8 decode: byte (kb, n, i) at kb*N*16 + n*16 + i.
#[allow(clippy::too_many_arguments)]
fn cpu_dequant_i8(
    weights: &[u8],
    scales: &[f32],
    n: usize,
    n_total: usize,
    k: usize,
    group: usize,
    kb: usize,
    i: usize,
) -> f32 {
    let byte = weights[kb * n_total * 16 + n * 16 + i];
    let s = scales[n * (k / group) + kb * 16 / group];
    (byte as i8 as f32) * s
}

fn cpu_gemv(x: &[f32], bytes: &[u8], scales: &[f32], n: usize, k: usize, group: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    for (nn, wy) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for kb in 0..k / 16 {
            for i in 0..16 {
                acc += x[kb * 16 + i] * cpu_dequant_i8(bytes, scales, nn, n, k, group, kb, i);
            }
        }
        *wy = acc;
    }
    y
}

fn check_gemv_tall(backend: &mut Backend, n: u32, k: u32, group: u32) -> Result<()> {
    let data: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.137 - 3.0).sin() * 0.5)
        .collect();
    let (bytes, scales) = quantize(&data, n as usize, k as usize, group as usize);
    let w = Weight::quant(
        backend.tensor_i8(&bytes, vec![n, k], "w"),
        backend.tensor_f32(&scales, vec![n, k / group], "ws"),
        group,
    );
    // x is a [16, K] tile; the kernel must read row 0. y is a [16, N] tile.
    let xs: Vec<f32> = (0..16 * k).map(|i| ((i as f32) * 0.31).cos()).collect();
    let xt = backend.tensor_f32(&xs, vec![16, k], "x");
    let yt = backend.zero_tensor(&[16, n], "y");
    let mut enc = backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "t");
        backend.gemv(&mut pass, Binding::Full(&xt), &w, Binding::Full(&yt))?;
    }
    backend.submit(enc);
    let y = backend.read_f32(&yt.buf, 0, n as usize)?;
    let ref_y = cpu_gemv(
        &xs[..k as usize],
        &bytes,
        &scales,
        n as usize,
        k as usize,
        group as usize,
    );
    let mut max_err = 0f32;
    for (a, b) in y.iter().zip(ref_y.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("gemv-tall N={n} K={k} G={group}: max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model(format!(
            "gemv-tall mismatch: max_err {max_err:.2e}"
        )));
    }
    Ok(())
}

fn check_gemv(backend: &mut Backend, n: u32, k: u32, group: u32) -> Result<()> {
    let data: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.137 - 3.0).sin() * 0.5)
        .collect();
    let (bytes, scales) = quantize(&data, n as usize, k as usize, group as usize);
    let w = Weight::quant(
        backend.tensor_i8(&bytes, vec![n, k], "w"),
        backend.tensor_f32(&scales, vec![n, k / group], "ws"),
        group,
    );
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.31).cos()).collect();
    let xt = backend.tensor_f32(&x, vec![k], "x");
    let yt = backend.zero_tensor(&[n], "y");
    let mut enc = backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "t");
        backend.gemv(&mut pass, Binding::Full(&xt), &w, Binding::Full(&yt))?;
    }
    backend.submit(enc);
    let y = backend.read_f32(&yt.buf, 0, n as usize)?;
    let ref_y = cpu_gemv(&x, &bytes, &scales, n as usize, k as usize, group as usize);
    let mut max_err = 0f32;
    for (a, b) in y.iter().zip(ref_y.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("gemv N={n} K={k} G={group}: max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model(format!(
            "gemv mismatch: max_err {max_err:.2e}"
        )));
    }
    Ok(())
}

fn cpu_gemv_bf16(x: &[f32], wf: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    for nn in 0..n {
        let mut acc = 0f32;
        for kk in 0..k {
            let bits = (wf[nn * k + kk].to_bits() >> 16) as u16 as u32;
            acc += x[kk] * f32::from_bits(bits << 16);
        }
        y[nn] = acc;
    }
    y
}

fn check_gemv_bf16(backend: &mut Backend, n: u32, k: u32) -> Result<()> {
    let wf: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.19 - 1.0).cos() * 0.7)
        .collect();
    let bytes: Vec<u8> = wf
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let w = Weight::plain(backend.tensor_bf16(&bytes, vec![n, k], "wb")?);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.31).cos()).collect();
    let xt = backend.tensor_f32(&x, vec![k], "x");
    let yt = backend.zero_tensor(&[n], "y");
    let mut enc = backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "t");
        backend.gemv(&mut pass, Binding::Full(&xt), &w, Binding::Full(&yt))?;
    }
    backend.submit(enc);
    let y = backend.read_f32(&yt.buf, 0, n as usize)?;
    let ref_y = cpu_gemv_bf16(&x, &wf, n as usize, k as usize);
    let mut max_err = 0f32;
    for (a, b) in y.iter().zip(ref_y.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("gemv-bf16 N={n} K={k}: max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model(format!(
            "gemv-bf16 mismatch: max_err {max_err:.2e}"
        )));
    }
    Ok(())
}

fn check_gemm(backend: &mut Backend, n: u32, k: u32, group: u32, m: u32) -> Result<()> {
    let data: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.137 - 3.0).sin() * 0.5)
        .collect();
    let (bytes, scales) = quantize(&data, n as usize, k as usize, group as usize);
    let w = Weight::quant(
        backend.tensor_i8(&bytes, vec![n, k], "w"),
        backend.tensor_f32(&scales, vec![n, k / group], "ws"),
        group,
    );
    let xs: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.71).cos() * 0.8)
        .collect();
    let xt = backend.tensor_f32(&xs, vec![m, k], "x");
    let yt = backend.zero_tensor(&[m, n], "y");
    let mut enc = backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "t");
        backend.gemm(&mut pass, Binding::Full(&xt), &w, Binding::Full(&yt), m)?;
    }
    backend.submit(enc);
    let y = backend.read_f32(&yt.buf, 0, (m * n) as usize)?;
    let mut max_err = 0f32;
    for row in 0..m {
        let ref_row = cpu_gemv(
            &xs[row as usize * k as usize..(row as usize + 1) * k as usize],
            &bytes,
            &scales,
            n as usize,
            k as usize,
            group as usize,
        );
        for (a, b) in y[row as usize * n as usize..(row as usize + 1) * n as usize]
            .iter()
            .zip(ref_row.iter())
        {
            max_err = max_err.max((a - b).abs());
        }
    }
    eprintln!("gemm N={n} K={k} G={group} M={m}: max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model(format!(
            "gemm mismatch: max_err {max_err:.2e}"
        )));
    }
    Ok(())
}

fn check_gemv_qkv(
    backend: &mut Backend,
    nq: u32,
    nk: u32,
    nv: u32,
    k: u32,
    group: u32,
) -> Result<()> {
    let mk = |n: u32| -> (flint_tensor::Weight, Vec<u8>, Vec<f32>) {
        let data: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.137 - 3.0).sin() * 0.5)
            .collect();
        let (bytes, scales) = quantize(&data, n as usize, k as usize, group as usize);
        (
            Weight::quant(
                backend.tensor_i8(&bytes, vec![n, k], "wq"),
                backend.tensor_f32(&scales, vec![n, k / group], "ws"),
                group,
            ),
            bytes,
            scales,
        )
    };
    let (wq, bq, sq) = mk(nq);
    let (wk, bk, sk) = mk(nk);
    let (wv, bv, sv) = mk(nv);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.71).cos() * 0.8).collect();
    let xt = backend.tensor_f32(&x, vec![k], "x");
    let yq_t = backend.zero_tensor(&[nq], "yq");
    let yk_t = backend.zero_tensor(&[nk], "yk");
    let yv_t = backend.zero_tensor(&[nv], "yv");
    let mut enc = backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "t");
        backend.gemv_qkv(
            &mut pass,
            Binding::Full(&xt),
            &wq,
            &wk,
            &wv,
            Binding::Full(&yq_t),
            Binding::Full(&yk_t),
            Binding::Full(&yv_t),
            nq,
            nk,
            nv,
            k,
        )?;
    }
    backend.submit(enc);
    let mut y = backend.read_f32(&yq_t.buf, 0, nq as usize)?;
    y.extend(backend.read_f32(&yk_t.buf, 0, nk as usize)?);
    y.extend(backend.read_f32(&yv_t.buf, 0, nv as usize)?);
    let rq = cpu_gemv(&x, &bq, &sq, nq as usize, k as usize, group as usize);
    let rk = cpu_gemv(&x, &bk, &sk, nk as usize, k as usize, group as usize);
    let rv = cpu_gemv(&x, &bv, &sv, nv as usize, k as usize, group as usize);
    let mut max_err = 0f32;
    for (i, v) in rq.iter().chain(rk.iter()).chain(rv.iter()).enumerate() {
        max_err = max_err.max((y[i] - v).abs());
    }
    eprintln!("gemv_qkv Nq={nq} Nk={nk} Nv={nv} K={k}: max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model(format!(
            "gemv_qkv mismatch: max_err {max_err:.2e}"
        )));
    }
    Ok(())
}

fn check_gemm_qkv(
    backend: &mut Backend,
    nq: u32,
    nk: u32,
    nv: u32,
    k: u32,
    group: u32,
    m: u32,
) -> Result<()> {
    let mk = |n: u32, tag: &str| -> (flint_tensor::Weight, Vec<u8>, Vec<f32>) {
        let data: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.137 - 3.0).sin() * 0.5)
            .collect();
        let (bytes, scales) = quantize(&data, n as usize, k as usize, group as usize);
        (
            Weight::quant(
                backend.tensor_i8(&bytes, vec![n, k], tag),
                backend.tensor_f32(&scales, vec![n, k / group], "ws"),
                group,
            ),
            bytes,
            scales,
        )
    };
    let (wq, bq, sq) = mk(nq, "wq");
    let (wk, bk, sk) = mk(nk, "wk");
    let (wv, bv, sv) = mk(nv, "wv");
    let xs: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.71).cos() * 0.8)
        .collect();
    let xt = backend.tensor_f32(&xs, vec![m, k], "x");
    let yt = backend.zero_tensor(&[m, nq + nk + nv], "y");
    let mut enc = backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "t");
        let ntot = nq + nk + nv;
        backend.gemm_strided(
            &mut pass,
            Binding::Full(&xt),
            &wq,
            Binding::Full(&yt),
            m,
            false,
            0,
            ntot,
        )?;
        backend.gemm_strided(
            &mut pass,
            Binding::Full(&xt),
            &wk,
            Binding::Full(&yt),
            m,
            false,
            nq,
            ntot,
        )?;
        backend.gemm_strided(
            &mut pass,
            Binding::Full(&xt),
            &wv,
            Binding::Full(&yt),
            m,
            false,
            nq + nk,
            ntot,
        )?;
    }
    backend.submit(enc);
    let y = backend.read_f32(&yt.buf, 0, (m * (nq + nk + nv)) as usize)?;
    let mut max_err = 0f32;
    for row in 0..m as usize {
        let xr = &xs[row * k as usize..(row + 1) * k as usize];
        for (off, (b, sc)) in [
            (0usize, (&bq, &sq)),
            (nq as usize, (&bk, &sk)),
            ((nq + nk) as usize, (&bv, &sv)),
        ] {
            let r = cpu_gemv(
                xr,
                b,
                sc,
                (if off == 0 {
                    nq
                } else if off == nq as usize {
                    nk
                } else {
                    nv
                }) as usize,
                k as usize,
                group as usize,
            );
            for (c, v) in r.iter().enumerate() {
                max_err = max_err.max((y[row * (nq + nk + nv) as usize + off + c] - v).abs());
            }
        }
    }
    eprintln!("gemm_qkv M={m} Nq={nq} Nk={nk} Nv={nv} K={k}: max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model(format!(
            "gemm_qkv mismatch: max_err {max_err:.2e}"
        )));
    }
    Ok(())
}

fn check_gemv_gateup(backend: &mut Backend, n: u32, k: u32, group: u32) -> Result<()> {
    let mk = |tag: &str| -> (flint_tensor::Weight, Vec<u8>, Vec<f32>) {
        let data: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.13 - 1.7).cos() * 0.4)
            .collect();
        let (bytes, scales) = quantize(&data, n as usize, k as usize, group as usize);
        (
            Weight::quant(
                backend.tensor_i8(&bytes, vec![n, k], tag),
                backend.tensor_f32(&scales, vec![n, k / group], "ws"),
                group,
            ),
            bytes,
            scales,
        )
    };
    let (wg, bg, sg) = mk("wg");
    let (wu, bu, su) = mk("wu");
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.71).cos() * 0.8).collect();
    let xt = backend.tensor_f32(&x, vec![k], "x");
    let yg_t = backend.zero_tensor(&[n], "yg");
    let yu_t = backend.zero_tensor(&[n], "yu");
    let mut enc = backend.encoder();
    {
        let mut pass = Pass::begin(&mut enc, "t");
        backend.gemv_gateup(
            &mut pass,
            Binding::Full(&xt),
            &wg,
            &wu,
            Binding::Full(&yg_t),
            Binding::Full(&yu_t),
            n,
            k,
        )?;
    }
    backend.submit(enc);
    let yg = backend.read_f32(&yg_t.buf, 0, n as usize)?;
    let yu = backend.read_f32(&yu_t.buf, 0, n as usize)?;
    let rg = cpu_gemv(&x, &bg, &sg, n as usize, k as usize, group as usize);
    let ru = cpu_gemv(&x, &bu, &su, n as usize, k as usize, group as usize);
    let mut max_err = 0f32;
    for (a, b) in yg.iter().zip(rg.iter()).chain(yu.iter().zip(ru.iter())) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("gemv_gateup N={n} K={k}: max_err={max_err:.2e}");
    if max_err > 1e-3 {
        return Err(flint_error::Error::Model(format!(
            "gemv_gateup mismatch: max_err {max_err:.2e}"
        )));
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut backend = Backend::new()?;
    eprintln!("adapter: {}", backend.adapter_name());

    check_gemv_bf16(&mut backend, 256, 2560)?;
    check_gemv_qkv(&mut backend, 4096, 1024, 1024, 2560, 128)?;
    check_gemv_gateup(&mut backend, 9728, 2560, 128)?;
    check_gemv_gateup(&mut backend, 640, 640, 64)?;
    check_gemm_qkv(&mut backend, 4096, 1024, 1024, 2560, 128, 16)?;
    check_gemm_qkv(&mut backend, 4096, 1024, 1024, 2560, 128, 92)?;
    check_gemv_qkv(&mut backend, 256, 64, 64, 640, 64)?;
    check_gemv_bf16(&mut backend, 151936, 2560)?;
    // gemm: 16-row (256 lanes) and 64-row (1024 lanes) row groups.
    for (n, k, g) in [
        (256u32, 2560u32, 128u32),
        (9728, 2560, 128),
        (2560, 9728, 128),
    ] {
        check_gemm(&mut backend, n, k, g, 16)?;
        check_gemm(&mut backend, n, k, g, 64)?;
        check_gemm(&mut backend, n, k, g, 92)?;
    }
    // i8 block-major: various shapes hitting odd/even segment lengths.
    for (n, k, g) in [
        (64u32, 256u32, 128u32),
        (256, 2560, 128),
        (128, 7680, 128),
        (7680, 2560, 128),
        (151936, 2560, 128),
        (64, 640, 64),
        (96, 1024, 32),
        (2560, 2560, 128),
        (4096, 4096, 128),
        (4096, 2560, 128),
        (1024, 2560, 128),
        (2560, 4096, 128),
        (2560, 9728, 128),
        (9728, 2560, 128),
    ] {
        check_gemv(&mut backend, n, k, g)?;
        check_gemv_tall(&mut backend, n, k, g)?;
    }
    eprintln!("all gemv/gemm checks passed");
    Ok(())
}
