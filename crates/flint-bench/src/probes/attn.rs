use flint_backend::Backend;
use flint_error::Result;
use flint_tensor::DType;

pub(super) fn attn_probe() -> Result<()> {
    use flint_backend::{Binding, Commands};

    let env = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (m, nq, nkv, hd, max_seq, pos, window) = (
        env("PROBE_M", 128),
        env("PROBE_Q_HEADS", 32),
        env("PROBE_KV_HEADS", 8),
        env("PROBE_HD", 128),
        env("PROBE_MAX_SEQ", 2048),
        env("PROBE_POS", 1024),
        env("PROBE_WINDOW", 0),
    );
    let mut backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
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
    let y = backend.zero_tensor(&[m, nq, hd], DType::F32);
    let args = flint_model::rows::row_meta(&backend);
    let mut meta = vec![0u32; 8 * m as usize];
    for i in 0..m as usize {
        meta[8 * i] = pos + i as u32;
        meta[8 * i + 1] = 0;
    }
    backend.write_u32(&args.buf, &meta);
    let pages = max_seq.div_ceil(flint_kernel::PAGE_LEN);
    let table: Vec<u32> = (0..pages).collect();
    let block_table = flint_tensor::Tensor::new(
        backend.storage(pages as u64 * 4),
        vec![pages],
        flint_tensor::DType::U32,
    );
    backend.write_u32(&block_table.buf, &table);
    let run_flash = |backend: &mut Backend| -> Result<()> {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            backend.dispatch(
                &mut commands,
                flint_kernel::shader::ATTN,
                &[
                    ("M", m as f64),
                    ("N_HEADS", nq as f64),
                    ("HEAD_DIM", hd as f64),
                    ("POOL_LEN", max_seq as f64),
                    ("SCALE", 1.0 / (hd as f64).sqrt()),
                    ("WINDOW", window as f64),
                    ("NQ_PER_KV", (nq / nkv) as f64),
                    ("SEQ", 0.0),
                    ("CAUSAL", 1.0),
                    ("MAX_PAGES", pages as f64),
                ],
                &[
                    Binding::Full(&qb),
                    Binding::Full(&kb),
                    Binding::Full(&vb),
                    Binding::Full(&y),
                    Binding::Full(&args),
                    Binding::Full(&block_table),
                ],
                [m.div_ceil(flint_kernel::ATTN_BR), nq, 1],
            )?;
        }
        backend.submit(&mut enc).unwrap();
        Ok(())
    };
    for _ in 0..10 {
        run_flash(&mut backend)?;
    }
    let t0 = std::time::Instant::now();
    let iters = 50;
    for _ in 0..iters {
        run_flash(&mut backend)?;
    }
    let _ = backend.read_f32(&y.buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let ms = secs / iters as f64 * 1e3;
    eprintln!(
        "[probe] attn identity table: {ms:.2} ms/call (M={m} kv={} hd={hd} nq={nq} nkv={nkv})",
        pos + m
    );

    let mut shuffled = (0..pages).collect::<Vec<u32>>();
    for i in (1..pages as usize).rev() {
        let j = (i * 73 + 11) % (i + 1);
        shuffled.swap(i, j);
    }
    backend.write_u32(&block_table.buf, &shuffled);
    for _ in 0..10 {
        run_flash(&mut backend)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        run_flash(&mut backend)?;
    }
    let _ = backend.read_f32(&y.buf, 0, 1)?;
    let secs = t0.elapsed().as_secs_f64();
    let ms_sh = secs / iters as f64 * 1e3;
    eprintln!(
        "[probe] attn shuffled table: {ms_sh:.2} ms/call ({:.1}% vs identity)",
        ms_sh / ms * 100.0
    );
    Ok(())
}
