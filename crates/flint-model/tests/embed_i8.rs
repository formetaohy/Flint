//! i8 embedding kernel conformance (large tables halve their footprint).

use flint_backend::{Backend, Binding, Pass};
use flint_model::ops;
use flint_tensor::{DType, Tensor, Weight};

#[test]
fn embed_i8_matches_cpu_dequant() {
    let mut backend = Backend::new().unwrap();
    let rows = 4u32;
    let dim = 64u32;
    let group = 32u32;
    let table: Vec<f32> = (0..rows * dim).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let groups = dim / group;
    let mut bytes = Vec::with_capacity((rows * dim) as usize);
    let mut scales = Vec::with_capacity((rows * groups) as usize);
    for r in 0..rows {
        for g in 0..groups {
            let block = &table[(r * dim + g * group) as usize..(r * dim + (g + 1) * group) as usize];
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales.push(scale);
            for v in block {
                bytes.push(((v / scale).round().clamp(-127.0, 127.0) as i8) as u8);
            }
        }
    }
    let w = Weight::quant(
        backend.tensor_i8(&bytes, vec![rows, dim], "t"),
        backend.tensor_f32(&scales, vec![rows, groups], "s"),
        group,
    );
    let ids_t = Tensor::new(backend.storage(16, "ids"), vec![4], DType::U32);
    backend.write_u32(&ids_t.buf, &[3, 1, 0, 2]);
    let y = backend.zero_tensor(&[16, dim], "y");
    let fallback = backend.tensor_f32(&[1.0], vec![1], "fb");
    {
        let mut enc = backend.encoder();
        {
            let mut pass = Pass::begin(&mut enc, "k");
            ops::embed(&mut backend, &mut pass, &ids_t, &w, &fallback, Binding::Full(&y), dim, 1.0)
                .unwrap();
        }
        backend.submit(enc);
    }
    let got = backend.read_f32(&y.buf, 0, (4 * dim) as usize).unwrap();
    let ids = [3u32, 1, 0, 2];
    for r in 0..rows {
        let src = ids[r as usize] as u32;
        let row = &table[(src * dim) as usize..((src + 1) * dim) as usize];
        let mut expect = vec![0f32; dim as usize];
        for g in 0..groups {
            let block = &row[g as usize * group as usize..(g as usize + 1) * group as usize];
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            for k in 0..group {
                let v = row[g as usize * group as usize + k as usize];
                expect[g as usize * group as usize + k as usize] =
                    (v / s).round().clamp(-127.0, 127.0) * s;
            }
        }
        let worst = (0..dim as usize)
            .map(|i| (got[r as usize * dim as usize + i] - expect[i]).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "row {r} (src {src}) got[0..4]={:?} expect[0..4]={:?} worst {worst}",
            &got[r as usize * dim as usize..r as usize * dim as usize + 4],
            &expect[..4],
        );
        assert!(worst < 1e-4, "row {r} worst diff {worst}");
    }
}
