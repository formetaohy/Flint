use flint_backend::{Backend, Binding, Commands};
use flint_model::ops;
use flint_model::quant::quantize;
use flint_tensor::{DType, Tensor, Weight};

#[test]
fn embed_i8_matches_cpu_dequant() {
    let mut backend = Backend::new().unwrap();
    let rows = 4u32;
    let dim = 64u32;
    let group = 32u32;
    let table: Vec<f32> = (0..rows * dim)
        .map(|i| (i % 17) as f32 * 0.1 - 0.8)
        .collect();
    let (bytes, scales) = quantize(&table, rows as usize, dim as usize, group as usize);
    let w = Weight::quant(
        backend.tensor_i8(&bytes, vec![rows, dim]),
        backend.tensor_f32(&scales, vec![rows, dim / group]),
        group,
    );
    let ids_t = Tensor::new(backend.storage(16), vec![4], DType::U32);
    backend.write_u32(&ids_t.buf, &[3, 1, 0, 2]);
    let y = backend.zero_tensor(&[16, dim]);
    {
        let mut enc = backend.encoder().unwrap();
        {
            let mut commands = Commands::begin(&mut enc);
            ops::embed(
                &mut backend,
                &mut commands,
                &ids_t,
                &w,
                Binding::Full(&y),
                &ops::EmbedSpec {
                    rows: 4,
                    dim,
                    scale: 1.0,
                    split: 0,
                },
            )
            .unwrap();
        }
        backend.submit(&mut enc).unwrap();
    }
    let got = backend
        .read_f32(&y.buf, 0, (4 * dim) as usize)
        .unwrap();
    let ids = [3u32, 1, 0, 2];
    for r in 0..rows {
        let src = ids[r as usize];
        let mut expect = vec![0f32; dim as usize];
        for kb in 0..dim / 16 {
            for i in 0..16 {
                let d = (kb * 16 + i) as usize;
                let q = bytes[(kb as usize * rows as usize + src as usize) * 16 + i as usize] as i8;
                expect[d] = q as f32 * scales[kb as usize / 2 * rows as usize + src as usize];
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
