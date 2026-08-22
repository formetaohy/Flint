use thuban_backend::{Backend, Binding, Commands};
use thuban_checkpoint::dequant::to_f32;
use thuban_model::ops;
use thuban_tensor::{DType, Quant, Tensor, Weight};

fn q8_0_blocks(rows: u32, dim: u32, table: &[f32]) -> Vec<u8> {
    let blocks = (rows * dim / 32) as usize;
    let mut raw = vec![0u8; blocks * 34];
    for b in 0..blocks {
        let r = b / (dim / 32) as usize;
        let blk = &table[r * dim as usize + (b % (dim / 32) as usize) * 32..][..32];
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let off = b * 34;
        raw[off..off + 2].copy_from_slice(&thuban_num::f32_to_f16(d).to_le_bytes());
        for (i, v) in blk.iter().enumerate() {
            raw[off + 2 + i] = (v / d).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
    raw
}

#[test]
fn embed_native_q8_0_matches_cpu_dequant() {
    let mut backend = Backend::new().unwrap();
    let rows = 4u32;
    let dim = 64u32;
    let table: Vec<f32> = (0..rows * dim)
        .map(|i| (i % 17) as f32 * 0.1 - 0.8)
        .collect();
    let raw = q8_0_blocks(rows, dim, &table);
    let padded = Quant::Q8_0.pad_blocks(&raw, (rows * dim) as usize).unwrap();
    let w = Weight::quantized(backend.tensor_quant(&padded, vec![rows, dim], Quant::Q8_0));
    let ids_t = Tensor::new(backend.storage(16), vec![4], DType::U32);
    backend.write_u32(&ids_t.buf, &[3, 1, 0, 2]);
    let y = backend.zero_tensor(&[16, dim], DType::F32);
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
                },
            )
            .unwrap();
        }
        backend.submit(&mut enc).unwrap();
    }
    let got = backend.read_f32(&y.buf, 0, (4 * dim) as usize).unwrap();
    let expect = to_f32(Quant::Q8_0, &raw, (rows * dim) as usize).unwrap();
    let ids = [3u32, 1, 0, 2];
    for r in 0..rows {
        let src = ids[r as usize];
        let worst = (0..dim as usize)
            .map(|i| {
                (got[r as usize * dim as usize + i]
                    - expect[src as usize * dim as usize + i])
                    .abs()
            })
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "row {r} worst diff {worst}");
    }
}
