use std::sync::{Mutex, MutexGuard};

use flint_architectures::bert::Bert;
use flint_backend::Backend;
use flint_checkpoint::{SafetensorEntry, open_checkpoint, write_tensors};
use flint_model::TextEmbedder;
use serde_json::json;

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn rng_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) * 0.02
        })
        .collect()
}

fn synth_bert() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("flint-bert-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut tensors: Vec<(String, Vec<u32>, Vec<f32>)> = Vec::new();
    let mut add = |name: &str, shape: &[u32], seed: u64| {
        tensors.push((
            name.to_string(),
            shape.to_vec(),
            rng_vec(shape.iter().product::<u32>() as usize, seed),
        ));
    };
    add("embeddings.word_embeddings.weight", &[64, 128], 1);
    add("embeddings.position_embeddings.weight", &[512, 128], 2);
    add("embeddings.token_type_embeddings.weight", &[2, 128], 3);
    add("embeddings.LayerNorm.weight", &[128], 4);
    add("embeddings.LayerNorm.bias", &[128], 5);
    for l in 0..2u64 {
        let p = format!("encoder.layer.{l}");
        for (part, seed) in [("query", 100), ("key", 101), ("value", 102)] {
            add(&format!("{p}.attention.self.{part}.weight"), &[128, 128], seed + l);
            add(&format!("{p}.attention.self.{part}.bias"), &[128], seed + 10 + l);
        }
        add(&format!("{p}.attention.output.dense.weight"), &[128, 128], 103 + l);
        add(&format!("{p}.attention.output.dense.bias"), &[128], 113 + l);
        add(
            &format!("{p}.attention.output.LayerNorm.weight"),
            &[128],
            104 + l,
        );
        add(
            &format!("{p}.attention.output.LayerNorm.bias"),
            &[128],
            114 + l,
        );
        add(&format!("{p}.intermediate.dense.weight"), &[256, 128], 105 + l);
        add(&format!("{p}.intermediate.dense.bias"), &[256], 115 + l);
        add(&format!("{p}.output.dense.weight"), &[128, 256], 106 + l);
        add(&format!("{p}.output.dense.bias"), &[128], 116 + l);
        add(&format!("{p}.output.LayerNorm.weight"), &[128], 107 + l);
        add(&format!("{p}.output.LayerNorm.bias"), &[128], 117 + l);
    }
    let config = json!({
        "model_type": "bert",
        "hidden_size": 128,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, d)| d.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let entries: Vec<SafetensorEntry> = tensors
        .iter()
        .zip(bytes.iter())
        .map(|((name, shape, _), b)| SafetensorEntry {
            name,
            shape,
            bytes: b,
            bf16: false,
        })
        .collect();
    write_tensors(&dir.join("model.safetensors"), &entries).unwrap();
    dir
}

#[test]
fn bert_embed_is_deterministic_unit_norm_and_content_sensitive() {
    let _g = gpu();
    let dir = synth_bert();
    let mut backend = Backend::new().unwrap();
    let source = open_checkpoint(&dir).unwrap();
    let mut model = Bert::load(source.as_ref(), &backend).unwrap();

    let a = model.embed(&mut backend, &[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(a.len(), 128);
    let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "sentence embeddings are L2 normalized, got {norm}"
    );
    assert!(a.iter().all(|v| v.is_finite()));

    let b = model.embed(&mut backend, &[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(
        a, b,
        "identical inputs must yield identical embeddings"
    );

    let c = model.embed(&mut backend, &[5, 4, 3, 2, 1]).unwrap();
    assert_ne!(
        a, c,
        "different inputs must yield different embeddings"
    );

    let short = model.embed(&mut backend, &[1]).unwrap();
    assert_eq!(short.len(), 128);
    assert!(short.iter().all(|v| v.is_finite()));

    std::fs::remove_dir_all(&dir).ok();
}
