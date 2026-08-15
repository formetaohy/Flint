use std::sync::{Mutex, MutexGuard};

use flint_architectures::llama;
use flint_backend::Backend;
use flint_checkpoint::{SafetensorEntry, open_checkpoint, write_tensors};
use flint_model::{LanguageModel, SeqChunk};
use serde_json::{Value, json};

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

struct Tensor {
    name: String,
    shape: Vec<u32>,
    data: Vec<f32>,
}

fn synth_llama() -> (std::path::PathBuf, Value) {
    let dir = std::env::temp_dir().join(format!("flint-batch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut tensors = Vec::new();
    let mut add = |name: &str, shape: &[u32], seed: u64| {
        tensors.push(Tensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: rng_vec(shape.iter().product::<u32>() as usize, seed),
        });
    };
    add("model.embed_tokens.weight", &[128, 128], 1);
    for l in 0..2u64 {
        let p = format!("model.layers.{l}");
        add(&format!("{p}.input_layernorm.weight"), &[128], 10 + l);
        add(&format!("{p}.post_attention_layernorm.weight"), &[128], 15 + l);
        add(&format!("{p}.self_attn.q_proj.weight"), &[256, 128], 20 + l);
        add(&format!("{p}.self_attn.k_proj.weight"), &[128, 128], 30 + l);
        add(&format!("{p}.self_attn.v_proj.weight"), &[128, 128], 40 + l);
        add(&format!("{p}.self_attn.o_proj.weight"), &[128, 256], 50 + l);
        add(&format!("{p}.mlp.gate_proj.weight"), &[256, 128], 60 + l);
        add(&format!("{p}.mlp.up_proj.weight"), &[256, 128], 70 + l);
        add(&format!("{p}.mlp.down_proj.weight"), &[128, 256], 80 + l);
    }
    add("model.norm.weight", &[128], 90);
    add("lm_head.weight", &[128, 128], 91);
    let config = json!({
        "model_type": "llama",
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 128,
        "rope_theta": 10000.0,
        "eos_token_id": [0],
        "tie_word_embeddings": false,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|t| t.data.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let entries: Vec<SafetensorEntry> = tensors
        .iter()
        .zip(bytes.iter())
        .map(|(t, b)| SafetensorEntry {
            name: &t.name,
            shape: &t.shape,
            bytes: b,
            bf16: false,
        })
        .collect();
    write_tensors(&dir.join("model.safetensors"), &entries).unwrap();
    (dir, config)
}

fn chunk<'a>(tokens: &'a [u32], slot: u32, logit_rows: &'a [u32]) -> SeqChunk<'a> {
    SeqChunk {
        tokens,
        slot,
        logit_rows,
        hidden_rows: &[],
    }
}

fn chunk_full<'a>(
    tokens: &'a [u32],
    slot: u32,
    logit_rows: &'a [u32],
    hidden_rows: &'a [u32],
) -> SeqChunk<'a> {
    SeqChunk {
        tokens,
        slot,
        logit_rows,
        hidden_rows,
    }
}

fn assert_same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: logit {i} differs");
    }
}

#[test]
fn self_speculator_drafts_from_the_captured_depth() {
    let _g = gpu();
    let (dir, config) = synth_llama();
    let mut backend = Backend::new().unwrap();
    let source = open_checkpoint(&dir).unwrap();
    let mut model = llama::load(source.as_ref(), &config, &[64], Some(1), &backend).unwrap();

    let tokens: Vec<u32> = (0..8).map(|i| (i * 3) % 127 + 1).collect();
    let last = [tokens.len() as u32 - 1];
    let out = model
        .forward(
            &mut backend,
            &[chunk_full(&tokens, 0, &last, &last)],
        )
        .unwrap();
    assert_eq!(out[0].logits.len(), 1, "logits for the last row");
    assert_eq!(out[0].hidden.len(), 1, "hidden captured at the spec depth");

    let pos_before = model.pos(0);
    model
        .speculator()
        .expect("spec_depth 1 enables the self-speculator")
        .snapshot(&backend, 0);
    let extra = [5u32, 6];
    let _ = model
        .forward(&mut backend, &[chunk(&extra, 0, &[1])])
        .unwrap();
    assert_eq!(model.pos(0), pos_before + 2);
    model
        .speculator()
        .expect("spec_depth 1 enables the self-speculator")
        .restore(&backend, 0);
    assert_eq!(
        model.pos(0),
        pos_before + 1,
        "restore rolls the rejected draft back by one position"
    );

    let draft = model
        .speculator()
        .expect("spec_depth 1 enables the self-speculator")
        .draft(&mut backend, 0, 7, &out[0].hidden[0])
        .unwrap();
    assert_eq!(draft.len(), 128, "draft logits span the vocab");
    assert!(draft.iter().all(|v| v.is_finite()));
    assert!(
        draft != out[0].logits[0],
        "the shallow head must differ from the full-depth logits"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn batch_slots_match_solo_and_isolate_sequences() {
    let _g = gpu();
    let (dir, config) = synth_llama();
    let backend = Backend::new().unwrap();
    let source = open_checkpoint(&dir).unwrap();
    let mut model = llama::load(source.as_ref(), &config, &[64, 64], None, &backend).unwrap();
    let mut backend = backend;

    let seq_a: Vec<u32> = (0..16).map(|i| (i * 3) % 127 + 1).collect();
    let last_a = [seq_a.len() as u32 - 1];
    let seq_b: Vec<u32> = (0..16).map(|i| (i * 5) % 127 + 1).collect();
    let last_b = [seq_b.len() as u32 - 1];

    let batch = model
        .forward(
            &mut backend,
            &[chunk(&seq_a, 0, &last_a), chunk(&seq_b, 1, &last_b)],
        )
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].logits.len(), 1);
    assert_eq!(batch[1].logits.len(), 1);

    model.reset(&backend, 0).unwrap();
    model.reset(&backend, 1).unwrap();
    let solo_a = model
        .forward(&mut backend, &[chunk(&seq_a, 0, &last_a)])
        .unwrap();
    let solo_b = model
        .forward(&mut backend, &[chunk(&seq_b, 1, &last_b)])
        .unwrap();

    assert_same(&batch[0].logits[0], &solo_a[0].logits[0], "slot 0 vs solo");
    assert_same(&batch[1].logits[0], &solo_b[0].logits[0], "slot 1 vs solo");
    assert_ne!(
        batch[0].logits[0], batch[1].logits[0],
        "different sequences must yield different logits"
    );

    model.reset(&backend, 0).unwrap();
    model.reset(&backend, 1).unwrap();
    let mirror = model
        .forward(
            &mut backend,
            &[chunk(&seq_a, 0, &last_a), chunk(&seq_a, 1, &last_a)],
        )
        .unwrap();
    assert_same(&mirror[0].logits[0], &mirror[1].logits[0], "same tokens across slots");

    model.reset(&backend, 0).unwrap();
    model.reset(&backend, 1).unwrap();
    let step1 = model
        .forward(
            &mut backend,
            &[chunk(&seq_a, 0, &[]), chunk(&seq_b, 1, &[])],
        )
        .unwrap();
    assert!(step1.iter().all(|o| o.logits.is_empty()));
    let tail_a = [1u32, 2, 3, 4];
    let tail_b = [9u32, 8, 7];
    let last_ta = [tail_a.len() as u32 - 1];
    let last_tb = [tail_b.len() as u32 - 1];
    let step2 = model
        .forward(
            &mut backend,
            &[chunk(&tail_a, 0, &last_ta), chunk(&tail_b, 1, &last_tb)],
        )
        .unwrap();

    let mut ctrl = llama::load(source.as_ref(), &config, &[64, 64], None, &backend).unwrap();
    ctrl.reset(&backend, 0).unwrap();
    ctrl.reset(&backend, 1).unwrap();
    let _ = ctrl
        .forward(
            &mut backend,
            &[chunk(&seq_a, 0, &[]), chunk(&seq_b, 1, &[])],
        )
        .unwrap();
    let ctrl2 = ctrl
        .forward(
            &mut backend,
            &[chunk(&tail_a, 0, &last_ta), chunk(&tail_b, 1, &last_tb)],
        )
        .unwrap();

    assert_same(&step2[0].logits[0], &ctrl2[0].logits[0], "slot 0 continuation");
    assert_same(&step2[1].logits[0], &ctrl2[1].logits[0], "slot 1 continuation");
    assert_ne!(step2[0].logits[0], step2[1].logits[0]);

    std::fs::remove_dir_all(&dir).ok();
}
