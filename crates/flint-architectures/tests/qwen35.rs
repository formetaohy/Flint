use std::sync::{Mutex, MutexGuard};

use flint_architectures::qwen35::{Qwen35, gguf_key};
use flint_backend::Backend;
use flint_checkpoint::{GgufWriter, open_checkpoint};
use flint_model::{LanguageModel, SeqChunk};
use flint_model::pool::ArenaSpec;

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

const HIDDEN: u32 = 64;
const INTER: u32 = 128;
const VOCAB: u32 = 48;
const Q_HEADS: u32 = 4;
const KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 64;
const LIN_KEY_HEADS: u32 = 8;
const LIN_VAL_HEADS: u32 = 16;
const LIN_KEY_DIM: u32 = 32;
const LIN_VAL_DIM: u32 = 32;
const CONV_DIM: u32 = LIN_KEY_HEADS * LIN_KEY_DIM * 2 + LIN_VAL_HEADS * LIN_VAL_DIM;

fn synth_qwen35(layers: u32, interval: u32) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("flint-qwen35-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", "qwen35");
    w.kv_u32("qwen35.embedding_length", HIDDEN);
    w.kv_u32("qwen35.feed_forward_length", INTER);
    w.kv_u32("qwen35.block_count", layers);
    w.kv_u32("qwen35.attention.head_count", Q_HEADS);
    w.kv_u32("qwen35.attention.head_count_kv", KV_HEADS);
    w.kv_u32("qwen35.attention.key_length", HEAD_DIM);
    w.kv_u32("qwen35.attention.value_length", HEAD_DIM);
    w.kv_u32("qwen35.rope.dimension_count", 32);
    w.kv_f32("qwen35.rope.freq_base", 10000.0);
    w.kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    w.kv_u32("qwen35.ssm.conv_kernel", 4);
    w.kv_u32("qwen35.ssm.inner_size", LIN_VAL_HEADS * LIN_VAL_DIM);
    w.kv_u32("qwen35.ssm.state_size", LIN_KEY_DIM);
    w.kv_u32("qwen35.ssm.group_count", LIN_KEY_HEADS);
    w.kv_u32("qwen35.ssm.time_step_rank", LIN_VAL_HEADS);
    w.kv_u32("qwen35.full_attention_interval", interval);
    w.kv_u32("tokenizer.ggml.eos_token_id", 1);

    fn add(w: &mut GgufWriter, name: &str, shape: &[u32], seed: u64) {
        w.tensor_f32(name, shape, &rng_vec(shape.iter().product::<u32>() as usize, seed));
    }
    fn add_bf16(w: &mut GgufWriter, name: &str, shape: &[u32], seed: u64) {
        w.tensor_bf16(name, shape, &rng_vec(shape.iter().product::<u32>() as usize, seed));
    }
    add_bf16(&mut w, "token_embd.weight", &[VOCAB, HIDDEN], 1);
    add(&mut w, "output_norm.weight", &[HIDDEN], 2);
    for l in 0..layers {
        let p = format!("blk.{l}");
        add(&mut w, &format!("{p}.attn_norm.weight"), &[HIDDEN], 10 + l as u64);
        add(&mut w, &format!("{p}.post_attention_norm.weight"), &[HIDDEN], 11 + l as u64);
        add_bf16(&mut w, &format!("{p}.ffn_gate.weight"), &[INTER, HIDDEN], 20 + l as u64);
        add_bf16(&mut w, &format!("{p}.ffn_up.weight"), &[INTER, HIDDEN], 21 + l as u64);
        add_bf16(&mut w, &format!("{p}.ffn_down.weight"), &[HIDDEN, INTER], 22 + l as u64);
        if (l + 1) % interval == 0 {
            add_bf16(&mut w, &format!("{p}.attn_q.weight"), &[Q_HEADS * HEAD_DIM * 2, HIDDEN], 30 + l as u64);
            add_bf16(&mut w, &format!("{p}.attn_k.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN], 31 + l as u64);
            add_bf16(&mut w, &format!("{p}.attn_v.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN], 32 + l as u64);
            add_bf16(&mut w, &format!("{p}.attn_output.weight"), &[HIDDEN, Q_HEADS * HEAD_DIM], 33 + l as u64);
            add(&mut w, &format!("{p}.attn_q_norm.weight"), &[HEAD_DIM], 34 + l as u64);
            add(&mut w, &format!("{p}.attn_k_norm.weight"), &[HEAD_DIM], 35 + l as u64);
        } else {
            add_bf16(&mut w, &format!("{p}.attn_qkv.weight"), &[CONV_DIM, HIDDEN], 40 + l as u64);
            add_bf16(&mut w, &format!("{p}.attn_gate.weight"), &[LIN_VAL_HEADS * LIN_VAL_DIM, HIDDEN], 41 + l as u64);
            add_bf16(&mut w, &format!("{p}.ssm_beta.weight"), &[LIN_VAL_HEADS, HIDDEN], 42 + l as u64);
            add_bf16(&mut w, &format!("{p}.ssm_alpha.weight"), &[LIN_VAL_HEADS, HIDDEN], 43 + l as u64);
            add(&mut w, &format!("{p}.ssm_conv1d.weight"), &[4, CONV_DIM], 44 + l as u64);
            add(&mut w, &format!("{p}.ssm_a"), &[LIN_VAL_HEADS], 45 + l as u64);
            add(&mut w, &format!("{p}.ssm_dt.bias"), &[LIN_VAL_HEADS], 46 + l as u64);
            add(&mut w, &format!("{p}.ssm_norm.weight"), &[LIN_VAL_DIM], 47 + l as u64);
            add_bf16(&mut w, &format!("{p}.ssm_out.weight"), &[HIDDEN, LIN_VAL_HEADS * LIN_VAL_DIM], 48 + l as u64);
        }
    }
    std::fs::write(dir.join("model.gguf"), w.finish()).unwrap();
    dir
}

fn arena(seqs: u32) -> ArenaSpec {
    ArenaSpec {
        seq_lens: vec![64; seqs as usize],
        pages: None,
    }
}

fn chunk<'a>(tokens: &'a [u32], seq: u32, logit_rows: &'a [u32]) -> SeqChunk<'a> {
    SeqChunk {
        tokens,
        seq,
        logit_rows,
        hidden_rows: &[],
    }
}

fn assert_same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: logit {i} differs");
    }
}

#[test]
fn keymap_covers_the_qwen35_family() {
    let cases: &[(&str, Option<&str>)] = &[
        (
            "blk.3.attn_qkv.weight",
            Some("layers.3.linear_attn.in_proj_qkv.weight"),
        ),
        (
            "blk.3.attn_gate.weight",
            Some("layers.3.linear_attn.in_proj_z.weight"),
        ),
        (
            "blk.3.ssm_beta.weight",
            Some("layers.3.linear_attn.in_proj_b.weight"),
        ),
        (
            "blk.3.ssm_alpha.weight",
            Some("layers.3.linear_attn.in_proj_a.weight"),
        ),
        (
            "blk.3.ssm_conv1d.weight",
            Some("layers.3.linear_attn.conv1d.weight"),
        ),
        ("blk.3.ssm_a", Some("layers.3.linear_attn.a_log")),
        (
            "blk.3.ssm_dt.bias",
            Some("layers.3.linear_attn.dt_bias"),
        ),
        (
            "blk.3.ssm_norm.weight",
            Some("layers.3.linear_attn.norm.weight"),
        ),
        (
            "blk.3.ssm_out.weight",
            Some("layers.3.linear_attn.out_proj.weight"),
        ),
        (
            "blk.3.attn_q.weight",
            Some("layers.3.self_attn.qg_proj.weight"),
        ),
        (
            "blk.3.attn_norm.weight",
            Some("layers.3.input_layernorm.weight"),
        ),
        (
            "blk.3.post_attention_norm.weight",
            Some("layers.3.post_attention_norm.weight"),
        ),
        (
            "blk.3.ffn_gate.weight",
            Some("layers.3.mlp.gate_proj.weight"),
        ),
        ("token_embd.weight", Some("embed_tokens.weight")),
        ("output_norm.weight", Some("norm.weight")),
        ("blk.24.nextn.eh_proj.weight", None),
    ];
    for &(name, want) in cases {
        assert_eq!(gguf_key(name).as_deref(), want, "{name}");
    }
    assert_eq!(
        gguf_key("blk.24.nextn.eh_proj.weight"),
        None,
        "mtp tensors are skipped"
    );
}

#[test]
fn forward_produces_finite_logits_and_isolates_sequences() {
    let _g = gpu();
    let dir = synth_qwen35(4, 4);
    let mut backend = Backend::new().unwrap();
    let source = open_checkpoint(&dir).unwrap();
    let mut model = Qwen35::load(
        &source,
        &serde_json::json!({
            "hidden_size": HIDDEN,
            "num_attention_heads": Q_HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTER,
            "num_hidden_layers": 4,
            "vocab_size": VOCAB,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "rotary_dim": 32,
            "rope_theta": 10000.0,
            "linear_num_key_heads": LIN_KEY_HEADS,
            "linear_num_value_heads": LIN_VAL_HEADS,
            "linear_key_head_dim": LIN_KEY_DIM,
            "linear_value_head_dim": LIN_VAL_DIM,
            "linear_conv_kernel_dim": 4,
            "eos_token_id": [1],
            "tie_word_embeddings": true,
        }),
        &arena(2),
        &backend,
    )
    .unwrap();

    let seq_a: Vec<u32> = (0..16).map(|i| (i * 3) % (VOCAB - 1) + 1).collect();
    let last_a = [seq_a.len() as u32 - 1];
    let seq_b: Vec<u32> = (0..16).map(|i| (i * 5) % (VOCAB - 1) + 1).collect();
    let last_b = [seq_b.len() as u32 - 1];

    model.alloc_pages(&backend, 0, seq_a.len() as u32).unwrap();
    model.alloc_pages(&backend, 1, seq_b.len() as u32).unwrap();
    let out = model
        .forward(
            &mut backend,
            &[chunk(&seq_a, 0, &last_a), chunk(&seq_b, 1, &last_b)],
        )
        .unwrap();
    assert_eq!(out.len(), 2);
    for o in &out {
        assert_eq!(o.logits.len(), 1);
        assert!(o.logits[0].iter().all(|v| v.is_finite()), "finite logits");
    }
    assert_ne!(out[0].logits[0], out[1].logits[0]);

    model.reset(&backend, 0).unwrap();
    model.reset(&backend, 1).unwrap();
    model.alloc_pages(&backend, 0, seq_a.len() as u32).unwrap();
    model.alloc_pages(&backend, 1, seq_a.len() as u32).unwrap();
    let mirror = model
        .forward(
            &mut backend,
            &[chunk(&seq_a, 0, &last_a), chunk(&seq_a, 1, &last_a)],
        )
        .unwrap();
    assert_same(
        &mirror[0].logits[0],
        &mirror[1].logits[0],
        "same tokens across sequences",
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn chunked_forward_matches_single_pass() {
    let _g = gpu();
    let dir = synth_qwen35(4, 4);
    let mut backend = Backend::new().unwrap();
    let source = open_checkpoint(&dir).unwrap();
    let config = serde_json::json!({
        "hidden_size": HIDDEN,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTER,
        "num_hidden_layers": 4,
        "vocab_size": VOCAB,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "rotary_dim": 32,
        "rope_theta": 10000.0,
        "linear_num_key_heads": LIN_KEY_HEADS,
        "linear_num_value_heads": LIN_VAL_HEADS,
        "linear_key_head_dim": LIN_KEY_DIM,
        "linear_value_head_dim": LIN_VAL_DIM,
        "linear_conv_kernel_dim": 4,
        "eos_token_id": [1],
        "tie_word_embeddings": true,
    });

    let tokens: Vec<u32> = (0..24).map(|i| (i * 7) % (VOCAB - 1) + 1).collect();
    let last = [tokens.len() as u32 - 1];

    let mut plain = Qwen35::load(&source, &config, &arena(1), &backend).unwrap();
    plain.alloc_pages(&backend, 0, tokens.len() as u32).unwrap();
    let want = plain
        .forward(&mut backend, &[chunk(&tokens, 0, &last)])
        .unwrap();

    let mut chunked = Qwen35::load(&source, &config, &arena(1), &backend).unwrap();
    chunked.alloc_pages(&backend, 0, 12).unwrap();
    let _ = chunked
        .forward(&mut backend, &[chunk(&tokens[..12], 0, &[])])
        .unwrap();
    chunked.alloc_pages(&backend, 0, 12).unwrap();
    let got = chunked
        .forward(&mut backend, &[chunk(&tokens[12..], 0, &[11])])
        .unwrap();

    assert_same(
        &want[0].logits[0],
        &got[0].logits[0],
        "chunked vs single pass",
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn decode_steps_extend_the_recurrent_state() {
    let _g = gpu();
    let dir = synth_qwen35(4, 4);
    let mut backend = Backend::new().unwrap();
    let source = open_checkpoint(&dir).unwrap();
    let config = serde_json::json!({
        "hidden_size": HIDDEN,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTER,
        "num_hidden_layers": 4,
        "vocab_size": VOCAB,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "rotary_dim": 32,
        "rope_theta": 10000.0,
        "linear_num_key_heads": LIN_KEY_HEADS,
        "linear_num_value_heads": LIN_VAL_HEADS,
        "linear_key_head_dim": LIN_KEY_DIM,
        "linear_value_head_dim": LIN_VAL_DIM,
        "linear_conv_kernel_dim": 4,
        "eos_token_id": [1],
        "tie_word_embeddings": true,
    });

    let tokens: Vec<u32> = (0..8).map(|i| (i * 11) % (VOCAB - 1) + 1).collect();

    let mut one_shot = Qwen35::load(&source, &config, &arena(1), &backend).unwrap();
    one_shot.alloc_pages(&backend, 0, 10).unwrap();
    let want = one_shot
        .forward(&mut backend, &[chunk(&tokens, 0, &[7])])
        .unwrap();

    let mut stepped = Qwen35::load(&source, &config, &arena(1), &backend).unwrap();
    let mut got = None;
    for i in 0..8 {
        stepped.alloc_pages(&backend, 0, 1).unwrap();
        let logit: &[u32] = if i == 7 { &[0] } else { &[] };
        let out = stepped
            .forward(&mut backend, &[chunk(&tokens[i..i + 1], 0, logit)])
            .unwrap();
        if i == 7 {
            got = Some(out);
        }
    }
    let got = got.unwrap();
    assert_eq!(got[0].logits[0].len(), want[0].logits[0].len());
    for (i, (g, w)) in got[0].logits[0].iter().zip(&want[0].logits[0]).enumerate() {
        let tol = w.abs() * 1e-4 + 1e-6;
        assert!(
            (g - w).abs() <= tol,
            "stepped decode vs single pass: logit {i} differs ({g} vs {w})"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
