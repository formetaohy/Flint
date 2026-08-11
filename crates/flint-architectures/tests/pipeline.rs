use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use flint_backend::Backend;
use flint_generate::{Engine, Sampler};
use flint_model::LanguageModel;

use flint_architectures::ChatModel;

mod common;
use common::toy::ToySpec;

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn toy_dir(test: &str, spec: ToySpec) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("flint-toy-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    spec.write(&dir).expect("toy checkpoint");
    dir
}

fn load(test: &str, spec: ToySpec, max_seq: u32) -> (Backend, ChatModel) {
    let dir = toy_dir(test, spec);
    let backend = Backend::new().unwrap();
    let cm = flint_architectures::load(&dir, max_seq, &backend).unwrap();
    (backend, cm)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn sequential_logits(
    model: &mut Box<dyn LanguageModel>,
    backend: &mut Backend,
    prompt: &[u32],
) -> Vec<f32> {
    let mut logits = Vec::new();
    for &t in prompt {
        logits = model
            .forward(backend, &[t], &[0], &[])
            .unwrap()
            .logits
            .pop()
            .unwrap();
    }
    logits
}

fn chunked_logits(
    model: &mut Box<dyn LanguageModel>,
    backend: &mut Backend,
    prompt: &[u32],
) -> Vec<f32> {
    let mut logits = Vec::new();
    let mut done = 0usize;
    while done < prompt.len() {
        let end = (done + 16).min(prompt.len());
        let chunk = &prompt[done..end];
        let last = end == prompt.len();
        let row = [chunk.len() as u32 - 1];
        let rows: &[u32] = if last { &row } else { &[] };
        let out = model.forward(backend, chunk, rows, &[]).unwrap();
        if let Some(l) = out.logits.into_iter().next() {
            logits = l;
        }
        done = end;
    }
    logits
}

fn prefill_equivalence(test: &str, spec: ToySpec) {
    let _g = gpu();
    let (mut backend, mut cm) = load(test, spec, 256);
    let prompt: Vec<u32> = (10..50).collect();

    let seq = sequential_logits(&mut cm.model, &mut backend, &prompt);
    cm.model.reset(&backend);
    let chunked = chunked_logits(&mut cm.model, &mut backend, &prompt);

    assert_eq!(chunked.len(), seq.len(), "vocab mismatch");
    let cos = cosine(&chunked, &seq);
    assert!(cos > 0.98, "logit distribution diverges (cosine {cos})");
}

fn long_sequence_stability(test: &str, spec: ToySpec) {
    let _g = gpu();
    let (mut backend, mut cm) = load(test, spec, 1024);
    let prompt: Vec<u32> = (10..138).collect();

    let seq = sequential_logits(&mut cm.model, &mut backend, &prompt);
    cm.model.reset(&backend);
    let chunked = chunked_logits(&mut cm.model, &mut backend, &prompt);

    assert_eq!(chunked.len(), seq.len(), "vocab mismatch");
    let cos = cosine(&chunked, &seq);
    assert!(cos > 0.98, "logit distribution diverges (cosine {cos})");

    let out = cm.model.forward(&mut backend, &[7], &[0], &[]).unwrap();
    let l = &out.logits[0];
    assert!(
        l.iter().all(|v| v.is_finite()),
        "non-finite logits after long decode"
    );
    let peak = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
    assert!(
        peak < 1e4,
        "logit magnitude exploded after long decode (peak {peak})"
    );
}

#[test]
fn long_sequence_llama() {
    long_sequence_stability("long_llama", ToySpec::Llama);
}

#[test]
fn prefill_llama_gguf() {
    prefill_equivalence("prefill_llama", ToySpec::Llama);
}

#[test]
fn prefill_llama_qkv_bias() {
    prefill_equivalence("prefill_llama_qkv_bias", ToySpec::LlamaQkvBias);
}

#[test]
fn prefill_llama_qk_norm() {
    prefill_equivalence("prefill_llama_qk_norm", ToySpec::LlamaQkNorm);
}

#[test]
fn prefill_gemma_sliding_window() {
    prefill_equivalence("prefill_gemma", ToySpec::Gemma);
}

#[test]
fn prefill_qwen35_hybrid() {
    prefill_equivalence("prefill_qwen35", ToySpec::Qwen35);
}

#[test]
fn prefill_qwen35_split_key_value_heads() {
    prefill_equivalence("prefill_qwen35_split", ToySpec::Qwen35Split);
}

#[test]
fn prefill_qwen35_untied_embeddings() {
    prefill_equivalence("prefill_qwen35_untied", ToySpec::Qwen35Untied);
}

#[test]
fn prefill_phi_partial_rotary() {
    prefill_equivalence("prefill_phi", ToySpec::Phi);
}

#[test]
fn prefill_phi_dense_layernorm() {
    prefill_equivalence("prefill_phi_layernorm", ToySpec::PhiLayerNorm);
}

#[test]
fn prefill_phimoe_sparsemixer() {
    prefill_equivalence("prefill_phimoe", ToySpec::PhiMoe);
}

#[test]
fn prefill_gemma4_kv_sharing_ple() {
    prefill_equivalence("prefill_gemma4", ToySpec::Gemma4);
}

fn generate(test: &str, spec: ToySpec, max_tokens: usize, speculate: bool) -> Vec<u32> {
    let _g = gpu();
    let (backend, cm) = load(test, spec, 256);
    let prompt = cm.chat.render("", &[], "count from one to five");
    let mut engine = Engine::new(
        backend,
        cm.model,
        cm.tokenizer,
        Sampler::greedy(7),
        cm.stop,
        speculate,
    );
    let mut tokens = Vec::new();
    for piece in engine.stream(&prompt, max_tokens).unwrap() {
        tokens.push(piece.unwrap().token);
    }
    tokens
}

#[test]
fn generation_llama_greedy_terminates() {
    let tokens = generate("gen_llama", ToySpec::Llama, 24, false);
    assert!(!tokens.is_empty(), "generation produced nothing");
}

#[test]
fn generation_qwen35_greedy_terminates() {
    let tokens = generate("gen_qwen35", ToySpec::Qwen35, 24, false);
    assert!(!tokens.is_empty(), "generation produced nothing");
}

#[test]
fn generation_phimoe_greedy_terminates() {
    let tokens = generate("gen_phimoe", ToySpec::PhiMoe, 24, false);
    assert!(!tokens.is_empty(), "generation produced nothing");
}

#[test]
fn generation_gemma4_greedy_terminates() {
    let tokens = generate("gen_gemma4", ToySpec::Gemma4, 24, false);
    assert!(!tokens.is_empty(), "generation produced nothing");
}

#[test]
fn generation_qwen35_speculative_matches_plain() {
    let plain = generate("gen_qwen35_plain", ToySpec::Qwen35, 40, false);
    let spec = generate("gen_qwen35_spec", ToySpec::Qwen35, 40, true);
    assert_eq!(spec, plain, "speculative decoding diverged from plain");
}

#[test]
fn multi_turn_reset_replays_identically() {
    for (test, spec) in [
        ("multi_llama", ToySpec::Llama),
        ("multi_qwen35", ToySpec::Qwen35),
        ("multi_gemma", ToySpec::Gemma),
    ] {
        let _g = gpu();
        let (backend, cm) = load(test, spec, 256);
        let prompt = cm.chat.render("", &[], "count from one to five");
        let mut engine = Engine::new(
            backend,
            cm.model,
            cm.tokenizer,
            Sampler::greedy(7),
            cm.stop,
            false,
        );
        let collect = |engine: &mut Engine| -> Vec<u32> {
            let mut stream = engine.stream(&prompt, 24).unwrap();
            let mut tokens = Vec::new();
            for piece in stream.by_ref() {
                tokens.push(piece.unwrap().token);
            }
            tokens
        };
        let first = collect(&mut engine);
        assert!(!first.is_empty(), "{test}: first turn produced nothing");
        let second = collect(&mut engine);
        assert_eq!(second, first, "{test}: reset must replay the turn");
    }
}

#[test]
fn unsupported_formats_fail_fast() {
    let _g = gpu();

    let dir = std::env::temp_dir().join(format!("flint-badfmt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    flint_checkpoint::write_tensors(
        &dir.join("model.safetensors"),
        &[("a".to_string(), vec![1], vec![0u8; 4], false)],
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type": "bert"}"#).unwrap();
    let backend = Backend::new().unwrap();
    let err = flint_architectures::load(&dir, 64, &backend).err().unwrap();
    assert!(err.to_string().contains("unsupported model_type"), "{err}");

    let gguf_dir = std::env::temp_dir().join(format!("flint-badarch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&gguf_dir);
    std::fs::create_dir_all(&gguf_dir).unwrap();
    let mut w = flint_checkpoint::GgufWriter::new(32);
    w.kv_u32("llama.block_count", 1);
    std::fs::write(gguf_dir.join("model.gguf"), w.finish()).unwrap();
    let err = flint_architectures::load(&gguf_dir, 64, &backend)
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("missing general.architecture"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&gguf_dir).ok();
}

#[test]
fn generation_qwen35_variants_speculative_matches_plain() {
    for spec in [ToySpec::Qwen35Split, ToySpec::Qwen35Untied] {
        let tag = format!("gen_{spec:?}");
        let plain = generate(&tag, spec, 40, false);
        let specd = generate(&format!("{tag}_spec"), spec, 40, true);
        assert_eq!(specd, plain, "{spec:?}: speculation diverged");
    }
}

#[test]
fn verify_chunk_row0_matches_single_forward() {
    let _g = gpu();
    let (mut backend, mut cm) = load("verify_qwen35", ToySpec::Qwen35, 256);
    let prompt: Vec<u32> = (10..40).collect();
    let mut done = 0usize;
    while done < prompt.len() {
        let end = (done + flint_model::MAX_M as usize).min(prompt.len());
        cm.model
            .forward(&mut backend, &prompt[done..end], &[], &[])
            .unwrap();
        done = end;
    }

    cm.model.speculator().unwrap().snapshot(&backend);
    let single = cm
        .model
        .forward(&mut backend, &[7], &[0], &[])
        .unwrap()
        .logits
        .pop()
        .unwrap();
    cm.model.speculator().unwrap().restore(&backend);
    let pair = cm
        .model
        .forward(&mut backend, &[7, 9], &[0], &[])
        .unwrap()
        .logits
        .pop()
        .unwrap();

    let max_diff = single
        .iter()
        .zip(&pair)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(max_diff < 1e-2, "row-0 drift {max_diff}");
}

#[test]
fn tokenizer_markers_and_streaming_roundtrip() {
    for (test, spec) in [("tok_gguf", ToySpec::Llama), ("tok_json", ToySpec::Qwen35)] {
        let (backend, cm) = load(test, spec, 256);
        let tok = &cm.tokenizer;

        let im_start = "<|im_start|>";
        assert_eq!(
            tok.encode(im_start).unwrap(),
            vec![tok.token_id(im_start).unwrap()],
            "{test}: {im_start:?} must be one special id"
        );

        let ids = tok.encode("hello world").unwrap();
        let mut state = tok.stream_decoder();
        let mut text = String::new();
        for id in ids {
            text.push_str(&tok.step_decode(&mut state, id).unwrap().unwrap_or_default());
        }
        assert!(text.contains("hello"), "{test}: decoded {text:?}");
        drop(backend);
    }
}

#[test]
fn engine_and_model_fail_fast() {
    let _g = gpu();
    let (mut backend, mut cm) = load("fail_fast", ToySpec::Llama, 64);

    assert!(
        cm.model.forward(&mut backend, &[], &[0], &[]).is_err(),
        "empty chunk"
    );
    assert!(
        cm.model
            .forward(&mut backend, &[1u32; 129], &[0], &[])
            .is_err(),
        "chunk above MAX_M"
    );
    for _ in 0..4 {
        cm.model
            .forward(&mut backend, &[1u32; 16], &[], &[])
            .unwrap();
    }
    assert!(
        cm.model.forward(&mut backend, &[1], &[], &[]).is_err(),
        "context overflow"
    );

    let mut engine = Engine::new(
        backend,
        cm.model,
        cm.tokenizer,
        Sampler::greedy(1),
        cm.stop,
        false,
    );
    assert!(engine.stream("", 8).is_err(), "empty prompt");
    assert!(
        engine.stream(&"hello ".repeat(100), 8).is_err(),
        "prompt exceeds context"
    );
}
