//! Checkpoint-level tests over the downloaded models. Every supported family
//! x format pair must load, prefill identically chunked and sequential, and
//! generate correctly; tokenizer markers and engine error paths are pinned
//! per family. GPU tests serialize on a lock: the adapter is a single shared,
//! memory-limited resource.
//!
//! Marker strings are built by concatenation so this file never embeds a
//! literal special-token string.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use flint_backend::Backend;
use flint_generate::{Engine, Sampler};
use flint_model::LanguageModel;
use flint_tokenizer::Tokenizer;

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

const QWEN35: &str = "Qwen3.5-0.8B";
const QWEN25: &str = "Qwen2.5-0.5B-GGUF";
const QWEN3: &str = "Qwen3-0.6B-GGUF";
const GEMMA: &str = "Gemma3-1B-GGUF";
const SMOLLM: &str = "SmolLM2-360M-GGUF";

fn model_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(name)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Replays the prompt one token at a time; returns the final logits.
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

/// Prefills in ROWS-wide chunks; returns the last chunk's final-row logits.
fn chunked_logits(
    model: &mut Box<dyn LanguageModel>,
    backend: &mut Backend,
    prompt: &[u32],
) -> Vec<f32> {
    let mut logits = Vec::new();
    let mut done = 0usize;
    while done < prompt.len() {
        let end = (done + flint_model::ROWS as usize).min(prompt.len());
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

/// Chunked prefill (16-wide) must reproduce a one-token-at-a-time replay.
/// The WGPU backend has small run-to-run reduction nondeterminism, so the
/// equivalence signal is cosine similarity of the full logit vector. The KV
/// cache is packed bf16, which lowers the agreement floor (truncation noise
/// compounds across layers); a correct graph still stays well above 0.98,
/// while a wrong load/dequant/forward graph collapses far below that.
fn prefill_equivalence(dir: &str) {
    let _g = gpu();
    let mut backend = Backend::new().unwrap();
    let mut cm = flint_archs::load(&model_dir(dir), 256, &backend).unwrap();
    let prompt: Vec<u32> = (1000..1040).collect();

    let seq = sequential_logits(&mut cm.model, &mut backend, &prompt);
    cm.model.reset(&backend);
    let chunked = chunked_logits(&mut cm.model, &mut backend, &prompt);

    assert_eq!(chunked.len(), seq.len(), "{dir}: vocab mismatch");
    let cos = cosine(&chunked, &seq);
    assert!(
        cos > 0.98,
        "{dir}: logit distribution diverges (cosine {cos})"
    );
}

// ---------------------------------------------------------------- prefill

#[test]
fn prefill_qwen35_safetensors_hybrid() {
    prefill_equivalence(QWEN35);
}

#[test]
fn prefill_qwen25_gguf_biased_qkv() {
    prefill_equivalence(QWEN25);
}

#[test]
fn prefill_qwen3_gguf_qk_norm() {
    prefill_equivalence(QWEN3);
}

#[test]
fn prefill_gemma_gguf_sliding_window() {
    prefill_equivalence(GEMMA);
}

#[test]
fn prefill_smollm_gguf_group64_quant() {
    prefill_equivalence(SMOLLM);
}

// ---------------------------------------------------------------- tokenizer

fn im_marker(start: bool) -> String {
    format!("<|im_{}|>", if start { "start" } else { "end" })
}

fn turn_marker(start: bool) -> String {
    format!("<{}_of_turn>", if start { "start" } else { "end" })
}

/// Turn markers encode as single special ids and streaming decode round-trips
/// plain text, for every tokenizer flavor (HF json, GGUF BPE, GGUF Unigram).
#[test]
fn tokenizer_markers_and_streaming_roundtrip() {
    let im_start = im_marker(true);
    let im_end = im_marker(false);
    let cases: &[(&str, Vec<(String, u32)>)] = &[
        (
            QWEN35,
            vec![(im_start.clone(), 248045), (im_end.clone(), 248046)],
        ),
        (QWEN25, vec![(im_start.clone(), 151644)]),
        (QWEN3, vec![(im_end.clone(), 151645)]),
        (SMOLLM, vec![(im_end.clone(), 49137)]),
        (
            GEMMA,
            vec![(turn_marker(true), 105), (turn_marker(false), 106)],
        ),
    ];
    for (dir, markers) in cases {
        let source = flint_checkpoint::open(&model_dir(dir)).unwrap();
        let tok = Tokenizer::load(&model_dir(dir), source.as_ref()).unwrap();
        for (text, id) in markers {
            assert_eq!(
                tok.encode(text).unwrap(),
                vec![*id],
                "{dir}: {text:?} must be one special id"
            );
        }

        let ids = tok.encode("Hello world").unwrap();
        let mut state = tok.decoder();
        let mut text = String::new();
        for id in ids {
            text.push_str(&tok.step_decode(&mut state, id).unwrap().unwrap_or_default());
        }
        assert!(
            text.contains("Hello world"),
            "{dir}: streamed decode produced {text:?}"
        );
    }
}

// ---------------------------------------------------------------- generation

fn generate(
    dir: &str,
    system: &str,
    user: &str,
    max_tokens: usize,
    speculate: bool,
) -> (Vec<u32>, String, usize) {
    let _g = gpu();
    let backend = Backend::new().unwrap();
    let cm = flint_archs::load(&model_dir(dir), 256, &backend).unwrap();
    let prompt = cm.chat.render(system, &[], user);
    let mut engine = Engine::new(
        backend,
        cm.model,
        cm.tokenizer,
        Sampler::greedy(7),
        cm.stop,
        speculate,
    );
    let mut stream = engine.stream(&prompt, max_tokens).unwrap();
    let mut tokens = Vec::new();
    let mut text = String::new();
    for piece in stream.by_ref() {
        let piece = piece.unwrap();
        tokens.push(piece.token);
        text.push_str(&piece.text);
    }
    let accepted = stream.stats().accepted;
    (tokens, text, accepted)
}

/// Greedy speculative decoding must emit exactly the same tokens as the plain
/// loop (speculation changes speed, never output) and the MTP draft head must
/// actually get drafts accepted.
#[test]
fn generation_qwen35_speculative_matches_plain() {
    let (plain, _, _) = generate(
        QWEN35,
        "You are a terse assistant.",
        "Count from 1 to 8.",
        40,
        false,
    );
    let (spec, _, accepted) = generate(
        QWEN35,
        "You are a terse assistant.",
        "Count from 1 to 8.",
        40,
        true,
    );
    assert!(!plain.is_empty(), "generation produced nothing");
    assert!(accepted > 0, "MTP draft never accepted");
    assert_eq!(spec, plain, "speculative decoding diverged from plain");
}

#[test]
fn generation_qwen25_greedy_replies() {
    let (tokens, text, _) = generate(
        QWEN25,
        "You are a terse assistant.",
        "Reply with the single word: ok",
        16,
        false,
    );
    assert!(!tokens.is_empty(), "gguf generation produced nothing");
    assert!(!text.trim().is_empty(), "gguf generation decoded no text");
}

#[test]
fn generation_gemma_arithmetic_and_stop() {
    let (_, text, _) = generate(
        GEMMA,
        "",
        "What is 2 + 2? Answer with just the number.",
        40,
        false,
    );
    assert!(text.contains('4'), "gemma arithmetic produced {text:?}");
    assert!(
        !text.contains(&turn_marker(false)),
        "stop token must not be emitted as text"
    );
}

// ---------------------------------------------------------------- model api

/// A 2-token verify chunk must produce the same row-0 logits as a single
/// forward from identical recurrent state (snapshot/restore correctness).
#[test]
fn verify_chunk_row0_matches_single_forward() {
    let _g = gpu();
    let mut backend = Backend::new().unwrap();
    let mut model = flint_archs::load(&model_dir(QWEN35), 256, &backend)
        .unwrap()
        .model;

    let prompt: Vec<u32> = (5000..5040).collect();
    let mut done = 0usize;
    while done < prompt.len() {
        let end = (done + flint_model::ROWS as usize).min(prompt.len());
        model
            .forward(&mut backend, &prompt[done..end], &[], &[])
            .unwrap();
        done = end;
    }

    model.speculator().unwrap().snapshot(&backend);
    let single = model
        .forward(&mut backend, &[7000], &[0], &[])
        .unwrap()
        .logits
        .pop()
        .unwrap();
    model.speculator().unwrap().restore(&backend);
    let pair = model
        .forward(&mut backend, &[7000, 9000], &[0], &[])
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

// ---------------------------------------------------------------- errors

/// Engine and model fail fast on invalid input (SmolLM2: smallest checkpoint,
/// loaded with a tiny context to make overflow cheap).
#[test]
fn engine_and_model_fail_fast() {
    let _g = gpu();
    let mut backend = Backend::new().unwrap();
    let mut cm = flint_archs::load(&model_dir(SMOLLM), 64, &backend).unwrap();

    assert!(
        cm.model.forward(&mut backend, &[], &[0], &[]).is_err(),
        "empty chunk"
    );
    assert!(
        cm.model
            .forward(&mut backend, &[1u32; 17], &[0], &[])
            .is_err(),
        "chunk above ROWS"
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
