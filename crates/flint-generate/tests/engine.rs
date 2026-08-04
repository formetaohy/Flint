//! Engine state machine over a scripted fake model: stop-token and token
//! budget termination, statistics accounting, and multi-turn reset
//! determinism. The fake model owns no GPU state — its logits are pure
//! functions of the position — so every behavioral claim is exact, unlike
//! pipeline tests over random toy weights.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use flint_backend::Backend;
use flint_checkpoint::{MetaVal, Metadata};
use flint_error::Result;
use flint_generate::{Engine, GenStats, Sampler};
use flint_model::{ChunkOut, LanguageModel, M_MAX};
use flint_tokenizer::Tokenizer;

const VOCAB: u32 = 32;
const EOS: u32 = 31; // last id, outside the plain-token cycle
const MAX_SEQ: u32 = 512;

/// The adapter is a single shared, memory-limited resource: serialize GPU work.
static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Scripted model: the next token after position p is (p*7 + 3) % 31, except
/// at `eos_at`, where it is the eos id. Deterministic and position-dependent,
/// so reset bugs show up as divergent second turns.
struct FakeModel {
    pos: u32,
    eos_at: Option<u32>,
}

impl FakeModel {
    fn new(eos_at: Option<u32>) -> Self {
        Self { pos: 0, eos_at }
    }

    fn next_token(&self, pos: u32) -> u32 {
        if self.eos_at == Some(pos) {
            EOS
        } else {
            (pos * 7 + 3) % (VOCAB - 1)
        }
    }

    fn logits_for(&self, pos: u32) -> Vec<f32> {
        let mut v = vec![-1.0; VOCAB as usize];
        v[self.next_token(pos) as usize] = 1.0;
        v
    }
}

impl LanguageModel for FakeModel {
    fn forward(
        &mut self,
        _backend: &mut Backend,
        tokens: &[u32],
        logit_rows: &[u32],
        hidden_rows: &[u32],
    ) -> Result<ChunkOut> {
        let m = tokens.len() as u32;
        assert!(m > 0 && m <= M_MAX, "chunk size {m} outside [1, {M_MAX}]");
        assert!(self.pos + m <= MAX_SEQ, "context overflow");
        let base = self.pos;
        let logits = logit_rows
            .iter()
            .map(|&r| self.logits_for(base + r))
            .collect();
        let hidden = hidden_rows
            .iter()
            .map(|&r| vec![(base + r) as f32; 4])
            .collect();
        self.pos += m;
        Ok(ChunkOut { logits, hidden })
    }

    fn reset(&mut self, _backend: &Backend) {
        self.pos = 0;
    }
    fn pos(&self) -> u32 {
        self.pos
    }
    fn max_seq(&self) -> u32 {
        MAX_SEQ
    }
    fn vocab(&self) -> u32 {
        VOCAB
    }
    fn eos(&self) -> &[u32] {
        &[EOS]
    }
}

/// A unigram tokenizer whose vocab covers ids 0..VOCAB, so any scripted token
/// sequence round-trips through the engine's streaming decoder.
fn tokenizer() -> Tokenizer {
    let tokens: Vec<String> = (0..VOCAB).map(|i| format!("t{i}")).collect();
    let scores: Vec<f64> = (0..VOCAB).map(|i| -(i as f64) - 1.0).collect();
    let mut kv = HashMap::new();
    kv.insert("tokenizer.ggml.model".into(), MetaVal::Str("llama".into()));
    kv.insert(
        "tokenizer.ggml.tokens".into(),
        MetaVal::Arr(tokens.iter().map(|t| MetaVal::Str(t.clone())).collect()),
    );
    kv.insert(
        "tokenizer.ggml.scores".into(),
        MetaVal::Arr(scores.iter().map(|s| MetaVal::Float(*s)).collect()),
    );
    kv.insert("tokenizer.ggml.unknown_token_id".into(), MetaVal::UInt(0));
    Tokenizer::from_gguf(&Metadata::new(kv)).unwrap()
}

/// Collects the committed token ids of one stream run. `eos_after` puts the
/// eos prediction k decoded positions past the prefill (0 = the very first
/// pending token is eos). The prompt's true encoded length is measured, not
/// assumed, because the unigram tokenizer folds spaces/unknowns into ids.
fn run(prompt: &str, max_tokens: usize, eos_after: Option<u32>) -> (Vec<u32>, GenStats, u32) {
    let _g = gpu();
    let backend = Backend::new().unwrap();
    let tok = tokenizer();
    let n = tok.encode(prompt).unwrap().len() as u32;
    let eos_at = eos_after.map(|k| n - 1 + k);
    let mut engine = Engine::new(
        backend,
        Box::new(FakeModel::new(eos_at)),
        tok,
        Sampler::greedy(1),
        vec![EOS],
        false,
    );
    let mut tokens = Vec::new();
    let mut stream = engine.stream(prompt, max_tokens).unwrap();
    for piece in stream.by_ref() {
        tokens.push(piece.unwrap().token);
    }
    let stats = {
        let s = stream.stats();
        GenStats {
            prefill_tokens: s.prefill_tokens,
            decode_tokens: s.decode_tokens,
            accepted: s.accepted,
            prefill_secs: s.prefill_secs,
            decode_secs: s.decode_secs,
        }
    };
    (tokens, stats, n)
}

#[test]
fn eos_terminates_generation_without_emitting_it() {
    // The eos is predicted two decoded positions past the prefill: the first
    // two committed tokens stream out, the third pending token halts.
    let (tokens, stats, n) = run("t0 t1 t2", 64, Some(2));
    let want = [((n - 1) * 7 + 3) % (VOCAB - 1), (n * 7 + 3) % (VOCAB - 1)];
    assert_eq!(tokens, want, "eos must not be emitted");
    assert_eq!(stats.decode_tokens, 2);
}

#[test]
fn eos_at_first_decode_stops_immediately() {
    let (tokens, stats, _) = run("t0 t1 t2", 64, Some(0));
    assert!(tokens.is_empty(), "eos predicted first: no output");
    assert_eq!(stats.decode_tokens, 0);
}

#[test]
fn token_budget_terminates_without_eos() {
    let (tokens, stats, n) = run("t0", 5, None);
    assert_eq!(tokens.len(), 5, "budget caps the stream");
    assert_eq!(stats.decode_tokens, 5);
    assert_eq!(
        tokens[0],
        ((n - 1) * 7 + 3) % (VOCAB - 1),
        "deterministic script"
    );
}

#[test]
fn multi_turn_reset_reproduces_output() {
    let _g = gpu();
    let backend = Backend::new().unwrap();
    let mut engine = Engine::new(
        backend,
        Box::new(FakeModel::new(None)),
        tokenizer(),
        Sampler::greedy(1),
        vec![EOS],
        false,
    );
    let collect = |engine: &mut Engine| -> Vec<u32> {
        let mut stream = engine.stream("t0 t1 t2", 8).unwrap();
        stream.by_ref().map(|p| p.unwrap().token).collect()
    };
    let first = collect(&mut engine);
    assert!(!first.is_empty(), "first turn generated tokens");
    let second = collect(&mut engine);
    assert_eq!(
        second, first,
        "reset must restore position so the second turn replays identically"
    );
}

#[test]
fn stats_account_prefill_and_decode() {
    // 200 space-separated words encode to well over M_MAX tokens, forcing
    // chunked prefill; the count is measured from the same tokenizer.
    let prompt = (0..200)
        .map(|i| format!("t{}", i % 31))
        .collect::<Vec<_>>()
        .join(" ");
    let n = tokenizer().encode(&prompt).unwrap().len();
    assert!(n > M_MAX as usize, "prompt must span prefill chunks");
    let (tokens, stats, _) = run(&prompt, 8, None);
    assert_eq!(stats.prefill_tokens, n, "all prompt tokens counted");
    assert_eq!(stats.decode_tokens, 8);
    assert_eq!(tokens.len(), 8);
    assert!(stats.prefill_secs >= 0.0 && stats.decode_secs >= 0.0);
}

#[test]
fn stats_summary_format() {
    let s = GenStats {
        prefill_tokens: 100,
        decode_tokens: 50,
        accepted: 7,
        prefill_secs: 1.0,
        decode_secs: 2.0,
    };
    let text = s.summary();
    assert!(text.contains("prefill: 100 tok"), "{text}");
    assert!(text.contains("decode: 50 tok"), "{text}");
    assert!(text.contains("accepted: 7"), "{text}");

    let idle = GenStats {
        prefill_secs: 0.0,
        decode_secs: 0.0,
        ..s
    };
    assert!(idle.summary().contains("0.0 tok/s"));
}
