use std::sync::{Mutex, MutexGuard};

use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, GenStats, Sampler};
use flint_model::{ChunkOut, LanguageModel, MAX_M};
use flint_tokenizer::Tokenizer;

const VOCAB: u32 = 32;
const EOS: u32 = 31;
const MAX_SEQ: u32 = 512;

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

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
        assert!(m > 0 && m <= MAX_M, "chunk size {m} outside [1, {MAX_M}]");
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

fn tokenizer() -> Tokenizer {
    let dir = std::env::temp_dir().join(format!("flint-gen-tok-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let vocab: Vec<String> = (0..VOCAB)
        .map(|i| format!("[\"t{i}\", -{}]", i as f64 + 1.0))
        .collect();
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{"type": "Whitespace"}},
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [{}]
            }}
        }}"#,
        vocab.join(",")
    );
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, json).unwrap();
    let tok = Tokenizer::from_file(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    tok
}

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
    let prompt = (0..200)
        .map(|i| format!("t{}", i % 31))
        .collect::<Vec<_>>()
        .join(" ");
    let n = tokenizer().encode(&prompt).unwrap().len();
    assert!(n > MAX_M as usize, "prompt must span prefill chunks");
    let (tokens, stats, _) = run(&prompt, 8, None);
    assert_eq!(stats.prefill_tokens, n, "all prompt tokens counted");
    assert_eq!(stats.decode_tokens, 8);
    assert_eq!(tokens.len(), 8);
    assert!(stats.prefill_secs >= 0.0 && stats.decode_secs >= 0.0);
}
