use std::sync::{Mutex, MutexGuard};

use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, GenStats, Grammar, Piece, SamplingParams, SessionId};
use flint_model::{ChunkOut, LanguageModel, MAX_M, SeqChunk, Speculator};
use flint_tokenizer::Tokenizer;
use serde_json::json;

const VOCAB: u32 = 32;
const EOS: u32 = 31;
const SEQ_LEN: u32 = 512;

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

struct FakeModel {
    pos: Vec<u32>,
    saved: Vec<u32>,
    eos_at: Option<u32>,
}

impl FakeModel {
    fn new(eos_at: Option<u32>) -> Self {
        Self {
            pos: vec![0; 2],
            saved: vec![0; 2],
            eos_at,
        }
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
    fn forward(&mut self, _backend: &mut Backend, batch: &[SeqChunk]) -> Result<Vec<ChunkOut>> {
        let m: u32 = batch.iter().map(SeqChunk::len).sum();
        assert!(m > 0 && m <= MAX_M, "chunk size {m} outside [1, {MAX_M}]");
        let mut outs = Vec::with_capacity(batch.len());
        for chunk in batch {
            let s = chunk.seq as usize;
            assert!(
                self.pos[s] + chunk.len() <= SEQ_LEN,
                "context overflow in seq {s}"
            );
            let base = self.pos[s];
            let logits = chunk
                .logit_rows
                .iter()
                .map(|&r| self.logits_for(base + r))
                .collect();
            let hidden = chunk
                .hidden_rows
                .iter()
                .map(|&r| vec![(base + r) as f32; 4])
                .collect();
            self.pos[s] += chunk.len();
            outs.push(ChunkOut { logits, hidden });
        }
        Ok(outs)
    }

    fn reset(&mut self, _backend: &Backend, seq: u32) -> Result<()> {
        self.pos[seq as usize] = 0;
        Ok(())
    }
    fn pos(&self, seq: u32) -> u32 {
        self.pos[seq as usize]
    }
    fn context_limit(&self, _seq: u32) -> u32 {
        SEQ_LEN
    }
    fn seq_count(&self) -> u32 {
        self.pos.len() as u32
    }
    fn vocab(&self) -> u32 {
        VOCAB
    }
    fn eos(&self) -> &[u32] {
        &[EOS]
    }

    fn alloc_pages(&mut self, _backend: &Backend, seq: u32, tokens: u32) -> Result<()> {
        assert!(
            self.pos[seq as usize] + tokens <= SEQ_LEN,
            "context overflow in seq {seq}"
        );
        Ok(())
    }

    fn free_pages(&mut self, _backend: &Backend, _seq: u32) -> Result<()> {
        Ok(())
    }

    fn truncate_pages(&mut self, _backend: &Backend, _seq: u32, _keep_tokens: u32) -> Result<()> {
        Ok(())
    }

    fn speculator(&mut self) -> Option<&mut dyn Speculator> {
        Some(self as &mut dyn Speculator)
    }
}

impl Speculator for FakeModel {
    fn draft(
        &mut self,
        _backend: &mut Backend,
        _seq: u32,
        _token: u32,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        Ok(self.logits_for(hidden[0] as u32 + 1))
    }

    fn prime(&mut self, _seq: u32) {}

    fn snapshot(&mut self, _backend: &Backend, seq: u32) {
        self.saved[seq as usize] = self.pos[seq as usize];
    }

    fn restore(&mut self, _backend: &Backend, seq: u32) {
        self.pos[seq as usize] = self.saved[seq as usize] + 1;
    }
}

fn tokenizer() -> Tokenizer {
    let vocab: Vec<String> = (0..VOCAB)
        .map(|i| format!("[\"t{i}\", -{}]", i as f64 + 1.0))
        .collect();
    tokenizer_with_vocab(vocab)
}

fn json_tokenizer() -> Tokenizer {
    let mut vocab: Vec<String> = ["{", "\"", "m", "o", "d", "e", ":", "f", "a", "s", "t", "}"]
        .iter()
        .map(|s| format!("[{}, -1.0]", serde_json::to_string(s).unwrap()))
        .collect();
    for i in 0..(VOCAB - 12) {
        vocab.push(format!("[\"x{i}\", -{}]", i as f64 + 2.0));
    }
    tokenizer_with_vocab(vocab)
}

fn tokenizer_with_vocab(vocab: Vec<String>) -> Tokenizer {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("flint-gen-tok-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
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

fn new_engine(eos_at: Option<u32>, sampling: SamplingParams, speculate: bool) -> Engine {
    let backend = Backend::new().unwrap();
    Engine::new(
        backend,
        Box::new(FakeModel::new(eos_at)),
        tokenizer(),
        sampling,
        1,
        vec![EOS],
        speculate,
    )
}

fn drain(engine: &mut Engine, id: SessionId) -> Vec<u32> {
    let mut tokens = Vec::new();
    loop {
        engine.step().unwrap();
        for p in engine.poll(id) {
            tokens.push(p.token);
        }
        if engine.finished(id) {
            break;
        }
        assert!(tokens.len() <= 4096, "runaway generation");
    }
    tokens
}

fn run(prompt: &str, max_tokens: usize, eos_after: Option<u32>) -> (Vec<u32>, GenStats, u32) {
    let _g = gpu();
    let n = tokenizer().encode(prompt).unwrap().len() as u32;
    let eos_at = eos_after.map(|k| n - 1 + k);
    let mut engine = new_engine(eos_at, SamplingParams::default(), false);
    let id = engine.create(prompt, max_tokens, None, None, &[]).unwrap();
    let tokens = drain(&mut engine, id);
    let stats = engine.stats(id).unwrap();
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
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let mut engine = new_engine(None, params, false);
    let collect = |engine: &mut Engine| -> Vec<u32> {
        let id = engine.create("t0 t1 t2", 8, None, None, &[]).unwrap();
        let tokens = drain(engine, id);
        engine.close(id).unwrap();
        tokens
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
fn concurrent_sessions_share_the_engine_without_interference() {
    let _g = gpu();
    let mut engine = new_engine(None, SamplingParams::default(), false);
    let a = engine.create("t0 t1 t2", 6, None, None, &[]).unwrap();
    let b = engine.create("t3 t4", 6, None, None, &[]).unwrap();
    let mut got_a = Vec::new();
    let mut got_b = Vec::new();
    loop {
        engine.step().unwrap();
        got_a.extend(engine.poll(a).into_iter().map(|p| p.token));
        got_b.extend(engine.poll(b).into_iter().map(|p| p.token));
        if engine.finished(a) && engine.finished(b) {
            break;
        }
        assert!(got_a.len() <= 64 && got_b.len() <= 64, "runaway batch");
    }
    assert_eq!(got_a.len(), 6);
    assert_eq!(got_b.len(), 6);
    assert_ne!(got_a, got_b, "sessions must stay independent");
    engine.close(a).unwrap();
    engine.close(b).unwrap();
}

#[test]
fn session_exhaustion_fails_fast() {
    let _g = gpu();
    let mut engine = new_engine(None, SamplingParams::default(), false);
    let a = engine.create("t0", 8, None, None, &[]).unwrap();
    let b = engine.create("t1", 8, None, None, &[]).unwrap();
    let err = engine.create("t2", 8, None, None, &[]).err().unwrap();
    assert!(err.to_string().contains("no free sequence"), "{err}");
    let _ = (a, b);
}

#[test]
fn greedy_speculation_matches_plain_greedy() {
    let _g = gpu();
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let plain = {
        let mut engine = new_engine(None, params, false);
        let id = engine.create("t0 t1", 24, None, None, &[]).unwrap();
        drain(&mut engine, id)
    };
    let spec = {
        let mut engine = new_engine(None, params, true);
        let id = engine.create("t0 t1", 24, None, None, &[]).unwrap();
        drain(&mut engine, id)
    };
    assert_eq!(
        spec, plain,
        "speculative decode must not alter greedy output"
    );
}

#[test]
fn grammar_forces_the_schema_literal() {
    let _g = gpu();
    let backend = Backend::new().unwrap();
    let mut engine = Engine::new(
        backend,
        Box::new(FakeModel::new(None)),
        json_tokenizer(),
        SamplingParams::default(),
        1,
        vec![EOS],
        false,
    );
    let grammar = Grammar::from_schema(&json!({
        "type": "object",
        "required": ["mode"],
        "properties": {"mode": {"type": "string", "enum": ["fast"]}}
    }))
    .unwrap();
    let id = engine.create("t0", 64, Some(grammar), None, &[]).unwrap();
    let mut tokens = Vec::new();
    loop {
        engine.step().unwrap();
        tokens.extend(engine.poll(id).into_iter().map(|p| p.token));
        if engine.finished(id) {
            break;
        }
        assert!(tokens.len() < 4096, "runaway constrained generation");
    }
    let want: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 1, 6, 1, 7, 8, 9, 10, 1, 11];
    assert_eq!(
        tokens, want,
        "the grammar must force the exact JSON literal"
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

#[test]
fn poll_drains_pieces_and_finished_holds_until_empty() {
    let _g = gpu();
    let mut engine = new_engine(None, SamplingParams::default(), false);
    let id = engine.create("t0", 3, None, None, &[]).unwrap();
    let mut all: Vec<Piece> = Vec::new();
    while !engine.finished(id) {
        engine.step().unwrap();
        all.extend(engine.poll(id));
    }
    assert_eq!(all.len(), 3);
    assert!(engine.poll(id).is_empty(), "poll must drain");
}
