//! Prefill/decode engine with optional speculative decoding.
//!
//! Speculation is exact by construction: the sampler's `transform` is the
//! single source of sampling distributions, so the draft distribution stored
//! in the spec phase and the target distribution computed at verification
//! carry the identical configuration over the identical context — the
//! rejection test compares the distribution the draft token was drawn from
//! against the target distribution, and committed tokens follow the target
//! distribution exactly (arXiv:2211.17192).

use std::collections::VecDeque;
use std::time::Instant;

use flint_backend::Backend;
use flint_error::{Error, Result};
use flint_model::{LanguageModel, ROWS};
use flint_tokenizer::{Decoder, Tokenizer};

use crate::sampler::{Dist, Sampler};

/// One committed token plus its freshly decoded text (may be empty while the
/// decoder buffers a partial character).
pub struct Piece {
    pub token: u32,
    pub text: String,
}

pub struct GenStats {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub accepted: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

impl GenStats {
    /// One-line throughput summary; printing is the caller's job.
    pub fn summary(&self) -> String {
        let pp = if self.prefill_secs > 0.0 {
            self.prefill_tokens as f64 / self.prefill_secs
        } else {
            0.0
        };
        let tg = if self.decode_secs > 0.0 {
            self.decode_tokens as f64 / self.decode_secs
        } else {
            0.0
        };
        format!(
            "[flint] prefill: {} tok in {:.2}s ({pp:.1} tok/s) | decode: {} tok in {:.2}s ({tg:.1} tok/s) | accepted: {}",
            self.prefill_tokens,
            self.prefill_secs,
            self.decode_tokens,
            self.decode_secs,
            self.accepted,
        )
    }
}

/// Owns backend, model, tokenizer and sampler; produces token streams.
pub struct Engine {
    backend: Backend,
    model: Box<dyn LanguageModel>,
    tokenizer: Tokenizer,
    sampler: Sampler,
    stop: Vec<u32>,
    speculate: bool,
}

impl Engine {
    pub fn new(
        backend: Backend,
        model: Box<dyn LanguageModel>,
        tokenizer: Tokenizer,
        sampler: Sampler,
        stop: Vec<u32>,
        speculate: bool,
    ) -> Self {
        Self {
            backend,
            model,
            tokenizer,
            sampler,
            stop,
            speculate,
        }
    }

    /// Formatted per-kernel GPU time breakdown, or None when profiling is off.
    pub fn profile_report(&self) -> Option<String> {
        if !self.backend.profiling() {
            return None;
        }
        let rows = self.backend.profile_report();
        let total: u64 = rows.iter().map(|r| r.total_ns).sum();
        let mut out = String::from("[flint] GPU kernel time breakdown (cumulative):\n");
        for r in rows {
            let pct = if total > 0 {
                r.total_ns as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:<12} {:9.2} ms  {:8} calls  {:5.1}%\n",
                r.label,
                r.total_ns as f64 / 1e6,
                r.count,
                pct
            ));
        }
        Some(out)
    }

    pub fn reset(&mut self) {
        self.model.reset(&self.backend);
    }

    /// Starts prompt prefill + autoregressive decode as a piece stream.
    pub fn stream(&mut self, prompt: &str, max_tokens: usize) -> Result<Stream<'_>> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        if prompt_ids.is_empty() {
            return Err(Error::Tokenizer("empty prompt".into()));
        }
        if prompt_ids.len() as u32 >= self.model.max_seq() {
            return Err(Error::Model(format!(
                "prompt of {} tokens exceeds context {}",
                prompt_ids.len(),
                self.model.max_seq()
            )));
        }
        self.reset();
        let speculate = self.speculate && self.model.speculator().is_some();
        Ok(Stream {
            backend: &mut self.backend,
            model: &mut self.model,
            tokenizer: &self.tokenizer,
            sampler: &mut self.sampler,
            stop: self.stop.clone(),
            speculate,
            decoder: self.tokenizer.decoder(),
            context: prompt_ids.clone(),
            prompt_ids,
            max_tokens,
            phase: Phase::Prefill { done: 0 },
            queue: VecDeque::new(),
            prefill_start: Instant::now(),
            decode_start: None,
            stats: GenStats {
                prefill_tokens: 0,
                decode_tokens: 0,
                accepted: 0,
                prefill_secs: 0.0,
                decode_secs: 0.0,
            },
        })
    }
}

enum Phase {
    Prefill {
        done: usize,
    },
    Plain {
        pending: u32,
    },
    /// pending is sampled from the target; draft_token was drawn from
    /// draft_dist, which verification must reproduce exactly.
    Spec {
        pending: u32,
        draft_token: u32,
        draft_dist: Dist,
    },
    Done,
}

/// Iterator over committed pieces; `stats()` is final once the stream ends.
pub struct Stream<'a> {
    backend: &'a mut Backend,
    model: &'a mut Box<dyn LanguageModel>,
    tokenizer: &'a Tokenizer,
    sampler: &'a mut Sampler,
    stop: Vec<u32>,
    speculate: bool,
    decoder: Decoder,
    prompt_ids: Vec<u32>,
    /// Every committed token; feeds the repetition penalty.
    context: Vec<u32>,
    max_tokens: usize,
    phase: Phase,
    queue: VecDeque<Piece>,
    prefill_start: Instant,
    decode_start: Option<Instant>,
    stats: GenStats,
}

impl Stream<'_> {
    pub fn stats(&mut self) -> &GenStats {
        if let Some(t) = self.decode_start {
            self.stats.decode_secs = t.elapsed().as_secs_f64();
        }
        &self.stats
    }

    /// Sampling distribution over plain committed context.
    fn dist(&mut self, logits: &[f32]) -> Dist {
        self.sampler.transform(logits, &self.context)
    }

    /// Sampling distribution for the token following `prev`: the repetition
    /// penalty sees the committed context plus `prev` itself, which is not
    /// committed yet (it is the pending token, or a draft about to be).
    fn dist_after(&mut self, logits: &[f32], prev: u32) -> Dist {
        self.context.push(prev);
        let d = self.sampler.transform(logits, &self.context);
        self.context.pop();
        d
    }

    fn halted(&self, token: u32) -> bool {
        self.stop.contains(&token) || self.stats.decode_tokens >= self.max_tokens
    }

    /// Commits one token: repetition context, decode text, decode stats.
    fn piece(&mut self, token: u32) -> Result<Piece> {
        let text = self
            .tokenizer
            .step_decode(&mut self.decoder, token)?
            .unwrap_or_default();
        self.context.push(token);
        self.stats.decode_tokens += 1;
        Ok(Piece { token, text })
    }

    fn start_decode(&mut self) {
        self.stats.prefill_secs = self.prefill_start.elapsed().as_secs_f64();
        self.decode_start = Some(Instant::now());
    }

    /// Advances the state machine until one piece is queued or generation ends.
    fn advance(&mut self) -> Result<()> {
        loop {
            if !self.queue.is_empty() {
                return Ok(());
            }
            let phase = std::mem::replace(&mut self.phase, Phase::Done);
            match phase {
                Phase::Done => return Ok(()),
                Phase::Prefill { done } => self.step_prefill(done)?,
                Phase::Plain { pending } => return self.step_plain(pending),
                Phase::Spec {
                    pending,
                    draft_token,
                    draft_dist,
                } => return self.step_spec(pending, draft_token, draft_dist),
            }
        }
    }

    fn step_prefill(&mut self, done: usize) -> Result<()> {
        let total = self.prompt_ids.len();
        let end = (done + ROWS as usize).min(total);
        let chunk = &self.prompt_ids[done..end];
        let m = chunk.len() as u32;
        let last = end == total;
        let row = [m - 1];
        let logit_rows: &[u32] = if last { &row } else { &[] };
        let hidden_rows: &[u32] = if last && self.speculate { &row } else { &[] };
        let out = self
            .model
            .forward(self.backend, chunk, logit_rows, hidden_rows)?;
        self.stats.prefill_tokens += chunk.len();
        if !last {
            self.phase = Phase::Prefill { done: end };
            return Ok(());
        }
        self.start_decode();
        let first = self.dist(&out.logits[0]);
        let pending = self.sampler.draw(&first);
        if self.speculate {
            let draft_logits = {
                let spec = self
                    .model
                    .speculator()
                    .expect("speculate implies a speculator");
                spec.prime();
                spec.draft(self.backend, pending, &out.hidden[0])?
            };
            let draft_dist = self.dist_after(&draft_logits, pending);
            let draft_token = self.sampler.draw(&draft_dist);
            self.phase = Phase::Spec {
                pending,
                draft_token,
                draft_dist,
            };
        } else {
            self.phase = Phase::Plain { pending };
        }
        Ok(())
    }

    fn step_plain(&mut self, pending: u32) -> Result<()> {
        if self.halted(pending) {
            return Ok(());
        }
        let out = self.model.forward(self.backend, &[pending], &[0], &[])?;
        let piece = self.piece(pending)?;
        let next_dist = self.dist(&out.logits[0]);
        let next = self.sampler.draw(&next_dist);
        self.phase = Phase::Plain { pending: next };
        self.queue.push_back(piece);
        Ok(())
    }

    fn step_spec(&mut self, pending: u32, draft_token: u32, draft_dist: Dist) -> Result<()> {
        if self.halted(pending) {
            return Ok(());
        }
        self.model
            .speculator()
            .expect("spec phase implies a speculator")
            .snapshot(self.backend);
        let out = self
            .model
            .forward(self.backend, &[pending, draft_token], &[0, 1], &[0, 1])?;

        // draft_dist was computed over context + [pending]; verification
        // reproduces that exact distribution for the target logits.
        let target_dist = self.dist_after(&out.logits[0], pending);
        let (accepted, chosen) = self.sampler.verify(&target_dist, &draft_dist, draft_token);

        if accepted {
            self.stats.accepted += 1;
            let p1 = self.piece(pending)?;
            self.queue.push_back(p1);
            if self.halted(draft_token) {
                return Ok(());
            }
            let p2 = self.piece(draft_token)?;
            self.queue.push_back(p2);
            // Context now ends with both committed tokens: sample the bonus,
            // then advance the draft head through each committed token (the
            // first draft call only syncs head state — the target already
            // supplied the bonus).
            let bonus_dist = self.dist(&out.logits[1]);
            let bonus = self.sampler.draw(&bonus_dist);
            let draft_logits = {
                let spec = self
                    .model
                    .speculator()
                    .expect("spec phase implies a speculator");
                spec.draft(self.backend, draft_token, &out.hidden[0])?;
                spec.draft(self.backend, bonus, &out.hidden[1])?
            };
            let draft_dist = self.dist_after(&draft_logits, bonus);
            let next_draft = self.sampler.draw(&draft_dist);
            self.phase = Phase::Spec {
                pending: bonus,
                draft_token: next_draft,
                draft_dist,
            };
            return Ok(());
        }

        // Rejected: roll the verification chunk back and replay the committed
        // token alone to restore generation state.
        self.model
            .speculator()
            .expect("spec phase implies a speculator")
            .restore(self.backend);
        let p1 = self.piece(pending)?;
        self.queue.push_back(p1);
        if self.halted(chosen) {
            return Ok(());
        }
        self.model.forward(self.backend, &[pending], &[], &[])?;
        let draft_logits = self
            .model
            .speculator()
            .expect("spec phase implies a speculator")
            .draft(self.backend, chosen, &out.hidden[0])?;
        let draft_dist = self.dist_after(&draft_logits, chosen);
        let next_draft = self.sampler.draw(&draft_dist);
        self.phase = Phase::Spec {
            pending: chosen,
            draft_token: next_draft,
            draft_dist,
        };
        Ok(())
    }
}

impl Iterator for Stream<'_> {
    type Item = Result<Piece>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(()) => self.queue.pop_front().map(Ok),
            Err(e) => {
                self.phase = Phase::Done;
                Some(Err(e))
            }
        }
    }
}
