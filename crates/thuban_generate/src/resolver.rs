use std::time::Instant;

use thuban_backend::Backend;
use thuban_error::{Error, Result};
use thuban_model::{ChunkOut, LanguageModel, SeqChunk};
use thuban_tokenizer::Tokenizer;

use crate::engine::{Phase, Piece, Session};
use crate::grammar::TokenTrie;
use crate::sampler::Dist;

pub(crate) struct Plan {
    pub(crate) id: u32,
    pub(crate) seq: u32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) logit_rows: Vec<u32>,
    pub(crate) hidden_rows: Vec<u32>,
    pub(crate) prefill_end: Option<usize>,
}

pub(crate) struct Resolver<'a> {
    pub(crate) backend: &'a mut Backend,
    pub(crate) model: &'a mut Box<dyn LanguageModel + Send>,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) trie: &'a Option<TokenTrie>,
    pub(crate) speculate: bool,
}

impl Resolver<'_> {
    pub(crate) fn resolve(&mut self, plan: &Plan, out: &ChunkOut, s: &mut Session) -> Result<()> {
        let phase = std::mem::replace(&mut s.phase, Phase::Done);
        match phase {
            Phase::Done => {}
            Phase::Prefill { .. } => self.prefill(plan, out, s)?,
            Phase::Plain { pending } => self.plain(out, s, pending)?,
            Phase::Spec {
                pending,
                draft_token,
                draft_dist,
            } => self.spec(out, s, pending, draft_token, draft_dist)?,
        }
        Ok(())
    }

    fn prefill(&mut self, plan: &Plan, out: &ChunkOut, s: &mut Session) -> Result<()> {
        let end = plan.prefill_end.expect("prefill plan carries its end");
        s.stats.prefill_tokens += plan.tokens.len();
        if end < s.prompt.len() {
            s.phase = Phase::Prefill { done: end };
            return Ok(());
        }
        s.start_decode();
        let mask = grammar_mask(self.trie, s)?;
        let first = s
            .sampler
            .transform(&out.logits[0], &s.context, mask.as_deref());
        let pending = s.sampler.draw(&first);
        if self.speculate {
            let spec = self
                .model
                .speculator()
                .expect("speculate implies a speculator");
            spec.prime(s.seq);
            let draft_logits = spec.draft(self.backend, s.seq, pending, &out.hidden[0])?;
            let draft_dist = dist_after_token(self.trie, s, pending, &draft_logits)?;
            let draft_token = s.sampler.draw(&draft_dist);
            s.phase = Phase::Spec {
                pending,
                draft_token,
                draft_dist,
            };
        } else {
            s.phase = Phase::Plain { pending };
        }
        Ok(())
    }

    fn plain(&mut self, out: &ChunkOut, s: &mut Session, pending: u32) -> Result<()> {
        if s.halted(pending) {
            s.phase = Phase::Done;
            return Ok(());
        }
        let piece = s.piece(self.tokenizer, self.trie, pending)?;
        s.queue.push_back(piece);
        let mask = grammar_mask(self.trie, s)?;
        let dist = s
            .sampler
            .transform(&out.logits[0], &s.context, mask.as_deref());
        let next = s.sampler.draw(&dist);
        s.phase = Phase::Plain { pending: next };
        Ok(())
    }

    fn spec(
        &mut self,
        out: &ChunkOut,
        s: &mut Session,
        pending: u32,
        draft_token: u32,
        draft_dist: Dist,
    ) -> Result<()> {
        if s.halted(pending) {
            s.phase = Phase::Done;
            return Ok(());
        }
        let target_dist = dist_after_token(self.trie, s, pending, &out.logits[0])?;
        let (accepted, chosen) = s.sampler.verify(&target_dist, &draft_dist, draft_token);
        if accepted {
            s.stats.accepted += 1;
            let p1 = s.piece(self.tokenizer, self.trie, pending)?;
            s.queue.push_back(p1);
            if s.halted(draft_token) {
                s.phase = Phase::Done;
                return Ok(());
            }
            let p2 = s.piece(self.tokenizer, self.trie, draft_token)?;
            s.queue.push_back(p2);
            let bonus_dist = masked_dist(self.trie, s, &out.logits[1])?;
            let bonus = s.sampler.draw(&bonus_dist);
            let spec = self
                .model
                .speculator()
                .expect("spec phase implies a speculator");
            spec.advance(self.backend, s.seq, draft_token, &out.hidden[0])?;
            let draft_logits = spec.draft(self.backend, s.seq, bonus, &out.hidden[1])?;
            let next_dist = dist_after_token(self.trie, s, bonus, &draft_logits)?;
            let next_draft = s.sampler.draw(&next_dist);
            s.phase = Phase::Spec {
                pending: bonus,
                draft_token: next_draft,
                draft_dist: next_dist,
            };
            return Ok(());
        }

        self.model
            .speculator()
            .expect("spec phase implies a speculator")
            .restore(self.backend, s.seq);
        let keep = self.model.pos(s.seq);
        self.model.truncate_pages(self.backend, s.seq, keep)?;
        self.model.alloc_pages(self.backend, s.seq, 1)?;
        let p1 = s.piece(self.tokenizer, self.trie, pending)?;
        s.queue.push_back(p1);
        if s.halted(chosen) {
            s.phase = Phase::Done;
            return Ok(());
        }
        let out2 = self.model.forward(
            self.backend,
            &[SeqChunk {
                tokens: &[chosen],
                seq: s.seq,
                logit_rows: &[],
                hidden_rows: &[0],
            }],
        )?;
        let spec = self
            .model
            .speculator()
            .expect("spec phase implies a speculator");
        let draft_logits = spec.draft(self.backend, s.seq, chosen, &out2[0].hidden[0])?;
        let next_dist = dist_after_token(self.trie, s, chosen, &draft_logits)?;
        let next_draft = s.sampler.draw(&next_dist);
        s.phase = Phase::Spec {
            pending: chosen,
            draft_token: next_draft,
            draft_dist: next_dist,
        };
        Ok(())
    }
}

fn grammar_mask(trie: &Option<TokenTrie>, s: &Session) -> Result<Option<Vec<f32>>> {
    let Some(matcher) = &s.matcher else {
        return Ok(None);
    };
    let trie = trie.as_ref().expect("grammar implies a token trie");
    let mut mask = matcher.mask(trie);
    if matcher.is_complete() {
        for &t in &s.stop {
            mask[t as usize] = 1.0;
        }
    }
    if !mask.iter().any(|&v| v > 0.0) {
        return Err(Error::Model("grammar reached a dead end".into()));
    }
    Ok(Some(mask))
}

fn masked_dist(trie: &Option<TokenTrie>, s: &mut Session, logits: &[f32]) -> Result<Dist> {
    let mask = grammar_mask(trie, s)?;
    Ok(s.sampler.transform(logits, &s.context, mask.as_deref()))
}

fn dist_after_token(
    trie: &Option<TokenTrie>,
    s: &mut Session,
    prev: u32,
    logits: &[f32],
) -> Result<Dist> {
    s.context.push(prev);
    let d = masked_dist(trie, s, logits);
    s.context.pop();
    d
}

impl Session {
    fn start_decode(&mut self) {
        self.stats.prefill_secs = self.prefill_start.elapsed().as_secs_f64();
        self.decode_start = Some(Instant::now());
    }

    fn halted(&self, token: u32) -> bool {
        self.stop.contains(&token) || self.stats.decode_tokens >= self.max_tokens
    }

    fn piece(
        &mut self,
        tokenizer: &Tokenizer,
        trie: &Option<TokenTrie>,
        token: u32,
    ) -> Result<Piece> {
        let text = tokenizer
            .step_decode(&mut self.decoder, token)?
            .unwrap_or_default();
        self.context.push(token);
        if let Some(matcher) = &mut self.matcher {
            matcher.commit(trie.as_ref().expect("grammar implies a token trie"), token);
        }
        self.stats.decode_tokens += 1;
        Ok(Piece { token, text })
    }
}
