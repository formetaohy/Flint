use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use flint_backend::Backend;
use flint_error::{Error, Result};
use flint_model::{ChunkOut, LanguageModel, MAX_M, SeqChunk};
use flint_tokenizer::{StreamDecoder, Tokenizer};

use crate::grammar::{Grammar, Matcher, TokenTrie};
use crate::sampler::{Dist, Sampler, SamplingParams};

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SessionId(pub u32);

enum Phase {
    Prefill {
        done: usize,
    },
    Plain {
        pending: u32,
    },
    Spec {
        pending: u32,
        draft_token: u32,
        draft_dist: Dist,
    },
    Done,
}

struct Session {
    seq: u32,
    sampler: Sampler,
    decoder: StreamDecoder,
    context: Vec<u32>,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: Vec<u32>,
    phase: Phase,
    queue: VecDeque<Piece>,
    stats: GenStats,
    prefill_start: Instant,
    decode_start: Option<Instant>,
    matcher: Option<Matcher>,
}

pub struct Engine {
    backend: Backend,
    model: Box<dyn LanguageModel>,
    tokenizer: Tokenizer,
    sampling: SamplingParams,
    seed: u64,
    stop: Vec<u32>,
    speculate: bool,
    free_seqs: Vec<u32>,
    next_id: u32,
    sessions: HashMap<u32, Session>,
    trie: Option<TokenTrie>,
}

impl Engine {
    pub fn new(
        backend: Backend,
        model: Box<dyn LanguageModel>,
        tokenizer: Tokenizer,
        sampling: SamplingParams,
        seed: u64,
        stop: Vec<u32>,
        speculate: bool,
    ) -> Self {
        let mut free_seqs: Vec<u32> = (0..model.seq_count()).collect();
        free_seqs.reverse();
        Self {
            backend,
            model,
            tokenizer,
            sampling,
            seed,
            stop,
            speculate,
            free_seqs,
            next_id: 0,
            sessions: HashMap::new(),
            trie: None,
        }
    }

    pub fn create(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        grammar: Option<Grammar>,
    ) -> Result<SessionId> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        if prompt_ids.is_empty() {
            return Err(Error::Tokenizer("empty prompt".into()));
        }
        let matcher = match grammar {
            Some(g) => {
                if self.trie.is_none() {
                    self.trie = Some(TokenTrie::from_tokenizer(&self.tokenizer));
                }
                Some(Matcher::new(Arc::new(g)))
            }
            None => None,
        };
        let seq = self.alloc_seq(&prompt_ids)?;
        self.model.reset(&self.backend, seq)?;
        self.model
            .alloc_pages(&self.backend, seq, prompt_ids.len() as u32)?;
        self.seed = self.seed.wrapping_add(1);
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(
            id,
            Session {
                seq,
                sampler: Sampler::new(self.sampling, self.seed),
                decoder: self.tokenizer.stream_decoder(),
                context: Vec::new(),
                prompt: prompt_ids,
                max_tokens,
                stop: self.stop.clone(),
                phase: Phase::Prefill { done: 0 },
                queue: VecDeque::new(),
                stats: GenStats {
                    prefill_tokens: 0,
                    decode_tokens: 0,
                    accepted: 0,
                    prefill_secs: 0.0,
                    decode_secs: 0.0,
                },
                prefill_start: Instant::now(),
                decode_start: None,
                matcher,
            },
        );
        Ok(SessionId(id))
    }

    fn alloc_seq(&mut self, prompt: &[u32]) -> Result<u32> {
        for i in (0..self.free_seqs.len()).rev() {
            if (prompt.len() as u32) < self.model.context_limit(self.free_seqs[i]) {
                return Ok(self.free_seqs.swap_remove(i));
            }
        }
        Err(Error::Model(format!(
            "prompt of {} tokens fits no free sequence",
            prompt.len()
        )))
    }

    pub fn step(&mut self) -> Result<()> {
        if self.sessions.is_empty() {
            return Ok(());
        }
        let speculate = self.speculate && self.model.speculator().is_some();
        if speculate {
            for s in self.sessions.values() {
                if matches!(s.phase, Phase::Spec { .. }) {
                    self.model
                        .speculator()
                        .expect("speculate implies a speculator")
                        .snapshot(&self.backend, s.seq);
                }
            }
        }

        let plan = self.assemble(speculate)?;
        if plan.is_empty() {
            return Ok(());
        }
        for p in &plan {
            self.model
                .alloc_pages(&self.backend, p.seq, p.tokens.len() as u32)?;
        }
        let batch: Vec<SeqChunk> = plan
            .iter()
            .map(|p| SeqChunk {
                tokens: &p.tokens,
                seq: p.seq,
                logit_rows: &p.logit_rows,
                hidden_rows: &p.hidden_rows,
            })
            .collect();
        let outs = self.model.forward(&mut self.backend, &batch)?;

        let Engine {
            backend,
            model,
            tokenizer,
            trie,
            sessions,
            ..
        } = self;
        let mut resolver = Resolver {
            backend,
            model,
            tokenizer,
            trie,
            speculate,
        };
        for (plan, out) in plan.into_iter().zip(outs.iter()) {
            let Some(mut s) = sessions.remove(&plan.id) else {
                continue;
            };
            resolver.resolve(&plan, out, &mut s)?;
            sessions.insert(plan.id, s);
        }
        Ok(())
    }

    fn assemble(&self, speculate: bool) -> Result<Vec<Plan>> {
        let mut plan = Vec::new();
        let mut used = 0usize;
        for (id, s) in &self.sessions {
            if used == MAX_M as usize {
                break;
            }
            let entry = match &s.phase {
                Phase::Done => None,
                Phase::Prefill { done } => {
                    let take = (s.prompt.len() - done).min(MAX_M as usize - used);
                    if take == 0 {
                        break;
                    }
                    let end = done + take;
                    let last = end == s.prompt.len();
                    let logit_rows: Vec<u32> = last.then_some(take as u32 - 1).into_iter().collect();
                    let hidden_rows: Vec<u32> =
                        (last && speculate).then_some(take as u32 - 1).into_iter().collect();
                    Some(Plan {
                        id: *id,
                        seq: s.seq,
                        tokens: s.prompt[*done..end].to_vec(),
                        logit_rows,
                        hidden_rows,
                        prefill_end: Some(end),
                    })
                }
                Phase::Plain { pending } => Some(Plan {
                    id: *id,
                    seq: s.seq,
                    tokens: vec![*pending],
                    logit_rows: vec![0],
                    hidden_rows: Vec::new(),
                    prefill_end: None,
                }),
                Phase::Spec {
                    pending,
                    draft_token,
                    ..
                } => {
                    if used + 2 > MAX_M as usize {
                        break;
                    }
                    Some(Plan {
                        id: *id,
                        seq: s.seq,
                        tokens: vec![*pending, *draft_token],
                        logit_rows: vec![0, 1],
                        hidden_rows: vec![0, 1],
                        prefill_end: None,
                    })
                }
            };
            if let Some(entry) = entry {
                used += entry.tokens.len();
                plan.push(entry);
            }
        }
        Ok(plan)
    }

    pub fn poll(&mut self, id: SessionId) -> Vec<Piece> {
        let Some(s) = self.sessions.get_mut(&id.0) else {
            return Vec::new();
        };
        s.queue.drain(..).collect()
    }

    pub fn finished(&self, id: SessionId) -> bool {
        self.sessions
            .get(&id.0)
            .map(|s| matches!(s.phase, Phase::Done) && s.queue.is_empty())
            .unwrap_or(true)
    }

    pub fn stats(&self, id: SessionId) -> Option<GenStats> {
        self.sessions.get(&id.0).map(|s| {
            let mut stats = GenStats {
                prefill_tokens: s.stats.prefill_tokens,
                decode_tokens: s.stats.decode_tokens,
                accepted: s.stats.accepted,
                prefill_secs: s.stats.prefill_secs,
                decode_secs: s.stats.decode_secs,
            };
            if let Some(t) = s.decode_start {
                stats.decode_secs = t.elapsed().as_secs_f64();
            }
            stats
        })
    }

    pub fn close(&mut self, id: SessionId) -> Result<()> {
        let Some(s) = self.sessions.remove(&id.0) else {
            return Ok(());
        };
        self.model.free_pages(&self.backend, s.seq)?;
        self.free_seqs.push(s.seq);
        Ok(())
    }
}

struct Plan {
    id: u32,
    seq: u32,
    tokens: Vec<u32>,
    logit_rows: Vec<u32>,
    hidden_rows: Vec<u32>,
    prefill_end: Option<usize>,
}

struct Resolver<'a> {
    backend: &'a mut Backend,
    model: &'a mut Box<dyn LanguageModel>,
    tokenizer: &'a Tokenizer,
    trie: &'a Option<TokenTrie>,
    speculate: bool,
}

impl Resolver<'_> {
    fn resolve(&mut self, plan: &Plan, out: &ChunkOut, s: &mut Session) -> Result<()> {
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
            let draft_dist = dist_after(self.trie, s, pending, &draft_logits)?;
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
        let target_dist = dist_after(self.trie, s, pending, &out.logits[0])?;
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
            let bonus_dist = dist(self.trie, s, &out.logits[1])?;
            let bonus = s.sampler.draw(&bonus_dist);
            let spec = self
                .model
                .speculator()
                .expect("spec phase implies a speculator");
            spec.advance(self.backend, s.seq, draft_token, &out.hidden[0])?;
            let draft_logits = spec.draft(self.backend, s.seq, bonus, &out.hidden[1])?;
            let next_dist = dist_after(self.trie, s, bonus, &draft_logits)?;
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
        let next_dist = dist_after(self.trie, s, chosen, &draft_logits)?;
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

fn dist(trie: &Option<TokenTrie>, s: &mut Session, logits: &[f32]) -> Result<Dist> {
    let mask = grammar_mask(trie, s)?;
    Ok(s.sampler.transform(logits, &s.context, mask.as_deref()))
}

fn dist_after(
    trie: &Option<TokenTrie>,
    s: &mut Session,
    prev: u32,
    logits: &[f32],
) -> Result<Dist> {
    s.context.push(prev);
    let d = dist(trie, s, logits);
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
