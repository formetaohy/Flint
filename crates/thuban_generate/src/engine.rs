use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use thuban_backend::Backend;
use thuban_error::{Error, Result};
use thuban_model::{LanguageModel, MAX_M, SeqChunk};
use thuban_tokenizer::{StreamDecoder, Tokenizer};

use crate::grammar::{Grammar, Matcher, TokenTrie};
use crate::resolver::{Plan, Resolver};
use crate::sampler::{Dist, Sampler, SamplingParams};

pub struct Piece {
    pub token: u32,
    pub text: String,
}

#[derive(Clone, Copy)]
pub struct GenStats {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub accepted: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SessionId(pub u32);

pub(crate) enum Phase {
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

pub(crate) struct Session {
    pub(crate) seq: u32,
    pub(crate) sampler: Sampler,
    pub(crate) decoder: StreamDecoder,
    pub(crate) context: Vec<u32>,
    pub(crate) prompt: Vec<u32>,
    pub(crate) max_tokens: usize,
    pub(crate) stop: Vec<u32>,
    pub(crate) phase: Phase,
    pub(crate) queue: VecDeque<Piece>,
    pub(crate) stats: GenStats,
    pub(crate) prefill_start: Instant,
    pub(crate) decode_start: Option<Instant>,
    pub(crate) matcher: Option<Matcher>,
}

pub struct Engine {
    backend: Backend,
    model: Box<dyn LanguageModel + Send>,
    tokenizer: Tokenizer,
    sampling: SamplingParams,
    seed: u64,
    pub(crate) stop: Vec<u32>,
    speculate: bool,
    free_seqs: Vec<u32>,
    next_id: u32,
    sessions: HashMap<u32, Session>,
    trie: Option<TokenTrie>,
}

impl Engine {
    pub fn new(
        backend: Backend,
        model: Box<dyn LanguageModel + Send>,
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
        sampling: Option<SamplingParams>,
        stop_extra: &[u32],
    ) -> Result<SessionId> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        if prompt_ids.is_empty() {
            return Err(Error::Tokenizer("empty prompt".into()));
        }
        let matcher = match grammar {
            Some(g) => {
                if self.trie.is_none() {
                    let vocab = self.model.vocab() as usize;
                    self.trie = Some(TokenTrie::from_tokenizer(&self.tokenizer, vocab));
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
        let mut stop = self.stop.clone();
        for &t in stop_extra {
            if !stop.contains(&t) {
                stop.push(t);
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(
            id,
            Session {
                seq,
                sampler: Sampler::new(sampling.unwrap_or(self.sampling), self.seed),
                decoder: self.tokenizer.stream_decoder(),
                context: Vec::new(),
                prompt: prompt_ids,
                max_tokens,
                stop,
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
                    let logit_rows: Vec<u32> =
                        last.then_some(take as u32 - 1).into_iter().collect();
                    let hidden_rows: Vec<u32> = (last && speculate)
                        .then_some(take as u32 - 1)
                        .into_iter()
                        .collect();
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
