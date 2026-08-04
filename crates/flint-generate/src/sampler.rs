//! Sampling pipeline: temperature, top-k, nucleus (top-p), min-p and a
//! repetition penalty, plus speculative-draft verification.
//!
//! `transform` is the single place raw logits become a sampling distribution,
//! and every consumer draws and compares only transformed distributions.
//! Speculative decoding is then exact (Leviathan et al., arXiv:2211.17192).

/// A processed sampling distribution — the only thing tokens are drawn from
/// and verification compares.
#[derive(Clone, Debug, PartialEq)]
pub enum Dist {
    /// temperature <= 0: deterministic argmax of the penalized logits.
    Greedy(u32),
    /// temperature > 0: probabilities after repeat penalty, temperature,
    /// top-k, top-p and min-p, renormalized to sum 1.
    Probs(Vec<f32>),
}

/// Knobs for one generation run. Defaults match common chat inference.
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    /// <= 0 selects greedy argmax and disables every random transform.
    pub temperature: f32,
    /// Keep only the top-k logits; 0 disables.
    pub top_k: usize,
    /// Nucleus: smallest set whose cumulative mass reaches top_p; 1.0 disables.
    pub top_p: f32,
    /// Drop tokens below `min_p * max_prob`; 0.0 disables.
    pub min_p: f32,
    /// Multiplicative repetition penalty (1.0 disables).
    pub repeat_penalty: f32,
    /// How many of the most recent context tokens the penalty sees.
    pub repeat_last_n: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: u64,
}

impl Sampler {
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self { params, rng: seed }
    }

    pub fn greedy(seed: u64) -> Self {
        Self {
            params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            rng: seed,
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Applies the full sampling configuration to raw logits, yielding the
    /// distribution tokens are drawn from. Pure: identical inputs yield
    /// identical distributions.
    pub fn transform(&self, logits: &[f32], context: &[u32]) -> Dist {
        let p = self.params;
        let mut scores = logits.to_vec();
        apply_repeat_penalty(&mut scores, context, p.repeat_penalty, p.repeat_last_n);
        if p.temperature <= 0.0 {
            return Dist::Greedy(argmax(&scores));
        }
        let probs = softmax(&scores, p.temperature);
        let mut kept: Vec<(f32, u32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        kept.sort_by(|a, b| b.0.total_cmp(&a.0));
        apply_top_k(&mut kept, p.top_k);
        apply_top_p(&mut kept, p.top_p);
        apply_min_p(&mut kept, p.min_p);

        // Zero everything outside the kept set and renormalize, so the
        // distribution is dense over the full vocab again.
        let mut out = vec![0.0f32; probs.len()];
        let mut sum = 0.0f32;
        for &(v, id) in &kept {
            out[id as usize] = v;
            sum += v;
        }
        for v in &mut out {
            *v /= sum;
        }
        Dist::Probs(out)
    }

    /// Draws one token from a transformed distribution.
    pub fn draw(&mut self, dist: &Dist) -> u32 {
        match dist {
            Dist::Greedy(token) => *token,
            Dist::Probs(probs) => {
                let target = self.next_f32();
                let mut acc = 0.0f32;
                for (i, v) in probs.iter().enumerate() {
                    acc += *v;
                    if acc >= target {
                        return i as u32;
                    }
                }
                (probs.len() - 1) as u32
            }
        }
    }

    /// Speculative verification: accepts the draft token with probability
    /// min(1, pt/pd); on rejection resamples from norm(max(0, pt - pd)).
    /// Under that rule the committed token follows pt exactly. Greedy runs
    /// accept iff the draft matches the target argmax.
    pub fn verify(&mut self, target: &Dist, draft: &Dist, draft_token: u32) -> (bool, u32) {
        match (target, draft) {
            (Dist::Greedy(t), Dist::Greedy(_)) => (*t == draft_token, *t),
            (Dist::Probs(pt), Dist::Probs(pd)) => {
                let d = draft_token as usize;
                if self.next_f32() * pd[d] < pt[d] {
                    return (true, draft_token);
                }
                let sum: f32 = pt.iter().zip(pd).map(|(&a, &b)| (a - b).max(0.0)).sum();
                let draw = self.next_f32() * sum;
                let mut acc = 0.0f32;
                for (i, (&a, &b)) in pt.iter().zip(pd).enumerate() {
                    acc += (a - b).max(0.0);
                    if acc >= draw {
                        return (false, i as u32);
                    }
                }
                (false, (pt.len() - 1) as u32)
            }
            // The sampler's mode is fixed for the whole run, so target and
            // draft distributions always agree on their variant.
            _ => unreachable!("greedy and stochastic distributions never mix"),
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.rng = self.rng.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Multiplicative repetition penalty over the last_n context tokens: positive
/// logits divided, negative multiplied, so repeats are discouraged
/// symmetrically around zero.
pub fn apply_repeat_penalty(scores: &mut [f32], context: &[u32], penalty: f32, last_n: usize) {
    if penalty == 1.0 || context.is_empty() {
        return;
    }
    let start = context.len().saturating_sub(last_n);
    for &tok in &context[start..] {
        let Some(s) = scores.get_mut(tok as usize) else {
            continue;
        };
        if *s > 0.0 {
            *s /= penalty;
        } else {
            *s *= penalty;
        }
    }
}

pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature;
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut out: Vec<f32> = logits.iter().map(|&v| ((v - max) * inv_t).exp()).collect();
    let sum: f32 = out.iter().sum();
    for v in &mut out {
        *v /= sum;
    }
    out
}

/// candidates must be sorted by descending probability.
fn apply_top_k(c: &mut Vec<(f32, u32)>, k: usize) {
    if k > 0 && k < c.len() {
        c.truncate(k);
    }
}

fn apply_top_p(c: &mut Vec<(f32, u32)>, p: f32) {
    if p >= 1.0 {
        return;
    }
    let sum: f32 = c.iter().map(|(v, _)| *v).sum();
    let mut acc = 0.0f32;
    let mut cutoff = c.len();
    for (i, (v, _)) in c.iter().enumerate() {
        acc += *v / sum;
        if acc >= p {
            cutoff = i + 1;
            break;
        }
    }
    c.truncate(cutoff.max(1));
}

fn apply_min_p(c: &mut Vec<(f32, u32)>, min_p: f32) {
    if min_p <= 0.0 || c.is_empty() {
        return;
    }
    let max_prob = c[0].0;
    let threshold = min_p * max_prob;
    let cutoff = c
        .iter()
        .position(|(v, _)| *v < threshold)
        .unwrap_or(c.len());
    c.truncate(cutoff.max(1));
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}
