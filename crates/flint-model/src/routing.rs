#[derive(Clone, Copy, Debug)]
pub enum RouteKind {
    Softmax,

    SparseMixer { jitter: f32 },
}

#[derive(Debug)]
pub struct Routing {
    pub starts: Vec<u32>,

    pub counts: Vec<u32>,

    pub rows: Vec<u32>,

    pub weights: Vec<f32>,
}

impl Routing {
    pub fn new(
        logits: &[f32],
        m: u32,
        experts: u32,
        top_k: u32,
        kind: RouteKind,
        shared_scale: f32,
    ) -> Self {
        let m = m as usize;
        let experts = experts as usize;
        let top_k = top_k as usize;
        let mut counts = vec![0u32; experts + 2];

        let mut pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(m * top_k);
        for r in 0..m {
            let logits = &logits[r * experts..(r + 1) * experts];
            let sel = match kind {
                RouteKind::Softmax => softmax_topk(logits, top_k),
                RouteKind::SparseMixer { jitter } => sparse_mixer(logits, top_k, jitter),
            };
            for (e, w) in sel {
                pairs.push((e, r, w));
            }
        }

        if shared_scale != 0.0 {
            counts[experts] = m as u32;
        }
        for (e, _, _) in &pairs {
            counts[*e] += 1;
        }
        let mut starts = vec![0u32; experts + 2];
        for e in 0..=experts {
            starts[e + 1] = align64(starts[e] + counts[e].max(1));
        }
        let mut slots = starts[..experts + 1].to_vec();
        let mut sorted = vec![(0usize, 0usize, 0f32); starts[experts + 1] as usize];
        for (e, r, w) in pairs {
            sorted[slots[e] as usize] = (e, r, w);
            slots[e] += 1;
        }

        let mut rows = vec![0u32; starts[experts + 1] as usize];
        let mut weights = vec![0f32; starts[experts + 1] as usize];
        for e in 0..=experts {
            let base = starts[e] as usize;
            for k in 0..counts[e] as usize {
                let (_, r, w) = sorted[base + k];
                rows[base + k] = r as u32;
                weights[base + k] = w;
            }
        }

        if shared_scale != 0.0 {
            let base = starts[experts] as usize;
            for (k, r) in (0..m).enumerate() {
                rows[base + k] = r as u32;
                weights[base + k] = shared_scale;
            }
        }
        Self {
            starts,
            counts,
            rows,
            weights,
        }
    }

    pub fn count(&self, e: usize) -> u32 {
        self.counts[e]
    }

    pub fn offset(&self, e: usize) -> u64 {
        self.starts[e] as u64 * 4
    }
}

fn align64(v: u32) -> u32 {
    v.div_ceil(64) * 64
}

fn softmax_topk(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|v| (v - mx).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, v)| (i, (v - mx).exp() / sum))
        .collect();
    probs.sort_by(|a, b| b.1.total_cmp(&a.1));
    probs.truncate(top_k);
    probs
}

fn sparse_mixer(scores: &[f32], top_k: usize, jitter: f32) -> Vec<(usize, f32)> {
    let keep = |scores: &[f32], max: f32, i: usize| -> bool {
        let factor = scores[i].abs().max(max);
        !(factor > 0.0 && (max - scores[i]) / factor > 2.0 * jitter)
    };
    let first_max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let masked: Vec<f32> = (0..scores.len())
        .map(|i| {
            if keep(scores, first_max, i) {
                scores[i]
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let p1 = softmax_topk(&masked, 1)[0];
    let mut sel = vec![p1];
    if top_k > 1 {
        let mut masked_scores = masked;
        masked_scores[p1.0] = f32::NEG_INFINITY;
        let second_max = masked_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let rest: Vec<f32> = (0..scores.len())
            .map(|i| {
                if i != p1.0 && keep(scores, second_max, i) {
                    scores[i]
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        sel.push(softmax_topk(&rest, 1)[0]);
    }
    sel
}
