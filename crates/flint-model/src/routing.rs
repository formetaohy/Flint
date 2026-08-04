//! CPU-side expert routing: softmax over router logits, top-k selection and
//! per-expert pair lists the GPU kernels consume. Pairs are sorted by expert
//! so each expert's rows are contiguous, giving fixed gather/scatter ranges.

/// How router logits become routing weights.
#[derive(Clone, Copy, Debug)]
pub enum RouteKind {
    /// Plain softmax over all experts; top-k weights are the softmax values.
    Softmax,
    /// Phi-MoE's sparsemixer: softmax over the experts within 2 * jitter of
    /// the max score, top-1 selected, then a re-softmax over the remainder
    /// for top-2 (transformers' `sparsemixer`, eval path).
    SparseMixer { jitter: f32 },
}

/// One forward's routing result: pair lists sorted by expert id, with the
/// shared expert (when present) as a final virtual expert covering every row.
/// Each expert's range starts at a 64-element boundary so the range binds
/// directly as a storage-buffer slice.
#[derive(Debug)]
pub struct Routing {
    /// Aligned start index of each expert's pairs (length experts + 2; the
    /// virtual shared expert sits at `experts`).
    pub starts: Vec<u32>,
    /// Pair count per expert (length experts + 1).
    pub counts: Vec<u32>,
    /// Row id per pair, contiguous per expert.
    pub rows: Vec<u32>,
    /// Routing weight per pair.
    pub weights: Vec<f32>,
}

impl Routing {
    /// Builds the routing for `m` rows over `experts` experts, appending the
    /// shared expert as a virtual expert covering every row.
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

        // (expert, row, weight) pairs, one per (row, slot).
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
        // Counting sort by expert; ties keep row order (stable). Each range
        // end pads to a 64-element boundary so the next expert's slice
        // satisfies the binding alignment. The virtual shared expert (when
        // active) covers every row at a constant weight.
        if shared_scale != 0.0 {
            counts[experts] = m as u32;
        }
        for (e, _, _) in &pairs {
            counts[*e] += 1;
        }
        let mut starts = vec![0u32; experts + 2];
        for e in 0..=experts {
            // Every expert advances at least one slot so zero-count experts
            // cannot collapse two non-empty ranges onto the same boundary.
            starts[e + 1] = align64(starts[e] + counts[e].max(1));
        }
        let mut slots = starts[..experts + 1].to_vec();
        let mut sorted = vec![(0usize, 0usize, 0f32); starts[experts + 1] as usize];
        for (e, r, w) in pairs {
            sorted[slots[e] as usize] = (e, r, w);
            slots[e] += 1;
        }
        // The pair buffers keep the padded layout so each expert's range
        // binds as a slice at its aligned start; inter-range padding stays 0.
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
        // The shared expert's slot was never counted into `pairs`; fill it
        // with the full row sweep at its constant weight.
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

    /// Pairs count of expert e (the shared expert's virtual id is `experts`).
    pub fn count(&self, e: usize) -> u32 {
        self.counts[e]
    }

    /// Byte offset of expert e's pair slice in the row/weight buffers.
    pub fn offset(&self, e: usize) -> u64 {
        self.starts[e] as u64 * 4
    }
}

/// Rounds up to the next 64-element (256-byte) boundary.
fn align64(v: u32) -> u32 {
    v.div_ceil(64) * 64
}

/// Softmax over all logits, then the top-k indices with their softmax values.
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

/// Phi-MoE sparsemixer (eval path): softmax over the experts within
/// 2 * jitter of the max score; take the max of that set, then mask it,
/// re-threshold against the new max and re-softmax for the second expert.
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
        // Mask the top-1 out and re-threshold against the new max.
        let mut masked_scores = masked;
        masked_scores[p1.0] = f32::NEG_INFINITY;
        let second_max = masked_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let rest: Vec<f32> = (0..scores.len())
            .map(|i| {
                if i != p1.0 && keep(scores, second_max, i) {
                    // The mask re-evaluates the original scores, so kept
                    // experts carry their original value (the top-1 slot and
                    // pass-1-masked experts are re-derived from `scores`).
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
