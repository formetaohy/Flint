use flint_kernel::{Act, NormMode};

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn bf16(v: f32) -> f32 {
    flint_num::bf16_to_f32(flint_num::f32_to_bf16(v))
}

pub fn gemm(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0f32;
            for ki in 0..k {
                s += x[mi * k + ki] * w[ni * k + ki];
            }
            y[mi * n + ni] = s;
        }
    }
    y
}

pub fn gemv(x: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    for ni in 0..n {
        let mut s = 0f32;
        for ki in 0..k {
            s += x[ki] * w[ni * k + ki];
        }
        y[ni] = s;
    }
    y
}

pub fn embed(ids: &[u32], table: &[f32], dim: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; ids.len() * dim];
    for (r, &id) in ids.iter().enumerate() {
        for d in 0..dim {
            out[r * dim + d] = table[id as usize * dim + d] * scale;
        }
    }
    out
}

pub struct NormArgs {
    pub rows: usize,
    pub dim: usize,
    pub w_dim: usize,
    pub eps: f32,
}

pub fn norm(mode: NormMode, x: &[f32], weight: &[f32], gate: &[f32], spec: NormArgs) -> Vec<f32> {
    let (rows, dim, w_dim, eps, stride) = (spec.rows, spec.dim, spec.w_dim, spec.eps, spec.dim);
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * stride..r * stride + dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let mean = match mode {
            NormMode::Layer => row.iter().sum::<f32>() / dim as f32,
            _ => 0.0,
        };
        let inv = match mode {
            NormMode::Layer => {
                let var = mean_sq - mean * mean;
                (var + eps).sqrt().recip()
            }
            _ => (mean_sq + eps).sqrt().recip(),
        };
        for d in 0..dim {
            let base = match mode {
                NormMode::Layer => (row[d] - mean) * inv,
                _ => row[d] * inv,
            };
            out[r * dim + d] = match mode {
                NormMode::Offset => base * (1.0 + weight[d % w_dim]),
                NormMode::Gated => base * weight[d % w_dim] * silu(gate[r * dim + d]),
                NormMode::Direct => base * weight[d % w_dim],
                NormMode::Layer => base * weight[d % w_dim] + gate[d % gate.len()],
            };
        }
    }
    out
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

pub fn bias(x: &mut [f32], bias: &[f32], dim: usize) {
    for (i, v) in x.iter_mut().enumerate() {
        *v += bias[i % dim];
    }
}

pub fn swiglu(gate: &[f32], up: &[f32], mode: Act) -> Vec<f32> {
    let act = |v: f32| match mode {
        Act::GeluTanh => 0.5 * v * (1.0 + (0.7978846 * (v + 0.044715 * v * v * v)).tanh()),
        Act::Silu => silu(v),
    };
    gate.iter().zip(up).map(|(g, u)| act(*g) * u).collect()
}

pub fn softcap(x: &mut [f32], cap: f32) {
    for v in x {
        *v = cap * (*v / cap).tanh();
    }
}

pub fn mul(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32> {
    (0..n).map(|i| a[i] * b[i % m]).collect()
}

pub fn concat(a: &[f32], b: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * 2 * d];
    for r in 0..rows {
        out[r * 2 * d..r * 2 * d + d].copy_from_slice(&a[r * d..(r + 1) * d]);
        out[r * 2 * d + d..(r + 1) * 2 * d].copy_from_slice(&b[r * d..(r + 1) * d]);
    }
    out
}

pub struct RopeArgs {
    pub m: usize,
    pub heads: usize,
    pub hd: usize,
    pub rot: usize,
    pub pos: usize,
}

pub fn rope(x: &mut [f32], cos: &[f32], sin: &[f32], spec: RopeArgs) {
    let (m, heads, hd, rot, pos) = (spec.m, spec.heads, spec.hd, spec.rot, spec.pos);
    let half = rot / 2;
    let orig = x.to_vec();
    for mi in 0..m {
        for h in 0..heads {
            for d in 0..rot {
                let t = (pos + mi) * half + d % half;
                let c = cos[t];
                let s = sin[t];
                let base = (mi * heads + h) * hd;
                x[base + d] = if d < half {
                    orig[base + d] * c - orig[base + d + half] * s
                } else {
                    orig[base + d] * c + orig[base + d - half] * s
                };
            }
        }
    }
}

pub fn kv_store(
    src: &[f32],
    cache: &mut [f32],
    m: usize,
    kv_heads: usize,
    hd: usize,
    max_seq: usize,
    pos: usize,
) {
    for mi in 0..m {
        for h in 0..kv_heads {
            for d in 0..hd {
                cache[(h * max_seq + pos + mi) * hd + d] = bf16(src[(mi * kv_heads + h) * hd + d]);
            }
        }
    }
}

pub struct AttnArgs {
    pub m: usize,
    pub nq: usize,
    pub nkv: usize,
    pub hd: usize,
    pub max_seq: usize,
    pub pos: usize,
    pub window: usize,
    pub causal: bool,
}

pub fn attn(q: &[f32], k_cache: &[f32], v_cache: &[f32], spec: AttnArgs) -> Vec<f32> {
    let (m, nq, nkv, hd, max_seq, pos, window) = (
        spec.m,
        spec.nq,
        spec.nkv,
        spec.hd,
        spec.max_seq,
        spec.pos,
        spec.window,
    );
    let scale = (hd as f32).sqrt().recip();
    let mut out = vec![0f32; m * nq * hd];
    for mi in 0..m {
        let kv_len = if spec.causal { pos + mi + 1 } else { pos + m };
        let win_start = if window != 0 && kv_len > window {
            kv_len - window
        } else {
            0
        };
        for h in 0..nq {
            let kvh = h / (nq / nkv);
            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for t in win_start..kv_len {
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += q[(mi * nq + h) * hd + d] * bf16(k_cache[(kvh * max_seq + t) * hd + d]);
                }
                scores[t] = dot * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for d in 0..hd {
                let mut o = 0f32;
                for t in 0..kv_len {
                    o += scores[t] / sum * bf16(v_cache[(kvh * max_seq + t) * hd + d]);
                }
                out[(mi * nq + h) * hd + d] = o;
            }
        }
    }
    out
}
