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

pub fn expert_gather(x: &[f32], ids: &[u32], rows: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * hidden];
    for (r, &id) in ids.iter().enumerate() {
        out[r * hidden..(r + 1) * hidden]
            .copy_from_slice(&x[id as usize * hidden..(id as usize + 1) * hidden]);
    }
    out
}

pub fn expert_scatter(acc: &mut [f32], src: &[f32], ids: &[u32], weights: &[f32], hidden: usize) {
    for (i, &id) in ids.iter().enumerate() {
        let w = weights[i];
        for c in 0..hidden {
            acc[id as usize * hidden + c] += w * src[i * hidden + c];
        }
    }
}

pub fn zero_rows(x: &mut [f32], n: usize) {
    x[..n].fill(0.0);
}

pub fn sigmoid_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x * (1.0 / (1.0 + (-y).exp())))
        .collect()
}

pub fn concat(a: &[f32], b: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * 2 * d];
    for r in 0..rows {
        out[r * 2 * d..r * 2 * d + d].copy_from_slice(&a[r * d..(r + 1) * d]);
        out[r * 2 * d + d..(r + 1) * 2 * d].copy_from_slice(&b[r * d..(r + 1) * d]);
    }
    out
}

pub fn split_qg(x: &[f32], rows: usize, heads: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0f32; rows * heads * hd];
    let mut g = vec![0f32; rows * heads * hd];
    for m in 0..rows {
        for h in 0..heads {
            for d in 0..hd {
                let base = (m * heads + h) * 2 * hd;
                q[(m * heads + h) * hd + d] = x[base + d];
                g[(m * heads + h) * hd + d] = x[base + hd + d];
            }
        }
    }
    (q, g)
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
    let (m, nq, nkv, hd, max_seq, pos, window) =
        (spec.m, spec.nq, spec.nkv, spec.hd, spec.max_seq, spec.pos, spec.window);
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

pub fn conv1d(x: &[f32], w: &[f32], state: &mut [f32]) -> Vec<f32> {
    let dim = x.len();
    let mut out = vec![0f32; dim];
    for c in 0..dim {
        out[c] = silu(
            w[c * 4] * state[c * 3]
                + w[c * 4 + 1] * state[c * 3 + 1]
                + w[c * 4 + 2] * state[c * 3 + 2]
                + w[c * 4 + 3] * x[c],
        );
        state[c * 3] = state[c * 3 + 1];
        state[c * 3 + 1] = state[c * 3 + 2];
        state[c * 3 + 2] = x[c];
    }
    out
}

pub fn repeat_qk(
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    n_k: usize,
    n_v: usize,
    kd: usize,
    vd: usize,
) {
    let ratio = n_v / n_k;
    let conv_dim = 2 * n_k * kd + n_v * vd;
    let out_dim = 2 * n_v * kd;
    for r in 0..rows {
        for seg in 0..2 {
            for h in 0..n_v {
                for d in 0..kd {
                    out[r * out_dim + seg * n_v * kd + h * kd + d] =
                        x[r * conv_dim + seg * n_k * kd + (h / ratio) * kd + d];
                }
            }
        }
    }
}

pub fn delta_gate(b: &[f32], a: &[f32], a_log: &[f32], dt_bias: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let heads = b.len();
    let mut beta = vec![0f32; heads];
    let mut g = vec![0f32; heads];
    for h in 0..heads {
        beta[h] = 1.0 / (1.0 + (-b[h]).exp());
        g[h] = -a_log[h].exp() * (1.0 + (a[h] + dt_bias[h]).exp()).ln();
    }
    (beta, g)
}

pub struct DeltaRecurArgs {
    pub heads: usize,
    pub kd: usize,
    pub vd: usize,
}

pub fn delta_recur(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    state: &mut [f32],
    spec: DeltaRecurArgs,
) -> Vec<f32> {
    let (heads, kd, vd) = (spec.heads, spec.kd, spec.vd);
    let l2norm = |row: &[f32]| -> Vec<f32> {
        let inv = (row.iter().map(|w| w * w).sum::<f32>() + 1e-6)
            .sqrt()
            .recip();
        row.iter().map(|w| w * inv).collect()
    };
    let scale = (kd as f32).sqrt().recip();
    let mut out = vec![0f32; heads * vd];
    for h in 0..heads {
        let qq: Vec<f32> = l2norm(&q[h * kd..(h + 1) * kd])
            .iter()
            .map(|x| x * scale)
            .collect();
        let kk = l2norm(&k[h * kd..(h + 1) * kd]);
        let decay = g[h].exp();
        let bt = beta[h];
        let base = h * kd * vd;
        for e in 0..kd * vd {
            state[base + e] *= decay;
        }
        let kv_mem: Vec<f32> = (0..vd)
            .map(|vi| (0..kd).map(|ki| state[base + ki * vd + vi] * kk[ki]).sum())
            .collect();
        let delta: Vec<f32> = (0..vd)
            .map(|vi| (v[h * vd + vi] - kv_mem[vi]) * bt)
            .collect();
        for ki in 0..kd {
            for vi in 0..vd {
                state[base + ki * vd + vi] += kk[ki] * delta[vi];
            }
        }
        for vi in 0..vd {
            out[h * vd + vi] = (0..kd).map(|ki| state[base + ki * vd + vi] * qq[ki]).sum();
        }
    }
    out
}
