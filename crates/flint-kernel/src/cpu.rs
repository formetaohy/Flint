//! CPU reference implementations of every compute kernel, in plain Rust.
//! They define the intended semantics; the WGPU kernels are tested against
//! them. Activations are row-major `[rows, dim]` tiles, matching the GPU.

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Round-to-bf16 (truncate the low 16 mantissa bits), modelling how the KV
/// cache stores activations so the reference matches the GPU kernels.
fn bf16(v: f32) -> f32 {
    flint_num::bf16_to_f32(flint_num::f32_to_bf16(v))
}

/// y[m, n] = x[m, k] @ w[n, k]^T.
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

/// y[n] = x[k] @ w[n, k]^T (single activation row, the decode fast path).
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

/// Gathers embedding rows: out[r] = scale * table[ids[r]].
pub fn embed(ids: &[u32], table: &[f32], dim: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; ids.len() * dim];
    for (r, &id) in ids.iter().enumerate() {
        for d in 0..dim {
            out[r * dim + d] = table[id as usize * dim + d] * scale;
        }
    }
    out
}

/// Normalization over `dim`. mode 0 = offset weight (1+w), 1 = gated
/// (weight * silu(gate)), 2 = direct weight (w), 3 = layer norm
/// ((x - mean) * inv_std * w + bias, `gate` holds the bias). `w_dim` is the
/// weight length (may repeat across dim).
#[allow(clippy::too_many_arguments)]
pub fn norm(
    mode: u32,
    x: &[f32],
    weight: &[f32],
    gate: &[f32],
    rows: usize,
    dim: usize,
    w_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let mean = match mode {
            3 => row.iter().sum::<f32>() / dim as f32,
            _ => 0.0,
        };
        let inv = match mode {
            3 => {
                let var = mean_sq - mean * mean;
                (var + eps).sqrt().recip()
            }
            _ => (mean_sq + eps).sqrt().recip(),
        };
        for d in 0..dim {
            let base = match mode {
                3 => (row[d] - mean) * inv,
                _ => row[d] * inv,
            };
            out[r * dim + d] = match mode {
                0 => base * (1.0 + weight[d % w_dim]),
                1 => base * weight[d % w_dim] * silu(gate[r * dim + d]),
                3 => base * weight[d % w_dim] + gate[d % gate.len()],
                _ => base * weight[d % w_dim],
            };
        }
    }
    out
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// In-place row-broadcast bias over a [rows, dim] tile.
pub fn bias(x: &mut [f32], bias: &[f32], dim: usize) {
    for (i, v) in x.iter_mut().enumerate() {
        *v += bias[i % dim];
    }
}

/// mode 0 = silu, 1 = gelu (pytorch tanh approximation).
pub fn swiglu(gate: &[f32], up: &[f32], mode: u32) -> Vec<f32> {
    let act = |v: f32| match mode {
        1 => 0.5 * v * (1.0 + (0.7978846 * (v + 0.044715 * v * v * v)).tanh()),
        _ => silu(v),
    };
    gate.iter().zip(up).map(|(g, u)| act(*g) * u).collect()
}

/// Logit softcapping: y[i] = cap * tanh(y[i] / cap), in place.
pub fn softcap(x: &mut [f32], cap: f32) {
    for v in x {
        *v = cap * (*v / cap).tanh();
    }
}

/// Elementwise multiply with broadcast: y[i] = a[i] * b[i % m].
pub fn mul(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32> {
    (0..n).map(|i| a[i] * b[i % m]).collect()
}

/// Copies COUNT rows of x [M_MAX, HIDDEN] selected by ids into out rows 0..COUNT.
pub fn expert_gather(x: &[f32], ids: &[u32], rows: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * hidden];
    for (r, &id) in ids.iter().enumerate() {
        out[r * hidden..(r + 1) * hidden]
            .copy_from_slice(&x[id as usize * hidden..(id as usize + 1) * hidden]);
    }
    out
}

/// MoE weighted accumulation: acc[ids[i]] += weights[i] * src[i] over packed rows.
pub fn expert_scatter(acc: &mut [f32], src: &[f32], ids: &[u32], weights: &[f32], hidden: usize) {
    for (i, &id) in ids.iter().enumerate() {
        let w = weights[i];
        for c in 0..hidden {
            acc[id as usize * hidden + c] += w * src[i * hidden + c];
        }
    }
}

/// Zeroes the first n elements.
pub fn zero_rows(x: &mut [f32], n: usize) {
    x[..n].fill(0.0);
}

pub fn sigmoid_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x * (1.0 / (1.0 + (-y).exp())))
        .collect()
}

/// Concatenates two [rows, d] tiles along the last dim -> [rows, 2d].
pub fn concat(a: &[f32], b: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * 2 * d];
    for r in 0..rows {
        out[r * 2 * d..r * 2 * d + d].copy_from_slice(&a[r * d..(r + 1) * d]);
        out[r * 2 * d + d..(r + 1) * 2 * d].copy_from_slice(&b[r * d..(r + 1) * d]);
    }
    out
}

/// Splits interleaved [rows, heads, 2*hd] into q and gate [rows, heads, hd].
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

/// In-place multi-row RoPE. cos/sin are [max_seq, rot/2]; only the first `rot`
/// dims of each head rotate. x is [m, heads, hd] starting at position `pos`.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    m: usize,
    heads: usize,
    hd: usize,
    rot: usize,
    pos: usize,
) {
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

/// Writes m rows of src [m, kv_heads, hd] into cache [kv_heads, max_seq, hd] at pos.
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

/// Multi-row causal GQA attention over a [kv_heads, max_seq, hd] KV cache.
/// q is [m, nq, hd]; output [m, nq, hd]. `window` > 0 restricts each query to
/// the trailing `window` keys (sliding window); 0 attends to the full prefix.
#[allow(clippy::too_many_arguments)]
pub fn attn(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    max_seq: usize,
    pos: usize,
    window: usize,
) -> Vec<f32> {
    let scale = (hd as f32).sqrt().recip();
    let mut out = vec![0f32; m * nq * hd];
    for mi in 0..m {
        let kv_len = pos + mi + 1;
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

/// Depthwise causal conv1d with a 3-tap ring state, per channel SiLU. One time
/// step: out[c] = silu(w0*s0 + w1*s1 + w2*s2 + w3*x[c]); state shifts to [s1,s2,x].
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

/// Expands the q/k segments of a conv tile from N_K key heads to N_V value
/// heads (repeat_interleave), matching the layout delta_recur consumes.
#[allow(clippy::too_many_arguments)]
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

/// Computes per-head beta = sigmoid(b) and g = -exp(A_log) * ln(1 + exp(a + dt)).
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

/// One Gated DeltaNet recurrence step per head. q/k are L2-normalized, q scaled
/// by 1/sqrt(key_dim). state [heads, kd, vd] is updated in place; returns
/// output [heads, vd].
#[allow(clippy::too_many_arguments)]
pub fn delta_recur(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    state: &mut [f32],
    heads: usize,
    kd: usize,
    vd: usize,
) -> Vec<f32> {
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
