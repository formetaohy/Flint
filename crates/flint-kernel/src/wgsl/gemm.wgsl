// y[m, n] = sum_k x[m, k] * dequant(w[n, k])
// Tiled GEMM: one workgroup computes a TM=32 x TN=32 output tile with 256
// lanes (16 row-lanes x 16 col-lanes, one 2x2 register block each). The
// small per-lane block keeps register pressure at ~80 regs so 3 workgroups
// stay resident per SM, and the 2x2 block halves shared-memory traffic per
// lane relative to 4x4 (shared reads dominate the inner loop).
//
// Activations stay f32; weights are dequantized to f32 in shared memory per
// K-window of BK=16.
// WDTYPE 1 (i8): w is block-major [K/16, N, 16] i8 viewed as vec4<u32>,
// dequantized with per-row group scales [N, K/GROUP].
// WDTYPE 0 (bf16): w is row-major [N, K/2] packed words.
//
// dispatch [N/TN, M/TM, SEGS]; SEGS splits long-K gemms across the z axis
// (partials merged by merge_gemm).

override N: u32 = 1u;
override K: u32 = 1u;
override M: u32 = 128u;
// K split across the dispatch z axis (partials merged by merge_gemm).
override SEGS: u32 = 1u;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;
// Accumulate into y instead of overwriting: y = y + result.
override ACC: u32 = 0u;
// Column stride and offset of y (fused qkv tiles: three gemms write their
// column ranges of one [m, N_TOTAL] buffer).
override Y_STRIDE: u32 = 1u;
override Y_OFF: u32 = 0u;
// Diagnostics: skip the inner-loop FMA work (stage + barrier only).
override SKIP_COMPUTE: u32 = 0u;

const TM: u32 = 32u;
const TN: u32 = 32u;
const BK: u32 = 16u;
const LANES: u32 = 256u;
// Shared x tile in vec4<f32> units: [TM, BK] f32 (4 vec4 per row).
const X_VECS: u32 = TM * BK / 4u;
// Shared dequantized w tile in vec4<f32> units, layout [TN, 9]: column,
// then 8 value vec4 (two 16-blocks) + 1 padding vec4. Stride 9 spreads the
// 16 column lanes over 8 bank groups (2-way conflict, the mathematical
// minimum at vec4 granularity).
const W_PAD: u32 = 9u;
const W_VECS: u32 = TN * W_PAD;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> xs: array<vec4<f32>, X_VECS>;
var<workgroup> wf: array<vec4<f32>, W_VECS>;

fn bf16f(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

// Sign-extends the 4 i8 bytes of an i8 word.
fn deq_i8(word: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((word & 0xFFu) << 24) >> 24),
        f32(i32((word & 0xFF00u) << 16) >> 24),
        f32(i32((word & 0xFF0000u) << 8) >> 24),
        f32(i32(word >> 24) << 24 >> 24),
    );
}

// Unpacks 4 bf16 values from two packed words.
fn deq_bf16(w0: u32, w1: u32) -> vec4<f32> {
    return vec4<f32>(
        bf16f(w0 & 0xFFFFu),
        bf16f(w0 >> 16),
        bf16f(w1 & 0xFFFFu),
        bf16f(w1 >> 16),
    );
}

// Stages the K-window at global 16-block `step` into shared memory.
// x rows beyond M and w columns beyond N are skipped.
fn stage(step: u32, m0: u32, n0: u32, lane: u32) {
    // x tile: 128 vec4, staged by lanes 0..127 (1 per lane).
    if (lane < 128u) {
        let xr = lane / 4u;
        let ir = lane % 4u;
        if (m0 + xr < M) {
            let xb = (m0 + xr) * (K / 4u) + step * 4u + ir;
            xs[lane] = x[xb];
        }
    }
    // w tile: 32 16-blocks (one per column), staged by lanes 0..31;
    // dequantize into shared f32 so the compute phase is pure FMA.
    if (lane < 32u) {
        let c = lane;
        let wbase = c * W_PAD;
        if (n0 + c < N) {
            if (WDTYPE == 1u) {
                let wb = step * N + n0 + c;
                let s = scales[(n0 + c) * (K / GROUP) + step / (GROUP / 16u)];
                let q = w[wb];
                wf[wbase] = deq_i8(q.x) * s;
                wf[wbase + 1u] = deq_i8(q.y) * s;
                wf[wbase + 2u] = deq_i8(q.z) * s;
                wf[wbase + 3u] = deq_i8(q.w) * s;
            } else {
                let wb = (n0 + c) * (K / 8u) + step * 2u;
                let a = w[wb];
                let b = w[wb + 1u];
                wf[wbase] = deq_bf16(a.x, a.y);
                wf[wbase + 1u] = deq_bf16(a.z, a.w);
                wf[wbase + 2u] = deq_bf16(b.x, b.y);
                wf[wbase + 3u] = deq_bf16(b.z, b.w);
            }
        }
    }
}

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let seg = wg.z;
    let m0 = wg.y * TM;
    let n0 = wg.x * TN;
    // Register block origin: 16 row-lanes x 16 col-lanes of 2x2 cells.
    let r0 = (lane / 16u) * 2u;
    let c0 = (lane % 16u) * 2u;
    // Split-K: this workgroup owns [k_lo, k_lo+ks) of K; partials land in
    // the [SEGS, M, Y_STRIDE] scratch that merge_gemm folds into y.
    let ks = K / SEGS;
    let k_lo = seg * ks;
    let steps = ks / BK;
    let yoff = select(0u, seg * (M * Y_STRIDE), SEGS > 1u);
    let ybase = yoff + (m0 + r0) * Y_STRIDE + Y_OFF + n0 + c0;
    let ystr = Y_STRIDE;

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;

    for (var step: u32 = 0u; step < steps; step += 1u) {
        stage(k_lo / 16u + step, m0, n0, lane);
        workgroupBarrier();
        // Compute the 2x2 block: two rows (8 vec4) x two columns (8 vec4).
        let xbase = r0 * 4u;
        let wb = c0 * W_PAD;
        let x0 = xs[xbase];
        let x1 = xs[xbase + 1u];
        let x2 = xs[xbase + 2u];
        let x3 = xs[xbase + 3u];
        let x4 = xs[xbase + 4u];
        let x5 = xs[xbase + 5u];
        let x6 = xs[xbase + 6u];
        let x7 = xs[xbase + 7u];
        let wv0 = wf[wb];
        let wv1 = wf[wb + 1u];
        let wv2 = wf[wb + 2u];
        let wv3 = wf[wb + 3u];
        let wv4 = wf[wb + W_PAD];
        let wv5 = wf[wb + W_PAD + 1u];
        let wv6 = wf[wb + W_PAD + 2u];
        let wv7 = wf[wb + W_PAD + 3u];
        if (SKIP_COMPUTE == 0u) {
            acc0 += dot(x0, wv0) + dot(x1, wv1) + dot(x2, wv2) + dot(x3, wv3);
            acc1 += dot(x0, wv4) + dot(x1, wv5) + dot(x2, wv6) + dot(x3, wv7);
            acc2 += dot(x4, wv0) + dot(x5, wv1) + dot(x6, wv2) + dot(x7, wv3);
            acc3 += dot(x4, wv4) + dot(x5, wv5) + dot(x6, wv6) + dot(x7, wv7);
        }
        // Staging of the next window must wait for every lane to finish
        // reading this one.
        workgroupBarrier();
    }

    // Write the 2x2 block; rows beyond M and columns beyond N are masked.
    if (m0 + r0 + 0u < M) {
        if (n0 + c0 + 0u < N) {
            y[ybase + 0u] = acc0 + f32(ACC) * y[ybase + 0u];
        }
        if (n0 + c0 + 1u < N) {
            y[ybase + 1u] = acc1 + f32(ACC) * y[ybase + 1u];
        }
    }
    if (m0 + r0 + 1u < M) {
        if (n0 + c0 + 0u < N) {
            y[ybase + ystr + 0u] = acc2 + f32(ACC) * y[ybase + ystr + 0u];
        }
        if (n0 + c0 + 1u < N) {
            y[ybase + ystr + 1u] = acc3 + f32(ACC) * y[ybase + ystr + 1u];
        }
    }
}
