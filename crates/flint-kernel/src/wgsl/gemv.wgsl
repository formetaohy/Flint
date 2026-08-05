// y[n] = sum_k x[k] * dequant(w[n, k])   (single activation row, M == 1)
// x: f32 [K] viewed as vec4, y: f32 [N]; dispatch [N/BN, SEGS, 1].
// N multiple of 16, K multiple of 16, SEGS in {1, 2, 4, 8}, K % SEGS == 0.
//
// WDTYPE 1 (i8): w is block-major [K/16, N, 16] i8 bytes viewed as vec4<u32>
// (one vec4 per (k-block, column) tile), dequantized with per-row group
// scales [N, K/GROUP]. Each lane owns one whole 16-k block of one column per
// iteration; a column's block stream is contiguous, so every fetched byte is
// used and short K segments never idle lanes.
// WDTYPE 0 (bf16): w stays row-major [N, K/2] packed words; a lane owns 16
// consecutive k of all 16 columns per iteration (two vec4s per column).
//
// Split-K: workgroup (n0, s) streams columns n0..n0+15 over K segment s.
// i8: column partials live on 8 lanes (stride 16), tree-reduced in 3 rounds;
// bf16: 128 lanes hold 16 column partials, reduced in 7 rounds.

override N: u32 = 1u;
override K: u32 = 1u;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;
override SEGS: u32 = 1u;
// Accumulate into y instead of overwriting: y = y + result.
override ACC: u32 = 0u;

const BN: u32 = 16u;
const LANES: u32 = 128u;
// k blocks per column per iteration (LANES / BN lanes span the columns).
const KB_PER_IT: u32 = LANES / BN;
// Reduction padding, coprime with 32 banks.
const SR: u32 = 17u;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, LANES * SR>;

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

// Column stride of a row-major bf16 weight in vec4<u32> units (K/2 words
// per row: K/8 vec4s).
fn bf16_wstride() -> u32 {
    return K / 8u;
}

// First vec4<u32> of row-major bf16 column n's 16-k block starting at k.
fn bf16_wbase(n: u32, k: u32) -> u32 {
    return (n * K + k) / 8u;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let n0 = wg.x * BN;
    let seg = wg.y;
    let ns = K / GROUP;

    if (WDTYPE == 1u) {
        // ---- i8 block-major path: two blocks (2 x 16 k x 1 column) per
        // lane per iteration, so two weight vec4 loads stay in flight
        // instead of stalling on one DRAM round trip per step. ----
        let c = lane % BN;
        let kb_rel = lane / BN;
        let seg_kb = (K / SEGS) / 16u;
        let kb_lo = seg * seg_kb;
        let iters = (seg_kb + 31u) / 32u;
        let lim = kb_lo + seg_kb;

        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;
        for (var it: u32 = 0u; it < iters; it += 1u) {
            let kb0 = kb_lo + it * 32u + kb_rel;
            if (kb0 >= lim) {
                break;
            }
            let kb1 = kb0 + 8u;
            let kb2 = kb0 + 16u;
            let kb3 = kb0 + 24u;
            let xb = kb0 * 4u;
            let xv0 = x[xb];
            let xv1 = x[xb + 1u];
            let xv2 = x[xb + 2u];
            let xv3 = x[xb + 3u];
            let w4 = w[kb0 * N + n0 + c];
            let sc = scales[(n0 + c) * ns + kb0 / (GROUP / 16u)];
            acc0 += dot(xv0, deq_i8(w4.x) * sc) + dot(xv1, deq_i8(w4.y) * sc)
                + dot(xv2, deq_i8(w4.z) * sc) + dot(xv3, deq_i8(w4.w) * sc);
            if (kb1 < lim) {
                let xb1 = kb1 * 4u;
                let xv4 = x[xb1];
                let xv5 = x[xb1 + 1u];
                let xv6 = x[xb1 + 2u];
                let xv7 = x[xb1 + 3u];
                let w4b = w[kb1 * N + n0 + c];
                let scb = scales[(n0 + c) * ns + kb1 / (GROUP / 16u)];
                acc1 += dot(xv4, deq_i8(w4b.x) * scb) + dot(xv5, deq_i8(w4b.y) * scb)
                    + dot(xv6, deq_i8(w4b.z) * scb) + dot(xv7, deq_i8(w4b.w) * scb);
            }
            if (kb2 < lim) {
                let xb2 = kb2 * 4u;
                let xv8 = x[xb2];
                let xv9 = x[xb2 + 1u];
                let xv10 = x[xb2 + 2u];
                let xv11 = x[xb2 + 3u];
                let w4c = w[kb2 * N + n0 + c];
                let scc = scales[(n0 + c) * ns + kb2 / (GROUP / 16u)];
                acc2 += dot(xv8, deq_i8(w4c.x) * scc) + dot(xv9, deq_i8(w4c.y) * scc)
                    + dot(xv10, deq_i8(w4c.z) * scc) + dot(xv11, deq_i8(w4c.w) * scc);
            }
            if (kb3 < lim) {
                let xb3 = kb3 * 4u;
                let xv12 = x[xb3];
                let xv13 = x[xb3 + 1u];
                let xv14 = x[xb3 + 2u];
                let xv15 = x[xb3 + 3u];
                let w4d = w[kb3 * N + n0 + c];
                let scd = scales[(n0 + c) * ns + kb3 / (GROUP / 16u)];
                acc3 += dot(xv12, deq_i8(w4d.x) * scd) + dot(xv13, deq_i8(w4d.y) * scd)
                    + dot(xv14, deq_i8(w4d.z) * scd) + dot(xv15, deq_i8(w4d.w) * scd);
            }
        }
        var acc = acc0 + acc1 + acc2 + acc3;

        // Tree-reduce the 8 lanes sharing each column: stride 64, 32, 16
        // (descending) so every partial lands in lanes 0..15 (ascending
        // strides would strand lanes 48/80/96/112).
        red[lane] = acc;
        workgroupBarrier();
        var stride = 64u;
        loop {
            if (stride < 16u) {
                break;
            }
            if (lane < stride) {
                red[lane] += red[lane + stride];
            }
            workgroupBarrier();
            stride >>= 1u;
        }
        if (lane < BN) {
            if (SEGS == 1u) {
                y[n0 + lane] = red[lane] + f32(ACC) * y[n0 + lane];
            } else {
                // Segments always overwrite their partial slot; the merge
                // applies ACC when folding them into y.
                y[seg * N + n0 + lane] = red[lane];
            }
        }
    } else {
        // ---- bf16 row-major path: 16 k x 16 columns per lane. ----
        let seg_len = K / SEGS;
        let k_lo = seg * seg_len;
        let k_hi = k_lo + seg_len;
        let iters = (seg_len + LANES * 16u - 1u) / (LANES * 16u);

        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;
        var acc4 = 0.0;
        var acc5 = 0.0;
        var acc6 = 0.0;
        var acc7 = 0.0;
        var acc8 = 0.0;
        var acc9 = 0.0;
        var acc10 = 0.0;
        var acc11 = 0.0;
        var acc12 = 0.0;
        var acc13 = 0.0;
        var acc14 = 0.0;
        var acc15 = 0.0;

        for (var it: u32 = 0u; it < iters; it += 1u) {
            let k0 = k_lo + it * LANES * 16u + lane * 16u;
            if (k0 >= k_hi) {
                break;
            }
            let xb = k0 / 4u;
            let xv0 = x[xb];
            let xv1 = x[xb + 1u];
            let xv2 = x[xb + 2u];
            let xv3 = x[xb + 3u];
            let wb = bf16_wbase(n0, k0);
            let ws = bf16_wstride();
            let w0a = w[wb + 0u * ws];
            let w0b = w[wb + 0u * ws + 1u];
            let w1a = w[wb + 1u * ws];
            let w1b = w[wb + 1u * ws + 1u];
            let w2a = w[wb + 2u * ws];
            let w2b = w[wb + 2u * ws + 1u];
            let w3a = w[wb + 3u * ws];
            let w3b = w[wb + 3u * ws + 1u];
            let w4a = w[wb + 4u * ws];
            let w4b = w[wb + 4u * ws + 1u];
            let w5a = w[wb + 5u * ws];
            let w5b = w[wb + 5u * ws + 1u];
            let w6a = w[wb + 6u * ws];
            let w6b = w[wb + 6u * ws + 1u];
            let w7a = w[wb + 7u * ws];
            let w7b = w[wb + 7u * ws + 1u];
            let w8a = w[wb + 8u * ws];
            let w8b = w[wb + 8u * ws + 1u];
            let w9a = w[wb + 9u * ws];
            let w9b = w[wb + 9u * ws + 1u];
            let w10a = w[wb + 10u * ws];
            let w10b = w[wb + 10u * ws + 1u];
            let w11a = w[wb + 11u * ws];
            let w11b = w[wb + 11u * ws + 1u];
            let w12a = w[wb + 12u * ws];
            let w12b = w[wb + 12u * ws + 1u];
            let w13a = w[wb + 13u * ws];
            let w13b = w[wb + 13u * ws + 1u];
            let w14a = w[wb + 14u * ws];
            let w14b = w[wb + 14u * ws + 1u];
            let w15a = w[wb + 15u * ws];
            let w15b = w[wb + 15u * ws + 1u];
            acc0 += dot(xv0, deq_bf16(w0a.x, w0a.y)) + dot(xv1, deq_bf16(w0a.z, w0a.w))
                + dot(xv2, deq_bf16(w0b.x, w0b.y)) + dot(xv3, deq_bf16(w0b.z, w0b.w));
            acc1 += dot(xv0, deq_bf16(w1a.x, w1a.y)) + dot(xv1, deq_bf16(w1a.z, w1a.w))
                + dot(xv2, deq_bf16(w1b.x, w1b.y)) + dot(xv3, deq_bf16(w1b.z, w1b.w));
            acc2 += dot(xv0, deq_bf16(w2a.x, w2a.y)) + dot(xv1, deq_bf16(w2a.z, w2a.w))
                + dot(xv2, deq_bf16(w2b.x, w2b.y)) + dot(xv3, deq_bf16(w2b.z, w2b.w));
            acc3 += dot(xv0, deq_bf16(w3a.x, w3a.y)) + dot(xv1, deq_bf16(w3a.z, w3a.w))
                + dot(xv2, deq_bf16(w3b.x, w3b.y)) + dot(xv3, deq_bf16(w3b.z, w3b.w));
            acc4 += dot(xv0, deq_bf16(w4a.x, w4a.y)) + dot(xv1, deq_bf16(w4a.z, w4a.w))
                + dot(xv2, deq_bf16(w4b.x, w4b.y)) + dot(xv3, deq_bf16(w4b.z, w4b.w));
            acc5 += dot(xv0, deq_bf16(w5a.x, w5a.y)) + dot(xv1, deq_bf16(w5a.z, w5a.w))
                + dot(xv2, deq_bf16(w5b.x, w5b.y)) + dot(xv3, deq_bf16(w5b.z, w5b.w));
            acc6 += dot(xv0, deq_bf16(w6a.x, w6a.y)) + dot(xv1, deq_bf16(w6a.z, w6a.w))
                + dot(xv2, deq_bf16(w6b.x, w6b.y)) + dot(xv3, deq_bf16(w6b.z, w6b.w));
            acc7 += dot(xv0, deq_bf16(w7a.x, w7a.y)) + dot(xv1, deq_bf16(w7a.z, w7a.w))
                + dot(xv2, deq_bf16(w7b.x, w7b.y)) + dot(xv3, deq_bf16(w7b.z, w7b.w));
            acc8 += dot(xv0, deq_bf16(w8a.x, w8a.y)) + dot(xv1, deq_bf16(w8a.z, w8a.w))
                + dot(xv2, deq_bf16(w8b.x, w8b.y)) + dot(xv3, deq_bf16(w8b.z, w8b.w));
            acc9 += dot(xv0, deq_bf16(w9a.x, w9a.y)) + dot(xv1, deq_bf16(w9a.z, w9a.w))
                + dot(xv2, deq_bf16(w9b.x, w9b.y)) + dot(xv3, deq_bf16(w9b.z, w9b.w));
            acc10 += dot(xv0, deq_bf16(w10a.x, w10a.y)) + dot(xv1, deq_bf16(w10a.z, w10a.w))
                + dot(xv2, deq_bf16(w10b.x, w10b.y)) + dot(xv3, deq_bf16(w10b.z, w10b.w));
            acc11 += dot(xv0, deq_bf16(w11a.x, w11a.y)) + dot(xv1, deq_bf16(w11a.z, w11a.w))
                + dot(xv2, deq_bf16(w11b.x, w11b.y)) + dot(xv3, deq_bf16(w11b.z, w11b.w));
            acc12 += dot(xv0, deq_bf16(w12a.x, w12a.y)) + dot(xv1, deq_bf16(w12a.z, w12a.w))
                + dot(xv2, deq_bf16(w12b.x, w12b.y)) + dot(xv3, deq_bf16(w12b.z, w12b.w));
            acc13 += dot(xv0, deq_bf16(w13a.x, w13a.y)) + dot(xv1, deq_bf16(w13a.z, w13a.w))
                + dot(xv2, deq_bf16(w13b.x, w13b.y)) + dot(xv3, deq_bf16(w13b.z, w13b.w));
            acc14 += dot(xv0, deq_bf16(w14a.x, w14a.y)) + dot(xv1, deq_bf16(w14a.z, w14a.w))
                + dot(xv2, deq_bf16(w14b.x, w14b.y)) + dot(xv3, deq_bf16(w14b.z, w14b.w));
            acc15 += dot(xv0, deq_bf16(w15a.x, w15a.y)) + dot(xv1, deq_bf16(w15a.z, w15a.w))
                + dot(xv2, deq_bf16(w15b.x, w15b.y)) + dot(xv3, deq_bf16(w15b.z, w15b.w));
        }

        // Tree-reduce the K-split lanes for each of the BN columns.
        red[lane * SR + 0u] = acc0;
        red[lane * SR + 1u] = acc1;
        red[lane * SR + 2u] = acc2;
        red[lane * SR + 3u] = acc3;
        red[lane * SR + 4u] = acc4;
        red[lane * SR + 5u] = acc5;
        red[lane * SR + 6u] = acc6;
        red[lane * SR + 7u] = acc7;
        red[lane * SR + 8u] = acc8;
        red[lane * SR + 9u] = acc9;
        red[lane * SR + 10u] = acc10;
        red[lane * SR + 11u] = acc11;
        red[lane * SR + 12u] = acc12;
        red[lane * SR + 13u] = acc13;
        red[lane * SR + 14u] = acc14;
        red[lane * SR + 15u] = acc15;
        workgroupBarrier();
        var stride = LANES >> 1u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (lane < stride) {
                let mine = lane * SR;
                let other = (lane + stride) * SR;
                for (var c: u32 = 0u; c < BN; c += 1u) {
                    red[mine + c] += red[other + c];
                }
            }
            workgroupBarrier();
            stride >>= 1u;
        }

        if (lane < BN) {
            if (SEGS == 1u) {
                y[n0 + lane] = red[lane] + f32(ACC) * y[n0 + lane];
            } else {
                y[seg * N + n0 + lane] = red[lane];
            }
        }
    }
}
