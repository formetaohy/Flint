// y[m, n] = sum_k x[m, k] * dequant(w[n, k])
// x: f32 [M, K] viewed as vec4 (16B-aligned), y: f32 [M, N]; dispatch
// [N/BN, ceil(M/ROWS_G), 1]. N multiple of 16, K multiple of 16, M even.
// WDTYPE 0: w is packed bf16 (two values per u32), scales unused.
// WDTYPE 1: w is i8 (four values per u32), dequantized with per-row group
//           scales [N, K/GROUP].
//
// One workgroup owns ROWS_G rows x BN=16 columns; every lane is one
// (row, column) cell sweeping the whole K range, so the kernel has no
// reduction, no barriers and no shared memory. The block-major weight
// column is contiguous, and its bytes broadcast across row lanes through
// L1, so each weight byte is read from L2 once per row group.

override N: u32 = 1u;
override K: u32 = 1u;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;
// Rows per workgroup: 16 (256 lanes) or 64 (1024 lanes).
override ROWS_G: u32 = 16u;
// Accumulate into y instead of overwriting: y = y + result.
override ACC: u32 = 0u;
// Column stride and offset of y (fused qkv tiles: three gemms write their
// column ranges of one [m, N_TOTAL] buffer).
override Y_STRIDE: u32 = 1u;
override Y_OFF: u32 = 0u;

const BN: u32 = 16u;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

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

@compute @workgroup_size(ROWS_G * BN)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    // Lane cell: row r of the group, column c.
    let r = lane / BN;
    let c = lane % BN;
    let m0 = wg.y * ROWS_G;
    let n0 = wg.x * BN;

    if (WDTYPE == 1u) {
        // ---- i8 block-major path: one (row, column) dot product; four
        // k-blocks per iteration so four weight loads stay in flight. ----
        // Column groups may cover a partial tail (N not a multiple of BN);
        // out-of-range lanes retire early.
        if (n0 + c >= N) {
            return;
        }
        let kbs = K / 16u;
        let ns = K / GROUP;
        let xr = (m0 + r) * (K / 4u);
        var acc = 0.0;
        for (var kb = 0u; kb < kbs; kb += 4u) {
            let xb = kb * 4u;
            let xv0 = x[xr + xb];
            let xv1 = x[xr + xb + 1u];
            let xv2 = x[xr + xb + 2u];
            let xv3 = x[xr + xb + 3u];
            let w4 = w[kb * N + n0 + c];
            let sc = scales[(n0 + c) * ns + kb / (GROUP / 16u)];
            acc += dot(xv0, deq_i8(w4.x) * sc) + dot(xv1, deq_i8(w4.y) * sc)
                + dot(xv2, deq_i8(w4.z) * sc) + dot(xv3, deq_i8(w4.w) * sc);
            let kb1 = kb + 1u;
            let kb2 = kb + 2u;
            let kb3 = kb + 3u;
            if (kb1 < kbs) {
                let xb1 = kb1 * 4u;
                let xv4 = x[xr + xb1];
                let xv5 = x[xr + xb1 + 1u];
                let xv6 = x[xr + xb1 + 2u];
                let xv7 = x[xr + xb1 + 3u];
                let w4b = w[kb1 * N + n0 + c];
                let scb = scales[(n0 + c) * ns + kb1 / (GROUP / 16u)];
                acc += dot(xv4, deq_i8(w4b.x) * scb) + dot(xv5, deq_i8(w4b.y) * scb)
                    + dot(xv6, deq_i8(w4b.z) * scb) + dot(xv7, deq_i8(w4b.w) * scb);
            }
            if (kb2 < kbs) {
                let xb2 = kb2 * 4u;
                let xv8 = x[xr + xb2];
                let xv9 = x[xr + xb2 + 1u];
                let xv10 = x[xr + xb2 + 2u];
                let xv11 = x[xr + xb2 + 3u];
                let w4c = w[kb2 * N + n0 + c];
                let scc = scales[(n0 + c) * ns + kb2 / (GROUP / 16u)];
                acc += dot(xv8, deq_i8(w4c.x) * scc) + dot(xv9, deq_i8(w4c.y) * scc)
                    + dot(xv10, deq_i8(w4c.z) * scc) + dot(xv11, deq_i8(w4c.w) * scc);
            }
            if (kb3 < kbs) {
                let xb3 = kb3 * 4u;
                let xv12 = x[xr + xb3];
                let xv13 = x[xr + xb3 + 1u];
                let xv14 = x[xr + xb3 + 2u];
                let xv15 = x[xr + xb3 + 3u];
                let w4d = w[kb3 * N + n0 + c];
                let scd = scales[(n0 + c) * ns + kb3 / (GROUP / 16u)];
                acc += dot(xv12, deq_i8(w4d.x) * scd) + dot(xv13, deq_i8(w4d.y) * scd)
                    + dot(xv14, deq_i8(w4d.z) * scd) + dot(xv15, deq_i8(w4d.w) * scd);
            }
        }
        y[(m0 + r) * Y_STRIDE + Y_OFF + n0 + c] = acc + f32(ACC) * y[(m0 + r) * Y_STRIDE + Y_OFF + n0 + c];
    } else {
        // ---- bf16 row-major path: 16 k per iteration, one column. ----
        if (n0 + c >= N) {
            return;
        }
        let xr = (m0 + r) * (K / 4u);
        let wr = n0 + c;
        var acc = 0.0;
        for (var kk = 0u; kk < K; kk += 16u) {
            let xb = xr + kk / 4u;
            let xv0 = x[xb];
            let xv1 = x[xb + 1u];
            let xv2 = x[xb + 2u];
            let xv3 = x[xb + 3u];
            let wb = (wr * K + kk) / 8u;
            let wv0 = w[wb];
            let wv1 = w[wb + 1u];
            acc += dot(xv0, deq_bf16(wv0.x, wv0.y)) + dot(xv1, deq_bf16(wv0.z, wv0.w))
                + dot(xv2, deq_bf16(wv1.x, wv1.y)) + dot(xv3, deq_bf16(wv1.z, wv1.w));
        }
        y[(m0 + r) * Y_STRIDE + Y_OFF + n0 + c] = acc + f32(ACC) * y[(m0 + r) * Y_STRIDE + Y_OFF + n0 + c];
    }
}
