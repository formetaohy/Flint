// y[n] = sum_k x[k] * dequant(w[n, k])   (single activation row, M == 1)
// x: f32 [K] (row 0 of the activation tile), y: f32 [N]; dispatch
// [N/BN, SEGS, 1]. N a multiple of 16, K a multiple of 128, SEGS in {1,4,8}
// with K % SEGS == 0.
// WDTYPE 0: w is packed bf16 (two values per u32), scales unused.
// WDTYPE 1: w is i8 (four values per u32), dequantized with per-row group
//           scales [N, K/GROUP].
//
// Split-K gemv: workgroup (n0, s) streams the weight rows of columns n0..n0+15
// over its K segment [s*K/SEGS, (s+1)*K/SEGS) — skinny outputs (small N) fan
// out over SEGS more workgroups instead of idling most SMs. Each workgroup
// tree-reduces its segment partials and writes them to `y` (SEGS == 1) or the
// [SEGS, N] partial buffer (SEGS > 1), which `merge_gemv` sums. 128 lanes
// split the segment; each lane holds 16 explicit register accumulators (one
// per column), the same shape that saturates bandwidth on wide outputs.

override N: u32 = 1u;
override K: u32 = 1u;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;
override SEGS: u32 = 1u;

const BN: u32 = 16u;
const BK: u32 = 128u;
const LANES: u32 = 128u;
const SR: u32 = 17u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, LANES * SR>;

fn bf16f(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

fn load_w(n: u32, k: u32) -> f32 {
    if (WDTYPE == 1u) {
        let word = w[(n * K + k) >> 2];
        let bits = (word >> ((k & 3u) << 3u)) & 0xFFu;
        let sb = i32(bits << 24) >> 24;
        return f32(sb) * scales[n * (K / GROUP) + k / GROUP];
    }
    let p = w[(n * K + k) >> 1];
    let bits = select(p >> 16, p & 0xFFFFu, (k & 1u) == 0u);
    return bf16f(bits);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let n0 = wg.x * BN;
    let seg = wg.y;
    let seg_len = K / SEGS;
    let k_lo = seg * seg_len;
    let k_hi = k_lo + seg_len;

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

    for (var k0: u32 = k_lo; k0 < k_hi; k0 += BK) {
        // Each lane owns x[k0+lane] outright (no cross-lane sharing), so it
        // is read straight from global memory — L2-resident and barrier-free.
        let kk = k0 + lane;
        if (kk < k_hi) {
            let xv = x[kk];
            acc0 += xv * load_w(n0 + 0u, kk);
            acc1 += xv * load_w(n0 + 1u, kk);
            acc2 += xv * load_w(n0 + 2u, kk);
            acc3 += xv * load_w(n0 + 3u, kk);
            acc4 += xv * load_w(n0 + 4u, kk);
            acc5 += xv * load_w(n0 + 5u, kk);
            acc6 += xv * load_w(n0 + 6u, kk);
            acc7 += xv * load_w(n0 + 7u, kk);
            acc8 += xv * load_w(n0 + 8u, kk);
            acc9 += xv * load_w(n0 + 9u, kk);
            acc10 += xv * load_w(n0 + 10u, kk);
            acc11 += xv * load_w(n0 + 11u, kk);
            acc12 += xv * load_w(n0 + 12u, kk);
            acc13 += xv * load_w(n0 + 13u, kk);
            acc14 += xv * load_w(n0 + 14u, kk);
            acc15 += xv * load_w(n0 + 15u, kk);
        }
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
            y[n0 + lane] = red[lane];
        } else {
            y[seg * N + n0 + lane] = red[lane];
        }
    }
}
