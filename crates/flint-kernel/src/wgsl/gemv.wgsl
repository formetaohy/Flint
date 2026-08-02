// y[n] = sum_k x[k] * dequant(w[n, k])   (single activation row, M == 1)
// x: f32 [K] (row 0 of the activation tile), y: f32 [N]; dispatch [N/BN, 1, 1].
// N a multiple of 16, K a multiple of 128.
// WDTYPE 0: w is packed bf16 (two values per u32), scales unused.
// WDTYPE 1: w is i8 (four values per u32), dequantized with per-row group
//           scales [N, K/GROUP].
//
// Decode is a pure matrix-vector product, so unlike the tiled matmul there is
// no activation reuse to stage: each workgroup takes BN output columns and
// streams their weights straight from global memory (consecutive threads read
// consecutive K, fully coalesced), multiplying by the activation row held in
// workgroup memory. One tree reduction over the K-split lanes finishes each
// column. This runs at a large fraction of peak memory bandwidth.

override N: u32 = 1u;
override K: u32 = 1u;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;

const BN: u32 = 16u;
const BK: u32 = 128u;
const LANES: u32 = 128u;
const SR: u32 = 17u; // reduction padding, coprime with 32 banks

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

    var acc: array<f32, BN>;
    for (var c: u32 = 0u; c < BN; c += 1u) {
        acc[c] = 0.0;
    }

    for (var k0: u32 = 0u; k0 < K; k0 += BK) {
        // Each lane owns x[k0+lane] outright (no cross-lane sharing), so it is
        // read straight from global memory — L2-resident and barrier-free.
        let kk = k0 + lane;
        if (kk < K) {
            let xv = x[kk];
            for (var c: u32 = 0u; c < BN; c += 1u) {
                acc[c] += xv * load_w(n0 + c, kk);
            }
        }
    }

    // Tree-reduce the K-split lanes for each of the BN columns.
    for (var c: u32 = 0u; c < BN; c += 1u) {
        red[lane * SR + c] = acc[c];
    }
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
        y[n0 + lane] = red[lane];
    }
}
