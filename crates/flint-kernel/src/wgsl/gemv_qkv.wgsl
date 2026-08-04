// Fused QKV projection: three projections q (NQ cols), k (NK) and v (NV)
// computed in one dispatch into three separate outputs. Every detail matches
// gemv's i8 block-major path (two k-blocks per lane per iteration, 8-lane
// tree reduction, SEGS-wide K splits); the segments' partials land in one
// contiguous [SEGS, NQ+NK+NV] scratch that merge_qkv folds into the three
// outputs.

override NQ: u32 = 1u;
override NK: u32 = 1u;
override NV: u32 = 1u;
override K: u32 = 1u;
override GROUP: u32 = 128u;
override SEGS: u32 = 8u;

const BN: u32 = 16u;
const LANES: u32 = 128u;
const SR: u32 = 17u;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scq: array<f32>;
@group(0) @binding(3) var<storage, read_write> yq: array<f32>;
@group(0) @binding(4) var<storage, read> wk: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> sck: array<f32>;
@group(0) @binding(6) var<storage, read_write> yk: array<f32>;
@group(0) @binding(7) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> scv: array<f32>;
@group(0) @binding(9) var<storage, read_write> yv: array<f32>;
// Segment partials [SEGS, NQ+NK+NV]; merge_qkv folds them into the outputs.
@group(0) @binding(10) var<storage, read_write> partial: array<f32>;

var<workgroup> red: array<f32, LANES * SR>;

// Sign-extends the 4 i8 bytes of an i8 word.
fn deq_i8(word: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((word & 0xFFu) << 24) >> 24),
        f32(i32((word & 0xFF00u) << 16) >> 24),
        f32(i32((word & 0xFF0000u) << 8) >> 24),
        f32(i32(word >> 24) << 24 >> 24),
    );
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let c = lane % BN;
    let kb_rel = lane / BN;
    let seg = wg.y;
    let col0 = wg.x * BN;
    let seg_kb = (K / SEGS) / 16u;
    let kb_lo = seg * seg_kb;
    let lim = kb_lo + seg_kb;
    let iters = (seg_kb + 15u) / 16u;
    let ns = K / GROUP;

    let is_q = col0 < NQ;
    let is_k = col0 >= NQ && col0 < NQ + NK;
    // v is the remainder.
    let n = select(select(NV, NK, is_k), NQ, is_q);
    let n0 = col0 - select(select(NQ + NK, NQ, is_k), 0u, is_q);
    let ntot = NQ + NK + NV;

    var acc0 = 0.0;
    var acc1 = 0.0;
    // Three weight streams; WGSL cannot index storage buffers, so the
    // projection selects its branch (uniform across the workgroup).
    if (is_q) {
        for (var it: u32 = 0u; it < iters; it += 1u) {
            let kb0 = kb_lo + it * 16u + kb_rel;
            if (kb0 >= lim) {
                break;
            }
            let kb1 = kb0 + 8u;
            let xb = kb0 * 4u;
            let xv0 = x[xb];
            let xv1 = x[xb + 1u];
            let xv2 = x[xb + 2u];
            let xv3 = x[xb + 3u];
            let w4 = wq[kb0 * NQ + n0 + c];
            let sc = scq[(n0 + c) * ns + kb0 / (GROUP / 16u)];
            acc0 += dot(xv0, deq_i8(w4.x) * sc) + dot(xv1, deq_i8(w4.y) * sc)
                + dot(xv2, deq_i8(w4.z) * sc) + dot(xv3, deq_i8(w4.w) * sc);
            if (kb1 < lim) {
                let xb1 = kb1 * 4u;
                let xv4 = x[xb1];
                let xv5 = x[xb1 + 1u];
                let xv6 = x[xb1 + 2u];
                let xv7 = x[xb1 + 3u];
                let w4b = wq[kb1 * NQ + n0 + c];
                let scb = scq[(n0 + c) * ns + kb1 / (GROUP / 16u)];
                acc1 += dot(xv4, deq_i8(w4b.x) * scb) + dot(xv5, deq_i8(w4b.y) * scb)
                    + dot(xv6, deq_i8(w4b.z) * scb) + dot(xv7, deq_i8(w4b.w) * scb);
            }
        }
    } else if (is_k) {
        for (var it: u32 = 0u; it < iters; it += 1u) {
            let kb0 = kb_lo + it * 16u + kb_rel;
            if (kb0 >= lim) {
                break;
            }
            let kb1 = kb0 + 8u;
            let xb = kb0 * 4u;
            let xv0 = x[xb];
            let xv1 = x[xb + 1u];
            let xv2 = x[xb + 2u];
            let xv3 = x[xb + 3u];
            let w4 = wk[kb0 * NK + n0 + c];
            let sc = sck[(n0 + c) * ns + kb0 / (GROUP / 16u)];
            acc0 += dot(xv0, deq_i8(w4.x) * sc) + dot(xv1, deq_i8(w4.y) * sc)
                + dot(xv2, deq_i8(w4.z) * sc) + dot(xv3, deq_i8(w4.w) * sc);
            if (kb1 < lim) {
                let xb1 = kb1 * 4u;
                let xv4 = x[xb1];
                let xv5 = x[xb1 + 1u];
                let xv6 = x[xb1 + 2u];
                let xv7 = x[xb1 + 3u];
                let w4b = wk[kb1 * NK + n0 + c];
                let scb = sck[(n0 + c) * ns + kb1 / (GROUP / 16u)];
                acc1 += dot(xv4, deq_i8(w4b.x) * scb) + dot(xv5, deq_i8(w4b.y) * scb)
                    + dot(xv6, deq_i8(w4b.z) * scb) + dot(xv7, deq_i8(w4b.w) * scb);
            }
        }
    } else {
        for (var it: u32 = 0u; it < iters; it += 1u) {
            let kb0 = kb_lo + it * 16u + kb_rel;
            if (kb0 >= lim) {
                break;
            }
            let kb1 = kb0 + 8u;
            let xb = kb0 * 4u;
            let xv0 = x[xb];
            let xv1 = x[xb + 1u];
            let xv2 = x[xb + 2u];
            let xv3 = x[xb + 3u];
            let w4 = wv[kb0 * NV + n0 + c];
            let sc = scv[(n0 + c) * ns + kb0 / (GROUP / 16u)];
            acc0 += dot(xv0, deq_i8(w4.x) * sc) + dot(xv1, deq_i8(w4.y) * sc)
                + dot(xv2, deq_i8(w4.z) * sc) + dot(xv3, deq_i8(w4.w) * sc);
            if (kb1 < lim) {
                let xb1 = kb1 * 4u;
                let xv4 = x[xb1];
                let xv5 = x[xb1 + 1u];
                let xv6 = x[xb1 + 2u];
                let xv7 = x[xb1 + 3u];
                let w4b = wv[kb1 * NV + n0 + c];
                let scb = scv[(n0 + c) * ns + kb1 / (GROUP / 16u)];
                acc1 += dot(xv4, deq_i8(w4b.x) * scb) + dot(xv5, deq_i8(w4b.y) * scb)
                    + dot(xv6, deq_i8(w4b.z) * scb) + dot(xv7, deq_i8(w4b.w) * scb);
            }
        }
    }

    // Tree-reduce the 8 lanes sharing each column (descending strides).
    red[lane] = acc0 + acc1;
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
        // Single-segment runs write the outputs directly; multi-segment
        // runs write the shared partial slab for merge_qkv.
        if (SEGS == 1u) {
            if (is_q) {
                yq[n0 + lane] = red[lane];
            } else if (is_k) {
                yk[n0 + lane] = red[lane];
            } else {
                yv[n0 + lane] = red[lane];
            }
        } else {
            partial[seg * ntot + col0 + lane] = red[lane];
        }
    }
}
