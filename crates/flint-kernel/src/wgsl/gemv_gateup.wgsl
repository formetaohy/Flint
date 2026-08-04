// Fused gate/up projection: the two MLP input projections (both NG cols)
// computed in one dispatch into separate outputs. Identical structure to
// gemv_qkv's i8 block-major path; SEGS=2 is the measured optimum for the
// 9728-wide projections. Partial slab [SEGS, 2*NG] folded by merge_gateup.

override NG: u32 = 1u;
override K: u32 = 1u;
override GROUP: u32 = 128u;
override SEGS: u32 = 2u;

const BN: u32 = 16u;
const LANES: u32 = 128u;
const SR: u32 = 17u;

@group(0) @binding(0) var<storage, read> xin: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wg: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scg: array<f32>;
@group(0) @binding(3) var<storage, read_write> yg: array<f32>;
@group(0) @binding(4) var<storage, read> wu: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> scu: array<f32>;
@group(0) @binding(6) var<storage, read_write> yu: array<f32>;
@group(0) @binding(7) var<storage, read_write> partial: array<f32>;

var<workgroup> red: array<f32, LANES * SR>;

fn deq_i8(word: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((word & 0xFFu) << 24) >> 24),
        f32(i32((word & 0xFF00u) << 16) >> 24),
        f32(i32((word & 0xFF0000u) << 8) >> 24),
        f32(i32(word >> 24) << 24 >> 24),
    );
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let c = lane % BN;
    let kb_rel = lane / BN;
    let seg = wgid.y;
    let col0 = wgid.x * BN;
    let is_g = col0 < NG;
    let n = select(NG, NG, is_g);
    let n0 = col0 - select(NG, 0u, is_g);
    let seg_kb = (K / SEGS) / 16u;
    let kb_lo = seg * seg_kb;
    let lim = kb_lo + seg_kb;
    let iters = (seg_kb + 15u) / 16u;
    let ns = K / GROUP;

    var acc0 = 0.0;
    var acc1 = 0.0;
    if (is_g) {
        for (var it: u32 = 0u; it < iters; it += 1u) {
            let kb0 = kb_lo + it * 16u + kb_rel;
            if (kb0 >= lim) {
                break;
            }
            let kb1 = kb0 + 8u;
            let xb = kb0 * 4u;
            let xv0 = xin[xb];
            let xv1 = xin[xb + 1u];
            let xv2 = xin[xb + 2u];
            let xv3 = xin[xb + 3u];
            let w4 = wg[kb0 * NG + n0 + c];
            let sc = scg[(n0 + c) * ns + kb0 / (GROUP / 16u)];
            acc0 += dot(xv0, deq_i8(w4.x) * sc) + dot(xv1, deq_i8(w4.y) * sc)
                + dot(xv2, deq_i8(w4.z) * sc) + dot(xv3, deq_i8(w4.w) * sc);
            if (kb1 < lim) {
                let xb1 = kb1 * 4u;
                let xv4 = xin[xb1];
                let xv5 = xin[xb1 + 1u];
                let xv6 = xin[xb1 + 2u];
                let xv7 = xin[xb1 + 3u];
                let w4b = wg[kb1 * NG + n0 + c];
                let scb = scg[(n0 + c) * ns + kb1 / (GROUP / 16u)];
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
            let xv0 = xin[xb];
            let xv1 = xin[xb + 1u];
            let xv2 = xin[xb + 2u];
            let xv3 = xin[xb + 3u];
            let w4 = wu[kb0 * NG + n0 + c];
            let sc = scu[(n0 + c) * ns + kb0 / (GROUP / 16u)];
            acc0 += dot(xv0, deq_i8(w4.x) * sc) + dot(xv1, deq_i8(w4.y) * sc)
                + dot(xv2, deq_i8(w4.z) * sc) + dot(xv3, deq_i8(w4.w) * sc);
            if (kb1 < lim) {
                let xb1 = kb1 * 4u;
                let xv4 = xin[xb1];
                let xv5 = xin[xb1 + 1u];
                let xv6 = xin[xb1 + 2u];
                let xv7 = xin[xb1 + 3u];
                let w4b = wu[kb1 * NG + n0 + c];
                let scb = scu[(n0 + c) * ns + kb1 / (GROUP / 16u)];
                acc1 += dot(xv4, deq_i8(w4b.x) * scb) + dot(xv5, deq_i8(w4b.y) * scb)
                    + dot(xv6, deq_i8(w4b.z) * scb) + dot(xv7, deq_i8(w4b.w) * scb);
            }
        }
    }

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
        if (SEGS == 1u) {
            if (is_g) {
                yg[n0 + lane] = red[lane];
            } else {
                yu[n0 + lane] = red[lane];
            }
        } else {
            partial[seg * (2u * NG) + col0 + lane] = red[lane];
        }
    }
}
