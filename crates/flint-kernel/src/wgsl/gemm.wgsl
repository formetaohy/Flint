// y[m, n] = sum_k x[m, k] * dequant(w[n, k])
// x: f32 [M, K] viewed as vec4 (16B-aligned), y: f32 [M, N]; dispatch
// [N/BN, M/2, 1]. N a multiple of 16, K a multiple of 64, M even.
// WDTYPE 0: w is packed bf16 (two values per u32), scales unused.
// WDTYPE 1: w is i8 (four values per u32), dequantized with per-row group
//           scales [N, K/GROUP].
//
// gemv-style streaming matmul for the skinny forward pass (M <= 16). Each
// workgroup owns 2 rows x BN=16 columns and streams their weight rows from
// global memory; sibling workgroups covering the same columns run
// concurrently, so the redundant weight reads hit L2, not VRAM. 128 lanes
// split K: each lane owns 4 consecutive K values (one i8 word per column),
// accumulated into 2x16 register accumulators held as 8 vec4s — the same
// register shape that makes the single-row gemv bandwidth-saturated. A tree
// reduction over the K-split lanes finishes each (row, column).

override N: u32 = 1u;
override K: u32 = 1u;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;

const BN: u32 = 16u;
const LANES: u32 = 128u;
// Per-chunk K width: LANES lanes x 4 K values each.
const BK: u32 = LANES * 4u;
// Reduction padding, coprime with 32 banks.
const SR: u32 = 33u;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

// K-split partials: 2 rows x 16 columns per lane.
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
// Column stride in u32 words (K/4 for i8, K/2 for bf16).
fn k4n() -> u32 {
    return K / 4u;
}

fn k2n() -> u32 {
    return K / 2u;
}

fn deq_bf16(w0: u32, w1: u32) -> vec4<f32> {
    return vec4<f32>(
        bf16f(w0 & 0xFFFFu),
        bf16f(w0 >> 16),
        bf16f(w1 & 0xFFFFu),
        bf16f(w1 >> 16),
    );
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let m0 = wg.y * 2u;
    let n0 = wg.x * BN;

    // 2 rows x 16 columns of register accumulators, held as vec4 column quads.
    var a0 = vec4(0.0); var a1 = vec4(0.0); var a2 = vec4(0.0); var a3 = vec4(0.0);
    var b0 = vec4(0.0); var b1 = vec4(0.0); var b2 = vec4(0.0); var b3 = vec4(0.0);

    let xr0 = m0 * (K / 4u);
    let xr1 = (m0 + 1u) * (K / 4u);
    let wr = n0 * (K / 4u) + lane;

    // Each lane owns x[m, k0+lane*4 .. +4] for both rows (L2-resident,
    // barrier-free) and streams the 16 weight words of its 16 columns.
    for (var k0: u32 = 0u; k0 < K; k0 += BK) {
        let wbase = k0 / 4u;
        let xv0 = x[xr0 + wbase + lane];
        let xv1 = x[xr1 + wbase + lane];
        // Lanes whose 4-K slice falls past K (only the last chunk of small
        // K) contribute zero; the guard keeps their loads in bounds.
        if (k0 + lane * 4u < K) {
        if (WDTYPE == 1u) {
            // Group scales: the lane's 4 K values lie in one K/GROUP block.
            let sb = (k0 + lane * 4u) / GROUP;
            let sc0 = scales[(n0 + 0u) * (K / GROUP) + sb];
            let sc1 = scales[(n0 + 1u) * (K / GROUP) + sb];
            let sc2 = scales[(n0 + 2u) * (K / GROUP) + sb];
            let sc3 = scales[(n0 + 3u) * (K / GROUP) + sb];
            let sc4 = scales[(n0 + 4u) * (K / GROUP) + sb];
            let sc5 = scales[(n0 + 5u) * (K / GROUP) + sb];
            let sc6 = scales[(n0 + 6u) * (K / GROUP) + sb];
            let sc7 = scales[(n0 + 7u) * (K / GROUP) + sb];
            let sc8 = scales[(n0 + 8u) * (K / GROUP) + sb];
            let sc9 = scales[(n0 + 9u) * (K / GROUP) + sb];
            let sc10 = scales[(n0 + 10u) * (K / GROUP) + sb];
            let sc11 = scales[(n0 + 11u) * (K / GROUP) + sb];
            let sc12 = scales[(n0 + 12u) * (K / GROUP) + sb];
            let sc13 = scales[(n0 + 13u) * (K / GROUP) + sb];
            let sc14 = scales[(n0 + 14u) * (K / GROUP) + sb];
            let sc15 = scales[(n0 + 15u) * (K / GROUP) + sb];
            let d0 = deq_i8(w[wr + wbase]) * sc0;
            let d1 = deq_i8(w[wr + wbase + k4n()]) * sc1;
            let d2 = deq_i8(w[wr + wbase + k4n() * 2u]) * sc2;
            let d3 = deq_i8(w[wr + wbase + k4n() * 3u]) * sc3;
            let d4 = deq_i8(w[wr + wbase + k4n() * 4u]) * sc4;
            let d5 = deq_i8(w[wr + wbase + k4n() * 5u]) * sc5;
            let d6 = deq_i8(w[wr + wbase + k4n() * 6u]) * sc6;
            let d7 = deq_i8(w[wr + wbase + k4n() * 7u]) * sc7;
            let d8 = deq_i8(w[wr + wbase + k4n() * 8u]) * sc8;
            let d9 = deq_i8(w[wr + wbase + k4n() * 9u]) * sc9;
            let d10 = deq_i8(w[wr + wbase + k4n() * 10u]) * sc10;
            let d11 = deq_i8(w[wr + wbase + k4n() * 11u]) * sc11;
            let d12 = deq_i8(w[wr + wbase + k4n() * 12u]) * sc12;
            let d13 = deq_i8(w[wr + wbase + k4n() * 13u]) * sc13;
            let d14 = deq_i8(w[wr + wbase + k4n() * 14u]) * sc14;
            let d15 = deq_i8(w[wr + wbase + k4n() * 15u]) * sc15;
            // Each d_c is column c's K-vector (4 k values); the acc quads are
            // column groups, so the K axis is spread via swizzled broadcasts.
            // Quad 0: columns 0..3.
            let e0 = vec4(d0.x, d1.x, d2.x, d3.x);
            let e1 = vec4(d0.y, d1.y, d2.y, d3.y);
            let e2 = vec4(d0.z, d1.z, d2.z, d3.z);
            let e3 = vec4(d0.w, d1.w, d2.w, d3.w);
            a0 += xv0.xxxx * e0 + xv0.yyyy * e1 + xv0.zzzz * e2 + xv0.wwww * e3;
            b0 += xv1.xxxx * e0 + xv1.yyyy * e1 + xv1.zzzz * e2 + xv1.wwww * e3;
            // Quad 1: columns 4..7.
            let e4 = vec4(d4.x, d5.x, d6.x, d7.x);
            let e5 = vec4(d4.y, d5.y, d6.y, d7.y);
            let e6 = vec4(d4.z, d5.z, d6.z, d7.z);
            let e7 = vec4(d4.w, d5.w, d6.w, d7.w);
            a1 += xv0.xxxx * e4 + xv0.yyyy * e5 + xv0.zzzz * e6 + xv0.wwww * e7;
            b1 += xv1.xxxx * e4 + xv1.yyyy * e5 + xv1.zzzz * e6 + xv1.wwww * e7;
            // Quad 2: columns 8..11.
            let e8 = vec4(d8.x, d9.x, d10.x, d11.x);
            let e9 = vec4(d8.y, d9.y, d10.y, d11.y);
            let e10 = vec4(d8.z, d9.z, d10.z, d11.z);
            let e11 = vec4(d8.w, d9.w, d10.w, d11.w);
            a2 += xv0.xxxx * e8 + xv0.yyyy * e9 + xv0.zzzz * e10 + xv0.wwww * e11;
            b2 += xv1.xxxx * e8 + xv1.yyyy * e9 + xv1.zzzz * e10 + xv1.wwww * e11;
            // Quad 3: columns 12..15.
            let e12 = vec4(d12.x, d13.x, d14.x, d15.x);
            let e13 = vec4(d12.y, d13.y, d14.y, d15.y);
            let e14 = vec4(d12.z, d13.z, d14.z, d15.z);
            let e15 = vec4(d12.w, d13.w, d14.w, d15.w);
            a3 += xv0.xxxx * e12 + xv0.yyyy * e13 + xv0.zzzz * e14 + xv0.wwww * e15;
            b3 += xv1.xxxx * e12 + xv1.yyyy * e13 + xv1.zzzz * e14 + xv1.wwww * e15;
        } else {
            // bf16: two words per column quad (4 bf16 values).
            let base = n0 * (K / 2u) + k0 / 2u + lane * 2u;
            let q0 = deq_bf16(w[base], w[base + 1u]);
            let q1 = deq_bf16(w[base + k2n()], w[base + k2n() + 1u]);
            let q2 = deq_bf16(w[base + k2n() * 2u], w[base + k2n() * 2u + 1u]);
            let q3 = deq_bf16(w[base + k2n() * 3u], w[base + k2n() * 3u + 1u]);
            let q4 = deq_bf16(w[base + k2n() * 4u], w[base + k2n() * 4u + 1u]);
            let q5 = deq_bf16(w[base + k2n() * 5u], w[base + k2n() * 5u + 1u]);
            let q6 = deq_bf16(w[base + k2n() * 6u], w[base + k2n() * 6u + 1u]);
            let q7 = deq_bf16(w[base + k2n() * 7u], w[base + k2n() * 7u + 1u]);
            let q8 = deq_bf16(w[base + k2n() * 8u], w[base + k2n() * 8u + 1u]);
            let q9 = deq_bf16(w[base + k2n() * 9u], w[base + k2n() * 9u + 1u]);
            let q10 = deq_bf16(w[base + k2n() * 10u], w[base + k2n() * 10u + 1u]);
            let q11 = deq_bf16(w[base + k2n() * 11u], w[base + k2n() * 11u + 1u]);
            let q12 = deq_bf16(w[base + k2n() * 12u], w[base + k2n() * 12u + 1u]);
            let q13 = deq_bf16(w[base + k2n() * 13u], w[base + k2n() * 13u + 1u]);
            let q14 = deq_bf16(w[base + k2n() * 14u], w[base + k2n() * 14u + 1u]);
            let q15 = deq_bf16(w[base + k2n() * 15u], w[base + k2n() * 15u + 1u]);
            let f0 = vec4(q0.x, q1.x, q2.x, q3.x);
            let f1 = vec4(q0.y, q1.y, q2.y, q3.y);
            let f2 = vec4(q0.z, q1.z, q2.z, q3.z);
            let f3 = vec4(q0.w, q1.w, q2.w, q3.w);
            a0 += xv0.xxxx * f0 + xv0.yyyy * f1 + xv0.zzzz * f2 + xv0.wwww * f3;
            b0 += xv1.xxxx * f0 + xv1.yyyy * f1 + xv1.zzzz * f2 + xv1.wwww * f3;
            let f4 = vec4(q4.x, q5.x, q6.x, q7.x);
            let f5 = vec4(q4.y, q5.y, q6.y, q7.y);
            let f6 = vec4(q4.z, q5.z, q6.z, q7.z);
            let f7 = vec4(q4.w, q5.w, q6.w, q7.w);
            a1 += xv0.xxxx * f4 + xv0.yyyy * f5 + xv0.zzzz * f6 + xv0.wwww * f7;
            b1 += xv1.xxxx * f4 + xv1.yyyy * f5 + xv1.zzzz * f6 + xv1.wwww * f7;
            let f8 = vec4(q8.x, q9.x, q10.x, q11.x);
            let f9 = vec4(q8.y, q9.y, q10.y, q11.y);
            let f10 = vec4(q8.z, q9.z, q10.z, q11.z);
            let f11 = vec4(q8.w, q9.w, q10.w, q11.w);
            a2 += xv0.xxxx * f8 + xv0.yyyy * f9 + xv0.zzzz * f10 + xv0.wwww * f11;
            b2 += xv1.xxxx * f8 + xv1.yyyy * f9 + xv1.zzzz * f10 + xv1.wwww * f11;
            let f12 = vec4(q12.x, q13.x, q14.x, q15.x);
            let f13 = vec4(q12.y, q13.y, q14.y, q15.y);
            let f14 = vec4(q12.z, q13.z, q14.z, q15.z);
            let f15 = vec4(q12.w, q13.w, q14.w, q15.w);
            a3 += xv0.xxxx * f12 + xv0.yyyy * f13 + xv0.zzzz * f14 + xv0.wwww * f15;
            b3 += xv1.xxxx * f12 + xv1.yyyy * f13 + xv1.zzzz * f14 + xv1.wwww * f15;
        }
        }
    }

    // Tree-reduce the K-split lanes for each of the 32 (row, column) outputs.
    let base = lane * SR;
    red[base + 0u] = a0.x; red[base + 1u] = a0.y; red[base + 2u] = a0.z; red[base + 3u] = a0.w;
    red[base + 4u] = a1.x; red[base + 5u] = a1.y; red[base + 6u] = a1.z; red[base + 7u] = a1.w;
    red[base + 8u] = a2.x; red[base + 9u] = a2.y; red[base + 10u] = a2.z; red[base + 11u] = a2.w;
    red[base + 12u] = a3.x; red[base + 13u] = a3.y; red[base + 14u] = a3.z; red[base + 15u] = a3.w;
    red[base + 16u] = b0.x; red[base + 17u] = b0.y; red[base + 18u] = b0.z; red[base + 19u] = b0.w;
    red[base + 20u] = b1.x; red[base + 21u] = b1.y; red[base + 22u] = b1.z; red[base + 23u] = b1.w;
    red[base + 24u] = b2.x; red[base + 25u] = b2.y; red[base + 26u] = b2.z; red[base + 27u] = b2.w;
    red[base + 28u] = b3.x; red[base + 29u] = b3.y; red[base + 30u] = b3.z; red[base + 31u] = b3.w;
    workgroupBarrier();
    var stride = LANES >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            let mine = lane * SR;
            let other = (lane + stride) * SR;
            for (var c: u32 = 0u; c < 32u; c += 1u) {
                red[mine + c] += red[other + c];
            }
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if (lane < 16u) {
        y[m0 * N + n0 + lane] = red[lane];
        y[(m0 + 1u) * N + n0 + lane] = red[16u + lane];
    }
}
