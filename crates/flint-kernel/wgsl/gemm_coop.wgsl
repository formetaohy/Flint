enable f16;
enable wgpu_cooperative_matrix;

struct Pc {
    N: u32,
    K: u32,
    M: u32,
    SEGS: u32,
    WDTYPE: u32,
    GROUP: u32,
    ACC: u32,
    Y_STRIDE: u32,
    Y_OFF: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM: u32 = 64;
const TN: u32 = 32;
const BK: u32 = 32;

var<workgroup> xs: array<f16, 2 * 64 * 32>;
var<workgroup> ws: array<f16, 2 * 32 * 32>;
var<workgroup> yt: array<f32, 64 * 32>;

fn sbyte(word: u32, b: u32) -> f32 {
    return f32(i32(((word >> (b * 8u)) & 255u) << 24u) >> 24);
}

fn deq2(word: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>((word & 65535u) << 16),
        bitcast<f32>((word >> 16) << 16),
    );
}

@compute @workgroup_size(256, 1, 1)
fn gemm_coop(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let M = pc.M;
    let N = pc.N;
    let K = pc.K;
    let SEGS = pc.SEGS;
    let WDTYPE = pc.WDTYPE;
    let GROUP = pc.GROUP;
    let ACC = pc.ACC;
    let Y_STRIDE = pc.Y_STRIDE;
    let Y_OFF = pc.Y_OFF;
    let seg = grid.z;
    let m0 = grid.y * TM;
    let n0 = grid.x * TN;
    let ks = K / SEGS;
    let k_lo = seg * ks;
    let steps = ks / BK;
    let sg = lid.x / 32;
    let wr = sg / 2;
    let wc = sg % 2;
    let yoff = select(0u, seg * (M * Y_STRIDE), SEGS > 1);

    for (var i = 0u; i < 8u; i++) {
        let idx = lid.x + i * 256;
        let row = idx / 32;
        let col = idx % 32;
        var v = 0.0;
        if ACC == 1 {
            if m0 + row < M && n0 + col < N {
                v = y[yoff + (m0 + row) * Y_STRIDE + Y_OFF + n0 + col];
            }
        }
        yt[row * 32 + col] = v;
    }

    for (var i = 0u; i < 8u; i++) {
        let idx = lid.x + i * 256;
        let row = idx / 32;
        let col = idx % 32;
        var v = 0.0;
        if m0 + row < M && k_lo + col < K {
            v = x[(m0 + row) * K + k_lo + col];
        }
        xs[row * 32 + col] = f16(v);
    }
    for (var i = 0u; i < 4u; i++) {
        let idx = lid.x + i * 256;
        let row = idx / 32;
        let col = idx % 32;
        var v = 0.0;
        if n0 + row < N && k_lo + col < K {
            if WDTYPE == 1u {
                let k = k_lo + col;
                let word = w[((k / 16u) * N + n0 + row) * 4u + (k % 16u) / 4u];
                v = sbyte(word, (k % 16u) % 4u) * scales[(k / GROUP) * N + n0 + row];
            } else {
                let k = k_lo + col;
                let word = w[(n0 + row) * (K / 2u) + k / 2u];
                v = select(deq2(word).x, deq2(word).y, k % 2u == 1u);
            }
        }
        ws[row * 32 + col] = f16(v);
    }
    workgroupBarrier();

    var acc = coopLoadT<coop_mat16x16<f32, C>>(&yt[wr * 16 * 32 + wc * 16], 32);
    var phase = 0u;
    for (var it = 0u; it < steps; it++) {
        let p = phase;
        phase = 1 - phase;
        let k1 = k_lo + (it + 1) * BK;
        if it + 1 < steps {
            let p1 = phase * 2048;
            for (var i = 0u; i < 8u; i++) {
                let idx = lid.x + i * 256;
                let row = idx / 32;
                let col = idx % 32;
                var v = 0.0;
                if m0 + row < M && k1 + col < K {
                    v = x[(m0 + row) * K + k1 + col];
                }
                xs[p1 + row * 32 + col] = f16(v);
            }
            for (var i = 0u; i < 4u; i++) {
                let idx = lid.x + i * 256;
                let row = idx / 32;
                let col = idx % 32;
                var v = 0.0;
                if n0 + row < N && k1 + col < K {
                    if WDTYPE == 1u {
                        let k = k1 + col;
                        let word = w[((k / 16u) * N + n0 + row) * 4u + (k % 16u) / 4u];
                        v = sbyte(word, (k % 16u) % 4u) * scales[(k / GROUP) * N + n0 + row];
                    } else {
                        let k = k1 + col;
                        let word = w[(n0 + row) * (K / 2u) + k / 2u];
                        v = select(deq2(word).x, deq2(word).y, k % 2u == 1u);
                    }
                }
                ws[phase * 1024 + row * 32 + col] = f16(v);
            }
        }
        for (var kk = 0u; kk < BK / 16; kk++) {
            let ca = coopLoadT<coop_mat16x16<f16, A>>(&xs[p * 2048 + wr * 16 * 32 + kk * 16], 32);
            let cb = coopLoad<coop_mat16x16<f16, B>>(&ws[p * 1024 + wc * 16 * 32 + kk * 16], 32);
            acc = coopMultiplyAdd(ca, cb, acc);
        }
        workgroupBarrier();
    }

    coopStoreT(acc, &yt[wr * 16 * 32 + wc * 16], 32);
    workgroupBarrier();

    for (var i = 0u; i < 8u; i++) {
        let idx = lid.x + i * 256;
        let row = idx / 32;
        let col = idx % 32;
        if m0 + row < M && n0 + col < N {
            y[yoff + (m0 + row) * Y_STRIDE + Y_OFF + n0 + col] = yt[row * 32 + col];
        }
    }
}
