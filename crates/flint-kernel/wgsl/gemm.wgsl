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

var<workgroup> xs: array<f32, 2112>;
var<workgroup> wf: array<f32, 2112>;

@compute @workgroup_size(256, 1, 1)
fn gemm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    const TM: u32 = 32;
    const TN: u32 = 32;
    const BK: u32 = 32;
    const PAD: u32 = 33;
    let N = pc.N;
    let K = pc.K;
    let M = pc.M;
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
    let ln = lid.x;
    let r0 = (ln / 16) * 2;
    let c0 = (ln % 16) * 2;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let idx = ln + i * 256;
        var v = 0.0;
        if m0 + idx / 32 < M {
            v = x[(m0 + idx / 32) * K + k_lo + idx % 32];
        }
        xs[(idx / 32) * PAD + idx % 32] = v;
    }
    for (var i = 0u; i < 4u; i++) {
        let idx = ln + i * 256;
        var v = 0.0;
        if n0 + idx / 32 < N {
            if WDTYPE == 1 {
                let kb = (k_lo + idx % 32) / 16;
                let word = w[(kb * N + n0 + idx / 32) * 4 + ((idx % 32) % 16) / 4];
                let byte = (word >> ((((idx % 32) % 16) % 4) * 8)) & 255;
                let sc = scales[(kb / (GROUP / 16)) * N + n0 + idx / 32];
                v = f32(i32(byte << 24) >> 24) * sc;
            } else {
                let word = w[(n0 + idx / 32) * (K / 2) + (k_lo + idx % 32) / 2];
                if (idx % 32) % 2 == 0 {
                    v = bitcast<f32>((word & 65535) << 16);
                } else {
                    v = bitcast<f32>((word >> 16) << 16);
                }
            }
        }
        wf[(idx / 32) * PAD + idx % 32] = v;
    }
    workgroupBarrier();
    for (var it = 0u; it < steps; it++) {
        let phase = it % 2;
        for (var s = 0u; s < 8u; s++) {
            let xb = phase * 1056 + r0 * PAD + s * 4;
            let wb = phase * 1056 + c0 * PAD + s * 4;
            let x0 = xs[xb];
            let x1 = xs[xb + 1];
            let x2 = xs[xb + 2];
            let x3 = xs[xb + 3];
            let x4 = xs[xb + PAD];
            let x5 = xs[xb + PAD + 1];
            let x6 = xs[xb + PAD + 2];
            let x7 = xs[xb + PAD + 3];
            let w0 = wf[wb];
            let w1 = wf[wb + 1];
            let w2 = wf[wb + 2];
            let w3 = wf[wb + 3];
            let w4 = wf[wb + PAD];
            let w5 = wf[wb + PAD + 1];
            let w6 = wf[wb + PAD + 2];
            let w7 = wf[wb + PAD + 3];
            acc0 = acc0 + x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
            acc1 = acc1 + x0 * w4 + x1 * w5 + x2 * w6 + x3 * w7;
            acc2 = acc2 + x4 * w0 + x5 * w1 + x6 * w2 + x7 * w3;
            acc3 = acc3 + x4 * w4 + x5 * w5 + x6 * w6 + x7 * w7;
        }
        if it + 1 < steps {
            let p1 = (1 - phase) * 1056;
            let k1 = k_lo + (it + 1) * BK;
            for (var i = 0u; i < 4u; i++) {
                let idx = ln + i * 256;
                var v = 0.0;
                if m0 + idx / 32 < M {
                    v = x[(m0 + idx / 32) * K + k1 + idx % 32];
                }
                xs[p1 + (idx / 32) * PAD + idx % 32] = v;
            }
            for (var i = 0u; i < 4u; i++) {
                let idx = ln + i * 256;
                var v = 0.0;
                if n0 + idx / 32 < N {
                    if WDTYPE == 1 {
                        let kb = (k1 + idx % 32) / 16;
                        let word = w[(kb * N + n0 + idx / 32) * 4 + ((idx % 32) % 16) / 4];
                        let byte = (word >> ((((idx % 32) % 16) % 4) * 8)) & 255;
                        let sc = scales[(kb / (GROUP / 16)) * N + n0 + idx / 32];
                        v = f32(i32(byte << 24) >> 24) * sc;
                    } else {
                        let word = w[(n0 + idx / 32) * (K / 2) + (k1 + idx % 32) / 2];
                        if (idx % 32) % 2 == 0 {
                            v = bitcast<f32>((word & 65535) << 16);
                        } else {
                            v = bitcast<f32>((word >> 16) << 16);
                        }
                    }
                }
                wf[p1 + (idx / 32) * PAD + idx % 32] = v;
            }
            workgroupBarrier();
        }
        workgroupBarrier();
    }
    let yoff = select(0u, seg * (M * Y_STRIDE), SEGS > 1);
    if m0 + r0 < M {
        if n0 + c0 < N {
            y[yoff + (m0 + r0) * Y_STRIDE + Y_OFF + n0 + c0] = acc0 + f32(ACC) * y[yoff + (m0 + r0) * Y_STRIDE + Y_OFF + n0 + c0];
        }
        if n0 + c0 + 1 < N {
            y[yoff + (m0 + r0) * Y_STRIDE + Y_OFF + n0 + c0 + 1] = acc1 + f32(ACC) * y[yoff + (m0 + r0) * Y_STRIDE + Y_OFF + n0 + c0 + 1];
        }
    }
    if m0 + r0 + 1 < M {
        if n0 + c0 < N {
            y[yoff + (m0 + r0 + 1) * Y_STRIDE + Y_OFF + n0 + c0] = acc2 + f32(ACC) * y[yoff + (m0 + r0 + 1) * Y_STRIDE + Y_OFF + n0 + c0];
        }
        if n0 + c0 + 1 < N {
            y[yoff + (m0 + r0 + 1) * Y_STRIDE + Y_OFF + n0 + c0 + 1] = acc3 + f32(ACC) * y[yoff + (m0 + r0 + 1) * Y_STRIDE + Y_OFF + n0 + c0 + 1];
        }
    }
}
