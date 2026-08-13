struct Pc {
    N: u32,
    K: u32,
    WDTYPE: u32,
    GROUP: u32,
    SEGS: u32,
    ACC: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn gemv(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let N = pc.N;
    let K = pc.K;
    let WDTYPE = pc.WDTYPE;
    let GROUP = pc.GROUP;
    let SEGS = pc.SEGS;
    let ACC = pc.ACC;
    let col = grid.x * 256 + lid.x;
    let seg = grid.y;
    let kb_total = (K / 16) / SEGS;
    let kb_lo = seg * kb_total;
    var acc = 0.0;
    if col < N {
        for (var it = 0u; it < kb_total; it++) {
            let kb0 = kb_lo + it;
            let xb = kb0 * 16;
            let x0 = x[xb];
            let x1 = x[xb + 1];
            let x2 = x[xb + 2];
            let x3 = x[xb + 3];
            let x4 = x[xb + 4];
            let x5 = x[xb + 5];
            let x6 = x[xb + 6];
            let x7 = x[xb + 7];
            let x8 = x[xb + 8];
            let x9 = x[xb + 9];
            let x10 = x[xb + 10];
            let x11 = x[xb + 11];
            let x12 = x[xb + 12];
            let x13 = x[xb + 13];
            let x14 = x[xb + 14];
            let x15 = x[xb + 15];
            if WDTYPE == 1 {
                let sc = scales[(kb0 / (GROUP / 16)) * N + col];
                let wb4 = (kb0 * N + col) * 4;
                let q0 = w[wb4];
                let q1 = w[wb4 + 1];
                let q2 = w[wb4 + 2];
                let q3 = w[wb4 + 3];
                let w00 = f32(i32((q0 & 255) << 24) >> 24);
                let w01 = f32(i32((q0 & 65280) << 16) >> 24);
                let w02 = f32(i32((q0 & 16711680) << 8) >> 24);
                let w03 = f32(i32((q0 >> 24) << 24) >> 24);
                let w10 = f32(i32((q1 & 255) << 24) >> 24);
                let w11 = f32(i32((q1 & 65280) << 16) >> 24);
                let w12 = f32(i32((q1 & 16711680) << 8) >> 24);
                let w13 = f32(i32((q1 >> 24) << 24) >> 24);
                let w20 = f32(i32((q2 & 255) << 24) >> 24);
                let w21 = f32(i32((q2 & 65280) << 16) >> 24);
                let w22 = f32(i32((q2 & 16711680) << 8) >> 24);
                let w23 = f32(i32((q2 >> 24) << 24) >> 24);
                let w30 = f32(i32((q3 & 255) << 24) >> 24);
                let w31 = f32(i32((q3 & 65280) << 16) >> 24);
                let w32 = f32(i32((q3 & 16711680) << 8) >> 24);
                let w33 = f32(i32((q3 >> 24) << 24) >> 24);
                var dotp = x0 * w00 + x1 * w01 + x2 * w02 + x3 * w03;
                dotp = dotp + x4 * w10 + x5 * w11 + x6 * w12 + x7 * w13;
                dotp = dotp + x8 * w20 + x9 * w21 + x10 * w22 + x11 * w23;
                dotp = dotp + x12 * w30 + x13 * w31 + x14 * w32 + x15 * w33;
                acc = acc + dotp * sc;
            } else {
                let wb = (col * (K / 8) + kb0 * 2) * 4;
                let a0 = w[wb];
                let a1 = w[wb + 1];
                let a2 = w[wb + 2];
                let a3 = w[wb + 3];
                let a4 = w[wb + 4];
                let a5 = w[wb + 5];
                let a6 = w[wb + 6];
                let a7 = w[wb + 7];
                let w00 = bitcast<f32>((a0 & 65535) << 16);
                let w01 = bitcast<f32>((a0 >> 16) << 16);
                let w02 = bitcast<f32>((a1 & 65535) << 16);
                let w03 = bitcast<f32>((a1 >> 16) << 16);
                let w10 = bitcast<f32>((a2 & 65535) << 16);
                let w11 = bitcast<f32>((a2 >> 16) << 16);
                let w12 = bitcast<f32>((a3 & 65535) << 16);
                let w13 = bitcast<f32>((a3 >> 16) << 16);
                let w20 = bitcast<f32>((a4 & 65535) << 16);
                let w21 = bitcast<f32>((a4 >> 16) << 16);
                let w22 = bitcast<f32>((a5 & 65535) << 16);
                let w23 = bitcast<f32>((a5 >> 16) << 16);
                let w30 = bitcast<f32>((a6 & 65535) << 16);
                let w31 = bitcast<f32>((a6 >> 16) << 16);
                let w32 = bitcast<f32>((a7 & 65535) << 16);
                let w33 = bitcast<f32>((a7 >> 16) << 16);
                var dotp = x0 * w00 + x1 * w01 + x2 * w02 + x3 * w03;
                dotp = dotp + x4 * w10 + x5 * w11 + x6 * w12 + x7 * w13;
                dotp = dotp + x8 * w20 + x9 * w21 + x10 * w22 + x11 * w23;
                dotp = dotp + x12 * w30 + x13 * w31 + x14 * w32 + x15 * w33;
                acc = acc + dotp;
            }
        }
        if SEGS == 1 {
            y[col] = acc + f32(ACC) * y[col];
        } else {
            y[seg * N + col] = acc;
        }
    }
}
