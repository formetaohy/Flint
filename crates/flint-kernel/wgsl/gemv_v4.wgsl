struct Pc {
    N: u32,
    K: u32,
    WDTYPE: u32,
    GROUP: u32,
    SEGS: u32,
    ACC: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn gemv_v4(
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
    let valid = col < N;
    let col2 = col + (N + 1) / 2;
    let valid2 = valid && col2 < N;
    let seg = grid.y;
    let kb_total = (K / 32) / SEGS;
    let kb_lo = seg * kb_total;
    var acc0a = 0.0;
    var acc0b = 0.0;
    var acc1a = 0.0;
    var acc1b = 0.0;
    for (var it = 0u; it < kb_total; it++) {
        let kb0 = kb_lo + it;
        if valid {
        let xb = kb0 * 8;
        let xv0 = x[xb];
        let xv1 = x[xb + 1];
        let xv2 = x[xb + 2];
        let xv3 = x[xb + 3];
        let xv4 = x[xb + 4];
        let xv5 = x[xb + 5];
        let xv6 = x[xb + 6];
        let xv7 = x[xb + 7];
        if WDTYPE == 1 {
            let sc_idx = kb0 / (GROUP / 32);
            let sc0 = scales[sc_idx * N + col];
            let wb4 = kb0 * 2 * N + col;
            let qv0 = w[wb4];
            let qv1 = w[wb4 + N];
            let w00 = f32(i32((qv0.x & 255) << 24) >> 24);
            let w01 = f32(i32((qv0.x & 65280) << 16) >> 24);
            let w02 = f32(i32((qv0.x & 16711680) << 8) >> 24);
            let w03 = f32(i32((qv0.x >> 24) << 24) >> 24);
            let w10 = f32(i32((qv0.y & 255) << 24) >> 24);
            let w11 = f32(i32((qv0.y & 65280) << 16) >> 24);
            let w12 = f32(i32((qv0.y & 16711680) << 8) >> 24);
            let w13 = f32(i32((qv0.y >> 24) << 24) >> 24);
            let w20 = f32(i32((qv0.z & 255) << 24) >> 24);
            let w21 = f32(i32((qv0.z & 65280) << 16) >> 24);
            let w22 = f32(i32((qv0.z & 16711680) << 8) >> 24);
            let w23 = f32(i32((qv0.z >> 24) << 24) >> 24);
            let w30 = f32(i32((qv0.w & 255) << 24) >> 24);
            let w31 = f32(i32((qv0.w & 65280) << 16) >> 24);
            let w32 = f32(i32((qv0.w & 16711680) << 8) >> 24);
            let w33 = f32(i32((qv0.w >> 24) << 24) >> 24);
            let w40 = f32(i32((qv1.x & 255) << 24) >> 24);
            let w41 = f32(i32((qv1.x & 65280) << 16) >> 24);
            let w42 = f32(i32((qv1.x & 16711680) << 8) >> 24);
            let w43 = f32(i32((qv1.x >> 24) << 24) >> 24);
            let w50 = f32(i32((qv1.y & 255) << 24) >> 24);
            let w51 = f32(i32((qv1.y & 65280) << 16) >> 24);
            let w52 = f32(i32((qv1.y & 16711680) << 8) >> 24);
            let w53 = f32(i32((qv1.y >> 24) << 24) >> 24);
            let w60 = f32(i32((qv1.z & 255) << 24) >> 24);
            let w61 = f32(i32((qv1.z & 65280) << 16) >> 24);
            let w62 = f32(i32((qv1.z & 16711680) << 8) >> 24);
            let w63 = f32(i32((qv1.z >> 24) << 24) >> 24);
            let w70 = f32(i32((qv1.w & 255) << 24) >> 24);
            let w71 = f32(i32((qv1.w & 65280) << 16) >> 24);
            let w72 = f32(i32((qv1.w & 16711680) << 8) >> 24);
            let w73 = f32(i32((qv1.w >> 24) << 24) >> 24);
            var dotp0 = xv0.x * w00 + xv0.y * w01 + xv0.z * w02 + xv0.w * w03;
            dotp0 = dotp0 + xv1.x * w10 + xv1.y * w11 + xv1.z * w12 + xv1.w * w13;
            dotp0 = dotp0 + xv2.x * w20 + xv2.y * w21 + xv2.z * w22 + xv2.w * w23;
            dotp0 = dotp0 + xv3.x * w30 + xv3.y * w31 + xv3.z * w32 + xv3.w * w33;
            var dotp1 = xv4.x * w40 + xv4.y * w41 + xv4.z * w42 + xv4.w * w43;
            dotp1 = dotp1 + xv5.x * w50 + xv5.y * w51 + xv5.z * w52 + xv5.w * w53;
            dotp1 = dotp1 + xv6.x * w60 + xv6.y * w61 + xv6.z * w62 + xv6.w * w63;
            dotp1 = dotp1 + xv7.x * w70 + xv7.y * w71 + xv7.z * w72 + xv7.w * w73;
            acc0a = acc0a + dotp0 * sc0;
            acc0b = acc0b + dotp1 * sc0;
            if valid2 {
                let sc1 = scales[sc_idx * N + col2];
                let wb5 = kb0 * 2 * N + col2;
                let rv0 = w[wb5];
                let rv1 = w[wb5 + N];
                let v00 = f32(i32((rv0.x & 255) << 24) >> 24);
                let v01 = f32(i32((rv0.x & 65280) << 16) >> 24);
                let v02 = f32(i32((rv0.x & 16711680) << 8) >> 24);
                let v03 = f32(i32((rv0.x >> 24) << 24) >> 24);
                let v10 = f32(i32((rv0.y & 255) << 24) >> 24);
                let v11 = f32(i32((rv0.y & 65280) << 16) >> 24);
                let v12 = f32(i32((rv0.y & 16711680) << 8) >> 24);
                let v13 = f32(i32((rv0.y >> 24) << 24) >> 24);
                let v20 = f32(i32((rv0.z & 255) << 24) >> 24);
                let v21 = f32(i32((rv0.z & 65280) << 16) >> 24);
                let v22 = f32(i32((rv0.z & 16711680) << 8) >> 24);
                let v23 = f32(i32((rv0.z >> 24) << 24) >> 24);
                let v30 = f32(i32((rv0.w & 255) << 24) >> 24);
                let v31 = f32(i32((rv0.w & 65280) << 16) >> 24);
                let v32 = f32(i32((rv0.w & 16711680) << 8) >> 24);
                let v33 = f32(i32((rv0.w >> 24) << 24) >> 24);
                let v40 = f32(i32((rv1.x & 255) << 24) >> 24);
                let v41 = f32(i32((rv1.x & 65280) << 16) >> 24);
                let v42 = f32(i32((rv1.x & 16711680) << 8) >> 24);
                let v43 = f32(i32((rv1.x >> 24) << 24) >> 24);
                let v50 = f32(i32((rv1.y & 255) << 24) >> 24);
                let v51 = f32(i32((rv1.y & 65280) << 16) >> 24);
                let v52 = f32(i32((rv1.y & 16711680) << 8) >> 24);
                let v53 = f32(i32((rv1.y >> 24) << 24) >> 24);
                let v60 = f32(i32((rv1.z & 255) << 24) >> 24);
                let v61 = f32(i32((rv1.z & 65280) << 16) >> 24);
                let v62 = f32(i32((rv1.z & 16711680) << 8) >> 24);
                let v63 = f32(i32((rv1.z >> 24) << 24) >> 24);
                let v70 = f32(i32((rv1.w & 255) << 24) >> 24);
                let v71 = f32(i32((rv1.w & 65280) << 16) >> 24);
                let v72 = f32(i32((rv1.w & 16711680) << 8) >> 24);
                let v73 = f32(i32((rv1.w >> 24) << 24) >> 24);
                var dotp2 = xv0.x * v00 + xv0.y * v01 + xv0.z * v02 + xv0.w * v03;
                dotp2 = dotp2 + xv1.x * v10 + xv1.y * v11 + xv1.z * v12 + xv1.w * v13;
                dotp2 = dotp2 + xv2.x * v20 + xv2.y * v21 + xv2.z * v22 + xv2.w * v23;
                dotp2 = dotp2 + xv3.x * v30 + xv3.y * v31 + xv3.z * v32 + xv3.w * v33;
                var dotp3 = xv4.x * v40 + xv4.y * v41 + xv4.z * v42 + xv4.w * v43;
                dotp3 = dotp3 + xv5.x * v50 + xv5.y * v51 + xv5.z * v52 + xv5.w * v53;
                dotp3 = dotp3 + xv6.x * v60 + xv6.y * v61 + xv6.z * v62 + xv6.w * v63;
                dotp3 = dotp3 + xv7.x * v70 + xv7.y * v71 + xv7.z * v72 + xv7.w * v73;
                acc1a = acc1a + dotp2 * sc1;
                acc1b = acc1b + dotp3 * sc1;
            }
        } else {
            let wb = col * (K / 8) + kb0 * 4;
            let av0 = w[wb];
            let av1 = w[wb + 1];
            let av2 = w[wb + 2];
            let av3 = w[wb + 3];
            let w00 = bitcast<f32>((av0.x & 65535) << 16);
            let w01 = bitcast<f32>((av0.x >> 16) << 16);
            let w02 = bitcast<f32>((av0.y & 65535) << 16);
            let w03 = bitcast<f32>((av0.y >> 16) << 16);
            let w10 = bitcast<f32>((av0.z & 65535) << 16);
            let w11 = bitcast<f32>((av0.z >> 16) << 16);
            let w12 = bitcast<f32>((av0.w & 65535) << 16);
            let w13 = bitcast<f32>((av0.w >> 16) << 16);
            let w20 = bitcast<f32>((av1.x & 65535) << 16);
            let w21 = bitcast<f32>((av1.x >> 16) << 16);
            let w22 = bitcast<f32>((av1.y & 65535) << 16);
            let w23 = bitcast<f32>((av1.y >> 16) << 16);
            let w30 = bitcast<f32>((av1.z & 65535) << 16);
            let w31 = bitcast<f32>((av1.z >> 16) << 16);
            let w32 = bitcast<f32>((av1.w & 65535) << 16);
            let w33 = bitcast<f32>((av1.w >> 16) << 16);
            let w40 = bitcast<f32>((av2.x & 65535) << 16);
            let w41 = bitcast<f32>((av2.x >> 16) << 16);
            let w42 = bitcast<f32>((av2.y & 65535) << 16);
            let w43 = bitcast<f32>((av2.y >> 16) << 16);
            let w50 = bitcast<f32>((av2.z & 65535) << 16);
            let w51 = bitcast<f32>((av2.z >> 16) << 16);
            let w52 = bitcast<f32>((av2.w & 65535) << 16);
            let w53 = bitcast<f32>((av2.w >> 16) << 16);
            let w60 = bitcast<f32>((av3.x & 65535) << 16);
            let w61 = bitcast<f32>((av3.x >> 16) << 16);
            let w62 = bitcast<f32>((av3.y & 65535) << 16);
            let w63 = bitcast<f32>((av3.y >> 16) << 16);
            let w70 = bitcast<f32>((av3.z & 65535) << 16);
            let w71 = bitcast<f32>((av3.z >> 16) << 16);
            let w72 = bitcast<f32>((av3.w & 65535) << 16);
            let w73 = bitcast<f32>((av3.w >> 16) << 16);
            var dotp0 = xv0.x * w00 + xv0.y * w01 + xv0.z * w02 + xv0.w * w03;
            dotp0 = dotp0 + xv1.x * w10 + xv1.y * w11 + xv1.z * w12 + xv1.w * w13;
            dotp0 = dotp0 + xv2.x * w20 + xv2.y * w21 + xv2.z * w22 + xv2.w * w23;
            dotp0 = dotp0 + xv3.x * w30 + xv3.y * w31 + xv3.z * w32 + xv3.w * w33;
            var dotp1 = xv4.x * w40 + xv4.y * w41 + xv4.z * w42 + xv4.w * w43;
            dotp1 = dotp1 + xv5.x * w50 + xv5.y * w51 + xv5.z * w52 + xv5.w * w53;
            dotp1 = dotp1 + xv6.x * w60 + xv6.y * w61 + xv6.z * w62 + xv6.w * w63;
            dotp1 = dotp1 + xv7.x * w70 + xv7.y * w71 + xv7.z * w72 + xv7.w * w73;
            acc0a = acc0a + dotp0;
            acc0b = acc0b + dotp1;
            if valid2 {
                let wb2 = col2 * (K / 8) + kb0 * 4;
                let bv0 = w[wb2];
                let bv1 = w[wb2 + 1];
                let bv2 = w[wb2 + 2];
                let bv3 = w[wb2 + 3];
                let v00 = bitcast<f32>((bv0.x & 65535) << 16);
                let v01 = bitcast<f32>((bv0.x >> 16) << 16);
                let v02 = bitcast<f32>((bv0.y & 65535) << 16);
                let v03 = bitcast<f32>((bv0.y >> 16) << 16);
                let v10 = bitcast<f32>((bv0.z & 65535) << 16);
                let v11 = bitcast<f32>((bv0.z >> 16) << 16);
                let v12 = bitcast<f32>((bv0.w & 65535) << 16);
                let v13 = bitcast<f32>((bv0.w >> 16) << 16);
                let v20 = bitcast<f32>((bv1.x & 65535) << 16);
                let v21 = bitcast<f32>((bv1.x >> 16) << 16);
                let v22 = bitcast<f32>((bv1.y & 65535) << 16);
                let v23 = bitcast<f32>((bv1.y >> 16) << 16);
                let v30 = bitcast<f32>((bv1.z & 65535) << 16);
                let v31 = bitcast<f32>((bv1.z >> 16) << 16);
                let v32 = bitcast<f32>((bv1.w & 65535) << 16);
                let v33 = bitcast<f32>((bv1.w >> 16) << 16);
                let v40 = bitcast<f32>((bv2.x & 65535) << 16);
                let v41 = bitcast<f32>((bv2.x >> 16) << 16);
                let v42 = bitcast<f32>((bv2.y & 65535) << 16);
                let v43 = bitcast<f32>((bv2.y >> 16) << 16);
                let v50 = bitcast<f32>((bv2.z & 65535) << 16);
                let v51 = bitcast<f32>((bv2.z >> 16) << 16);
                let v52 = bitcast<f32>((bv2.w & 65535) << 16);
                let v53 = bitcast<f32>((bv2.w >> 16) << 16);
                let v60 = bitcast<f32>((bv3.x & 65535) << 16);
                let v61 = bitcast<f32>((bv3.x >> 16) << 16);
                let v62 = bitcast<f32>((bv3.y & 65535) << 16);
                let v63 = bitcast<f32>((bv3.y >> 16) << 16);
                let v70 = bitcast<f32>((bv3.z & 65535) << 16);
                let v71 = bitcast<f32>((bv3.z >> 16) << 16);
                let v72 = bitcast<f32>((bv3.w & 65535) << 16);
                let v73 = bitcast<f32>((bv3.w >> 16) << 16);
                var dotp2 = xv0.x * v00 + xv0.y * v01 + xv0.z * v02 + xv0.w * v03;
                dotp2 = dotp2 + xv1.x * v10 + xv1.y * v11 + xv1.z * v12 + xv1.w * v13;
                dotp2 = dotp2 + xv2.x * v20 + xv2.y * v21 + xv2.z * v22 + xv2.w * v23;
                dotp2 = dotp2 + xv3.x * v30 + xv3.y * v31 + xv3.z * v32 + xv3.w * v33;
                var dotp3 = xv4.x * v40 + xv4.y * v41 + xv4.z * v42 + xv4.w * v43;
                dotp3 = dotp3 + xv5.x * v50 + xv5.y * v51 + xv5.z * v52 + xv5.w * v53;
                dotp3 = dotp3 + xv6.x * v60 + xv6.y * v61 + xv6.z * v62 + xv6.w * v63;
                dotp3 = dotp3 + xv7.x * v70 + xv7.y * v71 + xv7.z * v72 + xv7.w * v73;
                acc1a = acc1a + dotp2;
                acc1b = acc1b + dotp3;
            }
        }
        }
    }
    let acc0 = acc0a + acc0b;
    let acc1 = acc1a + acc1b;
    if valid {
        if SEGS == 1 {
            y[col] = acc0 + f32(ACC) * y[col];
            if valid2 {
                y[col2] = acc1 + f32(ACC) * y[col2];
            }
        } else {
            y[seg * N + col] = acc0;
            if valid2 {
                y[seg * N + col2] = acc1;
            }
        }
    }
}
