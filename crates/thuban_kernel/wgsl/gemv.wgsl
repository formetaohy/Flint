struct Pc {
    N: u32,
    K: u32,
    QTYPE: u32,
    SEGS: u32,
    ACC: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> ps: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn gemv(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let N = pc.N;
    let K = pc.K;
    let ty = pc.QTYPE;
    let SEGS = pc.SEGS;
    let ACC = pc.ACC;
    let t = lid.x;
    let sub = t / 64u;
    let col = t % 64u;
    let c0 = grid.x * 64u + col;
    let ks = K / SEGS;
    let seg_lo = grid.y * ks;
    let kstart = seg_lo + ((sub * ks / 4u + 31u) & 4294967264u);
    let kend = seg_lo + select((((sub + 1u) * ks / 4u + 31u) & 4294967264u), ks, sub == 3u);
    let iters = (kend - kstart) / 32u;
    var acc = 0.0;
    if c0 < N && iters > 0u {
        for (var it = 0u; it < iters; it++) {
            let kb = kstart + it * 32u;
            let xb = kb / 4u;
            let xv0 = x[xb];
            let xv1 = x[xb + 1u];
            let xv2 = x[xb + 2u];
            let xv3 = x[xb + 3u];
            let xv4 = x[xb + 4u];
            let xv5 = x[xb + 5u];
            let xv6 = x[xb + 6u];
            let xv7 = x[xb + 7u];
            var wv: array<vec4<f32>, 8>;
            tile32(ty, c0, kb, K, &wv);
            var dotp = dot(xv0, wv[0]);
            dotp = dotp + dot(xv1, wv[1]);
            dotp = dotp + dot(xv2, wv[2]);
            dotp = dotp + dot(xv3, wv[3]);
            dotp = dotp + dot(xv4, wv[4]);
            dotp = dotp + dot(xv5, wv[5]);
            dotp = dotp + dot(xv6, wv[6]);
            dotp = dotp + dot(xv7, wv[7]);
            acc = acc + dotp;
        }
    }
    ps[t] = acc;
    workgroupBarrier();
    if t < 64u && c0 < N {
        var o = 0.0;
        for (var s = 0u; s < 4u; s++) {
            o = o + ps[s * 64u + t];
        }
        if SEGS > 1u {
            y[grid.y * N + c0] = o;
        } else {
            y[c0] = o + f32(ACC) * y[c0];
        }
    }
}
