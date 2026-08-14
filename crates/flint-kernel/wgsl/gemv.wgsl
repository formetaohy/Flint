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

var<workgroup> ps: array<f32, 256>;

fn deq4(word: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((word & 255u) << 24) >> 24),
        f32(i32((word & 65280u) << 16) >> 24),
        f32(i32((word & 16711680u) << 8) >> 24),
        f32(i32((word >> 24) << 24) >> 24),
    );
}

fn deq2(word: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>((word & 65535u) << 16),
        bitcast<f32>((word >> 16) << 16),
    );
}

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
            if WDTYPE == 1u {
                let wb4 = (kb / 16u) * N + c0;
                let qv0 = w[wb4];
                let qv1 = w[wb4 + N];
                let sc = scales[(kb / GROUP) * N + c0];
                var dotp0 = dot(xv0, deq4(qv0.x)) + dot(xv1, deq4(qv0.y));
                dotp0 = dotp0 + dot(xv2, deq4(qv0.z)) + dot(xv3, deq4(qv0.w));
                var dotp1 = dot(xv4, deq4(qv1.x)) + dot(xv5, deq4(qv1.y));
                dotp1 = dotp1 + dot(xv6, deq4(qv1.z)) + dot(xv7, deq4(qv1.w));
                acc = acc + (dotp0 + dotp1) * sc;
            } else {
                let wb2 = c0 * (K / 8u) + kb / 8u;
                let av0 = w[wb2];
                let av1 = w[wb2 + 1u];
                let av2 = w[wb2 + 2u];
                let av3 = w[wb2 + 3u];
                var dotp0 = dot(xv0, vec4<f32>(deq2(av0.x), deq2(av0.y))) + dot(xv1, vec4<f32>(deq2(av0.z), deq2(av0.w)));
                dotp0 = dotp0 + dot(xv2, vec4<f32>(deq2(av1.x), deq2(av1.y))) + dot(xv3, vec4<f32>(deq2(av1.z), deq2(av1.w)));
                var dotp1 = dot(xv4, vec4<f32>(deq2(av2.x), deq2(av2.y))) + dot(xv5, vec4<f32>(deq2(av2.z), deq2(av2.w)));
                dotp1 = dotp1 + dot(xv6, vec4<f32>(deq2(av3.x), deq2(av3.y))) + dot(xv7, vec4<f32>(deq2(av3.z), deq2(av3.w)));
                acc = acc + dotp0 + dotp1;
            }
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
