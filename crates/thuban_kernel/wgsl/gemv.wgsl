struct Pc {
    N0: u32,
    N1: u32,
    N2: u32,
    K0: u32,
    K1: u32,
    K2: u32,
    QT0: u32,
    QT1: u32,
    QT2: u32,
    AC0: u32,
    AC1: u32,
    AC2: u32,
    O0: u32,
    O1: u32,
    O2: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y0: array<f32>;
@group(0) @binding(3) var<storage, read_write> y1: array<f32>;
@group(0) @binding(4) var<storage, read_write> y2: array<f32>;
@group(0) @binding(5) var<storage, read> lut: array<u32>;

const ROWS: u32 = 8u;
const LANES: u32 = 32u;

var<workgroup> red: array<f32, ROWS * LANES>;

fn f32v4(i: u32) -> vec4<f32> {
    return vec4<f32>(
        bitcast<f32>(w[i]),
        bitcast<f32>(w[i + 1u]),
        bitcast<f32>(w[i + 2u]),
        bitcast<f32>(w[i + 3u]),
    );
}

fn f16v4(i: u32) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(w[i]), unpack2x16float(w[i + 1u]));
}

fn bf16v4(i: u32) -> vec4<f32> {
    let v0 = w[i];
    let v1 = w[i + 1u];
    return vec4<f32>(
        bitcast<f32>((v0 & 65535u) << 16u),
        bitcast<f32>((v0 >> 16u) << 16u),
        bitcast<f32>((v1 & 65535u) << 16u),
        bitcast<f32>((v1 >> 16u) << 16u),
    );
}

fn iter_q8(acc: f32, it: u32, base: u32, lane: u32, off: u32) -> f32 {
    let bb = base + it * 144u;
    let d = unpack2x16float(w[bb / 4u]).x;
    return acc + dot(x[it * 32u + lane], qi8(qu32(bb + 2u + off)) * d);
}

fn iter_f32(acc: f32, it: u32, w0: u32, lane: u32) -> f32 {
    return acc + dot(x[it * 32u + lane], f32v4(w0 + it * 128u + 4u * lane));
}

fn iter_f16(acc: f32, it: u32, w0: u32, lane: u32) -> f32 {
    return acc + dot(x[it * 32u + lane], f16v4(w0 + it * 64u + 2u * lane));
}

fn iter_bf16(acc: f32, it: u32, w0: u32, lane: u32) -> f32 {
    return acc + dot(x[it * 32u + lane], bf16v4(w0 + it * 64u + 2u * lane));
}

fn gemv_fast(a: f32, ty: u32, iters: u32, nb: u32, lane: u32, lane8: u32, off: u32) -> f32 {
    var acc = a;
    var it = 0u;
    if ty == 8u {
        let base = nb + lane8 * 36u;
        while it + 4u <= iters {
            acc = iter_q8(acc, it, base, lane, off);
            acc = iter_q8(acc, it + 1u, base, lane, off);
            acc = iter_q8(acc, it + 2u, base, lane, off);
            acc = iter_q8(acc, it + 3u, base, lane, off);
            it = it + 4u;
        }
        while it < iters {
            acc = iter_q8(acc, it, base, lane, off);
            it = it + 1u;
        }
    } else if ty == 0u {
        let w0 = nb / 4u;
        while it + 4u <= iters {
            acc = iter_f32(acc, it, w0, lane);
            acc = iter_f32(acc, it + 1u, w0, lane);
            acc = iter_f32(acc, it + 2u, w0, lane);
            acc = iter_f32(acc, it + 3u, w0, lane);
            it = it + 4u;
        }
        while it < iters {
            acc = iter_f32(acc, it, w0, lane);
            it = it + 1u;
        }
    } else if ty == 1u {
        let w0 = nb / 4u;
        while it + 4u <= iters {
            acc = iter_f16(acc, it, w0, lane);
            acc = iter_f16(acc, it + 1u, w0, lane);
            acc = iter_f16(acc, it + 2u, w0, lane);
            acc = iter_f16(acc, it + 3u, w0, lane);
            it = it + 4u;
        }
        while it < iters {
            acc = iter_f16(acc, it, w0, lane);
            it = it + 1u;
        }
    } else {
        let w0 = nb / 4u;
        while it + 4u <= iters {
            acc = iter_bf16(acc, it, w0, lane);
            acc = iter_bf16(acc, it + 1u, w0, lane);
            acc = iter_bf16(acc, it + 2u, w0, lane);
            acc = iter_bf16(acc, it + 3u, w0, lane);
            it = it + 4u;
        }
        while it < iters {
            acc = iter_bf16(acc, it, w0, lane);
            it = it + 1u;
        }
    }
    return acc;
}

fn gemv_generic(a: f32, ty: u32, row: u32, K: u32, woff: u32, lane: u32) -> f32 {
    var acc = a;
    let blocks = K / 32u;
    var b = lane;
    while b + 32u <= blocks {
        let kb = b * 32u;
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
        tile32(ty, row, kb, K, woff, &wv);
        acc = acc + dot(xv0, wv[0]) + dot(xv1, wv[1]) + dot(xv2, wv[2])
            + dot(xv3, wv[3]) + dot(xv4, wv[4]) + dot(xv5, wv[5])
            + dot(xv6, wv[6]) + dot(xv7, wv[7]);
        let kb2 = (b + 32u) * 32u;
        let xb2 = kb2 / 4u;
        var wv2: array<vec4<f32>, 8>;
        tile32(ty, row, kb2, K, woff, &wv2);
        acc = acc + dot(x[xb2], wv2[0]) + dot(x[xb2 + 1u], wv2[1])
            + dot(x[xb2 + 2u], wv2[2]) + dot(x[xb2 + 3u], wv2[3])
            + dot(x[xb2 + 4u], wv2[4]) + dot(x[xb2 + 5u], wv2[5])
            + dot(x[xb2 + 6u], wv2[6]) + dot(x[xb2 + 7u], wv2[7]);
        b = b + 64u;
    }
    while b < blocks {
        let kb = b * 32u;
        let xb = kb / 4u;
        var wv3: array<vec4<f32>, 8>;
        tile32(ty, row, kb, K, woff, &wv3);
        acc = acc + dot(x[xb], wv3[0]) + dot(x[xb + 1u], wv3[1])
            + dot(x[xb + 2u], wv3[2]) + dot(x[xb + 3u], wv3[3])
            + dot(x[xb + 4u], wv3[4]) + dot(x[xb + 5u], wv3[5])
            + dot(x[xb + 6u], wv3[6]) + dot(x[xb + 7u], wv3[7]);
        b = b + 32u;
    }
    return acc;
}

@compute @workgroup_size(256, 1, 1)
fn gemv(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let z = grid.z;
    var N = 0u;
    var K = 0u;
    var ty = 0u;
    var acc_flag = 0u;
    var woff = 0u;
    if z == 0u {
        N = pc.N0;
        K = pc.K0;
        ty = pc.QT0;
        acc_flag = pc.AC0;
        woff = pc.O0;
    } else if z == 1u {
        N = pc.N1;
        K = pc.K1;
        ty = pc.QT1;
        acc_flag = pc.AC1;
        woff = pc.O1;
    } else {
        N = pc.N2;
        K = pc.K2;
        ty = pc.QT2;
        acc_flag = pc.AC2;
        woff = pc.O2;
    }
    if N == 0u {
        return;
    }
    let t = lid.x;
    let sub = t / LANES;
    let lane = t % LANES;
    let row = grid.x * ROWS + sub;
    let nb = woff + row * wrow(ty, K);
    var acc = 0.0;
    if row < N {
        let lane8 = lane / 8u;
        let off = 4u * (lane % 8u);
        if (ty == 0u || ty == 1u || ty == 30u || ty == 8u) && K % 128u == 0u {
            acc = gemv_fast(acc, ty, K / 128u, nb, lane, lane8, off);
        } else {
            acc = gemv_generic(acc, ty, row, K, woff, lane);
        }
    }
    red[t] = acc;
    workgroupBarrier();
    if lane == 0u && row < N {
        var o = 0.0;
        for (var l = 0u; l < LANES; l++) {
            o = o + red[sub * LANES + l];
        }
        if z == 0u {
            y0[row] = o + f32(acc_flag) * y0[row];
        } else if z == 1u {
            y1[row] = o + f32(acc_flag) * y1[row];
        } else {
            y2[row] = o + f32(acc_flag) * y2[row];
        }
    }
}
