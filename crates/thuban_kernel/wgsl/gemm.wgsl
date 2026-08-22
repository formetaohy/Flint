struct Pc {
    N: u32,
    K: u32,
    M: u32,
    SEGS: u32,
    QTYPE: u32,
    ACC: u32,
    Y_STRIDE: u32,
    Y_OFF: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM: u32 = 64;
const TN: u32 = 128;
const BK: u32 = 32;
const PAD: u32 = 36;

var<workgroup> xs: array<vec4<f32>, TM * PAD / 4>;
var<workgroup> ws: array<vec4<f32>, TN * PAD / 4>;

@compute @workgroup_size(256, 1, 1)
fn gemm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let N = pc.N;
    let K = pc.K;
    let M = pc.M;
    let SEGS = pc.SEGS;
    let ty = pc.QTYPE;
    let ACC = pc.ACC;
    let Y_STRIDE = pc.Y_STRIDE;
    let Y_OFF = pc.Y_OFF;
    let seg = grid.z;
    let m0 = grid.y * TM;
    let n0 = grid.x * TN;
    let ks = K / SEGS;
    let k_lo = seg * ks;
    let steps = ks / BK;
    let r = (lid.x % 16u) * 4u;
    let c = (lid.x / 16u) * 8u;
    var acc = array<vec4<f32>, 8>();

    var k1 = k_lo;
    for (var i = 0u; i < 2u; i++) {
        let idx = lid.x + i * 256u;
        let row = idx / 8u;
        let k4 = (idx % 8u) * 4u;
        var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if m0 + row < M && k1 + k4 < K {
            v = vec4<f32>(
                x[(m0 + row) * K + k1 + k4],
                x[(m0 + row) * K + k1 + k4 + 1u],
                x[(m0 + row) * K + k1 + k4 + 2u],
                x[(m0 + row) * K + k1 + k4 + 3u],
            );
        }
        xs[(row * PAD + k4) / 4u] = v;
    }
    {
        let col = lid.x / 2u;
        let h = (lid.x % 2u) * 16u;
        var wv: array<vec4<f32>, 8>;
        tile32(ty, n0 + col, k1, K, &wv);
        if n0 + col < N {
            for (var q = 0u; q < 4u; q++) {
                ws[(col * PAD + h + 4u * q) / 4u] = wv[h / 16u * 4u + q];
            }
        }
    }
    workgroupBarrier();

    let xr0 = r * PAD / 4u;
    let wc0 = c * PAD / 4u;
    let xr1 = xr0 + PAD / 4u;
    let xr2 = xr0 + 2u * PAD / 4u;
    let xr3 = xr0 + 3u * PAD / 4u;
    let wc1 = wc0 + PAD / 4u;
    let wc2 = wc0 + 2u * PAD / 4u;
    let wc3 = wc0 + 3u * PAD / 4u;
    let wc4 = wc0 + 4u * PAD / 4u;
    let wc5 = wc0 + 5u * PAD / 4u;
    let wc6 = wc0 + 6u * PAD / 4u;
    let wc7 = wc0 + 7u * PAD / 4u;
    for (var it = 0u; it < steps; it++) {
        for (var kk = 0u; kk < 8u; kk++) {
            let xa0 = xs[xr0 + kk];
            let xa1 = xs[xr1 + kk];
            let xa2 = xs[xr2 + kk];
            let xa3 = xs[xr3 + kk];
            let wa0 = ws[wc0 + kk];
            let wa1 = ws[wc1 + kk];
            let wa2 = ws[wc2 + kk];
            let wa3 = ws[wc3 + kk];
            let wa4 = ws[wc4 + kk];
            let wa5 = ws[wc5 + kk];
            let wa6 = ws[wc6 + kk];
            let wa7 = ws[wc7 + kk];
            acc[0] = acc[0] + vec4<f32>(dot(xa0, wa0), dot(xa0, wa1), dot(xa0, wa2), dot(xa0, wa3));
            acc[1] = acc[1] + vec4<f32>(dot(xa0, wa4), dot(xa0, wa5), dot(xa0, wa6), dot(xa0, wa7));
            acc[2] = acc[2] + vec4<f32>(dot(xa1, wa0), dot(xa1, wa1), dot(xa1, wa2), dot(xa1, wa3));
            acc[3] = acc[3] + vec4<f32>(dot(xa1, wa4), dot(xa1, wa5), dot(xa1, wa6), dot(xa1, wa7));
            acc[4] = acc[4] + vec4<f32>(dot(xa2, wa0), dot(xa2, wa1), dot(xa2, wa2), dot(xa2, wa3));
            acc[5] = acc[5] + vec4<f32>(dot(xa2, wa4), dot(xa2, wa5), dot(xa2, wa6), dot(xa2, wa7));
            acc[6] = acc[6] + vec4<f32>(dot(xa3, wa0), dot(xa3, wa1), dot(xa3, wa2), dot(xa3, wa3));
            acc[7] = acc[7] + vec4<f32>(dot(xa3, wa4), dot(xa3, wa5), dot(xa3, wa6), dot(xa3, wa7));
        }
        workgroupBarrier();
        if it + 1u < steps {
            k1 = k_lo + (it + 1u) * BK;
            for (var i = 0u; i < 2u; i++) {
                let idx = lid.x + i * 256u;
                let row = idx / 8u;
                let k4 = (idx % 8u) * 4u;
                var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                if m0 + row < M && k1 + k4 < K {
                    v = vec4<f32>(
                        x[(m0 + row) * K + k1 + k4],
                        x[(m0 + row) * K + k1 + k4 + 1u],
                        x[(m0 + row) * K + k1 + k4 + 2u],
                        x[(m0 + row) * K + k1 + k4 + 3u],
                    );
                }
                xs[(row * PAD + k4) / 4u] = v;
            }
            {
                let col = lid.x / 2u;
                let h = (lid.x % 2u) * 16u;
                var wv: array<vec4<f32>, 8>;
                tile32(ty, n0 + col, k1, K, &wv);
                if n0 + col < N {
                    for (var q = 0u; q < 4u; q++) {
                        ws[(col * PAD + h + 4u * q) / 4u] = wv[h / 16u * 4u + q];
                    }
                }
            }
        }
        workgroupBarrier();
    }
    let yoff = select(0u, seg * (M * Y_STRIDE), SEGS > 1u);
    for (var i = 0u; i < 4u; i++) {
        let row = m0 + r + i;
        if row < M {
            let a0 = acc[i * 2u];
            let a1 = acc[i * 2u + 1u];
            let yb = yoff + row * Y_STRIDE + Y_OFF + n0 + c;
            if n0 + c < N {
                y[yb] = a0.x + f32(ACC) * y[yb];
            }
            if n0 + c + 1u < N {
                y[yb + 1u] = a0.y + f32(ACC) * y[yb + 1u];
            }
            if n0 + c + 2u < N {
                y[yb + 2u] = a0.z + f32(ACC) * y[yb + 2u];
            }
            if n0 + c + 3u < N {
                y[yb + 3u] = a0.w + f32(ACC) * y[yb + 3u];
            }
            if n0 + c + 4u < N {
                y[yb + 4u] = a1.x + f32(ACC) * y[yb + 4u];
            }
            if n0 + c + 5u < N {
                y[yb + 5u] = a1.y + f32(ACC) * y[yb + 5u];
            }
            if n0 + c + 6u < N {
                y[yb + 6u] = a1.z + f32(ACC) * y[yb + 6u];
            }
            if n0 + c + 7u < N {
                y[yb + 7u] = a1.w + f32(ACC) * y[yb + 7u];
            }
        }
    }
}
