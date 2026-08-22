enable f16;
enable wgpu_cooperative_matrix;

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

@group(0) @binding(0) var<storage, read> xf: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM: u32 = 128;
const TN: u32 = 128;
const BK: u32 = 32;

var<workgroup> ws: array<f16, 2 * TN * BK>;

fn stage_b(p: u32, k1: u32, n0: u32, lid: u32, N: u32, K: u32, ty: u32) {
    let n = lid / 2u;
    let half = (lid % 2u) * 16u;
    var wv: array<vec4<f32>, 8>;
    tile32(ty, n0 + n, k1, K, &wv);
    let base = p * (TN * BK) + n * BK + half;
    for (var q = 0u; q < 4u; q++) {
        let v = wv[half / 16u * 4u + q];
        ws[base + 4u * q] = f16(v.x);
        ws[base + 4u * q + 1u] = f16(v.y);
        ws[base + 4u * q + 2u] = f16(v.z);
        ws[base + 4u * q + 3u] = f16(v.w);
    }
}
