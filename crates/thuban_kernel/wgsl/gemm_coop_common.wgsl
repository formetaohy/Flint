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

@group(0) @binding(0) var<storage, read> xf: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const TM: u32 = 128;
const TN: u32 = 128;
const BK: u32 = 32;

var<workgroup> ws: array<f16, 2 * TN * BK>;

fn deq4(word: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((word & 255u) << 24u) >> 24),
        f32(i32((word & 65280u) << 16u) >> 24),
        f32(i32((word & 16711680u) << 8u) >> 24),
        f32(i32((word >> 24u) << 24u) >> 24),
    );
}

fn deq2(word: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>((word & 65535u) << 16),
        bitcast<f32>((word >> 16) << 16),
    );
}

fn stage_b(p: u32, k1: u32, n0: u32, lid: u32, N: u32, K: u32, WDTYPE: u32, GROUP: u32) {
    let n = lid / 2u;
    let half = lid % 2u;
    let k0 = k1 + half * 16u;
    if WDTYPE == 1u {
        let words = vec4<u32>(
            w[((k0 / 16u) * N + n0 + n) * 4u],
            w[((k0 / 16u) * N + n0 + n) * 4u + 1u],
            w[((k0 / 16u) * N + n0 + n) * 4u + 2u],
            w[((k0 / 16u) * N + n0 + n) * 4u + 3u],
        );
        let sc = scales[(k0 / GROUP) * N + n0 + n];
        let v0 = deq4(words.x) * sc;
        let v1 = deq4(words.y) * sc;
        let v2 = deq4(words.z) * sc;
        let v3 = deq4(words.w) * sc;
        let base = p * (TN * BK) + n * BK + half * 16u;
        ws[base] = f16(v0.x);
        ws[base + 1u] = f16(v0.y);
        ws[base + 2u] = f16(v0.z);
        ws[base + 3u] = f16(v0.w);
        ws[base + 4u] = f16(v1.x);
        ws[base + 5u] = f16(v1.y);
        ws[base + 6u] = f16(v1.z);
        ws[base + 7u] = f16(v1.w);
        ws[base + 8u] = f16(v2.x);
        ws[base + 9u] = f16(v2.y);
        ws[base + 10u] = f16(v2.z);
        ws[base + 11u] = f16(v2.w);
        ws[base + 12u] = f16(v3.x);
        ws[base + 13u] = f16(v3.y);
        ws[base + 14u] = f16(v3.z);
        ws[base + 15u] = f16(v3.w);
    } else {
        let wb = (n0 + n) * (K / 2u) + k0 / 2u;
        let words0 = vec4<u32>(w[wb], w[wb + 1u], w[wb + 2u], w[wb + 3u]);
        let words1 = vec4<u32>(w[wb + 4u], w[wb + 5u], w[wb + 6u], w[wb + 7u]);
        let v0 = vec4<f32>(deq2(words0.x), deq2(words0.y));
        let v1 = vec4<f32>(deq2(words0.z), deq2(words0.w));
        let v2 = vec4<f32>(deq2(words1.x), deq2(words1.y));
        let v3 = vec4<f32>(deq2(words1.z), deq2(words1.w));
        let base = p * (TN * BK) + n * BK + half * 16u;
        ws[base] = f16(v0.x);
        ws[base + 1u] = f16(v0.y);
        ws[base + 2u] = f16(v0.z);
        ws[base + 3u] = f16(v0.w);
        ws[base + 4u] = f16(v1.x);
        ws[base + 5u] = f16(v1.y);
        ws[base + 6u] = f16(v1.z);
        ws[base + 7u] = f16(v1.w);
        ws[base + 8u] = f16(v2.x);
        ws[base + 9u] = f16(v2.y);
        ws[base + 10u] = f16(v2.z);
        ws[base + 11u] = f16(v2.w);
        ws[base + 12u] = f16(v3.x);
        ws[base + 13u] = f16(v3.y);
        ws[base + 14u] = f16(v3.z);
        ws[base + 15u] = f16(v3.w);
    }
}
