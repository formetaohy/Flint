// cache[h, POS + m, d] = bf16(src[(m * N_KV + h) * HEAD_DIM + d])
// The cache stores packed bf16 (two values per u32); each thread writes one
// u32 (a consecutive d pair) so no two threads touch the same word.

override N_KV: u32 = 1u;
override HEAD_DIM: u32 = 1u;
override MAX_SEQ: u32 = 1u;

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<u32>;
@group(0) @binding(2) var<storage, read> args: array<u32>;

fn bf16bits(v: f32) -> u32 {
    return bitcast<u32>(v) >> 16;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(global_invocation_id) g: vec3<u32>) {
    let pos = args[0];
    let i = g.x;
    let half_dim = HEAD_DIM >> 1;
    if (i >= N_KV * half_dim) {
        return;
    }
    let m = wg.y;
    let h = i / half_dim;
    let d2 = i % half_dim;
    let s_base = (m * N_KV + h) * HEAD_DIM + d2 * 2u;
    let lo = bf16bits(src[s_base]);
    let hi = bf16bits(src[s_base + 1u]);
    let c_base = (h * MAX_SEQ + pos + m) * half_dim + d2;
    cache[c_base] = lo | (hi << 16);
}
