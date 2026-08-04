// K/V cache store: writes both projections into their caches in one
// dispatch (packed bf16, two values per u32).

override N_KV: u32 = 1u;
override HEAD_DIM: u32 = 1u;
override MAX_SEQ: u32 = 1u;

@group(0) @binding(0) var<storage, read> k_src: array<f32>;
@group(0) @binding(1) var<storage, read> v_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<u32>;
@group(0) @binding(4) var<storage, read> args: array<u32>;

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
    let c_base = (h * MAX_SEQ + pos + m) * half_dim + d2;

    let k_lo = bf16bits(k_src[s_base]);
    let k_hi = bf16bits(k_src[s_base + 1u]);
    k_cache[c_base] = k_lo | (k_hi << 16);
    let v_lo = bf16bits(v_src[s_base]);
    let v_hi = bf16bits(v_src[s_base + 1u]);
    v_cache[c_base] = v_lo | (v_hi << 16);
}
