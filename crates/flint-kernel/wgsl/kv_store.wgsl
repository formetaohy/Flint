struct Pc {
    N_KV: u32,
    HEAD_DIM: u32,
    MAX_SEQ: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> k_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> v_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<u32>;
@group(0) @binding(4) var<storage, read_write> args: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn kv_store(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let N_KV = pc.N_KV;
    let HEAD_DIM = pc.HEAD_DIM;
    let MAX_SEQ = pc.MAX_SEQ;
    let half_dim = HEAD_DIM / 2;
    if gid.x < N_KV * half_dim {
        let m = grid.y;
        let h = gid.x / half_dim;
        let d2 = gid.x % half_dim;
        let s_base = (m * N_KV + h) * HEAD_DIM + d2 * 2;
        let c_base = (h * MAX_SEQ + args[0] + m) * half_dim + d2;
        k_cache[c_base] = (bitcast<u32>(k_src[s_base]) >> 16) | ((bitcast<u32>(k_src[s_base + 1]) >> 16) << 16);
        v_cache[c_base] = (bitcast<u32>(v_src[s_base]) >> 16) | ((bitcast<u32>(v_src[s_base + 1]) >> 16) << 16);
    }
}
