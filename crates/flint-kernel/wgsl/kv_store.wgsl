struct Pc {
    N_KV: u32,
    HEAD_DIM: u32,
    POOL_LEN: u32,
    MAX_PAGES: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> k_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> v_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<u32>;
@group(0) @binding(4) var<storage, read_write> args: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_table: array<u32>;

const PAGE_LEN: u32 = 32;
const PAGE_SHIFT: u32 = 5;
const PAGE_MASK: u32 = PAGE_LEN - 1;

fn pack2(a: f32, b: f32) -> u32 {
    return (bitcast<u32>(a) >> 16u) | ((bitcast<u32>(b) >> 16u) << 16u);
}

@compute @workgroup_size(256, 1, 1)
fn kv_store(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let N_KV = pc.N_KV;
    let HEAD_DIM = pc.HEAD_DIM;
    let POOL_LEN = pc.POOL_LEN;
    let MAX_PAGES = pc.MAX_PAGES;
    let half_dim = HEAD_DIM / 2;
    if gid.x < N_KV * half_dim {
        let m = grid.y;
        let h = gid.x / half_dim;
        let d2 = gid.x % half_dim;
        let pos = args[8 * m];
        let seq = args[8 * m + 1];
        let phys = block_table[seq * MAX_PAGES + (pos >> PAGE_SHIFT)] * PAGE_LEN
            + (pos & PAGE_MASK);
        let s_base = (m * N_KV + h) * HEAD_DIM + d2 * 2;
        let c_base = (h * POOL_LEN + phys) * half_dim + d2;
        k_cache[c_base] = pack2(k_src[s_base], k_src[s_base + 1]);
        v_cache[c_base] = pack2(v_src[s_base], v_src[s_base + 1]);
    }
}
