struct Pc {
    N_ELEM: u32,
    DIM: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> bias_buf: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_ELEM = pc.N_ELEM;
    let DIM = pc.DIM;
    if gid.x < N_ELEM {
        x[gid.x] = x[gid.x] + bias_buf[gid.x % DIM];
    }
}
