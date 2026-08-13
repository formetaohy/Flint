struct Pc {
    N_ELEM: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn zero_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_ELEM = pc.N_ELEM;
    if gid.x < N_ELEM {
        x[gid.x] = 0.0;
    }
}
