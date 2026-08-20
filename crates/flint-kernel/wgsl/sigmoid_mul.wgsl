struct Pc {
    N_ELEM: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn sigmoid_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_ELEM = pc.N_ELEM;
    if gid.x < N_ELEM {
        y[gid.x] = a[gid.x] / (1.0 + exp(-b[gid.x]));
    }
}
