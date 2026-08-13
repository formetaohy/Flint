struct Pc {
    N_ELEM: u32,
    CAP: f32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn softcap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_ELEM = pc.N_ELEM;
    let CAP = pc.CAP;
    if gid.x < N_ELEM {
        x[gid.x] = CAP * tanh(x[gid.x] / CAP);
    }
}
