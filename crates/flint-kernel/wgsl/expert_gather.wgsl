struct Pc {
    HIDDEN: u32,
    COUNT: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn expert_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let HIDDEN = pc.HIDDEN;
    let COUNT = pc.COUNT;
    if gid.x < COUNT * HIDDEN {
        let r = gid.x / HIDDEN;
        let c = gid.x % HIDDEN;
        out[r * HIDDEN + c] = x[ids[r] * HIDDEN + c];
    }
}
