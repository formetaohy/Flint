struct Pc {
    HIDDEN: u32,
    COUNT: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn expert_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let HIDDEN = pc.HIDDEN;
    let COUNT = pc.COUNT;
    if gid.x < COUNT * HIDDEN {
        let r = gid.x / HIDDEN;
        let c = gid.x % HIDDEN;
        acc[ids[r] * HIDDEN + c] = acc[ids[r] * HIDDEN + c] + weights[r] * src[gid.x];
    }
}
