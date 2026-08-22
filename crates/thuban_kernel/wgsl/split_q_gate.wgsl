struct Pc {
    ROWS: u32,
    HEADS: u32,
    HD: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn split_q_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ROWS = pc.ROWS;
    let HEADS = pc.HEADS;
    let HD = pc.HD;
    if gid.x < ROWS * HEADS * HD {
        let d = gid.x % HD;
        let h = (gid.x / HD) % HEADS;
        let m = gid.x / (HD * HEADS);
        let base = (m * HEADS + h) * (2 * HD);
        q[gid.x] = x[base + d];
        gate[gid.x] = x[base + HD + d];
    }
}
