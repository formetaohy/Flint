struct Pc {
    ROWS: u32,
    D: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn concat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ROWS = pc.ROWS;
    let D = pc.D;
    if gid.x < ROWS * 2 * D {
        let row = gid.x / (2 * D);
        let d = gid.x % (2 * D);
        if d < D {
            y[gid.x] = a[row * D + d];
        } else {
            y[gid.x] = b[row * D + d - D];
        }
    }
}
