struct Pc {
    N: u32,
    M: u32,
    MODE: u32,
    STRIDE: u32,
    OFFSET: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = pc.N;
    let M = pc.M;
    let MODE = pc.MODE;
    let STRIDE = pc.STRIDE;
    let OFFSET = pc.OFFSET;
    if gid.x < N {
        if MODE == 1 {
            y[gid.x] = a[gid.x] * b[(gid.x / M) * STRIDE + OFFSET + gid.x % M];
        } else {
            y[gid.x] = a[gid.x] * b[gid.x % M];
        }
    }
}
