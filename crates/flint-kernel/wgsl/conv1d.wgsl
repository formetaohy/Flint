struct Pc {
    DIM: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn conv1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let DIM = pc.DIM;
    if gid.x < DIM {
        let c = gid.x;
        let s0 = state[c * 3];
        let s1 = state[c * 3 + 1];
        let s2 = state[c * 3 + 2];
        let xc = x[c];
        let v = weight[c * 4] * s0 + weight[c * 4 + 1] * s1 + weight[c * 4 + 2] * s2 + weight[c * 4 + 3] * xc;
        state[c * 3] = s1;
        state[c * 3 + 1] = s2;
        state[c * 3 + 2] = xc;
        y[c] = v / (1.0 + exp(-v));
    }
}
