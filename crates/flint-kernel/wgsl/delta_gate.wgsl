struct Pc {
    HEADS: u32,
    ROW_T: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> b_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> a_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> a_log: array<f32>;
@group(0) @binding(3) var<storage, read_write> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> g: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn delta_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let HEADS = pc.HEADS;
    let ROW_T = pc.ROW_T;
    if gid.x < HEADS {
        let h = gid.x;
        let row = ROW_T * HEADS;
        beta[h] = 1.0 / (1.0 + exp(-b_in[row + h]));
        let x = a_in[row + h] + dt_bias[h];
        let sp = log(1.0 + exp(x));
        g[h] = -exp(a_log[h]) * sp;
    }
}
