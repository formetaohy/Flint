// beta[h] = sigmoid(b[ROW_T, h])
// g[h] = -exp(A_log[h]) * softplus(a[ROW_T, h] + dt_bias[h])
// HEADS must be <= 256. ROW_T selects the chunk row, avoiding unaligned
// sub-buffer bindings.

override HEADS: u32 = 1u;
override ROW_T: u32 = 0u;

@group(0) @binding(0) var<storage, read> b_in: array<f32>;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> a_log: array<f32>;
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> g: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= HEADS) {
        return;
    }
    let row = ROW_T * HEADS;
    beta[h] = 1.0 / (1.0 + exp(-b_in[row + h]));
    let x = a_in[row + h] + dt_bias[h];
    let sp = log(1.0 + exp(x));
    g[h] = -exp(a_log[h]) * sp;
}
