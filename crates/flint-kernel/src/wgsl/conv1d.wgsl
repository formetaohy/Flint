// Causal depthwise conv1d (kernel 4) with a rolling state buffer, silu after.
// state[c, 0..2] holds the previous three inputs for channel c.

override DIM: u32 = 1u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= DIM) {
        return;
    }
    let s0 = state[c * 3u];
    let s1 = state[c * 3u + 1u];
    let s2 = state[c * 3u + 2u];
    let xc = x[c];
    let v = weight[c * 4u] * s0 + weight[c * 4u + 1u] * s1 + weight[c * 4u + 2u] * s2 + weight[c * 4u + 3u] * xc;
    state[c * 3u] = s1;
    state[c * 3u + 1u] = s2;
    state[c * 3u + 2u] = xc;
    y[c] = v / (1.0 + exp(-v));
}
