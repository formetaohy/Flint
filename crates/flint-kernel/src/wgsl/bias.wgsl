// In-place row-broadcast bias: x[i] += bias[i % DIM] over a [rows, DIM] tile.

override N_ELEM: u32 = 1u;
override DIM: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N_ELEM) {
        return;
    }
    x[i] = x[i] + bias[i % DIM];
}
