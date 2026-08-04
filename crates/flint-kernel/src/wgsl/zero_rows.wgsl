// Zeroes the first N_ELEM elements of a buffer (the MoE accumulator before
// each block's scatters).

override N_ELEM: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N_ELEM) {
        return;
    }
    x[i] = 0.0;
}
