// Elementwise multiply with broadcast: y[i] = a[i] * b[i % M], M divides N.
// Covers [rows, dim] x [rows, dim] tiles and [rows, dim] x [1] scalars.

override N: u32 = 1u;
override M: u32 = 1u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N) {
        return;
    }
    y[i] = a[i] * b[i % M];
}
