// y[i] = a[i] * sigmoid(b[i])

override N_ELEM: u32 = 1u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N_ELEM) {
        return;
    }
    y[i] = a[i] * (1.0 / (1.0 + exp(-b[i])));
}
