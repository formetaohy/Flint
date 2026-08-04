// Logit softcapping (Gemma 4): y[i] = CAP * tanh(y[i] / CAP), in place.

override N_ELEM: u32 = 1u;
override CAP: f32 = 30.0;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N_ELEM) {
        return;
    }
    x[i] = CAP * tanh(x[i] / CAP);
}
