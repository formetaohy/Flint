// SwiGLU: y[i] = silu(gate[i]) * up[i]

override N_ELEM: u32 = 1u;

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N_ELEM) {
        return;
    }
    let gv = gate[i];
    y[i] = (gv / (1.0 + exp(-gv))) * up[i];
}
