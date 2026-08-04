// Gated MLP activation: y[i] = act(gate[i]) * up[i]
// MODE 0: silu, MODE 1: gelu (pytorch tanh approximation)

override N_ELEM: u32 = 1u;
override MODE: u32 = 0u;

const SQRT_2_PI: f32 = 0.7978845608028654;

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

fn gelu_tanh(v: f32) -> f32 {
    return 0.5 * v * (1.0 + tanh(SQRT_2_PI * (v + 0.044715 * v * v * v)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N_ELEM) {
        return;
    }
    let gv = gate[i];
    var a = silu(gv);
    if (MODE == 1u) {
        a = gelu_tanh(gv);
    }
    y[i] = a * up[i];
}
