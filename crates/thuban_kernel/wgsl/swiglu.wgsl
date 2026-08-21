struct Pc {
    N_ELEM: u32,
    MODE: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_ELEM = pc.N_ELEM;
    let MODE = pc.MODE;
    if gid.x < N_ELEM {
        let gv = gate[gid.x];
        if MODE == 1 {
            let v = gv * (gv * gv * 0.044715 + 1.0) * 0.7978845608028654;
            y[gid.x] = (tanh(v) + 1.0) * 0.5 * gv * up[gid.x];
        } else {
            y[gid.x] = gv / (1.0 + exp(-gv)) * up[gid.x];
        }
    }
}
