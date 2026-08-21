struct Pc {
    HEADS: u32,
    K_DIM: u32,
    V_DIM: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> q_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> g: array<f32>;
@group(0) @binding(5) var<storage, read_write> state: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> kk: array<f32, 128>;
var<workgroup> qq: array<f32, 128>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn delta_recur(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let HEADS = pc.HEADS;
    let K_DIM = pc.K_DIM;
    let V_DIM = pc.V_DIM;
    let h = grid.x;
    let t = lid.x;
    if t < K_DIM {
        kk[t] = k_in[h * K_DIM + t];
        qq[t] = q_in[h * K_DIM + t];
    } else {
        kk[t] = 0.0;
        qq[t] = 0.0;
    }
    workgroupBarrier();
    red[t] = kk[t] * kk[t];
    workgroupBarrier();
    var size = 64u;
    loop {
        if size == 0 {
            break;
        }
        if t < size {
            red[t] = red[t] + red[t + size];
        }
        workgroupBarrier();
        size = size / 2;
    }
    let k_scale = inverseSqrt(red[0] + 1.0e-6);
    workgroupBarrier();
    red[t] = qq[t] * qq[t];
    workgroupBarrier();
    size = 64u;
    loop {
        if size == 0 {
            break;
        }
        if t < size {
            red[t] = red[t] + red[t + size];
        }
        workgroupBarrier();
        size = size / 2;
    }
    let q_scale = inverseSqrt(red[0] + 1.0e-6);
    kk[t] = kk[t] * k_scale;
    qq[t] = qq[t] * q_scale / sqrt(f32(K_DIM));
    workgroupBarrier();
    if t < V_DIM {
        let decay = exp(g[h]);
        let bt = beta[h];
        let base = h * K_DIM * V_DIM;
        var kv_mem = 0.0;
        for (var k = 0u; k < K_DIM; k++) {
            kv_mem = kv_mem + state[base + k * V_DIM + t] * decay * kk[k];
        }
        let delta = (v_in[h * V_DIM + t] - kv_mem) * bt;
        var acc = 0.0;
        for (var k = 0u; k < K_DIM; k++) {
            let s = state[base + k * V_DIM + t] * decay + kk[k] * delta;
            state[base + k * V_DIM + t] = s;
            acc = acc + s * qq[k];
        }
        out[h * V_DIM + t] = acc;
    }
}
