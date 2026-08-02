// Per-token Gated DeltaNet recurrence, one workgroup per value head.
// S[h, k, v] = S[h, k, v] * decay + k'[k] * delta[v]
// out[h, v] = sum_k S[h, k, v] * q'[k]
// q'/k' are l2-normalized; q' is additionally scaled by 1/sqrt(K_DIM).

override HEADS: u32 = 1u;
override K_DIM: u32 = 128u;
override V_DIM: u32 = 128u;

const EPS: f32 = 1e-6;

@group(0) @binding(0) var<storage, read> q_in: array<f32>;
@group(0) @binding(1) var<storage, read> k_in: array<f32>;
@group(0) @binding(2) var<storage, read> v_in: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
@group(0) @binding(5) var<storage, read_write> state: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> kk: array<f32, 128>;
var<workgroup> qq: array<f32, 128>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x;
    let t = lid.x;

    kk[t] = select(0.0, k_in[h * K_DIM + t], t < K_DIM);
    qq[t] = select(0.0, q_in[h * K_DIM + t], t < K_DIM);
    workgroupBarrier();

    red[t] = kk[t] * kk[t];
    workgroupBarrier();
    var size = 64u;
    loop {
        if (size == 0u) {
            break;
        }
        if (t < size) {
            red[t] += red[t + size];
        }
        workgroupBarrier();
        size >>= 1;
    }
    let k_scale = inverseSqrt(red[0] + EPS);
    workgroupBarrier();

    red[t] = qq[t] * qq[t];
    workgroupBarrier();
    size = 64u;
    loop {
        if (size == 0u) {
            break;
        }
        if (t < size) {
            red[t] += red[t + size];
        }
        workgroupBarrier();
        size >>= 1;
    }
    let q_scale = inverseSqrt(red[0] + EPS);

    kk[t] *= k_scale;
    qq[t] = qq[t] * q_scale / sqrt(f32(K_DIM));
    workgroupBarrier();

    if (t >= V_DIM) {
        return;
    }

    let decay = exp(g[h]);
    let bt = beta[h];
    let base = h * K_DIM * V_DIM;

    var kv_mem = 0.0;
    for (var k = 0u; k < K_DIM; k += 1u) {
        kv_mem += state[base + k * V_DIM + t] * decay * kk[k];
    }
    let delta = (v_in[h * V_DIM + t] - kv_mem) * bt;

    var acc = 0.0;
    for (var k = 0u; k < K_DIM; k += 1u) {
        let s = state[base + k * V_DIM + t] * decay + kk[k] * delta;
        state[base + k * V_DIM + t] = s;
        acc += s * qq[k];
    }
    out[h * V_DIM + t] = acc;
}
