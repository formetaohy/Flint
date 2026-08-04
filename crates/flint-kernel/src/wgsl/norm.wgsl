// Normalization over the last dimension.
// MODE 0 (offset):  out = x * inv_rms * (1 + w)
// MODE 1 (gated):   out = x * inv_rms * w * silu(gate)
// MODE 2 (direct):  out = x * inv_rms * w
// MODE 3 (layer):   out = (x - mean) * inv_std * w + bias (gate slot holds bias)

override MODE: u32 = 0u;
override DIM: u32 = 1u;
// Weight length for gated mode (per-head norm weights repeat across the row).
override W_DIM: u32 = 1u;
override EPS: f32 = 1e-6;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 256>;
// Mean accumulator, used by layer mode only.
var<workgroup> mean: array<f32, 256>;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    let base = row * DIM;
    let t = lid.x;

    var s = 0.0;
    var m = 0.0;
    var i = t;
    loop {
        if (i >= DIM) {
            break;
        }
        let v = x[base + i];
        s += v * v;
        if (MODE == 3u) {
            m += v;
        }
        i += 256u;
    }
    red[t] = s;
    if (MODE == 3u) {
        mean[t] = m;
    }
    workgroupBarrier();
    var size = 128u;
    loop {
        if (size == 0u) {
            break;
        }
        if (t < size) {
            red[t] += red[t + size];
            if (MODE == 3u) {
                mean[t] += mean[t + size];
            }
        }
        workgroupBarrier();
        size >>= 1;
    }

    var inv: f32;
    var center = 0.0;
    if (MODE == 3u) {
        let avg = mean[0] / f32(DIM);
        let variance = red[0] / f32(DIM) - avg * avg;
        inv = inverseSqrt(variance + EPS);
        center = avg;
    } else {
        inv = inverseSqrt(red[0] / f32(DIM) + EPS);
    }
    i = t;
    loop {
        if (i >= DIM) {
            break;
        }
        var v = (x[base + i] - center) * inv;
        switch MODE {
            case 0u: {
                v *= 1.0 + w[i];
            }
            case 1u: {
                v *= w[i % W_DIM] * silu(gate[base + i]);
            }
            case 3u: {
                // Layer-norm bias: a [DIM] vector broadcast across rows.
                v = v * w[i] + gate[i];
            }
            default: {
                v *= w[i];
            }
        }
        y[base + i] = v;
        i += 256u;
    }
}
