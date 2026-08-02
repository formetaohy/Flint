// RMSNorm over the last dimension.
// MODE 0 (offset):  out = x * inverseSqrt(mean(x^2) + eps) * (1 + w)
// MODE 1 (gated):   out = x * inverseSqrt(mean(x^2) + eps) * w * silu(gate)
// MODE 2 (direct):  out = x * inverseSqrt(mean(x^2) + eps) * w

override MODE: u32 = 0u;
override DIM: u32 = 1u;
// Weight length for gated mode (per-head norm weights repeat across the row).
override W_DIM: u32 = 1u;

const EPS: f32 = 1e-6;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 256>;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    let base = row * DIM;
    let t = lid.x;

    var s = 0.0;
    var i = t;
    loop {
        if (i >= DIM) {
            break;
        }
        let v = x[base + i];
        s += v * v;
        i += 256u;
    }
    red[t] = s;
    workgroupBarrier();
    var size = 128u;
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

    let inv = inverseSqrt(red[0] / f32(DIM) + EPS);
    i = t;
    loop {
        if (i >= DIM) {
            break;
        }
        var v = x[base + i] * inv;
        switch MODE {
            case 0u: {
                v *= 1.0 + w[i];
            }
            case 1u: {
                v *= w[i % W_DIM] * silu(gate[base + i]);
            }
            default: {
                v *= w[i];
            }
        }
        y[base + i] = v;
        i += 256u;
    }
}
