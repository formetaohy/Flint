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

// The same buffers viewed as vec4 lanes (16B-aligned; every normalized dim
// is a multiple of 4).
@group(0) @binding(4) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> w4v: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> g4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> y4: array<vec4<f32>>;
// RoPE tables and step args (MODE 4 only).
@group(0) @binding(8) var<storage, read> cos_tbl: array<f32>;
@group(0) @binding(9) var<storage, read> sin_tbl: array<f32>;
@group(0) @binding(10) var<storage, read> args: array<u32>;

override HEADS: u32 = 1u;
override ROT: u32 = 1u;
override COS_STRIDE: u32 = 1u;

var<workgroup> red: array<f32, 256>;
// Mean accumulator, used by layer mode only.
var<workgroup> mean: array<f32, 256>;
// Norm+rope staging (MODE 4): the normalized row before rotation.
var<workgroup> tile: array<f32, 520>;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

fn silu4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(silu(v.x), silu(v.y), silu(v.z), silu(v.w));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    let base = row * DIM;
    let t = lid.x;
    let vbase = base / 4u;
    let vdim = DIM / 4u;

    var s = 0.0;
    var m = 0.0;
    var i = t;
    loop {
        if (i >= vdim) {
            break;
        }
        let xv = x4[vbase + i];
        s += dot(xv, xv);
        if (MODE == 3u) {
            m += xv.x + xv.y + xv.z + xv.w;
        }
        i += 256u;
    }
    let tail = DIM % 4u;
    if (t < tail) {
        let e = base + vdim * 4u + t;
        let v = x[e];
        s += v * v;
        if (MODE == 3u) {
            m += v;
        }
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
    if (MODE == 4u) {
        // Norm+rope: write the normalized row to shared memory, then apply
        // the partial rotation (the rotation crosses element pairs).
        i = t;
        loop {
            if (i >= vdim) {
                break;
            }
            let xv = x4[vbase + i];
            let v = (xv - vec4(center)) * inv * w4v[i];
            tile[i * 4u] = v.x;
            tile[i * 4u + 1u] = v.y;
            tile[i * 4u + 2u] = v.z;
            tile[i * 4u + 3u] = v.w;
            i += 256u;
        }
        if (t < tail) {
            let e = base + vdim * 4u + t;
            tile[vdim * 4u + t] = (x[e] - center) * inv * w[e];
        }
        workgroupBarrier();
        let pos_m = args[0] + row / HEADS;
        let half = ROT >> 1;
        if (t < DIM) {
            let cc = cos_tbl[pos_m * COS_STRIDE + t % half];
            let ss = sin_tbl[pos_m * COS_STRIDE + t % half];
            var rv: f32;
            if (t < ROT) {
                if (t < half) {
                    rv = tile[t] * cc - tile[t + half] * ss;
                } else {
                    rv = tile[t] * cc + tile[t - half] * ss;
                }
            } else {
                rv = tile[t];
            }
            y[base + t] = rv;
        }
        return;
    }
    i = t;
    loop {
        if (i >= vdim) {
            break;
        }
        let xv = x4[vbase + i];
        var v = (xv - vec4(center)) * inv;
        switch MODE {
            case 0u: {
                v *= vec4(1.0) + w4v[i];
            }
            case 1u: {
                let wdim4 = max(W_DIM / 4u, 1u);
                v *= w4v[i % wdim4] * silu4(g4[vbase + i]);
            }
            case 3u: {
                v = v * w4v[i] + g4[i];
            }
            default: {
                v *= w4v[i];
            }
        }
        y4[vbase + i] = v;
        i += 256u;
    }
    if (t < tail) {
        let e = base + vdim * 4u + t;
        var v = (x[e] - center) * inv;
        switch MODE {
            case 0u: {
                v *= 1.0 + w[e];
            }
            case 1u: {
                v *= w[e % W_DIM] * silu(gate[e]);
            }
            case 3u: {
                v = v * w[e] + gate[e];
            }
            default: {
                v *= w[e];
            }
        }
        y[e] = v;
    }
}
