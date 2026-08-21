struct Pc {
    MODE: u32,
    DIM: u32,
    W_DIM: u32,
    EPS: f32,
    HEADS: u32,
    ROT: u32,
    COS_STRIDE: u32,
    STRIDE: u32,
    PLE: u32,
    PLE_LAYERS: u32,
    PLE_STRIDE: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> cos_tbl: array<f32>;
@group(0) @binding(5) var<storage, read_write> sin_tbl: array<f32>;
@group(0) @binding(6) var<storage, read_write> args: array<u32>;

var<workgroup> red: array<f32, 256>;
var<workgroup> mean: array<f32, 256>;
var<workgroup> tile: array<f32, 520>;

fn blend(mode: u32, wv: f32, gv: f32, v: f32) -> f32 {
    if mode == 0u {
        return v * (1.0 + wv);
    }
    if mode == 1u {
        return v * wv * (gv / (1.0 + exp(-gv)));
    }
    if mode == 3u {
        return v * wv + gv;
    }
    return v * wv;
}

@compute @workgroup_size(256, 1, 1)
fn norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let MODE = pc.MODE;
    let DIM = pc.DIM;
    let W_DIM = pc.W_DIM;
    let EPS = pc.EPS;
    let HEADS = pc.HEADS;
    let ROT = pc.ROT;
    let COS_STRIDE = pc.COS_STRIDE;
    let STRIDE = pc.STRIDE;
    let PLE = pc.PLE;
    let PLE_LAYERS = pc.PLE_LAYERS;
    let PLE_STRIDE = pc.PLE_STRIDE;
    let row = grid.x;
    let base = select(
        row * STRIDE,
        (row / PLE_LAYERS) * PLE_STRIDE + (row % PLE_LAYERS) * DIM,
        PLE == 1,
    );
    let t = lid.x;
    let vbase = base / 4;
    let vdim = DIM / 4;
    var s = 0.0;
    var m = 0.0;
    var i = t;
    loop {
        if i >= vdim {
            break;
        }
        let xb = (vbase + i) * 4;
        let x0 = x[xb];
        let x1 = x[xb + 1];
        let x2 = x[xb + 2];
        let x3 = x[xb + 3];
        s = s + x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        if MODE == 3 {
            m = m + x0 + x1 + x2 + x3;
        }
        i = i + 256;
    }
    let tail = DIM % 4;
    if t < tail {
        let e = base + vdim * 4 + t;
        let v = x[e];
        s = s + v * v;
        if MODE == 3 {
            m = m + v;
        }
    }
    red[t] = s;
    if MODE == 3 {
        mean[t] = m;
    }
    workgroupBarrier();
    var size = 128u;
    loop {
        if size == 0 {
            break;
        }
        if t < size {
            red[t] = red[t] + red[t + size];
            if MODE == 3 {
                mean[t] = mean[t] + mean[t + size];
            }
        }
        workgroupBarrier();
        size = size / 2;
    }
    var inv = 0.0;
    var center = 0.0;
    if MODE == 3 {
        let avg = mean[0] / (f32(DIM));
        let variance = red[0] / (f32(DIM)) - avg * avg;
        inv = inverseSqrt(variance + EPS);
        center = avg;
    } else {
        inv = inverseSqrt(red[0] / (f32(DIM)) + EPS);
    }
    if MODE == 4 {
        i = t;
        loop {
            if i >= vdim {
                break;
            }
            let xb = (vbase + i) * 4;
            let wb = i * 4;
            tile[i * 4] = (x[xb] - center) * inv * w[wb];
            tile[i * 4 + 1] = (x[xb + 1] - center) * inv * w[wb + 1];
            tile[i * 4 + 2] = (x[xb + 2] - center) * inv * w[wb + 2];
            tile[i * 4 + 3] = (x[xb + 3] - center) * inv * w[wb + 3];
            i = i + 256;
        }
        if t < tail {
            let e = base + vdim * 4 + t;
            tile[vdim * 4 + t] = (x[e] - center) * inv * w[e];
        }
        workgroupBarrier();
        let pos_m = args[8 * (row / HEADS)];
        let half = ROT / 2;
        var wt = t;
        loop {
            if wt >= DIM {
                break;
            }
            let cc = cos_tbl[pos_m * COS_STRIDE + wt % half];
            let ss = sin_tbl[pos_m * COS_STRIDE + wt % half];
            if wt < ROT {
                if wt < half {
                    y[base + wt] = tile[wt] * cc - tile[wt + half] * ss;
                } else {
                    y[base + wt] = tile[wt] * cc + tile[wt - half] * ss;
                }
            } else {
                y[base + wt] = tile[wt];
            }
            wt = wt + 256;
        }
    } else {
        i = t;
        loop {
            if i >= vdim {
                break;
            }
            let xb = (vbase + i) * 4;
            let wb = select(i * 4u, (i % max(W_DIM / 4u, 1u)) * 4u, MODE == 1u);
            if MODE == 3u {
                y[xb] = blend(MODE, w[wb], gate[wb], (x[xb] - center) * inv);
                y[xb + 1u] = blend(MODE, w[wb + 1u], gate[wb + 1u], (x[xb + 1u] - center) * inv);
                y[xb + 2u] = blend(MODE, w[wb + 2u], gate[wb + 2u], (x[xb + 2u] - center) * inv);
                y[xb + 3u] = blend(MODE, w[wb + 3u], gate[wb + 3u], (x[xb + 3u] - center) * inv);
            } else if MODE == 1u {
                y[xb] = blend(MODE, w[wb], gate[xb], (x[xb] - center) * inv);
                y[xb + 1u] = blend(MODE, w[wb + 1u], gate[xb + 1u], (x[xb + 1u] - center) * inv);
                y[xb + 2u] = blend(MODE, w[wb + 2u], gate[xb + 2u], (x[xb + 2u] - center) * inv);
                y[xb + 3u] = blend(MODE, w[wb + 3u], gate[xb + 3u], (x[xb + 3u] - center) * inv);
            } else {
                y[xb] = blend(MODE, w[wb], 0.0, (x[xb] - center) * inv);
                y[xb + 1u] = blend(MODE, w[wb + 1u], 0.0, (x[xb + 1u] - center) * inv);
                y[xb + 2u] = blend(MODE, w[wb + 2u], 0.0, (x[xb + 2u] - center) * inv);
                y[xb + 3u] = blend(MODE, w[wb + 3u], 0.0, (x[xb + 3u] - center) * inv);
            }
            i = i + 256;
        }
        if t < tail {
            let e = base + vdim * 4 + t;
            if MODE == 1u {
                y[e] = blend(MODE, w[e % W_DIM], gate[e], (x[e] - center) * inv);
            } else if MODE == 3u {
                y[e] = blend(MODE, w[e], gate[e], (x[e] - center) * inv);
            } else {
                y[e] = blend(MODE, w[e], 0.0, (x[e] - center) * inv);
            }
        }
    }
}
