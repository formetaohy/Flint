// Partial half-rotation RoPE, applied in place on the first ROT dims.
// Layout: [M_PAD, HEADS, HEAD_DIM], positions are args[0]..+M.
// HEAD_DIM must be <= 256.

override HEADS: u32 = 1u;
override HEAD_DIM: u32 = 1u;
override ROT: u32 = 64u;
override COS_STRIDE: u32 = 64u;

@group(0) @binding(0) var<storage, read> cos_tbl: array<f32>;
@group(0) @binding(1) var<storage, read> sin_tbl: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> args: array<u32>;

var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pos = args[0];
    let m = wg.x;
    let h = wg.y;
    let t = lid.x;
    let head_base = (m * HEADS + h) * HEAD_DIM;

    if (t < HEAD_DIM) {
        tile[t] = x[head_base + t];
    }
    workgroupBarrier();

    if (t < ROT) {
        let half = ROT >> 1;
        let c = cos_tbl[(pos + m) * COS_STRIDE + t % half];
        let s = sin_tbl[(pos + m) * COS_STRIDE + t % half];
        var v: f32;
        if (t < half) {
            v = tile[t] * c - tile[t + half] * s;
        } else {
            v = tile[t] * c + tile[t - half] * s;
        }
        x[head_base + t] = v;
    }
}
