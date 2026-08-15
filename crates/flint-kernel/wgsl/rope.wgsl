struct Pc {
    HEADS: u32,
    HEAD_DIM: u32,
    ROT: u32,
    COS_STRIDE: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> cos_tbl: array<f32>;
@group(0) @binding(1) var<storage, read_write> sin_tbl: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> args: array<u32>;

var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn rope(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let HEADS = pc.HEADS;
    let HEAD_DIM = pc.HEAD_DIM;
    let ROT = pc.ROT;
    let COS_STRIDE = pc.COS_STRIDE;
    let head_base = (grid.x * HEADS + grid.y) * HEAD_DIM;
    if lid.x < HEAD_DIM {
        tile[lid.x] = x[head_base + lid.x];
    }
    workgroupBarrier();
    if lid.x < ROT {
        let half = ROT / 2;
        let c = cos_tbl[args[2 * grid.x] * COS_STRIDE + lid.x % half];
        let s = sin_tbl[args[2 * grid.x] * COS_STRIDE + lid.x % half];
        if lid.x < half {
            x[head_base + lid.x] = tile[lid.x] * c - tile[lid.x + half] * s;
        } else {
            x[head_base + lid.x] = tile[lid.x] * c + tile[lid.x - half] * s;
        }
    }
}
