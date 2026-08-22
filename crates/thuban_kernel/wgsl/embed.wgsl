struct Pc {
    M: u32,
    DIM: u32,
    SCALE: f32,
    QTYPE: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = pc.M;
    let DIM = pc.DIM;
    let ty = pc.QTYPE;
    let per_row = DIM / 32u;
    if gid.x < M * per_row {
        let row = ids[gid.x / per_row];
        let kb = (gid.x % per_row) * 32u;
        var wv: array<vec4<f32>, 8>;
        tile32(ty, row, kb, DIM, 0u, &wv);
        let o = gid.x * 32u;
        for (var q = 0u; q < 8u; q++) {
            let v = wv[q] * pc.SCALE;
            y[o + 4u * q] = v.x;
            y[o + 4u * q + 1u] = v.y;
            y[o + 4u * q + 2u] = v.z;
            y[o + 4u * q + 3u] = v.w;
        }
    }
}
