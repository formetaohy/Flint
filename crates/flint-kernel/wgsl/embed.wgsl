struct Pc {
    M: u32,
    DIM: u32,
    SCALE: f32,
    WDTYPE: u32,
    GROUP: u32,
    SPLIT: u32,
    ROWS: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> table0: array<u32>;
@group(0) @binding(2) var<storage, read_write> table1: array<u32>;
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = pc.M;
    let DIM = pc.DIM;
    let SCALE = pc.SCALE;
    let WDTYPE = pc.WDTYPE;
    let GROUP = pc.GROUP;
    let SPLIT = pc.SPLIT;
    let ROWS = pc.ROWS;
    if gid.x < M * DIM {
        let row = ids[gid.x / DIM];
        let d = gid.x % DIM;
        if WDTYPE == 1 {
            let word = table0[((d / 16) * ROWS + row) * 4 + (d % 16) / 4];
            let byte = (word >> (((d % 16) % 4) * 8)) & 255;
            let v = f32(i32(byte << 24) >> 24) * scales[(d / GROUP) * ROWS + row];
            y[gid.x] = v * SCALE;
        } else if row < SPLIT {
            let p = table0[row * (DIM / 2) + d / 2];
            if d % 2 == 0 {
                y[gid.x] = bitcast<f32>((p & 65535) << 16) * SCALE;
            } else {
                y[gid.x] = bitcast<f32>((p >> 16) << 16) * SCALE;
            }
        } else {
            let p = table1[(row - SPLIT) * (DIM / 2) + d / 2];
            if d % 2 == 0 {
                y[gid.x] = bitcast<f32>((p & 65535) << 16) * SCALE;
            } else {
                y[gid.x] = bitcast<f32>((p >> 16) << 16) * SCALE;
            }
        }
    }
}
