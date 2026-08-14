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

fn sbyte(word: u32, b: u32) -> f32 {
    return f32(i32(((word >> (b * 8u)) & 255u) << 24u) >> 24);
}

fn deq2(word: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>((word & 65535u) << 16),
        bitcast<f32>((word >> 16) << 16),
    );
}

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
            y[gid.x] = sbyte(word, (d % 16) % 4) * scales[(d / GROUP) * ROWS + row] * SCALE;
        } else {
            var p = 0u;
            if row < SPLIT {
                p = table0[row * (DIM / 2) + d / 2];
            } else {
                p = table1[(row - SPLIT) * (DIM / 2) + d / 2];
            }
            let d2 = deq2(p);
            y[gid.x] = select(d2.x, d2.y, d % 2u == 1u) * SCALE;
        }
    }
}
