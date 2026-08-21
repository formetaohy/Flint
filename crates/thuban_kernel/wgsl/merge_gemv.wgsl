struct Pc {
    N: u32,
    SEGS: u32,
    ACC: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn merge_gemv(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let N = pc.N;
    let SEGS = pc.SEGS;
    let ACC = pc.ACC;
    let col = grid.x * 256 + lid.x;
    if col < N {
        var s = 0.0;
        for (var i = 0u; i < SEGS; i++) {
            s = s + partial[i * N + col];
        }
        y[col] = s + f32(ACC) * y[col];
    }
}
