struct Pc {
    M: u32,
    N: u32,
    Y_STRIDE: u32,
    Y_OFF: u32,
    SEGS: u32,
    ACC: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn merge_gemm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let M = pc.M;
    let N = pc.N;
    let Y_STRIDE = pc.Y_STRIDE;
    let Y_OFF = pc.Y_OFF;
    let SEGS = pc.SEGS;
    let ACC = pc.ACC;
    let n0 = grid.x * 256 + lid.x;
    if n0 < N {
        let col = Y_OFF + n0;
        for (var m = 0u; m < M; m++) {
            var acc = f32(ACC) * y[m * Y_STRIDE + col];
            for (var s = 0u; s < SEGS; s++) {
                acc = acc + partial[s * (M * Y_STRIDE) + m * Y_STRIDE + col];
            }
            y[m * Y_STRIDE + col] = acc;
        }
    }
}
