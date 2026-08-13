struct Pc {
    ROWS: u32,
    N_K: u32,
    N_V: u32,
    K_DIM: u32,
    RATIO: u32,
    CONV_DIM: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn repeat_qk(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let ROWS = pc.ROWS;
    let N_K = pc.N_K;
    let N_V = pc.N_V;
    let K_DIM = pc.K_DIM;
    let RATIO = pc.RATIO;
    let CONV_DIM = pc.CONV_DIM;
    if grid.x < ROWS {
        let total = 2 * N_V * K_DIM;
        let base = grid.x * CONV_DIM;
        for (var i = lid.x; i < total; i++) {
            let seg = i / (N_V * K_DIM);
            let rem = i % (N_V * K_DIM);
            let h = rem / K_DIM;
            let d = rem % K_DIM;
            let src = base + seg * (N_K * K_DIM) + (h / RATIO) * K_DIM + d;
            y[grid.x * total + i] = x[src];
        }
    }
}
