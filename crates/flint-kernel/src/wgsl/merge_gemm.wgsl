// y[m, n] = sum_s partial[s, m, n]   (+ ACC * y)
// Folds the split-K partials of a gemm back into y. One workgroup covers a
// 256-column stripe of all M rows; the row loop is barrier-free streaming.
// Partials use the same column layout as gemm's y (Y_STRIDE / Y_OFF), so
// fused qkv writes merge in place too.
//
// dispatch [N/256, 1, 1]; N multiple of 256.

override M: u32 = 128u;
override N: u32 = 1u;
override Y_STRIDE: u32 = 1u;
override Y_OFF: u32 = 0u;
override SEGS: u32 = 4u;
override ACC: u32 = 0u;

const COLS: u32 = 256u;

@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(COLS)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n0 = wg.x * COLS + lid.x;
    if (n0 >= N) {
        return;
    }
    let col = Y_OFF + n0;
    for (var m: u32 = 0u; m < M; m += 1u) {
        var acc = f32(ACC) * y[m * Y_STRIDE + col];
        for (var s: u32 = 0u; s < SEGS; s += 1u) {
            acc += partial[s * (M * Y_STRIDE) + m * Y_STRIDE + col];
        }
        y[m * Y_STRIDE + col] = acc;
    }
}
