// Sums the SEGS partial columns written by `gemv` (SEGS > 1): y[n] =
// sum_s partial[s, n]. One workgroup per 16 columns; 128 lanes = 16 columns
// x 8 segment slots (SEGS <= 8), tree-reduced.

override N: u32 = 1u;
override SEGS: u32 = 8u;

const BN: u32 = 16u;
const SR: u32 = 33u;

@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 128u * SR>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let n0 = wg.x * BN;
    let c = lane % BN;
    let slot = lane / BN;

    red[lane * SR] = partial[slot * N + n0 + c];
    workgroupBarrier();
    // Columns live 16 lanes apart, so the tree halves from 128 to 16 lanes.
    var stride = 64u;
    loop {
        if (stride < 16u) {
            break;
        }
        if (lane < stride) {
            red[lane * SR] += red[(lane + stride) * SR];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if (lane < BN) {
        y[n0 + lane] = red[lane * SR];
    }
}
