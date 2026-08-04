// Folds the fused qkv segment partials [SEGS, NQ+NK+NV] into the three
// projection outputs. One workgroup per 16 columns; 128 lanes = 16 columns
// x 8 segment slots (SEGS <= 8), tree-reduced like merge_gemv.

override NQ: u32 = 1u;
override NK: u32 = 1u;
override NV: u32 = 1u;
override SEGS: u32 = 8u;

const BN: u32 = 16u;
const SR: u32 = 33u;

@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> yq: array<f32>;
@group(0) @binding(2) var<storage, read_write> yk: array<f32>;
@group(0) @binding(3) var<storage, read_write> yv: array<f32>;

var<workgroup> red: array<f32, 128u * SR>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let ntot = NQ + NK + NV;
    let col0 = wg.x * BN;
    let c = lane % BN;
    let slot = lane / BN;
    let is_q = col0 < NQ;
    let is_k = col0 >= NQ && col0 < NQ + NK;
    let n = select(select(NV, NK, is_k), NQ, is_q);
    let n0 = col0 - select(select(NQ + NK, NQ, is_k), 0u, is_q);

    red[lane * SR] = partial[slot * ntot + col0 + c];
    workgroupBarrier();
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
        if (is_q) {
            yq[n0 + lane] = red[lane * SR];
        } else if (is_k) {
            yk[n0 + lane] = red[lane * SR];
        } else {
            yv[n0 + lane] = red[lane * SR];
        }
    }
}
