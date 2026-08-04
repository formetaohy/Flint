// Folds the fused gate/up segment partials [SEGS, 2*NG] into the outputs.

override NG: u32 = 1u;
override SEGS: u32 = 2u;

const BN: u32 = 16u;
const SR: u32 = 33u;

@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> yg: array<f32>;
@group(0) @binding(2) var<storage, read_write> yu: array<f32>;

var<workgroup> red: array<f32, 128u * SR>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let col0 = wgid.x * BN;
    let c = lane % BN;
    let slot = lane / BN;
    let is_g = col0 < NG;
    let n0 = col0 - select(NG, 0u, is_g);

    red[lane * SR] = partial[slot * (2u * NG) + col0 + c];
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
        if (is_g) {
            yg[n0 + lane] = red[lane * SR];
        } else {
            yu[n0 + lane] = red[lane * SR];
        }
    }
}
