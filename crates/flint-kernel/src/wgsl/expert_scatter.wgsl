// MoE weighted accumulation: acc[ids[i]] += weights[i] * src[i] over the
// packed rows 0..COUNT. The accumulator is zeroed before the block's scatters.

override HIDDEN: u32 = 1u;
override COUNT: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= COUNT * HIDDEN) {
        return;
    }
    let r = i / HIDDEN;
    let c = i % HIDDEN;
    acc[ids[r] * HIDDEN + c] += weights[r] * src[i];
}
