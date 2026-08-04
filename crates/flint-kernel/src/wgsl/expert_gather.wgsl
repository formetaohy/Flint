// MoE row packing: copies COUNT rows of x [ROWS, HIDDEN] selected by ids
// [COUNT] into the packed tile out [ROWS, HIDDEN] at rows 0..COUNT. Rows
// COUNT..ROWS are left untouched (the following gemm may compute them; the
// scatter only consumes 0..COUNT).

override ROWS: u32 = 16u;
override HIDDEN: u32 = 1u;
override COUNT: u32 = 1u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= COUNT * HIDDEN) {
        return;
    }
    let r = i / HIDDEN;
    let c = i % HIDDEN;
    out[r * HIDDEN + c] = x[ids[r] * HIDDEN + c];
}
