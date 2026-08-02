// out[m, d] = SCALE * table[ids[m], d], table stored as packed bf16.
// SCALE is 1.0 for most models; Gemma scales its input embeddings by sqrt(dim).

override ROWS: u32 = 1u;
override DIM: u32 = 1u;
override SCALE: f32 = 1.0;

@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

fn bf16f(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= ROWS * DIM) {
        return;
    }
    let d2 = (i % DIM) >> 1;
    let p = table[ids[i / DIM] * (DIM >> 1) + d2];
    let bits = select(p >> 16, p & 0xFFFFu, (i & 1u) == 0u);
    y[i] = bf16f(bits) * SCALE;
}
