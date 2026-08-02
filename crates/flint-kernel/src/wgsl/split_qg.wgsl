// De-interleave q_proj output [ROWS, HEADS, 2*HD] into q and gate [ROWS, HEADS, HD].

override ROWS: u32 = 1u;
override HEADS: u32 = 1u;
override HD: u32 = 1u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    let total = ROWS * HEADS * HD;
    if (i >= total) {
        return;
    }
    let d = i % HD;
    let h = (i / HD) % HEADS;
    let m = i / (HD * HEADS);
    let base = (m * HEADS + h) * (2u * HD);
    q[i] = x[base + d];
    gate[i] = x[base + HD + d];
}
