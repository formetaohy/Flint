// y[row, d] = a[row, d] for d < D, else b[row, d - D].

override ROWS: u32 = 1u;
override D: u32 = 1u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= ROWS * 2u * D) {
        return;
    }
    let row = i / (2u * D);
    let d = i % (2u * D);
    y[i] = select(b[row * D + d - D], a[row * D + d], d < D);
}
