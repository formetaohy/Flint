// Expands the q/k segments of a conv tile from N_K key heads to N_V value
// heads (repeat_interleave), matching the layout delta_recur consumes.
// x is [ROWS, CONV_DIM] with segments [q: N_K*K_DIM, k: N_K*K_DIM, v: N_V*V_DIM];
// y is [ROWS, 2*N_V*K_DIM] with q/k duplicated RATIO = N_V/N_K times.

override ROWS: u32 = 1u;
override N_K: u32 = 1u;
override N_V: u32 = 1u;
override K_DIM: u32 = 1u;
override RATIO: u32 = 1u;
override CONV_DIM: u32 = 1u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wg.x;
    if (r >= ROWS) {
        return;
    }
    let total = 2u * N_V * K_DIM;
    let base = r * CONV_DIM;
    for (var i = lid.x; i < total; i += 256u) {
        let seg = i / (N_V * K_DIM);
        let rem = i % (N_V * K_DIM);
        let h = rem / K_DIM;
        let d = rem % K_DIM;
        let src = base + seg * (N_K * K_DIM) + (h / RATIO) * K_DIM + d;
        y[r * total + i] = x[src];
    }
}
