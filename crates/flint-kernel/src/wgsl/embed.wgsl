// out[m, d] = SCALE * dequant(table[ids[m], d]).
// WDTYPE 0: table stored as packed bf16, scales unused.
// WDTYPE 1: table stored as group-quantized i8 with per-row group scales
//           [vocab, DIM/GROUP] (large embedding tables halve their footprint).
// SCALE is 1.0 for most models; Gemma scales its input embeddings by sqrt(dim).

override ROWS: u32 = 1u;
override DIM: u32 = 1u;
override SCALE: f32 = 1.0;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;

@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

fn bf16f(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= ROWS * DIM) {
        return;
    }
    let row = ids[i / DIM];
    // `i` is the global flat index; the table offsets are within the row.
    let d = i % DIM;
    var v: f32;
    if (WDTYPE == 1u) {
        let word = table[row * (DIM >> 2) + (d >> 2)];
        let byte = (word >> ((d & 3u) << 3u)) & 0xFFu;
        v = f32(i32(byte << 24) >> 24)
            * scales[row * (DIM / GROUP) + (d / GROUP)];
    } else {
        let d2 = d >> 1;
        let p = table[row * (DIM >> 1) + d2];
        let bits = select(p >> 16, p & 0xFFFFu, (i & 1u) == 0u);
        v = bf16f(bits);
    }
    y[i] = v * SCALE;
}
