// Combines the SEGS per-segment attention partials written by `attn`: for
// each (row m, kv head g, query head hl) the exact softmax over the whole
// prefix is reassembled via the standard two-pass rescale: out = sum_s
// exp(m_s - M) * o_s / sum_s exp(m_s - M) * sum_s, with M the global max.
// One workgroup per (m, g); 512 threads = 8 head slots x 64 dims.

override N_HEADS: u32 = 1u;
override KV_HEADS: u32 = 1u;
override HEAD_DIM: u32 = 256u;
// Scratch slot stride; see attn.wgsl.
override STRIDE: u32 = 258u;

const CHUNK: u32 = 64u;
const SEGS: u32 = 32u;
const NEG_INF: f32 = -3.4e38;

@group(0) @binding(0) var<storage, read> scratch: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> args: array<u32>;

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = wg.x;
    let kvh = wg.y;
    let t = lid.x;
    let hl = t / CHUNK;
    let slot = t % CHUNK;
    // Only the segments attn actually wrote carry data; the rest hold stale
    // partials and must not be folded in.
    let segs = min(SEGS, max(1u, args[1]));

    // Global max over the segments (per head).
    var mx = NEG_INF;
    for (var s = 0u; s < segs; s += 1u) {
        let base = ((m * KV_HEADS + kvh) * SEGS + s) * (8u * STRIDE) + hl * STRIDE;
        mx = max(mx, scratch[base + HEAD_DIM]);
    }

    var wsum = 0.0;
    var o0 = 0.0;
    var o1 = 0.0;
    var o2 = 0.0;
    var o3 = 0.0;
    var o4 = 0.0;
    var o5 = 0.0;
    var o6 = 0.0;
    var o7 = 0.0;
    for (var s = 0u; s < segs; s += 1u) {
        let base = ((m * KV_HEADS + kvh) * SEGS + s) * (8u * STRIDE) + hl * STRIDE;
        // Segments with no keys carry max = NEG_INF; exp(NEG_INF - M) = 0.
        let w = exp(scratch[base + HEAD_DIM] - mx);
        wsum += w * scratch[base + HEAD_DIM + 1u];
        if (slot < HEAD_DIM) {
            o0 += w * scratch[base + slot];
        }
        if (slot + CHUNK < HEAD_DIM) {
            o1 += w * scratch[base + slot + CHUNK];
        }
        if (slot + 2u * CHUNK < HEAD_DIM) {
            o2 += w * scratch[base + slot + 2u * CHUNK];
        }
        if (slot + 3u * CHUNK < HEAD_DIM) {
            o3 += w * scratch[base + slot + 3u * CHUNK];
        }
        if (slot + 4u * CHUNK < HEAD_DIM) {
            o4 += w * scratch[base + slot + 4u * CHUNK];
        }
        if (slot + 5u * CHUNK < HEAD_DIM) {
            o5 += w * scratch[base + slot + 5u * CHUNK];
        }
        if (slot + 6u * CHUNK < HEAD_DIM) {
            o6 += w * scratch[base + slot + 6u * CHUNK];
        }
        if (slot + 7u * CHUNK < HEAD_DIM) {
            o7 += w * scratch[base + slot + 7u * CHUNK];
        }
    }

    if (hl < N_HEADS / KV_HEADS) {
        let ob = (m * N_HEADS + kvh * (N_HEADS / KV_HEADS) + hl) * HEAD_DIM;
        if (slot < HEAD_DIM) {
            y[ob + slot] = o0 / wsum;
        }
        if (slot + CHUNK < HEAD_DIM) {
            y[ob + slot + CHUNK] = o1 / wsum;
        }
        if (slot + 2u * CHUNK < HEAD_DIM) {
            y[ob + slot + 2u * CHUNK] = o2 / wsum;
        }
        if (slot + 3u * CHUNK < HEAD_DIM) {
            y[ob + slot + 3u * CHUNK] = o3 / wsum;
        }
        if (slot + 4u * CHUNK < HEAD_DIM) {
            y[ob + slot + 4u * CHUNK] = o4 / wsum;
        }
        if (slot + 5u * CHUNK < HEAD_DIM) {
            y[ob + slot + 5u * CHUNK] = o5 / wsum;
        }
        if (slot + 6u * CHUNK < HEAD_DIM) {
            y[ob + slot + 6u * CHUNK] = o6 / wsum;
        }
        if (slot + 7u * CHUNK < HEAD_DIM) {
            y[ob + slot + 7u * CHUNK] = o7 / wsum;
        }
    }
}
