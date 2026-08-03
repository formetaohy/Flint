// Combines the SEGS per-segment attention partials written by `attn`:
// for each (row m, kv head g, query head hl) the exact softmax over the
// whole prefix is reassembled from per-segment (max, sum, unnormalized o)
// via the standard two-pass rescale: out = sum_s exp(m_s - M) * o_s /
// sum_s exp(m_s - M) * sum_s, with M the global max. One workgroup per
// (m, g); 512 threads = 8 head slots x 64 dims (HEAD_DIM <= 256).

override N_HEADS: u32 = 1u;
override KV_HEADS: u32 = 1u;
override HEAD_DIM: u32 = 256u;

const CHUNK: u32 = 64u;
const SEGS: u32 = 32u;
const NEG_INF: f32 = -3.4e38;

@group(0) @binding(0) var<storage, read> scratch: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = wg.x;
    let kvh = wg.y;
    let t = lid.x;
    let hl = t / CHUNK;
    let slot = t % CHUNK;
    let d0 = slot;
    let d1 = slot + CHUNK;
    let d2 = slot + 2u * CHUNK;
    let d3 = slot + 3u * CHUNK;

    // Global max over the segments (per head).
    var mx = NEG_INF;
    for (var s = 0u; s < SEGS; s += 1u) {
        let base = ((m * KV_HEADS + kvh) * SEGS + s) * (8u * (HEAD_DIM + 2u))
            + hl * (HEAD_DIM + 2u);
        mx = max(mx, scratch[base + HEAD_DIM]);
    }

    var wsum = 0.0;
    var o0 = 0.0;
    var o1 = 0.0;
    var o2 = 0.0;
    var o3 = 0.0;
    for (var s = 0u; s < SEGS; s += 1u) {
        let base = ((m * KV_HEADS + kvh) * SEGS + s) * (8u * (HEAD_DIM + 2u))
            + hl * (HEAD_DIM + 2u);
        // Segments with no keys carry max = NEG_INF; exp(NEG_INF - M) = 0.
        let w = exp(scratch[base + HEAD_DIM] - mx);
        wsum += w * scratch[base + HEAD_DIM + 1u];
        if (d0 < HEAD_DIM) {
            o0 += w * scratch[base + d0];
        }
        if (d1 < HEAD_DIM) {
            o1 += w * scratch[base + d1];
        }
        if (d2 < HEAD_DIM) {
            o2 += w * scratch[base + d2];
        }
        if (d3 < HEAD_DIM) {
            o3 += w * scratch[base + d3];
        }
    }

    if (hl < N_HEADS / KV_HEADS) {
        let ob = (m * N_HEADS + kvh * (N_HEADS / KV_HEADS) + hl) * HEAD_DIM;
        if (d0 < HEAD_DIM) {
            y[ob + d0] = o0 / wsum;
        }
        if (d1 < HEAD_DIM) {
            y[ob + d1] = o1 / wsum;
        }
        if (d2 < HEAD_DIM) {
            y[ob + d2] = o2 / wsum;
        }
        if (d3 < HEAD_DIM) {
            y[ob + d3] = o3 / wsum;
        }
    }
}
