// Split-K fused GQA attention: one workgroup per (query row m, kv head g,
// segment s). The KV range of each (m, g) is split into SEGS segments so
// decode (m == 1) fans out over the whole GPU instead of idling 40 SMs on
// 8 workgroups. Each workgroup computes the exact partial softmax statistics
// (max, sum) and the unnormalized output of its segment and writes them to
// the scratch buffer; `merge_attn` combines the SEGS partials.
//
// Within a segment, the K and V tiles are staged in workgroup memory and
// shared by the NQ_PER_KV query heads of this kv head (GQA batching).
// HEAD_DIM in [64, 256]; NQ_PER_KV in [1, 8]. 512 threads = 8 head slots
// x 64 key slots; heads beyond NQ_PER_KV idle (their barriers still sync).

override N_HEADS: u32 = 1u;
override KV_HEADS: u32 = 1u;
override HEAD_DIM: u32 = 256u;
override MAX_SEQ: u32 = 8192u;
override SCALE: f32 = 0.0625;
override WINDOW: u32 = 0u;
override NQ_PER_KV: u32 = 1u;

const CHUNK: u32 = 64u;
const SEGS: u32 = 32u;
const NEG_INF: f32 = -3.4e38;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(4) var<storage, read> args: array<u32>;

var<workgroup> kt: array<u32, CHUNK * HEAD_DIM / 2u>;
var<workgroup> vt: array<u32, CHUNK * HEAD_DIM / 2u>;
var<workgroup> scs: array<f32, 8u * CHUNK>;
var<workgroup> red: array<f32, 8u * CHUNK>;
// Segment softmax stats per head (max, sum); kept out of `red` because the
// tree reduction uses every slot of the red region.
var<workgroup> segstat: array<f32, 16u>;

fn bf16f(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

fn load_k(e: u32) -> f32 {
    let word = kt[e >> 1];
    return bf16f(select(word >> 16, word & 0xFFFFu, (e & 1u) == 0u));
}

fn load_v(e: u32) -> f32 {
    let word = vt[e >> 1];
    return bf16f(select(word >> 16, word & 0xFFFFu, (e & 1u) == 0u));
}

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pos = args[0];
    let m = wg.x;
    let kvh = wg.y;
    let seg = wg.z;
    let qpos = pos + m;
    let kv_len = qpos + 1u;
    var win_start = 0u;
    if (WINDOW != 0u && kv_len > WINDOW) {
        win_start = kv_len - WINDOW;
    }
    // This workgroup's key range: [seg * seg_len, min((seg+1) * seg_len, kv_len)).
    let seg_len = (kv_len + SEGS - 1u) / SEGS;
    let seg_lo = seg * seg_len;
    let seg_hi = min(seg_lo + seg_len, kv_len);
    let q0 = (m * N_HEADS + kvh * NQ_PER_KV) * HEAD_DIM;
    let cache_plane = kvh * MAX_SEQ * HEAD_DIM;
    let t = lid.x;

    let hl = t / CHUNK;
    let slot = t % CHUNK;
    let d0 = slot;
    let d1 = slot + CHUNK;
    let d2 = slot + 2u * CHUNK;
    let d3 = slot + 3u * CHUNK;

    var o0 = 0.0;
    var o1 = 0.0;
    var o2 = 0.0;
    var o3 = 0.0;

    // Segment stats (max, sum) per head; empty segments stay NEG_INF/0 so
    // the merge absorbs them.
    if (slot == 0u && hl < 8u) {
        segstat[hl * 2u] = NEG_INF;
        segstat[hl * 2u + 1u] = 0.0;
    }
    workgroupBarrier();

    for (var c0: u32 = seg_lo; c0 < seg_hi; c0 += CHUNK) {
        let limit = min(CHUNK, seg_hi - c0);

        // ---- Stage the K and V tiles. ----
        for (var j: u32 = 0u; j < CHUNK * HEAD_DIM / 2u / 256u; j += 1u) {
            let e = t + j * 256u;
            let key = e / (HEAD_DIM / 2u);
            let d = (e % (HEAD_DIM / 2u)) * 2u;
            if (key < limit) {
                kt[e] = k_cache[(cache_plane + (c0 + key) * HEAD_DIM + d) >> 1];
                vt[e] = v_cache[(cache_plane + (c0 + key) * HEAD_DIM + d) >> 1];
            } else {
                kt[e] = 0u;
                vt[e] = 0u;
            }
        }
        workgroupBarrier();

        // ---- Phase 1: scores (masked by window and segment tail). ----
        var sc = NEG_INF;
        if (hl < NQ_PER_KV && c0 + slot >= win_start && c0 + slot < seg_hi) {
            let qb = q0 + hl * HEAD_DIM;
            var dot = 0.0;
            for (var dd = 0u; dd < HEAD_DIM; dd += 1u) {
                dot += q[qb + dd] * load_k(slot * HEAD_DIM + dd);
            }
            sc = dot * SCALE;
        }
        scs[hl * CHUNK + slot] = sc;
        red[hl * CHUNK + slot] = sc;
        workgroupBarrier();

        // ---- Phase 2: chunk max, then chunk sum (per head). ----
        var size = 32u;
        loop {
            if (size == 0u) {
                break;
            }
            if (slot < size) {
                let base = hl * CHUNK + slot;
                red[base] = max(red[base], red[base + size]);
            }
            workgroupBarrier();
            size >>= 1u;
        }
        let c_max = red[hl * CHUNK];
        // Every thread must finish reading the reduced max before any thread
        // overwrites red with the exp values below; without this barrier the
        // overwrite races the read (llvmpipe exposes it, GPUs hide it).
        workgroupBarrier();
        var e = 0.0;
        if (hl < NQ_PER_KV && c0 + slot >= win_start && c0 + slot < seg_hi) {
            // c_max may be NEG_INF when the whole chunk is masked (empty
            // segment); exp(NEG_INF - NEG_INF) is NaN, so guard on sc.
            e = exp(sc - c_max);
            if (sc == NEG_INF) {
                e = 0.0;
            }
        }
        scs[hl * CHUNK + slot] = e;
        red[hl * CHUNK + slot] = e;
        workgroupBarrier();
        size = 32u;
        loop {
            if (size == 0u) {
                break;
            }
            if (slot < size) {
                let base = hl * CHUNK + slot;
                red[base] += red[base + size];
            }
            workgroupBarrier();
            size >>= 1u;
        }
        let c_sum = red[hl * CHUNK];

        // ---- Phase 3: fold this chunk into the segment partials. ----
        if (hl < NQ_PER_KV) {
            for (var kk = 0u; kk < limit; kk += 1u) {
                let w = scs[hl * CHUNK + kk];
                if (d0 < HEAD_DIM) {
                    o0 += w * load_v(kk * HEAD_DIM + d0);
                }
                if (d1 < HEAD_DIM) {
                    o1 += w * load_v(kk * HEAD_DIM + d1);
                }
                if (d2 < HEAD_DIM) {
                    o2 += w * load_v(kk * HEAD_DIM + d2);
                }
                if (d3 < HEAD_DIM) {
                    o3 += w * load_v(kk * HEAD_DIM + d3);
                }
            }
        }
        // Fold the chunk stats into the running segment stats (exact max;
        // the sum rescales by the max delta).
        if (hl < NQ_PER_KV && slot == 0u) {
            let prev_max = segstat[hl * 2u];
            let prev_sum = segstat[hl * 2u + 1u];
            if (prev_max == NEG_INF) {
                segstat[hl * 2u] = c_max;
                segstat[hl * 2u + 1u] = c_sum;
            } else if (c_max > prev_max) {
                segstat[hl * 2u + 1u] = prev_sum * exp(prev_max - c_max) + c_sum;
                segstat[hl * 2u] = c_max;
            } else if (c_max > NEG_INF / 2.0) {
                segstat[hl * 2u + 1u] = prev_sum + c_sum * exp(c_max - prev_max);
            }
            // else: this chunk is fully masked; stats unchanged.
        }
        workgroupBarrier();
    }

    // ---- Write the segment partials: [SEGS, m, kvh, 4 heads, hd] ----
    // plus per-head max/sum at the tail of each head's row.
    if (hl < NQ_PER_KV) {
        let sbase = ((m * KV_HEADS + kvh) * SEGS + seg) * (8u * (HEAD_DIM + 2u));
        if (d0 < HEAD_DIM) {
            scratch[sbase + hl * (HEAD_DIM + 2u) + d0] = o0;
        }
        if (d1 < HEAD_DIM) {
            scratch[sbase + hl * (HEAD_DIM + 2u) + d1] = o1;
        }
        if (d2 < HEAD_DIM) {
            scratch[sbase + hl * (HEAD_DIM + 2u) + d2] = o2;
        }
        if (d3 < HEAD_DIM) {
            scratch[sbase + hl * (HEAD_DIM + 2u) + d3] = o3;
        }
        if (slot == 0u) {
            scratch[sbase + hl * (HEAD_DIM + 2u) + HEAD_DIM] = segstat[hl * 2u];
            scratch[sbase + hl * (HEAD_DIM + 2u) + HEAD_DIM + 1u] = segstat[hl * 2u + 1u];
        }
    }
}
