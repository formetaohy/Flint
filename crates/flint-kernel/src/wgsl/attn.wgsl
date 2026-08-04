// Split-K fused GQA attention: one workgroup per (query row m, kv head g,
// segment s). The KV range of each (m, g) is split into SEGS segments so
// decode (m == 1) fans out over the whole GPU instead of idling 40 SMs on
// 8 workgroups. Each workgroup computes the exact partial softmax statistics
// (max, sum) and the unnormalized output of its segment and writes them to
// the scratch buffer; `merge_attn` combines the SEGS partials.
//
// Within a segment, the K and V tiles are staged in workgroup memory and
// shared by the NQ_PER_KV query heads of this kv head (GQA batching).
// HEAD_DIM in [64, 512]; NQ_PER_KV in [1, 8]. 512 threads = 8 head slots
// x 64 key slots; heads beyond NQ_PER_KV idle (their barriers still sync).
// Tiles are staged in 128-dim halves so the workgroup storage stays at
// 32 KiB regardless of HEAD_DIM.

override N_HEADS: u32 = 1u;
override KV_HEADS: u32 = 1u;
override HEAD_DIM: u32 = 256u;
override MAX_SEQ: u32 = 8192u;
override SCALE: f32 = 0.0625;
override WINDOW: u32 = 0u;
override NQ_PER_KV: u32 = 1u;
// Scratch slot stride (HEAD_DIM + 2, or the largest layer head dim when
// layers vary; keeps one shared scratch valid for every layer).
override STRIDE: u32 = 258u;

const CHUNK: u32 = 64u;
const SEGS: u32 = 32u;
// Dims staged per step; a half-tile is 64 keys x 128 dims packed bf16.
const HALF: u32 = 128u;
const STAGE: u32 = CHUNK * HALF / 2u;
const NEG_INF: f32 = -3.4e38;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(4) var<storage, read> args: array<u32>;

var<workgroup> kt: array<u32, STAGE>;
var<workgroup> vt: array<u32, STAGE>;
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

// Stages one 128-dim half of the K tile: half h covers dims [h*128, h*128+128)
// of the chunk's keys. `d2` is the dim-PAIR index (e % 64): the staged word e
// holds the cache word for dims (d2*2, d2*2+1). Out-of-range keys and dims
// beyond HEAD_DIM stage zeros.
fn stage_k(e: u32, key: u32, d2: u32, limit: u32, c0: u32, plane: u32, half: u32) {
    if (key < limit && half * HALF + d2 * 2u < HEAD_DIM) {
        let ckey = plane + (c0 + key) * HEAD_DIM + half * HALF + d2 * 2u;
        kt[e] = k_cache[ckey >> 1];
    } else {
        kt[e] = 0u;
    }
}

fn stage_v(e: u32, key: u32, d2: u32, limit: u32, c0: u32, plane: u32, half: u32) {
    if (key < limit && half * HALF + d2 * 2u < HEAD_DIM) {
        let ckey = plane + (c0 + key) * HEAD_DIM + half * HALF + d2 * 2u;
        vt[e] = v_cache[ckey >> 1];
    } else {
        vt[e] = 0u;
    }
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

    var o0 = 0.0;
    var o1 = 0.0;
    var o2 = 0.0;
    var o3 = 0.0;
    var o4 = 0.0;
    var o5 = 0.0;
    var o6 = 0.0;
    var o7 = 0.0;

    // Segment stats (max, sum) per head; empty segments stay NEG_INF/0 so
    // the merge absorbs them.
    if (slot == 0u && hl < 8u) {
        segstat[hl * 2u] = NEG_INF;
        segstat[hl * 2u + 1u] = 0.0;
    }
    workgroupBarrier();

    for (var c0: u32 = seg_lo; c0 < seg_hi; c0 += CHUNK) {
        let limit = min(CHUNK, seg_hi - c0);

        // ---- Phase 1: scores, staged in 128-dim halves. ----
        var dot = 0.0;
        for (var half = 0u; half < (HEAD_DIM + HALF - 1u) / HALF; half += 1u) {
            // 512 threads cover STAGE words in STAGE/512 steps of 512.
            for (var j: u32 = 0u; j < STAGE / 512u; j += 1u) {
                let e = t + j * 512u;
                stage_k(e, e / (HALF / 2u), e % (HALF / 2u), limit, c0, cache_plane, half);
            }
            workgroupBarrier();
            if (hl < NQ_PER_KV && c0 + slot >= win_start && c0 + slot < seg_hi) {
                let qb = q0 + hl * HEAD_DIM + half * HALF;
                for (var dd = 0u; dd < HALF && half * HALF + dd < HEAD_DIM; dd += 1u) {
                    dot += q[qb + dd] * load_k(slot * HALF + dd);
                }
            }
            workgroupBarrier();
        }
        var sc = NEG_INF;
        if (hl < NQ_PER_KV && c0 + slot >= win_start && c0 + slot < seg_hi) {
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
        // Each half contributes two of the eight per-thread columns:
        // half h feeds accumulators 2h and 2h+1 (dims h*128+slot, +64).
        for (var half = 0u; half < (HEAD_DIM + HALF - 1u) / HALF; half += 1u) {
            for (var j: u32 = 0u; j < STAGE / 512u; j += 1u) {
                let e = t + j * 512u;
                stage_v(e, e / (HALF / 2u), e % (HALF / 2u), limit, c0, cache_plane, half);
            }
            workgroupBarrier();
            if (hl < NQ_PER_KV) {
                for (var kk = 0u; kk < limit; kk += 1u) {
                    let w = scs[hl * CHUNK + kk];
                    if (half == 0u) {
                        o0 += w * load_v(kk * HALF + slot);
                        o1 += w * load_v(kk * HALF + slot + CHUNK);
                    }
                    if (half == 1u) {
                        o2 += w * load_v(kk * HALF + slot);
                        o3 += w * load_v(kk * HALF + slot + CHUNK);
                    }
                    if (half == 2u) {
                        o4 += w * load_v(kk * HALF + slot);
                        o5 += w * load_v(kk * HALF + slot + CHUNK);
                    }
                    if (half == 3u) {
                        o6 += w * load_v(kk * HALF + slot);
                        o7 += w * load_v(kk * HALF + slot + CHUNK);
                    }
                }
            }
            workgroupBarrier();
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

    // ---- Write the segment partials: [SEGS, m, kvh, 8 heads, hd] ----
    // plus per-head max/sum at the tail of each head's row.
    if (hl < NQ_PER_KV) {
        let sbase = ((m * KV_HEADS + kvh) * SEGS + seg) * (8u * STRIDE);
        if (slot < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot] = o0;
        }
        if (slot + CHUNK < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot + CHUNK] = o1;
        }
        if (slot + 2u * CHUNK < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot + 2u * CHUNK] = o2;
        }
        if (slot + 3u * CHUNK < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot + 3u * CHUNK] = o3;
        }
        if (slot + 4u * CHUNK < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot + 4u * CHUNK] = o4;
        }
        if (slot + 5u * CHUNK < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot + 5u * CHUNK] = o5;
        }
        if (slot + 6u * CHUNK < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot + 6u * CHUNK] = o6;
        }
        if (slot + 7u * CHUNK < HEAD_DIM) {
            scratch[sbase + hl * STRIDE + slot + 7u * CHUNK] = o7;
        }
        if (slot == 0u) {
            scratch[sbase + hl * STRIDE + HEAD_DIM] = segstat[hl * 2u];
            scratch[sbase + hl * STRIDE + HEAD_DIM + 1u] = segstat[hl * 2u + 1u];
        }
    }
}
