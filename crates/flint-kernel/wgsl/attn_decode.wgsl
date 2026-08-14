struct Pc {
    N_HEADS: u32,
    KV_HEADS: u32,
    HEAD_DIM: u32,
    MAX_SEQ: u32,
    SCALE: f32,
    WINDOW: u32,
    NQ_PER_KV: u32,
    STRIDE: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> args: array<u32>;

const BATCH: u32 = 16;

var<workgroup> qt: array<f32, 512>;
var<workgroup> kt: array<vec4<u32>, 2 * BATCH * 16>;
var<workgroup> scs: array<f32, 2 * BATCH * 8>;
var<workgroup> dts: array<f32, 128>;

fn deq2(word: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>((word & 65535u) << 16),
        bitcast<f32>((word >> 16) << 16),
    );
}

fn pick(word: vec4<u32>, wi: u32) -> u32 {
    return select(select(word.x, word.y, wi == 1u), select(word.z, word.w, wi == 3u), wi >= 2u);
}

@compute @workgroup_size(256, 1, 1)
fn attn_decode(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let KV_HEADS = pc.KV_HEADS;
    let HEAD_DIM = pc.HEAD_DIM;
    let MAX_SEQ = pc.MAX_SEQ;
    let SCALE = pc.SCALE;
    let NQ_PER_KV = pc.NQ_PER_KV;
    let STRIDE = pc.STRIDE;
    let pos = args[0];
    let segs = min(32u, max(1u, args[1]));
    let kv_len = pos + 1u;
    let seg_len = (kv_len + segs - 1u) / segs;
    let kvh = grid.x;
    let seg = grid.y;
    let seg_lo = seg * seg_len;
    let seg_hi = min(seg_lo + seg_len, kv_len);
    let half = HEAD_DIM / 2u;
    let t = lid.x;
    let hl = t / 64u;
    let slot = t % 64u;
    if hl < NQ_PER_KV && seg_lo < seg_hi {
        let qb = (kvh * NQ_PER_KV + hl) * HEAD_DIM;
        if slot < half {
            qt[hl * HEAD_DIM + slot] = q[qb + slot];
        }
        if half + slot < HEAD_DIM {
            qt[hl * HEAD_DIM + half + slot] = q[qb + half + slot];
        }
    }
    workgroupBarrier();
    if hl < NQ_PER_KV && slot < half && seg_lo < seg_hi {
        let cache_plane = kvh * MAX_SEQ * HEAD_DIM;
        let kb0 = (cache_plane + seg_lo * HEAD_DIM) / 8u;
        let wi = (slot % 8u) / 2u;
        let lo_byte = select(0.0, 1.0, slot % 2u == 1u);
        var m = -3.4e38;
        var s = 0.0;
        var o0 = 0.0;
        var o1 = 0.0;
        let batches = (seg_hi - seg_lo + BATCH - 1u) / BATCH;
        var phase = 0u;
        if t < 256u {
            let pj = t / 16u;
            let j = t % 16u;
            kt[pj * 16u + j] = k_cache[kb0 + pj * 16u + j];
        }
        workgroupBarrier();
        for (var b = 0u; b < batches; b++) {
            let p0 = seg_lo + b * BATCH;
            let count = min(BATCH, seg_hi - p0);
            if b + 1u < batches {
                let p1 = p0 + BATCH;
                let c1 = min(BATCH, seg_hi - p1);
                if t < 256u {
                    let pj = t / 16u;
                    let j = t % 16u;
                    if pj < c1 {
                        kt[(1u - phase) * BATCH * 16u + pj * 16u + j] = k_cache[kb0 + (p1 + pj - seg_lo) * 16u + j];
                    }
                }
            }
            if t < 64u {
                let dh = t % NQ_PER_KV;
                let pj = t / NQ_PER_KV;
                var dot = 0.0;
                if pj < count {
                    var a0 = 0.0;
                    var a1 = 0.0;
                    var a2 = 0.0;
                    var a3 = 0.0;
                    for (var j = 0u; j < HEAD_DIM / 8u; j++) {
                        let wv = kt[phase * BATCH * 16u + pj * 16u + j];
                        let d0 = deq2(wv.x);
                        let d1 = deq2(wv.y);
                        let d2 = deq2(wv.z);
                        let d3 = deq2(wv.w);
                        let j8 = j * 8u;
                        a0 = a0 + qt[dh * HEAD_DIM + j8] * d0.x + qt[dh * HEAD_DIM + j8 + 1u] * d0.y;
                        a1 = a1 + qt[dh * HEAD_DIM + j8 + 2u] * d1.x + qt[dh * HEAD_DIM + j8 + 3u] * d1.y;
                        a2 = a2 + qt[dh * HEAD_DIM + j8 + 4u] * d2.x + qt[dh * HEAD_DIM + j8 + 5u] * d2.y;
                        a3 = a3 + qt[dh * HEAD_DIM + j8 + 6u] * d3.x + qt[dh * HEAD_DIM + j8 + 7u] * d3.y;
                    }
                    dot = (a0 + a1) + (a2 + a3);
                }
                dts[t] = dot * SCALE;
            }
            workgroupBarrier();
            if t < NQ_PER_KV {
                var mh = m;
                var sh = s;
                for (var pj = 0u; pj < count; pj++) {
                    let sc = dts[pj * NQ_PER_KV + t];
                    let m_new = max(mh, sc);
                    let r = exp(mh - m_new);
                    let e = exp(sc - m_new);
                    sh = sh * r + e;
                    mh = m_new;
                    scs[phase * BATCH * 8u + pj * 4u + t] = r;
                    scs[phase * BATCH * 8u + BATCH * 4u + pj * 4u + t] = e;
                }
                m = mh;
                s = sh;
            }
            workgroupBarrier();
            let vb0 = (cache_plane + p0 * HEAD_DIM) / 8u + slot / 8u;
            var vw0_pre = pick(v_cache[vb0], wi);
            var vw1_pre = 0u;
            if slot + half < HEAD_DIM {
                vw1_pre = pick(v_cache[vb0 + HEAD_DIM / 16u], wi);
            }
            for (var pj = 0u; pj < count; pj++) {
                let r = scs[phase * BATCH * 8u + pj * 4u + hl];
                let e = scs[phase * BATCH * 8u + BATCH * 4u + pj * 4u + hl];
                let vw0 = vw0_pre;
                let vw1 = vw1_pre;
                if pj + 1u < count {
                    vw0_pre = pick(v_cache[vb0 + (pj + 1u) * HEAD_DIM / 8u], wi);
                    if slot + half < HEAD_DIM {
                        vw1_pre = pick(v_cache[vb0 + HEAD_DIM / 16u + (pj + 1u) * HEAD_DIM / 8u], wi);
                    }
                }
                let d = deq2(vw0);
                let v0 = select(d.x, d.y, lo_byte == 1.0);
                o0 = o0 * r + e * v0;
                if slot + half < HEAD_DIM {
                    let d1v = deq2(vw1);
                    let v1 = select(d1v.x, d1v.y, lo_byte == 1.0);
                    o1 = o1 * r + e * v1;
                }
            }
            phase = 1u - phase;
            workgroupBarrier();
        }
        let sbase = (kvh * 32u + seg) * (8u * STRIDE);
        scratch[sbase + hl * STRIDE + slot] = o0;
        if slot + half < HEAD_DIM {
            scratch[sbase + hl * STRIDE + slot + half] = o1;
        }
        if t < NQ_PER_KV {
            scratch[sbase + t * STRIDE + HEAD_DIM] = m;
            scratch[sbase + t * STRIDE + HEAD_DIM + 1u] = s;
        }
    }
}
