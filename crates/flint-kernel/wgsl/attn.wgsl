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
@group(0) @binding(1) var<storage, read_write> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> args: array<u32>;

var<workgroup> kt: array<u32, 4160>;
var<workgroup> scs: array<f32, 512>;
var<workgroup> red: array<f32, 512>;
var<workgroup> segstat: array<f32, 16>;

@compute @workgroup_size(512, 1, 1)
fn attn(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    const CHUNK: u32 = 64;
    const SEGS: u32 = 32;
    const HALF: u32 = 128;
    const NEG_INF: f32 = -3.4e38;
    let N_HEADS = pc.N_HEADS;
    let KV_HEADS = pc.KV_HEADS;
    let HEAD_DIM = pc.HEAD_DIM;
    let MAX_SEQ = pc.MAX_SEQ;
    let SCALE = pc.SCALE;
    let WINDOW = pc.WINDOW;
    let NQ_PER_KV = pc.NQ_PER_KV;
    let STRIDE = pc.STRIDE;
    let pos = args[0];
    let segs = min(SEGS, max(1u, args[1]));
    let m = grid.x;
    let kvh = grid.y;
    let seg = grid.z;
    if seg < segs {
        let qpos = pos + m;
        let kv_len = qpos + 1;
        var win_start = 0u;
        if WINDOW != 0 && kv_len > WINDOW {
            win_start = kv_len - WINDOW;
        }
        let seg_len = (kv_len + segs - 1) / segs;
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
        var c0 = seg_lo;
        loop {
            if c0 >= seg_hi {
                break;
            }
            if c0 == seg_lo && slot == 0 && hl < 8 {
                segstat[hl * 2] = NEG_INF;
                segstat[hl * 2 + 1] = 0.0;
            }
            workgroupBarrier();
            let limit = min(CHUNK, seg_hi - c0);
            var dotp = 0.0;
            let halves = (HEAD_DIM + HALF - 1) / HALF;
            for (var half = 0u; half < halves; half++) {
                for (var j = 0u; j < 8u; j++) {
                    let e = t + j * 512;
                    let key = e / 64;
                    let d2 = e % 64;
                    if key < limit && half * HALF + d2 * 2 < HEAD_DIM {
                        let ckey = cache_plane + (c0 + key) * HEAD_DIM + half * HALF + d2 * 2;
                        kt[key * 65 + d2] = k_cache[ckey / 2];
                    } else {
                        kt[key * 65 + d2] = 0;
                    }
                }
                workgroupBarrier();
                if hl < NQ_PER_KV && c0 + slot >= win_start && c0 + slot < seg_hi {
                    let qb = q0 + hl * HEAD_DIM + half * HALF;
                    for (var dd = 0u; dd < 128u; dd++) {
                        if half * HALF + dd >= HEAD_DIM {
                            break;
                        }
                        let e = slot * HALF + dd;
                        let word = kt[slot * 65 + dd / 2];
                        if e % 2 == 0 {
                            dotp = dotp + q[qb + dd] * bitcast<f32>((word & 65535) << 16);
                        } else {
                            dotp = dotp + q[qb + dd] * bitcast<f32>((word >> 16) << 16);
                        }
                    }
                }
                workgroupBarrier();
            }
            var sc = NEG_INF;
            if hl < NQ_PER_KV && c0 + slot >= win_start && c0 + slot < seg_hi {
                sc = dotp * SCALE;
            }
            scs[hl * CHUNK + slot] = sc;
            var mx = subgroupMax(sc);
            if slot % 32 == 0 {
                red[hl * (CHUNK / 32) + slot / 32] = mx;
            }
            workgroupBarrier();
            var c_max = NEG_INF;
            if slot == 0 {
                var cm = NEG_INF;
                for (var w = 0u; w < (CHUNK / 32); w++) {
                    cm = max(cm, red[hl * (CHUNK / 32) + w]);
                }
                c_max = cm;
                red[16 + hl] = c_max;
            }
            workgroupBarrier();
            c_max = red[16 + hl];
            workgroupBarrier();
            var prev_max = NEG_INF;
            var align = 1.0;
            if hl < NQ_PER_KV && c0 != seg_lo {
                prev_max = segstat[hl * 2];
                if prev_max > NEG_INF / 2.0 {
                    if c_max > prev_max {
                        let r = exp(prev_max - c_max);
                        o0 = o0 * r;
                        o1 = o1 * r;
                        o2 = o2 * r;
                        o3 = o3 * r;
                        o4 = o4 * r;
                        o5 = o5 * r;
                        o6 = o6 * r;
                        o7 = o7 * r;
                    } else {
                        align = exp(c_max - prev_max);
                    }
                }
            }
            workgroupBarrier();
            var e = 0.0;
            if hl < NQ_PER_KV && c0 + slot >= win_start && c0 + slot < seg_hi {
                e = exp(sc - c_max) * align;
                if sc == NEG_INF {
                    e = 0.0;
                }
            }
            scs[hl * CHUNK + slot] = e;
            var sm = subgroupAdd(e);
            if slot % 32 == 0 {
                red[hl * (CHUNK / 32) + slot / 32] = sm;
            }
            workgroupBarrier();
            var c_sum = 0.0;
            if slot == 0 {
                var cs = 0.0;
                for (var w = 0u; w < (CHUNK / 32); w++) {
                    cs = cs + red[hl * (CHUNK / 32) + w];
                }
                c_sum = cs;
                red[16 + hl] = c_sum;
            }
            workgroupBarrier();
            c_sum = red[16 + hl];
            workgroupBarrier();
            if hl < NQ_PER_KV {
                for (var kk = 0u; kk < limit; kk++) {
                    let w = scs[hl * CHUNK + kk];
                    let vk = (cache_plane + (c0 + kk) * HEAD_DIM) / 2;
                    if slot % 2 == 0 {
                        o0 = o0 + w * bitcast<f32>((v_cache[vk + slot / 2] & 65535) << 16);
                        if slot + CHUNK < HEAD_DIM {
                            o1 = o1 + w * bitcast<f32>((v_cache[vk + CHUNK / 2 + slot / 2] & 65535) << 16);
                        }
                        if slot + 2 * CHUNK < HEAD_DIM {
                            o2 = o2 + w * bitcast<f32>((v_cache[vk + CHUNK + slot / 2] & 65535) << 16);
                            o3 = o3 + w * bitcast<f32>((v_cache[vk + CHUNK + CHUNK / 2 + slot / 2] & 65535) << 16);
                        }
                        if slot + 4 * CHUNK < HEAD_DIM {
                            o4 = o4 + w * bitcast<f32>((v_cache[vk + 2 * CHUNK + slot / 2] & 65535) << 16);
                            o5 = o5 + w * bitcast<f32>((v_cache[vk + 2 * CHUNK + CHUNK / 2 + slot / 2] & 65535) << 16);
                        }
                        if slot + 6 * CHUNK < HEAD_DIM {
                            o6 = o6 + w * bitcast<f32>((v_cache[vk + 3 * CHUNK + slot / 2] & 65535) << 16);
                            o7 = o7 + w * bitcast<f32>((v_cache[vk + 3 * CHUNK + CHUNK / 2 + slot / 2] & 65535) << 16);
                        }
                    } else {
                        o0 = o0 + w * bitcast<f32>((v_cache[vk + slot / 2] >> 16) << 16);
                        if slot + CHUNK < HEAD_DIM {
                            o1 = o1 + w * bitcast<f32>((v_cache[vk + CHUNK / 2 + slot / 2] >> 16) << 16);
                        }
                        if slot + 2 * CHUNK < HEAD_DIM {
                            o2 = o2 + w * bitcast<f32>((v_cache[vk + CHUNK + slot / 2] >> 16) << 16);
                            o3 = o3 + w * bitcast<f32>((v_cache[vk + CHUNK + CHUNK / 2 + slot / 2] >> 16) << 16);
                        }
                        if slot + 4 * CHUNK < HEAD_DIM {
                            o4 = o4 + w * bitcast<f32>((v_cache[vk + 2 * CHUNK + slot / 2] >> 16) << 16);
                            o5 = o5 + w * bitcast<f32>((v_cache[vk + 2 * CHUNK + CHUNK / 2 + slot / 2] >> 16) << 16);
                        }
                        if slot + 6 * CHUNK < HEAD_DIM {
                            o6 = o6 + w * bitcast<f32>((v_cache[vk + 3 * CHUNK + slot / 2] >> 16) << 16);
                            o7 = o7 + w * bitcast<f32>((v_cache[vk + 3 * CHUNK + CHUNK / 2 + slot / 2] >> 16) << 16);
                        }
                    }
                }
            }
            if hl < NQ_PER_KV && slot == 0 {
                let prev_max = segstat[hl * 2];
                let prev_sum = segstat[hl * 2 + 1];
                if prev_max == NEG_INF {
                    segstat[hl * 2] = c_max;
                    segstat[hl * 2 + 1] = c_sum;
                } else if c_max > prev_max {
                    segstat[hl * 2 + 1] = prev_sum * exp(prev_max - c_max) + c_sum;
                    segstat[hl * 2] = c_max;
                } else if c_max > NEG_INF / 2.0 {
                    segstat[hl * 2 + 1] = prev_sum + c_sum;
                }
            }
            workgroupBarrier();
            c0 = c0 + CHUNK;
        }
        if hl < NQ_PER_KV {
            let sbase = ((m * KV_HEADS + kvh) * SEGS + seg) * (8 * STRIDE);
            if slot < HEAD_DIM {
                scratch[sbase + hl * STRIDE + slot] = o0;
            }
            if slot + CHUNK < HEAD_DIM {
                scratch[sbase + hl * STRIDE + slot + CHUNK] = o1;
            }
            if slot + 2 * CHUNK < HEAD_DIM {
                scratch[sbase + hl * STRIDE + slot + 2 * CHUNK] = o2;
                scratch[sbase + hl * STRIDE + slot + 3 * CHUNK] = o3;
            }
            if slot + 4 * CHUNK < HEAD_DIM {
                scratch[sbase + hl * STRIDE + slot + 4 * CHUNK] = o4;
                scratch[sbase + hl * STRIDE + slot + 5 * CHUNK] = o5;
            }
            if slot + 6 * CHUNK < HEAD_DIM {
                scratch[sbase + hl * STRIDE + slot + 6 * CHUNK] = o6;
                scratch[sbase + hl * STRIDE + slot + 7 * CHUNK] = o7;
            }
            if slot == 0 {
                scratch[sbase + hl * STRIDE + HEAD_DIM] = segstat[hl * 2];
                scratch[sbase + hl * STRIDE + HEAD_DIM + 1] = segstat[hl * 2 + 1];
            }
        }
    }
}
