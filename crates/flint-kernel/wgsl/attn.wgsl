struct Pc {
    M: u32,
    N_HEADS: u32,
    HEAD_DIM: u32,
    MAX_SEQ: u32,
    SCALE: f32,
    WINDOW: u32,
    NQ_PER_KV: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> args: array<u32>;

const BR: u32 = 8;
const BC: u32 = 128;
const KD: u32 = 32;
const COLS: u32 = 4;
const LANES: u32 = 32;
const NEG_INF: f32 = -3.4e38;

var<workgroup> kt: array<f32, 2 * BC * KD>;

fn deq2(word: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>((word & 65535u) << 16),
        bitcast<f32>((word >> 16) << 16),
    );
}

@compute @workgroup_size(256, 1, 1)
fn attn(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let M = pc.M;
    let N_HEADS = pc.N_HEADS;
    let HEAD_DIM = pc.HEAD_DIM;
    let MAX_SEQ = pc.MAX_SEQ;
    let SCALE = pc.SCALE;
    let WINDOW = pc.WINDOW;
    let NQ_PER_KV = pc.NQ_PER_KV;
    let pos = args[0];
    let t = lid.x;
    let row = t / LANES;
    let lane = t % LANES;
    let m0 = grid.x * BR;
    let qh = grid.y;
    let kvh = qh / NQ_PER_KV;
    let mi = m0 + row;
    let half_dim = HEAD_DIM / 2;
    let cache_plane = kvh * MAX_SEQ * half_dim;
    let qb = (mi * N_HEADS + qh) * HEAD_DIM;
    let qpos = pos + mi;
    var win_start = 0u;
    if WINDOW != 0 && qpos + 1 > WINDOW {
        win_start = qpos + 1 - WINDOW;
    }
    let k_segs = (HEAD_DIM + KD - 1) / KD;
    let kv_len = pos + min(m0 + BR, M);

    var m_old = NEG_INF;
    var s_old = 0.0;
    var o = array<f32, 16>();
    var c0 = 0u;
    var w = t;
    loop {
        if w >= BC * min(KD, HEAD_DIM) / 2 {
            break;
        }
        let j2 = w / (min(KD, HEAD_DIM) / 2);
        let d2 = w % (min(KD, HEAD_DIM) / 2);
        var kd2 = vec2<f32>(0.0, 0.0);
        if j2 < kv_len {
            kd2 = deq2(k_cache[cache_plane + j2 * half_dim + d2]);
        }
        kt[d2 * 2 * BC + j2] = kd2.x;
        kt[(d2 * 2 + 1) * BC + j2] = kd2.y;
        w = w + 256;
    }
    workgroupBarrier();
    loop {
        if c0 >= kv_len {
            break;
        }
        var s = array<f32, COLS>();
        let limit = min(BC, kv_len - c0);
        for (var ds = 0u; ds < k_segs; ds++) {
            let buf = ds % 2;
            let d_base = ds * KD;
            let seg_w = min(KD, HEAD_DIM - d_base);
            if ds + 1 < k_segs {
                let nb = (ds + 1) % 2;
                let nb_base = (ds + 1) * KD;
                let nb_w = min(KD, HEAD_DIM - nb_base);
                let words = BC * nb_w / 2;
                var w = t;
                loop {
                    if w >= words {
                        break;
                    }
                    let j2 = w / (nb_w / 2);
                    let d2 = w % (nb_w / 2);
                    var kd2 = vec2<f32>(0.0, 0.0);
                    if c0 + j2 < kv_len {
                        kd2 = deq2(
                            k_cache[cache_plane + (c0 + j2) * half_dim + nb_base / 2 + d2],
                        );
                    }
                    kt[nb * BC * KD + d2 * 2 * BC + j2] = kd2.x;
                    kt[nb * BC * KD + (d2 * 2 + 1) * BC + j2] = kd2.y;
                    w = w + 256;
                }
            }
            for (var d = 0u; d < seg_w; d++) {
                var qd = 0.0;
                if mi < M {
                    qd = q[qb + d_base + d];
                }
                for (var c = 0u; c < COLS; c++) {
                    s[c] = s[c] + qd * kt[buf * BC * KD + d * BC + lane * COLS + c];
                }
            }
            workgroupBarrier();
        }
        var m_cur = NEG_INF;
        for (var c = 0u; c < COLS; c++) {
            let col = c0 + lane * COLS + c;
            var sc = s[c] * SCALE;
            if col > qpos || col < win_start {
                sc = NEG_INF;
            }
            s[c] = sc;
            m_cur = max(m_cur, sc);
        }
        let m_row = subgroupMax(m_cur);
        let m_new = max(m_old, m_row);
        let r = exp(m_old - m_new);
        var s_cur = 0.0;
        for (var c = 0u; c < COLS; c++) {
            let e = select(exp(s[c] - m_new), 0.0, s[c] == NEG_INF);
            s[c] = e;
            s_cur = s_cur + e;
        }
        let s_row = subgroupAdd(s_cur);
        s_old = s_old * r + s_row;
        if HEAD_DIM <= 128 {
            let d8 = lane % 16 * 8;
            for (var i8 = 0u; i8 < 8u; i8++) {
                if d8 + i8 < HEAD_DIM {
                    o[i8] = o[i8] * r;
                }
            }
        } else {
            var nb0 = 0u;
            loop {
                let d8 = lane * 8 + nb0 * LANES * 8;
                if d8 >= HEAD_DIM {
                    break;
                }
                for (var i8 = 0u; i8 < 8u; i8++) {
                    o[nb0 * 8 + i8] = o[nb0 * 8 + i8] * r;
                }
                nb0 = nb0 + 1;
            }
        }
        for (var c = 0u; c < COLS; c++) {
            kt[row * BC + lane * COLS + c] = s[c];
        }
        workgroupBarrier();
        if HEAD_DIM <= 128 {
            let col_lo = (lane / 16) * 64;
            for (var col = 0u; col < 64u; col++) {
                let gcol = c0 + col_lo + col;
                if gcol >= kv_len {
                    break;
                }
                let e = kt[row * BC + col_lo + col];
                let d8 = lane % 16 * 8;
                if d8 + 8 <= HEAD_DIM && (gcol * half_dim + d8 / 2) % 4 == 0 {
                    let vw = v_cache[cache_plane / 4 + (gcol * half_dim + d8 / 2) / 4];
                    let v0 = deq2(vw.x);
                    let v1 = deq2(vw.y);
                    let v2 = deq2(vw.z);
                    let v3 = deq2(vw.w);
                    o[0] = o[0] + e * v0.x;
                    o[1] = o[1] + e * v0.y;
                    o[2] = o[2] + e * v1.x;
                    o[3] = o[3] + e * v1.y;
                    o[4] = o[4] + e * v2.x;
                    o[5] = o[5] + e * v2.y;
                    o[6] = o[6] + e * v3.x;
                    o[7] = o[7] + e * v3.y;
                } else {
                    for (var i8 = 0u; i8 < 8u; i8++) {
                        let d = d8 + i8;
                        if d >= HEAD_DIM {
                            break;
                        }
                        let col_u32 = gcol * half_dim;
                        let u32idx = col_u32 + d / 2;
                        let v4 = min(u32idx / 4, (col_u32 + half_dim - 1) / 4);
                        let vw = v_cache[cache_plane / 4 + v4];
                        let word = select(
                            select(vw.x, vw.y, u32idx - v4 * 4 == 1),
                            select(vw.z, vw.w, u32idx - v4 * 4 == 3),
                            u32idx - v4 * 4 >= 2,
                        );
                        let dd = deq2(word);
                        let v = select(dd.x, dd.y, d % 2 == 1);
                        o[i8] = o[i8] + e * v;
                    }
                }
            }
        } else {
            for (var col = 0u; col < limit; col++) {
                let e = kt[row * BC + col];
                var nb = 0u;
                loop {
                    let d8 = lane * 8 + nb * LANES * 8;
                    if d8 >= HEAD_DIM {
                        break;
                    }
                    if d8 + 8 <= HEAD_DIM && ((c0 + col) * half_dim + d8 / 2) % 4 == 0 {
                        let vw = v_cache[cache_plane / 4 + ((c0 + col) * half_dim + d8 / 2) / 4];
                        let v0 = deq2(vw.x);
                        let v1 = deq2(vw.y);
                        let v2 = deq2(vw.z);
                        let v3 = deq2(vw.w);
                        o[nb * 8] = o[nb * 8] + e * v0.x;
                        o[nb * 8 + 1] = o[nb * 8 + 1] + e * v0.y;
                        o[nb * 8 + 2] = o[nb * 8 + 2] + e * v1.x;
                        o[nb * 8 + 3] = o[nb * 8 + 3] + e * v1.y;
                        o[nb * 8 + 4] = o[nb * 8 + 4] + e * v2.x;
                        o[nb * 8 + 5] = o[nb * 8 + 5] + e * v2.y;
                        o[nb * 8 + 6] = o[nb * 8 + 6] + e * v3.x;
                        o[nb * 8 + 7] = o[nb * 8 + 7] + e * v3.y;
                    } else {
                        for (var i8 = 0u; i8 < 8u; i8++) {
                            let d = d8 + i8;
                            if d >= HEAD_DIM {
                                break;
                            }
                            let col_u32 = (c0 + col) * half_dim;
                            let u32idx = col_u32 + d / 2;
                            let v4 = min(u32idx / 4, (col_u32 + half_dim - 1) / 4);
                            let vw = v_cache[cache_plane / 4 + v4];
                            let word = select(
                                select(vw.x, vw.y, u32idx - v4 * 4 == 1),
                                select(vw.z, vw.w, u32idx - v4 * 4 == 3),
                                u32idx - v4 * 4 >= 2,
                            );
                            let dd = deq2(word);
                            let v = select(dd.x, dd.y, d % 2 == 1);
                            o[nb * 8 + i8] = o[nb * 8 + i8] + e * v;
                        }
                    }
                    nb = nb + 1;
                }
            }
        }
        workgroupBarrier();
        m_old = m_new;
        c0 = c0 + BC;
        if c0 < kv_len {
            let words = BC * min(KD, HEAD_DIM) / 2;
            var w = t;
            loop {
                if w >= words {
                    break;
                }
                let j2 = w / (min(KD, HEAD_DIM) / 2);
                let d2 = w % (min(KD, HEAD_DIM) / 2);
                var kd2 = vec2<f32>(0.0, 0.0);
                if c0 + j2 < kv_len {
                    kd2 = deq2(k_cache[cache_plane + (c0 + j2) * half_dim + d2]);
                }
                kt[d2 * 2 * BC + j2] = kd2.x;
                kt[(d2 * 2 + 1) * BC + j2] = kd2.y;
                w = w + 256;
            }
        }
        workgroupBarrier();
    }
    if mi < M {
        let ob = qb;
        if HEAD_DIM <= 128 {
            let d8 = lane % 16 * 8;
            for (var i8 = 0u; i8 < 8u; i8++) {
                if d8 + i8 < HEAD_DIM {
                    y[ob + d8 + i8] =
                        (o[i8] + subgroupShuffleXor(o[i8], 16u)) / s_old;
                }
            }
        } else {
            var nb = 0u;
            loop {
                let d8 = lane * 8 + nb * LANES * 8;
                if d8 >= HEAD_DIM {
                    break;
                }
                for (var i8 = 0u; i8 < 8u; i8++) {
                    if d8 + i8 < HEAD_DIM {
                        y[ob + d8 + i8] = o[nb * 8 + i8] / s_old;
                    }
                }
                nb = nb + 1;
            }
        }
    }
}
