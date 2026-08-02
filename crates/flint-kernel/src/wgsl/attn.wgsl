// Fused grouped-query attention over a [KV_HEADS, MAX_SEQ, HEAD_DIM] KV cache.
// One workgroup per (query row m, query head); online softmax over the causal
// prefix. WINDOW > 0 restricts each query to the trailing WINDOW keys (sliding
// window, Gemma 3); WINDOW == 0 attends to the full causal prefix. HEAD_DIM must
// be in [64, 256].

override N_HEADS: u32 = 1u;
override KV_HEADS: u32 = 1u;
override HEAD_DIM: u32 = 256u;
override MAX_SEQ: u32 = 8192u;
override SCALE: f32 = 0.0625;
override WINDOW: u32 = 0u;

const CHUNK: u32 = 64u;
const NEG_INF: f32 = -3.4e38;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> args: array<u32>;

var<workgroup> scs: array<f32, CHUNK>;
var<workgroup> red: array<f32, 256>;

fn bf16f(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

// Unpack one bf16 element (index `e`) from a packed cache.
fn load_k(e: u32) -> f32 {
    let word = k_cache[e >> 1];
    let bits = select(word >> 16, word & 0xFFFFu, (e & 1u) == 0u);
    return bf16f(bits);
}

fn load_v(e: u32) -> f32 {
    let word = v_cache[e >> 1];
    let bits = select(word >> 16, word & 0xFFFFu, (e & 1u) == 0u);
    return bf16f(bits);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pos = args[0];
    let m = wg.x;
    let h = wg.y;
    let kvh = h / (N_HEADS / KV_HEADS);
    let qpos = pos + m;
    let kv_len = qpos + 1u;
    // Sliding window: the oldest attended key. 0 (WINDOW off) or qpos+1-WINDOW.
    var win_start = 0u;
    if (WINDOW != 0u && kv_len > WINDOW) {
        win_start = kv_len - WINDOW;
    }
    let c_start = (win_start / CHUNK) * CHUNK;
    let q_base = (m * N_HEADS + h) * HEAD_DIM;
    let cache_plane = kvh * MAX_SEQ * HEAD_DIM;
    let t = lid.x;
    let owns_dim = t < HEAD_DIM;

    var run_max = NEG_INF;
    var run_sum = 0.0;
    var o = 0.0;

    for (var c0: u32 = c_start; c0 < kv_len; c0 += CHUNK) {
        // Phase 1: scores for this chunk (threads 0..CHUNK).
        var sc = NEG_INF;
        if (t < CHUNK && c0 + t >= win_start && c0 + t < kv_len) {
            let k_base = cache_plane + (c0 + t) * HEAD_DIM;
            var dot = 0.0;
            for (var d = 0u; d < HEAD_DIM; d += 1u) {
                dot += q[q_base + d] * load_k(k_base + d);
            }
            sc = dot * SCALE;
        }
        red[t] = sc;
        workgroupBarrier();
        var size = 128u;
        loop {
            if (size == 0u) {
                break;
            }
            if (t < size) {
                red[t] = max(red[t], red[t + size]);
            }
            workgroupBarrier();
            size >>= 1;
        }
        let new_max = max(run_max, red[0]);

        // Phase 2: exponentiated scores + chunk sum.
        var e = 0.0;
        if (t < CHUNK && c0 + t >= win_start && c0 + t < kv_len) {
            e = exp(sc - new_max);
        }
        if (t < CHUNK) {
            scs[t] = e;
        }
        red[t] = e;
        workgroupBarrier();
        size = 128u;
        loop {
            if (size == 0u) {
                break;
            }
            if (t < size) {
                red[t] += red[t + size];
            }
            workgroupBarrier();
            size >>= 1;
        }
        let chunk_sum = red[0];

        // Phase 3: every output dimension consumes the shared scores.
        let corr = exp(run_max - new_max);
        o *= corr;
        workgroupBarrier();
        if (owns_dim) {
            let limit = min(CHUNK, kv_len - c0);
            for (var j = 0u; j < limit; j += 1u) {
                o += scs[j] * load_v(cache_plane + (c0 + j) * HEAD_DIM + t);
            }
        }
        run_sum = run_sum * corr + chunk_sum;
        run_max = new_max;
        workgroupBarrier();
    }

    if (owns_dim) {
        y[(m * N_HEADS + h) * HEAD_DIM + t] = o / run_sum;
    }
}
