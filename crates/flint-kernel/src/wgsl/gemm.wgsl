// y[m, n] = sum_k x[m, k] * dequant(w[n, k])
// x: f32 [M, K], y: f32 [M, N]; dispatch [N/BN, M/BM, 1].
// N a multiple of 16, K a multiple of 64, M a multiple of 16.
// WDTYPE 0: w is packed bf16 (two values per u32), scales unused.
// WDTYPE 1: w is i8 (four values per u32), dequantized with per-row group
//           scales [N, K/GROUP].
//
// Skinny forward (M <= 16), memory-bound on the weight matrix. Each workgroup
// streams a [BN, K] weight slice exactly once. Threads are assigned along K so
// global loads are coalesced (consecutive threads read consecutive bytes), and
// the dequantized tiles are staged in workgroup memory with a padded stride
// (17, coprime to the 32 banks) so both the stage writes and the accumulate
// reads are bank-conflict free. 128 threads, each a 2-row register tile.

override N: u32 = 1u;
override K: u32 = 1u;
override WDTYPE: u32 = 0u;
override GROUP: u32 = 128u;

const BM: u32 = 16u;
const BN: u32 = 16u;
const BK: u32 = 64u;
// Padded row stride (BN+1 == BM+1): coprime with 32 banks => conflict-free.
const SX: u32 = 17u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

// Transposed, padded tiles: xs[k][m] at xs[k*SX + m], ws[k][n] at ws[k*SX + n].
var<workgroup> xs: array<f32, BK * SX>;
var<workgroup> ws: array<f32, BK * SX>;

fn bf16f(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg3: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let m0 = wg3.y * BM;
    let n0 = wg3.x * BN;

    var acc0 = 0.0;
    var acc1 = 0.0;

    for (var k0: u32 = 0u; k0 < K; k0 += BK) {
        // ---- Stage activation tile xs[k][m] (16 rows x 64 cols). ----
        // Slot s = m*64 + k; consecutive threads take consecutive k (coalesced
        // global read) and write xs[k*SX + m] (conflict-free, stride 17).
        for (var j: u32 = 0u; j < 8u; j += 1u) {
            let s = tid + j * 128u;
            let m = s / BK;
            let col = s % BK;
            xs[col * SX + m] = x[(m0 + m) * K + k0 + col];
        }

        // ---- Stage weight tile ws[k][n] (16 rows x 64 cols), dequantized. ----
        // Slot s = n_local*64 + k; consecutive threads take consecutive k.
        if (WDTYPE == 1u) {
            for (var j: u32 = 0u; j < 8u; j += 1u) {
                let s = tid + j * 128u;
                let row = s / BK;
                let kk = s % BK;
                let n = n0 + row;
                let gk = k0 + kk;
                let word = w[(n * K + gk) >> 2];
                let bits = (word >> ((kk & 3u) << 3u)) & 0xFFu;
                let sb = i32(bits << 24) >> 24;
                let scale = scales[n * (K / GROUP) + gk / GROUP];
                ws[kk * SX + row] = f32(sb) * scale;
            }
        } else {
            for (var j: u32 = 0u; j < 8u; j += 1u) {
                let s = tid + j * 128u;
                let row = s / BK;
                let kk = s % BK;
                let n = n0 + row;
                let p = w[(n * K + k0 + kk) >> 1];
                let bits = select(p >> 16, p & 0xFFFFu, (kk & 1u) == 0u);
                ws[kk * SX + row] = bf16f(bits);
            }
        }
        workgroupBarrier();

        // ---- Accumulate: thread owns N column (tid % 16) and two M rows. ----
        let n_local = tid % BN;
        let m_base = (tid / BN) * 2u;
        for (var kk: u32 = 0u; kk < BK; kk += 1u) {
            let bval = ws[kk * SX + n_local];
            acc0 += xs[kk * SX + m_base] * bval;
            acc1 += xs[kk * SX + m_base + 1u] * bval;
        }
        workgroupBarrier();
    }

    let n_local = tid % BN;
    let m_base = (tid / BN) * 2u;
    y[(m0 + m_base) * N + n0 + n_local] = acc0;
    y[(m0 + m_base + 1u) * N + n0 + n_local] = acc1;
}
