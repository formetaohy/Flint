use std::process::Command;

use saturn_compiler::{Kernel, Source};
use saturn_shader::{to_msl, to_spirv};

fn compile(src: &str) -> Result<Kernel, Vec<saturn_compiler::Diagnostic>> {
    saturn_compiler::compile(&Source::new("<test>", src))
}

fn validate_spirv(source: &str, name: &str) {
    let kernel = compile(source).expect("compile");
    let spirv = to_spirv(&kernel).expect("spirv");
    let path = std::env::temp_dir().join(format!("saturn_{name}_{}.spv", std::process::id()));
    std::fs::write(&path, &spirv).expect("write spv");
    let status = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&path)
        .status()
        .expect("spirv-val not found");
    let _ = std::fs::remove_file(&path);
    assert!(status.success(), "spirv-val rejected generated SPIR-V");
}

#[test]
fn spirv_scale_valid() {
    validate_spirv(
        "kernel scale [workgroup(64, 1, 1)] (src: buf<f32>, dst: buf<f32>) {
            dst[gid.x] = src[gid.x] * 2.0;
        }",
        "spirv_scale_valid",
    );
}

#[test]
fn spirv_f16_valid() {
    validate_spirv(
        "kernel f16k [workgroup(16, 16, 1)] (a: buf<f16>, b: buf<f16>, c: buf<f32>) {
            let acc: f32 = a[gid.x] as f32 * b[gid.x] as f32;
            c[gid.x] = acc + 0.5;
        }",
        "spirv_f16_valid",
    );
}

#[test]
fn spirv_control_flow_valid() {
    validate_spirv(
        r#"
        kernel cf [workgroup(8, 8, 1)] (a: buf<f32>, b: buf<f32>) {
            var acc = 0.0;
            for i in 0..16 {
                if i % 2 == 0 {
                    acc += a[i * 64 + gid.x];
                } else {
                    acc -= b[i * 64 + gid.x];
                }
            }
            loop {
                if acc > 1000.0 {
                    break;
                }
                acc = acc * 1.01;
            }
            b[gid.y * 64 + gid.x] = max(acc, 0.0);
        }
        "#,
        "spirv_control_flow_valid",
    );
}

#[test]
fn spirv_all_builtins_valid() {
    validate_spirv(
        r#"
        kernel blt [workgroup(4, 1, 1)] (a: buf<f32>, b: buf<f32>) {
            let x = a[gid.x];
            let r = floor(x) + ceil(x) + round(x) + sqrt(x) + rsqrt(x) + exp(x)
                + exp2(x) + log(x) + log2(x) + tanh(x) + abs(x) + fma(x, 2.0, 1.0)
                + pow(x, 2.0) + min(x, 1.0) + max(x, 0.0) + clamp(x, 0.0, 1.0);
            b[gid.x] = select(r, 0.0, r > 1000.0);
        }
        "#,
        "spirv_all_builtins_valid",
    );
}

#[test]
fn spirv_int_ops_valid() {
    validate_spirv(
        r#"
        kernel intk [workgroup(4, 1, 1)] (a: buf<u32>, b: buf<i32>) {
            let x = a[gid.x];
            let y = (x + 1) * 3 - 2;
            let z = y / 4;
            let w = y % 4;
            let bits = (x << 2) | (x >> 1) ^ 0xFF;
            a[gid.x] = min(w, 7) & bits;
            let n: i32 = -1;
            let neg = n * 2;
            b[gid.x] = neg + max(n, 0) + abs(n) + clamp(n, -10, 10);
        }
        "#,
        "spirv_int_ops_valid",
    );
}

#[test]
fn spirv_thread_block_valid() {
    validate_spirv(
        "kernel tb [workgroup(2, 3, 4)] (a: buf<u32>) {
            let idx = block.x * 2 + thread.x;
            let flat = idx + block.y * 6 + block.z * 18 + block_dim.x;
            a[flat] = thread.y + thread.z;
        }",
        "spirv_thread_block_valid",
    );
}

#[test]
fn spirv_bool_ops_valid() {
    validate_spirv(
        "kernel boll [workgroup(4, 1, 1)] (a: buf<u32>, b: buf<u32>) {
            let p = a[gid.x] == 1;
            let q = b[gid.x] != 2;
            let r = p && q;
            let s = p || q;
            let t = !r;
            let eq = r == t;
            a[gid.x] = (eq && s) ? 1 : 0;
        }",
        "spirv_bool_ops_valid",
    );
}

#[test]
fn spirv_bool_local_and_shared_valid() {
    validate_spirv(
        r#"
        kernel bv [workgroup(8, 1, 1)] (a: buf<f32>, b: buf<f32>) {
            shared flags: [bool; 8];
            let p = a[gid.x] > 0.0;
            let q = b[gid.x] > 1.0;
            flags[gid.x] = p && q;
            a[gid.x] = flags[gid.x] ? 1.0 : 0.0;
        }
        "#,
        "spirv_bool_local_and_shared_valid",
    );
}

#[test]
fn msl_scale_snapshot() {
    let kernel = compile(
        "kernel scale [workgroup(64, 1, 1)] (src: buf<f32>, dst: buf<f32>) {
            dst[gid.x] = src[gid.x] * 2.0;
        }",
    )
    .expect("compile");
    let (msl, entry) = to_msl(&kernel).expect("msl");
    assert_eq!(entry, "scale");
    assert!(msl.contains("kernel void scale("));
    assert!(msl.contains("device float* src [[buffer(0)]]"));
    assert!(msl.contains("uint3 gid [[thread_position_in_grid]]"));
    assert!(msl.contains("dst[gid.x] = (src[gid.x] * 2.0);"));
}

#[test]
fn msl_f16_snapshot() {
    let kernel = compile(
        "kernel fk [workgroup(16, 1, 1)] (a: buf<f16>) {
            let x: f16 = 0.5;
            a[gid.x] = x * 2.0;
        }",
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("device half* a [[buffer(0)]]"));
    assert!(msl.contains("const half x = (half)0.5;"));
}

#[test]
fn msl_control_flow_snapshot() {
    let kernel = compile(
        r#"
        kernel cf [workgroup(8, 1, 1)] (a: buf<f32>) {
            var acc = 0.0;
            for i in 0..16 {
                acc += a[i * 8 + gid.x];
            }
            loop {
                if acc > 100.0 {
                    break;
                }
                acc += 1.0;
            }
            a[gid.x] = acc;
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("for (uint i = 0u; i < 16u; ++i)"));
    assert!(msl.contains("while (true)"));
    assert!(msl.contains("break;"));
}

#[test]
fn spirv_push_constant_shared_barrier_valid() {
    validate_spirv(
        r#"
        kernel gemm [workgroup(16, 16, 1)]
            (a: buf<f16>, b: buf<f16>, c: buf<f32>, m: u32, n: u32, k: u32)
        {
            const TILE: u32 = 16;
            shared a_tile: [f16; 16 * 16];
            shared b_tile: [f16; 16 * 16];
            let row = block.y * TILE + thread.y;
            let col = block.x * TILE + thread.x;
            var acc: f32 = 0.0;
            for i in 0..((k + 15) / 16) {
                barrier();
                let base = i * 16;
                let ai = (block.y * 16 + thread.y) * k + base + thread.x;
                a_tile[thread.y * 16 + thread.x] = ai < m * k ? a[ai] : 0.0 as f16;
                let bi = (base + thread.y) * n + block.x * 16 + thread.x;
                b_tile[thread.y * 16 + thread.x] = bi < k * n ? b[bi] : 0.0 as f16;
                barrier();
                for j in 0..16 {
                    acc += a_tile[thread.y * 16 + j] as f32 * b_tile[j * 16 + thread.x] as f32;
                }
            }
            if row < m && col < n {
                c[row * n + col] = acc;
            }
        }
        "#,
        "spirv_push_constant_shared_barrier_valid",
    );
}

#[test]
fn spirv_scalar_params_valid() {
    validate_spirv(
        "kernel sp [workgroup(4, 1, 1)] (a: buf<f32>, s: u32, t: f16, u: f32) {
            a[gid.x] = s as f32 + t as f32 + u;
        }",
        "spirv_scalar_params_valid",
    );
}

#[test]
fn msl_shared_barrier_snapshot() {
    let kernel = compile(
        r#"
        kernel sh [workgroup(8, 1, 1)] (a: buf<f32>, s: u32) {
            shared buf: [f32; 64];
            barrier();
            buf[gid.x] = a[gid.x] + s as f32;
            barrier();
            a[gid.x] = buf[gid.x];
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("threadgroup float buf[64];"));
    assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"));
    assert!(msl.contains("constant uint& s [[buffer(1)]]"));
}

#[test]
fn msl_scalar_params_snapshot() {
    let kernel = compile(
        "kernel sp [workgroup(4, 1, 1)] (a: buf<f32>, s: u32, t: f16, u: f32) {
            a[gid.x] = s as f32 + t as f32 + u;
        }",
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("device float* a [[buffer(0)]]"));
    assert!(msl.contains("constant uint& s [[buffer(1)]]"));
    assert!(msl.contains("constant half& t [[buffer(2)]]"));
    assert!(msl.contains("constant float& u [[buffer(3)]]"));
    assert!(msl.contains("a[gid.x] = (((float)(s) + (float)(t)) + u);"));
}

#[test]
fn spirv_subgroup_valid() {
    validate_spirv(
        r#"
        kernel sg [workgroup(64, 1, 1)] (a: buf<f32>, b: buf<u32>) {
            let v = a[gid.x];
            let w = subgroup_broadcast(v, 0);
            let x = subgroup_shuffle(v, lane);
            let y = subgroup_shuffle_down(v, 2);
            let z = subgroup_shuffle_up(v, 1);
            let sum = subgroup_reduce_add(v);
            let mx = subgroup_reduce_max(v);
            let mn = subgroup_reduce_min(w);
            let scan = subgroup_inclusive_add(v);
            let all = subgroup_all(v > 0.0);
            let any = subgroup_any(w < 1.0);
            b[gid.x] = (all && any) ? (x as u32 + y as u32 + z as u32 + lane) : 0;
            a[gid.x] = sum + mx + mn + scan + w;
        }
        "#,
        "spirv_subgroup_valid",
    );
}

#[test]
fn spirv_subgroup_int_valid() {
    validate_spirv(
        r#"
        kernel sgi [workgroup(32, 1, 1)] (a: buf<i32>) {
            let v = a[gid.x];
            let sum = subgroup_reduce_add(v);
            let mx = subgroup_reduce_max(v);
            a[gid.x] = sum + mx;
        }
        "#,
        "spirv_subgroup_int_valid",
    );
}

#[test]
fn msl_subgroup_snapshot() {
    let kernel = compile(
        r#"
        kernel sg [workgroup(64, 1, 1)] (a: buf<f32>) {
            let v = a[gid.x];
            a[gid.x] = subgroup_reduce_add(v) + subgroup_broadcast(v, lane)
                + subgroup_inclusive_add(v);
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("uint lane [[simd_lane_id]]"));
    assert!(msl.contains("simd_sum(v)"));
    assert!(msl.contains("simd_broadcast(v, lane)"));
    assert!(msl.contains("simd_prefix_sum(v)"));
}

#[test]
fn spirv_coop_gemm_valid() {
    validate_spirv(
        r#"
        kernel gemm_cm [workgroup(16, 16, 1)]
            (a: buf<f16>, b: buf<f16>, c: buf<f32>, m: u32, n: u32, k: u32)
        {
            const TILE: u32 = 16;
            shared a_tile: [f16; 16 * 16];
            shared b_tile: [f16; 16 * 16];
            var acc: matrix<f32, acc> = coop_zero();
            let row = block.y * TILE + thread.y;
            let col = block.x * TILE + thread.x;
            for i in 0..((k + 15) / 16) {
                barrier();
                let base = i * 16;
                let ai = (block.y * 16 + thread.y) * k + base + thread.x;
                a_tile[thread.y * 16 + thread.x] = ai < m * k ? a[ai] : 0.0 as f16;
                let bi = (base + thread.y) * n + block.x * 16 + thread.x;
                b_tile[thread.y * 16 + thread.x] = bi < k * n ? b[bi] : 0.0 as f16;
                barrier();
                let am = coop_load_a(a_tile, 16, true);
                let bm = coop_load_b(b_tile, 16, true);
                acc = coop_mul_add(am, bm, acc);
            }
            coop_store(c, acc, n, false);
        }
        "#,
        "spirv_coop_gemm_valid",
    );
}

#[test]
fn spirv_coop_f32_valid() {
    validate_spirv(
        r#"
        kernel coop32 [workgroup(16, 16, 1)] (a: buf<f32>, c: buf<f32>) {
            var acc: matrix<f32, acc> = coop_zero();
            let am = coop_load_a(a, 16, true);
            let bm = coop_load_b(a, 16, false);
            acc = coop_mul_add(am, bm, acc);
            coop_store(c, acc, 16, true);
        }
        "#,
        "spirv_coop_f32_valid",
    );
}

#[test]
fn msl_coop_snapshot() {
    let kernel = compile(
        r#"
        kernel gemm_cm [workgroup(16, 16, 1)]
            (a: buf<f16>, b: buf<f16>, c: buf<f32>, m: u32, n: u32, k: u32)
        {
            shared a_tile: [f16; 16 * 16];
            var acc: matrix<f32, acc> = coop_zero();
            barrier();
            let am = coop_load_a(a_tile, 16, true);
            let bm = coop_load_b(b, 16, true);
            acc = coop_mul_add(am, bm, acc);
            coop_store(c, acc, n, false);
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("metal::simdgroup_float16x16"));
    assert!(msl.contains("NagaCooperativeLoad"));
    assert!(msl.contains("simdgroup_multiply_accumulate(d, a, b, c);"));
    assert!(msl.contains("metal::make_filled_simdgroup_matrix<float, 16, 16>(0.0)"));
    assert!(msl.contains("NagaCooperativeStore"));
}

#[test]
fn spirv_bf16_valid() {
    validate_spirv(
        r#"
        kernel bf [workgroup(64, 1, 1)] (src: buf<f32>, w: buf<bf16>) {
            w[gid.x] = src[gid.x] as bf16;
            let v = w[gid.x] as f32;
            w[gid.x] = (v * 2.0) as bf16;
        }
        "#,
        "spirv_bf16_valid",
    );
}

#[test]
fn spirv_int8_valid() {
    validate_spirv(
        r#"
        kernel q [workgroup(64, 1, 1)] (src: buf<f32>, w: buf<u8>, q: buf<i8>, scale: f32) {
            let v = src[gid.x];
            w[gid.x] = (v * scale) as u8;
            q[gid.x] = (v * scale) as i8;
            let a = w[gid.x] as u32;
            let b = q[gid.x] as i32;
            src[gid.x] = a as f32 + b as f32;
        }
        "#,
        "spirv_int8_valid",
    );
}

#[test]
fn msl_bf16_snapshot() {
    let kernel = compile(
        r#"
        kernel bf [workgroup(4, 1, 1)] (src: buf<f32>, w: buf<bf16>) {
            w[gid.x] = src[gid.x] as bf16;
            src[gid.x] = w[gid.x] as f32;
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("device ushort* w [[buffer(1)]]"));
    assert!(msl.contains("as_type<float>(uint("));
}

#[test]
fn spirv_vec_construct_swizzle_valid() {
    validate_spirv(
        r#"
        kernel vk [workgroup(4, 1, 1)] (a: buf<f32>, b: buf<f32>) {
            let v = vec4<f32>(a[gid.x], 1.0, 2.0, 3.0);
            let w = v * vec4<f32>(2.0, 2.0, 2.0, 2.0);
            let x = w.xy;
            let y = w.zyx;
            b[gid.x * 4 + 0] = x.x;
            b[gid.x * 4 + 1] = x.y;
            b[gid.x * 4 + 2] = y.x + y.z;
            b[gid.x * 4 + 3] = w.w;
        }
        "#,
        "spirv_vec_construct_swizzle_valid",
    );
}

#[test]
fn msl_vec_snapshot() {
    let kernel = compile(
        r#"
        kernel vk [workgroup(4, 1, 1)] (a: buf<f32>, b: buf<f32>) {
            let v = vec4<f32>(a[gid.x], 1.0, 2.0, 3.0);
            let x = v.xy;
            b[0] = x.x + v.w;
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("const float4 v = (float4(a[gid.x], 1.0, 2.0, 3.0));"));
    assert!(msl.contains("v.xy"));
}

#[test]
fn msl_bitcast_snapshot() {
    let kernel = compile(
        r#"
        kernel bk [workgroup(4, 1, 1)] (a: buf<u32>, b: buf<f32>) {
            b[gid.x] = bitcast_f32(a[gid.x]);
            a[gid.x] = bitcast_u32(b[gid.x]);
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("as_type<float>(a[gid.x])"));
    assert!(msl.contains("as_type<uint>(b[gid.x])"));
}

#[test]
fn spirv_bitcast_valid() {
    validate_spirv(
        r#"
        kernel bk [workgroup(4, 1, 1)] (a: buf<u32>, b: buf<f32>) {
            b[gid.x] = bitcast_f32(a[gid.x]);
            a[gid.x] = bitcast_u32(b[gid.x]);
        }
        "#,
        "spirv_bitcast_valid",
    );
}

#[test]
fn spirv_tanh_valid() {
    validate_spirv(
        "kernel tk [workgroup(4, 1, 1)] (a: buf<f32>) {
            a[gid.x] = tanh(a[gid.x]);
        }",
        "spirv_tanh_valid",
    );
}

#[test]
fn msl_tanh_snapshot() {
    let kernel = compile(
        "kernel tk [workgroup(4, 1, 1)] (a: buf<f32>) {
            a[gid.x] = tanh(a[gid.x]);
        }",
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("tanh(a[gid.x])"));
}

#[test]
fn spirv_atomic_valid() {
    validate_spirv(
        r#"
        kernel at [workgroup(64, 1, 1)] (a: buf<u32>, b: buf<i32>) {
            let old = atomic_add(a, gid.x, 1);
            let om = atomic_max(b, gid.x, 5);
            let on = atomic_min(a, gid.x, 3);
            let oe = atomic_exchange(b, gid.x, 7);
            a[gid.x] = old + om as u32 + on + oe as u32;
        }
        "#,
        "spirv_atomic_valid",
    );
}

#[test]
fn msl_atomic_snapshot() {
    let kernel = compile(
        r#"
        kernel at [workgroup(4, 1, 1)] (a: buf<u32>) {
            let old = atomic_add(a, gid.x, 1);
            a[gid.x] = old;
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("atomic_fetch_add_explicit((device atomic_uint*)&a["));
}
