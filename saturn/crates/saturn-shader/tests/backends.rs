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
    let output = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&path)
        .output()
        .expect("spirv-val not found");
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "spirv-val rejected generated SPIR-V:
{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn spirv_scale_valid() {
    validate_spirv(
        "@workgroup_size(64, 1, 1)
        kernel scale (@binding(0) src: buf<f32>, @binding(1) dst: buf<f32>) {
            dst[global_id().x] = src[global_id().x] * 2.0;
        }",
        "spirv_scale_valid",
    );
}

#[test]
fn spirv_f16_valid() {
    validate_spirv(
        "@workgroup_size(16, 16, 1)
        kernel f16k (@binding(0) a: buf<f16>, @binding(1) b: buf<f16>, @binding(2) c: buf<f32>) {
            let acc: f32 = a[global_id().x] as f32 * b[global_id().x] as f32;
            c[global_id().x] = acc + 0.5;
        }",
        "spirv_f16_valid",
    );
}

#[test]
fn spirv_control_flow_valid() {
    validate_spirv(
        r#"
        @workgroup_size(8, 8, 1)
        kernel cf (@binding(0) a: buf<f32>, @binding(1) b: buf<f32>) {
            let mut acc = 0.0;
            for i in 0..16 {
                if i % 2 == 0 {
                    acc += a[i * 64 + global_id().x];
                } else {
                    acc -= b[i * 64 + global_id().x];
                }
            }
            loop {
                if acc > 1000.0 {
                    break;
                }
                acc = acc * 1.01;
            }
            b[global_id().y * 64 + global_id().x] = max(acc, 0.0);
        }
        "#,
        "spirv_control_flow_valid",
    );
}

#[test]
fn spirv_all_builtins_valid() {
    validate_spirv(
        r#"
        @workgroup_size(4, 1, 1)
        kernel blt (@binding(0) a: buf<f32>, @binding(1) b: buf<f32>) {
            let x = a[global_id().x];
            let r = floor(x) + ceil(x) + round(x) + sqrt(x) + rsqrt(x) + exp(x)
                + exp2(x) + log(x) + log2(x) + tanh(x) + abs(x) + fma(x, 2.0, 1.0)
                + pow(x, 2.0) + min(x, 1.0) + max(x, 0.0) + clamp(x, 0.0, 1.0);
            b[global_id().x] = r > 1000.0 ? 0.0 : r;
        }
        "#,
        "spirv_all_builtins_valid",
    );
}

#[test]
fn spirv_int_ops_valid() {
    validate_spirv(
        r#"
        @workgroup_size(4, 1, 1)
        kernel intk (@binding(0) a: buf<u32>, @binding(1) b: buf<i32>) {
            let x = a[global_id().x];
            let y = (x + 1) * 3 - 2;
            let z = y / 4;
            let w = y % 4;
            let bits = (x << 2) | (x >> 1) ^ 0xFF;
            a[global_id().x] = min(w, 7) & bits;
            let n: i32 = -1i;
            let neg = n * 2i;
            b[global_id().x] = neg + max(n, 0i) + abs(n) + clamp(n, -10i, 10i);
        }
        "#,
        "spirv_int_ops_valid",
    );
}

#[test]
fn spirv_thread_block_valid() {
    validate_spirv(
        "@workgroup_size(2, 3, 4)
        kernel tb (@binding(0) a: buf<u32>) {
            let idx = group_id().x * 2 + local_id().x;
            let flat = idx + group_id().y * 6 + group_id().z * 18 + group_size().x;
            a[flat] = local_id().y + local_id().z;
        }",
        "spirv_thread_block_valid",
    );
}

#[test]
fn spirv_bool_ops_valid() {
    validate_spirv(
        "@workgroup_size(4, 1, 1)
        kernel boll (@binding(0) a: buf<u32>, @binding(1) b: buf<u32>) {
            let p = a[global_id().x] == 1;
            let q = b[global_id().x] != 2;
            let r = p && q;
            let s = p || q;
            let t = !r;
            let eq = r == t;
            a[global_id().x] = (eq && s) ? 1 : 0;
        }",
        "spirv_bool_ops_valid",
    );
}

#[test]
fn spirv_bool_local_and_shared_valid() {
    validate_spirv(
        r#"
        @workgroup_size(8, 1, 1)
        kernel bv (@binding(0) a: buf<f32>, @binding(1) b: buf<f32>) {
            let mut flags: threadgroup<[bool; 8]>;
            let p = a[global_id().x] > 0.0;
            let q = b[global_id().x] > 1.0;
            flags[global_id().x] = p && q;
            a[global_id().x] = flags[global_id().x] ? 1.0 : 0.0;
        }
        "#,
        "spirv_bool_local_and_shared_valid",
    );
}

#[test]
fn msl_scale_snapshot() {
    let kernel = compile(
        "@workgroup_size(64, 1, 1)
        kernel scale (@binding(0) src: buf<f32>, @binding(1) dst: buf<f32>) {
            dst[global_id().x] = src[global_id().x] * 2.0;
        }",
    )
    .expect("compile");
    let (msl, entry) = to_msl(&kernel).expect("msl");
    assert_eq!(entry, "scale");
    assert!(msl.contains("kernel void scale("));
    assert!(msl.contains("device float* src [[buffer(0)]]"));
    assert!(msl.contains("uint3 global_id [[thread_position_in_grid]]"));
    assert!(msl.contains("dst[global_id.x] = (src[global_id.x] * 2.0);"));
}

#[test]
fn msl_f16_snapshot() {
    let kernel = compile(
        "@workgroup_size(16, 1, 1)
        kernel fk (@binding(0) a: buf<f16>) {
            let x: f16 = 0.5h;
            a[global_id().x] = x * 2.0h;
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
        @workgroup_size(8, 1, 1)
        kernel cf (@binding(0) a: buf<f32>) {
            let mut acc = 0.0;
            for i in 0..16 {
                acc += a[i * 8 + global_id().x];
            }
            loop {
                if acc > 100.0 {
                    break;
                }
                acc += 1.0;
            }
            a[global_id().x] = acc;
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
        @workgroup_size(16, 16, 1)
        kernel gemm (@binding(0) a: buf<f16>, @binding(1) b: buf<f16>, @binding(2) c: buf<f32>,
                     m: u32, n: u32, k: u32)
        {
            const TILE: u32 = 16;
            let mut a_tile: threadgroup<[f16; 16 * 16]>;
            let mut b_tile: threadgroup<[f16; 16 * 16]>;
            let row = group_id().y * TILE + local_id().y;
            let col = group_id().x * TILE + local_id().x;
            let mut acc: f32 = 0.0;
            for i in 0..((k + 15) / 16) {
                barrier();
                let base = i * 16;
                let ai = (group_id().y * 16 + local_id().y) * k + base + local_id().x;
                a_tile[local_id().y * 16 + local_id().x] =
                    ai < m * k ? a[ai] : 0.0h;
                let bi = (base + local_id().y) * n + group_id().x * 16 + local_id().x;
                b_tile[local_id().y * 16 + local_id().x] =
                    bi < k * n ? b[bi] : 0.0h;
                barrier();
                @unroll for j in 0..16 {
                    acc += a_tile[local_id().y * 16 + j] as f32
                        * b_tile[j * 16 + local_id().x] as f32;
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
        "@workgroup_size(4, 1, 1) kernel sp (@binding(0) a: buf<f32>, s: u32, t: f16, u: f32) {
            a[global_id().x] = s as f32 + t as f32 + u;
        }",
        "spirv_scalar_params_valid",
    );
}

#[test]
fn msl_shared_barrier_snapshot() {
    let kernel = compile(
        r#"
        @workgroup_size(8, 1, 1)
        kernel sh (@binding(0) a: buf<f32>, s: u32) {
            let mut tile: threadgroup<[f32; 64]>;
            barrier();
            tile[global_id().x] = a[global_id().x] + s as f32;
            barrier();
            a[global_id().x] = tile[global_id().x];
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("threadgroup float tile[64];"));
    assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"));
    assert!(msl.contains("constant uint& s [[buffer(1)]]"));
}

#[test]
fn msl_scalar_params_snapshot() {
    let kernel = compile(
        "@workgroup_size(4, 1, 1) kernel sp (@binding(0) a: buf<f32>, s: u32, t: f16, u: f32) {
            a[global_id().x] = s as f32 + t as f32 + u;
        }",
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("device float* a [[buffer(0)]]"));
    assert!(msl.contains("constant uint& s [[buffer(1)]]"));
    assert!(msl.contains("constant half& t [[buffer(2)]]"));
    assert!(msl.contains("constant float& u [[buffer(3)]]"));
    assert!(msl.contains("a[global_id.x] = (((float)(s) + (float)(t)) + u);"));
}

#[test]
fn spirv_subgroup_valid() {
    validate_spirv(
        r#"
        @workgroup_size(64, 1, 1)
        kernel sg (@binding(0) a: buf<f32>, @binding(1) b: buf<u32>) {
            let v = a[global_id().x];
            let w = subgroup_broadcast(v, 0);
            let x = subgroup_shuffle(v, lane());
            let y = subgroup_shuffle_down(v, 2);
            let z = subgroup_shuffle_up(v, 1);
            let sum = subgroup_reduce_add(v);
            let mx = subgroup_reduce_max(v);
            let mn = subgroup_reduce_min(w);
            let scan = subgroup_inclusive_add(v);
            let all = subgroup_all(v > 0.0);
            let any = subgroup_any(w < 1.0);
            b[global_id().x] =
                (all && any) ? (x as u32 + y as u32 + z as u32 + lane()) : 0;
            a[global_id().x] = sum + mx + mn + scan + w;
        }
        "#,
        "spirv_subgroup_valid",
    );
}

#[test]
fn spirv_subgroup_int_valid() {
    validate_spirv(
        r#"
        @workgroup_size(32, 1, 1)
        kernel sgi (@binding(0) a: buf<i32>) {
            let v = a[global_id().x];
            let sum = subgroup_reduce_add(v);
            let mx = subgroup_reduce_max(v);
            a[global_id().x] = sum + mx;
        }
        "#,
        "spirv_subgroup_int_valid",
    );
}

#[test]
fn msl_subgroup_snapshot() {
    let kernel = compile(
        r#"
        @workgroup_size(64, 1, 1)
        kernel sg (@binding(0) a: buf<f32>) {
            let v = a[global_id().x];
            a[global_id().x] = subgroup_reduce_add(v) + subgroup_broadcast(v, lane())
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
        @workgroup_size(16, 16, 1)
        kernel gemm_cm (@binding(0) a: buf<f16>, @binding(1) b: buf<f16>, @binding(2) c: buf<f32>,
                        m: u32, n: u32, k: u32)
        {
            const TILE: u32 = 16;
            let mut a_tile: threadgroup<[f16; 16 * 16]>;
            let mut b_tile: threadgroup<[f16; 16 * 16]>;
            let mut acc: matrix<f32> = coop_zero();
            let row = group_id().y * TILE + local_id().y;
            let col = group_id().x * TILE + local_id().x;
            for i in 0..((k + 15) / 16) {
                barrier();
                let base = i * 16;
                let ai = (group_id().y * 16 + local_id().y) * k + base + local_id().x;
                a_tile[local_id().y * 16 + local_id().x] =
                    ai < m * k ? a[ai] : 0.0h;
                let bi = (base + local_id().y) * n + group_id().x * 16 + local_id().x;
                b_tile[local_id().y * 16 + local_id().x] =
                    bi < k * n ? b[bi] : 0.0h;
                barrier();
                let am = coop_load_a(a_tile[0], 16, true);
                let bm = coop_load_b(b_tile[0], 16, true);
                acc = coop_mul_add(am, bm, acc);
            }
            coop_store(c[0], acc, n, false);
        }
        "#,
        "spirv_coop_gemm_valid",
    );
}

#[test]
fn spirv_coop_f32_valid() {
    validate_spirv(
        r#"
        @workgroup_size(16, 16, 1)
        kernel coop32 (@binding(0) a: buf<f32>, @binding(1) c: buf<f32>) {
            let mut acc: matrix<f32> = coop_zero();
            let am = coop_load_a(a[0], 16, true);
            let bm = coop_load_b(a[0], 16, false);
            acc = coop_mul_add(am, bm, acc);
            coop_store(c[0], acc, 16, true);
        }
        "#,
        "spirv_coop_f32_valid",
    );
}

#[test]
fn msl_coop_snapshot() {
    let kernel = compile(
        r#"
        @workgroup_size(16, 16, 1)
        kernel gemm_cm (@binding(0) a: buf<f16>, @binding(1) b: buf<f16>, @binding(2) c: buf<f32>,
                        m: u32, n: u32, k: u32)
        {
            let mut a_tile: threadgroup<[f16; 16 * 16]>;
            let mut acc: matrix<f32> = coop_zero();
            barrier();
            let am = coop_load_a(a_tile[0], 16, true);
            let bm = coop_load_b(b[0], 16, true);
            acc = coop_mul_add(am, bm, acc);
            coop_store(c[0], acc, n, false);
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
        @workgroup_size(64, 1, 1)
        kernel bf (@binding(0) src: buf<f32>, @binding(1) w: buf<bf16>) {
            w[global_id().x] = src[global_id().x] as bf16;
            let v = w[global_id().x] as f32;
            w[global_id().x] = (v * 2.0) as bf16;
        }
        "#,
        "spirv_bf16_valid",
    );
}

#[test]
fn spirv_int8_valid() {
    validate_spirv(
        r#"
        @workgroup_size(64, 1, 1)
        kernel q (@binding(0) src: buf<f32>, @binding(1) w: buf<u8>, @binding(2) q: buf<i8>,
                  scale: f32) {
            let v = src[global_id().x];
            w[global_id().x] = (v * scale) as u8;
            q[global_id().x] = (v * scale) as i8;
            let a = w[global_id().x] as u32;
            let b = q[global_id().x] as i32;
            src[global_id().x] = a as f32 + b as f32;
        }
        "#,
        "spirv_int8_valid",
    );
}

#[test]
fn msl_bf16_snapshot() {
    let kernel = compile(
        r#"
        @workgroup_size(4, 1, 1)
        kernel bf (@binding(0) src: buf<f32>, @binding(1) w: buf<bf16>) {
            w[global_id().x] = src[global_id().x] as bf16;
            src[global_id().x] = w[global_id().x] as f32;
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
        @workgroup_size(4, 1, 1)
        kernel vk (@binding(0) a: buf<f32>, @binding(1) b: buf<f32>) {
            let v = vec4<f32>(a[global_id().x], 1.0, 2.0, 3.0);
            let w = v * vec4<f32>(2.0, 2.0, 2.0, 2.0);
            let x = w.xy;
            let y = w.zyx;
            b[global_id().x * 4 + 0] = x.x;
            b[global_id().x * 4 + 1] = x.y;
            b[global_id().x * 4 + 2] = y.x + y.z;
            b[global_id().x * 4 + 3] = w.w;
        }
        "#,
        "spirv_vec_construct_swizzle_valid",
    );
}

#[test]
fn msl_vec_snapshot() {
    let kernel = compile(
        r#"
        @workgroup_size(4, 1, 1)
        kernel vk (@binding(0) a: buf<f32>, @binding(1) b: buf<f32>) {
            let v = vec4<f32>(a[global_id().x], 1.0, 2.0, 3.0);
            let x = v.xy;
            b[0] = x.x + v.w;
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("const float4 v = (float4(a[global_id.x], 1.0, 2.0, 3.0));"));
    assert!(msl.contains("v.xy"));
}

#[test]
fn msl_bitcast_snapshot() {
    let kernel = compile(
        r#"
        @workgroup_size(4, 1, 1)
        kernel bk (@binding(0) a: buf<u32>, @binding(1) b: buf<f32>) {
            b[global_id().x] = bitcast_f32(a[global_id().x]);
            a[global_id().x] = bitcast_u32(b[global_id().x]);
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("as_type<float>(a[global_id.x])"));
    assert!(msl.contains("as_type<uint>(b[global_id.x])"));
}

#[test]
fn spirv_bitcast_valid() {
    validate_spirv(
        r#"
        @workgroup_size(4, 1, 1)
        kernel bk (@binding(0) a: buf<u32>, @binding(1) b: buf<f32>) {
            b[global_id().x] = bitcast_f32(a[global_id().x]);
            a[global_id().x] = bitcast_u32(b[global_id().x]);
        }
        "#,
        "spirv_bitcast_valid",
    );
}

#[test]
fn spirv_tanh_valid() {
    validate_spirv(
        "@workgroup_size(4, 1, 1) kernel tk (@binding(0) a: buf<f32>) {
            a[global_id().x] = tanh(a[global_id().x]);
        }",
        "spirv_tanh_valid",
    );
}

#[test]
fn msl_tanh_snapshot() {
    let kernel = compile(
        "@workgroup_size(4, 1, 1) kernel tk (@binding(0) a: buf<f32>) {
            a[global_id().x] = tanh(a[global_id().x]);
        }",
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("tanh(a[global_id.x])"));
}

#[test]
fn spirv_atomic_valid() {
    validate_spirv(
        r#"
        @workgroup_size(64, 1, 1)
        kernel at (@binding(0) a: buf<u32>, @binding(1) b: buf<i32>) {
            let old = atomic_add(a[global_id().x], 1, .relaxed);
            let om = atomic_max(b[global_id().x], 5i, .acquire);
            let on = atomic_min(a[global_id().x], 3, .release);
            let oe = atomic_exchange(b[global_id().x], 7i, .seq_cst);
            a[global_id().x] = old + om as u32 + on + oe as u32;
        }
        "#,
        "spirv_atomic_valid",
    );
}

#[test]
fn msl_atomic_snapshot() {
    let kernel = compile(
        r#"
        @workgroup_size(4, 1, 1)
        kernel at (@binding(0) a: buf<u32>) {
            let old = atomic_add(a[global_id().x], 1, .relaxed);
            a[global_id().x] = old;
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains(
        "atomic_fetch_add_explicit((device atomic_uint*)&a[global_id.x], 1u, memory_order_relaxed)"
    ));
}

#[test]
fn spirv_struct_buf_valid() {
    validate_spirv(
        r#"
        struct Pair {
            lo: f32,
            hi: f32,
        }
        @workgroup_size(4, 1, 1)
        kernel sk (@binding(0) p: buf<Pair>, @binding(1) out: buf<f32>) {
            let m = p[global_id().x];
            out[global_id().x] = m.lo + m.hi;
        }
        "#,
        "spirv_struct_buf_valid",
    );
}

#[test]
fn msl_struct_snapshot() {
    let kernel = compile(
        r#"
        struct Pair {
            lo: f32,
            hi: f32,
        }
        @workgroup_size(4, 1, 1)
        kernel sk (@binding(0) p: buf<Pair>, @binding(1) out: buf<f32>) {
            let m = p[global_id().x];
            out[global_id().x] = m.lo + m.hi;
        }
        "#,
    )
    .expect("compile");
    let (msl, _) = to_msl(&kernel).expect("msl");
    assert!(msl.contains("struct Pair {"));
    assert!(msl.contains("device Pair* p [[buffer(0)]]"));
    assert!(msl.contains("m.lo + m.hi"));
}
