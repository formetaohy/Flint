use saturn_compiler::ir::{Expr, Scalar, Stmt, Type};
use saturn_compiler::{Diagnostic, Driver, Kernel, Source};

fn compile(src: &str) -> std::result::Result<Kernel, Vec<Diagnostic>> {
    saturn_compiler::compile(&Source::new("<test>", src))
}

fn diag_msg(diags: &[Diagnostic]) -> String {
    diags
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

fn params(src: &str) -> Kernel {
    compile(src).expect("compile")
}

#[test]
fn scale_kernel_compiles() {
    let kernel = compile(
        "@workgroup_size(64, 1, 1)
        kernel scale (@binding(0) src: buf<f32>, @binding(1) dst: buf<f32>) {
            dst[global_id().x] = src[global_id().x] * 2.0;
        }",
    )
    .expect("compile");
    assert_eq!(kernel.name, "scale");
    assert_eq!(kernel.workgroup_size, [64, 1, 1]);
    assert_eq!(kernel.params.len(), 2);
    assert_eq!(kernel.params[0].binding, 0);
    assert_eq!(kernel.params[1].binding, 1);
}

#[test]
fn shipped_kernels_compile() {
    let kernels = [
        "@workgroup_size(64, 1, 1)
        kernel hist (@binding(0) a: buf<u32>, @binding(1) out: buf<u32>) {
            atomic_add(out[0], a[global_id().x], .relaxed);
            barrier();
            if local_id().x == 0 {
                out[1] = out[0];
            }
        }",
        "@workgroup_size(4, 4, 1)
        kernel blk (@binding(0) out: buf<u32>) {
            out[global_id().y * 128 + global_id().x] =
                group_id().y * 16 + group_id().x;
        }",
        "@workgroup_size(16, 16, 1)
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
        }",
        "@workgroup_size(64, 1, 1)
        kernel quant (@binding(0) src: buf<f32>, @binding(1) w: buf<bf16>,
                      @binding(2) q: buf<u8>, scale: f32)
        {
            let v = src[global_id().x];
            w[global_id().x] = v as bf16;
            let r = w[global_id().x] as f32;
            let scaled = r * scale;
            q[global_id().x] =
                scaled > 255.0 ? 255 as u8 : (scaled < 0.0 ? 0 as u8 : scaled as u8);
        }",
        "@workgroup_size(64, 1, 1)
        kernel sgtest (@binding(0) a: buf<f32>, @binding(1) b: buf<f32>,
                       @binding(2) c: buf<f32>, @binding(3) d: buf<f32>) {
            let v = a[global_id().x];
            b[global_id().x] = subgroup_reduce_add(v);
            c[global_id().x] = subgroup_broadcast(v, 0);
            d[global_id().x] = subgroup_shuffle(v, (lane() + 1) % 4);
        }",
    ];
    for source in kernels {
        saturn_compiler::compile(&Source::new("<corpus>", source))
            .unwrap_or_else(|diags| panic!("kernel failed: {}", diag_msg(&diags)));
    }
}

#[test]
fn full_language_kernel_compiles() {
    let kernel = params(
        r#"
        @workgroup_size(16, 8, 1)
        kernel probe (@binding(0) a: buf<f16>, @binding(1) b: buf<f32>, @binding(2) c: buf<i32>) {
            let row = group_id().y * 16 + local_id().y;
            let col = group_id().x * 16 + local_id().x;
            let mut acc: f32 = 0.0;
            for i in 0..4 {
                let base = i * 16;
                let ai = row * 16 + base + local_id().x;
                let av = ai < 256 ? a[ai] as f32 : 0.0;
                acc += av * 2.0;
                if acc > 100.0 {
                    acc = min(acc, 99.0);
                }
            }
            loop {
                if col >= 8 {
                    break;
                }
                acc += 1.0;
            }
            let flag = acc >= 0.0 && row < 128;
            let picked = flag ? acc : 0.0;
            b[col * 64 + row] = clamp(picked, 0.0, 128.0);
            c[row * 16 + col] = acc as i32;
        }
        "#,
    );
    assert_eq!(kernel.params.len(), 3);
    assert_eq!(kernel.params[0].binding, 0);
    assert_eq!(kernel.params[2].ty.elem(), Some(Scalar::I32));
}

#[test]
fn f16_literals_are_explicit() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f16>) {
            let x: f16 = 0.5h;
            let y = 1.0h;
            a[0] = x * y;
        }",
    );
    assert_eq!(kernel.params[0].ty.elem(), Some(Scalar::F16));
}

#[test]
fn rejects_var_keyword() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            var x = 1.0;
            a[0] = x;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("use 'let mut'"), "got: {err:?}");
}

#[test]
fn rejects_immutable_mutation() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let x = 1.0;
            x = 2.0;
        }",
    )
    .expect_err("should fail");
    assert!(
        diag_msg(&err).contains("cannot assign to immutable")
            || diag_msg(&err).contains("cannot assign to constant"),
        "got: {err:?}"
    );
}

#[test]
fn rejects_immutable_threadgroup() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let tile: threadgroup<[f32; 4]>;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("'let mut'"), "got: {err:?}");
}

#[test]
fn rejects_uninitialized_scalar() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut x: f32;
            a[0] = x;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("must be an array type"), "got: {err:?}");
}

#[test]
fn rejects_threadgroup_non_array() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut x: threadgroup<f32>;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("requires an array element"), "got: {err:?}");
}

#[test]
fn rejects_threadgroup_with_init() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut x: threadgroup<[f32; 4]> = 1.0;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot have an initializer"), "got: {err:?}");
}

#[test]
fn threadgroup_in_block_rejected() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            if true { let mut s: threadgroup<[f32; 4]>; }
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("kernel top level"), "got: {err:?}");
}

#[test]
fn rejects_immutable_field_assignment() {
    let err = compile(
        r#"
        struct P { lo: f32, hi: f32 }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let meta = P { lo: 1.0, hi: 2.0 };
            meta.lo = 3.0;
            a[0] = meta.lo;
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot assign to immutable"), "got: {err:?}");
}

#[test]
fn float_exponent_without_dot_compiles() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            a[0] = 1e5 + 2.5e-3;
        }",
    );
    assert_eq!(kernel.params.len(), 1);
}

#[test]
fn binding_gap_preserved() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>, @binding(2) b: buf<f32>) {
            b[0] = a[0];
        }",
    );
    assert_eq!(kernel.params[0].binding, 0);
    assert_eq!(kernel.params[1].binding, 2);
}

#[test]
fn field_assignment_pollutes_uniformity() {
    let err = compile(
        r#"
        struct P { lo: f32, hi: f32 }
        @workgroup_size(64,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut meta = P { lo: 1.0, hi: 2.0 };
            if local_id().x < 32 { meta.lo = a[0]; }
            if meta.lo > 0.5 { barrier(); }
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("non-uniform"), "got: {err:?}");
}

#[test]
fn unsuffixed_integers_default_to_u32() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            let x = 1;
            a[0] = x + 2;
        }",
    );
    assert_eq!(kernel.params.len(), 1);
}

#[test]
fn unsuffixed_integers_adapt_to_float_context() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            a[0] = 1;
            a[1] = 255;
        }",
    );
    let Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let Expr::FloatLit { ty, .. } = value else {
        panic!("integer literal must adapt to f32 context, got {value:?}");
    };
    assert_eq!(*ty, Scalar::F32);
}

#[test]
fn unsuffixed_floats_adapt_to_int_context() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            a[0] = 2.0;
        }",
    );
    let Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let Expr::IntLit { ty, .. } = value else {
        panic!("float literal must adapt to u32 context, got {value:?}");
    };
    assert_eq!(*ty, Scalar::U32);
}

#[test]
fn rejects_fractional_float_in_int_context() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            a[0] = 2.5;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("not an integer"), "got: {err:?}");
}

#[test]
fn unsuffixed_int_float_mix_promotes_to_f32() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            a[0] = 1 + 2.5;
            a[1] = 3.5 * 2;
        }",
    );
    assert_eq!(kernel.params.len(), 1);
}

#[test]
fn unsuffixed_literal_as_conversion() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u8>, @binding(1) b: buf<f16>) {
            a[0] = 255 as u8;
            b[0] = 0.5 as f16;
        }",
    );
    let Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let Expr::IntLit { ty, .. } = value else {
        panic!("literal as conversion must resolve directly, got {value:?}");
    };
    assert_eq!(*ty, Scalar::U8);
}

#[test]
fn rejects_unsuffixed_int_exceeding_float_precision() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            a[0] = 16777217;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("not exactly representable"), "got: {err:?}");
}

#[test]
fn unsuffixed_literals_in_ternary_adapt() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u8>) {
            let x = 1;
            a[0] = x > 0 ? 255 : 0;
        }",
    );
    assert_eq!(kernel.params.len(), 1);
}

#[test]
fn const_declaration_with_unsuffixed_literal() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            const TILE: u32 = 16;
            a[0] = TILE;
        }",
    );
    assert_eq!(kernel.body.len(), 1);
}

#[test]
fn attribute_integers_without_suffix() {
    let kernel = params(
        "@workgroup_size(64, 1, 1) kernel k (@binding(0) a: buf<f32>) {
            a[0] = 1.0;
        }",
    );
    assert_eq!(kernel.workgroup_size, [64, 1, 1]);
}

#[test]
fn rejects_undefined_variable() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) { a[0] = missing; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("undefined variable"));
}

#[test]
fn rejects_type_mismatch() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut x = 1;
            a[0] = x;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("type mismatch"));
}

#[test]
fn rejects_negated_unsigned_literal() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { let mut x: u32 = -1u; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot negate a u32 literal"));
}

#[test]
fn negative_literals_compile() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<i32>, @binding(1) b: buf<f32>) {
            let lo = -1i;
            a[0] = lo;
            b[0] = -1.0;
        }",
    );
    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn rejects_break_outside_loop() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { break; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("break outside loop"));
}

#[test]
fn rejects_assign_to_param() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { a = 1; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot assign to parameter"));
}

#[test]
fn rejects_bool_as_conversion() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { a[0] = 1.0 as bool ? 1 : 0; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("bool"));
}

#[test]
fn rejects_redundant_conversion() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { a[0] = 1 as u32; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("redundant conversion"));
}

#[test]
fn rejects_duplicate_parameter() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>, @binding(1) a: buf<u32>) {}",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("duplicate parameter"));
}

#[test]
fn rejects_duplicate_binding() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>, @binding(0) b: buf<u32>) {}",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("duplicate @buffer"));
}

#[test]
fn rejects_buffer_without_binding() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (a: buf<u32>) {}",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("@buffer"));
}

#[test]
fn rejects_reserved_name_shadowing() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { let buf = 1; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("reserved name"));
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { let barrier = 1; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("reserved name"));
}

#[test]
fn rejects_bad_workgroup_size() {
    let err = compile(
        "@workgroup_size(0,1,1) kernel k (@binding(0) a: buf<u32>) {}",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("workgroup size"));
}

#[test]
fn rejects_unknown_builtin() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { a[0] = frobnicate(1.0); }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("unknown function or builtin"));
}

#[test]
fn rejects_at_expression() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { a[0] = @nonsense; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("expected expression"), "got: {err:?}");
}

#[test]
fn bitcast_builtins() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>, @binding(1) b: buf<f32>) {
            b[0] = bitcast_f32(a[0]);
            a[0] = bitcast_u32(b[0]);
        }",
    );
    let Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let Expr::Call { name, ty, .. } = value else {
        panic!("expected call");
    };
    assert_eq!(*name, "bitcast_f32");
    assert_eq!(*ty, Type::Scalar(Scalar::F32));
}

#[test]
fn rejects_bitcast_wrong_arg_type() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) { a[0] = bitcast_f32(a[0]); }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("type mismatch"), "got: {}", diag_msg(&err));
}

#[test]
fn scalar_params_and_offsets() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>, s: u32, t: f16, u: f32) {
            a[0] = s as f32 + t as f32 + u;
        }",
    );
    assert_eq!(kernel.params.len(), 1);
    assert_eq!(kernel.scalars.len(), 3);
    assert_eq!(kernel.scalars[0].offset, 0);
    assert_eq!(kernel.scalars[1].offset, 4);
    assert_eq!(kernel.scalars[2].offset, 8);
}

#[test]
fn rejects_bool_buf_param() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<bool>, s: u32) { a[0] = s > 0; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("buf<bool>"));
}

#[test]
fn rejects_bool_scalar_param() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>, s: bool) { a[0] = s ? 1.0 : 0.0; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot be bool"));
}

#[test]
fn threadgroup_bool_array_compiles() {
    let kernel = params(
        r#"
        @workgroup_size(8, 1, 1)
        kernel k (@binding(0) a: buf<f32>, @binding(1) b: buf<f32>) {
            let mut flags: threadgroup<[bool; 8]>;
            let p = a[global_id().x] > 0.0;
            let q = b[global_id().x] > 1.0;
            flags[global_id().x] = p && q;
            a[global_id().x] = flags[global_id().x] ? 1.0 : 0.0;
        }
        "#,
    );
    assert!(contains_threadgroup(&kernel.body));
}

fn contains_threadgroup(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Var { ty: Type::Threadgroup(_), .. } => true,
        Stmt::If { then, els, .. } => contains_threadgroup(then) || contains_threadgroup(els),
        Stmt::Loop { body, .. } | Stmt::For { body, .. } => contains_threadgroup(body),
        _ => false,
    })
}

#[test]
fn threadgroup_and_barrier_compile() {
    let kernel = params(
        r#"
        @workgroup_size(8, 1, 1)
        kernel k (@binding(0) a: buf<f32>) {
            const TILE: u32 = 16;
            let mut tile: threadgroup<[f32; TILE * TILE]>;
            barrier();
            tile[global_id().x] = a[global_id().x];
            barrier();
            a[global_id().x] = tile[global_id().x];
        }
        "#,
    );
    assert!(contains_barrier(&kernel.body));
    assert!(contains_threadgroup(&kernel.body));
}

#[test]
fn const_expression() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            const SCALE: f32 = 2.0 * 3.0;
            a[0] = a[0] * SCALE;
        }",
    );
    assert_eq!(kernel.body.len(), 1);
}

#[test]
fn rejects_threadgroup_in_block() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            if true { let mut s: threadgroup<[f32; 4]>; }
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("kernel top level"));
}

#[test]
fn rejects_threadgroup_size_nonconstant() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut s: threadgroup<[f32; global_id().x]>;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("constant"));
}

#[test]
fn rejects_assign_to_scalar_param() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>, s: u32) { s = 1; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot assign"));
}

#[test]
fn rejects_barrier_with_args() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) { barrier(1); }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("takes no arguments"));
}

#[test]
fn unroll_expands() {
    let kernel = params(
        r#"
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            @unroll for i in 0..4 {
                a[i] = a[i] * 2.0;
            }
        }
        "#,
    );
    assert_eq!(kernel.body.len(), 4);
}

#[test]
fn rejects_unroll_nonconstant() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            @unroll for i in 0..global_id().x { a[0] = 1.0; }
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("constant"));
}

#[test]
fn const_fold_simplifies_int_arithmetic() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            let x = (1 + 2) * 3;
            a[0] = x + 4;
        }",
    );
    let Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let Expr::IntLit { value, .. } = value else {
        panic!("expected fully folded literal, got {value:?}");
    };
    assert_eq!(*value, 13);
}

#[test]
fn const_fold_resolves_constant_cond() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            let x = 2 > 3 ? 1 : 7;
            a[0] = x;
        }",
    );
    let Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let Expr::IntLit { value, .. } = value else {
        panic!("expected folded literal, got {value:?}");
    };
    assert_eq!(*value, 7);
}

#[test]
fn diagnostic_renders_source_location() {
    let source = Source::new(
        "kern.scl",
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {\n    a[0] = missing;\n}",
    );
    let diags = saturn_compiler::compile(&source).expect_err("should fail");
    let rendered = source.render(&diags[0]);
    assert!(rendered.contains("kern.scl:2:12"), "got: {rendered}");
    assert!(rendered.contains("^"), "got: {rendered}");
}

#[test]
fn functions_expand_inline() {
    let kernel = params(
        r#"
        fn lerp(a: f32, b: f32, t: f32) -> f32 {
            return a + (b - a) * t;
        }
        fn double(x: u32) -> u32 {
            return x * 2;
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>, @binding(1) b: buf<u32>) {
            a[0] = lerp(0.0, 1.0, 0.5);
            b[0] = double(21);
        }
        "#,
    );
    assert!(contains_assign(&kernel.body));
}

#[test]
fn function_parameter_is_value_copy() {
    let kernel = params(
        r#"
        fn bump(x: u32) -> u32 {
            x = x + 1;
            return x;
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            let mut v = 1;
            a[0] = bump(v);
            a[1] = v;
        }
        "#,
    );
    assert!(contains_assign(&kernel.body));
}

#[test]
fn void_function_call_statement() {
    let kernel = params(
        r#"
        fn zero_out(dst: buf<f32>) {
            dst[0] = 0.0;
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            zero_out(a);
        }
        "#,
    );
    assert!(contains_assign(&kernel.body));
}

fn contains_assign(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Assign { .. } => true,
        Stmt::If { then, els, .. } => contains_assign(then) || contains_assign(els),
        Stmt::Loop { body, .. } | Stmt::For { body, .. } => contains_assign(body),
        _ => false,
    })
}

#[test]
fn rejects_recursive_function() {
    let err = compile(
        r#"
        fn f(x: u32) -> u32 {
            if x == 0 { return 1; }
            return f(x - 1);
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            a[0] = f(3);
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("recursive"));
}

#[test]
fn rejects_function_missing_return_path() {
    let err = compile(
        r#"
        fn f(x: u32) -> u32 {
            if x > 0 { return 1; }
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            a[0] = f(3);
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("return on all paths"));
}

#[test]
fn spec_default_specializes() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            spec MODE: u32 = 1;
            if MODE == 1 {
                a[0] = 1.0;
            } else {
                a[0] = 2.0;
            }
        }",
    );
    assert_eq!(kernel.body.len(), 1);
}

#[test]
fn spec_override_specializes() {
    let source = Source::new(
        "<spec>",
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            spec MODE: u32 = 1;
            if MODE == 1 {
                a[0] = 1.0;
            } else {
                a[0] = 2.0;
            }
        }",
    );
    let kernel = Driver::new()
        .compile_with_specs(&source, &[("MODE", 0.0)])
        .expect("compile with override");
    assert_eq!(kernel.body.len(), 1);
    let Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let Expr::FloatLit { value, .. } = value else {
        panic!("expected literal, got {value:?}");
    };
    assert_eq!(*value, 2.0);
}

#[test]
fn rejects_unknown_spec() {
    let source = Source::new(
        "<spec>",
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            spec MODE: u32 = 1;
            a[0] = MODE as f32;
        }",
    );
    let err = Driver::new()
        .compile_with_specs(&source, &[("NOPE", 0.0)])
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("unknown spec"));
}

#[test]
fn rejects_spec_out_of_range() {
    let source = Source::new(
        "<spec>",
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            spec MODE: u32 = 1;
            a[0] = MODE;
        }",
    );
    let err = Driver::new()
        .compile_with_specs(&source, &[("MODE", -1.0)])
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("out of range"));
}

#[test]
fn rejects_barrier_in_divergent_flow() {
    let err = compile(
        "@workgroup_size(64,1,1) kernel k (@binding(0) a: buf<f32>) {
            if local_id().x < 32 {
                barrier();
            }
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("non-uniform"));
}

#[test]
fn barrier_under_uniform_condition_passes() {
    let kernel = params(
        "@workgroup_size(64,1,1) kernel k (@binding(0) a: buf<f32>, n: u32) {
            let limit = min(n, 32);
            if group_id().x < limit {
                barrier();
            }
        }",
    );
    assert!(contains_barrier(&kernel.body));
}

fn contains_barrier(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Barrier { .. } => true,
        Stmt::If { then, els, .. } => contains_barrier(then) || contains_barrier(els),
        Stmt::Loop { body, .. } | Stmt::For { body, .. } => contains_barrier(body),
        _ => false,
    })
}

#[test]
fn new_builtins_compile() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>, @binding(1) f: buf<f32>) {
            a[0] = popcount(a[0]) + clz(a[0]) + ctz(a[0]);
            f[0] = trunc(f[0]) + sign(f[0]) + fract(f[0]);
            let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
            f[1] = dot(v, v);
            atomic_add(a[1], 1, .relaxed);
            atomic_and(a[2], 2, .acquire);
            atomic_or(a[3], 3, .release);
            atomic_xor(a[4], 5, .seq_cst);
            atomic_exchange(a[5], 7, .acq_rel);
        }",
    );
    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn rejects_atomic_missing_order() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            atomic_add(a[0], 1);
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("expects 3 arguments"));
}

#[test]
fn rejects_atomic_on_scalar_target() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>, s: u32) {
            atomic_add(s, 1, .relaxed);
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("buffer or threadgroup array element"));
}

#[test]
fn rejects_write_to_readonly_buffer() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0, readonly) a: buf<u32>) {
            a[0] = 1;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("readonly"));
}

#[test]
fn readonly_buffer_read_compiles() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0, readonly) a: buf<u32>, @binding(1) b: buf<u32>) {
            b[0] = a[0];
        }",
    );
    assert_eq!(kernel.params[0].access, saturn_compiler::ir::Access::ReadOnly);
}

#[test]
fn struct_construct_and_field() {
    let kernel = params(
        r#"
        struct QuantMeta {
            scale: f32,
            zero: f32,
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut meta = QuantMeta { scale: 2.0, zero: 1.0 };
            let s = meta.scale;
            meta.zero = s * 3.0;
            a[0] = meta.zero;
        }
        "#,
    );
    assert_eq!(kernel.structs.len(), 1);
    assert_eq!(kernel.structs[0].name, "QuantMeta");
}

#[test]
fn rejects_struct_field_type() {
    let err = compile(
        r#"
        struct Bad {
            scale: f16,
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {}
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("4-byte"));
}

#[test]
fn rejects_struct_missing_field() {
    let err = compile(
        r#"
        struct QuantMeta {
            scale: f32,
            zero: f32,
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let meta = QuantMeta { scale: 2.0 };
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("requires 2 fields"));
}

#[test]
fn buf_struct_element_compiles() {
    let kernel = params(
        r#"
        struct QuantMeta {
            scale: f32,
            zero: f32,
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) meta: buf<QuantMeta>, @binding(1) a: buf<f32>) {
            let m = meta[0];
            a[0] = m.scale + m.zero;
        }
        "#,
    );
    assert_eq!(kernel.params[0].ty.elem(), Some(Scalar::F32));
}

#[test]
fn local_array_compiles() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let mut acc: [f32; 4];
            acc[0] = a[0];
            acc[1] = acc[0] * 2.0;
            a[1] = acc[1];
        }",
    );
    assert!(contains_assign(&kernel.body));
}

#[test]
fn swizzle_and_member_compile() {
    let kernel = params(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
            let x = v.x;
            let yz = v.yz;
            a[0] = x + yz.x;
        }",
    );
    assert!(contains_assign(&kernel.body));
}

#[test]
fn multi_error_recovery() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            let x = ;
            let y = ;
        }",
    )
    .expect_err("should fail");
    assert!(err.len() >= 2, "expected multiple diagnostics, got {err:?}");
}

#[test]
fn return_outside_function_rejected() {
    let err = compile(
        "@workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) { return; }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("only allowed inside functions"));
}

#[test]
fn threadgroup_inside_function_rejected() {
    let err = compile(
        r#"
        fn f(x: u32) -> u32 {
            let mut s: threadgroup<[u32; 4]>;
            return x;
        }
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<u32>) {
            a[0] = f(1);
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("not allowed inside functions"));
}

#[test]
fn barrier_termination_awareness() {
    let kernel = params(
        r#"
        @workgroup_size(64, 1, 1)
        kernel k (@binding(0) a: buf<f32>, n: u32) {
            for i in 0..4 {
                if local_id().x == 0 {
                    break;
                }
                barrier();
            }
        }
        "#,
    );
    assert!(contains_barrier(&kernel.body));
}

#[test]
fn imports_merge_functions() {
    let entry = Source::new(
        "<entry>",
        r#"
        import "util.scl";
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {
            a[0] = twice(2.0);
        }
        "#,
    );
    let kernel = Driver::new()
        .with_import_resolver(Box::new(|name, _| {
            assert_eq!(name, "util.scl");
            Ok("fn twice(x: f32) -> f32 { return x * 2.0; }".to_string())
        }))
        .compile(&entry)
        .expect("imported function must resolve");
    assert!(contains_assign(&kernel.body));
}

#[test]
fn rejects_circular_import() {
    let entry = Source::new(
        "<entry>",
        r#"
        import "a.scl";
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {}
        "#,
    );
    let err = Driver::new()
        .with_import_resolver(Box::new(|name, _| {
            Ok(match name {
                "a.scl" => "import \"b.scl\";".to_string(),
                "b.scl" => "import \"a.scl\";".to_string(),
                _ => unreachable!(),
            })
        }))
        .compile(&entry)
        .expect_err("circular import must fail");
    assert!(diag_msg(&err).contains("circular import"));
}

#[test]
fn rejects_kernel_in_import() {
    let entry = Source::new(
        "<entry>",
        r#"
        import "bad.scl";
        @workgroup_size(1,1,1) kernel k (@binding(0) a: buf<f32>) {}
        "#,
    );
    let err = Driver::new()
        .with_import_resolver(Box::new(|_, _| {
            Ok("@workgroup_size(1,1,1) kernel other (@binding(0) a: buf<f32>) {}".to_string())
        }))
        .compile(&entry)
        .expect_err("kernel in import must fail");
    assert!(diag_msg(&err).contains("must not contain a kernel"));
}

#[test]
fn cooperative_matrix_compiles() {
    let kernel = params(
        r#"
        @workgroup_size(16, 16, 1)
        kernel k (@binding(0) a: buf<f16>, @binding(1) c: buf<f32>) {
            let mut tile: threadgroup<[f16; 256]>;
            tile[local_id().y * 16 + local_id().x] = a[0];
            barrier();
            let m = coop_load_a(tile[0], 16, true);
            let n = coop_load_b(tile[0], 16, true);
            let mut acc: matrix<f32> = coop_zero();
            acc = coop_mul_add(m, n, acc);
            coop_store(c[0], acc, 16, true);
        }
        "#,
    );
    assert_eq!(kernel.coop_roles.len(), 3);
    assert_eq!(kernel.coop_triples.len(), 1);
}

#[test]
fn rejects_matrix_role_in_type() {
    let err = compile(
        "@workgroup_size(16,16,1) kernel k (@binding(0) a: buf<f16>) {
            let mut acc: matrix<f32, acc> = coop_zero();
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("unknown type") || diag_msg(&err).contains("expected"));
}

#[test]
fn group_size_and_subgroup_builtins() {
    let kernel = params(
        r#"
        @workgroup_size(16, 16, 1)
        kernel k (@binding(0) a: buf<u32>) {
            let dim = group_size();
            let sg = subgroup_id();
            let size = subgroup_size();
            let lane_id = lane();
            let wid = group_id();
            let lid = local_id();
            let gid = global_id();
            a[0] = dim.x + dim.y + dim.z + sg + size + lane_id + wid.x + lid.x + gid.x;
        }
        "#,
    );
    assert!(contains_assign(&kernel.body));
}
