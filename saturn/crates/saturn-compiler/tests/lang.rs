use saturn_compiler::{Diagnostic, Kernel, Source};

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

#[test]
fn scale_kernel_compiles() {
    let kernel = compile(
        "kernel scale [workgroup(64, 1, 1)] (src: buf<f32>, dst: buf<f32>) {
            dst[gid.x] = src[gid.x] * 2.0;
        }",
    )
    .expect("compile");
    assert_eq!(kernel.name, "scale");
    assert_eq!(kernel.workgroup_size, [64, 1, 1]);
    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn shipped_kernels_compile() {
    let kernels = [
        "kernel hist [workgroup(64, 1, 1)] (a: buf<u32>, out: buf<u32>) {
            atomic_add(out, 0, a[gid.x]);
            barrier();
            if thread.x == 0 {
                out[1] = out[0];
            }
        }",
        "kernel blk [workgroup(4, 4, 1)] (out: buf<u32>) {
            out[gid.y * 128 + gid.x] = block.y * 16 + block.x;
        }",
        "kernel gemm [workgroup(16, 16, 1)]
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
                for j in 0..16 unroll {
                    acc += a_tile[thread.y * 16 + j] as f32 * b_tile[j * 16 + thread.x] as f32;
                }
            }
            if row < m && col < n {
                c[row * n + col] = acc;
            }
        }",
        "kernel quant [workgroup(64, 1, 1)]
            (src: buf<f32>, w: buf<bf16>, q: buf<u8>, scale: f32)
        {
            let v = src[gid.x];
            w[gid.x] = v as bf16;
            let r = w[gid.x] as f32;
            let scaled = r * scale;
            q[gid.x] = scaled > 255.0 ? 255 : (scaled < 0.0 ? 0 : scaled as u8);
        }",
        "kernel sgtest [workgroup(64, 1, 1)] (a: buf<f32>, b: buf<f32>, c: buf<f32>, d: buf<f32>) {
            let v = a[gid.x];
            b[gid.x] = subgroup_reduce_add(v);
            c[gid.x] = subgroup_broadcast(v, 0);
            d[gid.x] = subgroup_shuffle(v, (lane + 1) % 4);
        }",
    ];
    for source in kernels {
        saturn_compiler::compile(&Source::new("<corpus>", source))
            .unwrap_or_else(|diags| panic!("kernel failed: {}", diag_msg(&diags)));
    }
}

#[test]
fn full_language_kernel_compiles() {
    let kernel = compile(
        r#"
        kernel probe [workgroup(16, 8, 1)] (a: buf<f16>, b: buf<f32>, c: buf<i32>) {
            let row = block.y * 16 + thread.y;
            let col = block.x * 16 + thread.x;
            var acc: f32 = 0.0;
            for i in 0..4 {
                let base = i * 16;
                let ai = row * 16 + base + thread.x;
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
            let picked = select(acc, 0.0, flag);
            b[col * 64 + row] = clamp(picked, 0.0, 128.0);
            c[row * 16 + col] = acc as i32;
        }
        "#,
    )
    .expect("compile");
    assert_eq!(kernel.params.len(), 3);
    assert_eq!(kernel.params[0].binding, 0);
    assert_eq!(kernel.params[2].elem, saturn_compiler::ir::Scalar::I32);
}

#[test]
fn f16_literals_adapt_to_context() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f16>) {
            let x: f16 = 0.5;
            let y = 1.0;
            a[0] = x * y as f16;
        }",
    )
    .expect("compile");
    assert_eq!(kernel.params[0].elem, saturn_compiler::ir::Scalar::F16);
}

#[test]
fn rejects_undefined_variable() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<f32>) { a[0] = missing; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("undefined variable"));
}

#[test]
fn rejects_type_mismatch() {
    let err = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) {
            let x = 1;
            a[0] = x;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("type mismatch"));
}

#[test]
fn rejects_negated_u32() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { a[0] = -1; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot negate u32"));
}

#[test]
fn rejects_break_outside_loop() {
    let err =
        compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { break; }").expect_err("should fail");
    assert!(diag_msg(&err).contains("break outside loop"));
}

#[test]
fn rejects_assign_to_param() {
    let err =
        compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { a = 1; }").expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot assign to parameter"));
}

#[test]
fn rejects_bool_as_conversion() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { a[0] = 1.0 as bool ? 1 : 0; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("bool"));
}

#[test]
fn rejects_redundant_conversion() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { a[0] = 1 as u32; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("redundant conversion"));
}

#[test]
fn rejects_duplicate_parameter() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>, a: buf<u32>) {}")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("duplicate parameter"));
}

#[test]
fn rejects_builtin_name_shadowing() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { let gid = 1; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("reserved builtin"));
}

#[test]
fn rejects_bad_workgroup_size() {
    let err = compile("kernel k [workgroup(0,1,1)] (a: buf<u32>) {}").expect_err("should fail");
    assert!(diag_msg(&err).contains("workgroup size"));
}

#[test]
fn rejects_unknown_builtin() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { a[0] = frobnicate(1.0); }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("unknown builtin"));
}

#[test]
fn bitcast_builtins() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<u32>, b: buf<f32>) {
            b[0] = bitcast_f32(a[0]);
            a[0] = bitcast_u32(b[0]);
        }",
    )
    .expect("compile");
    let saturn_compiler::ir::Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let saturn_compiler::ir::Expr::Call { name, ty, .. } = value else {
        panic!("expected call");
    };
    assert_eq!(*name, "bitcast_f32");
    assert_eq!(*ty, saturn_compiler::ir::Type::Scalar(saturn_compiler::ir::Scalar::F32));
}

#[test]
fn rejects_bitcast_wrong_arg_type() {
    let err = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) { a[0] = bitcast_f32(a[0]); }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("type mismatch"), "got: {}", diag_msg(&err));
}

#[test]
fn tanh_builtin() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) { a[0] = tanh(a[0]); }",
    )
    .expect("compile");
    let saturn_compiler::ir::Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let saturn_compiler::ir::Expr::Call { name, .. } = value else {
        panic!("expected call");
    };
    assert_eq!(*name, "tanh");
}

#[test]
fn scalar_params_and_offsets() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>, s: u32, t: f16, u: f32) {
            a[0] = s as f32 + t as f32 + u;
        }",
    )
    .expect("compile");
    assert_eq!(kernel.params.len(), 1);
    assert_eq!(kernel.scalars.len(), 3);
    assert_eq!(kernel.scalars[0].offset, 0);
    assert_eq!(kernel.scalars[1].offset, 4);
    assert_eq!(kernel.scalars[2].offset, 8);
}

#[test]
fn rejects_bool_buf_param() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<bool>, s: u32) { a[0] = s > 0; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot be buf<bool>"));
}

#[test]
fn rejects_bool_scalar_param() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<f32>, s: bool) { a[0] = s ? 1.0 : 0.0; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot be bool"));
}

#[test]
fn bool_locals_and_shared_compile() {
    let kernel = compile(
        r#"
        kernel k [workgroup(8, 1, 1)] (a: buf<f32>, b: buf<f32>) {
            shared flags: [bool; 8];
            let p = a[gid.x] > 0.0;
            let q = b[gid.x] > 1.0;
            flags[gid.x] = p && q;
            a[gid.x] = flags[gid.x] ? 1.0 : 0.0;
        }
        "#,
    )
    .expect("compile");
    assert_eq!(kernel.shareds.len(), 1);
    assert_eq!(kernel.shareds[0].elem, saturn_compiler::ir::Scalar::Bool);
}

#[test]
fn shared_and_barrier_compile() {
    let kernel = compile(
        r#"
        kernel k [workgroup(8, 1, 1)] (a: buf<f32>) {
            const TILE: u32 = 16;
            shared buf: [f32; TILE * TILE];
            barrier();
            buf[gid.x] = a[gid.x];
            barrier();
            a[gid.x] = buf[gid.x];
        }
        "#,
    )
    .expect("compile");
    assert_eq!(kernel.shareds.len(), 1);
    assert_eq!(kernel.shareds[0].len, 256);
    assert!(
        kernel
            .body
            .iter()
            .any(|s| matches!(s, saturn_compiler::ir::Stmt::Barrier { .. }))
    );
}

#[test]
fn const_expression() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) {
            const SCALE: f32 = 2.0 * 3.0;
            a[0] = a[0] * SCALE;
        }",
    )
    .expect("compile");
    assert_eq!(kernel.body.len(), 1);
}

#[test]
fn rejects_shared_in_block() {
    let err =
        compile("kernel k [workgroup(1,1,1)] (a: buf<f32>) { if true { shared s: [f32; 4]; } }")
            .expect_err("should fail");
    assert!(diag_msg(&err).contains("kernel top level"));
}

#[test]
fn rejects_shared_size_nonconstant() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<f32>) { shared s: [f32; gid.x]; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("constant"));
}

#[test]
fn rejects_assign_to_scalar_param() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<f32>, s: u32) { s = 1; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot assign"));
}

#[test]
fn rejects_barrier_with_args() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<f32>) { barrier(1); }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("unknown builtin"));
}

#[test]
fn unroll_expands() {
    let kernel = compile(
        r#"
        kernel k [workgroup(1,1,1)] (a: buf<f32>) {
            for i in 0..4 unroll {
                a[i] = a[i] * 2.0;
            }
        }
        "#,
    )
    .expect("compile");
    assert_eq!(kernel.body.len(), 4);
}

#[test]
fn rejects_unroll_nonconstant() {
    let err = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) { for i in 0..gid.x unroll { a[0] = 1.0; } }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("constant"));
}

#[test]
fn const_fold_simplifies_int_arithmetic() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<u32>) {
            let x = (1 + 2) * 3;
            a[0] = x + 4;
        }",
    )
    .expect("compile");
    let saturn_compiler::ir::Stmt::Let { init, .. } = &kernel.body[0] else {
        panic!("expected let");
    };
    let saturn_compiler::ir::Expr::IntLit { value, .. } = init else {
        panic!("expected folded literal, got {init:?}");
    };
    assert_eq!(*value, 9);
    let saturn_compiler::ir::Stmt::Assign { value, .. } = &kernel.body[1] else {
        panic!("expected assign");
    };
    let saturn_compiler::ir::Expr::Binary { lhs, rhs, .. } = value else {
        panic!("expected binary");
    };
    assert!(matches!(&**lhs, saturn_compiler::ir::Expr::LocalRef { .. }));
    let saturn_compiler::ir::Expr::IntLit { value: r, .. } = &**rhs else {
        panic!("expected folded literal");
    };
    assert_eq!(*r, 4);
}

#[test]
fn const_fold_resolves_constant_cond() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<u32>) {
            let x = 2 > 3 ? 1 : 7;
            a[0] = x;
        }",
    )
    .expect("compile");
    let saturn_compiler::ir::Stmt::Let { init, .. } = &kernel.body[0] else {
        panic!("expected let");
    };
    let saturn_compiler::ir::Expr::IntLit { value, .. } = init else {
        panic!("expected folded literal, got {init:?}");
    };
    assert_eq!(*value, 7);
}

#[test]
fn diagnostic_renders_source_location() {
    let source = Source::new(
        "kern.sat",
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) {\n    a[0] = missing;\n}",
    );
    let diags = saturn_compiler::compile(&source).expect_err("should fail");
    let rendered = source.render(&diags[0]);
    assert!(rendered.contains("kern.sat:2:12"), "got: {rendered}");
    assert!(rendered.contains("^"), "got: {rendered}");
}
