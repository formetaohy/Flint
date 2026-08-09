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
            var x = 1;
            a[0] = x;
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("type mismatch"));
}

#[test]
fn literal_binding_adapts_to_context() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>, b: buf<u32>) {
            let x = 1;
            a[0] = x;
            b[0] = x;
        }",
    )
    .expect("literal binding must adapt to each use site");
    assert_eq!(kernel.body.len(), 2);
}

#[test]
fn rejects_negated_unsigned_literal() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { var x: u32 = -1; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("cannot negate an unsigned integer literal"));
}

#[test]
fn negative_literals_compile() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<i32>, b: buf<f32>) {
            let lo = -1;
            let min = -2147483648;
            a[0] = lo;
            a[1] = min;
            b[0] = -1;
        }",
    )
    .expect("negative literals must compile");
    assert_eq!(kernel.params.len(), 2);
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
fn rejects_reserved_name_shadowing() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { let gid = 1; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("reserved name"));
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { let buf = 1; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("reserved name"));
    let err = compile("kernel k [workgroup(1,1,1)] (let: buf<u32>) {}")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("reserved name"));
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
            shared tile: [f32; TILE * TILE];
            barrier();
            tile[gid.x] = a[gid.x];
            barrier();
            a[gid.x] = tile[gid.x];
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
    let saturn_compiler::ir::Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let saturn_compiler::ir::Expr::IntLit { value, .. } = value else {
        panic!("expected fully folded literal, got {value:?}");
    };
    assert_eq!(*value, 13);
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
    let saturn_compiler::ir::Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let saturn_compiler::ir::Expr::IntLit { value, .. } = value else {
        panic!("expected folded literal, got {value:?}");
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

#[test]
fn functions_expand_inline() {
    let kernel = compile(
        r#"
        fn lerp(a: f32, b: f32, t: f32) -> f32 {
            return a + (b - a) * t;
        }
        fn double(x: u32) -> u32 {
            return x * 2;
        }
        kernel k [workgroup(1,1,1)] (a: buf<f32>, b: buf<u32>) {
            a[0] = lerp(0.0, 1.0, 0.5);
            b[0] = double(21);
        }
        "#,
    )
    .expect("compile");
    assert!(kernel
        .body
        .iter()
        .filter(|s| matches!(s, saturn_compiler::ir::Stmt::Assign { .. }))
        .count()
        >= 2);
}

#[test]
fn void_function_call_statement() {
    let kernel = compile(
        r#"
        fn zero_out(dst: buf<f32>) {
            dst[0] = 0.0;
        }
        kernel k [workgroup(1,1,1)] (a: buf<f32>) {
            zero_out(a);
        }
        "#,
    )
    .expect("compile");
    assert!(contains_assign(&kernel.body));
}

fn contains_assign(stmts: &[saturn_compiler::ir::Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        saturn_compiler::ir::Stmt::Assign { .. } => true,
        saturn_compiler::ir::Stmt::If { then, els, .. } => {
            contains_assign(then) || contains_assign(els)
        }
        saturn_compiler::ir::Stmt::Loop { body, .. }
        | saturn_compiler::ir::Stmt::For { body, .. } => contains_assign(body),
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
        kernel k [workgroup(1,1,1)] (a: buf<u32>) {
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
        kernel k [workgroup(1,1,1)] (a: buf<u32>) {
            a[0] = f(3);
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("return on all paths"));
}

#[test]
fn const_argument_specializes() {
    let kernel = compile(
        r#"
        fn decode(w: u32, const WDTYPE: u32) -> u32 {
            if WDTYPE == 1 {
                return w * 2;
            }
            return w;
        }
        kernel k [workgroup(1,1,1)] (a: buf<u32>) {
            a[0] = decode(a[0], 1);
        }
        "#,
    )
    .expect("compile");
    let last = kernel.body.last().unwrap();
    let saturn_compiler::ir::Stmt::Assign { value, .. } = last else {
        panic!("expected assign, got {last:?}");
    };
    let saturn_compiler::ir::Expr::LocalRef { .. } = value else {
        panic!("expected local ref, got {value:?}");
    };
    assert!(contains_mul(&kernel.body));
}

fn contains_mul(stmts: &[saturn_compiler::ir::Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        saturn_compiler::ir::Stmt::Let { init, .. }
        | saturn_compiler::ir::Stmt::Var { init, .. }
        | saturn_compiler::ir::Stmt::Assign { value: init, .. }
        | saturn_compiler::ir::Stmt::ExprStmt { expr: init, .. } => {
            expr_has_mul(init)
        }
        saturn_compiler::ir::Stmt::If { then, els, .. } => {
            contains_mul(then) || contains_mul(els)
        }
        saturn_compiler::ir::Stmt::Loop { body, .. }
        | saturn_compiler::ir::Stmt::For { body, .. } => contains_mul(body),
        _ => false,
    })
}

fn expr_has_mul(expr: &saturn_compiler::ir::Expr) -> bool {
    match expr {
        saturn_compiler::ir::Expr::Binary {
            op: saturn_compiler::ir::BinOp::Mul,
            ..
        } => true,
        saturn_compiler::ir::Expr::Binary { lhs, rhs, .. } => {
            expr_has_mul(lhs) || expr_has_mul(rhs)
        }
        saturn_compiler::ir::Expr::Index { base, index, .. } => {
            expr_has_mul(base) || expr_has_mul(index)
        }
        saturn_compiler::ir::Expr::Unary { expr: e, .. }
        | saturn_compiler::ir::Expr::Convert { expr: e, .. } => expr_has_mul(e),
        saturn_compiler::ir::Expr::Cond {
            cond, then, els, ..
        } => expr_has_mul(cond) || expr_has_mul(then) || expr_has_mul(els),
        saturn_compiler::ir::Expr::Call { args, .. } => args.iter().any(expr_has_mul),
        _ => false,
    }
}

#[test]
fn spec_default_specializes() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) {
            spec MODE: u32 = 1;
            if MODE == 1 {
                a[0] = 1.0;
            } else {
                a[0] = 2.0;
            }
        }",
    )
    .expect("compile");
    assert_eq!(kernel.body.len(), 1);
}

#[test]
fn spec_override_specializes() {
    let source = Source::new(
        "<spec>",
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) {
            spec MODE: u32 = 1;
            if MODE == 1 {
                a[0] = 1.0;
            } else {
                a[0] = 2.0;
            }
        }",
    );
    let kernel = saturn_compiler::Driver::new()
        .compile_with_specs(&source, &[("MODE", 0.0)])
        .expect("compile with override");
    assert_eq!(kernel.body.len(), 1);
    let saturn_compiler::ir::Stmt::Assign { value, .. } = &kernel.body[0] else {
        panic!("expected assign");
    };
    let saturn_compiler::ir::Expr::FloatLit { value, .. } = value else {
        panic!("expected literal, got {value:?}");
    };
    assert_eq!(*value, 2.0);
}

#[test]
fn rejects_unknown_spec() {
    let source = Source::new(
        "<spec>",
        "kernel k [workgroup(1,1,1)] (a: buf<f32>) {
            spec MODE: u32 = 1;
            a[0] = MODE as f32;
        }",
    );
    let err = saturn_compiler::Driver::new()
        .compile_with_specs(&source, &[("NOPE", 0.0)])
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("unknown spec"));
}

#[test]
fn rejects_spec_out_of_range() {
    let source = Source::new(
        "<spec>",
        "kernel k [workgroup(1,1,1)] (a: buf<u32>) {
            spec MODE: u32 = 1;
            a[0] = MODE;
        }",
    );
    let err = saturn_compiler::Driver::new()
        .compile_with_specs(&source, &[("MODE", -1.0)])
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("out of range"));
}

#[test]
fn rejects_barrier_in_divergent_flow() {
    let err = compile(
        "kernel k [workgroup(64,1,1)] (a: buf<f32>) {
            if thread.x < 32 {
                barrier();
            }
        }",
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("non-uniform"));
}

#[test]
fn barrier_under_uniform_condition_passes() {
    let kernel = compile(
        "kernel k [workgroup(64,1,1)] (a: buf<f32>, n: u32) {
            let limit = min(n, 32);
            if block.x < limit {
                barrier();
            }
        }",
    )
    .expect("uniform condition barrier must compile");
    assert!(contains_barrier(&kernel.body));
}

fn contains_barrier(stmts: &[saturn_compiler::ir::Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        saturn_compiler::ir::Stmt::Barrier { .. } => true,
        saturn_compiler::ir::Stmt::If { then, els, .. } => {
            contains_barrier(then) || contains_barrier(els)
        }
        saturn_compiler::ir::Stmt::Loop { body, .. }
        | saturn_compiler::ir::Stmt::For { body, .. } => contains_barrier(body),
        _ => false,
    })
}

#[test]
fn new_builtins_compile() {
    let kernel = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<u32>, f: buf<f32>) {
            a[0] = popcount(a[0]) + clz(a[0]) + ctz(a[0]);
            f[0] = trunc(f[0]) + sign(f[0]) + fract(f[0]);
            let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
            f[1] = dot(v, v);
            atomic_add(a, 1, 1);
            atomic_and(a, 2, 3);
            atomic_or(a, 3, 1);
            atomic_xor(a, 4, 5);
        }",
    )
    .expect("new builtins must compile");
    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn multi_error_recovery() {
    let err = compile(
        "kernel k [workgroup(1,1,1)] (a: buf<u32>) {
            let x = ;
            let y = ;
        }",
    )
    .expect_err("should fail");
    assert!(err.len() >= 2, "expected multiple diagnostics, got {err:?}");
}

#[test]
fn return_outside_function_rejected() {
    let err = compile("kernel k [workgroup(1,1,1)] (a: buf<u32>) { return; }")
        .expect_err("should fail");
    assert!(diag_msg(&err).contains("only allowed inside functions"));
}

#[test]
fn shared_inside_function_rejected() {
    let err = compile(
        r#"
        fn f(x: u32) -> u32 {
            shared s: [u32; 4];
            return x;
        }
        kernel k [workgroup(1,1,1)] (a: buf<u32>) {
            a[0] = f(1);
        }
        "#,
    )
    .expect_err("should fail");
    assert!(diag_msg(&err).contains("shared"));
}
