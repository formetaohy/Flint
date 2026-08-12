use saturn_compiler::ir::{BinOp, Expr, Kernel, MemOrder, Scalar, Stmt, Type, UnOp};

type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Wrapped {
    Load { elem: Scalar, space: &'static str },
    Store { elem: Scalar, space: &'static str },
    MulAdd { ab: Scalar, c: Scalar },
}

struct Msl {
    out: String,
    builtins: Vec<&'static str>,
    wrapped: std::collections::HashSet<Wrapped>,
}

pub fn to_msl(kernel: &Kernel) -> Result<(String, String)> {
    let mut msl = Msl {
        out: String::new(),
        builtins: Vec::new(),
        wrapped: std::collections::HashSet::new(),
    };
    msl.emit_kernel(kernel)?;
    Ok((msl.out, kernel.name.clone()))
}

impl Msl {
    fn emit_kernel(&mut self, kernel: &Kernel) -> Result<()> {
        self.collect_builtins(&kernel.body);
        self.emit_structs(kernel);
        self.emit_wrappers();
        self.out.push_str("kernel void ");
        self.out.push_str(&kernel.name);
        self.out.push('(');
        let mut first = true;
        for param in kernel.params.iter() {
            if !first {
                self.out.push_str(", ");
            }
            first = false;
            let Type::Buf(elem) = &param.ty else {
                return Err("buffer parameter has non-buffer type".to_string());
            };
            if param.access == saturn_compiler::ir::Access::ReadOnly {
                self.out.push_str("const ");
            }
            self.out.push_str("device ");
            self.out.push_str(&msl_type(elem));
            self.out.push_str("* ");
            self.out.push_str(&param.name);
            self.out.push_str(&format!(" [[buffer({})]]", param.binding));
        }
        let scalars_base = kernel
            .params
            .iter()
            .map(|p| p.binding)
            .max()
            .map_or(0, |max| max + 1);
        for (index, scalar) in kernel.scalars.iter().enumerate() {
            if !first {
                self.out.push_str(", ");
            }
            first = false;
            self.out.push_str("constant ");
            self.out.push_str(msl_scalar(scalar.ty));
            self.out.push_str("& ");
            self.out.push_str(&scalar.name);
            self.out.push_str(&format!(
                " [[buffer({})]]",
                scalars_base + index as u32
            ));
        }
        for builtin in &self.builtins {
            if !first {
                self.out.push_str(", ");
            }
            first = false;
            let attr = match *builtin {
                "global_id" => "thread_position_in_grid",
                "local_id" => "thread_position_in_threadgroup",
                "group_id" => "threadgroup_position_in_grid",
                "group_size" => "threads_per_threadgroup",
                "lane" => "simd_lane_id",
                "subgroup_id" => "threadgroup_position_in_simdgroup",
                "subgroup_size" => "simdgroups_per_threadgroup",
                _ => unreachable!(),
            };
            if matches!(
                *builtin,
                "lane" | "subgroup_id" | "subgroup_size"
            ) {
                self.out.push_str(&format!("uint {builtin} [[{attr}]]"));
            } else {
                self.out.push_str(&format!("uint3 {builtin} [[{attr}]]"));
            }
        }
        self.out.push_str(") {\n");
        let threadgroup_vars = collect_threadgroup_vars(&kernel.body);
        if !threadgroup_vars.is_empty() {
            for (name, elem, len) in &threadgroup_vars {
                self.out.push_str("    threadgroup ");
                self.out.push_str(&msl_type(elem));
                self.out.push(' ');
                self.out.push_str(name);
                self.out.push_str(&format!("[{len}];\n"));
            }
            self.out.push('\n');
        }
        self.emit_stmts(&kernel.body, 1)?;
        self.out.push_str(
            "}
",
        );
        Ok(())
    }

    fn emit_structs(&mut self, kernel: &Kernel) {
        for decl in &kernel.structs {
            self.out.push_str(&format!("struct {} {{\n", decl.name));
            for (name, ty) in &decl.fields {
                self.out.push_str(&format!("    {} {};\n", msl_scalar(scalar_of(ty)), name));
            }
            self.out.push_str("};\n\n");
        }
    }

    fn emit_wrappers(&mut self) {
        let mut loads: Vec<Wrapped> = self
            .wrapped
            .iter()
            .filter(|w| matches!(w, Wrapped::Load { .. }))
            .copied()
            .collect();
        loads.sort_by_key(|w| match w {
            Wrapped::Load { elem, space } => (msl_scalar(*elem), *space),
            _ => unreachable!(),
        });
        for wrapped in &loads {
            let Wrapped::Load { elem, space } = *wrapped else {
                continue;
            };
            let name = msl_scalar(elem);
            let ty = msl_simdgroup(elem);
            self.out.push_str(&format!(
                "metal::{ty} NagaCooperativeLoad(const {space} {name}* ptr, ulong stride, bool is_row_major) {{\n"
            ));
            self.out.push_str(&format!("    metal::{ty} m;\n"));
            self.out
                .push_str("    simdgroup_load(m, ptr, stride, 0, is_row_major);\n");
            self.out.push_str("    return m;\n}\n\n");
        }
        let mut mul_adds: Vec<Wrapped> = self
            .wrapped
            .iter()
            .filter(|w| matches!(w, Wrapped::MulAdd { .. }))
            .copied()
            .collect();
        mul_adds.sort_by_key(|w| match w {
            Wrapped::MulAdd { ab, c } => (msl_scalar(*ab), msl_scalar(*c)),
            _ => unreachable!(),
        });
        for wrapped in &mul_adds {
            let Wrapped::MulAdd { ab, c } = *wrapped else {
                continue;
            };
            let ab_ty = msl_simdgroup(ab);
            let c_ty = msl_simdgroup(c);
            self.out.push_str(&format!(
                "metal::{c_ty} NagaCooperativeMultiplyAdd(const metal::{ab_ty}& a, const metal::{ab_ty}& b, const metal::{c_ty}& c) {{\n"
            ));
            self.out.push_str(&format!("    metal::{c_ty} d;\n"));
            self.out
                .push_str("    simdgroup_multiply_accumulate(d, a, b, c);\n");
            self.out.push_str("    return d;\n}\n\n");
        }
        let mut stores: Vec<Wrapped> = self
            .wrapped
            .iter()
            .filter(|w| matches!(w, Wrapped::Store { .. }))
            .copied()
            .collect();
        stores.sort_by_key(|w| match w {
            Wrapped::Store { elem, space } => (msl_scalar(*elem), *space),
            _ => unreachable!(),
        });
        for wrapped in &stores {
            let Wrapped::Store { elem, space } = *wrapped else {
                continue;
            };
            let name = msl_scalar(elem);
            let ty = msl_simdgroup(elem);
            self.out.push_str(&format!(
                "void NagaCooperativeStore({space} {name}* ptr, metal::{ty} m, ulong stride, bool is_row_major) {{\n"
            ));
            self.out
                .push_str("    simdgroup_store(m, ptr, stride, 0, is_row_major);\n}\n\n");
        }
    }

    fn collect_builtins(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init, .. } => self.collect_expr(init),
                Stmt::Var { init: Some(init), .. } => self.collect_expr(init),
                Stmt::Var { init: None, .. } => {}
                Stmt::Assign { target, value, .. } => {
                    self.collect_expr(target);
                    self.collect_expr(value);
                }
                Stmt::If {
                    cond, then, els, ..
                } => {
                    self.collect_expr(cond);
                    self.collect_builtins(then);
                    self.collect_builtins(els);
                }
                Stmt::Loop { body, .. } => self.collect_builtins(body),
                Stmt::For {
                    start, end, body, ..
                } => {
                    self.collect_expr(start);
                    self.collect_expr(end);
                    self.collect_builtins(body);
                }
                Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Barrier { .. } => {}
                Stmt::ExprStmt { expr, .. } => self.collect_expr(expr),
            }
        }
    }

    fn collect_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Call { name, args, ty, .. } => {
                for arg in args {
                    self.collect_expr(arg);
                }
                if matches!(
                    *name,
                    "global_id"
                        | "local_id"
                        | "group_id"
                        | "group_size"
                        | "subgroup_id"
                        | "lane"
                        | "subgroup_size"
                ) && !self.builtins.contains(name)
                {
                    self.builtins.push(name);
                }
                match *name {
                    "coop_load_a" | "coop_load_b" => {
                        let elem = scalar_of(ty);
                        let space = match &args[0] {
                            Expr::LocalRef { .. } => "threadgroup",
                            _ => "device",
                        };
                        self.wrapped.insert(Wrapped::Load { elem, space });
                    }
                    "coop_mul_add" => {
                        let ab = scalar_of(match &args[0] {
                            Expr::LocalRef { ty, .. } => ty,
                            _ => &Type::Scalar(Scalar::F16),
                        });
                        let c = scalar_of(ty);
                        self.wrapped.insert(Wrapped::MulAdd { ab, c });
                    }
                    "coop_store" => {
                        let elem = scalar_of(match &args[1] {
                            Expr::LocalRef { ty, .. } => ty,
                            _ => &Type::Scalar(Scalar::F32),
                        });
                        let space = match &args[0] {
                            Expr::LocalRef { .. } => "threadgroup",
                            _ => "device",
                        };
                        self.wrapped.insert(Wrapped::Store { elem, space });
                    }
                    _ => {}
                }
            }
            Expr::Index { base, index, .. } => {
                self.collect_expr(base);
                self.collect_expr(index);
            }
            Expr::Field { base, .. } => self.collect_expr(base),
            Expr::Unary { expr, .. } => self.collect_expr(expr),
            Expr::Binary { lhs, rhs, .. } => {
                self.collect_expr(lhs);
                self.collect_expr(rhs);
            }
            Expr::Cond {
                cond, then, els, ..
            } => {
                self.collect_expr(cond);
                self.collect_expr(then);
                self.collect_expr(els);
            }
            Expr::Convert { expr, .. } => self.collect_expr(expr),
            Expr::ConstructStruct { fields, .. } => {
                for (_, value) in fields {
                    self.collect_expr(value);
                }
            }
            _ => {}
        }
    }

    fn emit_stmts(&mut self, stmts: &[Stmt], indent: usize) -> Result<()> {
        for stmt in stmts {
            self.emit_stmt(stmt, indent)?;
        }
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: &Stmt, indent: usize) -> Result<()> {
        let pad = "    ".repeat(indent);
        match stmt {
            Stmt::Let { name, ty, init, .. } => {
                self.out.push_str(&pad);
                self.out.push_str("const ");
                self.out.push_str(&msl_decl_type(ty));
                self.out.push(' ');
                self.out.push_str(name);
                self.out.push_str(" = ");
                self.emit_expr(init)?;
                self.out.push_str(";\n");
            }
            Stmt::Var {
                name,
                ty,
                init,
                ..
            } => {
                if matches!(ty, Type::Threadgroup(_)) {
                    return Ok(());
                }
                self.out.push_str(&pad);
                self.out.push_str(&msl_decl_type(ty));
                self.out.push(' ');
                self.out.push_str(name);
                if let Some(init) = init {
                    self.out.push_str(" = ");
                    self.emit_expr(init)?;
                }
                self.out.push_str(";\n");
            }
            Stmt::Assign { target, value, .. } => {
                self.out.push_str(&pad);
                emit_lvalue(&mut self.out, target)?;
                self.out.push_str(" = ");
                self.emit_expr(value)?;
                self.out.push_str(";\n");
            }
            Stmt::If {
                cond, then, els, ..
            } => {
                self.out.push_str(&pad);
                self.out.push_str("if (");
                self.emit_expr(cond)?;
                self.out.push_str(") {\n");
                self.emit_stmts(then, indent + 1)?;
                if els.is_empty() {
                    self.out.push_str(&pad);
                    self.out.push_str("}\n");
                } else {
                    self.out.push_str(&pad);
                    self.out.push_str("} else {\n");
                    self.emit_stmts(els, indent + 1)?;
                    self.out.push_str(&pad);
                    self.out.push_str("}\n");
                }
            }
            Stmt::Loop { body, .. } => {
                self.out.push_str(&pad);
                self.out.push_str("while (true) {\n");
                self.emit_stmts(body, indent + 1)?;
                self.out.push_str(&pad);
                self.out.push_str("}\n");
            }
            Stmt::For {
                var,
                start,
                end,
                body,
                ..
            } => {
                self.out.push_str(&pad);
                self.out.push_str("for (uint ");
                self.out.push_str(var);
                self.out.push_str(" = ");
                self.emit_expr(start)?;
                self.out.push_str("; ");
                self.out.push_str(var);
                self.out.push_str(" < ");
                self.emit_expr(end)?;
                self.out.push_str("; ++");
                self.out.push_str(var);
                self.out.push_str(") {\n");
                self.emit_stmts(body, indent + 1)?;
                self.out.push_str(&pad);
                self.out.push_str("}\n");
            }
            Stmt::Break { .. } => {
                self.out.push_str(&pad);
                self.out.push_str("break;\n");
            }
            Stmt::Barrier { .. } => {
                self.out.push_str(&pad);
                self.out.push_str(
                    "threadgroup_barrier(mem_flags::mem_threadgroup);
",
                );
            }
            Stmt::Continue { .. } => {
                self.out.push_str(&pad);
                self.out.push_str("continue;\n");
            }
            Stmt::ExprStmt { expr, .. } => {
                self.out.push_str(&pad);
                self.emit_expr(expr)?;
                self.out.push_str(";\n");
            }
        }
        Ok(())
    }

    fn emit_coop_ptr(&mut self, expr: &Expr) -> Result<()> {
        match expr {
            Expr::LocalRef { name, .. } | Expr::ParamRef { name, .. } => {
                self.out.push_str(name);
            }
            Expr::Index { base, index, .. } => {
                self.emit_coop_ptr(base)?;
                self.out.push_str(" + ");
                let mut msl = Msl {
                    out: String::new(),
                    builtins: Vec::new(),
                    wrapped: std::collections::HashSet::new(),
                };
                msl.emit_expr(index)?;
                self.out.push_str(&msl.out);
            }
            _ => return Err("coop source must be a buffer or shared array".to_string()),
        }
        Ok(())
    }

    fn emit_expr(&mut self, expr: &Expr) -> Result<()> {
        match expr {
            Expr::IntLit { value, ty, .. } => {
                self.out.push_str(&value.to_string());
                if *ty == Scalar::U32 {
                    self.out.push('u');
                }
            }
            Expr::FloatLit { value, ty, .. } => match ty {
                Scalar::F16 => {
                    self.out.push_str(&format!("(half){value}"));
                }
                _ => {
                    let text = value.to_string();
                    if !text.contains('.') && !text.contains('e') && !text.contains('E') {
                        self.out.push_str(&text);
                        self.out.push_str(".0");
                    } else {
                        self.out.push_str(&text);
                    }
                }
            },
            Expr::BoolLit { value, .. } => {
                self.out.push_str(if *value { "true" } else { "false" });
            }
            Expr::LocalRef { name, .. } => self.out.push_str(name),
            Expr::ParamRef { name, .. } => self.out.push_str(name),
            Expr::ScalarRef { name, .. } => self.out.push_str(name),
            Expr::Index { base, index, .. } => {
                self.emit_expr(base)?;
                self.out.push('[');
                self.emit_expr(index)?;
                self.out.push(']');
            }
            Expr::Field { base, name, .. } => {
                self.emit_expr(base)?;
                self.out.push('.');
                self.out.push_str(name);
            }
            Expr::Unary { op, expr, .. } => {
                match op {
                    UnOp::Neg => self.out.push('-'),
                    UnOp::Not => self.out.push('!'),
                }
                self.out.push('(');
                self.emit_expr(expr)?;
                self.out.push(')');
            }
            Expr::Binary {
                op, lhs, rhs, ty, ..
            } => {
                self.out.push('(');
                self.emit_expr(lhs)?;
                let text = match op {
                    BinOp::Add => " + ",
                    BinOp::Sub => " - ",
                    BinOp::Mul => " * ",
                    BinOp::Div => " / ",
                    BinOp::Rem => " % ",
                    BinOp::And => " & ",
                    BinOp::Or => " | ",
                    BinOp::Xor => " ^ ",
                    BinOp::Shl => " << ",
                    BinOp::Shr => " >> ",
                    BinOp::Eq => " == ",
                    BinOp::Ne => " != ",
                    BinOp::Lt => " < ",
                    BinOp::Le => " <= ",
                    BinOp::Gt => " > ",
                    BinOp::Ge => " >= ",
                    BinOp::LAnd => " && ",
                    BinOp::LOr => " || ",
                };
                self.out.push_str(text);
                if scalar_of(ty) == Scalar::F16
                    && matches!(
                        op,
                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
                    )
                {
                    self.out.push_str("(half)");
                }
                self.emit_expr(rhs)?;
                self.out.push(')');
            }
            Expr::Cond {
                cond, then, els, ..
            } => {
                self.out.push('(');
                self.emit_expr(cond)?;
                self.out.push_str(" ? ");
                self.emit_expr(then)?;
                self.out.push_str(" : ");
                self.emit_expr(els)?;
                self.out.push(')');
            }
            Expr::Convert { ty, expr, .. } => {
                let source = source_scalar(expr);
                match (source, *ty) {
                    (Some(Scalar::F32), Scalar::Bf16) => {
                        self.out.push_str("(ushort)((as_type<uint>(");
                        self.emit_expr(expr)?;
                        self.out.push_str(") >> 16) & 0xFFFFu)");
                        return Ok(());
                    }
                    (Some(Scalar::Bf16), Scalar::F32) => {
                        self.out.push_str("as_type<float>(uint(");
                        self.emit_expr(expr)?;
                        self.out.push_str(") << 16)");
                        return Ok(());
                    }
                    (Some(Scalar::I8 | Scalar::U8), Scalar::F32) => {
                        self.out.push_str("(float)(");
                        self.emit_expr(expr)?;
                        self.out.push(')');
                        return Ok(());
                    }
                    (Some(Scalar::F32), Scalar::I8 | Scalar::U8) => {
                        self.out.push_str("(char)(int)(");
                        self.emit_expr(expr)?;
                        self.out.push(')');
                        return Ok(());
                    }
                    _ => {}
                }
                self.out.push('(');
                self.out.push_str(msl_scalar(*ty));
                self.out.push_str(")(");
                self.emit_expr(expr)?;
                self.out.push(')');
            }
            Expr::OrderLit { order, .. } => {
                self.out.push_str(msl_order(*order));
            }
            Expr::ConstructStruct { name, fields, .. } => {
                self.out.push_str(name);
                self.out.push('{');
                for (index, (_, value)) in fields.iter().enumerate() {
                    if index > 0 {
                        self.out.push_str(", ");
                    }
                    self.emit_expr(value)?;
                }
                self.out.push('}');
            }
            Expr::Call { name, args, ty, .. } => {
                if matches!(
                    *name,
                    "global_id"
                        | "local_id"
                        | "group_id"
                        | "group_size"
                        | "subgroup_id"
                        | "lane"
                        | "subgroup_size"
                ) {
                    self.out.push_str(name);
                    return Ok(());
                }
                if *name == "construct_vec" {
                    self.out.push('(');
                    self.out.push_str(&msl_decl_type(ty));
                    self.out.push('(');
                    for (index, arg) in args.iter().enumerate() {
                        if index > 0 {
                            self.out.push_str(", ");
                        }
                        self.emit_expr(arg)?;
                    }
                    self.out.push_str("))");
                    return Ok(());
                }
                if *name == "swizzle_vec" {
                    self.emit_expr(&args[0])?;
                    self.out.push('.');
                    for mask_arg in &args[1..] {
                        let idx = match mask_arg {
                            Expr::IntLit { value, .. } => *value as usize,
                            _ => 0,
                        };
                        self.out.push(match idx {
                            0 => 'x',
                            1 => 'y',
                            2 => 'z',
                            _ => 'w',
                        });
                    }
                    return Ok(());
                }
                if matches!(
                    *name,
                    "atomic_add"
                        | "atomic_max"
                        | "atomic_min"
                        | "atomic_exchange"
                        | "atomic_and"
                        | "atomic_or"
                        | "atomic_xor"
                ) {
                    let (space, base_name, index) = split_addr(&args[0]);
                    let ty_name = msl_scalar(scalar_of(ty));
                    let fn_name = match *name {
                        "atomic_add" => "atomic_fetch_add_explicit",
                        "atomic_max" => "atomic_fetch_max_explicit",
                        "atomic_min" => "atomic_fetch_min_explicit",
                        "atomic_and" => "atomic_fetch_and_explicit",
                        "atomic_or" => "atomic_fetch_or_explicit",
                        "atomic_xor" => "atomic_fetch_xor_explicit",
                        _ => "atomic_exchange_explicit",
                    };
                    self.out.push_str(fn_name);
                    self.out.push_str("((");
                    self.out.push_str(space);
                    self.out.push_str(" atomic_");
                    self.out.push_str(ty_name);
                    self.out.push_str("*)&");
                    self.out.push_str(base_name);
                    self.out.push('[');
                    let mut msl = Msl {
                        out: String::new(),
                        builtins: Vec::new(),
                        wrapped: std::collections::HashSet::new(),
                    };
                    match index {
                        Some(index) => msl.emit_expr(index)?,
                        None => msl.out.push('0'),
                    }
                    self.out.push_str(&msl.out);
                    self.out.push_str("], ");
                    self.emit_expr(&args[1])?;
                    self.out.push_str(", ");
                    let order = match &args[2] {
                        Expr::OrderLit { order, .. } => *order,
                        _ => MemOrder::Relaxed,
                    };
                    self.out.push_str(msl_order(order));
                    self.out.push(')');
                    return Ok(());
                }
                if *name == "coop_zero" {
                    self.out.push_str("metal::make_filled_simdgroup_matrix<");
                    self.out.push_str(msl_scalar(scalar_of(ty)));
                    self.out.push_str(", 16, 16>(0.0)");
                    return Ok(());
                }
                if *name == "coop_load_a" || *name == "coop_load_b" {
                    self.out.push_str("NagaCooperativeLoad(");
                    self.emit_coop_ptr(&args[0])?;
                    self.out.push_str(", ");
                    self.emit_expr(&args[1])?;
                    self.out.push_str(", ");
                    let row_major = match &args[2] {
                        Expr::IntLit { value, .. } => *value == 0,
                        _ => false,
                    };
                    self.out.push_str(if row_major { "true" } else { "false" });
                    self.out.push(')');
                    return Ok(());
                }
                if *name == "coop_mul_add" {
                    self.out.push_str("NagaCooperativeMultiplyAdd(");
                    self.emit_expr(&args[0])?;
                    self.out.push_str(", ");
                    self.emit_expr(&args[1])?;
                    self.out.push_str(", ");
                    self.emit_expr(&args[2])?;
                    self.out.push(')');
                    return Ok(());
                }
                if *name == "coop_store" {
                    self.out.push_str("NagaCooperativeStore(");
                    self.emit_coop_ptr(&args[0])?;
                    self.out.push_str(", ");
                    self.emit_expr(&args[1])?;
                    self.out.push_str(", ");
                    self.emit_expr(&args[2])?;
                    self.out.push_str(", ");
                    let row_major = match &args[3] {
                        Expr::IntLit { value, .. } => *value == 0,
                        _ => false,
                    };
                    self.out.push_str(if row_major { "true" } else { "false" });
                    self.out.push(')');
                    return Ok(());
                }
                if *name == "bitcast_f32" {
                    self.out.push_str("as_type<float>(");
                    self.emit_expr(&args[0])?;
                    self.out.push(')');
                    return Ok(());
                }
                if *name == "bitcast_u32" {
                    self.out.push_str("as_type<uint>(");
                    self.emit_expr(&args[0])?;
                    self.out.push(')');
                    return Ok(());
                }
                let msl_name = match *name {
                    "subgroup_broadcast" => "simd_broadcast",
                    "subgroup_shuffle" => "simd_shuffle",
                    "subgroup_shuffle_down" => "simd_shuffle_down",
                    "subgroup_shuffle_up" => "simd_shuffle_up",
                    "subgroup_reduce_add" => "simd_sum",
                    "subgroup_reduce_max" => "simd_max",
                    "subgroup_reduce_min" => "simd_min",
                    "subgroup_inclusive_add" => "simd_prefix_sum",
                    "subgroup_all" => "simd_all",
                    "subgroup_any" => "simd_any",
                    _ => name,
                };
                let ty_scalar = scalar_of(ty);
                if ty_scalar == Scalar::F16 {
                    self.out.push_str("(half)");
                }
                self.out.push_str(msl_name);
                self.out.push('(');
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        self.out.push_str(", ");
                    }
                    self.emit_expr(arg)?;
                }
                self.out.push(')');
            }
        }
        Ok(())
    }
}

fn emit_lvalue(out: &mut String, expr: &Expr) -> Result<()> {
    match expr {
        Expr::LocalRef { name, .. }
        | Expr::ParamRef { name, .. }
        | Expr::ScalarRef { name, .. } => out.push_str(name),
        Expr::Index { base, index, .. } => {
            emit_lvalue(out, base)?;
            out.push('[');
            let mut msl = Msl {
                out: String::new(),
                builtins: Vec::new(),
                wrapped: std::collections::HashSet::new(),
            };
            msl.emit_expr(index)?;
            out.push_str(&msl.out);
            out.push(']');
        }
        Expr::Field { base, name, .. } => {
            emit_lvalue(out, base)?;
            out.push('.');
            out.push_str(name);
        }
        _ => return Err("invalid assignment target".to_string()),
    }
    Ok(())
}

fn split_addr<'a>(expr: &'a Expr) -> (&'static str, &'a str, Option<&'a Expr>) {
    match expr {
        Expr::Index { base, index, .. } => match &**base {
            Expr::LocalRef { name, .. } => ("threadgroup", name, Some(index)),
            _ => ("device", name_of(&**base), Some(index)),
        },
        _ => ("device", name_of(expr), None),
    }
}

fn name_of(expr: &Expr) -> &str {
    match expr {
        Expr::LocalRef { name, .. } | Expr::ParamRef { name, .. } => name,
        _ => "",
    }
}

fn collect_threadgroup_vars(stmts: &[Stmt]) -> Vec<(String, Type, u64)> {
    let mut out = Vec::new();
    for stmt in stmts {
        match stmt {
            Stmt::Var { name, ty, .. } => {
                if let Type::Threadgroup(inner) = ty {
                    if let Type::Array { elem, len } = &**inner {
                        out.push((name.clone(), (**elem).clone(), *len));
                    }
                }
            }
            Stmt::If { then, els, .. } => {
                out.extend(collect_threadgroup_vars(then));
                out.extend(collect_threadgroup_vars(els));
            }
            Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                out.extend(collect_threadgroup_vars(body));
            }
            _ => {}
        }
    }
    out
}

fn source_scalar(expr: &Expr) -> Option<Scalar> {
    match expr {
        Expr::LocalRef { ty, .. } => ty.scalar(),
        Expr::ScalarRef { ty, .. } => Some(*ty),
        Expr::IntLit { ty, .. } | Expr::FloatLit { ty, .. } => Some(*ty),
        Expr::Index { ty, .. } | Expr::Field { ty, .. } => ty.scalar(),
        Expr::Unary { ty, .. } | Expr::Convert { ty, .. } => Some(*ty),
        Expr::Cond { ty, .. } | Expr::Binary { ty, .. } | Expr::Call { ty, .. } => ty.scalar(),
        _ => None,
    }
}

fn msl_order(order: MemOrder) -> &'static str {
    match order {
        MemOrder::Relaxed => "memory_order_relaxed",
        MemOrder::Acquire => "memory_order_acquire",
        MemOrder::Release => "memory_order_release",
        MemOrder::AcqRel => "memory_order_acq_rel",
        MemOrder::SeqCst => "memory_order_seq_cst",
    }
}

fn msl_simdgroup(elem: Scalar) -> &'static str {
    match elem {
        Scalar::F16 => "simdgroup_half16x16",
        Scalar::F32 => "simdgroup_float16x16",
        _ => unreachable!("sema rejects cooperative matrices with non-float elements"),
    }
}

fn scalar_of(ty: &Type) -> Scalar {
    match ty {
        Type::Scalar(scalar) => *scalar,
        Type::Matrix { elem, .. } => *elem,
        Type::Vec { elem, .. } => *elem,
        _ => unreachable!("not a scalar type"),
    }
}

fn msl_decl_type(ty: &Type) -> String {
    match ty {
        Type::Scalar(scalar) => msl_scalar(*scalar).to_string(),
        Type::Matrix { elem, .. } => {
            let name = match elem {
                Scalar::F16 => "half",
                _ => "float",
            };
            format!("metal::simdgroup_{name}16x16")
        }
        Type::Vec { size, elem } => format!("{}{}", msl_scalar(*elem), size),
        Type::Array { elem, len } => format!("{} [{len}]", msl_decl_type(elem)),
        Type::Threadgroup(elem) => msl_decl_type(elem),
        Type::Struct { name, .. } => name.clone(),
        _ => unreachable!("not a local type"),
    }
}

fn msl_type(ty: &Type) -> String {
    match ty {
        Type::Scalar(scalar) => msl_scalar(*scalar).to_string(),
        Type::Vec { size, elem } => format!("{}{}", msl_scalar(*elem), size),
        Type::Struct { name, .. } => name.clone(),
        _ => unreachable!("not a buffer element type"),
    }
}

fn msl_scalar(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::F32 => "float",
        Scalar::F16 => "half",
        Scalar::Bf16 => "ushort",
        Scalar::I32 => "int",
        Scalar::U32 => "uint",
        Scalar::I8 => "char",
        Scalar::U8 => "uchar",
        Scalar::Bool => "bool",
    }
}
