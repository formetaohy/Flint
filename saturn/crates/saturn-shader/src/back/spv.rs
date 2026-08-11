use std::collections::{HashMap, HashSet};

use saturn_compiler::ir::{BinOp, Expr, Kernel, MatrixRole, Scalar, Stmt, Type, UnOp};
use spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, GlslStd450Op,
    MemoryModel, Op, StorageClass,
};

type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum T {
    Void,
    Bool,
    Int {
        width: u32,
        signed: bool,
    },
    Float {
        width: u32,
    },
    Vec {
        size: u32,
        elem: Box<T>,
    },
    RtArray(Box<T>),
    Array {
        elem: Box<T>,
        len: u64,
    },
    Struct(Box<T>),
    PushStruct(Vec<Scalar>),
    CoopMat {
        elem: Scalar,
        role: MatrixRole,
    },
    Ptr {
        class: StorageClass,
        pointee: Box<T>,
    },
    Function,
}

impl T {
    fn from_scalar(scalar: Scalar) -> T {
        match scalar {
            Scalar::F32 => T::Float { width: 32 },
            Scalar::F16 => T::Float { width: 16 },
            Scalar::Bf16 => T::Int {
                width: 16,
                signed: false,
            },
            Scalar::I32 => T::Int {
                width: 32,
                signed: true,
            },
            Scalar::U32 => T::Int {
                width: 32,
                signed: false,
            },
            Scalar::I8 => T::Int {
                width: 8,
                signed: true,
            },
            Scalar::U8 => T::Int {
                width: 8,
                signed: false,
            },
            Scalar::Bool => T::Bool,
        }
    }

    fn width(&self) -> u32 {
        match self {
            T::Void => 0,
            T::Bool => 1,
            T::Int { width, .. } | T::Float { width } => *width,
            T::Vec { size, elem } => elem.width() * *size,
            _ => unreachable!("no scalar width"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ConstKey {
    Int { ty: T, bits: u32 },
    Float { ty: T, bits: u32 },
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BuiltinVar {
    Gid,
    Thread,
    Block,
    BlockDim,
    Lane,
    SubgroupId,
    SubgroupSize,
}

struct Collect {
    types: HashSet<T>,
    builtins: HashSet<BuiltinVar>,
    subgroup_caps: HashSet<u32>,
    has_f16: bool,
    has_int16: bool,
    has_int8: bool,
    has_glsl_ext: bool,
    has_coop: bool,
}

struct Spv {
    caps: Vec<u32>,
    extensions: Vec<u32>,
    import: Option<u32>,
    import_words: Vec<u32>,
    entry: Vec<u32>,
    exec: Vec<u32>,
    modes: Vec<u32>,
    types: Vec<u32>,
    fns: Vec<u32>,
    next_id: u32,
    type_ids: HashMap<T, u32>,
    const_ids: HashMap<ConstKey, u32>,
    param_index: HashMap<String, usize>,
    param_vars: Vec<u32>,
    push_var: Option<u32>,
    push_members: HashMap<String, u32>,
    shared_vars: HashMap<String, u32>,
    builtin_vars: HashMap<BuiltinVar, u32>,
    locals: HashMap<u32, u32>,
    loop_stack: Vec<LoopCtx>,
    block_dim_const: Option<u32>,
    coop_zero_ids: HashMap<(Scalar, MatrixRole), u32>,
    block_terminated: bool,
}

struct LoopCtx {
    continue_label: u32,
    merge: u32,
}

const U32: T = T::Int {
    width: 32,
    signed: false,
};
const BOOL: T = T::Bool;

pub fn to_spirv(kernel: &Kernel) -> Result<Vec<u8>> {
    let mut collect = Collect {
        types: HashSet::new(),
        builtins: HashSet::new(),
        subgroup_caps: HashSet::new(),
        has_f16: false,
        has_int16: false,
        has_int8: false,
        has_glsl_ext: false,
        has_coop: false,
    };
    for param in &kernel.params {
        match param.elem {
            Scalar::F16 => collect.has_f16 = true,
            Scalar::Bf16 => collect.has_int16 = true,
            Scalar::I8 | Scalar::U8 => collect.has_int8 = true,
            _ => {}
        }
    }
    for scalar in &kernel.scalars {
        match scalar.ty {
            Scalar::F16 => collect.has_f16 = true,
            Scalar::Bf16 => collect.has_int16 = true,
            Scalar::I8 | Scalar::U8 => collect.has_int8 = true,
            _ => {}
        }
    }
    collect.types.insert(BOOL);
    for stmt in &kernel.body {
        collect_stmt(&mut collect, stmt);
    }

    let mut spv = Spv {
        caps: Vec::new(),
        extensions: Vec::new(),
        import: None,
        import_words: Vec::new(),
        entry: Vec::new(),
        exec: Vec::new(),
        modes: Vec::new(),
        types: Vec::new(),
        fns: Vec::new(),
        next_id: 1,
        type_ids: HashMap::new(),
        const_ids: HashMap::new(),
        param_index: HashMap::new(),
        param_vars: Vec::new(),
        push_var: None,
        push_members: kernel
            .scalars
            .iter()
            .enumerate()
            .map(|(index, p)| (p.name.clone(), index as u32))
            .collect(),
        shared_vars: HashMap::new(),
        builtin_vars: HashMap::new(),
        locals: HashMap::new(),
        loop_stack: Vec::new(),
        block_dim_const: None,
        coop_zero_ids: HashMap::new(),
        block_terminated: false,
    };

    spv.caps.push(Capability::Shader as u32);
    if collect.has_f16 {
        spv.caps.push(Capability::Float16 as u32);
    }
    if collect.has_int16 {
        spv.caps.push(Capability::Int16 as u32);
    }
    if collect.has_int8 {
        spv.caps.push(Capability::Int8 as u32);
    }
    spv.caps.push(Capability::GroupNonUniform as u32);
    for cap in &collect.subgroup_caps {
        spv.caps.push(*cap);
    }
    if collect.has_coop {
        spv.caps.push(6022);
        spv.caps.push(5345);
    }
    spv.caps = spv
        .caps
        .iter()
        .flat_map(|cap| inst(Op::Capability, &[*cap]))
        .collect();
    let mut extensions = Vec::new();
    if collect.has_coop {
        extensions.extend(extension_words("SPV_KHR_cooperative_matrix"));
        extensions.extend(extension_words("SPV_KHR_vulkan_memory_model"));
    }
    spv.extensions = extensions;
    if collect.has_glsl_ext {
        let id = spv.alloc();
        spv.import = Some(id);
        spv.import_words = ext_inst_import_words(id);
    }

    let mut decorated: HashSet<T> = HashSet::new();
    for (index, param) in kernel.params.iter().enumerate() {
        let elem = T::from_scalar(param.elem);
        let rt_key = T::RtArray(Box::new(elem.clone()));
        let rt = spv.ensure_type(&rt_key);
        let struct_ty = spv.ensure_type(&T::Struct(Box::new(T::RtArray(Box::new(elem.clone())))));
        if decorated.insert(rt_key) {
            spv.modes.extend(inst(
                Op::Decorate,
                &[rt, Decoration::ArrayStride as u32, elem.width() / 8],
            ));
            spv.modes
                .extend(inst(Op::Decorate, &[struct_ty, Decoration::Block as u32]));
            spv.modes.extend(inst(
                Op::MemberDecorate,
                &[struct_ty, 0, Decoration::Offset as u32, 0],
            ));
        }
        let ptr = spv.ensure_type(&T::Ptr {
            class: StorageClass::StorageBuffer,
            pointee: Box::new(T::Struct(Box::new(T::RtArray(Box::new(elem))))),
        });
        let var_id = spv.alloc();
        spv.param_index.insert(param.name.clone(), index);
        spv.param_vars.push(var_id);
        spv.types.extend(inst(
            Op::Variable,
            &[ptr, var_id, StorageClass::StorageBuffer as u32],
        ));
        spv.modes.extend(inst(
            Op::Decorate,
            &[var_id, Decoration::DescriptorSet as u32, 0],
        ));
        spv.modes.extend(inst(
            Op::Decorate,
            &[var_id, Decoration::Binding as u32, index as u32],
        ));
    }

    let push_var = if !kernel.scalars.is_empty() {
        let struct_ty = spv.ensure_type(&T::PushStruct(
            kernel.scalars.iter().map(|p| p.ty).collect(),
        ));
        spv.modes
            .extend(inst(Op::Decorate, &[struct_ty, Decoration::Block as u32]));
        for (index, scalar) in kernel.scalars.iter().enumerate() {
            spv.modes.extend(inst(
                Op::MemberDecorate,
                &[
                    struct_ty,
                    index as u32,
                    Decoration::Offset as u32,
                    scalar.offset,
                ],
            ));
        }
        let ptr_ty = spv.ensure_type(&T::Ptr {
            class: StorageClass::PushConstant,
            pointee: Box::new(T::PushStruct(kernel.scalars.iter().map(|p| p.ty).collect())),
        });
        let var_id = spv.alloc();
        spv.types.extend(inst(
            Op::Variable,
            &[ptr_ty, var_id, StorageClass::PushConstant as u32],
        ));
        Some(var_id)
    } else {
        None
    };
    spv.push_var = push_var;

    let mut interface = Vec::new();
    for builtin in &collect.builtins {
        let builtin_kind = match builtin {
            BuiltinVar::Gid => BuiltIn::GlobalInvocationId,
            BuiltinVar::Thread => BuiltIn::LocalInvocationId,
            BuiltinVar::Block => BuiltIn::WorkgroupId,
            BuiltinVar::Lane => BuiltIn::SubgroupLocalInvocationId,
            BuiltinVar::SubgroupId => BuiltIn::SubgroupId,
            BuiltinVar::SubgroupSize => BuiltIn::SubgroupSize,
            BuiltinVar::BlockDim => continue,
        };
        let var_id = spv.alloc();
        spv.builtin_vars.insert(builtin.clone(), var_id);
        interface.push(var_id);
        let pointee = match builtin {
            BuiltinVar::Gid | BuiltinVar::Thread | BuiltinVar::Block => T::Vec {
                size: 3,
                elem: Box::new(U32),
            },
            BuiltinVar::Lane | BuiltinVar::SubgroupId | BuiltinVar::SubgroupSize => U32,
            BuiltinVar::BlockDim => unreachable!(),
        };
        let ptr_ty = spv.ensure_type(&T::Ptr {
            class: StorageClass::Input,
            pointee: Box::new(pointee),
        });
        spv.types.extend(inst(
            Op::Variable,
            &[ptr_ty, var_id, StorageClass::Input as u32],
        ));
        spv.modes.extend(inst(
            Op::Decorate,
            &[var_id, Decoration::BuiltIn as u32, builtin_kind as u32],
        ));
    }

    for shared in &kernel.shareds {
        let elem = T::from_scalar(shared.elem);
        let array_key = T::Array {
            elem: Box::new(elem.clone()),
            len: shared.len,
        };
        let array_ty = spv.ensure_type(&array_key);
        let _ = (array_ty, &decorated);
        let ptr_ty = spv.ensure_type(&T::Ptr {
            class: StorageClass::Workgroup,
            pointee: Box::new(T::Array {
                elem: Box::new(elem),
                len: shared.len,
            }),
        });
        let var_id = spv.alloc();
        spv.types.extend(inst(
            Op::Variable,
            &[ptr_ty, var_id, StorageClass::Workgroup as u32],
        ));
        spv.shared_vars.insert(shared.name.clone(), var_id);
    }

    let entry_id = spv.alloc();
    let mut entry_words = vec![0, ExecutionModel::GLCompute as u32, entry_id];
    entry_words.extend(string_words(&kernel.name));
    entry_words.extend(&interface);
    let count = entry_words.len() as u32;
    entry_words[0] = (Op::EntryPoint as u32) | (count << 16);
    spv.entry = entry_words;

    spv.exec.extend(inst(
        Op::ExecutionMode,
        &[
            entry_id,
            ExecutionMode::LocalSize as u32,
            kernel.workgroup_size[0],
            kernel.workgroup_size[1],
            kernel.workgroup_size[2],
        ],
    ));

    if collect.builtins.contains(&BuiltinVar::BlockDim) {
        let x = spv.constant(&ConstKey::Int {
            ty: U32,
            bits: kernel.workgroup_size[0],
        });
        let y = spv.constant(&ConstKey::Int {
            ty: U32,
            bits: kernel.workgroup_size[1],
        });
        let z = spv.constant(&ConstKey::Int {
            ty: U32,
            bits: kernel.workgroup_size[2],
        });
        let vec_ty = spv.ensure_type(&T::Vec {
            size: 3,
            elem: Box::new(U32),
        });
        let id = spv.alloc();
        spv.block_dim_const = Some(id);
        spv.types
            .extend(inst(Op::ConstantComposite, &[vec_ty, id, x, y, z]));
    }
    let fn_ty = spv.ensure_type(&T::Function);
    for ty in &collect.types {
        spv.ensure_type(ty);
    }
    spv.fns.extend(inst(
        Op::Function,
        &[spv.type_id(&T::Void), entry_id, 0, fn_ty],
    ));
    let entry_label = spv.alloc();
    spv.fns.extend(inst(Op::Label, &[entry_label]));
    let mut local_vars = Vec::new();
    collect_locals(&kernel.body, &mut local_vars);
    for (id, ty) in local_vars {
        let ptr_ty = spv.ensure_type(&T::Ptr {
            class: StorageClass::Function,
            pointee: Box::new(type_of(&ty)),
        });
        let var_id = spv.alloc();
        spv.locals.insert(id, var_id);
        spv.fns.extend(inst(
            Op::Variable,
            &[ptr_ty, var_id, StorageClass::Function as u32],
        ));
    }
    spv.emit_stmts(&kernel.body)?;
    spv.fns.extend(inst(Op::Return, &[]));
    spv.fns.extend(inst(Op::FunctionEnd, &[]));

    let bound = spv.next_id;
    let mut words = vec![0x0723_0203, 0x0001_0300, 0, bound, 0];
    words.extend(&spv.caps);
    words.extend(&spv.extensions);
    words.extend(&spv.import_words);
    words.extend(inst(
        Op::MemoryModel,
        &[
            AddressingModel::Logical as u32,
            if collect.has_coop {
                MemoryModel::Vulkan as u32
            } else {
                MemoryModel::GLSL450 as u32
            },
        ],
    ));
    words.extend(&spv.entry);
    words.extend(&spv.exec);
    words.extend(&spv.modes);
    words.extend(&spv.types);
    words.extend(&spv.fns);

    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    Ok(bytes)
}

impl Spv {
    fn alloc(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn type_id(&self, ty: &T) -> u32 {
        self.type_ids[ty]
    }

    fn ensure_type(&mut self, ty: &T) -> u32 {
        if let Some(&id) = self.type_ids.get(ty) {
            return id;
        }
        let id = self.alloc();
        self.type_ids.insert(ty.clone(), id);
        let words = match ty {
            T::Void => inst(Op::TypeVoid, &[id]),
            T::Bool => inst(Op::TypeBool, &[id]),
            T::Int { width, signed } => inst(Op::TypeInt, &[id, *width, *signed as u32]),
            T::Float { width } => inst(Op::TypeFloat, &[id, *width]),
            T::Vec { size, elem } => {
                let elem_id = self.ensure_type(elem);
                inst(Op::TypeVector, &[id, elem_id, *size])
            }
            T::RtArray(elem) => {
                let elem_id = self.ensure_type(elem);
                inst(Op::TypeRuntimeArray, &[id, elem_id])
            }
            T::Struct(field) => {
                let field_id = self.ensure_type(field);
                inst(Op::TypeStruct, &[id, field_id])
            }
            T::Array { elem, len } => {
                let elem_id = self.ensure_type(elem);
                let len_id = self.constant(&ConstKey::Int {
                    ty: U32,
                    bits: *len as u32,
                });
                inst(Op::TypeArray, &[id, elem_id, len_id])
            }
            T::CoopMat { elem, role } => {
                let comp = self.ensure_type(&T::from_scalar(*elem));
                let scope = self.constant(&ConstKey::Int { ty: U32, bits: 3 });
                let rows = self.constant(&ConstKey::Int { ty: U32, bits: 16 });
                let cols = self.constant(&ConstKey::Int { ty: U32, bits: 16 });
                let coop_use = self.constant(&ConstKey::Int {
                    ty: U32,
                    bits: match role {
                        MatrixRole::A => 0,
                        MatrixRole::B => 1,
                        MatrixRole::Acc => 2,
                    },
                });
                inst_raw(4456, &[id, comp, scope, rows, cols, coop_use])
            }
            T::PushStruct(fields) => {
                let mut operands = vec![id];
                for scalar in fields {
                    let member = self.ensure_type(&T::from_scalar(*scalar));
                    operands.push(member);
                }
                inst(Op::TypeStruct, &operands)
            }
            T::Ptr { class, pointee } => {
                let pointee_id = self.ensure_type(pointee);
                inst(Op::TypePointer, &[id, *class as u32, pointee_id])
            }
            T::Function => {
                let void_id = self.ensure_type(&T::Void);
                inst(Op::TypeFunction, &[id, void_id])
            }
        };
        self.types.extend(words);
        id
    }

    fn constant(&mut self, key: &ConstKey) -> u32 {
        if let Some(&id) = self.const_ids.get(key) {
            return id;
        }
        let id = self.alloc();
        self.const_ids.insert(key.clone(), id);
        let words = match key {
            ConstKey::Int { ty, bits } => {
                let ty_id = self.ensure_type(ty);
                inst(Op::Constant, &[ty_id, id, *bits])
            }
            ConstKey::Float { ty, bits } => {
                let ty_id = self.ensure_type(ty);
                inst(Op::Constant, &[ty_id, id, *bits])
            }
            ConstKey::Bool(value) => {
                let ty_id = self.ensure_type(&BOOL);
                inst(
                    if *value {
                        Op::ConstantTrue
                    } else {
                        Op::ConstantFalse
                    },
                    &[ty_id, id],
                )
            }
        };
        self.types.extend(words);
        id
    }

    fn push_fn(&mut self, words: Vec<u32>) {
        let op = words.first().map(|word| word & 0xFFFF);
        if op == Some(Op::Label as u32) {
            self.block_terminated = false;
        } else if op.is_some_and(|op| matches!(op, 249..=255)) {
            self.block_terminated = true;
        }
        self.fns.extend(words);
    }

    fn end_block(&mut self, target: u32) {
        if !self.block_terminated {
            self.push_fn(inst(Op::Branch, &[target]));
        }
    }

    fn emit_stmts(&mut self, stmts: &[Stmt]) -> Result<()> {
        for stmt in stmts {
            self.emit_stmt(stmt)?;
        }
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { id, init, .. } | Stmt::Var { id, init, .. } => {
                let var_id = self.locals[id];
                let init_id = self.emit_expr(init)?;
                self.push_fn(inst(Op::Store, &[var_id, init_id]));
            }
            Stmt::Assign { target, value, .. } => {
                let ptr_id = self.emit_target(target)?;
                let value_id = self.emit_expr(value)?;
                self.push_fn(inst(Op::Store, &[ptr_id, value_id]));
            }
            Stmt::If {
                cond, then, els, ..
            } => {
                let cond_id = self.emit_expr(cond)?;
                let then_label = self.alloc();
                let els_label = self.alloc();
                let merge = self.alloc();
                self.push_fn(inst(Op::SelectionMerge, &[merge, 0]));
                self.push_fn(inst(
                    Op::BranchConditional,
                    &[cond_id, then_label, els_label],
                ));
                self.push_fn(inst(Op::Label, &[then_label]));
                self.emit_stmts(then)?;
                self.end_block(merge);
                self.push_fn(inst(Op::Label, &[els_label]));
                self.emit_stmts(els)?;
                self.end_block(merge);
                self.push_fn(inst(Op::Label, &[merge]));
            }
            Stmt::Loop { body, .. } => {
                let header = self.alloc();
                let body_label = self.alloc();
                let continue_label = self.alloc();
                let merge = self.alloc();
                self.push_fn(inst(Op::Branch, &[header]));
                self.push_fn(inst(Op::Label, &[header]));
                self.fns
                    .extend(inst(Op::LoopMerge, &[merge, continue_label, 0]));
                self.push_fn(inst(Op::Branch, &[body_label]));
                self.push_fn(inst(Op::Label, &[body_label]));
                self.loop_stack.push(LoopCtx {
                    continue_label,
                    merge,
                });
                self.emit_stmts(body)?;
                self.end_block(continue_label);
                self.push_fn(inst(Op::Label, &[continue_label]));
                self.push_fn(inst(Op::Branch, &[header]));
                self.push_fn(inst(Op::Label, &[merge]));
                self.loop_stack.pop();
            }
            Stmt::For {
                id,
                start,
                end,
                body,
                ..
            } => {
                let start_id = self.emit_expr(start)?;
                let end_id = self.emit_expr(end)?;
                let u32_ty = self.type_id(&U32);
                let var_id = self.locals[id];
                self.push_fn(inst(Op::Store, &[var_id, start_id]));
                let header = self.alloc();
                let cond_label = self.alloc();
                let body_label = self.alloc();
                let continue_label = self.alloc();
                let merge = self.alloc();
                self.push_fn(inst(Op::Branch, &[header]));
                self.push_fn(inst(Op::Label, &[header]));
                self.fns
                    .extend(inst(Op::LoopMerge, &[merge, continue_label, 0]));
                self.push_fn(inst(Op::Branch, &[cond_label]));
                self.push_fn(inst(Op::Label, &[cond_label]));
                let load_id = self.alloc();
                self.push_fn(inst(Op::Load, &[u32_ty, load_id, var_id]));
                let cmp_id = self.alloc();
                self.push_fn(inst(
                    Op::ULessThan,
                    &[self.type_id(&BOOL), cmp_id, load_id, end_id],
                ));
                self.fns
                    .extend(inst(Op::BranchConditional, &[cmp_id, body_label, merge]));
                self.push_fn(inst(Op::Label, &[body_label]));
                self.loop_stack.push(LoopCtx {
                    continue_label,
                    merge,
                });
                self.emit_stmts(body)?;
                self.end_block(continue_label);
                self.push_fn(inst(Op::Label, &[continue_label]));
                let one = self.constant(&ConstKey::Int { ty: U32, bits: 1 });
                let load2 = self.alloc();
                self.push_fn(inst(Op::Load, &[u32_ty, load2, var_id]));
                let inc_id = self.alloc();
                self.push_fn(inst(Op::IAdd, &[u32_ty, inc_id, load2, one]));
                self.push_fn(inst(Op::Store, &[var_id, inc_id]));
                self.push_fn(inst(Op::Branch, &[header]));
                self.push_fn(inst(Op::Label, &[merge]));
                self.loop_stack.pop();
            }
            Stmt::Break { .. } => {
                let merge = self.loop_stack.last().unwrap().merge;
                self.push_fn(inst(Op::Branch, &[merge]));
            }
            Stmt::Continue { .. } => {
                let continue_label = self.loop_stack.last().unwrap().continue_label;
                self.push_fn(inst(Op::Branch, &[continue_label]));
            }
            Stmt::ExprStmt { expr, .. } => {
                self.emit_expr(expr)?;
            }
            Stmt::Barrier { .. } => {
                let scope = self.constant(&ConstKey::Int { ty: U32, bits: 2 });
                let semantics = self.constant(&ConstKey::Int { ty: U32, bits: 264 });
                self.push_fn(inst(Op::ControlBarrier, &[scope, scope, semantics]));
            }
        }
        Ok(())
    }

    fn emit_target(&mut self, target: &Expr) -> Result<u32> {
        match target {
            Expr::LocalRef { id, .. } => Ok(self.locals[id]),
            Expr::Index { base, index, .. } => match &**base {
                Expr::ParamRef { name, elem, .. } => {
                    let var_id = self.param_vars[self.param_index[name]];
                    let index_id = self.emit_expr(index)?;
                    let ptr_elem = self.ensure_type(&T::Ptr {
                        class: StorageClass::StorageBuffer,
                        pointee: Box::new(T::from_scalar(*elem)),
                    });
                    let zero = self.constant(&ConstKey::Int { ty: U32, bits: 0 });
                    let result = self.alloc();
                    self.push_fn(inst(
                        Op::AccessChain,
                        &[ptr_elem, result, var_id, zero, index_id],
                    ));
                    Ok(result)
                }
                Expr::SharedRef { name, elem, .. } => {
                    let var_id = self.shared_vars[name];
                    let index_id = self.emit_expr(index)?;
                    let ptr_elem = self.ensure_type(&T::Ptr {
                        class: StorageClass::Workgroup,
                        pointee: Box::new(T::from_scalar(*elem)),
                    });
                    let result = self.alloc();
                    self.push_fn(inst(Op::AccessChain, &[ptr_elem, result, var_id, index_id]));
                    Ok(result)
                }
                _ => Err("index base must be a buffer or shared array".to_string()),
            },
            _ => Err("invalid assignment target".to_string()),
        }
    }

    fn emit_expr(&mut self, expr: &Expr) -> Result<u32> {
        match expr {
            Expr::IntLit { value, ty, .. } => {
                let scalar = T::from_scalar(*ty);
                let bits = *value as u32;
                Ok(self.constant(&ConstKey::Int { ty: scalar, bits }))
            }
            Expr::FloatLit { value, ty, .. } => {
                let scalar = T::from_scalar(*ty);
                let bits = match ty {
                    Scalar::F32 => (*value as f32).to_bits(),
                    Scalar::F16 => f32_to_f16(*value as f32),
                    _ => unreachable!(),
                };
                Ok(self.constant(&ConstKey::Float { ty: scalar, bits }))
            }
            Expr::BoolLit { value, .. } => Ok(self.constant(&ConstKey::Bool(*value))),
            Expr::LocalRef { id, ty, .. } => {
                let var_id = self.locals[id];
                let result = self.alloc();
                let load_ty = self.ensure_type(&type_of(ty));
                self.push_fn(inst(Op::Load, &[load_ty, result, var_id]));
                Ok(result)
            }
            Expr::ParamRef { .. } => Err("buffer parameter used as value".to_string()),
            Expr::ScalarRef { name, ty, .. } => {
                let var_id = self.push_var.expect("push constant missing");
                let member = self.constant(&ConstKey::Int {
                    ty: U32,
                    bits: self.push_members[name],
                });
                let ptr_ty = self.ensure_type(&T::Ptr {
                    class: StorageClass::PushConstant,
                    pointee: Box::new(T::from_scalar(*ty)),
                });
                let ptr_id = self.alloc();
                self.push_fn(inst(Op::AccessChain, &[ptr_ty, ptr_id, var_id, member]));
                let result = self.alloc();
                self.push_fn(inst(
                    Op::Load,
                    &[self.type_id(&T::from_scalar(*ty)), result, ptr_id],
                ));
                Ok(result)
            }
            Expr::SharedRef { .. } => Err("shared array used as value".to_string()),
            Expr::Builtin { name, .. } => {
                if *name == "block_dim" {
                    return Ok(self.block_dim_const.expect("block_dim constant missing"));
                }
                if matches!(*name, "lane" | "subgroup_id" | "subgroup_size") {
                    let var = match *name {
                        "lane" => &BuiltinVar::Lane,
                        "subgroup_id" => &BuiltinVar::SubgroupId,
                        _ => &BuiltinVar::SubgroupSize,
                    };
                    let var_id = self.builtin_vars[var];
                    let result = self.alloc();
                    let u32_ty = self.ensure_type(&U32);
                    self.push_fn(inst(Op::Load, &[u32_ty, result, var_id]));
                    return Ok(result);
                }
                let var_id = match *name {
                    "gid" => self.builtin_vars[&BuiltinVar::Gid],
                    "thread" => self.builtin_vars[&BuiltinVar::Thread],
                    "block" => self.builtin_vars[&BuiltinVar::Block],
                    _ => unreachable!(),
                };
                let vec_ty = self.ensure_type(&T::Vec {
                    size: 3,
                    elem: Box::new(U32),
                });
                let result = self.alloc();
                self.push_fn(inst(Op::Load, &[vec_ty, result, var_id]));
                Ok(result)
            }
            Expr::Index {
                base, index, ty, ..
            } => {
                let ptr_id = self.emit_target(&Expr::Index {
                    base: base.clone(),
                    index: index.clone(),
                    ty: *ty,
                    span: saturn_compiler::Span::dummy(),
                })?;
                let result = self.alloc();
                self.push_fn(inst(
                    Op::Load,
                    &[self.type_id(&T::from_scalar(*ty)), result, ptr_id],
                ));
                Ok(result)
            }
            Expr::Member { base, idx, ty, .. } => {
                let base_id = self.emit_expr(base)?;
                let result = self.alloc();
                self.push_fn(inst(
                    Op::CompositeExtract,
                    &[self.type_id(&T::from_scalar(*ty)), result, base_id, *idx],
                ));
                Ok(result)
            }
            Expr::Unary { op, expr, ty, .. } => match op {
                UnOp::Neg => {
                    let operand = self.emit_expr(expr)?;
                    let result = self.alloc();
                    let op = match ty {
                        Scalar::F32 | Scalar::F16 => Op::FNegate,
                        Scalar::I32 => Op::SNegate,
                        _ => unreachable!(),
                    };
                    self.push_fn(inst(
                        op,
                        &[self.type_id(&T::from_scalar(*ty)), result, operand],
                    ));
                    Ok(result)
                }
                UnOp::Not => {
                    let operand = self.emit_expr(expr)?;
                    let result = self.alloc();
                    self.fns.extend(inst(
                        Op::LogicalNot,
                        &[self.type_id(&BOOL), result, operand],
                    ));
                    Ok(result)
                }
            },
            Expr::Binary {
                op, lhs, rhs, ty, ..
            } => {
                let lhs_id = self.emit_expr(lhs)?;
                let rhs_id = self.emit_expr(rhs)?;
                let result = self.alloc();
                let operand_ty = self.expr_type(lhs);
                let operand_scalar = match operand_ty {
                    T::Float { width } => {
                        if width == 16 {
                            Scalar::F16
                        } else {
                            Scalar::F32
                        }
                    }
                    T::Int { signed: true, .. } => Scalar::I32,
                    T::Int { signed: false, .. } => Scalar::U32,
                    T::Bool => Scalar::Bool,
                    T::Vec { elem, .. } => match *elem {
                        T::Float { width } => {
                            if width == 16 {
                                Scalar::F16
                            } else {
                                Scalar::F32
                            }
                        }
                        _ => unreachable!("non-scalar vec operand"),
                    },
                    _ => unreachable!("non-scalar operand"),
                };
                let result_ty = match op {
                    BinOp::Eq
                    | BinOp::Ne
                    | BinOp::Lt
                    | BinOp::Le
                    | BinOp::Gt
                    | BinOp::Ge
                    | BinOp::LAnd
                    | BinOp::LOr => BOOL,
                    _ => type_of(ty),
                };
                let op = match op {
                    BinOp::Add => match operand_scalar {
                        Scalar::F32 | Scalar::F16 => Op::FAdd,
                        _ => Op::IAdd,
                    },
                    BinOp::Sub => match operand_scalar {
                        Scalar::F32 | Scalar::F16 => Op::FSub,
                        _ => Op::ISub,
                    },
                    BinOp::Mul => match operand_scalar {
                        Scalar::F32 | Scalar::F16 => Op::FMul,
                        _ => Op::IMul,
                    },
                    BinOp::Div => match operand_scalar {
                        Scalar::F32 | Scalar::F16 => Op::FDiv,
                        Scalar::I32 => Op::SDiv,
                        _ => Op::UDiv,
                    },
                    BinOp::Rem => match operand_scalar {
                        Scalar::F32 | Scalar::F16 => Op::FRem,
                        Scalar::I32 => Op::SRem,
                        _ => Op::UMod,
                    },
                    BinOp::And => Op::BitwiseAnd,
                    BinOp::Or => Op::BitwiseOr,
                    BinOp::Xor => Op::BitwiseXor,
                    BinOp::Shl => Op::ShiftLeftLogical,
                    BinOp::Shr => match operand_scalar {
                        Scalar::I32 => Op::ShiftRightArithmetic,
                        _ => Op::ShiftRightLogical,
                    },
                    BinOp::Eq => match operand_scalar {
                        Scalar::F32 | Scalar::F16 => Op::FOrdEqual,
                        Scalar::Bool => {
                            self.push_fn(inst(
                                Op::LogicalEqual,
                                &[self.type_id(&BOOL), result, lhs_id, rhs_id],
                            ));
                            return Ok(result);
                        }
                        _ => Op::IEqual,
                    },
                    BinOp::Ne => match operand_scalar {
                        Scalar::F32 | Scalar::F16 => Op::FOrdNotEqual,
                        Scalar::Bool => {
                            let eq = self.alloc();
                            self.push_fn(inst(
                                Op::LogicalEqual,
                                &[self.type_id(&BOOL), eq, lhs_id, rhs_id],
                            ));
                            self.fns
                                .extend(inst(Op::LogicalNot, &[self.type_id(&BOOL), result, eq]));
                            return Ok(result);
                        }
                        _ => Op::INotEqual,
                    },
                    BinOp::Lt => cmp_op(operand_scalar, false, false),
                    BinOp::Le => cmp_op(operand_scalar, false, true),
                    BinOp::Gt => cmp_op(operand_scalar, true, false),
                    BinOp::Ge => cmp_op(operand_scalar, true, true),
                    BinOp::LAnd => Op::LogicalAnd,
                    BinOp::LOr => Op::LogicalOr,
                };
                self.push_fn(inst(
                    op,
                    &[self.type_id(&result_ty), result, lhs_id, rhs_id],
                ));
                Ok(result)
            }
            Expr::Cond {
                cond,
                then,
                els,
                ty,
                ..
            } => {
                let cond_id = self.emit_expr(cond)?;
                let then_id = self.emit_expr(then)?;
                let els_id = self.emit_expr(els)?;
                let result = self.alloc();
                self.push_fn(inst(
                    Op::Select,
                    &[
                        self.type_id(&T::from_scalar(*ty)),
                        result,
                        cond_id,
                        then_id,
                        els_id,
                    ],
                ));
                Ok(result)
            }
            Expr::Convert { ty, expr, .. } => {
                let operand = self.emit_expr(expr)?;
                let source = self.expr_type(expr);
                let target = T::from_scalar(*ty);
                let result = self.alloc();
                let op = convert_op(&target, &source);
                let mut chain = vec![operand];
                let mut final_ty = target.clone();
                match (&source, *ty) {
                    (T::Float { width: 32 }, Scalar::Bf16) => {
                        let u32_ty = self.type_id(&U32);
                        let u16_ty = self.type_id(&T::from_scalar(Scalar::Bf16));
                        let bitcast = self.alloc();
                        let shift = self.alloc();
                        let result_ty = self.type_id(&target);
                        self.push_fn(inst(Op::Bitcast, &[u32_ty, bitcast, chain[0]]));
                        let sixteen = self.constant(&ConstKey::Int { ty: U32, bits: 16 });
                        self.push_fn(inst(
                            Op::ShiftRightLogical,
                            &[u32_ty, shift, bitcast, sixteen],
                        ));
                        self.push_fn(inst(Op::UConvert, &[u16_ty, result, shift]));
                        let _ = result_ty;
                        return Ok(result);
                    }
                    (
                        T::Int {
                            width: 16,
                            signed: false,
                        },
                        Scalar::F32,
                    ) => {
                        let u32_ty = self.type_id(&U32);
                        let widen = self.alloc();
                        let shift = self.alloc();
                        self.push_fn(inst(Op::UConvert, &[u32_ty, widen, chain[0]]));
                        let sixteen = self.constant(&ConstKey::Int { ty: U32, bits: 16 });
                        self.push_fn(inst(Op::ShiftLeftLogical, &[u32_ty, shift, widen, sixteen]));
                        self.push_fn(inst(Op::Bitcast, &[self.type_id(&target), result, shift]));
                        return Ok(result);
                    }
                    (
                        T::Int {
                            width: 8,
                            signed: s8,
                        },
                        Scalar::F32,
                    ) => {
                        let mid = if *s8 {
                            T::Int {
                                width: 32,
                                signed: true,
                            }
                        } else {
                            U32
                        };
                        let widen_ty = self.type_id(&mid);
                        let widen = self.alloc();
                        let widen_op = if *s8 { Op::SConvert } else { Op::UConvert };
                        self.push_fn(inst(widen_op, &[widen_ty, widen, chain[0]]));
                        chain.push(widen);
                        final_ty = mid;
                    }
                    (T::Float { width: 32 }, Scalar::I8 | Scalar::U8) => {
                        let signed = *ty == Scalar::I8;
                        let mid = if signed {
                            T::Int {
                                width: 32,
                                signed: true,
                            }
                        } else {
                            U32
                        };
                        let trunc_ty = self.type_id(&mid);
                        let trunc = self.alloc();
                        let trunc_op = if signed {
                            Op::ConvertFToS
                        } else {
                            Op::ConvertFToU
                        };
                        self.push_fn(inst(trunc_op, &[trunc_ty, trunc, chain[0]]));
                        chain.push(trunc);
                        final_ty = mid;
                    }
                    (
                        T::Int {
                            width: 8,
                            signed: s1,
                        },
                        Scalar::I32 | Scalar::U32,
                    )
                    | (
                        T::Int {
                            width: 32,
                            signed: s1,
                        },
                        Scalar::I8 | Scalar::U8,
                    ) => {
                        let s2 = matches!(*ty, Scalar::I32 | Scalar::I8);
                        let mid = if s2 {
                            T::Int {
                                width: 32,
                                signed: true,
                            }
                        } else {
                            U32
                        };
                        let mid_ty = self.type_id(&mid);
                        let mid_id = self.alloc();
                        let mid_op = if *s1 { Op::SConvert } else { Op::UConvert };
                        self.push_fn(inst(mid_op, &[mid_ty, mid_id, chain[0]]));
                        chain.push(mid_id);
                        final_ty = mid;
                    }
                    _ => {}
                }
                if chain.len() > 1 && final_ty == target {
                    return Ok(chain[chain.len() - 1]);
                }
                let final_op = if chain.len() > 1 {
                    convert_op(&target, &final_ty)
                } else {
                    op
                };
                self.push_fn(inst(
                    final_op,
                    &[self.type_id(&target), result, chain[chain.len() - 1]],
                ));
                Ok(result)
            }
            Expr::Call { name, args, ty, .. } => {
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
                    let class = match &args[0] {
                        Expr::SharedRef { .. } => StorageClass::Workgroup,
                        _ => StorageClass::StorageBuffer,
                    };
                    let scope_bits = match class {
                        StorageClass::Workgroup => 2,
                        _ => 1,
                    };
                    let scope = self.constant(&ConstKey::Int {
                        ty: U32,
                        bits: scope_bits,
                    });
                    let relaxed = self.constant(&ConstKey::Int { ty: U32, bits: 0 });
                    let var_id = match &args[0] {
                        Expr::ParamRef { name, .. } => self.param_vars[self.param_index[name]],
                        Expr::SharedRef { name, .. } => self.shared_vars[name],
                        _ => unreachable!("atomic base"),
                    };
                    let elem = scalar_of(ty);
                    let elem_ty = self.ensure_type(&T::from_scalar(elem));
                    let ptr_ty = self.ensure_type(&T::Ptr {
                        class,
                        pointee: Box::new(T::from_scalar(elem)),
                    });
                    let index_id = self.emit_expr(&args[1])?;
                    let ptr_id = self.alloc();
                    if class == StorageClass::Workgroup {
                        self.push_fn(inst(Op::AccessChain, &[ptr_ty, ptr_id, var_id, index_id]));
                    } else {
                        let zero = self.constant(&ConstKey::Int { ty: U32, bits: 0 });
                        self.push_fn(inst(
                            Op::AccessChain,
                            &[ptr_ty, ptr_id, var_id, zero, index_id],
                        ));
                    }
                    let value_id = self.emit_expr(&args[2])?;
                    let result = self.alloc();
                    let op = match (*name, elem) {
                        ("atomic_add", _) => Op::AtomicIAdd,
                        ("atomic_max", Scalar::I32) => Op::AtomicSMax,
                        ("atomic_max", _) => Op::AtomicUMax,
                        ("atomic_min", Scalar::I32) => Op::AtomicSMin,
                        ("atomic_min", _) => Op::AtomicUMin,
                        ("atomic_exchange", _) => Op::AtomicExchange,
                        ("atomic_and", _) => Op::AtomicAnd,
                        ("atomic_or", _) => Op::AtomicOr,
                        ("atomic_xor", _) => Op::AtomicXor,
                        _ => unreachable!(),
                    };
                    self.push_fn(inst(
                        op,
                        &[elem_ty, result, ptr_id, scope, relaxed, value_id],
                    ));
                    return Ok(result);
                }
                if *name == "coop_zero" {
                    let (elem, role) = match ty {
                        Type::Matrix { elem, role } => (*elem, *role),
                        _ => unreachable!(),
                    };
                    let key = (elem, role);
                    if let Some(&id) = self.coop_zero_ids.get(&key) {
                        return Ok(id);
                    }
                    let coop_ty = self.ensure_type(&type_of(ty));
                    let zero = self.constant(&ConstKey::Float {
                        ty: T::from_scalar(elem),
                        bits: 0,
                    });
                    let id = self.alloc();
                    self.types
                        .extend(inst(Op::ConstantComposite, &[coop_ty, id, zero]));
                    self.coop_zero_ids.insert(key, id);
                    return Ok(id);
                }
                if *name == "coop_load_a" || *name == "coop_load_b" {
                    let ptr_id = self.emit_coop_ptr(&args[0])?;
                    let stride_id = self.emit_expr(&args[1])?;
                    let layout_id = self.emit_expr(&args[2])?;
                    let result = self.alloc();
                    let coop_ty = self.type_id(&type_of(ty));
                    self.push_fn(inst_raw(
                        4457,
                        &[coop_ty, result, ptr_id, layout_id, stride_id, 32],
                    ));
                    return Ok(result);
                }
                if *name == "coop_mul_add" {
                    let result = self.alloc();
                    let coop_ty = self.type_id(&type_of(ty));
                    let a = self.emit_expr(&args[0])?;
                    let b = self.emit_expr(&args[1])?;
                    let c = self.emit_expr(&args[2])?;
                    self.push_fn(inst_raw(4459, &[coop_ty, result, a, b, c]));
                    return Ok(result);
                }
                if *name == "construct_vec" {
                    let vec_ty = self.ensure_type(&type_of(ty));
                    let id = self.alloc();
                    let mut operands = vec![vec_ty, id];
                    for arg in args {
                        operands.push(self.emit_expr(arg)?);
                    }
                    self.push_fn(inst(Op::CompositeConstruct, &operands));
                    return Ok(id);
                }
                if *name == "swizzle_vec" {
                    let base_id = self.emit_expr(&args[0])?;
                    let id = self.alloc();
                    let mut operands = vec![self.ensure_type(&type_of(ty)), id, base_id, base_id];
                    for mask_arg in &args[1..] {
                        let value = match mask_arg {
                            Expr::IntLit { value, .. } => *value as u32,
                            _ => 0,
                        };
                        operands.push(value);
                    }
                    self.push_fn(inst(Op::VectorShuffle, &operands));
                    return Ok(id);
                }
                if *name == "coop_store" {
                    let ptr_id = self.emit_coop_ptr(&args[0])?;
                    let mat_id = self.emit_expr(&args[1])?;
                    let stride_id = self.emit_expr(&args[2])?;
                    let layout_id = self.emit_expr(&args[3])?;
                    self.push_fn(inst_raw(4458, &[ptr_id, mat_id, layout_id, stride_id, 32]));
                    return Ok(0);
                }
                let mut arg_ids = Vec::new();
                for arg in args {
                    arg_ids.push(self.emit_expr(arg)?);
                }
                let scalar_ty = scalar_of(ty);
                if *name == "bitcast_f32" || *name == "bitcast_u32" {
                    let result = self.alloc();
                    self.push_fn(inst(
                        Op::Bitcast,
                        &[self.type_id(&T::from_scalar(scalar_ty)), result, arg_ids[0]],
                    ));
                    return Ok(result);
                }
                if *name == "popcount" {
                    let result = self.alloc();
                    self.push_fn(inst(
                        Op::BitCount,
                        &[self.type_id(&T::from_scalar(scalar_ty)), result, arg_ids[0]],
                    ));
                    return Ok(result);
                }
                if *name == "clz" || *name == "ctz" {
                    let u32_ty = self.type_id(&U32);
                    let bool_ty = self.type_id(&BOOL);
                    let zero = self.constant(&ConstKey::Int { ty: U32, bits: 0 });
                    let is_zero = self.alloc();
                    self.push_fn(inst(Op::IEqual, &[bool_ty, is_zero, arg_ids[0], zero]));
                    let idx = self.alloc();
                    let ext = if *name == "clz" {
                        GlslStd450Op::FindUMsb
                    } else {
                        GlslStd450Op::FindILsb
                    };
                    self.gl_ext(ext, u32_ty, idx, &[arg_ids[0]]);
                    let width = self.constant(&ConstKey::Int { ty: U32, bits: 32 });
                    let val = if *name == "clz" {
                        let thirty_one = self.constant(&ConstKey::Int { ty: U32, bits: 31 });
                        let sub = self.alloc();
                        self.push_fn(inst(Op::ISub, &[u32_ty, sub, thirty_one, idx]));
                        sub
                    } else {
                        idx
                    };
                    let result = self.alloc();
                    self.push_fn(inst(Op::Select, &[u32_ty, result, is_zero, width, val]));
                    return Ok(result);
                }
                if *name == "dot" {
                    let result = self.alloc();
                    self.push_fn(inst(
                        Op::Dot,
                        &[
                            self.type_id(&T::from_scalar(scalar_ty)),
                            result,
                            arg_ids[0],
                            arg_ids[1],
                        ],
                    ));
                    return Ok(result);
                }
                match *name {
                    "min" | "max" | "clamp" => {
                        let result = self.alloc();
                        let ty_id = self.type_id(&T::from_scalar(scalar_ty));
                        let ext = match (scalar_ty, *name) {
                            (Scalar::F32 | Scalar::F16, "min") => GlslStd450Op::FMin,
                            (Scalar::F32 | Scalar::F16, "max") => GlslStd450Op::FMax,
                            (Scalar::F32 | Scalar::F16, _) => GlslStd450Op::FClamp,
                            (Scalar::I32, "min") => GlslStd450Op::SMin,
                            (Scalar::I32, "max") => GlslStd450Op::SMax,
                            (Scalar::I32, _) => GlslStd450Op::SClamp,
                            (_, "min") => GlslStd450Op::UMin,
                            (_, "max") => GlslStd450Op::UMax,
                            (_, _) => GlslStd450Op::UClamp,
                        };
                        self.gl_ext(ext, ty_id, result, &arg_ids);
                        Ok(result)
                    }
                    "abs" => {
                        let result = self.alloc();
                        let ty_id = self.type_id(&T::from_scalar(scalar_ty));
                        let ext = match scalar_ty {
                            Scalar::I32 => GlslStd450Op::SAbs,
                            _ => GlslStd450Op::FAbs,
                        };
                        self.gl_ext(ext, ty_id, result, &arg_ids);
                        Ok(result)
                    }
                    "floor" | "ceil" | "round" | "trunc" | "sign" | "fract" | "sqrt" | "rsqrt"
                    | "exp" | "exp2" | "log" | "log2" | "tanh" | "pow" | "fma" => {
                        let result = self.alloc();
                        let ty_id = self.type_id(&T::from_scalar(scalar_ty));
                        let ext = match *name {
                            "floor" => GlslStd450Op::Floor,
                            "ceil" => GlslStd450Op::Ceil,
                            "round" => GlslStd450Op::Round,
                            "trunc" => GlslStd450Op::Trunc,
                            "sign" => GlslStd450Op::FSign,
                            "fract" => GlslStd450Op::Fract,
                            "sqrt" => GlslStd450Op::Sqrt,
                            "rsqrt" => GlslStd450Op::InverseSqrt,
                            "exp" => GlslStd450Op::Exp,
                            "exp2" => GlslStd450Op::Exp2,
                            "log" => GlslStd450Op::Log,
                            "log2" => GlslStd450Op::Log2,
                            "tanh" => GlslStd450Op::Tanh,
                            "pow" => GlslStd450Op::Pow,
                            "fma" => GlslStd450Op::Fma,
                            _ => unreachable!(),
                        };
                        self.gl_ext(ext, ty_id, result, &arg_ids);
                        Ok(result)
                    }
                    "select" => {
                        let result = self.alloc();
                        self.push_fn(inst(
                            Op::Select,
                            &[
                                self.type_id(&T::from_scalar(scalar_ty)),
                                result,
                                arg_ids[2],
                                arg_ids[0],
                                arg_ids[1],
                            ],
                        ));
                        Ok(result)
                    }
                    "subgroup_broadcast"
                    | "subgroup_shuffle"
                    | "subgroup_shuffle_down"
                    | "subgroup_shuffle_up" => {
                        let scope = self.constant(&ConstKey::Int { ty: U32, bits: 3 });
                        let result = self.alloc();
                        let ty_id = self.type_id(&T::from_scalar(scalar_ty));
                        let op = match *name {
                            "subgroup_broadcast" => 337,
                            "subgroup_shuffle" => 345,
                            "subgroup_shuffle_down" => 348,
                            "subgroup_shuffle_up" => 347,
                            _ => unreachable!(),
                        };
                        self.push_fn(inst_raw(
                            op,
                            &[ty_id, result, scope, arg_ids[0], arg_ids[1]],
                        ));
                        Ok(result)
                    }
                    "subgroup_reduce_add"
                    | "subgroup_reduce_max"
                    | "subgroup_reduce_min"
                    | "subgroup_inclusive_add" => {
                        let scope = self.constant(&ConstKey::Int { ty: U32, bits: 3 });
                        let group_op = if *name == "subgroup_inclusive_add" {
                            spirv::GroupOperation::InclusiveScan as u32
                        } else {
                            spirv::GroupOperation::Reduce as u32
                        };
                        let result = self.alloc();
                        let ty_id = self.type_id(&T::from_scalar(scalar_ty));
                        let op = match (*name, scalar_ty) {
                            ("subgroup_reduce_add", Scalar::F32 | Scalar::F16)
                            | ("subgroup_inclusive_add", Scalar::F32 | Scalar::F16) => {
                                Op::GroupNonUniformFAdd
                            }
                            ("subgroup_reduce_add", _) | ("subgroup_inclusive_add", _) => {
                                Op::GroupNonUniformIAdd
                            }
                            ("subgroup_reduce_max", Scalar::I32) => Op::GroupNonUniformSMax,
                            ("subgroup_reduce_max", Scalar::F32 | Scalar::F16) => {
                                Op::GroupNonUniformFMax
                            }
                            ("subgroup_reduce_max", _) => Op::GroupNonUniformUMax,
                            ("subgroup_reduce_min", Scalar::I32) => Op::GroupNonUniformSMin,
                            ("subgroup_reduce_min", Scalar::F32 | Scalar::F16) => {
                                Op::GroupNonUniformFMin
                            }
                            ("subgroup_reduce_min", _) => Op::GroupNonUniformUMin,
                            _ => unreachable!(),
                        };
                        self.push_fn(inst(op, &[ty_id, result, scope, group_op, arg_ids[0]]));
                        Ok(result)
                    }
                    "subgroup_all" | "subgroup_any" => {
                        let scope = self.constant(&ConstKey::Int { ty: U32, bits: 3 });
                        let result = self.alloc();
                        let op = match *name {
                            "subgroup_all" => 334,
                            _ => 335,
                        };
                        self.push_fn(inst_raw(
                            op,
                            &[self.type_id(&BOOL), result, scope, arg_ids[0]],
                        ));
                        Ok(result)
                    }
                    _ => Err(format!("unknown builtin '{name}'")),
                }
            }
        }
    }

    fn emit_coop_ptr(&mut self, expr: &Expr) -> Result<u32> {
        match expr {
            Expr::Index {
                base, index, ty, ..
            } => {
                let index_id = self.emit_expr(index)?;
                match &**base {
                    Expr::SharedRef { name, .. } => {
                        let var_id = self.shared_vars[name];
                        let ptr_ty = self.ensure_type(&T::Ptr {
                            class: StorageClass::Workgroup,
                            pointee: Box::new(T::from_scalar(*ty)),
                        });
                        let result = self.alloc();
                        self.push_fn(inst(Op::AccessChain, &[ptr_ty, result, var_id, index_id]));
                        Ok(result)
                    }
                    Expr::ParamRef { name, .. } => {
                        let var_id = self.param_vars[self.param_index[name]];
                        let zero = self.constant(&ConstKey::Int { ty: U32, bits: 0 });
                        let ptr_ty = self.ensure_type(&T::Ptr {
                            class: StorageClass::StorageBuffer,
                            pointee: Box::new(T::from_scalar(*ty)),
                        });
                        let result = self.alloc();
                        self.push_fn(inst(
                            Op::AccessChain,
                            &[ptr_ty, result, var_id, zero, index_id],
                        ));
                        Ok(result)
                    }
                    _ => Err("coop source must be a buffer or shared array".to_string()),
                }
            }
            Expr::SharedRef { name, elem, .. } => {
                let var_id = self.shared_vars[name];
                let zero = self.constant(&ConstKey::Int { ty: U32, bits: 0 });
                let ptr_ty = self.ensure_type(&T::Ptr {
                    class: StorageClass::Workgroup,
                    pointee: Box::new(T::from_scalar(*elem)),
                });
                let result = self.alloc();
                self.push_fn(inst(Op::AccessChain, &[ptr_ty, result, var_id, zero]));
                Ok(result)
            }
            Expr::ParamRef { name, elem, .. } => {
                let var_id = self.param_vars[self.param_index[name]];
                let zero = self.constant(&ConstKey::Int { ty: U32, bits: 0 });
                let ptr_ty = self.ensure_type(&T::Ptr {
                    class: StorageClass::StorageBuffer,
                    pointee: Box::new(T::from_scalar(*elem)),
                });
                let result = self.alloc();
                self.push_fn(inst(Op::AccessChain, &[ptr_ty, result, var_id, zero, zero]));
                Ok(result)
            }
            _ => Err("coop source must be a buffer or shared array".to_string()),
        }
    }

    fn gl_ext(&mut self, ext: GlslStd450Op, ty_id: u32, result: u32, args: &[u32]) {
        let set = self.import.expect("glsl ext import missing");
        let mut operands = vec![ty_id, result, set, ext as u32];
        operands.extend(args);
        self.push_fn(inst(Op::ExtInst, &operands));
    }

    fn expr_type(&self, expr: &Expr) -> T {
        match expr {
            Expr::Builtin { name, .. }
                if matches!(*name, "lane" | "subgroup_id" | "subgroup_size") =>
            {
                U32
            }
            Expr::Builtin { .. } => T::Vec {
                size: 3,
                elem: Box::new(U32),
            },
            Expr::IntLit { ty, .. } | Expr::FloatLit { ty, .. } => T::from_scalar(*ty),
            Expr::BoolLit { .. } => BOOL,
            Expr::LocalRef { ty, .. } => type_of(ty),
            Expr::ScalarRef { ty, .. } => T::from_scalar(*ty),
            Expr::Index { ty, .. } | Expr::Member { ty, .. } => T::from_scalar(*ty),
            Expr::Unary { ty, .. } | Expr::Cond { ty, .. } | Expr::Convert { ty, .. } => {
                T::from_scalar(*ty)
            }
            Expr::Binary { ty, .. } => type_of(ty),
            Expr::Call { ty, .. } => type_of(ty),
            _ => unreachable!("no value type"),
        }
    }
}

fn type_of(ty: &Type) -> T {
    match ty {
        Type::Scalar(scalar) => T::from_scalar(*scalar),
        Type::Matrix { elem, role } => T::CoopMat {
            elem: *elem,
            role: *role,
        },
        Type::Vec { size, elem } => T::Vec {
            size: *size,
            elem: Box::new(T::from_scalar(*elem)),
        },
        _ => unreachable!("not a local type"),
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

fn cmp_op(ty: Scalar, gt: bool, eq: bool) -> Op {
    match ty {
        Scalar::F32 | Scalar::F16 => match (gt, eq) {
            (false, false) => Op::FOrdLessThan,
            (false, true) => Op::FOrdLessThanEqual,
            (true, false) => Op::FOrdGreaterThan,
            (true, true) => Op::FOrdGreaterThanEqual,
        },
        Scalar::I32 => match (gt, eq) {
            (false, false) => Op::SLessThan,
            (false, true) => Op::SLessThanEqual,
            (true, false) => Op::SGreaterThan,
            (true, true) => Op::SGreaterThanEqual,
        },
        _ => match (gt, eq) {
            (false, false) => Op::ULessThan,
            (false, true) => Op::ULessThanEqual,
            (true, false) => Op::UGreaterThan,
            (true, true) => Op::UGreaterThanEqual,
        },
    }
}

fn convert_op(target: &T, source: &T) -> Op {
    match (source, target) {
        (T::Float { width: 32 }, T::Float { width: 16 })
        | (T::Float { width: 16 }, T::Float { width: 32 }) => Op::FConvert,
        (T::Float { .. }, T::Int { signed: true, .. }) => Op::ConvertFToS,
        (T::Float { .. }, T::Int { signed: false, .. }) => Op::ConvertFToU,
        (T::Int { signed: true, .. }, T::Float { .. }) => Op::ConvertSToF,
        (T::Int { signed: false, .. }, T::Float { .. }) => Op::ConvertUToF,
        (
            T::Int {
                width: 32,
                signed: true,
            },
            T::Int {
                width: 32,
                signed: false,
            },
        )
        | (
            T::Int {
                width: 32,
                signed: false,
            },
            T::Int {
                width: 32,
                signed: true,
            },
        ) => Op::Bitcast,
        (T::Int { signed: true, .. }, T::Int { .. }) => Op::SConvert,
        (T::Int { signed: false, .. }, T::Int { .. }) => Op::UConvert,
        _ => unreachable!("no conversion"),
    }
}

fn f32_to_f16(value: f32) -> u32 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7C00 | (mant >> 13)
        }
    } else if exp >= 0x8F {
        sign | 0x7C00
    } else if exp <= 0x70 {
        sign
    } else {
        let mut e = exp - 127 + 15;
        let mut m = mant;
        if e <= 0 {
            m |= 0x80_0000;
            let shift = 14 - e;
            m >>= shift;
            e = 0;
        }
        sign | ((e as u32) << 10) | (m >> 13)
    }
}

fn inst_raw(op: u32, operands: &[u32]) -> Vec<u32> {
    let word_count = 1 + operands.len();
    let mut words = vec![op | ((word_count as u32) << 16)];
    words.extend_from_slice(operands);
    words
}

fn inst(op: Op, operands: &[u32]) -> Vec<u32> {
    let word_count = 1 + operands.len();
    let mut words = vec![(op as u32) | ((word_count as u32) << 16)];
    words.extend_from_slice(operands);
    words
}

fn string_words(s: &str) -> Vec<u32> {
    let mut words = Vec::new();
    let mut current = 0u32;
    let mut shift = 0;
    for byte in s.bytes().chain(std::iter::once(0)) {
        current |= (byte as u32) << shift;
        shift += 8;
        if shift == 32 {
            words.push(current);
            current = 0;
            shift = 0;
        }
    }
    if shift != 0 {
        words.push(current);
    }
    words
}

fn extension_words(name: &str) -> Vec<u32> {
    let mut words = vec![(Op::Extension as u32) | (2u32 << 16)];
    words.extend(string_words(name));
    let count = words.len() as u32;
    words[0] = (Op::Extension as u32) | (count << 16);
    words
}

fn ext_inst_import_words(id: u32) -> Vec<u32> {
    let mut words = vec![0, id];
    words.extend(string_words("GLSL.std.450"));
    let count = words.len() as u32;
    words[0] = (Op::ExtInstImport as u32) | (count << 16);
    words
}

fn collect_locals(stmts: &[Stmt], out: &mut Vec<(u32, Type)>) {
    for stmt in stmts {
        match stmt {
            Stmt::Let { id, ty, .. } | Stmt::Var { id, ty, .. } => out.push((*id, *ty)),
            Stmt::For { id, body, .. } => {
                out.push((*id, Type::Scalar(Scalar::U32)));
                collect_locals(body, out);
            }
            Stmt::If { then, els, .. } => {
                collect_locals(then, out);
                collect_locals(els, out);
            }
            Stmt::Loop { body, .. } => collect_locals(body, out),
            _ => {}
        }
    }
}

fn collect_stmt(collect: &mut Collect, stmt: &Stmt) {
    match stmt {
        Stmt::Let { ty, init, .. } | Stmt::Var { ty, init, .. } => {
            collect.types.insert(type_of(ty));
            if let Type::Matrix { elem, .. } = ty {
                if *elem == Scalar::F16 {
                    collect.has_f16 = true;
                }
                collect.has_coop = true;
            }
            collect_expr(collect, init);
        }
        Stmt::Assign { target, value, .. } => {
            collect_expr(collect, target);
            collect_expr(collect, value);
        }
        Stmt::If {
            cond, then, els, ..
        } => {
            collect_expr(collect, cond);
            for stmt in then {
                collect_stmt(collect, stmt);
            }
            for stmt in els {
                collect_stmt(collect, stmt);
            }
        }
        Stmt::Loop { body, .. } => {
            for stmt in body {
                collect_stmt(collect, stmt);
            }
        }
        Stmt::For {
            start, end, body, ..
        } => {
            collect_expr(collect, start);
            collect_expr(collect, end);
            for stmt in body {
                collect_stmt(collect, stmt);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Barrier { .. } => {}
        Stmt::ExprStmt { expr, .. } => collect_expr(collect, expr),
    }
}

fn collect_expr(collect: &mut Collect, expr: &Expr) {
    match expr {
        Expr::IntLit { ty, .. } => {
            collect.types.insert(T::from_scalar(*ty));
            if *ty == Scalar::Bf16 {
                collect.has_int16 = true;
            }
            if matches!(*ty, Scalar::I8 | Scalar::U8) {
                collect.has_int8 = true;
            }
        }
        Expr::FloatLit { ty, .. } => {
            collect.types.insert(T::from_scalar(*ty));
            if *ty == Scalar::F16 {
                collect.has_f16 = true;
            }
        }
        Expr::BoolLit { .. } => {
            collect.types.insert(BOOL);
        }
        Expr::LocalRef { ty, .. } => {
            collect.types.insert(type_of(ty));
            if let Type::Scalar(scalar) = ty {
                match scalar {
                    Scalar::F16 => collect.has_f16 = true,
                    Scalar::Bf16 => collect.has_int16 = true,
                    Scalar::I8 | Scalar::U8 => collect.has_int8 = true,
                    _ => {}
                }
            }
        }
        Expr::ParamRef { .. } => {}
        Expr::ScalarRef { ty, .. } => {
            collect.types.insert(T::from_scalar(*ty));
            match *ty {
                Scalar::F16 => collect.has_f16 = true,
                Scalar::Bf16 => collect.has_int16 = true,
                Scalar::I8 | Scalar::U8 => collect.has_int8 = true,
                _ => {}
            }
        }
        Expr::SharedRef { elem, len, .. } => {
            collect.types.insert(T::Array {
                elem: Box::new(T::from_scalar(*elem)),
                len: *len,
            });
        }
        Expr::Builtin { name, .. } => {
            if matches!(*name, "lane" | "subgroup_id" | "subgroup_size") {
                collect.types.insert(U32);
                collect.builtins.insert(match *name {
                    "lane" => BuiltinVar::Lane,
                    "subgroup_id" => BuiltinVar::SubgroupId,
                    _ => BuiltinVar::SubgroupSize,
                });
            } else {
                collect.types.insert(T::Vec {
                    size: 3,
                    elem: Box::new(U32),
                });
                collect.builtins.insert(match *name {
                    "gid" => BuiltinVar::Gid,
                    "thread" => BuiltinVar::Thread,
                    "block" => BuiltinVar::Block,
                    "block_dim" => BuiltinVar::BlockDim,
                    _ => unreachable!(),
                });
            }
        }
        Expr::Index {
            base, index, ty, ..
        } => {
            collect_expr(collect, base);
            collect_expr(collect, index);
            collect.types.insert(T::from_scalar(*ty));
        }
        Expr::Member { base, ty, .. } => {
            collect_expr(collect, base);
            collect.types.insert(T::from_scalar(*ty));
        }
        Expr::Unary { expr, ty, .. } => {
            collect_expr(collect, expr);
            collect.types.insert(T::from_scalar(*ty));
        }
        Expr::Binary { lhs, rhs, ty, .. } => {
            collect_expr(collect, lhs);
            collect_expr(collect, rhs);
            collect.types.insert(type_of(ty));
            if scalar_of(ty) == Scalar::F16 {
                collect.has_f16 = true;
            }
        }
        Expr::Cond {
            cond,
            then,
            els,
            ty,
            ..
        } => {
            collect_expr(collect, cond);
            collect_expr(collect, then);
            collect_expr(collect, els);
            collect.types.insert(T::from_scalar(*ty));
        }
        Expr::Convert { ty, expr, .. } => {
            collect_expr(collect, expr);
            collect.types.insert(T::from_scalar(*ty));
            match *ty {
                Scalar::F16 => collect.has_f16 = true,
                Scalar::Bf16 => collect.has_int16 = true,
                Scalar::I8 | Scalar::U8 => collect.has_int8 = true,
                _ => {}
            }
        }
        Expr::Call { name, args, ty, .. } => {
            for arg in args {
                collect_expr(collect, arg);
            }
            collect.types.insert(type_of(ty));
            if scalar_of(ty) == Scalar::F16 {
                collect.has_f16 = true;
            }
            if matches!(
                *name,
                "min"
                    | "max"
                    | "clamp"
                    | "abs"
                    | "floor"
                    | "ceil"
                    | "round"
                    | "trunc"
                    | "sign"
                    | "fract"
                    | "sqrt"
                    | "rsqrt"
                    | "exp"
                    | "exp2"
                    | "log"
                    | "log2"
                    | "tanh"
                    | "pow"
                    | "fma"
                    | "clz"
                    | "ctz"
            ) {
                collect.has_glsl_ext = true;
            }
            if matches!(
                *name,
                "coop_zero" | "coop_load_a" | "coop_load_b" | "coop_mul_add" | "coop_store"
            ) {
                collect.has_coop = true;
            }
            if matches!(
                *name,
                "subgroup_broadcast"
                    | "subgroup_shuffle"
                    | "subgroup_shuffle_down"
                    | "subgroup_shuffle_up"
                    | "subgroup_reduce_add"
                    | "subgroup_reduce_max"
                    | "subgroup_reduce_min"
                    | "subgroup_inclusive_add"
                    | "subgroup_all"
                    | "subgroup_any"
            ) {
                let cap = match *name {
                    "subgroup_broadcast" => Capability::GroupNonUniformBallot as u32,
                    "subgroup_shuffle" => Capability::GroupNonUniformShuffle as u32,
                    "subgroup_shuffle_down" | "subgroup_shuffle_up" => {
                        Capability::GroupNonUniformShuffleRelative as u32
                    }
                    "subgroup_reduce_add"
                    | "subgroup_reduce_max"
                    | "subgroup_reduce_min"
                    | "subgroup_inclusive_add" => Capability::GroupNonUniformArithmetic as u32,
                    "subgroup_all" | "subgroup_any" => Capability::GroupNonUniformVote as u32,
                    _ => unreachable!(),
                };
                collect.subgroup_caps.insert(cap);
            }
        }
    }
}
