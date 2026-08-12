use std::collections::HashMap;
use std::path::Path;

use crate::ast;
use crate::diag::{Diagnostic, Result as StageResult, Source};
use crate::ir::Kernel;
use crate::opt::PassManager;
use crate::{expand, lexer, parser, sema, uniformity};

pub type ImportResolver = dyn Fn(&str, &Path) -> Result<String, String>;

pub struct Driver {
    passes: PassManager,
    resolve: Option<Box<ImportResolver>>,
}

impl Driver {
    pub fn new() -> Self {
        Self {
            passes: PassManager::default(),
            resolve: None,
        }
    }

    pub fn with_passes(passes: PassManager) -> Self {
        Self {
            passes,
            resolve: None,
        }
    }

    pub fn with_import_resolver(mut self, resolve: Box<ImportResolver>) -> Self {
        self.resolve = Some(resolve);
        self
    }

    pub fn compile(&self, source: &Source) -> std::result::Result<Kernel, Vec<Diagnostic>> {
        self.compile_with_specs(source, &[])
    }

    pub fn compile_with_specs(
        &self,
        source: &Source,
        specs: &[(&str, f64)],
    ) -> std::result::Result<Kernel, Vec<Diagnostic>> {
        let mut modules = ModuleSet::new(self.resolve.as_deref());
        modules.load_entry(source)?;
        let (fns, structs) = modules.merge()?;
        let kernel = expand::expand(&ast::Program {
            imports: Vec::new(),
            structs,
            fns,
            kernel: modules.entry.and_then(|e| e.kernel),
        })
        .map_err(|e| vec![e])?;
        let mut kernel = sema::check(&kernel, specs).map_err(|e| vec![e])?;
        uniformity::check(&kernel).map_err(|e| vec![e])?;
        self.passes.run(&mut kernel);
        Ok(kernel)
    }

    pub fn lex(source: &Source) -> StageResult<Vec<lexer::Token>> {
        lexer::lex(source)
    }

    pub fn parse(tokens: &[lexer::Token]) -> std::result::Result<ast::Program, Vec<Diagnostic>> {
        parser::parse(tokens)
    }

    pub fn check(program: &ast::Program) -> StageResult<Kernel> {
        let kernel = expand::expand(program)?;
        let kernel = sema::check(&kernel, &[])?;
        uniformity::check(&kernel)?;
        Ok(kernel)
    }

    pub fn optimize(&mut self, kernel: &mut Kernel) {
        self.passes.run(kernel);
    }
}

impl Default for Driver {
    fn default() -> Self {
        Self::new()
    }
}

struct ModuleSet<'a> {
    resolve: Option<&'a ImportResolver>,
    loaded: HashMap<String, ast::Program>,
    order: Vec<String>,
    stack: Vec<String>,
    entry: Option<ast::Program>,
}

impl<'a> ModuleSet<'a> {
    fn new(resolve: Option<&'a ImportResolver>) -> Self {
        Self {
            resolve,
            loaded: HashMap::new(),
            order: Vec::new(),
            stack: Vec::new(),
            entry: None,
        }
    }

    fn load_entry(&mut self, source: &Source) -> std::result::Result<(), Vec<Diagnostic>> {
        let program = self.parse_source(source)?;
        if program.kernel.is_none() {
            return Err(vec![Diagnostic::new(
                crate::diag::Span::dummy(),
                format!("entry file {} must contain a kernel", source.name()),
            )]);
        }
        self.entry = Some(program.clone());
        self.load_imports(&program, source.name())
    }

    fn parse_source(&self, source: &Source) -> std::result::Result<ast::Program, Vec<Diagnostic>> {
        let tokens = lexer::lex(source).map_err(|e| vec![e])?;
        parser::parse(&tokens)
    }

    fn load_imports(
        &mut self,
        program: &ast::Program,
        current: &str,
    ) -> std::result::Result<(), Vec<Diagnostic>> {
        for (name, span) in &program.imports {
            if self.stack.iter().any(|s| s == name) {
                let mut chain = self.stack.clone();
                chain.push(name.clone());
                chain.push(current.to_string());
                return Err(vec![Diagnostic::new(
                    *span,
                    format!("circular import: {}", chain.join(" -> ")),
                )]);
            }
            if self.loaded.contains_key(name) {
                continue;
            }
            let resolver = self.resolve.ok_or_else(|| {
                vec![Diagnostic::new(
                    *span,
                    format!("import '{name}' requires a file resolver"),
                )]
            })?;
            let current_path = Path::new(current);
            let text = resolver(name, current_path).map_err(|msg| {
                vec![Diagnostic::new(*span, format!("cannot import '{name}': {msg}"))]
            })?;
            let imported = Source::new(
                current_path
                    .parent()
                    .map(|dir| dir.join(name).display().to_string())
                    .unwrap_or_else(|| name.clone()),
                text,
            );
            let imported_program = self.parse_source(&imported).map_err(|e| {
                let msg = e
                    .first()
                    .map(|d| d.msg.clone())
                    .unwrap_or_else(|| "parse error".to_string());
                vec![Diagnostic::new(*span, format!("error in import '{name}': {msg}"))]
            })?;
            if imported_program.kernel.is_some() {
                return Err(vec![Diagnostic::new(
                    *span,
                    format!("import '{name}' must not contain a kernel"),
                )]);
            }
            self.stack.push(name.clone());
            self.load_imports(&imported_program, imported.name())?;
            self.stack.pop();
            self.loaded.insert(name.clone(), imported_program);
            self.order.push(name.clone());
        }
        Ok(())
    }

    fn merge(&mut self) -> std::result::Result<(Vec<ast::FnDecl>, Vec<ast::StructDecl>), Vec<Diagnostic>> {
        let mut fns = Vec::new();
        let mut structs = Vec::new();
        let mut fn_names = std::collections::HashSet::new();
        let mut struct_names = std::collections::HashSet::new();
        if let Some(entry) = &self.entry {
            for f in &entry.fns {
                if !fn_names.insert(f.name.clone()) {
                    return Err(vec![Diagnostic::new(
                        f.span,
                        format!("duplicate function '{}'", f.name),
                    )]);
                }
                fns.push(f.clone());
            }
            for s in &entry.structs {
                if !struct_names.insert(s.name.clone()) {
                    return Err(vec![Diagnostic::new(
                        s.span,
                        format!("duplicate struct '{}' across imports", s.name),
                    )]);
                }
                structs.push(s.clone());
            }
        }
        for name in &self.order {
            let program = &self.loaded[name];
            for f in &program.fns {
                if !fn_names.insert(f.name.clone()) {
                    return Err(vec![Diagnostic::new(
                        f.span,
                        format!("duplicate function '{}' across imports", f.name),
                    )]);
                }
                fns.push(f.clone());
            }
            for s in &program.structs {
                if !struct_names.insert(s.name.clone()) {
                    return Err(vec![Diagnostic::new(
                        s.span,
                        format!("duplicate struct '{}' across imports", s.name),
                    )]);
                }
                structs.push(s.clone());
            }
        }
        Ok((fns, structs))
    }
}
