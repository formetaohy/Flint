use crate::ast;
use crate::diag::{Diagnostic, Result as StageResult, Source};
use crate::ir::Kernel;
use crate::opt::PassManager;
use crate::{expand, lexer, parser, sema, uniformity};

pub struct Driver {
    passes: PassManager,
}

impl Driver {
    pub fn new() -> Self {
        Self {
            passes: PassManager::default(),
        }
    }

    pub fn with_passes(passes: PassManager) -> Self {
        Self { passes }
    }

    pub fn compile(&self, source: &Source) -> std::result::Result<Kernel, Vec<Diagnostic>> {
        self.compile_with_specs(source, &[])
    }

    pub fn compile_with_specs(
        &self,
        source: &Source,
        specs: &[(&str, f64)],
    ) -> std::result::Result<Kernel, Vec<Diagnostic>> {
        let tokens = lexer::lex(source).map_err(|e| vec![e])?;
        let program = parser::parse(&tokens)?;
        let kernel = expand::expand(&program).map_err(|e| vec![e])?;
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
        sema::check(&kernel, &[])
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
