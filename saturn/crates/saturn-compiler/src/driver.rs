use crate::ast;
use crate::diag::{Diagnostic, Result as StageResult, Source};
use crate::ir::Kernel;
use crate::opt::PassManager;
use crate::{lexer, parser, sema};

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
        let tokens = lexer::lex(source).map_err(|e| vec![e])?;
        let program = parser::parse(&tokens).map_err(|e| vec![e])?;
        let mut kernel = sema::check(&program).map_err(|e| vec![e])?;
        self.passes.run(&mut kernel);
        Ok(kernel)
    }

    pub fn lex(source: &Source) -> StageResult<Vec<lexer::Token>> {
        lexer::lex(source)
    }

    pub fn parse(tokens: &[lexer::Token]) -> StageResult<ast::Kernel> {
        parser::parse(tokens)
    }

    pub fn check(program: &ast::Kernel) -> StageResult<Kernel> {
        sema::check(program)
    }

    pub fn optimize(&self, kernel: &mut Kernel) {
        self.passes.run(kernel);
    }
}

impl Default for Driver {
    fn default() -> Self {
        Self::new()
    }
}
