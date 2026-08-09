pub mod ast;
pub mod diag;
pub mod driver;
pub mod ir;
pub mod lexer;
pub mod opt;
pub mod parser;
pub mod sema;

pub use diag::{Diagnostic, Source, Span};
pub use driver::Driver;
pub use ir::Kernel;

pub fn compile(source: &Source) -> std::result::Result<Kernel, Vec<Diagnostic>> {
    Driver::new().compile(source)
}
