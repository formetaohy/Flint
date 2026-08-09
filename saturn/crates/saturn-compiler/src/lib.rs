pub mod ast;
pub mod consts;
pub mod diag;
pub mod driver;
pub mod expand;
pub mod ir;
pub mod lexer;
pub mod opt;
pub mod parser;
pub mod sema;
pub mod uniformity;

pub use diag::{Diagnostic, Source, Span};
pub use driver::Driver;
pub use ir::Kernel;

pub fn compile(source: &Source) -> std::result::Result<Kernel, Vec<Diagnostic>> {
    Driver::new().compile(source)
}
