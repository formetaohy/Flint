use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("gpu: {0}")]
    Gpu(String),
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    #[error("model: {0}")]
    Model(String),
    #[error("config: {0}")]
    Config(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("profiler: {0}")]
    Profiler(String),
}
