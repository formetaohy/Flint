use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("gpu: {0}")]
    Gpu(String),
    #[error("model: {0}")]
    Model(String),
    #[error("config: {0}")]
    Config(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Config(e.to_string())
    }
}
