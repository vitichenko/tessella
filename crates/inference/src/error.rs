use tessella_core::error::DomainError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Inference generic error: {0}")]
    Other(String),
}

impl From<InferenceError> for DomainError {
    fn from(err: InferenceError) -> Self {
        DomainError::Inference(err.to_string())
    }
}
