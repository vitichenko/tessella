use tessella_core::error::DomainError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Storage generic error: {0}")]
    Other(String),
}

impl From<StorageError> for DomainError {
    fn from(err: StorageError) -> Self {
        DomainError::Storage(err.to_string())
    }
}
