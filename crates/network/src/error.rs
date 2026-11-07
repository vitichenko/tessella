use tessella_core::error::DomainError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Libp2p dial error: {0}")]
    Dial(String),

    #[error("Libp2p transport error: {0}")]
    Transport(String),

    #[error("Swarm command error: {0}")]
    Swarm(String),

    #[error("Cryptographic / signature error: {0}")]
    Crypto(String),

    #[error("Network generic error: {0}")]
    Other(String),
}

impl From<NetworkError> for DomainError {
    fn from(err: NetworkError) -> Self {
        DomainError::Network(err.to_string())
    }
}
