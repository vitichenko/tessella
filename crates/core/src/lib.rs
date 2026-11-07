#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::use_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::used_underscore_binding)]

pub mod assignment;
pub mod callback;
pub mod chat;
pub mod config;
pub mod constants;
pub mod domain;
pub mod error;
pub mod generation;
pub mod net;
pub mod network;
pub mod routing;
pub mod sampling;
pub mod shutdown;
pub mod telemetry;
pub mod time;
pub mod traits;
pub mod vram;

pub mod models {
    pub use crate::chat::*;
    pub use crate::config::QuantizationConfig;
    pub use crate::domain::*;
    pub use crate::network::*;
}

pub use telemetry::init_tracing;

pub use assignment::*;
pub use chat::*;
pub use config::{NetworkConfig, QuantizationConfig, ReputationConfig};
pub use domain::*;
pub use error::DomainError;
pub use generation::*;
pub use network::*;
pub use routing::{
    DEFAULT_PIN_TTL, EXPLORATION_SHARE, MAX_PINNED_SESSIONS, NO_EXPLORATION, PinnedRoute,
    RouteSelection, SessionPinRegistry, UNMEASURED_LATENCY, select_route_with,
};
pub use sampling::*;
pub use traits::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedConfig {
    pub musivums: Vec<String>,
}

impl SeedConfig {
    pub fn load() -> Result<Self, DomainError> {
        // In production load from seed_musivums.json
        let musivums = vec![]; // Removed hardcoded localhost

        if musivums.is_empty() {
            return Err(DomainError::Config(
                "tracker_url is required but not set in config. Add tracker_url = \"http://<host>:<port>\" to your config file.".to_string()
            ));
        }

        Ok(Self { musivums })
    }

    pub fn get_primary_url(&self) -> String {
        self.musivums.first().cloned().unwrap_or_default()
    }
}
