//! `tessella-api` as an implementation of the data plane's ingress port (`SOLID-89`).
//!
//! The daemon does not name this crate any more: its composition root (`tessella-node`) builds this
//! value and hands it over as an `Arc<dyn ApiHost>`. Everything else in the crate stays where it was —
//! the handlers, the router and `start_api_server`, which this adapter is a one-line pass-through to.

use std::net::SocketAddr;

use async_trait::async_trait;
use tessella_ingress::{ApiHost, ApiState};

/// The Axum router of this crate, seen by the daemon through [`ApiHost`].
#[derive(Debug, Default)]
pub struct AxumApiHost;

impl AxumApiHost {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ApiHost for AxumApiHost {
    /// Serves until the server stops, exactly like [`crate::start_api_server`].
    async fn serve(&self, addr: SocketAddr, state: ApiState) -> anyhow::Result<()> {
        crate::start_api_server(addr, state).await
    }
}
