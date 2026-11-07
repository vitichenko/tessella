//! Token stream routing and callback channel collaborator for NetworkManager (`SOLID-90`).

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Shared thread-safe map associating in-flight request IDs with token-stream channels.
pub type TokenStreamSenderMap =
    Arc<DashMap<String, mpsc::Sender<tessella_protocol::TokenStreamPayload>>>;

/// Creates an empty token-stream map.
pub fn new_token_stream_map() -> TokenStreamSenderMap {
    Arc::new(DashMap::new())
}
