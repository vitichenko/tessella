use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tessella_core::models::LayerPeer;
use tracing::info;

/// The on-disk shape of the cache.
///
/// It stores [`LayerPeer`] (`TASK-58`/S64): the same record the routing table uses and the one
/// `peer_discovery::layer_peer_from_node_info` produces, so `/peers`, the gossip topology and this
/// file cannot disagree about what a peer looks like. A file written by an older build carried the
/// retired `WorkerInfo` shape (`model_id` extra, same fields); it no longer deserialises, which is
/// harmless — this is a cache, and its TTL is 300 s.
#[derive(Debug, Serialize, Deserialize)]
struct CacheData {
    peers: Vec<LayerPeer>,
    timestamp: u64, // unix timestamp
}

#[derive(Debug)]
pub struct TopologyCache {
    peers: Vec<LayerPeer>,
    last_updated: Instant,
    ttl: Duration,
    cache_file: Option<PathBuf>,
}

impl TopologyCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            peers: Vec::new(),
            last_updated: Instant::now() - ttl * 2, // Start expired
            ttl,
            cache_file: None,
        }
    }

    pub fn with_persistence(mut self, path: PathBuf) -> Self {
        self.cache_file = Some(path);
        let _ = self.load_from_disk();
        self
    }

    pub fn update(&mut self, peers: Vec<LayerPeer>) {
        self.peers = peers;
        self.last_updated = Instant::now();
        let _ = self.save_to_disk();
    }

    pub fn get(&self) -> Option<Vec<LayerPeer>> {
        if self.peers.is_empty() {
            None
        } else {
            Some(self.peers.clone())
        }
    }

    pub fn is_stale(&self) -> bool {
        self.last_updated.elapsed() > self.ttl
    }

    pub fn age_secs(&self) -> u64 {
        self.last_updated.elapsed().as_secs()
    }

    fn save_to_disk(&self) -> anyhow::Result<()> {
        if let Some(ref path) = self.cache_file {
            let data = CacheData {
                peers: self.peers.clone(),
                timestamp: tessella_core::time::unix_now_secs(),
            };
            let json = serde_json::to_string_pretty(&data)?;
            std::fs::write(path, json)?;
        }
        Ok(())
    }

    fn load_from_disk(&mut self) -> anyhow::Result<()> {
        if let Some(ref path) = self.cache_file {
            if path.exists() {
                let content = std::fs::read_to_string(path)?;
                let data: CacheData = serde_json::from_str(&content)?;

                // Adjust last_updated based on file timestamp
                let now = tessella_core::time::unix_now_secs();
                let elapsed = now.saturating_sub(data.timestamp);

                self.peers = data.peers;
                self.last_updated = Instant::now() - Duration::from_secs(elapsed);
                info!("📂 Loaded topology cache from disk (age: {}s)", elapsed);
            }
        }
        Ok(())
    }
}
