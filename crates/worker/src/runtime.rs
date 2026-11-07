use crate::config::WorkerConfig;
use std::sync::Arc;

pub struct WorkerRuntime {
    pub config: WorkerConfig,
    pub peer_id: String,
    pub orchestrator_url: String,
    pub bind_ip: String,
    pub p2p_addr: String,
    pub http_client: reqwest::Client,
    /// What this node knows about hops that failed (`TASK-50`/S56).
    ///
    /// It replaces the `tessella_network::CircuitBreaker`, which was **write-only** (nothing in the
    /// workspace read `is_allowed`/`state_of`/`get_all_infos`): the datum survives, the state machine
    /// that governed nothing does not, and `GET /status` publishes the count.
    pub peer_failures: Arc<tessella_core::telemetry::metrics::PeerFailureRegistry>,
    /// The worker's persistent libp2p key, shared with the swarm.
    ///
    /// Since D-3 (`TASK-03` Step B) the destructive HTTP requests (`/announce`, `/ready`,
    /// `/deregister`) are signed with it, and the orchestrator only accepts a signature whose
    /// key owns the `peer_id` in the body, so it must be reachable from every call site (heartbeat
    /// tick, rebalance, graceful shutdown) and not only from the `Swarm`.
    /// The wait this node declares for a hop of its own pipeline (`TASK-76`/S83).
    ///
    /// It is `p2p_inference_timeout_secs` of `worker.toml`, validated **once** at boot
    /// ([`WorkerConfig::hop_wait`]) instead of being read at each forwarding site: a key that only
    /// exists as a `u64` in the struct is how it ended up with zero consumers while the router used a
    /// 5 s literal. Every `HttpPipelineRouter` of this process is built with this same `Duration`.
    pub hop_wait: std::time::Duration,
    pub keypair: Arc<tessella_network::NetworkKeypair>,
    /// Telemetry and Prometheus metrics for this worker node (`TASK-94`).
    pub metrics: Arc<tessella_core::telemetry::WorkerMetrics>,
}
