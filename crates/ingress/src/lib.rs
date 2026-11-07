//! The HTTP ingress **port** of the data plane: what the API server is served *for*, without the
//! HTTP crate (`SOLID-89`).
//!
//! Before this crate existed, `tessella-worker` declared `tessella-api` as a dependency only to call
//! `start_api_server` and to name `ApiState`: the data-plane daemon linked Axum and the whole HTTP
//! crate, and it could not be built (nor tested) without them. The dependency was inverted here:
//!
//! - the **state** the surface publishes ([`ApiState`]) and the **port** that serves it
//!   ([`ApiHost`]) live in this crate, which depends on no HTTP framework;
//! - `tessella-api` implements the port with its Axum router and re-exports the state, so the
//!   handlers did not have to change;
//! - the daemon receives an `Arc<dyn ApiHost>` from its composition root (`tessella-node`), builds
//!   the state from its own internals and calls [`ApiHost::serve`] from the same `select!` that used
//!   to call `start_api_server`.
//!
//! The state is **process data**, not a wire contract: every field is either a trait object the
//! daemon already owns (`DistributedInference`, `StorageService`, the GPU health of its engine) or a
//! handle to state it shares with it (the token-stream channels, the capability registry). That is
//! why it does not belong with the wire DTOs of `tessella-protocol`.

#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc};

use tessella_core::callback::capability::CallbackCapabilities;
use tessella_core::traits::{
    CreditService, DiscoveryService, DistributedInference, PeerSlashingService, SlashingService,
    StorageService,
};
use tessella_protocol::{
    ModelConfig, RebalancePlan, TokenStreamPayload, WorkerAssignment, WorkerRegistration,
    WorkerStatus,
};

/// A worker as the orchestrator knows it (the `Musivum` role of the surface).
#[derive(Clone)]
pub struct RegisteredWorker {
    pub registration: WorkerRegistration,
    pub assignment: WorkerAssignment,
    pub status: WorkerStatus,
    pub registered_at: Instant,
    pub queue_depth: usize,
}

/// The registry the `Musivum` role keeps in memory. The daemon always builds it empty (its role is
/// the worker one), which is why the handlers that read it publish `null` instead of a model.
pub struct MusivumState {
    pub workers: Vec<RegisteredWorker>,
    pub blacklist: std::collections::HashSet<String>,
    pub rebalance_version: u64,
    pub current_plan: Option<RebalancePlan>,
    pub auto_rebalance: bool,
}

impl MusivumState {
    pub fn get_total_layers_for(
        &self,
        model_id: &str,
        _fallback_config: Option<&ModelConfig>,
    ) -> usize {
        if let Some(w) = self
            .workers
            .iter()
            .find(|w| w.assignment.model_repo == model_id)
            && w.registration.total_layers > 0
        {
            return w.registration.total_layers;
        }
        // Fallback to core logic
        tessella_protocol::get_total_layers(model_id).unwrap_or(32)
    }
}

/// GLOBAL API STATE (Worker Role)
#[derive(Clone)]
pub struct ApiState {
    pub dist_inference: Option<Arc<dyn DistributedInference>>,
    pub storage: Option<Arc<dyn StorageService + Send + Sync>>,
    pub credits: Option<Arc<dyn CreditService + Send + Sync>>,
    pub discovery: Option<Arc<dyn DiscoveryService + Send + Sync>>,
    pub slashing: Option<Arc<dyn SlashingService + Send + Sync>>,
    pub peer_slashing: Option<Arc<dyn PeerSlashingService>>,
    pub musivum_state: Option<Arc<Mutex<MusivumState>>>,
    /// `[worker] max_new_tokens`: the answer's upper bound when the client omits `max_tokens`
    /// (`TASK-40`/S46). It used to be a field nobody read — the handlers hardcoded 512 — and the
    /// only knob of the same name was dead in `worker.toml`.
    pub max_new_tokens: usize,
    /// How many tokens of hidden state this node's pipeline can carry in **one hop** (`TASK-47`/S53):
    /// the **minimum of the two limits that bind it** (`TASK-76`/S83) — what the wire can carry
    /// (`MAX_HIDDEN_STATE_BYTES / (hidden_size × bytes per activation element)`) and what this node's
    /// VRAM declares it can prefill (`max_input_tokens` of `worker.toml`). It is published by
    /// `GET /status` and it is what `Session::forward` and the receiving hop enforce before paying for
    /// a prefill that would not fit. `None` when neither limit is known (the model config is not
    /// readable and nothing is declared), in which case the check is skipped.
    pub max_input_tokens: Option<usize>,
    /// **Which** of the two limits produced [`ApiState::max_input_tokens`]
    /// ([`tessella_core::constants::inference::HopCeilingSource::as_str`], `TASK-76`/S83). `None`
    /// exactly when the number is `None`: a ceiling always has a reason, and the operator's next move
    /// depends on it (shard the model differently, or lower the prompt).
    pub max_input_tokens_source: Option<&'static str>,
    /// How long **this node** waits for a hop of its own pipeline, in seconds (`TASK-76`/S83).
    ///
    /// It is `p2p_inference_timeout_secs` of `worker.toml`, resolved once at boot
    /// (`WorkerConfig::hop_wait`) and it is the **only** timer a forward has: the router used to wrap
    /// each request in a hard-coded 5 s, which overrode both this key (which had no reader at all) and
    /// the 30 s of its own HTTP client. A prompt of the published ceiling needs more than those 5 s to
    /// be applied — measured in `S83`: 2 037 tokens —, so the literal declared a working hop dead and
    /// the client ended in a `500 … Pipeline timeout`.
    ///
    /// `GET /status` publishes it next to the ceiling because the two numbers are the same deal: the
    /// ceiling says *how much* this node accepts and this one says *until when* it will wait for it.
    /// It is not optional: a node always has a wait (there is a default), so a `null` here would mean
    /// the operator is reading a struct that was never built by the boot.
    pub hop_timeout_secs: u64,
    pub musivum_keypair: Option<tessella_network::NetworkKeypair>,
    pub admin_token: Option<String>,
    pub default_model: String,
    pub model_config: Option<ModelConfig>,
    /// `S52`/`TASK-46`: a `kv_cache` field used to be here, fed with the worker's shared
    /// `KvCacheStore` — which was never written to. It was retired with the store; the KV cache that
    /// exists is per session inside the backend and nothing in the HTTP layer reads it.
    /// `TASK-50`/S56: what this node has seen fail towards its peers, published by `GET /status` as
    /// `peer_failures`. It replaces a `CircuitBreaker` that the HTTP layer never read; the count is
    /// the local half of the answer to a hop that does not answer (there is no third-party report
    /// channel: `/report_failure` authenticates a single identity, D-3). `None` (never in production,
    /// where the worker always builds it) publishes `null`.
    pub peer_failures: Option<Arc<tessella_core::telemetry::metrics::PeerFailureRegistry>>,
    /// `TASK-77`/S84: la salud **de este nodo**, no la de sus pares — de ahí que no viva en
    /// `peer_failures`. `GET /status` la publica como `gpu`: si el nodo sirve, con qué veredicto juzgó
    /// el último fallo (`out_of_memory` = juicio sobre la petición, el nodo sigue; `device_lost` =
    /// juicio sobre el nodo, que sólo se declara caído si su device deja de contestar), desde cuándo y
    /// cada cuánto se revalida. Es de sólo lectura para el HTTP —el estado lo mueve el engine—, y
    /// `None` (nunca en producción, donde el worker siempre la construye) publica `null`.
    pub gpu_health: Option<Arc<tessella_core::telemetry::gpu_health::GpuHealth>>,
    pub tokenizer: Option<Arc<dyn tessella_core::traits::TokenizerService>>,
    pub token_stream_senders:
        Option<Arc<dashmap::DashMap<String, mpsc::Sender<TokenStreamPayload>>>>,
    pub orchestrator_url: Option<String>,
    pub own_api_url: Option<String>,
    /// `(session_id, nonce)` pairs already seen by this worker's HTTP surface (H-09 ·
    /// `TASK-05`). The engine keeps its own cache because it also authenticates the payloads it
    /// is handed by the P2P path; each of the two components counts a frame exactly once.
    pub replay_cache: Arc<tessella_protocol::signing::ReplayCache>,
    /// Capabilities accepted by `/v1/token_stream/{request_id}` (H-10 · `TASK-06` mitigation 2):
    /// the entry node registers the signature of **every** payload it seals here, and a callback
    /// must present one of them for the stream it targets. It is the same registry the engine
    /// fills while sealing, so a callback can only come from a node that saw this pipeline's
    /// traffic. `None` (never in production, where the worker always builds it) fails closed.
    pub callback_capabilities: Option<Arc<CallbackCapabilities>>,
    /// Telemetry and Prometheus metrics for this worker node (`TASK-94`).
    pub metrics: Option<Arc<tessella_core::telemetry::WorkerMetrics>>,
    /// Monotonic version of the active worker shard assignment (`MSV-18`).
    /// Rejects stale rebalance notifications (`version < current_assignment_version`).
    pub current_assignment_version: Arc<AtomicU64>,
}

impl ApiState {
    /// Stops accepting callbacks for `request_id`: drops the stream channel **and** the entry of
    /// the capability registry, so a capability of a finished stream cannot be replayed (H-10).
    pub fn forget_stream(&self, request_id: &str) {
        if let Some(senders) = &self.token_stream_senders {
            senders.remove(request_id);
        }
        if let Some(capabilities) = &self.callback_capabilities {
            capabilities.remove(request_id);
        }
    }
}

/// The **port** the daemon asks its composition root for: an HTTP surface it can start without
/// naming a framework (`SOLID-89`).
///
/// It replaces the direct call to `tessella_api::start_api_server`, and it keeps its contract: like
/// the function it wraps, an implementation **never returns on success** — it serves until the
/// process stops — so its `Ok` is not a value the caller acts on, only its `Err` (bind failure,
/// server death) is, which the daemon logs as `API Server halted` and then shuts down.
#[async_trait]
pub trait ApiHost: Send + Sync + 'static {
    /// Serves `state` on `addr` until the server stops.
    async fn serve(&self, addr: SocketAddr, state: ApiState) -> anyhow::Result<()>;
}
