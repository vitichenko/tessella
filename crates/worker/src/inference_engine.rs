use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use candle_core::Device;
use candle_core::Tensor;
use tessella_core::constants::inference::{hop_ceiling, max_input_tokens_per_hop, HopCeiling};
use tessella_core::models::{
    InferenceMode, KarmaTicket, ModelConfig, QueuePriority, WorkTicket, WorkerAssignment,
};
use tessella_core::telemetry::gpu_health::{
    GpuFailureKind, GpuHealth, GpuHealthReport, GpuVerdict,
};
use tessella_core::telemetry::metrics::ExpertActivationRegistry;
use tessella_core::{
    error::DomainError,
    traits::{DistributedInference, DistributedSession, PipelineRouter},
};
use tessella_inference::backends::activation_dtype_bytes;
use tessella_inference::{
    backends::{BackendRegistry, InferenceBackend, InferenceSession},
    HiddenStatePayload, PayloadTensorExt, ShardedWeights,
};
use tessella_protocol::layer_ranges;
use tessella_protocol::signing::{ReplayCache, SignedRequest};

use crate::callback_guard::CallbackGuard;
use crate::engine::canary_audit::{
    audit_order_of, is_marked_step, resolve_ticket_priority, sign_and_store_work_proof, unix_now,
};
use crate::scheduler::{LayerRunner, SchedulerCommand};

pub use crate::engine::canary_audit::{build_hop_work_proof, new_session_for_step};
pub use crate::engine::health::{device_probe, DeviceProbe, GpuHealthMonitor};
pub use crate::engine::sessions::{
    apply_session_metadata, issue_session_id, Session, SessionManager, SESSION_TTL_SECS,
};

/// The ceiling of **one hop** for this node (`TASK-47`/S53, `TASK-76`/S83).
pub fn hop_token_ceiling(
    model_config: &ModelConfig,
    device: &Device,
    declared_tokens: usize,
) -> Option<HopCeiling> {
    hop_ceiling(
        max_input_tokens_per_hop(model_config.hidden_size, activation_dtype_bytes(device)),
        declared_tokens,
    )
}

/// What a hop has to do with a hidden state of `tokens` tokens that must travel to the next node.
#[derive(Debug, PartialEq, Eq)]
pub enum HopTransport {
    Fits,
    NearLimit { tokens: usize, ceiling: HopCeiling },
    TooLarge { tokens: usize, ceiling: HopCeiling },
}

/// Decides the fate of a hop (`TASK-47`/S53, extended by `TASK-76`/S83).
pub fn hop_transport_verdict(
    tokens: usize,
    hidden_size: usize,
    dtype_bytes: usize,
    declared_tokens: usize,
) -> HopTransport {
    let Some(ceiling) = hop_ceiling(
        max_input_tokens_per_hop(hidden_size, dtype_bytes),
        declared_tokens,
    ) else {
        return HopTransport::Fits;
    };
    if tokens > ceiling.tokens {
        return HopTransport::TooLarge { tokens, ceiling };
    }
    if tessella_core::constants::inference::is_near_the_limit(tokens, ceiling.tokens) {
        return HopTransport::NearLimit { tokens, ceiling };
    }
    HopTransport::Fits
}

/// Builds the token-stream callback of the final shard (H-10 · `TASK-06` mitigation 2).
fn token_stream_request(
    client: &reqwest::Client,
    token_url: &str,
    signature: &[u8],
    payload: &tessella_protocol::TokenStreamPayload,
) -> reqwest::RequestBuilder {
    client
        .post(token_url)
        .bearer_auth(tessella_core::callback::capability::callback_capability(
            signature,
        ))
        .json(payload)
        .timeout(std::time::Duration::from_secs(2))
}

/// The weights this node holds, one entry per range of its assignment (`TASK-96`/S102).
///
/// A node used to hold exactly one shard and build exactly one backend (`backend: Option<…>`), and
/// that is why «one range per node» looked true all the way down. With two disjoint ranges the node
/// holds two shards of the same model — the loader already filters each call to the layers of the
/// range it is given — and builds two backends, because `Qwen2Model` owns the `Vec<Layer>` of *its*
/// range and `apply_layers` walks that vector. A hop is served by the backend of its step, and the
/// range travels with it: nothing in the inference side had to learn about sets.
pub type ShardSet = Vec<((usize, usize), Arc<ShardedWeights>)>;

/// The backends built from those shards, keyed by the same range and in ascending order.
pub type BackendSet = Vec<((usize, usize), Arc<dyn InferenceBackend>)>;

/// Orchestrates distributed inference, composing SessionManager, GpuHealthMonitor,
/// and backend dispatch (SOLID-08).
pub struct InferenceEngine {
    /// One backend per range of the assignment (`TASK-96`/S102). A node with a single range — every
    /// deployment until the orchestrator started handing out second chunks — holds exactly one.
    pub backends: RwLock<BackendSet>,
    /// The weights each backend was built from, keyed by the same range. Kept because a reload
    /// replaces them and the operator-facing paths ask what this node downloaded.
    pub shards: Arc<RwLock<ShardSet>>,
    pub model_config: ModelConfig,
    pub assignment: RwLock<WorkerAssignment>,
    pub device: Device,
    pub registry: Arc<BackendRegistry>,
    pub session_mgr: Arc<SessionManager>,
    pub sessions: Arc<DashMap<String, Box<dyn InferenceSession>>>,
    pub session_timestamps: Arc<DashMap<String, std::time::Instant>>,
    pub scheduler_tx: Option<mpsc::Sender<SchedulerCommand>>,
    pub router: RwLock<Option<Arc<dyn PipelineRouter>>>,
    pub own_api_url: Option<String>,
    pub http_client: reqwest::Client,
    pub storage: Option<Arc<tessella_storage::SqliteStorage>>,
    pub keypair: Option<tessella_network::NetworkKeypair>,
    pub orchestrator_public_key: Option<ed25519_dalek::VerifyingKey>,
    pub callback_guard: CallbackGuard,
    pub replay_cache: Arc<ReplayCache>,
    pub callback_capabilities: Arc<tessella_core::callback::capability::CallbackCapabilities>,
    pub expert_activations: Arc<ExpertActivationRegistry>,
    pub health: GpuHealthMonitor,
    pub gpu_health: Arc<GpuHealth>,
    pub canary: crate::config::CanaryMode,
    pub max_input_tokens: usize,
}

impl InferenceEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shards: ShardSet,
        model_config: ModelConfig,
        assignment: WorkerAssignment,
        device: Device,
        registry: Arc<BackendRegistry>,
        own_api_url: Option<String>,
        storage: Option<Arc<tessella_storage::SqliteStorage>>,
        keypair: Option<tessella_network::NetworkKeypair>,
        orchestrator_public_key: Option<ed25519_dalek::VerifyingKey>,
        callback_guard: CallbackGuard,
        callback_capabilities: Arc<tessella_core::callback::capability::CallbackCapabilities>,
        expert_activations: Arc<ExpertActivationRegistry>,
        canary: crate::config::CanaryMode,
        max_input_tokens: usize,
        heartbeat_interval_secs: u64,
        oom_streak: u32,
        streak_window_secs: u64,
    ) -> Self {
        let session_mgr = Arc::new(SessionManager::new());
        let sessions = session_mgr.sessions.clone();
        let session_timestamps = session_mgr.session_timestamps.clone();
        let health = GpuHealthMonitor::new(
            device.clone(),
            heartbeat_interval_secs,
            oom_streak,
            streak_window_secs,
        );
        let gpu_health = health.gpu_health();

        Self {
            backends: RwLock::new(Vec::new()),
            shards: Arc::new(RwLock::new(shards)),
            model_config,
            assignment: RwLock::new(assignment),
            device,
            registry,
            session_mgr,
            sessions,
            session_timestamps,
            scheduler_tx: None,
            router: RwLock::new(None),
            own_api_url,
            http_client: reqwest::Client::new(),
            storage,
            keypair,
            orchestrator_public_key,
            callback_guard,
            replay_cache: Arc::new(ReplayCache::new()),
            callback_capabilities,
            expert_activations,
            health,
            gpu_health,
            canary,
            max_input_tokens,
        }
    }

    fn authenticate<T: SignedRequest>(&self, req: &T, context: &str) -> Result<(), DomainError> {
        tessella_network::verify_signed_request(req, &self.replay_cache, unix_now(), context)
            .map_err(|e| {
                warn!("🔏 Rejecting unsigned/forged request on {context}: {e} (H-09)");
                DomainError::Validation(format!("signature rejected on {context}: {e}"))
            })
    }

    pub async fn set_router(&self, router: Arc<dyn PipelineRouter>) {
        let mut router_lock = self.router.write().await;
        *router_lock = Some(router);
    }

    pub fn set_device_probe(&mut self, probe: DeviceProbe) {
        self.health.set_device_probe(probe);
    }

    pub fn probe_device(&self) -> Result<(), String> {
        self.health.probe_device()
    }

    #[must_use]
    pub fn gpu_report(&self) -> GpuHealthReport {
        self.health.gpu_report()
    }

    pub fn note_gpu_failure(&self, context: &str, failure: &str) -> Option<GpuVerdict> {
        self.health.note_gpu_failure(context, failure)
    }

    pub fn note_gpu_failure_kind(
        &self,
        context: &str,
        kind: GpuFailureKind,
        failure: &str,
    ) -> GpuVerdict {
        self.health.note_gpu_failure_kind(context, kind, failure)
    }

    pub fn revalidate_gpu(&self) -> Result<(), String> {
        self.health.revalidate_gpu()
    }

    pub fn refuse_if_gpu_down(&self) -> Result<(), DomainError> {
        self.health.refuse_if_gpu_down()
    }

    pub(crate) fn refuse_missing_prefill(
        &self,
        key: &str,
        seq_start_pos: usize,
        layer_range: (usize, usize),
    ) -> Option<DomainError> {
        self.session_mgr
            .refuse_missing_prefill(key, seq_start_pos, layer_range)
    }

    pub fn set_scheduler_tx(&mut self, tx: mpsc::Sender<SchedulerCommand>) {
        self.scheduler_tx = Some(tx);
    }

    pub(crate) fn new_session_for_step(
        &self,
        backend: Option<&Arc<dyn InferenceBackend>>,
    ) -> Result<Box<dyn InferenceSession>, DomainError> {
        crate::engine::canary_audit::new_session_for_step(self.canary, backend)
    }

    /// The range of this node that serves the hop `payload` asks for, with the backend built for it
    /// (`TASK-96`/S102).
    ///
    /// The step is read from the payload — the last layer the node before us executed (`layer_idx`),
    /// or the front of the prompt when the payload still carries its tokens — and **never** from the
    /// assignment: with more than one range there is no «the node's range» to execute. A hop asking
    /// for a step this node does not serve is refused here, naming the ranges it does serve, instead
    /// of being answered with the weights of the other range (invariant 17).
    async fn step_target(
        &self,
        payload: &HiddenStatePayload,
    ) -> Result<((usize, usize), Option<Arc<dyn InferenceBackend>>), DomainError> {
        let step = if payload.tokens.is_empty() {
            payload.layer_idx + 1
        } else {
            // A payload that still carries the prompt tokens is the front of the pipeline: there is
            // no previous step to continue from, so the range that starts at layer 0 is the one.
            0
        };

        // The **assignment** decides whether this node serves the step; the backends are what it
        // runs it with. They are two different questions on purpose: a canary serves its step with
        // no backend and no weight at all (`TASK-71`/S78), and it is precisely the node whose
        // answer the audit compares.
        let ranges = self.assignment.read().await.layer_ranges.clone();
        let range = layer_ranges::range_covering(&ranges, step).ok_or_else(|| {
            DomainError::Validation(format!(
                "Refusing the hop for layer {step} of req {}: this node serves {ranges:?} \
                 (TASK-96/S102)",
                payload.request_id
            ))
        })?;

        let backend = self.backend_for(range).await.ok();
        Ok((range, backend))
    }

    /// The backend loaded for `range`, or the refusal that names what this node does hold.
    async fn backend_for(
        &self,
        range: (usize, usize),
    ) -> Result<Arc<dyn InferenceBackend>, DomainError> {
        let backends = self.backends.read().await;
        backends
            .iter()
            .find(|(loaded, _)| *loaded == range)
            .map(|(_, backend)| backend.clone())
            .ok_or_else(|| {
                DomainError::Inference(format!(
                    "No backend is loaded for range {range:?} of this node (loaded: {:?})",
                    backends
                        .iter()
                        .map(|(loaded, _)| *loaded)
                        .collect::<Vec<_>>()
                ))
            })
    }

    /// The backend that holds the front of the model — the only one an entry node can start a
    /// prompt with (`TASK-96`/S102). `None` when this node does not serve layer 0.
    pub(crate) async fn head_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        let assignment = self.assignment.read().await;
        let head = layer_ranges::head_range(&assignment.layer_ranges)?;
        drop(assignment);
        self.backend_for(head).await.ok()
    }

    pub async fn initialize_models(
        &self,
        model_dir: &std::path::Path,
        tokenizer: Option<Arc<tessella_inference::TokenizerWrapper>>,
    ) -> Result<(), DomainError> {
        let assignment = self.assignment.read().await.clone();
        let shards = self.shards.read().await.clone();
        self.model_config.validate_assignment(&assignment)?;

        if self.canary.is_on() {
            info!(
                "🐤 Canary node ({:?}): no backend is created and no weight is loaded — every step \
                 served answers with a known-wrong tensor of the right shape (TASK-71/S78)",
                self.canary
            );
            return Ok(());
        }

        let tokenizer = tokenizer.ok_or_else(|| {
            DomainError::Inference(
                "[TESERROR-07] No tokenizer to load the model with: entry nodes require a tokenizer; verify HF_TOKEN or model files"
                    .to_string(),
            )
        })?;

        let mut built: BackendSet = Vec::with_capacity(shards.len());
        for (range, weights) in &shards {
            let backend = self.registry.create(
                &self.model_config,
                model_dir,
                weights,
                &self.device,
                tokenizer.clone(),
                self.expert_activations.clone(),
            )?;
            info!(
                "✅ Homogeneous model loaded (layers [{}, {}), heads={}/{})",
                range.0,
                range.1,
                self.model_config.num_attention_heads,
                self.model_config.num_key_value_heads
            );
            built.push((*range, Arc::from(backend)));
        }

        // The assignment is what the node answers hops with, so a mismatch here is a node that would
        // serve a range it holds no weights for (`TASK-96`/S102). Loud, before any hop arrives.
        let without_weights = assignment
            .layer_ranges
            .iter()
            .filter(|range| !built.iter().any(|(built, _)| built == *range))
            .collect::<Vec<_>>();
        if !without_weights.is_empty() {
            return Err(DomainError::Inference(format!(
                "Refusing to serve: the assignment declares {without_weights:?} and no shard was \
                 loaded for them (loaded: {:?})",
                built.iter().map(|(range, _)| *range).collect::<Vec<_>>()
            )));
        }

        *self.backends.write().await = built;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn begin_session_with_router(
        self: Arc<Self>,
        router: Arc<dyn PipelineRouter>,
        session_hint: Option<String>,
        entry_node_url: Option<String>,
        karma_ticket: Option<KarmaTicket>,
        work_ticket: Option<WorkTicket>,
    ) -> Result<(Box<dyn DistributedSession>, String), DomainError> {
        let request_id = work_ticket
            .as_ref()
            .map(|ticket| ticket.request_id.clone())
            .unwrap_or_else(|| format!("{:x}", rand::random::<u64>()));

        let session_key = self.session_mgr.resolve_session_key(session_hint);

        // A conversation starts at the **front** of the model, so the session of the entry node is
        // the head backend's (`TASK-96`/S102): a node with a second chunk owns two backends and only
        // one of them has the token embedding table that consumes the prompt's tokens. A node that
        // serves no front cannot start a conversation at all — `covers_layer_0` refuses it at boot,
        // and this is the same refusal at the point of use.
        let backend = self.head_backend().await.ok_or_else(|| {
            DomainError::Inference(
                "No backend holds the front of the model, so this node cannot start a conversation \
                 (TASK-96/S102): only the range that contains layer 0 can consume prompt tokens"
                    .to_string(),
            )
        })?;

        let (backend_sess, seq_start_pos) =
            if let Some(existing) = self.sessions.remove(&session_key) {
                let pos = existing.1.get_pos_offset();
                self.session_timestamps
                    .insert(session_key.clone(), std::time::Instant::now());
                info!(
                    "♻️  Resuming existing session '{}' for multi-turn inference at pos {}",
                    session_key, pos
                );
                (existing.1, pos)
            } else {
                let sess = backend.begin_session()?;
                self.session_timestamps
                    .insert(session_key.clone(), std::time::Instant::now());
                (sess, 0)
            };

        Ok((
            Box::new(Session {
                engine: self.clone(),
                router,
                request_id: request_id.clone(),
                session_key: session_key.clone(),
                seq_start_pos,
                backend_sess: Some(backend_sess),
                entry_node_url: entry_node_url.or_else(|| self.own_api_url.clone()),
                karma_ticket,
                work_ticket,
                local_work_reported: false,
                prompt_consumed: false,
            }),
            request_id,
        ))
    }

    #[tracing::instrument(skip(self, bytes), level = "debug")]
    pub async fn forward_hidden(&self, bytes: &[u8]) -> Result<Vec<u8>, DomainError> {
        let payload = HiddenStatePayload::from_bytes(bytes)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        self.authenticate(&payload, "forward_hidden")?;

        let callback_target = match payload.entry_node_url.as_deref() {
            Some(raw) => self.callback_guard.sanitize(raw).await,
            None => None,
        };

        let mut priority = QueuePriority::Leech;
        if let Some(ticket) = &payload.karma_ticket {
            priority = resolve_ticket_priority(
                ticket,
                &payload.client_pubkey,
                &payload.request_id,
                unix_now(),
                self.orchestrator_public_key.as_ref(),
            );
        }

        let rid = payload.request_id.clone();
        let hidden = payload
            .to_tensor(&self.device)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        self.refuse_if_gpu_down()?;

        // The range this hop serves, its backend, and the K/V slot **of that range**
        // (`TASK-96`/S102): a node holding two ranges owns two caches for the same conversation, and
        // a pipeline can come back to it (front `[0, 4)`, tail `[8, 12)`) with the same continuity
        // key. Sharing one slot would hand the second visit the cache of the other backend.
        let (layer_range, backend) = self.step_target(&payload).await?;
        let key = cache_key(payload.session_key(), layer_range);

        if let Some(refusal) =
            self.refuse_missing_prefill(&key, payload.seq_start_pos as usize, layer_range)
        {
            return Err(refusal);
        }

        let input_tensor =
            if layer_range.0 == 0 && !payload.tokens.is_empty() && !self.canary.is_on() {
                candle_core::Tensor::new(payload.tokens.as_slice(), &candle_core::Device::Cpu)
                    .map_err(|e| DomainError::Inference(e.to_string()))?
                    .to_device(&self.device)
                    .map_err(|e| DomainError::Inference(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| DomainError::Inference(e.to_string()))?
            } else {
                hidden
            };

        let output_hidden = if let Some(tx) = &self.scheduler_tx {
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            tx.send(SchedulerCommand {
                request_id: rid.clone(),
                session_key: key.clone(),
                range: layer_range,
                seq_start_pos: payload.seq_start_pos as usize,
                hidden_tensor: input_tensor,
                priority,
                timestamp: std::time::Instant::now(),
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| DomainError::Inference("Scheduler dropped".into()))?;

            let scheduler_res = resp_rx
                .await
                .map_err(|_| DomainError::Inference("Scheduler response channel dropped".into()))?;
            match scheduler_res {
                Ok(val) => {
                    self.gpu_health.note_request_served();
                    val
                }
                Err(e) => {
                    self.note_gpu_failure("forward_hidden/scheduler", &e.to_string());
                    return Err(e);
                }
            }
        } else {
            if !self.sessions.contains_key(&key) {
                let sess = self.new_session_for_step(backend.as_ref())?;
                self.sessions.insert(key.clone(), sess);
                self.session_timestamps
                    .insert(key.clone(), std::time::Instant::now());
            } else {
                self.session_timestamps
                    .insert(key.clone(), std::time::Instant::now());
            }

            let mut session = {
                let Some((_, s)) = self.sessions.remove(&key) else {
                    return Err(DomainError::Inference(
                        "Active session not found or expired".into(),
                    ));
                };
                s
            };
            session.set_pos_offset(payload.seq_start_pos as usize);
            let seq_start_pos = payload.seq_start_pos as usize;

            let join_res = tokio::task::spawn_blocking(move || {
                let res = session.apply_layers(input_tensor, layer_range, seq_start_pos);
                (res, session)
            })
            .await;

            let (res, session) = match join_res {
                Ok((inner_res, session)) => match inner_res {
                    Ok(val) => {
                        self.gpu_health.note_request_served();
                        (Ok(val), session)
                    }
                    Err(e) => {
                        self.note_gpu_failure("forward_hidden", &e.to_string());
                        (Err(e), session)
                    }
                },
                Err(join_err) => {
                    self.note_gpu_failure("forward_hidden/join", &join_err.to_string());
                    return Err(DomainError::Inference(format!(
                        "compute thread panicked: {join_err}"
                    )));
                }
            };

            self.sessions.insert(key.clone(), session);
            res?
        };

        debug!(
            "⚡ Processed hidden state: layers {:?} (request={})",
            layer_range, rid
        );

        let mut res_payload = HiddenStatePayload::from_tensor(
            &output_hidden,
            rid.clone(),
            payload.seq_start_pos,
            layer_range.1 - 1,
        )
        .map_err(|e| DomainError::Inference(e.to_string()))?;

        res_payload.session_id = payload.session_id.clone();
        res_payload.entry_node_url = payload.entry_node_url.clone();
        res_payload.signature = payload.signature.clone();
        res_payload.client_pubkey = payload.client_pubkey.clone();
        res_payload.signature_version = payload.signature_version;
        res_payload.nonce = payload.nonce;
        res_payload.iat = payload.iat;
        res_payload.karma_ticket = payload.karma_ticket.clone();
        res_payload.work_ticket = payload.work_ticket.clone();

        let payload_bytes = res_payload
            .to_bytes()
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        if let (Some(storage), Some(keypair)) = (&self.storage, &self.keypair) {
            match payload.work_ticket.as_ref() {
                Some(ticket) => {
                    let input_data = if layer_range.0 == 0 {
                        None
                    } else {
                        Some(payload.data.as_slice())
                    };
                    sign_and_store_work_proof(
                        storage,
                        keypair,
                        &rid,
                        layer_range,
                        input_data,
                        &res_payload.data,
                        ticket,
                    )
                    .await;
                }
                None => warn!(
                    "🧾 Hop of req {rid} carries no work ticket: no evidence stored, so this work \
                     cannot be credited (TASK-57/S63)"
                ),
            }
        }

        let audit_order = audit_order_of(payload.work_ticket.as_ref());
        let marked_step = is_marked_step(audit_order.as_ref(), layer_range.0);
        let total_layers = self.model_config.total_layers();
        if marked_step {
            info!(
                "🧾 Marked step of the sampled session {rid}: layers {layer_range:?} are returned in \
                 isolation for the compute audit (TASK-70/S77)"
            );
            Ok(payload_bytes)
        } else if layer_range.1 < total_layers {
            let router_lock = self.router.read().await;
            if let Some(ref router) = *router_lock {
                debug!(
                    "➡️ Routing hidden state to next hop (current_layer={})",
                    layer_range.1
                );
                router
                    .route(
                        layer_range.1,
                        total_layers,
                        payload_bytes,
                        &self.model_config.model_id,
                        audit_order.clone(),
                    )
                    .await
            } else {
                warn!("⚠️ Missing router in forward_hidden hop — returning raw bytes");
                Ok(payload_bytes)
            }
        } else {
            if let Some(ref entry_url) = callback_target {
                let token_url = format!(
                    "{}/v1/token_stream/{}",
                    entry_url.trim_end_matches('/'),
                    rid
                );
                let token_payload = tessella_protocol::TokenStreamPayload {
                    request_id: rid.clone(),
                    token: String::new(),
                    finish_reason: None,
                };
                let signature = payload.signature.clone();
                let client = self.http_client.clone();
                tokio::spawn(async move {
                    let request =
                        token_stream_request(&client, &token_url, &signature, &token_payload);
                    if let Err(e) = request.send().await {
                        error!("⚠️ Token stream callback to {} failed: {}", token_url, e);
                    }
                });
            }
            Ok(payload_bytes)
        }
    }

    pub fn spawn_cleanup_task(
        &self,
        cancel_token: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        self.session_mgr
            .spawn_cleanup_task(self.replay_cache.clone(), cancel_token)
    }

    pub async fn drain_sessions(&self, timeout: Duration) -> bool {
        self.session_mgr.drain_sessions(timeout).await
    }
}

#[async_trait::async_trait]
impl DistributedInference for InferenceEngine {
    async fn reset_distributed_cache(&self) {
        let dropped = self.sessions.len();
        self.session_mgr.clear();
        if dropped > 0 {
            debug!("🧹 Dropped {dropped} local inference session(s) of this engine");
        }
    }

    async fn apply_prompt_template(&self, mode: &InferenceMode, raw_prompt: &str) -> String {
        // Every backend of a node is built from the same model config, so the template is the same
        // in all of them (`TASK-96`/S102); the first one is read rather than the head's to keep this
        // answer independent of which chunk happens to hold layer 0.
        let backends = self.backends.read().await;
        if let Some((_, backend)) = backends.first() {
            backend.apply_prompt_template(mode, raw_prompt).await
        } else {
            raw_prompt.to_string()
        }
    }

    async fn begin_session(
        self: Arc<Self>,
        session_hint: Option<String>,
    ) -> Result<(Box<dyn DistributedSession>, String), DomainError> {
        let router = {
            let router_lock = self.router.read().await;
            router_lock
                .clone()
                .ok_or_else(|| DomainError::Inference("Router not initialized".into()))?
        };

        let (session, request_id) = self
            .begin_session_with_router(router, session_hint, None, None, None)
            .await?;
        Ok((session, request_id))
    }

    async fn forward_hidden(&self, bytes: &[u8]) -> Result<Vec<u8>, DomainError> {
        self.forward_hidden(bytes).await
    }

    async fn current_assignment(&self) -> Option<WorkerAssignment> {
        Some(self.assignment.read().await.clone())
    }
}

#[async_trait::async_trait]
impl LayerRunner for InferenceEngine {
    async fn run_single(
        &self,
        session_key: &str,
        range: (usize, usize),
        seq_start_pos: usize,
        hidden_tensor: Tensor,
    ) -> Result<Tensor, DomainError> {
        let key = session_key;
        // The range comes with the command, resolved by `step_target` when the hop arrived
        // (`TASK-96`/S102): the scheduler is a queue of steps and a step belongs to one range.
        let layer_range = range;

        if let Some(refusal) = self.refuse_missing_prefill(key, seq_start_pos, layer_range) {
            return Err(refusal);
        }

        if !self.sessions.contains_key(key) {
            let backend = self.backend_for(range).await.ok();
            let sess = self.new_session_for_step(backend.as_ref())?;
            self.sessions.insert(key.to_string(), sess);
            self.session_timestamps
                .insert(key.to_string(), std::time::Instant::now());
        }

        let mut sess = {
            let Some((_, s)) = self.sessions.remove(key) else {
                return Err(DomainError::Inference("Session evicted or expired".into()));
            };
            s
        };
        sess.set_pos_offset(seq_start_pos);

        self.refuse_if_gpu_down()?;

        let join_res = tokio::task::spawn_blocking(move || {
            let res = sess.apply_layers(hidden_tensor, layer_range, seq_start_pos);
            (res, sess)
        })
        .await;

        let (res, sess_back) = match join_res {
            Ok((inner_res, sess_back)) => match inner_res {
                Ok(val) => {
                    self.gpu_health.note_request_served();
                    (Ok(val), sess_back)
                }
                Err(e) => {
                    self.note_gpu_failure("scheduler.run_single", &e.to_string());
                    (Err(e), sess_back)
                }
            },
            Err(join_err) => {
                self.note_gpu_failure("scheduler.run_single/join", &join_err.to_string());
                return Err(DomainError::Inference(format!(
                    "compute thread panicked: {join_err}"
                )));
            }
        };

        self.sessions.insert(key.to_string(), sess_back);
        self.session_timestamps
            .insert(key.to_string(), std::time::Instant::now());

        res
    }
}

/// The key of the K/V cache slot of **one range** of **one** session (`TASK-96`/S102).
///
/// A node with two ranges owns two backends and therefore two caches, and one pipeline can come back
/// to it — front `[0, 4)` and tail `[8, 12)` of the same conversation — under the same continuity
/// key. Sharing the slot would hand the second visit the cache of the other backend, which is a
/// wrong answer with no error attached.
pub(crate) fn cache_key(session_key: &str, range: (usize, usize)) -> String {
    format!("{session_key}#{}-{}", range.0, range.1)
}

#[cfg(test)]
#[path = "engine/tests.rs"]
mod tests;
