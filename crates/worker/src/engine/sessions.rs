use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use tessella_core::error::DomainError;
use tessella_core::models::{KarmaTicket, WorkTicket};
use tessella_core::traits::{DistributedSession, ForwardOutput, PipelineRouter};
use tessella_inference::backends::{activation_dtype_bytes, InferenceSession};
use tessella_inference::{HiddenStatePayload, PayloadTensorExt};
use tessella_protocol::signing::{ReplayCache, SIGNATURE_VERSION_V1};

use crate::engine::canary_audit::{audit_order_of, sign_and_store_work_proof, unix_now};
use crate::inference_engine::{hop_transport_verdict, HopTransport, InferenceEngine};

/// How long an idle inference session (and with it the KV cache it holds) survives without a request.
pub const SESSION_TTL_SECS: u64 = 300;

/// Issues the id of a session: the canonical name of one conversation's K/V cache (`TASK-78`/S85, `D-36`).
pub fn issue_session_id() -> String {
    format!("sess-{:032x}", rand::random::<u128>())
}

/// Copies onto `payload` the metadata that belongs to the **session** that emits it.
pub fn apply_session_metadata(
    payload: &mut HiddenStatePayload,
    session_id: &str,
    entry_node_url: &Option<String>,
    karma_ticket: &Option<KarmaTicket>,
    work_ticket: &Option<WorkTicket>,
) {
    payload.session_id = Some(session_id.to_string());
    payload.entry_node_url = entry_node_url.clone();
    payload.karma_ticket = karma_ticket.clone();
    payload.work_ticket = work_ticket.clone();
}

/// Manages active inference sessions, KV-cache tables, and TTL background sweeping (SOLID-08).
pub struct SessionManager {
    pub sessions: Arc<DashMap<String, Box<dyn InferenceSession>>>,
    pub session_timestamps: Arc<DashMap<String, std::time::Instant>>,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            session_timestamps: Arc::new(DashMap::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    pub fn clear(&self) {
        self.sessions.clear();
        self.session_timestamps.clear();
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.sessions.contains_key(key)
    }

    pub fn contains_timestamp(&self, key: &str) -> bool {
        self.session_timestamps.contains_key(key)
    }

    pub fn touch(&self, key: &str) {
        self.session_timestamps
            .insert(key.to_string(), std::time::Instant::now());
    }

    pub fn insert(&self, key: String, session: Box<dyn InferenceSession>) {
        self.sessions.insert(key.clone(), session);
        self.session_timestamps
            .insert(key, std::time::Instant::now());
    }

    pub fn remove(&self, key: &str) -> Option<Box<dyn InferenceSession>> {
        self.sessions.remove(key).map(|(_, s)| s)
    }

    pub fn remove_timestamp(&self, key: &str) {
        self.session_timestamps.remove(key);
    }

    /// Resolves or generates the session key given an optional client hint.
    pub fn resolve_session_key(&self, hint: Option<String>) -> String {
        let hint = hint.filter(|id| !id.trim().is_empty());
        match hint {
            Some(hint) if self.session_timestamps.contains_key(&hint) => hint,
            _ => issue_session_id(),
        }
    }

    /// The guard of `TASK-68`/`S75` (`F-1`): checks if a prefix is declared but no session exists.
    pub fn refuse_missing_prefill(
        &self,
        key: &str,
        seq_start_pos: usize,
        layer_range: (usize, usize),
    ) -> Option<DomainError> {
        let session_is_new = !self.sessions.contains_key(key);
        let declares_a_prefix = seq_start_pos > 0;
        if !(session_is_new && declares_a_prefix) {
            return None;
        }
        let refusal = tessella_core::error::missing_prefill_error(key, seq_start_pos, layer_range);
        warn!("🚫 {refusal}");
        Some(refusal)
    }

    /// Spawns the periodic cache/nonce/session sweeper and returns its `JoinHandle`.
    pub fn spawn_cleanup_task(
        &self,
        replay_cache: Arc<ReplayCache>,
        cancel_token: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let cleanup_sessions = self.sessions.clone();
        let cleanup_timestamps = self.session_timestamps.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let dropped = replay_cache.prune(unix_now());
                        if dropped > 0 {
                            debug!("🧹 Dropped {dropped} expired signature nonces from the replay cache");
                        }
                        let now = std::time::Instant::now();
                        let mut to_remove = Vec::new();
                        for entry in cleanup_timestamps.iter() {
                            if now.duration_since(*entry.value()).as_secs() > SESSION_TTL_SECS {
                                to_remove.push(entry.key().clone());
                            }
                        }
                        for rid in to_remove {
                            cleanup_sessions.remove(&rid);
                            cleanup_timestamps.remove(&rid);
                            debug!("🧹 Cleaned up idle session: {}", rid);
                        }
                    },
                    _ = cancel_token.cancelled() => break,
                }
            }
        })
    }

    pub async fn drain_sessions(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if self.sessions.is_empty() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        self.sessions.is_empty()
    }
}

pub struct Session {
    pub engine: Arc<InferenceEngine>,
    pub router: Arc<dyn PipelineRouter>,
    pub request_id: String,
    pub session_key: String,
    pub seq_start_pos: usize,
    pub backend_sess: Option<Box<dyn InferenceSession>>,
    pub entry_node_url: Option<String>,
    pub karma_ticket: Option<KarmaTicket>,
    pub prompt_consumed: bool,
    pub work_ticket: Option<WorkTicket>,
    pub local_work_reported: bool,
}

#[async_trait::async_trait]
impl DistributedSession for Session {
    fn session_id(&self) -> &str {
        &self.session_key
    }

    async fn forward(&mut self, tokens: Vec<u32>) -> Result<ForwardOutput, DomainError> {
        self.engine.refuse_if_gpu_down()?;

        let is_empty = !self.prompt_consumed;
        self.prompt_consumed = true;

        let input_tokens = if is_empty {
            if self.seq_start_pos > 0 && self.seq_start_pos < tokens.len() {
                let sliced = tokens[self.seq_start_pos..].to_vec();
                debug!(
                    "Slicing input tokens: seq_start_pos={}, total_tokens={}, sliced_tokens={:?}, prefix_tokens={:?}",
                    self.seq_start_pos,
                    tokens.len(),
                    sliced,
                    &tokens[..self.seq_start_pos]
                );
                sliced
            } else {
                if self.seq_start_pos >= tokens.len() && self.seq_start_pos > 0 {
                    warn!(
                        "seq_start_pos {} is out of bounds for tokens length {} during session resumption",
                        self.seq_start_pos,
                        tokens.len()
                    );
                }
                tokens.clone()
            }
        } else {
            vec![*tokens
                .last()
                .ok_or_else(|| DomainError::Inference("Empty tokens".into()))?]
        };

        let original_seq_start_pos = self.seq_start_pos;
        let n_input_tokens = input_tokens.len();

        // The front of a prompt is executed by the range that holds the token embedding table
        // (`TASK-96`/S102): on a node with a second chunk that is no longer «its range», and a node
        // that serves no front cannot be an entry point at all (`covers_layer_0` refuses it at boot,
        // naming the reason).
        let layer_range = {
            let assignment = self.engine.assignment.read().await;
            tessella_protocol::layer_ranges::head_range(&assignment.layer_ranges).ok_or_else(
                || {
                    DomainError::Inference(
                    "This node serves no front (no range starting at layer 0), so it cannot start \
                     a conversation (TASK-96/S102)"
                        .to_string(),
                )
                },
            )?
        };
        let total_layers = self.engine.model_config.total_layers();
        let will_be_routed = layer_range.1 < total_layers;
        if will_be_routed {
            let hidden_size = self.engine.model_config.hidden_size;
            let dtype_bytes = activation_dtype_bytes(&self.engine.device);
            match hop_transport_verdict(
                n_input_tokens,
                hidden_size,
                dtype_bytes,
                self.engine.max_input_tokens,
            ) {
                HopTransport::Fits => {}
                HopTransport::NearLimit { tokens, ceiling } => warn!(
                    "⚠️ Hidden state near the ceiling of one hop: {tokens}/{} tokens \
                     ({hidden_size}-wide, {dtype_bytes} bytes per element, source: {}) \
                     (TASK-47/S53 · TASK-76/S83)",
                    ceiling.tokens,
                    ceiling.source.as_str()
                ),
                HopTransport::TooLarge { tokens, ceiling } => {
                    warn!(
                        "🚫 Rejecting a request whose hidden state does not fit in one hop: \
                         {tokens} > {} tokens ({hidden_size}-wide, {dtype_bytes} bytes per element, \
                         source: {}, req {})",
                        ceiling.tokens,
                        ceiling.source.as_str(),
                        self.request_id
                    );
                    return Err(tessella_core::error::prompt_too_long_error(
                        tokens,
                        hidden_size,
                        dtype_bytes,
                        &ceiling,
                    ));
                }
            }
        }

        if self.backend_sess.is_none() {
            let backend = self.engine.head_backend().await.ok_or_else(|| {
                DomainError::Inference(
                    "[TESERROR-08] No backend holds the front of the model (layer 0); the pipeline is incomplete"
                        .to_string(),
                )
            })?;
            self.backend_sess = Some(backend.begin_session()?);
        }

        let Some(mut backend_sess) = self.backend_sess.take() else {
            return Err(DomainError::Inference(
                "backend session lost before applying layers (internal error: the session was taken twice)".into(),
            ));
        };

        let input_tensor =
            candle_core::Tensor::new(input_tokens.as_slice(), &candle_core::Device::Cpu)
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .to_device(&self.engine.device)
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| DomainError::Inference(e.to_string()))?;

        let join_res = tokio::task::spawn_blocking(move || {
            let res = backend_sess.apply_layers(input_tensor, layer_range, original_seq_start_pos);
            (res, backend_sess)
        })
        .await;

        let (res, backend_sess) = match join_res {
            Ok((inner_res, backend_sess)) => match inner_res {
                Ok(val) => {
                    self.engine.gpu_health.note_request_served();
                    (Ok(val), backend_sess)
                }
                Err(e) => {
                    self.engine
                        .note_gpu_failure("session_forward", &e.to_string());
                    (Err(e), backend_sess)
                }
            },
            Err(join_err) => {
                self.engine
                    .note_gpu_failure("session_forward/join", &join_err.to_string());
                return Err(DomainError::Inference(format!(
                    "compute thread panicked: {join_err}"
                )));
            }
        };

        self.backend_sess = Some(backend_sess);
        let current_hidden = res?;

        let mut payload = HiddenStatePayload::from_tensor(
            &current_hidden,
            self.request_id.clone(),
            original_seq_start_pos as u32,
            layer_range.1 - 1,
        )
        .map_err(|e| DomainError::Inference(e.to_string()))?;

        apply_session_metadata(
            &mut payload,
            &self.session_key,
            &self.entry_node_url,
            &self.karma_ticket,
            &self.work_ticket,
        );

        let audit_order = audit_order_of(self.work_ticket.as_ref());
        if audit_order
            .as_ref()
            .is_some_and(|order| order.marks(layer_range.0))
        {
            payload.tokens = input_tokens.clone();
        }

        let Some(keypair) = self.engine.keypair.as_ref() else {
            return Err(DomainError::Network(
                "no local keypair: refusing to emit an unsigned hidden-state payload (H-09)"
                    .to_string(),
            ));
        };
        let nonce: u64 = rand::random();
        tessella_network::seal_signed_request(keypair, &mut payload, nonce, unix_now())
            .map_err(|e| DomainError::Network(format!("could not seal payload: {e}")))?;

        self.engine.callback_capabilities.register(
            &self.request_id,
            &payload.signature,
            std::time::Instant::now(),
        );
        debug!(
            "🔏 Sealed hidden-state payload for req {} (nonce={}, version={})",
            self.request_id, nonce, SIGNATURE_VERSION_V1
        );

        let payload_bytes = payload
            .to_bytes()
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        if !self.local_work_reported {
            self.local_work_reported = true;
            if let (Some(storage), Some(keypair)) = (&self.engine.storage, &self.engine.keypair) {
                match self.work_ticket.as_ref() {
                    Some(ticket) => {
                        sign_and_store_work_proof(
                            storage,
                            keypair,
                            &self.request_id,
                            layer_range,
                            None,
                            &payload.data,
                            ticket,
                        )
                        .await;
                    }
                    None => warn!(
                        "🧾 Entry node of req {} has no work ticket: no evidence stored, so its own \
                         layers cannot be credited (TASK-57/S63)",
                        self.request_id
                    ),
                }
            }
        }

        if will_be_routed
            && payload_bytes.len() > tessella_core::constants::inference::MAX_HIDDEN_STATE_BYTES
        {
            warn!(
                "🚫 Rejecting a payload of {} bytes on the next hop (limit {} on 'POST /v1/forward', req {})",
                payload_bytes.len(),
                tessella_core::constants::inference::MAX_HIDDEN_STATE_BYTES,
                self.request_id
            );
            return Err(tessella_core::error::payload_too_large_error(
                payload_bytes.len(),
                tessella_core::constants::inference::MAX_HIDDEN_STATE_BYTES,
            ));
        }

        if let Some(order) = audit_order.as_ref() {
            if order.marks(layer_range.0) {
                let my_peer_id = self
                    .engine
                    .keypair
                    .as_ref()
                    .map(tessella_network::peer_id_from_keypair)
                    .unwrap_or_default();
                self.router
                    .audit_front(
                        layer_range,
                        &payload_bytes,
                        &my_peer_id,
                        &payload_bytes,
                        order,
                    )
                    .await?;
            }
        }

        let final_bytes = if will_be_routed {
            match self
                .router
                .route(
                    layer_range.1,
                    total_layers,
                    payload_bytes,
                    &self.engine.model_config.model_id,
                    audit_order,
                )
                .await
            {
                Ok(bytes) => bytes,
                Err(error) => {
                    self.discard_lost_session(&error);
                    return Err(error);
                }
            }
        } else {
            payload_bytes
        };

        let final_payload = HiddenStatePayload::from_bytes(&final_bytes)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        let logits = final_payload
            .to_tensor(&candle_core::Device::Cpu)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        self.seq_start_pos += n_input_tokens;

        debug!(
            "Forwarded layers up to the pipeline tail: seq_start_pos was {}, now {}",
            self.seq_start_pos - n_input_tokens,
            self.seq_start_pos
        );

        Ok(ForwardOutput {
            logits: Some(
                logits
                    .to_vec1::<f32>()
                    .map_err(|e| DomainError::Inference(e.to_string()))?,
            ),
            seq_start_pos: self.seq_start_pos,
        })
    }
}

impl Session {
    fn discard_lost_session(&mut self, error: &DomainError) {
        if !tessella_core::error::is_missing_prefill(error) {
            return;
        }
        warn!(
            "🗑️ Dropping session '{key}': a hop of this pipeline no longer has its prefix, so the \
             next request has to prefill it from scratch (S76/D-30) — {error}",
            key = self.session_key
        );
        self.backend_sess = None;
        self.engine.sessions.remove(&self.session_key);
        self.engine.session_timestamps.remove(&self.session_key);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(backend_sess) = self.backend_sess.take() {
            debug!(
                "Saving entry node session '{}' back to engine.sessions on drop",
                self.session_key
            );
            self.engine
                .sessions
                .insert(self.session_key.clone(), backend_sess);
            self.engine
                .session_timestamps
                .insert(self.session_key.clone(), std::time::Instant::now());
        }
    }
}
