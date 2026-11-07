use crate::error::DomainError;
use crate::{AnnounceRequest, AnnounceResponse, InferenceMode, NodeInfo};
use std::sync::Arc;

pub mod pipeline_router;
pub use pipeline_router::{AuditOrder, PipelineRouter};

pub mod peer_network;
pub use peer_network::PeerNetwork;

pub mod tokenizer;
pub use tokenizer::TokenizerService;

pub struct ForwardOutput {
    /// The logits that close this node's slice of the pipeline. `None` if the backend does not
    /// expose them (the entry node cannot sample without them).
    pub logits: Option<Vec<f32>>,
    pub seq_start_pos: usize,
}

/// Interface for distributed inference across the network.
///
/// There was a `distributed_forward(tokens)` here until S45 (`TASK-39`): it was the only producer
/// of the `BatchCollector` (the worker's second, *unreachable* batching path) and it had **zero**
/// call sites — the entry-node work of a request goes through `begin_session` +
/// `DistributedSession::forward`. It is gone, together with the batching tables nobody fed.
#[async_trait::async_trait]
pub trait DistributedInference: Send + Sync {
    async fn reset_distributed_cache(&self);

    /// Applies the prompt template according to the active model
    async fn apply_prompt_template(&self, mode: &InferenceMode, raw_prompt: &str) -> String;

    // New session-based API
    //
    // Returns the session **and** the `request_id` the orchestrator authorised for it
    // (`TASK-07`, H-12). The work identifier is minted by the orchestrator, so the entry node
    // cannot know it before asking: it comes back from `/api/v1/session_grant` together with the
    // Karma ticket, in the single round-trip this call already paid for the QoS ticket.
    //
    // `session_hint` is the client's `session_id` and it is only a **hint** (`TASK-78`/S85, `D-36`):
    // an id of a conversation is issued by the node that owns the K/V cache, and the hint is honoured
    // only when it names a session that node still holds. Nothing in this signature can promise
    // continuity — the caller has to ask the returned session for
    // [`DistributedSession::session_id`] and publish *that* — because the id a client sends cannot be
    // trusted to name a session of its own.
    async fn begin_session(
        self: Arc<Self>,
        session_hint: Option<String>,
    ) -> Result<(Box<dyn DistributedSession>, String), DomainError>;

    async fn forward_hidden(&self, _bytes: &[u8]) -> Result<Vec<u8>, DomainError> {
        Err(DomainError::Inference(
            "forward_hidden not implemented".into(),
        ))
    }

    async fn rebalance(&self, _layer_start: usize, _layer_end: usize) -> Result<(), DomainError> {
        Err(DomainError::Inference("rebalance not implemented".into()))
    }

    async fn update_assignment(
        &self,
        _assignment: crate::WorkerAssignment,
    ) -> Result<(), DomainError> {
        Err(DomainError::Inference(
            "update_assignment not implemented".into(),
        ))
    }

    async fn current_assignment(&self) -> Option<crate::WorkerAssignment> {
        None
    }
}

#[async_trait::async_trait]
pub trait DistributedSession: Send + Sync {
    /// The canonical id of this session (`TASK-78`/S85, `D-36`), issued by the node that owns the K/V
    /// cache: it is the value a client has to send back to resume the conversation, so the node has to
    /// publish it (the JSON answer and its header, in the HTTP surface) instead of echoing the hint the
    /// client sent, which named no session at all.
    fn session_id(&self) -> &str;

    async fn forward(&mut self, tokens: Vec<u32>) -> Result<ForwardOutput, DomainError>;
}

/// Interface for node discovery and role assignment.
#[async_trait::async_trait]
pub trait DiscoveryService: Send + Sync {
    async fn announce(&self, req: AnnounceRequest) -> Result<AnnounceResponse, DomainError>;
    async fn discover(&self, model_id: &str) -> Result<Vec<NodeInfo>, DomainError>;
}

/// Interface for persistent storage of network state.
#[async_trait::async_trait]
pub trait StorageService: Send + Sync {
    async fn save_node(&self, node: NodeInfo) -> Result<(), DomainError>;
    async fn get_nodes(&self) -> Result<Vec<NodeInfo>, DomainError>;
    async fn get_all_nodes(&self) -> Result<Vec<NodeInfo>, DomainError>;
    async fn deregister_node(&self, peer_id: &str) -> Result<(), DomainError>;
    async fn save_manifest(&self, manifest: crate::WeightManifest) -> Result<(), DomainError>;
    async fn get_manifest(
        &self,
        model_id: &str,
        role_json: &str,
    ) -> Result<Option<crate::WeightManifest>, DomainError>;
    async fn get_manifests_for_model(
        &self,
        model_id: &str,
    ) -> Result<Vec<crate::WeightManifest>, DomainError>;
    async fn list_manifests(&self) -> Result<Vec<crate::WeightManifest>, DomainError>;
    async fn save_receipt(&self, receipt: &crate::SignedTransferReceipt)
    -> Result<(), DomainError>;
    async fn get_receipt(
        &self,
        sender_peer_id: &str,
        receiver_peer_id: &str,
        model_id: &str,
        role: &str,
    ) -> Result<Option<crate::SignedTransferReceipt>, DomainError>;
    async fn list_receipts(
        &self,
        peer_id: &str,
    ) -> Result<Vec<crate::TransferReceipt>, DomainError>;
    /// The Karma balance of a peer, persisted in `karma_ledger` (`TASK-30`).
    ///
    /// It is the whole persistence of the economy between restarts (SEC-06), so the table is created
    /// by `SqliteStorage`'s constructor and a failure to read it back is loud at boot: before
    /// `TASK-30` the write always failed with `no such table` and the read was masked as "a brand new
    /// cluster".
    async fn update_karma(&self, peer_id: &str, balance: i64) -> Result<(), DomainError>;
    /// Every `(peer_id, balance)` of `karma_ledger`, used to seed the ledger at boot (`TASK-30`).
    async fn get_all_karma(&self) -> Result<Vec<(String, i64)>, DomainError>;

    /// The credited proofs of one session (`TASK-27`/S33), read back to analyse the chain
    /// `output_hash(N) == input_hash(N+1)`.
    ///
    /// It replaces the old `check_conflicting_proof(request_id, output_hash)`, which compared two
    /// proofs of the same session **without looking at their layer range**: every hop of a pipeline
    /// covers a different range (and every decoding step has a different input), so the predicate
    /// flagged honest work as a conflict — and its caller answered with a blacklist. Divergence is
    /// now decided by `musivum_api::audit` over `(request_id, layer_range, input_hash)`.
    ///
    /// The default is empty: an implementation that does not keep the evidence (a test double, an
    /// in-memory credit service) has nothing to analyse instead of failing the credit path.
    async fn get_session_work_proofs(
        &self,
        _request_id: &str,
    ) -> Result<Vec<crate::WorkProof>, DomainError> {
        Ok(Vec::new())
    }

    /// Returns the number of work proofs currently pending in the local WAL (`TASK-94`).
    ///
    /// Used by worker `/metrics` to expose `tessella_wal_pending_proofs_depth`.
    /// The default returns 0 (test doubles or non-worker storage implementations).
    async fn count_pending_work_proofs(&self) -> Result<u64, DomainError> {
        Ok(0)
    }

    /// Monotonic version of the active worker shard assignment (`MSV-19`).
    async fn get_assignment_version(&self) -> Result<u64, DomainError> {
        Ok(0)
    }

    /// Persists the monotonic version of the active worker shard assignment (`MSV-19`).
    async fn set_assignment_version(&self, _version: u64) -> Result<(), DomainError> {
        Ok(())
    }

    /// Returns the last persisted active worker shard assignment, if any (`MSV-20`).
    async fn get_active_assignment(&self) -> Result<Option<crate::WorkerAssignment>, DomainError> {
        Ok(None)
    }

    /// Persists the active worker shard assignment payload alongside its monotonic version (`MSV-20`).
    async fn set_active_assignment(
        &self,
        _version: u64,
        _assignment: &crate::WorkerAssignment,
    ) -> Result<(), DomainError> {
        Ok(())
    }
}

/// Una fila del ledger append-only con su `id` monotónico.
///
/// El `id` es lo que hace reconstruible la federación (`TASK-35`/S41): la réplica avanza su cursor
/// con el último `id` que recibió, en vez de con un recuento de filas. El `source` viaja dentro de
/// [`crate::LedgerEntry`], así que un asiento local del primario y uno que él mismo importó de otro
/// nodo son distinguibles una vez copiados.
#[derive(Debug, Clone)]
pub struct LedgerPageEntry {
    pub id: u64,
    pub entry: crate::LedgerEntry,
}

/// Qué hizo aplicar un lote federado (SP-02 · `TASK-35`).
///
/// Antes el `INSERT OR IGNORE` no devolvía nada: un asiento colapsado por la clave de dedup
/// equivocada era indistinguible de uno aplicado. Ahora el llamante recibe los dos números y los
/// publica (`inserted` / `ignored`), y cada ignorado deja además una línea de log con su hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FederatedApplyReport {
    /// Asientos nuevos escritos en el ledger.
    pub inserted: usize,
    /// Asientos que ya estaban aplicados (mismo enunciado firmado) y no se duplicaron.
    pub ignored: usize,
}

impl FederatedApplyReport {
    /// Asientos que el lote traía (`inserted + ignored`).
    pub fn total(&self) -> usize {
        self.inserted + self.ignored
    }
}

/// Interface for the Barter economy (Credits).
#[async_trait::async_trait]
pub trait CreditService: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    async fn add_work(
        &self,
        peer_id: &str,
        amount: f32,
        proof_json: Option<String>,
    ) -> Result<crate::LedgerEntry, DomainError>;
    async fn add_work_batch(
        &self,
        batch: Vec<(String, f32, Option<String>)>,
    ) -> Result<(), DomainError>;
    async fn record_consumption(
        &self,
        peer_id: &str,
        amount: f32,
        proof_json: Option<String>,
    ) -> Result<crate::LedgerEntry, DomainError>;
    async fn get_balance(&self, peer_id: &str) -> Result<f32, DomainError>;
    /// Una página del ledger para la **vista de inspección** (`GET /ledger`), lo más reciente primero
    /// (`TASK-42`/S48).
    ///
    /// Devuelve los asientos, no su JSON: serializarlos es del handler, y antes de S48 esta firma era
    /// `Vec<String>` —cada asiento serializado dentro del array—, que es justo lo que el único
    /// consumidor (el CLI de economía) no podía parsear.
    ///
    /// Ordena por `id` **descendente**, así que `limit` corta por el lado que importa: los últimos
    /// movimientos. Antes era ascendente y el endpoint enseñaba el pasado remoto del log.
    async fn get_ledger(&self, limit: u32) -> Result<Vec<crate::LedgerEntry>, DomainError>;
    /// Final del log del ledger: el `id` más alto (0 con el ledger vacío).
    ///
    /// Era un contador en `ledger_meta` que sumaba filas **locales y federadas** y que la federación
    /// usaba como cursor con un `skip(...)`: por eso una réplica se desviaba por cada asiento que ya
    /// había aplicado. Desde `TASK-35`/S41 el cursor de federación es su propio valor
    /// (`SqliteStorage::get_federation_cursor`) y esto solo describe el final del log.
    async fn get_ledger_height(&self) -> Result<u64, DomainError>;

    /// Una página del ledger estrictamente **posterior** a `after_id`, en orden ascendente de `id`.
    ///
    /// Es la primitiva que hace reconstruible la copia (`TASK-35`/S41, SP-02): el llamante recibe
    /// asientos con su `id` y avanza el cursor con el último de la página, así que no puede saltarse
    /// asientos legítimos ni atascarse a partir de `limit` filas.
    async fn get_ledger_page(
        &self,
        after_id: u64,
        limit: u32,
    ) -> Result<Vec<LedgerPageEntry>, DomainError>;
}

/// Reports counted over the `failure_reports` table inside a sliding window (H-11 · `TASK-22`).
///
/// The two numbers answer the two different questions the corroboration policy asks: `by_reporter`
/// is the quota of **one** reporter against one target, and `distinct_reporters` is how many peers
/// **independently** agree that the target failed. An accusation without cryptographic proof only
/// escalates when the second one crosses its threshold, so a single keypair can no longer blacklist
/// anybody (before `TASK-22`, three POSTs from one identity were enough).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FailureReportCounters {
    /// Accepted reports this reporter sent about this target inside the window.
    pub by_reporter: u32,
    /// **Distinct** reporters that reported this target inside the window (corroboration).
    pub distinct_reporters: u32,
}

/// Interface for penalizing bad behavior in the network.
#[async_trait::async_trait]
pub trait SlashingService: Send + Sync {
    async fn record_incident(
        &self,
        peer_id: &str,
        incident: crate::IncidentType,
    ) -> Result<u32, DomainError>;
    async fn slash_peer(
        &self,
        peer_id: &str,
        amount: f32,
        reason: &str,
    ) -> Result<crate::LedgerEntry, DomainError>;

    /// Registers a [`crate::WorkTicket`] the orchestrator has just minted (`TASK-07`, H-12).
    ///
    /// Storing what was issued is what makes the credit authoritative: a proof is only paid for a
    /// `request_id` this orchestrator minted itself, so a worker can no longer invent ids.
    async fn store_work_ticket(&self, ticket: &crate::WorkTicket) -> Result<(), DomainError>;

    /// Consumes a work ticket for `worker_id`, returning what should happen with the credit.
    ///
    /// `credits` is **computed by the caller** from the worker's real assignment (never from the
    /// value the worker sent). Implementations must be idempotent per `(request_id, worker_id)`:
    /// the WAL sweeper retries whole batches, so a second call for the same pair has to report
    /// [`crate::WorkTicketOutcome::Duplicate`] instead of crediting again.
    async fn consume_work_ticket(
        &self,
        ticket: &crate::WorkTicket,
        worker_id: &str,
        model_id: &str,
        credits: f32,
        now: u64,
    ) -> Result<crate::WorkTicketOutcome, DomainError>;

    /// `true` when this reporter may still report this target inside the current window.
    ///
    /// This is the **quota** half of H-11 (`TASK-22`): three accepted reports per
    /// `(reporter, target)` per hour. It is a *soft* cap — two reports accepted concurrently for the
    /// same pair can overshoot it by one — while the escalation decision itself is race-free, because
    /// it is taken from the counters of the very insert it belongs to (see below).
    async fn can_report_failure(
        &self,
        reporter_id: &str,
        target_id: &str,
    ) -> Result<bool, DomainError>;

    /// Records a failure report and returns the counters of the window **including it**
    /// (H-11 · `TASK-22`).
    ///
    /// The insert, the pruning and both counts happen inside a single transaction on purpose: the
    /// policy fires on the **crossing** of a threshold (`==`, never `>=`), and that is only immune to
    /// duplicates if the value it reads is the one this very report produced. With the counters taken
    /// separately, two concurrent reports could both see the threshold reached and escalate twice.
    ///
    /// Rows older than the retention are dropped in the same transaction, so the table cannot grow
    /// without bound while the corroboration window is only one hour wide.
    async fn record_failure_report(
        &self,
        reporter_id: &str,
        target_id: &str,
        window_secs: u64,
    ) -> Result<FailureReportCounters, DomainError>;
    async fn get_score(&self, peer_id: &str) -> Result<crate::ReputationScore, DomainError>;
    async fn update_score(
        &self,
        peer_id: &str,
        event: crate::ScoreEvent,
    ) -> Result<crate::ReputationScore, DomainError>;
    async fn get_all_scores(&self) -> Result<Vec<crate::ReputationScore>, DomainError>;
    async fn upsert_score(&self, score: crate::ReputationScore) -> Result<(), DomainError>;
    async fn upsert_manifest(&self, manifest: crate::WeightManifest) -> Result<(), DomainError>;

    /// Applies a batch of ledger entries that came from a primary (`TASK-34`/S40).
    ///
    /// Deduplication is by the **signed statement** (the `source_hash` of the row), never by
    /// `(peer_id, ts, entry_type)`: `ts` has one-second resolution, so that key collapsed two
    /// legitimate credits of the same peer in the same second, silently (SP-02 · `TASK-35`).
    ///
    /// The returned [`FederatedApplyReport`] is what turns the copy into something an operator can
    /// audit: how many entries were written and how many were already there.
    async fn add_federated_entries(
        &self,
        entries: Vec<crate::SignedLedgerEntry>,
    ) -> Result<FederatedApplyReport, DomainError>;
}

/// Interface for providing model weights (e.g. from disk or network).
pub trait WeightProvider {
    fn get_weights(&self, name: &str) -> Result<Vec<u8>, DomainError>;
}

/// High-level async interface for slashing and blacklisting peers.
/// This is distinct from the broader `SlashingService` which handles the
/// full reputation/scoring pipeline. This trait handles the checksum-mismatch
/// slash path and is wired directly into the tracker runtime.
#[async_trait::async_trait]
pub trait PeerSlashingService: Send + Sync {
    /// Record a slash for the given peer.
    /// Returns the updated slash count. Auto-blacklists at threshold (>= 3).
    async fn slash(&self, peer_id: &str, reason: &str) -> anyhow::Result<u32>;
    /// Returns true if the peer is currently blacklisted.
    async fn is_blacklisted(&self, peer_id: &str) -> bool;
    /// Permanently blacklist a peer regardless of slash count.
    async fn force_blacklist(&self, peer_id: &str) -> anyhow::Result<()>;
}

/// Interface for slashing storage operations.
#[async_trait::async_trait]
pub trait SlashStorage: Send + Sync {
    async fn increment_slash_count(&self, peer_id: &str, reason: &str) -> anyhow::Result<u32>;
    async fn blacklist_node(&self, peer_id: &str) -> anyhow::Result<()>;
}
