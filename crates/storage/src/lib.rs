pub mod credits;
pub mod error;
pub mod manifests;
pub mod migrations;
pub mod nodes;
pub mod slashing;
pub mod wal;

#[cfg(test)]
mod tests;

use anyhow::Result;
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tessella_core::error::DomainError;
use tessella_core::models::{
    IncidentType, LayerPeer, LedgerEntry, NodeInfo, ReputationScore, ScoreEvent, SignedLedgerEntry,
    SignedTransferReceipt, SignedWorkProof, TransferReceipt, WeightManifest, WorkProof, WorkTicket,
    WorkTicketOutcome, WorkerAssignment,
};
use tessella_core::traits::{
    CreditService, FailureReportCounters, FederatedApplyReport, LedgerPageEntry, SlashStorage,
    SlashingService, StorageService, WeightProvider,
};

pub use credits::{SqliteCreditStore, federated_source_hash, parse_ledger_entry_type};
pub use manifests::SqliteManifestStore;
pub use migrations::{
    LEDGER_DEDUP_INDEX, SQLITE_BUSY_TIMEOUT_MS, column_declaration, declared_type_in,
    ensure_column, ensure_column_renamed, ensure_ledger_dedup_index, run_migrations,
};
pub use nodes::{KNOWN_PEERS_MAX_ROWS, KNOWN_PEERS_TTL_SECS, SqliteNodeStore};
pub use slashing::{
    DEFAULT_SCORE_PENALTY, FAILURE_REPORTS_RETENTION_SECS, SCORE_RECOVERY_PER_HOUR,
    SCORE_RECOVERY_PER_WORK, SqliteSlashingStore, WORK_TICKET_RETENTION_SECS,
};
pub use wal::{WAL_CHECKPOINT_INTERVAL_SECS, WalCheckpoint, WalMaintenance};

pub fn unix_now() -> u64 {
    tessella_core::time::unix_now_secs()
}

/// Composite storage facade coordinating modular SQLite sub-stores.
///
/// Implements [`CreditService`], [`SlashingService`], [`SlashStorage`], [`StorageService`],
/// and [`WeightProvider`] by delegating to specialized domain stores:
/// - [`SqliteCreditStore`]: barter credits, ledger accounting, and federated replication.
/// - [`SqliteSlashingStore`]: peer incidents, reputation scoring, and slash tracking.
/// - [`SqliteManifestStore`]: weight manifests, receipts, and disk safetensors reading.
/// - [`SqliteNodeStore`]: mesh discovery and persistent peer cache.
#[derive(Clone)]
pub struct SqliteStorage {
    conn: Arc<Mutex<Connection>>,
    pub weights_base_path: PathBuf,
    score_penalty: f64,
    pub credits: SqliteCreditStore,
    pub slashing: SqliteSlashingStore,
    pub manifests: SqliteManifestStore,
    pub nodes: SqliteNodeStore,
    pub metrics: Option<Arc<tessella_core::telemetry::WorkerMetrics>>,
}

impl SqliteStorage {
    /// Opens (and migrates) the database with the default reputation policy ([`DEFAULT_SCORE_PENALTY`]).
    pub fn new(
        path: impl AsRef<Path>,
        weights_base_path: Option<PathBuf>,
    ) -> Result<Self, DomainError> {
        Self::with_slash_penalty(path, weights_base_path, DEFAULT_SCORE_PENALTY)
    }

    /// Same as [`Self::new`] with an explicit reputation policy multiplier.
    pub fn with_slash_penalty(
        path: impl AsRef<Path>,
        weights_base_path: Option<PathBuf>,
        slash_penalty: f64,
    ) -> Result<Self, DomainError> {
        let db_path = path.as_ref().to_path_buf();
        let conn = Connection::open(path).map_err(|e| DomainError::Storage(e.to_string()))?;
        migrations::run_migrations(&conn)?;

        let base_path = weights_base_path
            .unwrap_or_else(|| db_path.parent().unwrap_or(Path::new("")).to_path_buf());

        let conn_arc = Arc::new(Mutex::new(conn));
        let credits = SqliteCreditStore::new(conn_arc.clone());
        let slashing = SqliteSlashingStore::new(conn_arc.clone(), slash_penalty);
        let manifests = SqliteManifestStore::new(conn_arc.clone(), base_path.clone());
        let nodes = SqliteNodeStore::new(conn_arc.clone());

        Ok(Self {
            conn: conn_arc,
            weights_base_path: base_path,
            score_penalty: slash_penalty,
            credits,
            slashing,
            manifests,
            nodes,
            metrics: None,
        })
    }

    pub fn with_metrics(mut self, metrics: Arc<tessella_core::telemetry::WorkerMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn set_metrics(&mut self, metrics: Arc<tessella_core::telemetry::WorkerMetrics>) {
        self.metrics = Some(metrics);
    }

    pub fn score_penalty(&self) -> f64 {
        self.score_penalty
    }

    /// Consolidates the write-ahead log into the main database file and truncates it.
    pub async fn checkpoint_wal(&self) -> Result<WalCheckpoint, DomainError> {
        wal::checkpoint_wal_conn(self.conn.clone()).await
    }

    pub async fn increment_slash_count(
        &self,
        peer_id: &str,
        reason: &str,
    ) -> Result<u32, DomainError> {
        self.slashing.increment_slash_count(peer_id, reason).await
    }

    pub async fn blacklist_node(&self, peer_id: &str) -> Result<(), DomainError> {
        self.slashing.blacklist_node(peer_id).await
    }

    pub async fn save_pending_work_proof(
        &self,
        proof: &SignedWorkProof,
    ) -> Result<i64, DomainError> {
        let res = self.nodes.save_pending_work_proof(proof).await;
        if res.is_ok()
            && let Some(metrics) = &self.metrics
        {
            metrics.inc_proofs_queued(1);
        }
        res
    }

    pub async fn count_pending_work_proofs(&self) -> Result<u64, DomainError> {
        self.nodes.count_pending_work_proofs().await
    }

    pub async fn get_pending_work_proofs(
        &self,
    ) -> Result<Vec<(i64, SignedWorkProof)>, DomainError> {
        self.nodes.get_pending_work_proofs().await
    }

    pub async fn delete_pending_work_proofs(&self, ids: Vec<i64>) -> Result<(), DomainError> {
        self.nodes.delete_pending_work_proofs(ids).await
    }

    pub async fn remember_known_peers(
        &self,
        peers: &[LayerPeer],
        now: u64,
    ) -> Result<usize, DomainError> {
        self.nodes.remember_known_peers(peers, now).await
    }

    pub async fn load_known_peers(
        &self,
        now: u64,
        ttl_secs: u64,
    ) -> Result<Vec<LayerPeer>, DomainError> {
        self.nodes.load_known_peers(now, ttl_secs).await
    }

    pub async fn get_federation_cursor(&self) -> Result<u64, DomainError> {
        self.credits.get_federation_cursor().await
    }

    pub async fn set_federation_cursor(&self, cursor: u64) -> Result<(), DomainError> {
        self.credits.set_federation_cursor(cursor).await
    }

    pub async fn get_assignment_version(&self) -> Result<u64, DomainError> {
        self.credits.get_assignment_version().await
    }

    pub fn get_assignment_version_sync(&self) -> Result<u64, DomainError> {
        self.credits.get_assignment_version_sync()
    }

    pub async fn set_assignment_version(&self, version: u64) -> Result<(), DomainError> {
        self.credits.set_assignment_version(version).await
    }

    pub async fn get_active_assignment(&self) -> Result<Option<WorkerAssignment>, DomainError> {
        self.credits.get_active_assignment().await
    }

    pub fn get_active_assignment_sync(&self) -> Result<Option<WorkerAssignment>, DomainError> {
        self.credits.get_active_assignment_sync()
    }

    pub async fn set_active_assignment(
        &self,
        version: u64,
        assignment: &WorkerAssignment,
    ) -> Result<(), DomainError> {
        self.credits
            .set_active_assignment(version, assignment)
            .await
    }

    pub async fn update_karma(&self, peer_id: &str, balance: i64) -> Result<(), DomainError> {
        self.credits.update_karma(peer_id, balance).await
    }

    pub async fn get_all_karma(&self) -> Result<Vec<(String, i64)>, DomainError> {
        self.credits.get_all_karma().await
    }

    pub async fn save_manifest(&self, manifest: WeightManifest) -> Result<(), DomainError> {
        self.manifests.save_manifest(manifest).await
    }

    pub async fn get_manifest(
        &self,
        model_id: &str,
        role_json: &str,
    ) -> Result<Option<WeightManifest>, DomainError> {
        self.manifests.get_manifest(model_id, role_json).await
    }

    pub async fn get_manifests_for_model(
        &self,
        model_id: &str,
    ) -> Result<Vec<WeightManifest>, DomainError> {
        self.manifests.get_manifests_for_model(model_id).await
    }

    pub async fn list_manifests(&self) -> Result<Vec<WeightManifest>, DomainError> {
        self.manifests.list_manifests().await
    }

    pub async fn save_receipt(&self, signed: &SignedTransferReceipt) -> Result<(), DomainError> {
        self.manifests.save_receipt(signed).await
    }

    pub async fn get_receipt(
        &self,
        sender_peer_id: &str,
        receiver_peer_id: &str,
        model_id: &str,
        role: &str,
    ) -> Result<Option<SignedTransferReceipt>, DomainError> {
        self.manifests
            .get_receipt(sender_peer_id, receiver_peer_id, model_id, role)
            .await
    }

    pub async fn list_receipts(&self, peer_id: &str) -> Result<Vec<TransferReceipt>, DomainError> {
        self.manifests.list_receipts(peer_id).await
    }

    pub async fn store_work_ticket(&self, ticket: &WorkTicket) -> Result<(), DomainError> {
        self.slashing.store_work_ticket(ticket).await
    }

    pub async fn consume_work_ticket(
        &self,
        ticket: &WorkTicket,
        worker_id: &str,
        model_id: &str,
        credits: f32,
        now: u64,
    ) -> Result<WorkTicketOutcome, DomainError> {
        self.slashing
            .consume_work_ticket(ticket, worker_id, model_id, credits, now)
            .await
    }

    pub async fn can_report_failure(
        &self,
        reporter_id: &str,
        target_id: &str,
    ) -> Result<bool, DomainError> {
        self.slashing
            .can_report_failure(reporter_id, target_id)
            .await
    }

    pub async fn record_failure_report(
        &self,
        reporter_id: &str,
        target_id: &str,
        window_secs: u64,
    ) -> Result<FailureReportCounters, DomainError> {
        self.slashing
            .record_failure_report(reporter_id, target_id, window_secs)
            .await
    }

    pub async fn get_score(&self, peer_id: &str) -> Result<ReputationScore, DomainError> {
        self.slashing.get_score(peer_id).await
    }

    pub async fn update_score(
        &self,
        peer_id: &str,
        event: ScoreEvent,
    ) -> Result<ReputationScore, DomainError> {
        self.slashing.update_score(peer_id, event).await
    }

    pub async fn get_all_scores(&self) -> Result<Vec<ReputationScore>, DomainError> {
        self.slashing.get_all_scores().await
    }

    pub async fn upsert_score(&self, score: ReputationScore) -> Result<(), DomainError> {
        self.slashing.upsert_score(score).await
    }

    pub async fn upsert_manifest(&self, manifest: WeightManifest) -> Result<(), DomainError> {
        self.manifests.upsert_manifest(manifest).await
    }

    pub async fn add_federated_entries(
        &self,
        entries: Vec<SignedLedgerEntry>,
    ) -> Result<FederatedApplyReport, DomainError> {
        self.credits.add_federated_entries(entries).await
    }
}

#[async_trait::async_trait]
impl CreditService for SqliteStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn add_work(
        &self,
        peer_id: &str,
        amount: f32,
        proof_json: Option<String>,
    ) -> Result<LedgerEntry, DomainError> {
        self.credits.add_work(peer_id, amount, proof_json).await
    }

    async fn add_work_batch(
        &self,
        batch: Vec<(String, f32, Option<String>)>,
    ) -> Result<(), DomainError> {
        self.credits.add_work_batch(batch).await
    }

    async fn record_consumption(
        &self,
        peer_id: &str,
        amount: f32,
        proof_json: Option<String>,
    ) -> Result<LedgerEntry, DomainError> {
        self.credits
            .record_consumption(peer_id, amount, proof_json)
            .await
    }

    async fn get_balance(&self, peer_id: &str) -> Result<f32, DomainError> {
        self.credits.get_balance(peer_id).await
    }

    async fn get_ledger(&self, limit: u32) -> Result<Vec<LedgerEntry>, DomainError> {
        self.credits.get_ledger(limit).await
    }

    async fn get_ledger_height(&self) -> Result<u64, DomainError> {
        self.credits.get_ledger_height().await
    }

    async fn get_ledger_page(
        &self,
        after_id: u64,
        limit: u32,
    ) -> Result<Vec<LedgerPageEntry>, DomainError> {
        self.credits.get_ledger_page(after_id, limit).await
    }
}

#[async_trait::async_trait]
impl SlashStorage for SqliteStorage {
    async fn increment_slash_count(&self, peer_id: &str, reason: &str) -> anyhow::Result<u32> {
        SlashStorage::increment_slash_count(&self.slashing, peer_id, reason).await
    }

    async fn blacklist_node(&self, peer_id: &str) -> anyhow::Result<()> {
        SlashStorage::blacklist_node(&self.slashing, peer_id).await
    }
}

#[async_trait::async_trait]
impl SlashingService for SqliteStorage {
    async fn record_incident(
        &self,
        peer_id: &str,
        incident: IncidentType,
    ) -> Result<u32, DomainError> {
        self.slashing.record_incident(peer_id, incident).await
    }

    async fn slash_peer(
        &self,
        peer_id: &str,
        amount: f32,
        reason: &str,
    ) -> Result<LedgerEntry, DomainError> {
        self.slashing.slash_peer(peer_id, amount, reason).await
    }

    async fn store_work_ticket(&self, ticket: &WorkTicket) -> Result<(), DomainError> {
        self.slashing.store_work_ticket(ticket).await
    }

    async fn consume_work_ticket(
        &self,
        ticket: &WorkTicket,
        worker_id: &str,
        model_id: &str,
        credits: f32,
        now: u64,
    ) -> Result<WorkTicketOutcome, DomainError> {
        self.slashing
            .consume_work_ticket(ticket, worker_id, model_id, credits, now)
            .await
    }

    async fn can_report_failure(
        &self,
        reporter_id: &str,
        target_id: &str,
    ) -> Result<bool, DomainError> {
        self.slashing
            .can_report_failure(reporter_id, target_id)
            .await
    }

    async fn record_failure_report(
        &self,
        reporter_id: &str,
        target_id: &str,
        window_secs: u64,
    ) -> Result<FailureReportCounters, DomainError> {
        self.slashing
            .record_failure_report(reporter_id, target_id, window_secs)
            .await
    }

    async fn get_score(&self, peer_id: &str) -> Result<ReputationScore, DomainError> {
        self.slashing.get_score(peer_id).await
    }

    async fn update_score(
        &self,
        peer_id: &str,
        event: ScoreEvent,
    ) -> Result<ReputationScore, DomainError> {
        self.slashing.update_score(peer_id, event).await
    }

    async fn get_all_scores(&self) -> Result<Vec<ReputationScore>, DomainError> {
        self.slashing.get_all_scores().await
    }

    async fn upsert_score(&self, score: ReputationScore) -> Result<(), DomainError> {
        self.slashing.upsert_score(score).await
    }

    async fn upsert_manifest(&self, manifest: WeightManifest) -> Result<(), DomainError> {
        self.manifests.upsert_manifest(manifest).await
    }

    async fn add_federated_entries(
        &self,
        entries: Vec<SignedLedgerEntry>,
    ) -> Result<FederatedApplyReport, DomainError> {
        self.credits.add_federated_entries(entries).await
    }
}

#[async_trait::async_trait]
impl StorageService for SqliteStorage {
    async fn get_session_work_proofs(
        &self,
        request_id: &str,
    ) -> Result<Vec<WorkProof>, DomainError> {
        self.credits.get_session_work_proofs(request_id).await
    }

    async fn save_node(&self, node: NodeInfo) -> Result<(), DomainError> {
        self.nodes.save_node(node).await
    }

    async fn get_nodes(&self) -> Result<Vec<NodeInfo>, DomainError> {
        self.nodes.get_nodes().await
    }

    async fn get_all_nodes(&self) -> Result<Vec<NodeInfo>, DomainError> {
        self.nodes.get_all_nodes().await
    }

    async fn deregister_node(&self, peer_id: &str) -> Result<(), DomainError> {
        self.nodes.deregister_node(peer_id).await
    }

    async fn save_manifest(&self, manifest: WeightManifest) -> Result<(), DomainError> {
        self.manifests.save_manifest(manifest).await
    }

    async fn get_manifest(
        &self,
        model_id: &str,
        role_json: &str,
    ) -> Result<Option<WeightManifest>, DomainError> {
        self.manifests.get_manifest(model_id, role_json).await
    }

    async fn get_manifests_for_model(
        &self,
        model_id: &str,
    ) -> Result<Vec<WeightManifest>, DomainError> {
        self.manifests.get_manifests_for_model(model_id).await
    }

    async fn list_manifests(&self) -> Result<Vec<WeightManifest>, DomainError> {
        self.manifests.list_manifests().await
    }

    async fn save_receipt(&self, receipt: &SignedTransferReceipt) -> Result<(), DomainError> {
        self.manifests.save_receipt(receipt).await
    }

    async fn get_receipt(
        &self,
        sender_peer_id: &str,
        receiver_peer_id: &str,
        model_id: &str,
        role: &str,
    ) -> Result<Option<SignedTransferReceipt>, DomainError> {
        self.manifests
            .get_receipt(sender_peer_id, receiver_peer_id, model_id, role)
            .await
    }

    async fn list_receipts(&self, peer_id: &str) -> Result<Vec<TransferReceipt>, DomainError> {
        self.manifests.list_receipts(peer_id).await
    }

    async fn update_karma(&self, peer_id: &str, balance: i64) -> Result<(), DomainError> {
        self.credits.update_karma(peer_id, balance).await
    }

    async fn get_all_karma(&self) -> Result<Vec<(String, i64)>, DomainError> {
        self.credits.get_all_karma().await
    }

    async fn count_pending_work_proofs(&self) -> Result<u64, DomainError> {
        self.count_pending_work_proofs().await
    }

    async fn get_assignment_version(&self) -> Result<u64, DomainError> {
        self.get_assignment_version().await
    }

    async fn set_assignment_version(&self, version: u64) -> Result<(), DomainError> {
        self.set_assignment_version(version).await
    }

    async fn get_active_assignment(&self) -> Result<Option<WorkerAssignment>, DomainError> {
        self.get_active_assignment().await
    }

    async fn set_active_assignment(
        &self,
        version: u64,
        assignment: &WorkerAssignment,
    ) -> Result<(), DomainError> {
        self.set_active_assignment(version, assignment).await
    }
}

impl WeightProvider for SqliteStorage {
    fn get_weights(&self, name: &str) -> Result<Vec<u8>, DomainError> {
        self.manifests.get_weights(name)
    }
}
