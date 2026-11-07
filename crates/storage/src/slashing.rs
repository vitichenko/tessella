use rusqlite::{Connection, OptionalExtension, params};
use std::sync::{Arc, Mutex};
use tessella_core::error::DomainError;
use tessella_core::models::{
    IncidentType, LedgerEntry, LedgerEntryType, ReputationScore, ScoreEvent, SignedLedgerEntry,
    WeightManifest, WorkTicket, WorkTicketOutcome,
};
use tessella_core::traits::{
    FailureReportCounters, FederatedApplyReport, SlashStorage, SlashingService,
};
use tracing::warn;

/// How long an issued work ticket (and its consumptions) is kept after it expires (`TASK-07`).
pub const WORK_TICKET_RETENTION_SECS: u64 = 3600;

/// How long a row of `failure_reports` is kept (H-11 · `TASK-22`).
pub const FAILURE_REPORTS_RETENTION_SECS: u64 = 24 * 3600;

/// Default score penalty factor for slashable incidents.
pub const DEFAULT_SCORE_PENALTY: f64 = 0.5;

pub const SCORE_RECOVERY_PER_WORK: f64 = 0.01;
pub const SCORE_RECOVERY_PER_HOUR: f64 = 0.001;

fn unix_now() -> u64 {
    tessella_core::time::unix_now_secs()
}

/// Store handling peer incidents, reputation scoring, failure corroboration, and slashing.
#[derive(Clone)]
pub struct SqliteSlashingStore {
    conn: Arc<Mutex<Connection>>,
    score_penalty: f64,
}

impl SqliteSlashingStore {
    pub fn new(conn: Arc<Mutex<Connection>>, score_penalty: f64) -> Self {
        Self {
            conn,
            score_penalty,
        }
    }

    pub fn score_penalty(&self) -> f64 {
        self.score_penalty
    }

    pub async fn increment_slash_count(
        &self,
        peer_id: &str,
        reason: &str,
    ) -> Result<u32, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        let reason = reason.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn.transaction().map_err(|e| DomainError::Storage(e.to_string()))?;

            let affected = tx.execute(
                "UPDATE discovered_nodes SET slash_count = slash_count + 1, last_error = ?1, last_seen = CURRENT_TIMESTAMP WHERE peer_id = ?2",
                params![reason, peer_id],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            if affected == 0 {
                return Err(DomainError::Storage(format!("Peer not found: {}", peer_id)));
            }

            tx.commit().map_err(|e| DomainError::Storage(e.to_string()))?;

            let new_count: u32 = conn.query_row(
                "SELECT slash_count FROM discovered_nodes WHERE peer_id = ?1",
                params![peer_id],
                |row| row.get(0),
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            Ok(new_count)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn blacklist_node(&self, peer_id: &str) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let affected = conn.execute(
                "UPDATE discovered_nodes SET blacklisted = 1, status = 'Offline', last_seen = CURRENT_TIMESTAMP WHERE peer_id = ?1",
                params![peer_id],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            if affected == 0 {
                return Err(DomainError::Storage(format!("Peer not found: {}", peer_id)));
            }
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }
}

#[async_trait::async_trait]
impl SlashStorage for SqliteSlashingStore {
    async fn increment_slash_count(&self, peer_id: &str, reason: &str) -> anyhow::Result<u32> {
        self.increment_slash_count(peer_id, reason)
            .await
            .map_err(|e| anyhow::anyhow!(e))
    }

    async fn blacklist_node(&self, peer_id: &str) -> anyhow::Result<()> {
        self.blacklist_node(peer_id)
            .await
            .map_err(|e| anyhow::anyhow!(e))
    }
}

#[async_trait::async_trait]
impl SlashingService for SqliteSlashingStore {
    async fn record_incident(
        &self,
        peer_id: &str,
        incident: IncidentType,
    ) -> Result<u32, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let type_str = format!("{:?}", incident);
            conn.execute(
                "INSERT INTO incidents (peer_id, incident_type) VALUES (?1, ?2)",
                params![peer_id, type_str],
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
            conn.query_row(
                "SELECT COUNT(*) FROM incidents WHERE peer_id = ?1 AND incident_type = ?2",
                params![peer_id, type_str],
                |row| row.get(0),
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn slash_peer(
        &self,
        peer_id: &str,
        amount: f32,
        reason: &str,
    ) -> Result<LedgerEntry, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        let reason = reason.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let now = unix_now();
            tx.execute(
                "UPDATE peer_credits SET credits_spent = credits_spent + ?2 WHERE peer_id = ?1",
                params![peer_id, amount],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            tx.execute("INSERT INTO ledger (peer_id, entry_type, amount, proof, source, ts) VALUES (?1, ?2, ?3, ?4, 'local', ?5)", params![peer_id, LedgerEntryType::Slashed.to_db(), amount, reason, now]).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            tx.execute(
                "UPDATE discovered_nodes SET status = 'Slashed' WHERE peer_id = ?1",
                params![peer_id],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok(LedgerEntry {
                peer_id,
                entry_type: LedgerEntryType::Slashed,
                amount,
                proof: reason,
                timestamp: now,
                source: "local".to_string(),
            })
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn store_work_ticket(&self, ticket: &WorkTicket) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        let request_id = ticket.request_id.clone();
        let model_id = ticket.model_id.clone();
        let entry_peer_id = ticket.entry_peer_id.clone();
        let issued_at = i64::try_from(ticket.issued_at).unwrap_or(i64::MAX);
        let expires_at = i64::try_from(ticket.expires_at).unwrap_or(i64::MAX);
        let now = unix_now();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "INSERT OR IGNORE INTO work_tickets
                     (request_id, model_id, entry_peer_id, issued_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![request_id, model_id, entry_peer_id, issued_at, expires_at],
            )
            .map_err(|e| DomainError::Storage(e.to_string()))?;

            let horizon = now.saturating_sub(WORK_TICKET_RETENTION_SECS) as i64;
            conn.execute(
                "DELETE FROM work_tickets WHERE expires_at < ?1",
                params![horizon],
            )
            .map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "DELETE FROM work_ticket_consumptions
                 WHERE request_id NOT IN (SELECT request_id FROM work_tickets)",
                [],
            )
            .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn consume_work_ticket(
        &self,
        ticket: &WorkTicket,
        worker_id: &str,
        model_id: &str,
        credits: f32,
        now: u64,
    ) -> Result<WorkTicketOutcome, DomainError> {
        let conn = self.conn.clone();
        let request_id = ticket.request_id.clone();
        let worker_id = worker_id.to_string();
        let model_id = model_id.to_string();
        let now_i64 = i64::try_from(now).unwrap_or(i64::MAX);
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let issued: Option<(String, i64)> = conn
                .query_row(
                    "SELECT model_id, expires_at FROM work_tickets WHERE request_id = ?1",
                    params![request_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let Some((issued_model, expires_at)) = issued else {
                return Ok(WorkTicketOutcome::Unknown);
            };
            let expires_at = u64::try_from(expires_at).unwrap_or(u64::MAX);
            if now > expires_at {
                return Ok(WorkTicketOutcome::Expired { expires_at });
            }
            if issued_model != model_id {
                return Ok(WorkTicketOutcome::ModelMismatch {
                    expected: issued_model,
                    got: model_id,
                });
            }

            let inserted = conn
                .execute(
                    "INSERT OR IGNORE INTO work_ticket_consumptions
                         (request_id, worker_id, credits, consumed_at)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![request_id, worker_id, f64::from(credits), now_i64],
                )
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            if inserted == 0 {
                Ok(WorkTicketOutcome::Duplicate { credits })
            } else {
                Ok(WorkTicketOutcome::Accepted { credits })
            }
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn can_report_failure(
        &self,
        reporter_id: &str,
        target_id: &str,
    ) -> Result<bool, DomainError> {
        let conn = self.conn.clone();
        let reporter_id = reporter_id.to_string();
        let target_id = target_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let count: u32 = conn.query_row(
                "SELECT COUNT(*) FROM failure_reports WHERE reporter_id = ?1 AND target_id = ?2 AND timestamp > datetime('now', '-1 hour')",
                params![reporter_id, target_id],
                |row| row.get(0),
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(count < 3)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn record_failure_report(
        &self,
        reporter_id: &str,
        target_id: &str,
        window_secs: u64,
    ) -> Result<FailureReportCounters, DomainError> {
        let conn = self.conn.clone();
        let reporter_id = reporter_id.to_string();
        let target_id = target_id.to_string();
        let window = format!("-{} seconds", window_secs);
        let retention = format!("-{} seconds", FAILURE_REPORTS_RETENTION_SECS);
        tokio::task::spawn_blocking(move || {
            let mut conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            tx.execute(
                "INSERT INTO failure_reports (reporter_id, target_id) VALUES (?1, ?2)",
                params![reporter_id, target_id],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            tx.execute(
                "DELETE FROM failure_reports WHERE timestamp < datetime('now', ?1)",
                params![retention],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            let counters = tx.query_row(
                "SELECT
                     (SELECT COUNT(*) FROM failure_reports
                        WHERE reporter_id = ?1 AND target_id = ?2 AND timestamp > datetime('now', ?3)),
                     (SELECT COUNT(DISTINCT reporter_id) FROM failure_reports
                        WHERE target_id = ?2 AND timestamp > datetime('now', ?3))",
                params![reporter_id, target_id, window],
                |row| {
                    Ok(FailureReportCounters {
                        by_reporter: row.get(0)?,
                        distinct_reporters: row.get(1)?,
                    })
                },
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            Ok(counters)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn get_score(&self, peer_id: &str) -> Result<ReputationScore, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let now = unix_now();
            let res: Result<(f64, u64), rusqlite::Error> = conn.query_row(
                "SELECT score, last_updated FROM reputation_scores WHERE peer_id = ?1",
                params![peer_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            );
            match res {
                Ok((mut score, last_updated)) => {
                    let elapsed_hours = (now.saturating_sub(last_updated) / 3600) as f64;
                    if elapsed_hours >= 1.0 {
                        score = (score + elapsed_hours * SCORE_RECOVERY_PER_HOUR).min(1.0);
                        if let Err(e) = conn.execute(
                            "UPDATE reputation_scores SET score = ?1, last_updated = ?2 WHERE peer_id = ?3",
                            params![score, now, peer_id],
                        ) {
                            warn!("⚠️ Could not persist the reputation recovery of {peer_id}: {e}");
                        }
                    }
                    Ok(ReputationScore {
                        peer_id,
                        score,
                        last_updated: now,
                    })
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => {
                    if let Err(e) = conn.execute(
                        "INSERT OR IGNORE INTO reputation_scores (peer_id, score, last_updated) VALUES (?1, 1.0, ?2)",
                        params![peer_id, now],
                    ) {
                        warn!("⚠️ Could not record the first reputation score of {peer_id}: {e}");
                    }
                    Ok(ReputationScore {
                        peer_id,
                        score: 1.0,
                        last_updated: now,
                    })
                }
                Err(e) => Err(DomainError::Storage(e.to_string())),
            }
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn update_score(
        &self,
        peer_id: &str,
        event: ScoreEvent,
    ) -> Result<ReputationScore, DomainError> {
        let score_obj = self.get_score(peer_id).await?;
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        let score_penalty = self.score_penalty;
        tokio::task::spawn_blocking(move || {
            let mut score_obj = score_obj;
            match event {
                ScoreEvent::IncidentRecorded(incident) => match incident {
                    IncidentType::InvalidSignature
                    | IncidentType::ExcessiveFailures
                    | IncidentType::UncorroboratedWork => {
                        score_obj.score *= score_penalty;
                    }
                    IncidentType::ConflictingProof | IncidentType::WeightPoisoning => {
                        score_obj.score = 0.0;
                    }
                },
                ScoreEvent::WorkProofAccepted => {
                    score_obj.score = (score_obj.score + SCORE_RECOVERY_PER_WORK).min(1.0)
                }
            }
            let now = unix_now();
            score_obj.last_updated = now;
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "UPDATE reputation_scores SET score = ?1, last_updated = ?2 WHERE peer_id = ?3",
                params![score_obj.score, now, peer_id],
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
            Ok(score_obj)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn get_all_scores(&self) -> Result<Vec<ReputationScore>, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn
                .prepare("SELECT peer_id, score, last_updated FROM reputation_scores")
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            let iter = stmt
                .query_map([], |row| {
                    Ok(ReputationScore {
                        peer_id: row.get(0)?,
                        score: row.get(1)?,
                        last_updated: row.get(2)?,
                    })
                })
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            Ok(iter.flatten().collect())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn upsert_score(&self, score: ReputationScore) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute("INSERT INTO reputation_scores (peer_id, score, last_updated) VALUES (?1, ?2, ?3) ON CONFLICT(peer_id) DO UPDATE SET score = ?2, last_updated = ?3", params![score.peer_id, score.score, score.last_updated]).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn upsert_manifest(&self, manifest: WeightManifest) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let role_json = serde_json::to_string(&manifest.role)
                .map_err(|e| DomainError::Validation(e.to_string()))?;
            conn.execute(
                "INSERT INTO weight_manifests (model_id, filename, role, blake3_hash, size_bytes, registered_at, tracker_public_key, tracker_signature) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                 ON CONFLICT(model_id, role) DO UPDATE SET \
                 filename = ?2, blake3_hash = ?4, size_bytes = ?5, registered_at = ?6, \
                 tracker_public_key = ?7, tracker_signature = ?8",
                params![
                    manifest.model_id,
                    manifest.filename,
                    role_json,
                    manifest.blake3_hash,
                    manifest.size_bytes,
                    manifest.registered_at,
                    manifest.tracker_public_key,
                    manifest.tracker_signature,
                ],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn add_federated_entries(
        &self,
        entries: Vec<SignedLedgerEntry>,
    ) -> Result<FederatedApplyReport, DomainError> {
        crate::credits::SqliteCreditStore::new(self.conn.clone())
            .add_federated_entries(entries)
            .await
    }
}
