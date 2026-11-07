use rusqlite::{Connection, params};
use std::sync::{Arc, Mutex};
use tessella_core::WorkerAssignment;
use tessella_core::error::DomainError;
use tessella_core::models::{
    LedgerEntry, LedgerEntryType, SignedLedgerEntry, SignedWorkProof, WorkProof,
};
use tessella_core::traits::{CreditService, FederatedApplyReport, LedgerPageEntry};
use tracing::warn;

/// Deduplication key of a federated entry: blake3 over the **signed statement**
/// (`canonical JSON of the entry ‖ signature`).
///
/// It is what replaces the old `(peer_id, ts, entry_type, source)` key (SP-02 · `TASK-35`). Ed25519
/// is deterministic (RFC 8032), so the same statement issued by the same orchestrator always carries
/// the same signature — and two *different* credits of the same peer in the same second carry
/// different ones, which is exactly what the old key could not tell apart. The signature is included
/// so that the same entry signed by two primaries stays two distinct rows.
pub fn federated_source_hash(entry: &SignedLedgerEntry) -> Result<String, DomainError> {
    let canonical = entry.entry.to_canonical_bytes()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&canonical);
    hasher.update(&entry.signature);
    Ok(hex::encode(hasher.finalize().as_bytes()))
}

/// Reads `entry_type` from a row (`TASK-42`/S48).
///
/// The conversion itself lives in [`LedgerEntryType::from_db`] —the protocol owns the text of its own
/// variants and the compiler checks that `match`—, so what stays here is the **policy** for a text this
/// build does not know: keep the historical fallback (`Spent`) and say it. The log line is what turns
/// «a value I cannot represent» into something an operator can act on; the fallback is deliberately not
/// changed here (that would be a different decision, and one nobody has measured).
pub fn parse_ledger_entry_type(raw: &str) -> LedgerEntryType {
    LedgerEntryType::from_db(raw).unwrap_or_else(|| {
        warn!(
            "Unknown LedgerEntryType in the ledger table: '{}', defaulting to Spent",
            raw
        );
        LedgerEntryType::Spent
    })
}

fn unix_now() -> u64 {
    tessella_core::time::unix_now_secs()
}

/// Store handling barter credits, ledger accounting, karma tracking, and federated replication.
#[derive(Clone)]
pub struct SqliteCreditStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteCreditStore {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// The primary ledger `id` this node has applied as a replica up to (`TASK-35`/S41).
    pub async fn get_federation_cursor(&self) -> Result<u64, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let cursor: i64 = conn
                .query_row(
                    "SELECT COALESCE((SELECT value FROM ledger_meta WHERE key = 'federation_cursor'), 0)",
                    [],
                    |row| row.get(0),
                )
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            Ok(cursor.max(0) as u64)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    /// Advances the federation cursor.
    pub async fn set_federation_cursor(&self, cursor: u64) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "INSERT INTO ledger_meta (key, value) VALUES ('federation_cursor', ?1) \
                 ON CONFLICT(key) DO UPDATE SET value = ?1",
                params![cursor as i64],
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    /// Returns the monotonic version of the active worker shard assignment (`MSV-19`).
    pub async fn get_assignment_version(&self) -> Result<u64, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let version: i64 = conn
                .query_row(
                    "SELECT COALESCE((SELECT value FROM ledger_meta WHERE key = 'worker_assignment_version'), 0)",
                    [],
                    |row| row.get(0),
                )
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            Ok(version.max(0) as u64)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    /// Synchronous read of the monotonic assignment version for boot initialization (`MSV-19`).
    pub fn get_assignment_version_sync(&self) -> Result<u64, DomainError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| DomainError::Storage(e.to_string()))?;
        let version: i64 = conn
            .query_row(
                "SELECT COALESCE((SELECT value FROM ledger_meta WHERE key = 'worker_assignment_version'), 0)",
                [],
                |row| row.get(0),
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
        Ok(version.max(0) as u64)
    }

    /// Advances the monotonic assignment version (`MSV-19`).
    pub async fn set_assignment_version(&self, version: u64) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "INSERT INTO ledger_meta (key, value) VALUES ('worker_assignment_version', ?1) \
                 ON CONFLICT(key) DO UPDATE SET value = MAX(value, ?1)",
                params![version as i64],
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    fn query_active_assignment(conn: &Connection) -> Result<Option<WorkerAssignment>, DomainError> {
        let mut stmt = conn
            .prepare("SELECT assignment_json FROM active_worker_assignment WHERE id = 1")
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
        let mut rows = stmt.query([]).map_err(|e| {
            tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
        })?;

        if let Some(row) = rows.next().map_err(|e| {
            tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
        })? {
            let json_str: String = row.get(0).map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
            let assignment = serde_json::from_str::<WorkerAssignment>(&json_str).map_err(|e| {
                DomainError::Storage(format!(
                    "Failed to deserialize active WorkerAssignment: {}",
                    e
                ))
            })?;
            Ok(Some(assignment))
        } else {
            Ok(None)
        }
    }

    /// Returns the last persisted active worker shard assignment, if any (`MSV-20`).
    pub async fn get_active_assignment(&self) -> Result<Option<WorkerAssignment>, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            Self::query_active_assignment(&conn)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    /// Synchronous read of the active worker shard assignment for boot initialization (`MSV-20`).
    pub fn get_active_assignment_sync(&self) -> Result<Option<WorkerAssignment>, DomainError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| DomainError::Storage(e.to_string()))?;
        Self::query_active_assignment(&conn)
    }

    /// Persists the active worker shard assignment alongside its monotonic version (`MSV-20`).
    pub async fn set_active_assignment(
        &self,
        version: u64,
        assignment: &WorkerAssignment,
    ) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        let json_str = serde_json::to_string(assignment).map_err(|e| {
            DomainError::Storage(format!("Failed to serialize WorkerAssignment: {}", e))
        })?;
        let now = tessella_core::time::unix_now_secs() as i64;
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "INSERT INTO active_worker_assignment (id, version, assignment_json, updated_at) \
                 VALUES (1, ?1, ?2, ?3) \
                 ON CONFLICT(id) DO UPDATE SET \
                    version = ?1, \
                    assignment_json = ?2, \
                    updated_at = ?3 \
                 WHERE ?1 >= version",
                params![version as i64, json_str, now],
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
            conn.execute(
                "INSERT INTO ledger_meta (key, value) VALUES ('worker_assignment_version', ?1) \
                 ON CONFLICT(key) DO UPDATE SET value = MAX(value, ?1)",
                params![version as i64],
            )
            .map_err(|e| {
                tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    /// Applies a batch of ledger entries that came from a primary (`TASK-34`/S40).
    pub async fn add_federated_entries(
        &self,
        entries: Vec<SignedLedgerEntry>,
    ) -> Result<FederatedApplyReport, DomainError> {
        let conn = self.conn.clone();
        let outcome: (usize, Vec<String>) = tokio::task::spawn_blocking(move || {
            let mut conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let mut inserted = 0usize;
            let mut ignored_hashes: Vec<String> = Vec::new();

            for signed in entries {
                let hash = federated_source_hash(&signed)?;
                let entry = &signed.entry;
                let changes = tx
                    .execute(
                        "INSERT OR IGNORE INTO ledger \
                     (peer_id, entry_type, amount, proof, source, source_hash, ts) \
                     VALUES (?1, ?2, ?3, ?4, 'federated', ?5, ?6)",
                        params![
                            entry.peer_id,
                            entry.entry_type.to_db(),
                            entry.amount,
                            entry.proof,
                            hash,
                            entry.timestamp
                        ],
                    )
                    .map_err(|e| {
                        tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(
                            e,
                        ))
                    })?;
                if changes > 0 {
                    inserted += 1;
                } else {
                    ignored_hashes.push(hash);
                }
            }

            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok::<_, DomainError>((inserted, ignored_hashes))
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))??;

        let (inserted, ignored_hashes) = outcome;
        for hash in &ignored_hashes {
            warn!("♻️ Federated ledger entry already applied, not duplicated (source_hash={hash})");
        }

        Ok(FederatedApplyReport {
            inserted,
            ignored: ignored_hashes.len(),
        })
    }

    /// Updates the persistent Karma balance of a peer in `karma_ledger`.
    pub async fn update_karma(&self, peer_id: &str, balance: i64) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(format!("DB lock poisoned: {}", e)))?;
            conn.execute(
                "INSERT INTO karma_ledger (peer_id, balance) VALUES (?1, ?2)
                 ON CONFLICT(peer_id) DO UPDATE SET balance = ?2",
                params![peer_id, balance],
            )
            .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    /// Retrieves all `(peer_id, balance)` pairs from `karma_ledger`.
    pub async fn get_all_karma(&self) -> Result<Vec<(String, i64)>, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(format!("DB lock poisoned: {}", e)))?;
            let mut stmt = conn
                .prepare("SELECT peer_id, balance FROM karma_ledger")
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let rows = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let mut results = Vec::new();
            for row in rows {
                results.push(row.map_err(|e| DomainError::Storage(e.to_string()))?);
            }
            Ok(results)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    /// The credited proofs of one session (`TASK-27`/S33), read back for the chain analysis.
    pub async fn get_session_work_proofs(
        &self,
        request_id: &str,
    ) -> Result<Vec<WorkProof>, DomainError> {
        let conn = self.conn.clone();
        let request_id = request_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn
                .prepare(
                    "SELECT proof FROM ledger WHERE request_id = ?1 AND entry_type = 'Earned' AND proof IS NOT NULL AND proof != '' ORDER BY id ASC",
                )
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let rows = stmt
                .query_map(params![request_id], |row| row.get::<_, String>(0))
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let mut proofs = Vec::new();
            for row in rows {
                let serialized = row.map_err(|e| DomainError::Storage(e.to_string()))?;
                match serde_json::from_str::<SignedWorkProof>(&serialized) {
                    Ok(signed) => proofs.push(signed.proof),
                    Err(e) => tracing::warn!(
                        "⚠️ Skipping an unreadable work proof of session '{request_id}' in the ledger: {e}"
                    ),
                }
            }
            Ok(proofs)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }
}

#[async_trait::async_trait]
impl CreditService for SqliteCreditStore {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn add_work(
        &self,
        peer_id: &str,
        amount: f32,
        proof_json: Option<String>,
    ) -> Result<LedgerEntry, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let mut request_id = None;
            let mut output_hash = None;
            if let Some(ref json) = proof_json
                && let Ok(signed) = serde_json::from_str::<SignedWorkProof>(json)
            {
                request_id = Some(signed.proof.request_id);
                output_hash = Some(signed.proof.output_hash.to_vec());
            }

            let now = unix_now();

            tx.execute(
                "INSERT INTO peer_credits (peer_id, credits_earned) VALUES (?1, ?2) \
                 ON CONFLICT(peer_id) DO UPDATE SET credits_earned = credits_earned + ?2, last_seen = CURRENT_TIMESTAMP",
                params![peer_id, amount],
            )
            .map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            tx.execute(
                "INSERT INTO ledger (peer_id, entry_type, amount, proof, request_id, output_hash, source, ts) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'local', ?7)",
                params![
                    peer_id,
                    LedgerEntryType::Earned.to_db(),
                    amount,
                    proof_json.as_deref().unwrap_or_default(),
                    request_id,
                    output_hash,
                    now
                ],
            )
            .map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            Ok(LedgerEntry {
                peer_id,
                entry_type: LedgerEntryType::Earned,
                amount,
                proof: proof_json.unwrap_or_default(),
                timestamp: now,
                source: "local".to_string(),
            })
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn add_work_batch(
        &self,
        batch: Vec<(String, f32, Option<String>)>,
    ) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let now = unix_now();

            for (peer_id, amount, proof_json) in batch {
                let mut request_id = None;
                let mut output_hash = None;
                if let Some(ref json) = proof_json
                    && let Ok(signed) = serde_json::from_str::<SignedWorkProof>(json)
                {
                    request_id = Some(signed.proof.request_id);
                    output_hash = Some(signed.proof.output_hash.to_vec());
                }

                tx.execute(
                    "INSERT INTO peer_credits (peer_id, credits_earned) VALUES (?1, ?2) \
                     ON CONFLICT(peer_id) DO UPDATE SET credits_earned = credits_earned + ?2, last_seen = CURRENT_TIMESTAMP",
                    params![peer_id, amount],
                )
                .map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

                tx.execute(
                    "INSERT INTO ledger (peer_id, entry_type, amount, proof, request_id, output_hash, source, ts) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'local', ?7)",
                    params![
                        peer_id,
                        LedgerEntryType::Earned.to_db(),
                        amount,
                        proof_json.as_deref().unwrap_or_default(),
                        request_id,
                        output_hash,
                        now
                    ],
                )
                .map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            }

            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn record_consumption(
        &self,
        peer_id: &str,
        amount: f32,
        proof_json: Option<String>,
    ) -> Result<LedgerEntry, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let now = unix_now();

            tx.execute(
                "INSERT INTO peer_credits (peer_id, credits_spent) VALUES (?1, ?2) \
                 ON CONFLICT(peer_id) DO UPDATE SET credits_spent = credits_spent + ?2, last_seen = CURRENT_TIMESTAMP",
                params![peer_id, amount],
            )
            .map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            tx.execute(
                "INSERT INTO ledger (peer_id, entry_type, amount, proof, source, ts) VALUES (?1, ?2, ?3, ?4, 'local', ?5)",
                params![
                    peer_id,
                    LedgerEntryType::Spent.to_db(),
                    amount,
                    proof_json.as_deref().unwrap_or_default(),
                    now
                ],
            )
            .map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;

            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            Ok(LedgerEntry {
                peer_id,
                entry_type: LedgerEntryType::Spent,
                amount,
                proof: proof_json.unwrap_or_default(),
                timestamp: now,
                source: "local".to_string(),
            })
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn get_balance(&self, peer_id: &str) -> Result<f32, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            match conn.query_row(
                "SELECT credits_earned - credits_spent FROM peer_credits WHERE peer_id = ?1",
                params![peer_id],
                |row| row.get(0),
            ) {
                Ok(balance) => Ok(balance),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(0.0),
                Err(e) => Err(DomainError::Storage(e.to_string())),
            }
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn get_ledger(&self, limit: u32) -> Result<Vec<LedgerEntry>, DomainError> {
        let conn = self.conn.clone();
        let (entries, unreadable) = tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn
                .prepare(
                    "SELECT peer_id, entry_type, amount, proof, ts, source FROM ledger \
                 ORDER BY id DESC LIMIT ?1",
                )
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            let rows = stmt
                .query_map(params![limit], |row| {
                    Ok(LedgerEntry {
                        peer_id: row.get(0)?,
                        entry_type: parse_ledger_entry_type(&row.get::<_, String>(1)?),
                        amount: row.get(2)?,
                        proof: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        timestamp: row.get(4)?,
                        source: row.get(5).unwrap_or_else(|_| "local".into()),
                    })
                })
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;

            let mut entries = Vec::new();
            let mut unreadable: Vec<String> = Vec::new();
            for (index, row) in rows.enumerate() {
                match row {
                    Ok(entry) => entries.push(entry),
                    Err(e) => unreadable.push(format!("#{}: {e}", index + 1)),
                }
            }

            Ok::<_, DomainError>((entries, unreadable))
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))??;

        for reason in &unreadable {
            warn!(
                "⚠️ The ledger has a row that cannot be read and was left out of this view: {reason}"
            );
        }

        Ok(entries)
    }

    async fn get_ledger_height(&self) -> Result<u64, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let height: Option<i64> = conn
                .query_row("SELECT MAX(id) FROM ledger", [], |row| row.get(0))
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            Ok(height.unwrap_or(0).max(0) as u64)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    async fn get_ledger_page(
        &self,
        after_id: u64,
        limit: u32,
    ) -> Result<Vec<LedgerPageEntry>, DomainError> {
        let conn = self.conn.clone();
        let limit = limit.max(1);
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn
                .prepare(
                    "SELECT id, peer_id, entry_type, amount, proof, ts, source FROM ledger \
                 WHERE id > ?1 ORDER BY id ASC LIMIT ?2",
                )
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            let rows = stmt
                .query_map(params![after_id, limit], |row| {
                    let id: i64 = row.get(0)?;
                    Ok(LedgerPageEntry {
                        id: id.max(0) as u64,
                        entry: LedgerEntry {
                            peer_id: row.get(1)?,
                            entry_type: parse_ledger_entry_type(&row.get::<_, String>(2)?),
                            amount: row.get(3)?,
                            proof: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                            timestamp: row.get(5)?,
                            source: row.get(6).unwrap_or_else(|_| "local".into()),
                        },
                    })
                })
                .map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?;
            let mut page = Vec::new();
            for entry in rows {
                page.push(entry.map_err(|e| {
                    tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e))
                })?);
            }
            Ok(page)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }
}
