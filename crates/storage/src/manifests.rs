use rusqlite::{Connection, params};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tessella_core::error::DomainError;
use tessella_core::models::{SignedTransferReceipt, TransferReceipt, WeightManifest};
use tessella_core::traits::WeightProvider;

/// Store handling model weight manifests, transfer receipts, and local safetensors weights files.
#[derive(Clone)]
pub struct SqliteManifestStore {
    conn: Arc<Mutex<Connection>>,
    pub weights_base_path: PathBuf,
}

impl SqliteManifestStore {
    pub fn new(conn: Arc<Mutex<Connection>>, weights_base_path: PathBuf) -> Self {
        Self {
            conn,
            weights_base_path,
        }
    }

    pub async fn save_manifest(&self, manifest: WeightManifest) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
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
                    manifest.blake3_hash.to_vec(),
                    manifest.size_bytes,
                    manifest.registered_at,
                    manifest.tracker_public_key,
                    manifest.tracker_signature
                ],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn get_manifest(
        &self,
        model_id: &str,
        role_json: &str,
    ) -> Result<Option<WeightManifest>, DomainError> {
        let conn = self.conn.clone();
        let model_id = model_id.to_string();
        let role_json = role_json.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let res: Result<WeightManifest, rusqlite::Error> = conn.query_row(
                "SELECT model_id, filename, role, blake3_hash, size_bytes, registered_at, tracker_public_key, tracker_signature \
                 FROM weight_manifests WHERE model_id = ?1 AND role = ?2",
                params![model_id, role_json],
                |row| {
                    let role_str: String = row.get(2)?;
                    let role = serde_json::from_str(&role_str).map_err(|_| rusqlite::Error::InvalidQuery)?;
                    let hash_vec: Vec<u8> = row.get(3)?;
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hash_vec);
                    Ok(WeightManifest {
                        model_id: row.get(0)?,
                        filename: row.get(1)?,
                        role,
                        blake3_hash: hash,
                        size_bytes: row.get(4)?,
                        registered_at: row.get(5)?,
                        tracker_public_key: row.get(6)?,
                        tracker_signature: row.get(7)?,
                    })
                },
            );
            match res {
                Ok(m) => Ok(Some(m)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(DomainError::Storage(e.to_string())),
            }
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn get_manifests_for_model(
        &self,
        model_id: &str,
    ) -> Result<Vec<WeightManifest>, DomainError> {
        let conn = self.conn.clone();
        let model_id = model_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn.prepare(
                "SELECT model_id, filename, role, blake3_hash, size_bytes, registered_at, tracker_public_key, tracker_signature \
                 FROM weight_manifests WHERE model_id = ?1"
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            let iter = stmt.query_map(params![model_id], |row| {
                let role_str: String = row.get(2)?;
                let role =
                    serde_json::from_str(&role_str).map_err(|_| rusqlite::Error::InvalidQuery)?;
                let hash_vec: Vec<u8> = row.get(3)?;
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&hash_vec);
                Ok(WeightManifest {
                    model_id: row.get(0)?,
                    filename: row.get(1)?,
                    role,
                    blake3_hash: hash,
                    size_bytes: row.get(4)?,
                    registered_at: row.get(5)?,
                    tracker_public_key: row.get(6)?,
                    tracker_signature: row.get(7)?,
                })
            }).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(iter.flatten().collect())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn list_manifests(&self) -> Result<Vec<WeightManifest>, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn.prepare(
                "SELECT model_id, filename, role, blake3_hash, size_bytes, registered_at, tracker_public_key, tracker_signature \
                 FROM weight_manifests"
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            let iter = stmt.query_map([], |row| {
                let role_str: String = row.get(2)?;
                let role =
                    serde_json::from_str(&role_str).map_err(|_| rusqlite::Error::InvalidQuery)?;
                let hash_vec: Vec<u8> = row.get(3)?;
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&hash_vec);
                Ok(WeightManifest {
                    model_id: row.get(0)?,
                    filename: row.get(1)?,
                    role,
                    blake3_hash: hash,
                    size_bytes: row.get(4)?,
                    registered_at: row.get(5)?,
                    tracker_public_key: row.get(6)?,
                    tracker_signature: row.get(7)?,
                })
            }).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(iter.flatten().collect())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn upsert_manifest(&self, manifest: WeightManifest) -> Result<(), DomainError> {
        self.save_manifest(manifest).await
    }

    pub async fn save_receipt(&self, signed: &SignedTransferReceipt) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        let signed = signed.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let receipt_json =
                serde_json::to_string(&signed).map_err(|e| DomainError::Validation(e.to_string()))?;
            conn.execute(
                "INSERT INTO transfer_receipts (sender_peer_id, receiver_peer_id, model_id, role, blake3_hash, size_bytes, timestamp, receipt_json) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    signed.receipt.sender_peer_id,
                    signed.receipt.receiver_peer_id,
                    signed.receipt.model_id,
                    serde_json::to_string(&signed.receipt.role).unwrap_or_default(),
                    signed.receipt.blake3_hash.to_vec(),
                    signed.receipt.size_bytes,
                    signed.receipt.timestamp,
                    receipt_json
                ],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn get_receipt(
        &self,
        sender_peer_id: &str,
        receiver_peer_id: &str,
        model_id: &str,
        role: &str,
    ) -> Result<Option<SignedTransferReceipt>, DomainError> {
        let conn = self.conn.clone();
        let sender_peer_id = sender_peer_id.to_string();
        let receiver_peer_id = receiver_peer_id.to_string();
        let model_id = model_id.to_string();
        let role = role.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let res: Result<String, rusqlite::Error> = conn.query_row(
                "SELECT receipt_json FROM transfer_receipts WHERE sender_peer_id = ?1 AND receiver_peer_id = ?2 AND model_id = ?3 AND role = ?4 ORDER BY timestamp DESC LIMIT 1",
                params![sender_peer_id, receiver_peer_id, model_id, role],
                |row| row.get(0),
            );
            match res {
                Ok(json) => Ok(serde_json::from_str(&json).ok()),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(DomainError::Storage(e.to_string())),
            }
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn list_receipts(&self, peer_id: &str) -> Result<Vec<TransferReceipt>, DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(format!("DB lock poisoned: {}", e)))?;
            let mut stmt = conn.prepare(
                "SELECT id, sender_peer_id, receiver_peer_id, model_id, role, blake3_hash, size_bytes, timestamp, receipt_json
                 FROM transfer_receipts WHERE sender_peer_id = ?1 OR receiver_peer_id = ?1"
            ).map_err(|e| DomainError::Storage(e.to_string()))?;

            let rows = stmt
                .query_map(params![peer_id], |row| {
                    let role_json: String = row.get(4)?;
                    let role = serde_json::from_str(&role_json)
                        .unwrap_or(tessella_core::models::Role::Unassigned);

                    Ok(TransferReceipt {
                        sender_peer_id: row.get(1)?,
                        receiver_peer_id: row.get(2)?,
                        model_id: row.get(3)?,
                        role,
                        blake3_hash: row.get::<_, Vec<u8>>(5)?.try_into().unwrap_or_default(),
                        size_bytes: row.get(6)?,
                        timestamp: row.get(7)?,
                    })
                })
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
}

impl WeightProvider for SqliteManifestStore {
    fn get_weights(&self, name: &str) -> Result<Vec<u8>, DomainError> {
        let path = self.weights_base_path.join(format!("{}.safetensors", name));
        std::fs::read(&path)
            .map_err(|e| DomainError::Storage(format!("Failed to read weights {:?}: {}", path, e)))
    }
}
