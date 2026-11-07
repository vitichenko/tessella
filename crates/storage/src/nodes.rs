use rusqlite::{Connection, params};
use std::sync::{Arc, Mutex};
use tessella_core::error::DomainError;
use tessella_core::models::{LayerPeer, NodeInfo, PeerStatus, Role, SignedWorkProof};
use tracing::warn;

/// How long a peer stays in the persistent cache (`TASK-15`, H-06) after the last time we saw it.
pub const KNOWN_PEERS_TTL_SECS: u64 = 3600;

/// Hard cap of the persistent cache, so a mesh of hundreds of nodes cannot grow the table without
/// bound on a small disk.
pub const KNOWN_PEERS_MAX_ROWS: i64 = 256;

/// Store handling node discovery, active topology tracking, and offline peer recovery cache.
#[derive(Clone)]
pub struct SqliteNodeStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteNodeStore {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    pub async fn save_node(&self, node: NodeInfo) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let role_json = serde_json::to_string(&node.role)
                .map_err(|e| DomainError::Validation(e.to_string()))?;
            conn.execute(
                "INSERT INTO discovered_nodes (peer_id, host, port, vram_mb, role, model_id, p2p_address, http_api_url, relay_address, status, failed_requests, successful_requests, latency_p99_ms, queue_depth, last_error, slash_count, blacklisted, public_ip, public_api_port) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19) \
                 ON CONFLICT(peer_id) DO UPDATE SET host = ?2, port = ?3, vram_mb = ?4, role = ?5, model_id = ?6, p2p_address = ?7, http_api_url = ?8, relay_address = ?9, \
                 status = CASE WHEN status = 'Slashed' THEN 'Slashed' ELSE ?10 END, failed_requests = MAX(failed_requests, ?11), successful_requests = MAX(successful_requests, ?12), \
                 latency_p99_ms = CASE WHEN ?13 = 0 THEN latency_p99_ms ELSE ?13 END, queue_depth = ?14, last_error = COALESCE(?15, last_error), slash_count = MAX(slash_count, ?16), \
                 blacklisted = ?17 OR blacklisted, public_ip = ?18, public_api_port = ?19, last_seen = CURRENT_TIMESTAMP",
                params![
                    node.peer_id,
                    node.host,
                    node.port,
                    node.vram_mb,
                    role_json,
                    node.model_id,
                    node.p2p_address,
                    node.http_api_url,
                    node.relay_address,
                    format!("{:?}", node.status),
                    node.failed_requests,
                    node.successful_requests,
                    node.latency_p99_ms,
                    node.queue_depth,
                    node.last_error,
                    node.slash_count,
                    node.blacklisted,
                    node.public_ip,
                    node.public_api_port
                ],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn get_nodes(&self) -> Result<Vec<NodeInfo>, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn.prepare(
                "SELECT peer_id, host, port, vram_mb, role, model_id, p2p_address, http_api_url, relay_address, status, \
                        failed_requests, successful_requests, latency_p99_ms, queue_depth, last_error, slash_count, blacklisted, public_ip, public_api_port \
                 FROM discovered_nodes \
                 WHERE last_seen > datetime('now', '-10 minutes') AND status != 'Offline'"
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            let node_iter = stmt.query_map([], |row| {
                let role_str: String = row.get(4)?;
                let role = serde_json::from_str::<Role>(&role_str).unwrap_or(Role::Unassigned);
                let status_str: String = row.get(9).unwrap_or_else(|_| "Healthy".to_string());
                let status = match status_str.as_str() {
                    "Degraded" => PeerStatus::Degraded,
                    "Slashed" => PeerStatus::Slashed,
                    "Offline" => PeerStatus::Offline,
                    _ => PeerStatus::Healthy,
                };
                Ok(NodeInfo {
                    peer_id: row.get(0)?,
                    host: row.get(1)?,
                    port: row.get(2)?,
                    vram_mb: row.get(3)?,
                    role,
                    model_id: row.get(5)?,
                    p2p_address: row.get(6)?,
                    http_api_url: row.get(7)?,
                    relay_address: row.get(8)?,
                    status,
                    failed_requests: row.get(10)?,
                    successful_requests: row.get(11)?,
                    latency_p99_ms: row.get(12)?,
                    queue_depth: row.get(13)?,
                    last_error: row.get(14)?,
                    slash_count: row.get(15)?,
                    blacklisted: row.get(16)?,
                    public_ip: row.get(17)?,
                    public_api_port: row.get(18)?,
                })
            }).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(node_iter.flatten().collect())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn get_all_nodes(&self) -> Result<Vec<NodeInfo>, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn.prepare(
                "SELECT peer_id, host, port, vram_mb, role, model_id, p2p_address, http_api_url, relay_address, status, \
                        failed_requests, successful_requests, latency_p99_ms, queue_depth, last_error, slash_count, blacklisted, public_ip, public_api_port \
                 FROM discovered_nodes"
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            let node_iter = stmt.query_map([], |row| {
                let role_str: String = row.get(4)?;
                let role = serde_json::from_str::<Role>(&role_str).unwrap_or(Role::Unassigned);
                let status_str: String = row.get(9).unwrap_or_else(|_| "Healthy".to_string());
                let status = match status_str.as_str() {
                    "Degraded" => PeerStatus::Degraded,
                    "Slashed" => PeerStatus::Slashed,
                    "Offline" => PeerStatus::Offline,
                    _ => PeerStatus::Healthy,
                };
                Ok(NodeInfo {
                    peer_id: row.get(0)?,
                    host: row.get(1)?,
                    port: row.get(2)?,
                    vram_mb: row.get(3)?,
                    role,
                    model_id: row.get(5)?,
                    p2p_address: row.get(6)?,
                    http_api_url: row.get(7)?,
                    relay_address: row.get(8)?,
                    status,
                    failed_requests: row.get(10)?,
                    successful_requests: row.get(11)?,
                    latency_p99_ms: row.get(12)?,
                    queue_depth: row.get(13)?,
                    last_error: row.get(14)?,
                    slash_count: row.get(15)?,
                    blacklisted: row.get(16)?,
                    public_ip: row.get(17)?,
                    public_api_port: row.get(18)?,
                })
            }).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(node_iter.flatten().collect())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn deregister_node(&self, peer_id: &str) -> Result<(), DomainError> {
        let conn = self.conn.clone();
        let peer_id = peer_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "UPDATE discovered_nodes SET last_seen = datetime('now', '-10 minutes'), status = 'Offline' WHERE peer_id = ?1",
                params![peer_id],
            ).map_err(|e| tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(e)))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn save_pending_work_proof(
        &self,
        proof: &SignedWorkProof,
    ) -> Result<i64, DomainError> {
        let conn = self.conn.clone();
        let serialized = serde_json::to_string(proof)
            .map_err(|e| DomainError::Storage(format!("Failed to serialize work proof: {}", e)))?;

        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            conn.execute(
                "INSERT INTO pending_work_proofs (serialized_proof) VALUES (?1)",
                params![serialized],
            )
            .map_err(|e| DomainError::Storage(e.to_string()))?;
            let row_id = conn.last_insert_rowid();
            Ok(row_id)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn count_pending_work_proofs(&self) -> Result<u64, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM pending_work_proofs", [], |row| {
                    row.get(0)
                })
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok(count.max(0) as u64)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn get_pending_work_proofs(
        &self,
    ) -> Result<Vec<(i64, SignedWorkProof)>, DomainError> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn
                .prepare("SELECT id, serialized_proof FROM pending_work_proofs ORDER BY id ASC")
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let rows = stmt
                .query_map([], |row| {
                    let id: i64 = row.get(0)?;
                    let serialized: String = row.get(1)?;
                    Ok((id, serialized))
                })
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let mut results = Vec::new();
            for r in rows {
                let (id, serialized) = r.map_err(|e| DomainError::Storage(e.to_string()))?;
                let proof: SignedWorkProof = serde_json::from_str(&serialized)
                    .map_err(|e| DomainError::Storage(format!("Deserialization error: {}", e)))?;
                results.push((id, proof));
            }
            Ok(results)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn delete_pending_work_proofs(&self, ids: Vec<i64>) -> Result<(), DomainError> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            for id in ids {
                tx.execute("DELETE FROM pending_work_proofs WHERE id = ?1", params![id])
                    .map_err(|e| {
                        tessella_core::error::DomainError::from(crate::error::StorageError::Sqlite(
                            e,
                        ))
                    })?;
            }

            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn remember_known_peers(
        &self,
        peers: &[LayerPeer],
        now: u64,
    ) -> Result<usize, DomainError> {
        if peers.is_empty() {
            return Ok(0);
        }
        let rows: Vec<(String, String)> = peers
            .iter()
            .map(|peer| {
                let payload = serde_json::to_string(peer)
                    .map_err(|e| DomainError::Storage(format!("Failed to serialize peer: {e}")))?;
                Ok((peer.peer_id.clone(), payload))
            })
            .collect::<Result<_, DomainError>>()?;

        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            for (peer_id, payload) in &rows {
                tx.execute(
                    "INSERT INTO known_peers (peer_id, payload, updated_at) VALUES (?1, ?2, ?3) \
                     ON CONFLICT(peer_id) DO UPDATE SET payload = ?2, updated_at = ?3",
                    params![peer_id, payload, now as i64],
                )
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            }

            let cutoff = now.saturating_sub(KNOWN_PEERS_TTL_SECS) as i64;
            tx.execute(
                "DELETE FROM known_peers WHERE updated_at < ?1",
                params![cutoff],
            )
            .map_err(|e| DomainError::Storage(e.to_string()))?;
            tx.execute(
                "DELETE FROM known_peers WHERE peer_id NOT IN (
                     SELECT peer_id FROM known_peers ORDER BY updated_at DESC LIMIT ?1
                 )",
                params![KNOWN_PEERS_MAX_ROWS],
            )
            .map_err(|e| DomainError::Storage(e.to_string()))?;

            tx.commit()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            Ok(rows.len())
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }

    pub async fn load_known_peers(
        &self,
        now: u64,
        ttl_secs: u64,
    ) -> Result<Vec<LayerPeer>, DomainError> {
        let cutoff = now.saturating_sub(ttl_secs) as i64;
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| DomainError::Storage(e.to_string()))?;
            let mut stmt = conn
                .prepare(
                    "SELECT peer_id, payload FROM known_peers WHERE updated_at >= ?1 \
                     ORDER BY updated_at DESC",
                )
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let rows = stmt
                .query_map(params![cutoff], |row| {
                    let peer_id: String = row.get(0)?;
                    let payload: String = row.get(1)?;
                    Ok((peer_id, payload))
                })
                .map_err(|e| DomainError::Storage(e.to_string()))?;

            let mut peers = Vec::new();
            for row in rows {
                let (peer_id, payload) = row.map_err(|e| DomainError::Storage(e.to_string()))?;
                match serde_json::from_str::<LayerPeer>(&payload) {
                    Ok(peer) => peers.push(peer),
                    Err(e) => warn!(
                        "⚠️ known_peers: dropping the cached record of {} ({}); the mesh view is \
                         re-learned from the tracker",
                        peer_id, e
                    ),
                }
            }
            Ok(peers)
        })
        .await
        .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
    }
}
