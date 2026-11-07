use crate::*;
use rusqlite::{Connection, params};
use tessella_core::models::*;
use tessella_core::traits::*;

#[cfg(test)]
mod work_ticket_tests {
    use super::*;

    fn storage() -> SqliteStorage {
        SqliteStorage::new(":memory:", None).unwrap()
    }

    fn ticket(request_id: &str, model_id: &str, expires_at: u64) -> WorkTicket {
        WorkTicket {
            request_id: request_id.to_string(),
            model_id: model_id.to_string(),
            entry_peer_id: "12D3KooWEntry".to_string(),
            issued_at: unix_now(),
            expires_at,
            audit_layer: None,
            audit_replicas: 2,
            audit_canary: None,
            audit_reward: 0.0,
            orchestrator_signature: vec![1, 2, 3],
        }
    }

    /// The core of H-12: a `request_id` the orchestrator minted can be paid once per worker, and a
    /// second attempt (the WAL sweeper retries whole batches) must not credit twice.
    #[tokio::test]
    async fn an_issued_ticket_is_credited_once_per_worker() {
        let db = storage();
        let issued = ticket("work-1", "model-a", unix_now() + 600);
        db.store_work_ticket(&issued).await.unwrap();

        assert_eq!(
            db.consume_work_ticket(&issued, "peerA", "model-a", 3.0, unix_now())
                .await
                .unwrap(),
            WorkTicketOutcome::Accepted { credits: 3.0 }
        );
        assert_eq!(
            db.consume_work_ticket(&issued, "peerA", "model-a", 3.0, unix_now())
                .await
                .unwrap(),
            WorkTicketOutcome::Duplicate { credits: 3.0 },
            "a retried proof must be a no-op, not a second payment"
        );

        // A different hop of the same session worked too and gets its own (single) credit.
        assert_eq!(
            db.consume_work_ticket(&issued, "peerB", "model-a", 5.0, unix_now())
                .await
                .unwrap(),
            WorkTicketOutcome::Accepted { credits: 5.0 }
        );
        assert_eq!(
            db.consume_work_ticket(&issued, "peerB", "model-a", 5.0, unix_now())
                .await
                .unwrap(),
            WorkTicketOutcome::Duplicate { credits: 5.0 }
        );
    }

    /// A `request_id` nobody issued is exactly what H-12 called "inventing ids": refused, and the
    /// refusal is distinguishable from an expired ticket so the log can tell them apart.
    #[tokio::test]
    async fn an_invented_request_id_is_refused() {
        let db = storage();
        let forged = ticket("invented-by-the-worker", "model-a", unix_now() + 600);

        assert_eq!(
            db.consume_work_ticket(&forged, "peerA", "model-a", 999.0, unix_now())
                .await
                .unwrap(),
            WorkTicketOutcome::Unknown
        );
    }

    #[tokio::test]
    async fn an_expired_ticket_is_refused() {
        let db = storage();
        let mut expired = ticket("work-2", "model-a", unix_now() - 1);
        expired.issued_at = unix_now() - 3600;
        db.store_work_ticket(&expired).await.unwrap();

        let expires_at = expired.expires_at;
        assert_eq!(
            db.consume_work_ticket(&expired, "peerA", "model-a", 3.0, unix_now())
                .await
                .unwrap(),
            WorkTicketOutcome::Expired { expires_at }
        );

        // Right on the expiry second it is still valid (same inclusive rule as `KarmaTicket`).
        assert_eq!(
            db.consume_work_ticket(&expired, "peerA", "model-a", 3.0, expires_at)
                .await
                .unwrap(),
            WorkTicketOutcome::Accepted { credits: 3.0 }
        );
    }

    #[tokio::test]
    async fn a_ticket_authorises_one_model_only() {
        let db = storage();
        let issued = ticket("work-3", "model-a", unix_now() + 600);
        db.store_work_ticket(&issued).await.unwrap();

        assert_eq!(
            db.consume_work_ticket(&issued, "peerA", "model-b", 3.0, unix_now())
                .await
                .unwrap(),
            WorkTicketOutcome::ModelMismatch {
                expected: "model-a".to_string(),
                got: "model-b".to_string(),
            }
        );
    }

    /// Pruning keeps both tables bounded: tickets that expired longer ago than the retention window
    /// disappear, and the consumptions that hang from them go with them.
    #[tokio::test]
    async fn old_tickets_and_their_consumptions_are_pruned() {
        let db = storage();
        let now = unix_now();

        let fresh = ticket("work-fresh", "model-a", now + 600);
        db.store_work_ticket(&fresh).await.unwrap();
        assert_eq!(
            db.consume_work_ticket(&fresh, "peerA", "model-a", 1.0, now)
                .await
                .unwrap(),
            WorkTicketOutcome::Accepted { credits: 1.0 }
        );

        let ancient = ticket(
            "work-ancient",
            "model-a",
            now - WORK_TICKET_RETENTION_SECS - 1,
        );
        db.store_work_ticket(&ancient).await.unwrap();
        // Storing the next ticket is what triggers the opportunistic prune.
        let newer = ticket("work-newer", "model-a", now + 600);
        db.store_work_ticket(&newer).await.unwrap();

        assert_eq!(
            db.consume_work_ticket(&ancient, "peerB", "model-a", 1.0, now)
                .await
                .unwrap(),
            WorkTicketOutcome::Unknown,
            "a ticket expired beyond the retention window must have been pruned"
        );
        assert_eq!(
            db.consume_work_ticket(&fresh, "peerA", "model-a", 1.0, now)
                .await
                .unwrap(),
            WorkTicketOutcome::Duplicate { credits: 1.0 },
            "the fresh ticket and its consumption must survive the prune"
        );
    }
}

/// Tests of the periodic WAL maintenance (`TASK-13`, H-21). Offline: a SQLite file under the system
/// temp dir, no daemon, no socket and no GPU.
///
/// Unlike the rest of the suite these tests cannot use `:memory:`: a `-wal` sidecar only exists for
/// an on-disk database, and the whole point of H-21 is what happens to that file.
#[cfg(test)]
mod known_peers_tests {
    use super::*;
    use tessella_core::models::LayerPeer;

    fn storage() -> SqliteStorage {
        SqliteStorage::new(":memory:", None).unwrap()
    }

    fn peer(peer_id: &str, p2p_addr: &str) -> LayerPeer {
        LayerPeer {
            peer_id: peer_id.to_string(),
            http_addr: format!("http://{peer_id}:10001"),
            p2p_addr: p2p_addr.to_string(),
            layer_ranges: vec![(0, 12)],
            expert_range: None,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        }
    }

    /// The whole point of the cache (criterion 2 of `TASK-15`): a worker that restarts while the
    /// tracker is down gets its mesh view back from SQLite, with the addresses it needs to dial.
    #[tokio::test]
    async fn a_cached_peer_survives_a_restart() {
        let now = 1_000_000;
        {
            let db = storage();
            assert_eq!(
                db.remember_known_peers(
                    &[
                        peer("peerA", "/ip4/10.0.0.1/tcp/9001"),
                        peer("peerB", "/ip4/10.0.0.2/tcp/9001"),
                    ],
                    now,
                )
                .await
                .unwrap(),
                2
            );
        }

        // A new process opening the same file/db (`:memory:` here shares nothing, so use a file).
        let dir = std::env::temp_dir().join(format!("tessella-known-peers-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("peers.db");
        let _ = std::fs::remove_file(&path);

        let db = SqliteStorage::new(path.to_str().unwrap(), None).unwrap();
        db.remember_known_peers(&[peer("peerA", "/ip4/10.0.0.1/tcp/9001")], now)
            .await
            .unwrap();
        drop(db);

        let reopened = SqliteStorage::new(path.to_str().unwrap(), None).unwrap();
        let cached = reopened
            .load_known_peers(now, KNOWN_PEERS_TTL_SECS)
            .await
            .unwrap();
        assert_eq!(cached.len(), 1, "the cache must survive the process");
        assert_eq!(cached[0].peer_id, "peerA");
        assert_eq!(cached[0].p2p_addr, "/ip4/10.0.0.1/tcp/9001");
        assert_eq!(cached[0].layer_ranges, vec![(0, 12)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn stale_rows_are_not_returned_and_are_pruned_on_write() {
        let db = storage();
        let first = 100_000;
        db.remember_known_peers(&[peer("old", "/ip4/10.0.0.9/tcp/9001")], first)
            .await
            .unwrap();
        let now = first + KNOWN_PEERS_TTL_SECS + 1;
        db.remember_known_peers(&[peer("fresh", "/ip4/10.0.0.1/tcp/9001")], now)
            .await
            .unwrap();

        let cached = db
            .load_known_peers(now, KNOWN_PEERS_TTL_SECS)
            .await
            .unwrap();
        assert_eq!(
            cached
                .iter()
                .map(|p| p.peer_id.as_str())
                .collect::<Vec<_>>(),
            vec!["fresh"],
            "a row older than the TTL must not seed a cold start"
        );

        // The prune is also a write-side guarantee, so the table does not grow forever.
        let everything = db
            .load_known_peers(now, 100 * KNOWN_PEERS_TTL_SECS)
            .await
            .unwrap();
        assert_eq!(
            everything.len(),
            1,
            "the stale row was deleted by the upsert"
        );
    }

    #[tokio::test]
    async fn the_cache_is_capped() {
        let db = storage();
        let peers: Vec<LayerPeer> = (0..KNOWN_PEERS_MAX_ROWS + 20)
            .map(|i| peer(&format!("peer{i}"), "/ip4/10.0.0.1/tcp/9001"))
            .collect();
        db.remember_known_peers(&peers, 5_000).await.unwrap();

        let cached = db
            .load_known_peers(5_000, KNOWN_PEERS_TTL_SECS)
            .await
            .unwrap();
        assert_eq!(
            cached.len() as i64,
            KNOWN_PEERS_MAX_ROWS,
            "the cache must stay bounded"
        );
    }

    /// An empty mesh view is not an error and must not delete the cache: a single failed `/peers`
    /// call must not cost us the cold-start information we already had.
    #[tokio::test]
    async fn remembering_nothing_keeps_what_was_cached() {
        let db = storage();
        db.remember_known_peers(&[peer("peerA", "/ip4/10.0.0.1/tcp/9001")], 7_000)
            .await
            .unwrap();
        assert_eq!(db.remember_known_peers(&[], 7_000).await.unwrap(), 0);

        let cached = db
            .load_known_peers(7_000, KNOWN_PEERS_TTL_SECS)
            .await
            .unwrap();
        assert_eq!(cached.len(), 1);
    }
}

#[cfg(test)]
mod wal_tests {
    use super::*;
    // `SystemTime`/`UNIX_EPOCH` used to be imported at the top of the file, where the only user is
    // this test: after `TASK-14` moved the other five expressions into `unix_now()`, the crate-level
    // import became dead weight for the library target (`clippy -D warnings` catches it here).
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "tessella-wal-{}-{}-{}.db",
            std::process::id(),
            name,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        remove_db_files(&path);
        path
    }

    /// Removes the database and both sidecars, so a test never reads a previous run's WAL.
    fn remove_db_files(db: &Path) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", db.display(), suffix));
        }
    }

    /// Size of the `-wal` sidecar in bytes; `0` when SQLite already removed it.
    fn wal_bytes(db: &Path) -> u64 {
        std::fs::metadata(format!("{}-wal", db.display()))
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// One committed transaction per row, which is what puts pages in the WAL.
    async fn write_nodes(db: &SqliteStorage, rows: u32) {
        for i in 0..rows {
            db.save_node(test_node(i)).await.unwrap();
        }
    }

    fn test_node(index: u32) -> NodeInfo {
        NodeInfo {
            peer_id: format!("12D3KooWpeer{index}"),
            host: "127.0.0.1".to_string(),
            port: 9001,
            vram_mb: 8192,
            role: Role::Unassigned,
            model_id: "model-a".to_string(),
            p2p_address: "/ip4/127.0.0.1/tcp/9001".to_string(),
            http_api_url: None,
            relay_address: None,
            status: PeerStatus::Healthy,
            failed_requests: 0,
            successful_requests: 0,
            latency_p99_ms: 0,
            queue_depth: 0,
            last_error: Some("x".repeat(64)),
            slash_count: 0,
            blacklisted: false,
            public_ip: None,
            public_api_port: None,
        }
    }

    /// Polls `predicate` until it holds or the deadline expires. Returns its last value, so a
    /// failure message can tell "never happened" from "not yet".
    async fn wait_for(predicate: impl Fn() -> bool, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if predicate() {
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        predicate()
    }

    /// The acceptance criterion of H-21: after inserting rows, `wal_checkpoint(TRUNCATE)` leaves the
    /// `-wal` at 0 and the data is still readable (it now lives in the main database file).
    #[tokio::test]
    async fn checkpoint_truncates_the_wal_and_keeps_the_data_readable() {
        let path = temp_db("truncate");
        let db = SqliteStorage::new(&path, None).unwrap();
        write_nodes(&db, 40).await;

        // SQLite allocates the sidecar in fixed-size chunks the moment the first transaction is
        // committed, and a PASSIVE/FULL checkpoint never gives that space back.
        let before = wal_bytes(&path);
        assert!(
            before > 0,
            "the -wal sidecar must exist after 40 commits (got {before} bytes)"
        );

        let outcome = db.checkpoint_wal().await.unwrap();
        assert!(
            !outcome.busy,
            "this test holds the only connection, so nothing can block it: {outcome:?}"
        );
        assert_eq!(
            wal_bytes(&path),
            0,
            "TRUNCATE must leave the -wal empty (was {before} bytes, outcome: {outcome:?})"
        );

        let nodes = db.get_all_nodes().await.unwrap();
        assert_eq!(
            nodes.len(),
            40,
            "every row must survive the truncation (in the main db file now)"
        );
        remove_db_files(&path);
    }

    /// The periodic task is what actually runs in the two binaries: it must truncate the WAL on its
    /// own and stop when it is cancelled (the `TASK-10`/H-08 contract for long-lived tasks).
    #[tokio::test]
    async fn the_maintenance_task_truncates_the_wal_on_its_own_and_stops_on_shutdown() {
        let path = temp_db("periodic");
        let db = Arc::new(SqliteStorage::new(&path, None).unwrap());
        let task = WalMaintenance::spawn(db.clone(), std::time::Duration::from_millis(300))
            .expect("a non-zero interval spawns the task");

        // Phase 1 — the immediate first tick (the first `interval` tick of `tokio` resolves at
        // once), which is what reclaims the WAL of a previous run without waiting for a full period.
        write_nodes(&db, 5).await;
        assert!(
            wal_bytes(&path) > 0,
            "the writes must land in the -wal sidecar"
        );
        assert!(
            wait_for(|| wal_bytes(&path) == 0, std::time::Duration::from_secs(5)).await,
            "the maintenance task must truncate the -wal without being asked"
        );

        // Phase 2 — periodic behaviour. The first tick already happened, so writing is no longer
        // racing it: the next one is a whole interval away.
        write_nodes(&db, 20).await;
        assert!(
            wal_bytes(&path) > 0,
            "the 20 commits must land in the -wal sidecar"
        );
        assert!(
            wait_for(|| wal_bytes(&path) == 0, std::time::Duration::from_secs(5)).await,
            "the next tick must truncate the -wal again"
        );
        assert!(
            !task.is_finished(),
            "the task is periodic: it must still be alive after its checkpoints"
        );

        task.cancel();
        assert!(
            wait_for(|| task.is_finished(), std::time::Duration::from_secs(5)).await,
            "a cancelled maintenance task must leave its loop"
        );
        task.shutdown().await; // idempotent: cancel + join
        remove_db_files(&path);
    }

    /// `wal_checkpoint_interval_secs = 0` disables the task (documented, so nobody guesses, and
    /// nobody gets a panic from `interval(Duration::ZERO)`).
    #[tokio::test]
    async fn a_zero_interval_disables_the_maintenance_task() {
        let path = temp_db("disabled");
        let db = Arc::new(SqliteStorage::new(&path, None).unwrap());
        write_nodes(&db, 20).await;
        assert!(
            wal_bytes(&path) > 0,
            "the writes must land in the -wal sidecar"
        );

        assert!(
            WalMaintenance::spawn(db.clone(), std::time::Duration::ZERO).is_none(),
            "interval 0 means disabled, not a zero-period timer"
        );
        assert!(
            wal_bytes(&path) > 0,
            "with the task disabled nothing truncates the -wal while the process runs"
        );

        // ... and this is why a long-lived process needs the task: the sidecar is only reclaimed
        // when the last connection closes.
        drop(db);
        assert_eq!(
            wal_bytes(&path),
            0,
            "closing the last connection is the only cleanup SQLite does on its own"
        );
        remove_db_files(&path);
    }

    /// Most of the suite (and every unit test of the two binaries) runs against `:memory:`, so the
    /// checkpoint must be a usable no-op there instead of an error.
    #[tokio::test]
    async fn checkpointing_an_in_memory_database_is_a_no_op() {
        let db = SqliteStorage::new(":memory:", None).unwrap();
        let outcome = db
            .checkpoint_wal()
            .await
            .expect("an in-memory database has no WAL, but the call must not fail");
        assert!(!outcome.busy);
    }
}

/// Tests of the reputation multiplier (`TASK-17`, residue of H-20). All offline, in memory: no
/// daemon, no socket, no GPU (PLAN §1).
///
/// The point of the module: `[reputation] slash_penalty` was parsed and **never used**, while the
/// factor that actually punished a peer lived here as the literals `SCORE_PENALTY_INVALID_SIGNATURE
/// = 0.5` and `SCORE_PENALTY_EXCESSIVE_FAILURES = 0.6`. The storage now takes the value from the
/// configuration (through [`SqliteStorage::with_slash_penalty`]) and the default reproduces the old
/// live behaviour exactly.
#[cfg(test)]
mod reputation_penalty_tests {
    use super::*;
    use tessella_core::config::ReputationConfig;

    fn storage(slash_penalty: f64) -> SqliteStorage {
        SqliteStorage::with_slash_penalty(":memory:", None, slash_penalty).unwrap()
    }

    /// El default de `new()` viene de `ReputationConfig`, así que la clave del TOML y el valor que
    /// usa un `SqliteStorage` construido sin política no pueden divergir (misma trampa que el
    /// `wal_checkpoint_interval_secs` compartido de S15).
    #[test]
    fn the_default_penalty_is_the_shared_config_default() {
        assert_eq!(
            DEFAULT_SCORE_PENALTY,
            ReputationConfig::default().slash_penalty,
            "el default del storage debe ser el de [reputation]"
        );
        assert_eq!(
            DEFAULT_SCORE_PENALTY, 0.5,
            "…y ese default es el literal que había aquí antes de TASK-17"
        );
    }

    /// Dos valores de la clave, dos notas distintas sobre el mismo incidente: eso es lo que hace
    /// que la clave deje de ser decorativa.
    #[tokio::test]
    async fn a_recorded_incident_multiplies_the_score_by_the_configured_penalty() {
        let lenient = storage(0.2);
        let after_first = lenient
            .update_score(
                "peer",
                ScoreEvent::IncidentRecorded(IncidentType::InvalidSignature),
            )
            .await
            .unwrap();
        assert!(
            (after_first.score - 0.2).abs() < 1e-9,
            "1.0 * 0.2 = 0.2, got {}",
            after_first.score
        );

        // Y el castigo se compone: el siguiente incidente multiplica otra vez.
        let second = lenient
            .update_score(
                "peer",
                ScoreEvent::IncidentRecorded(IncidentType::InvalidSignature),
            )
            .await
            .unwrap();
        assert!(
            (second.score - 0.04).abs() < 1e-9,
            "0.2 * 0.2 = 0.04, got {}",
            second.score
        );

        let harsh = storage(0.01);
        let hardened = harsh
            .update_score(
                "peer",
                ScoreEvent::IncidentRecorded(IncidentType::InvalidSignature),
            )
            .await
            .unwrap();
        assert!(
            hardened.score < second.score,
            "un operador más severo deja la nota más baja: {} vs {}",
            hardened.score,
            second.score
        );
    }

    /// Sin política explícita (el caso del worker y de los tests) el comportamiento es **idéntico**
    /// al de antes de TASK-17: 0.5 por firma inválida.
    #[tokio::test]
    async fn the_default_storage_keeps_scoring_like_before_task17() {
        let db = SqliteStorage::new(":memory:", None).unwrap();
        let score = db
            .update_score(
                "peer",
                ScoreEvent::IncidentRecorded(IncidentType::InvalidSignature),
            )
            .await
            .unwrap();
        assert_eq!(score.score, 0.5);
    }

    /// `ExcessiveFailures` era el literal `0.6` y **no lo invocaba nadie**; ahora usa la misma
    /// perilla que el resto de incidentes (documentado en el `changelog`).
    #[tokio::test]
    async fn excessive_failures_uses_the_same_configured_multiplier() {
        let db = storage(0.2);
        let score = db
            .update_score(
                "peer",
                ScoreEvent::IncidentRecorded(IncidentType::ExcessiveFailures),
            )
            .await
            .unwrap();
        assert!((score.score - 0.2).abs() < 1e-9, "got {}", score.score);
    }

    /// Fraude **probado** (doble prueba en conflicto, envenenamiento de pesos) es un reset total:
    /// la clave no puede suavizarlo, ni siquiera con el valor más indulgente.
    #[tokio::test]
    async fn proven_poisoning_is_a_full_reset_regardless_of_the_penalty() {
        for incident in [
            IncidentType::ConflictingProof,
            IncidentType::WeightPoisoning,
        ] {
            let db = storage(0.9);
            let score = db
                .update_score("peer", ScoreEvent::IncidentRecorded(incident))
                .await
                .unwrap();
            assert_eq!(
                score.score, 0.0,
                "{incident:?} con slash_penalty = 0.9 debe dejar la nota en 0.0"
            );
        }
    }

    /// El camino de recuperación no ha cambiado: trabajo aceptado sube la nota y sigue topada a 1.
    #[tokio::test]
    async fn an_accepted_work_proof_recovers_the_score() {
        let db = storage(0.5);
        let lowered = db
            .update_score(
                "peer",
                ScoreEvent::IncidentRecorded(IncidentType::InvalidSignature),
            )
            .await
            .unwrap();

        let recovered = db
            .update_score("peer", ScoreEvent::WorkProofAccepted)
            .await
            .unwrap();
        assert!(
            (recovered.score - (lowered.score + SCORE_RECOVERY_PER_WORK)).abs() < 1e-9,
            "got {}",
            recovered.score
        );

        let fresh = db.get_score("another-peer").await.unwrap();
        assert_eq!(fresh.score, 1.0, "una identidad nueva arranca en 1.0");
        let top = db
            .update_score("another-peer", ScoreEvent::WorkProofAccepted)
            .await
            .unwrap();
        assert_eq!(top.score, 1.0, "…y sigue topada en 1.0");
    }
}

/// Informes de fallo: cuota y corroboración (H-11 · `TASK-22`).
///
/// S8 autenticó `/report_failure` (D-3), así que el griefing anónimo murió; quedaron dos agujeros de
/// la misma familia: `can_report_failure` **no tenía ni un consumidor** (la cuota de 3 informes por
/// hora y por par existía y nadie la aplicaba) y un informe de terceros **sin prueba** escalaba la
/// reputación del objetivo en el acto, de modo que tres POST de una sola keypair bastaban para
/// blacklistear a cualquier peer (`MusivumSlashingService::slash` blacklistea a `slash_count >= 3`).
///
/// Esta tabla (`failure_reports`) es el único sitio donde vive el historial con el que se decide la
/// corroboración, así que los tests de la política real (la que convierte los contadores en un
/// castigo o en nada) están en `musivum/src/tests.rs`; aquí se fijan las dos consultas.
#[cfg(test)]
mod failure_report_tests {
    use super::*;

    /// La ventana que la política usa (una hora). Se repite aquí para que el test fije el valor y no
    /// dependa de la constante del orquestador.
    const WINDOW: u64 = 3600;

    fn storage() -> SqliteStorage {
        SqliteStorage::new(":memory:", None).unwrap()
    }

    /// La mitad «cuota» de H-11: tres informes aceptados por `(reporter, target)` y hora, y el cuarto
    /// se rechaza. Antes de `TASK-22` esta función no la llamaba nadie en todo el workspace.
    #[tokio::test]
    async fn the_quota_allows_three_reports_and_refuses_the_fourth() {
        let db = storage();

        for accepted in 1..=3 {
            assert!(
                db.can_report_failure("reporter", "target").await.unwrap(),
                "el informe {accepted} entra en la cuota"
            );
            db.record_failure_report("reporter", "target", WINDOW)
                .await
                .unwrap();
        }

        assert!(
            !db.can_report_failure("reporter", "target").await.unwrap(),
            "el cuarto informe del mismo reportero contra el mismo objetivo no entra"
        );

        // …y la cuota es **por par**: no la gastan otros.
        assert!(
            db.can_report_failure("other-reporter", "target")
                .await
                .unwrap()
        );
        assert!(
            db.can_report_failure("reporter", "other-target")
                .await
                .unwrap()
        );
    }

    /// Los dos contadores miden cosas distintas: la cuota es del reportero, la corroboración del
    /// objetivo y exige reporteros **distintos**. Es la diferencia entre «este peer insiste» y «tres
    /// peers independientes están de acuerdo», que es justo lo que decide el castigo.
    #[tokio::test]
    async fn the_two_counters_measure_quota_and_corroboration_separately() {
        let db = storage();

        let first = db.record_failure_report("A", "T", WINDOW).await.unwrap();
        assert_eq!(
            first,
            FailureReportCounters {
                by_reporter: 1,
                distinct_reporters: 1
            }
        );

        // La misma identidad insiste: sube su cuota, **no** la corroboración.
        let second = db.record_failure_report("A", "T", WINDOW).await.unwrap();
        assert_eq!(
            second,
            FailureReportCounters {
                by_reporter: 2,
                distinct_reporters: 1
            },
            "repetir no es corroborar"
        );

        // Otra identidad sobre el mismo objetivo: la corroboración sube.
        let third = db.record_failure_report("B", "T", WINDOW).await.unwrap();
        assert_eq!(
            third,
            FailureReportCounters {
                by_reporter: 1,
                distinct_reporters: 2
            }
        );

        // Un objetivo distinto tiene su propia cuenta, y no contamina la del anterior.
        let other = db.record_failure_report("A", "U", WINDOW).await.unwrap();
        assert_eq!(
            other,
            FailureReportCounters {
                by_reporter: 1,
                distinct_reporters: 1
            }
        );
        let back_to_t = db.record_failure_report("C", "T", WINDOW).await.unwrap();
        assert_eq!(
            back_to_t,
            FailureReportCounters {
                by_reporter: 1,
                distinct_reporters: 3
            },
            "los informes sobre U no cuentan para T"
        );
    }

    /// La ventana se aplica **al contar**, no solo al escribir: con `window_secs = 0` ni el informe
    /// que se acaba de insertar entra (`timestamp > datetime('now')` es falso por igualdad). Es la
    /// forma de fijar, sin inyectar filas antiguas, que la comparación no es «todo lo que hay».
    #[tokio::test]
    async fn the_window_is_applied_when_counting() {
        let db = storage();

        db.record_failure_report("A", "T", WINDOW).await.unwrap();
        let inside = db.record_failure_report("A", "T", WINDOW).await.unwrap();
        assert_eq!(inside.by_reporter, 2, "la hora de la ventana sí los cuenta");

        let outside = db.record_failure_report("A", "T", 0).await.unwrap();
        assert_eq!(
            outside,
            FailureReportCounters {
                by_reporter: 0,
                distinct_reporters: 0
            },
            "una ventana de 0 s no cuenta ni el informe recién insertado"
        );
    }

    /// La tabla no puede crecer sin fin: al registrar un informe se podan los que ya no sirven a
    /// ninguna ventana (más de un día). Sin esta poda, `failure_reports` era la única tabla con
    /// historial y sin TTL (la corroboración solo mira la última hora).
    #[tokio::test]
    async fn recording_prunes_reports_older_than_the_retention() {
        let db = storage();

        // `conn` es privado, pero este módulo vive dentro del crate: es la forma de tener filas
        // antiguas sin esperar un día. El de 2 h se queda (podría entrar en una ventana futura más
        // ancha), los de 25 h y 48 h son basura.
        {
            let conn = db.conn.lock().unwrap();
            for hours in [48, 25, 2] {
                conn.execute(
                    "INSERT INTO failure_reports (reporter_id, target_id, timestamp) \
                     VALUES (?1, ?2, datetime('now', ?3))",
                    params!["old-reporter", "target", format!("-{hours} hours")],
                )
                .unwrap();
            }
        }

        let counters = db
            .record_failure_report("fresh-reporter", "target", WINDOW)
            .await
            .unwrap();
        assert_eq!(
            counters.distinct_reporters, 1,
            "solo el informe nuevo está dentro de la hora"
        );

        let remaining: u32 = {
            let conn = db.conn.lock().unwrap();
            conn.query_row("SELECT COUNT(*) FROM failure_reports", [], |row| row.get(0))
                .unwrap()
        };
        assert_eq!(
            remaining, 2,
            "se podan los de 48 h y 25 h; el de 2 h y el nuevo se quedan"
        );
    }
}

/// Tests of the Karma ledger persistence (`TASK-30`, SEC-06): `update_karma`/`get_all_karma` are the
/// only persistence of the balance, and they used to write to a table nobody created. Offline: no
/// daemon, no socket, no GPU.
#[cfg(test)]
mod karma_tests {
    use super::*;

    fn storage() -> SqliteStorage {
        SqliteStorage::new(":memory:", None).unwrap()
    }

    fn temp_db(name: &str) -> String {
        let mut path = std::env::temp_dir();
        path.push(format!("tessella-karma-{}-{}.db", std::process::id(), name));
        let path = path.to_string_lossy().to_string();
        remove_db_files(&path);
        path
    }

    fn remove_db_files(db: &str) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{db}{suffix}"));
        }
    }

    /// El corazón de `TASK-30`: acreditar y leer el balance deja de ser un camino que falla siempre.
    /// Y una segunda acreditación **actualiza** la fila (la clave primaria es el `peer_id`), no crea
    /// una segunda.
    #[tokio::test]
    async fn karma_round_trips_through_sqlite() {
        let db = storage();

        assert!(
            db.get_all_karma().await.unwrap().is_empty(),
            "un ledger recién creado está vacío, no roto"
        );

        db.update_karma("peerA", 120).await.unwrap();
        db.update_karma("peerB", -30).await.unwrap();
        db.update_karma("peerA", 200).await.unwrap();

        let mut records = db.get_all_karma().await.unwrap();
        records.sort();
        assert_eq!(
            records,
            vec![("peerA".to_string(), 200), ("peerB".to_string(), -30)],
            "el balance se conserva tal cual, incluido el negativo de un consumidor"
        );
    }

    /// Criterio 4 de `TASK-30`: la tabla es **aditiva**. Una base creada por un binario anterior
    /// (sin `karma_ledger`) se migra al abrirla, como el resto del esquema, sin tocar sus datos.
    #[tokio::test]
    async fn an_existing_database_without_the_table_gets_it_created() {
        let path = temp_db("migration");
        {
            let db = SqliteStorage::new(&path, None).unwrap();
            db.update_karma("peerA", 10).await.unwrap();
            db.save_node(node("peerB")).await.unwrap();
        }

        // La traza de lo que hacía el código antes de `TASK-30`: la tabla no existía.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute("DROP TABLE karma_ledger", []).unwrap();
        }

        let reopened = SqliteStorage::new(&path, None).unwrap();
        assert!(
            reopened.get_all_karma().await.unwrap().is_empty(),
            "el constructor recrea la tabla en vez de fallar con 'no such table'"
        );
        reopened.update_karma("peerA", 42).await.unwrap();
        assert_eq!(
            reopened.get_all_karma().await.unwrap(),
            vec![("peerA".to_string(), 42)]
        );
        assert_eq!(
            reopened.get_all_nodes().await.unwrap().len(),
            1,
            "la migración no toca ninguna tabla existente"
        );

        remove_db_files(&path);
    }

    fn node(peer_id: &str) -> NodeInfo {
        NodeInfo {
            peer_id: peer_id.to_string(),
            host: "127.0.0.1".to_string(),
            port: 8001,
            vram_mb: 6144,
            role: Role::Unassigned,
            model_id: "org/model".to_string(),
            p2p_address: "/ip4/127.0.0.1/tcp/9001".to_string(),
            http_api_url: None,
            relay_address: None,
            status: PeerStatus::Healthy,
            failed_requests: 0,
            successful_requests: 0,
            latency_p99_ms: 0,
            queue_depth: 0,
            last_error: None,
            slash_count: 0,
            blacklisted: false,
            public_ip: None,
            public_api_port: None,
        }
    }
}

/// Tests del ledger **federado**: el cursor de la réplica y el dedup honesto (`TASK-35`/S41, SP-02).
///
/// El hallazgo era triple y silencioso: la clave de deduplicación tenía resolución de **segundo**
/// (`(peer_id, ts, entry_type, source)`), así que dos acreditaciones legítimas del mismo peer en el
/// mismo segundo se colapsaban en una; el «cursor» era un `skip(local_height)` sobre un recuento de
/// filas locales + federadas; y `get_ledger_height()` convertía cualquier error de lectura en `0`,
/// que con aquel `skip` significaba **volver a aplicar todo**.
#[cfg(test)]
mod federated_ledger_tests {
    use super::*;

    fn storage() -> SqliteStorage {
        SqliteStorage::new(":memory:", None).unwrap()
    }

    fn temp_db(name: &str) -> String {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "tessella-federated-{}-{}.db",
            std::process::id(),
            name
        ));
        let path = path.to_string_lossy().to_string();
        remove_db_files(&path);
        path
    }

    fn remove_db_files(db: &str) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{db}{suffix}"));
        }
    }

    /// Un asiento tal como llega de un primario. La `signature` es lo que identifica el enunciado
    /// firmado (`tag` representa «emitido por esta clave en esta posición») y es de donde sale la
    /// clave de deduplicación.
    fn signed_entry(
        peer_id: &str,
        ts: u64,
        amount: f32,
        proof: &str,
        tag: u8,
    ) -> SignedLedgerEntry {
        SignedLedgerEntry {
            entry: LedgerEntry {
                peer_id: peer_id.to_string(),
                entry_type: LedgerEntryType::Earned,
                amount,
                proof: proof.to_string(),
                timestamp: ts,
                source: "local".to_string(),
            },
            signature: vec![tag; 64],
        }
    }

    fn index_columns(db: &str, index: &str) -> Vec<String> {
        let conn = Connection::open(db).unwrap();
        let mut stmt = conn
            .prepare(&format!("PRAGMA index_info({index})"))
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(2))
            .unwrap()
            .map(|c| c.unwrap())
            .collect()
    }

    fn ledger_rows(db: &str) -> i64 {
        let conn = Connection::open(db).unwrap();
        conn.query_row("SELECT COUNT(*) FROM ledger", [], |row| row.get(0))
            .unwrap()
    }

    /// Criterio 1 de `TASK-35` (SP-02): dos asientos **distintos** del mismo peer, en el mismo
    /// segundo y del mismo tipo entraban en conflicto por la clave
    /// `(peer_id, ts, entry_type, source)` —resolución de segundo— y uno se perdía en silencio.
    #[tokio::test]
    async fn distinct_federated_entries_of_the_same_second_are_all_stored() {
        let db = storage();

        let report = db
            .add_federated_entries(vec![
                signed_entry("peerA", 100, 1.0, "proof-1", 1),
                signed_entry("peerA", 100, 2.0, "proof-2", 2),
                signed_entry("peerA", 100, 3.0, "proof-3", 3),
            ])
            .await
            .unwrap();

        assert_eq!(
            report.inserted, 3,
            "las tres acreditaciones del mismo segundo son asientos distintos"
        );
        assert_eq!(report.ignored, 0);

        let page = db.get_ledger_page(0, 10).await.unwrap();
        assert_eq!(page.len(), 3, "el ledger federado conserva los tres");
        assert_eq!(
            page.iter().map(|p| p.entry.amount).collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0],
            "y en el orden en que llegaron (id ascendente)"
        );
    }

    /// Criterio 4: el **mismo** asiento firmado dos veces (un reintento del primario, una página
    /// reenviada) sí es un duplicado: se cuenta, se registra y **no** se aplica dos veces.
    #[tokio::test]
    async fn the_same_signed_entry_is_applied_once_and_reported_as_ignored() {
        let db = storage();
        let entry = signed_entry("peerA", 100, 5.0, "proof-1", 7);

        let first = db.add_federated_entries(vec![entry.clone()]).await.unwrap();
        let second = db
            .add_federated_entries(vec![entry.clone(), entry])
            .await
            .unwrap();

        assert_eq!(
            (first.inserted, first.ignored),
            (1, 0),
            "el primer envío se aplica y no ignora nada"
        );
        assert_eq!(
            (second.inserted, second.ignored),
            (0, 2),
            "el reintento se cuenta como ignorado en vez de duplicar el asiento"
        );
        assert_eq!(db.get_ledger_page(0, 10).await.unwrap().len(), 1);
    }

    /// Criterios 2 y 3: la réplica avanza por el `id` monotónico del primario (cada página es el
    /// rango `(since, last_id]`), no por un recuento local de filas, y las páginas encadenan sin
    /// atascarse ni saltarse asientos.
    #[tokio::test]
    async fn the_page_read_follows_the_cursor_and_not_the_row_count() {
        let db = storage();
        db.add_federated_entries(
            (1..=5)
                .map(|i| signed_entry("peerA", 100, i as f32, &format!("proof-{i}"), i))
                .collect(),
        )
        .await
        .unwrap();

        let first = db.get_ledger_page(0, 2).await.unwrap();
        assert_eq!(
            first.iter().map(|p| p.id).collect::<Vec<_>>(),
            vec![1, 2],
            "la primera página es el principio del log"
        );

        let second = db
            .get_ledger_page(first.last().unwrap().id, 2)
            .await
            .unwrap();
        assert_eq!(second.iter().map(|p| p.id).collect::<Vec<_>>(), vec![3, 4]);

        let third = db
            .get_ledger_page(second.last().unwrap().id, 2)
            .await
            .unwrap();
        assert_eq!(third.iter().map(|p| p.id).collect::<Vec<_>>(), vec![5]);

        let exhausted = db
            .get_ledger_page(third.last().unwrap().id, 2)
            .await
            .unwrap();
        assert!(
            exhausted.is_empty(),
            "con el cursor al final del log no hay nada más: la réplica converge"
        );
    }

    /// Criterio 3 y 5: la altura es el **final del log** (`MAX(id)`), no un contador que mezclaba
    /// filas locales y federadas, y el cursor de federación es un valor **aparte** que sobrevive a
    /// un reinicio. Un ledger vacío es `0`; un error de lectura ya no puede disfrazarse de `0`.
    #[tokio::test]
    async fn the_ledger_height_is_the_log_end_and_the_cursor_is_separate() {
        let db = storage();

        assert_eq!(db.get_ledger_height().await.unwrap(), 0, "log vacío");
        assert_eq!(db.get_federation_cursor().await.unwrap(), 0, "sin cursor");

        db.add_work("peerA", 1.0, None).await.unwrap();
        let inserted = db
            .add_federated_entries(vec![
                signed_entry("peerB", 100, 1.0, "p", 1),
                signed_entry("peerB", 100, 2.0, "p", 2),
            ])
            .await
            .unwrap()
            .inserted;
        assert_eq!(inserted, 2);

        assert_eq!(
            db.get_ledger_height().await.unwrap(),
            3,
            "la altura es el id más alto, con locales y federados en el mismo log"
        );

        db.set_federation_cursor(2).await.unwrap();
        assert_eq!(db.get_federation_cursor().await.unwrap(), 2);
        db.set_federation_cursor(3).await.unwrap();
        assert_eq!(
            db.get_federation_cursor().await.unwrap(),
            3,
            "el cursor avanza por su cuenta y no lo mueve la altura"
        );
    }

    /// El cursor sobrevive a un reinicio (si no, una réplica que se apaga volvería a pedir el ledger
    /// entero cada vez) y la migración del índice de dedup (SP-02) es aditiva e idempotente: una
    /// base escrita por el binario anterior se abre, conserva sus asientos y pasa a deduplicar por el
    /// **contenido firmado**.
    #[tokio::test]
    async fn the_cursor_persists_and_the_dedup_index_is_migrated_to_the_content_key() {
        let path = temp_db("cursor-and-index");
        {
            let db = SqliteStorage::new(&path, None).unwrap();
            db.add_federated_entries(vec![signed_entry("peerA", 100, 1.0, "p", 1)])
                .await
                .unwrap();
            db.set_federation_cursor(41).await.unwrap();
        }

        // Traza de lo que hacía el código antes de S41: el índice viejo (por segundo) y un asiento
        // federado **sin** `source_hash`, que es lo que escribía `add_federated_entries`.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute("DROP INDEX idx_ledger_dedup", []).unwrap();
            conn.execute(
                "CREATE UNIQUE INDEX idx_ledger_dedup ON ledger(peer_id, ts, entry_type, source) \
                 WHERE source = 'federated'",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO ledger (peer_id, entry_type, amount, proof, source, ts) \
                 VALUES ('legacy', 'Earned', 9.0, 'p', 'federated', 100)",
                [],
            )
            .unwrap();
        }

        let reopened = SqliteStorage::new(&path, None).unwrap();
        assert_eq!(
            reopened.get_federation_cursor().await.unwrap(),
            41,
            "el cursor sobrevive al reinicio"
        );
        assert_eq!(
            ledger_rows(&path),
            2,
            "la migración no pierde ni duplica asientos"
        );
        assert_eq!(
            index_columns(&path, "idx_ledger_dedup"),
            vec!["source".to_string(), "source_hash".to_string()],
            "el índice pasa a ser por contenido firmado, no por (peer_id, ts, entry_type, source)"
        );

        // Abrirla otra vez no cambia nada: la migración no es acumulativa.
        drop(reopened);
        let _again = SqliteStorage::new(&path, None).unwrap();
        assert_eq!(ledger_rows(&path), 2);
        assert_eq!(
            index_columns(&path, "idx_ledger_dedup"),
            vec!["source".to_string(), "source_hash".to_string()]
        );

        remove_db_files(&path);
    }

    /// Criterio 4 de `TASK-35`: cada asiento ignorado por dedup deja una línea de log **con su
    /// hash**, así que un duplicado deja de ser invisible (`INSERT OR IGNORE` no decía nada).
    ///
    /// Se captura con un subscriber **de ámbito** (`with_default`, sin tocar el global del binario
    /// de tests) sobre un runtime propio, así que el `warn!` que cuenta se verifica de verdad.
    #[test]
    fn an_ignored_entry_is_logged_with_its_hash() {
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct Captured(Arc<Mutex<Vec<u8>>>);
        impl Write for Captured {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = Captured(buffer.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_max_level(tracing::Level::WARN)
            .finish();

        let entry = signed_entry("peerA", 100, 1.0, "p", 5);
        let expected_hash = federated_source_hash(&entry).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        tracing::subscriber::with_default(subscriber, || {
            // `TASK-61`/S67: la caché de `Interest` es de proceso; se recalcula con este subscriber
            // como default del hilo para que un `never` cacheado por otro test no descarte el `warn!`.
            tracing::callsite::rebuild_interest_cache();
            runtime.block_on(async {
                let db = storage();
                db.add_federated_entries(vec![entry.clone()]).await.unwrap();
                let report = db.add_federated_entries(vec![entry.clone()]).await.unwrap();
                assert_eq!(report.ignored, 1, "el segundo envío es el duplicado");
            });
        });

        let logs = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(
            logs.contains(&expected_hash),
            "el asiento ignorado debe nombrarse por su hash ({expected_hash}); log capturado: {logs}"
        );
    }
}

/// Tests del esquema y de los **silencios** de SQLite (`TASK-42`/S48 · SP-09).
///
/// El hallazgo era de diagnóstico: 18 `let _ = conn.execute(…)` (16 migraciones y 2 escrituras de
/// reputación) hacían que «la columna ya existe» y «la base está bloqueada / el disco está lleno»
/// fueran el mismo evento — no pasaba nada, y en silencio —, así que un esquema a medio migrar solo
/// se manifestaba mucho después (el `no such table: karma_ledger` de `TASK-30` fue el mismo patrón).
#[cfg(test)]
mod sqlite_schema_tests {
    use super::*;

    fn temp_db(name: &str) -> String {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "tessella-schema-{}-{}.db",
            std::process::id(),
            name
        ));
        let path = path.to_string_lossy().to_string();
        remove_db_files(&path);
        path
    }

    fn remove_db_files(db: &str) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{db}{suffix}"));
        }
    }

    fn columns_of(conn: &Connection, table: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .unwrap();
        let mut columns: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        columns.sort();
        columns
    }

    fn index_columns_of(conn: &Connection, index: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA index_info({index})"))
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(2))
            .unwrap()
            .map(|row| row.unwrap())
            .collect()
    }

    /// Criterio 1: una columna que falta se añade, y encontrarla ya puesta **no** es un error ni una
    /// segunda migración. Antes las dos cosas eran el mismo `let _ = …`: el segundo arranque sobre la
    /// misma base era indistinguible de un `ALTER TABLE` que había fallado.
    #[test]
    fn ensure_column_adds_what_is_missing_and_leaves_the_rest_alone() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE t (a INTEGER)", []).unwrap();

        ensure_column(&conn, "t", "b", "ALTER TABLE t ADD COLUMN b TEXT").unwrap();
        assert_eq!(
            columns_of(&conn, "t"),
            vec!["a".to_string(), "b".to_string()]
        );

        ensure_column(&conn, "t", "b", "ALTER TABLE t ADD COLUMN b TEXT").unwrap();
        assert_eq!(
            columns_of(&conn, "t"),
            vec!["a".to_string(), "b".to_string()],
            "la segunda vez es un no-op, no un error"
        );
    }

    /// Buffer compartido donde un subscriber **de ámbito** escribe lo que se loguea, para poder
    /// afirmar sobre un aviso concreto (precedente de `federated_ledger_tests`, S41).
    fn capture_warnings() -> std::sync::Arc<std::sync::Mutex<Vec<u8>>> {
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))
    }

    fn with_captured_warnings(
        buffer: &std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        body: impl FnOnce(),
    ) {
        use std::io::Write;

        #[derive(Clone)]
        struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl Write for Captured {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let writer = Captured(buffer.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_max_level(tracing::Level::WARN)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            // `TASK-61`/S67: la caché de `Interest` de `tracing-core` es **de proceso** y se calcula
            // con el dispatcher del hilo que registra el callsite, así que otro test de este binario
            // (sin subscriber) puede haberla cacheado como `never` y el `warn!` se descarta antes de
            // consultar `enabled()` — el test quedaría rojo sin que el camino se haya roto. Forzar el
            // recálculo aquí, con este subscriber como default de este hilo, lo repara. Receta
            // completa (incluido el caso «el callsite aún no existe») y causa raíz medida: `TASK-61`
            // y el test `test_an_oversize_frame_logs_exactly_one_warning` de
            // `tessella-network` (que además calienta el callsite desde su propio hilo).
            tracing::callsite::rebuild_interest_cache();
            body();
        });
    }

    fn captured(buffer: &std::sync::Arc<std::sync::Mutex<Vec<u8>>>) -> String {
        String::from_utf8(buffer.lock().unwrap().clone()).unwrap()
    }

    /// Riesgo declarado y asumido: `PRAGMA table_info` **no** distingue «columna con otro tipo». El
    /// helper no migra nada por su cuenta —SQLite no cambia el tipo de una columna en sitio—, pero lo
    /// **dice**: eso es la diferencia entre un operador avisado y un `i64` guardado en un `TEXT`.
    #[test]
    fn a_divergent_column_type_is_reported_and_not_migrated() {
        let buffer = capture_warnings();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE t (amount TEXT)", []).unwrap();

        with_captured_warnings(&buffer, || {
            ensure_column(&conn, "t", "amount", "ALTER TABLE t ADD COLUMN amount REAL")
                .expect("un tipo distinto no rompe el arranque: se avisa y se sigue");
        });

        assert_eq!(
            columns_of(&conn, "t").len(),
            1,
            "no se añade una segunda columna con el mismo nombre"
        );
        let logs = captured(&buffer);
        assert!(
            logs.contains("amount") && logs.contains("TEXT") && logs.contains("REAL"),
            "el aviso debe nombrar la columna y los dos tipos; log capturado: {logs}"
        );
    }

    /// Criterio 2 (el corazón del paso): un fallo **real** de migración se propaga. Con `let _ =`, un
    /// `ALTER TABLE` sobre una tabla que no existe —o sobre una base bloqueada, o con el disco lleno—
    /// no decía nada y el proceso arrancaba con un esquema a medias que nadie miraba.
    #[test]
    fn ensure_column_propagates_a_real_failure() {
        let conn = Connection::open_in_memory().unwrap();

        let error = ensure_column(
            &conn,
            "no_such_table",
            "b",
            "ALTER TABLE no_such_table ADD COLUMN b TEXT",
        )
        .expect_err("un ALTER TABLE imposible no puede ser un no-op silencioso");

        let message = error.to_string();
        assert!(
            message.contains("no_such_table") && message.contains("ALTER TABLE"),
            "el error debe nombrar la tabla y la sentencia: {message}"
        );
    }

    /// Una base escrita por un binario anterior (columnas nuevas ausentes y los nombres viejos de
    /// `weight_manifests`) se migra al abrirla, y abrirla otra vez no cambia nada. Es el guard que
    /// hace segura la decisión de propagar errores: lo normal (la columna ya está) sigue siendo un
    /// no-op.
    #[test]
    fn a_legacy_database_is_migrated_and_reopening_it_is_a_no_op() {
        let path = temp_db("legacy");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE discovered_nodes (
                     peer_id TEXT PRIMARY KEY, host TEXT, port INTEGER, vram_mb INTEGER,
                     role TEXT, model_id TEXT, p2p_address TEXT, relay_address TEXT,
                     last_seen DATETIME DEFAULT CURRENT_TIMESTAMP);
                 CREATE TABLE ledger (
                     id INTEGER PRIMARY KEY AUTOINCREMENT, peer_id TEXT, entry_type TEXT,
                     amount REAL, proof TEXT, request_id TEXT, output_hash BLOB,
                     source TEXT DEFAULT 'local', source_hash TEXT);
                 CREATE TABLE ledger_meta (key TEXT PRIMARY KEY, value INTEGER);
                 CREATE TABLE weight_manifests (
                     model_id TEXT NOT NULL, role TEXT NOT NULL, blake3_hash BLOB NOT NULL,
                     size_bytes INTEGER NOT NULL, registered_at INTEGER NOT NULL,
                     musivum_public_key BLOB NOT NULL, musivum_signature BLOB NOT NULL,
                     PRIMARY KEY (model_id, role));
                 CREATE TABLE transfer_receipts (
                     id INTEGER PRIMARY KEY AUTOINCREMENT, sender_peer_id TEXT NOT NULL,
                     receiver_peer_id TEXT NOT NULL, model_id TEXT NOT NULL, role TEXT NOT NULL,
                     blake3_hash BLOB NOT NULL, timestamp INTEGER NOT NULL,
                     receipt_json TEXT NOT NULL);",
            )
            .unwrap();
        }

        let db = SqliteStorage::new(&path, None).expect("una base antigua se migra al abrirla");
        drop(db);

        let conn = Connection::open(&path).unwrap();
        let discovered = columns_of(&conn, "discovered_nodes");
        for column in [
            "status",
            "http_api_url",
            "public_ip",
            "public_api_port",
            "queue_depth",
        ] {
            assert!(
                discovered.contains(&column.to_string()),
                "falta {column}: {discovered:?}"
            );
        }
        let manifests = columns_of(&conn, "weight_manifests");
        assert!(manifests.contains(&"tracker_public_key".to_string()));
        assert!(manifests.contains(&"filename".to_string()));
        assert!(!manifests.contains(&"musivum_public_key".to_string()));

        // Abrirla otra vez es un no-op: todas las migraciones encuentran su columna ya puesta.
        drop(SqliteStorage::new(&path, None).expect("el segundo arranque no migra nada"));
        drop(conn);
        remove_db_files(&path);
    }

    /// Y si la tabla no es la que este binario espera —no tiene **ninguno** de los dos nombres—, el
    /// arranque lo dice en vez de dejar un esquema roto en silencio (que es lo que hacía el `let _ =`).
    #[test]
    fn a_weight_manifests_table_without_either_column_refuses_to_boot() {
        let path = temp_db("broken-manifests");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE weight_manifests (
                     model_id TEXT NOT NULL, role TEXT NOT NULL, blake3_hash BLOB NOT NULL,
                     size_bytes INTEGER NOT NULL, registered_at INTEGER NOT NULL,
                     PRIMARY KEY (model_id, role));",
            )
            .unwrap();
        }

        let error = match SqliteStorage::new(&path, None) {
            Ok(_) => panic!("una tabla que no es la esperada no puede abrirse en silencio"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains("weight_manifests")
                && message.contains("tracker_public_key")
                && message.contains("musivum_public_key"),
            "el error debe nombrar la tabla y los dos nombres: {message}"
        );

        remove_db_files(&path);
    }

    /// Criterio 5: la ventana de corroboración de `TASK-22` se consulta por `(reporter_id,
    /// target_id)` + `timestamp`, y hasta S48 esa tabla no tenía ningún índice.
    #[test]
    fn the_failure_reports_table_has_its_corroboration_index() {
        let path = temp_db("failure-index");
        drop(SqliteStorage::new(&path, None).unwrap());

        let conn = Connection::open(&path).unwrap();
        assert_eq!(
            index_columns_of(&conn, "idx_failure_reports"),
            vec![
                "reporter_id".to_string(),
                "target_id".to_string(),
                "timestamp".to_string()
            ],
            "el índice que la política de corroboración necesita"
        );

        drop(conn);
        remove_db_files(&path);
    }

    /// Criterio 5: `busy_timeout`, para que un escritor concurrente (otro proceso abriendo la misma
    /// base, el checkpoint del WAL, el CLI del operador) **espere** en vez de fallar al instante —
    /// que con `let _ =` era, además, un fallo que no se veía.
    #[test]
    fn the_connection_waits_for_a_locked_writer() {
        let path = temp_db("busy-timeout");
        let db = SqliteStorage::new(&path, None).unwrap();

        let timeout: i64 = db
            .conn
            .lock()
            .unwrap()
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            timeout, SQLITE_BUSY_TIMEOUT_MS as i64,
            "el timeout es de conexión: hay que leerlo de la conexión del propio almacenamiento"
        );

        drop(db);
        remove_db_files(&path);
    }
}

/// Tests de **lectura** del ledger: qué devuelve y qué deja de ser silencioso (`TASK-42`/S48).
///
/// El endpoint público de inspección (`GET /ledger`) leía `ORDER BY id ASC LIMIT ?1`, es decir el
/// **pasado remoto** del ledger, y descartaba con `entries.flatten()` cualquier fila que no pudiera
/// mapear. Lo segundo importa más que lo primero: un asiento ilegible desaparecía de la vista sin
/// dejar rastro, así que «el ledger está completo» era una afirmación que el código no podía sostener.
#[cfg(test)]
mod ledger_read_tests {
    use super::*;

    fn storage() -> SqliteStorage {
        SqliteStorage::new(":memory:", None).unwrap()
    }

    /// Subscriber **de ámbito** que escribe en el buffer (mismo patrón que
    /// `federated_ledger_tests::an_ignored_entry_is_logged_with_its_hash`).
    fn subscriber(
        buffer: &std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    ) -> impl tracing::Subscriber + Send + Sync + 'static {
        use std::io::Write;

        #[derive(Clone)]
        struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl Write for Captured {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let writer = Captured(buffer.clone());
        tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_max_level(tracing::Level::WARN)
            .finish()
    }

    /// Inserta una fila que **no** se puede mapear: el `amount` es texto, así que leerlo como número
    /// falla. Es el caso que `flatten()` convertía en «no había nada que ver».
    fn insert_unreadable_row(db: &SqliteStorage) {
        let conn = db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO ledger (peer_id, entry_type, amount, proof, source, ts) \
             VALUES ('broken', 'Earned', 'not-a-number', 'p', 'local', 7)",
            [],
        )
        .unwrap();
    }

    /// SP-09: el endpoint de inspección devolvía las N filas **más antiguas**, así que enseñaba el
    /// pasado remoto del ledger. Ahora lo más reciente va primero y el `limit` corta por ese lado.
    #[tokio::test]
    async fn the_ledger_view_returns_the_most_recent_entries_first() {
        let db = storage();
        db.add_work("oldest", 1.0, None).await.unwrap();
        db.add_work("middle", 2.0, None).await.unwrap();
        db.record_consumption("newest", 3.0, None).await.unwrap();

        let all = db.get_ledger(10).await.unwrap();
        assert_eq!(
            all.iter().map(|e| e.peer_id.as_str()).collect::<Vec<_>>(),
            vec!["newest", "middle", "oldest"],
            "lo más reciente primero"
        );

        let one = db.get_ledger(1).await.unwrap();
        assert_eq!(one.len(), 1, "el límite se aplica a lo más reciente");
        assert_eq!(one[0].peer_id, "newest");
        assert_eq!(one[0].entry_type, LedgerEntryType::Spent);
        assert_eq!(one[0].amount, 3.0);
        assert_eq!(one[0].source, "local");
    }

    /// Y una fila que no se puede leer **se nombra**: las filas buenas siguen saliendo (el endpoint no
    /// se queda mudo) y el operador lee qué se quedó fuera y por qué.
    #[test]
    fn a_row_that_cannot_be_read_is_reported_instead_of_dropped() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let entries = tracing::subscriber::with_default(subscriber(&buffer), || {
            // `TASK-61`/S67: recalcula la caché de `Interest` (de proceso) con este subscriber como
            // default del hilo, para que un `never` cacheado por otro test no descarte el `warn!`.
            tracing::callsite::rebuild_interest_cache();
            runtime.block_on(async {
                let db = storage();
                db.add_work("good", 1.0, None).await.unwrap();
                insert_unreadable_row(&db);
                db.get_ledger(10).await.unwrap()
            })
        });

        assert_eq!(
            entries.len(),
            1,
            "la fila legible sigue saliendo: {entries:?}"
        );
        assert_eq!(entries[0].peer_id, "good");

        let logs = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(
            logs.contains("cannot be read") && logs.contains("amount"),
            "la fila ilegible debe nombrarse con su posición y el motivo (columna `amount`); \
             log capturado: {logs}"
        );
    }

    /// Criterio 4: un tipo de asiento que este binario no conoce conserva el fallback histórico
    /// (`Spent`) —cambiar el sentido de un movimiento sería peor— pero **se ve** en el log. La
    /// conversión de verdad vive en `LedgerEntryType::to_db`/`from_db` (protocolo, con su test).
    #[test]
    fn an_unknown_entry_type_keeps_the_historical_fallback_and_says_so() {
        assert_eq!(parse_ledger_entry_type("Earned"), LedgerEntryType::Earned);
        assert_eq!(parse_ledger_entry_type("Spent"), LedgerEntryType::Spent);
        assert_eq!(parse_ledger_entry_type("Slashed"), LedgerEntryType::Slashed);

        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let parsed = tracing::subscriber::with_default(subscriber(&buffer), || {
            // `TASK-61`/S67: misma razón que en `a_row_that_cannot_be_read_is_reported_instead_of_dropped`.
            tracing::callsite::rebuild_interest_cache();
            parse_ledger_entry_type("Withdrawal")
        });

        assert_eq!(parsed, LedgerEntryType::Spent, "el fallback no cambia");
        let logs = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(
            logs.contains("Withdrawal"),
            "el texto desconocido debe aparecer en el log; capturado: {logs}"
        );
    }

    #[tokio::test]
    async fn count_pending_work_proofs_and_metrics_hook() {
        use std::sync::Arc;
        use tessella_core::models::{SignedWorkProof, WorkProof, WorkTicket};
        use tessella_core::telemetry::WorkerMetrics;

        let metrics = Arc::new(WorkerMetrics::new());
        let db = storage().with_metrics(metrics.clone());

        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 0);
        assert_eq!(
            metrics
                .proofs_queued
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );

        let proof = SignedWorkProof {
            proof: WorkProof {
                request_id: "req-wal-1".to_string(),
                worker_id: "worker-1".to_string(),
                credits_delta: 1.0,
                output_hash: [0u8; 32],
                timestamp: 100,
                input_hash: [0u8; 32],
                layer_range: (0, 2),
                work_ticket: WorkTicket {
                    request_id: "req-wal-1".to_string(),
                    model_id: "model-x".to_string(),
                    entry_peer_id: "worker-1".to_string(),
                    issued_at: 100,
                    expires_at: 200,
                    audit_layer: None,
                    audit_replicas: 2,
                    audit_canary: None,
                    audit_reward: 0.0,
                    orchestrator_signature: vec![0; 64],
                },
            },
            signature: vec![1, 2, 3],
            public_key: vec![4, 5, 6],
        };

        let id = db.save_pending_work_proof(&proof).await.unwrap();
        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 1);
        assert_eq!(
            metrics
                .proofs_queued
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        db.delete_pending_work_proofs(vec![id]).await.unwrap();
        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 0);
        assert_eq!(
            metrics
                .proofs_queued
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn worker_assignment_version_persists_and_is_monotonic() {
        let db = SqliteStorage::new(":memory:", None).unwrap();
        assert_eq!(db.get_assignment_version().await.unwrap(), 0);
        assert_eq!(db.get_assignment_version_sync().unwrap(), 0);

        db.set_assignment_version(5).await.unwrap();
        assert_eq!(db.get_assignment_version().await.unwrap(), 5);
        assert_eq!(db.get_assignment_version_sync().unwrap(), 5);

        // Stale/lower version does not regress the stored value (MAX)
        db.set_assignment_version(3).await.unwrap();
        assert_eq!(db.get_assignment_version().await.unwrap(), 5);
        assert_eq!(db.get_assignment_version_sync().unwrap(), 5);

        // Higher version advances it
        db.set_assignment_version(10).await.unwrap();
        assert_eq!(db.get_assignment_version().await.unwrap(), 10);
        assert_eq!(db.get_assignment_version_sync().unwrap(), 10);
    }

    #[tokio::test]
    async fn worker_active_assignment_persists_and_restores() {
        let db = SqliteStorage::new(":memory:", None).unwrap();
        assert!(db.get_active_assignment().await.unwrap().is_none());
        assert!(db.get_active_assignment_sync().unwrap().is_none());
        assert_eq!(db.get_assignment_version().await.unwrap(), 0);

        let assignment_v5 = WorkerAssignment {
            layer_ranges: vec![(0, 14)],
            model_repo: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            replica_group_id: 1,
            expert_range: None,
        };

        db.set_active_assignment(5, &assignment_v5).await.unwrap();

        // Verifies recovery matches the persisted version and payload (both async and sync)
        assert_eq!(
            db.get_active_assignment().await.unwrap(),
            Some(assignment_v5.clone())
        );
        assert_eq!(
            db.get_active_assignment_sync().unwrap(),
            Some(assignment_v5.clone())
        );
        assert_eq!(db.get_assignment_version().await.unwrap(), 5);
        assert_eq!(db.get_assignment_version_sync().unwrap(), 5);

        // Stale/lower version (v3) does not overwrite the assignment or regress the version
        let assignment_stale = WorkerAssignment {
            layer_ranges: vec![(0, 4)],
            model_repo: "Obsolete-Model".to_string(),
            replica_group_id: 99,
            expert_range: None,
        };
        db.set_active_assignment(3, &assignment_stale)
            .await
            .unwrap();

        assert_eq!(
            db.get_active_assignment().await.unwrap(),
            Some(assignment_v5.clone())
        );
        assert_eq!(db.get_assignment_version().await.unwrap(), 5);

        // Higher version (v12) updates both version and payload
        let assignment_v12 = WorkerAssignment {
            layer_ranges: vec![(14, 28)],
            model_repo: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            replica_group_id: 2,
            expert_range: Some((0, 8)),
        };
        db.set_active_assignment(12, &assignment_v12).await.unwrap();

        assert_eq!(
            db.get_active_assignment().await.unwrap(),
            Some(assignment_v12.clone())
        );
        assert_eq!(
            db.get_active_assignment_sync().unwrap(),
            Some(assignment_v12)
        );
        assert_eq!(db.get_assignment_version().await.unwrap(), 12);
        assert_eq!(db.get_assignment_version_sync().unwrap(), 12);
    }
}
