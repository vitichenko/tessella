use rusqlite::Connection;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tessella_core::error::DomainError;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::SqliteStorage;

/// Default cadence of the WAL maintenance task (`TASK-13`, H-21).
pub const WAL_CHECKPOINT_INTERVAL_SECS: u64 = 300;

/// What a `PRAGMA wal_checkpoint(TRUNCATE)` run did, as reported by SQLite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalCheckpoint {
    /// `true` when another connection was reading and the checkpoint could not finish.
    pub busy: bool,
    /// Frames the WAL held when the checkpoint started.
    pub log_frames: i64,
    /// Frames that were moved into the main database file.
    pub checkpointed_frames: i64,
}

/// Consolidates the write-ahead log into the main database file and truncates it.
pub async fn checkpoint_wal_conn(
    conn: Arc<Mutex<Connection>>,
) -> Result<WalCheckpoint, DomainError> {
    tokio::task::spawn_blocking(move || {
        let conn = conn
            .lock()
            .map_err(|e| DomainError::Storage(format!("DB lock poisoned: {}", e)))?;
        let (busy, log_frames, checkpointed_frames) = conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(|e| DomainError::Storage(format!("WAL checkpoint failed: {}", e)))?;

        Ok(WalCheckpoint {
            busy: busy != 0,
            log_frames,
            checkpointed_frames,
        })
    })
    .await
    .map_err(|e| DomainError::Storage(format!("Blocking task failed: {}", e)))?
}

/// Handle of the running WAL maintenance task (`TASK-13`, H-21).
pub struct WalMaintenance {
    cancel_token: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl WalMaintenance {
    /// Starts the periodic checkpoint, or returns `None` when `interval` is zero.
    pub fn spawn(storage: Arc<SqliteStorage>, interval: Duration) -> Option<Self> {
        if interval.is_zero() {
            info!(
                "🧹 WAL maintenance disabled (wal_checkpoint_interval_secs = 0): the -wal file is \
                 only consolidated when the process closes the database"
            );
            return None;
        }

        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(run_wal_maintenance(storage, interval, cancel_token.clone()));
        info!(
            "🧹 WAL maintenance started: PRAGMA wal_checkpoint(TRUNCATE) every {:?}",
            interval
        );

        Some(Self {
            cancel_token,
            handle,
        })
    }

    /// Asks the task to stop. Idempotent and safe to call from anywhere.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// `true` once the task has left its loop (cancelled or not).
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Cancels the task and waits for it.
    pub async fn shutdown(self) {
        self.cancel_token.cancel();
        if let Err(e) = self.handle.await {
            warn!("⚠️ WAL maintenance task did not stop cleanly: {}", e);
        }
    }
}

/// The loop of [`WalMaintenance`]: checkpoint, wait, repeat — and leave at once when cancelled.
async fn run_wal_maintenance(
    storage: Arc<SqliteStorage>,
    interval: Duration,
    cancel_token: CancellationToken,
) {
    let mut ticker = tokio::time::interval(interval);
    let mut first_run = true;

    loop {
        tokio::select! {
            _ = ticker.tick() => match storage.checkpoint_wal().await {
                Ok(outcome) => {
                    if first_run {
                        info!(
                            "🧹 WAL checkpoint: {} frames consolidated, {} left in the -wal (busy={})",
                            outcome.checkpointed_frames, outcome.log_frames, outcome.busy
                        );
                        first_run = false;
                    } else {
                        debug!(
                            "🧹 WAL checkpoint: {} frames consolidated, {} left in the -wal (busy={})",
                            outcome.checkpointed_frames, outcome.log_frames, outcome.busy
                        );
                    }
                }
                Err(e) => {
                    warn!("⚠️ WAL checkpoint failed (retrying in {:?}): {}", interval, e);
                }
            },
            _ = cancel_token.cancelled() => {
                info!("🛑 WAL maintenance task stopped by cancellation token");
                break;
            }
        }
    }
}
