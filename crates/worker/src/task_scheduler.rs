use crate::inference_engine::InferenceEngine;
use crate::runtime::WorkerRuntime;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

pub struct TaskScheduler {
    engine: Arc<InferenceEngine>,
    storage: Arc<tessella_storage::SqliteStorage>,
    runtime: Arc<WorkerRuntime>,
}

impl TaskScheduler {
    pub fn new(
        engine: Arc<InferenceEngine>,
        storage: Arc<tessella_storage::SqliteStorage>,
        runtime: Arc<WorkerRuntime>,
    ) -> Self {
        Self {
            engine,
            storage,
            runtime,
        }
    }

    /// Starts the two background loops of the worker and returns their `JoinHandle`s.
    ///
    /// `TASK-10` Part A: every loop shares `cancel_token` and every handle is handed back, so
    /// `WorkerService::run` can cancel them and wait for them (they used to be detached spawns
    /// whose handles were dropped on the floor).
    ///
    /// There were three until S45 (`TASK-39`): the `BatchCollector` flush loop was spawned when
    /// `[batching] enabled = true`, but nothing ever submitted a request to the collector (its only
    /// producer, `WorkerService::distributed_forward`, had zero call sites), so it never flushed.
    pub fn start(
        self: Arc<Self>,
        scheduler_rx: mpsc::Receiver<crate::scheduler::SchedulerCommand>,
        cancel_token: CancellationToken,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let mut handles = Vec::with_capacity(2);

        // 1. Spawn the priority-QoS scheduler (each hop applies its sessions one by one).
        let engine_cl = self.engine.clone();
        let cancel_scheduler = cancel_token.clone();
        handles.push(tokio::spawn(async move {
            let scheduler = crate::scheduler::Scheduler::new(engine_cl, scheduler_rx);
            scheduler.run(cancel_scheduler).await;
        }));

        // 2. Spawn WAL sweeper loop
        let self_cl = self.clone();
        let cancel_token_cl = cancel_token.clone();
        handles.push(tokio::spawn(async move {
            self_cl.run_sweeper_loop(cancel_token_cl).await;
        }));

        handles
    }

    async fn run_sweeper_loop(self: Arc<Self>, cancel_token: CancellationToken) {
        let storage = self.storage.clone();
        let orchestrator_url = self.runtime.config.orchestrator_url.clone();
        let client = self.runtime.http_client.clone();

        let metrics = self.runtime.metrics.clone();

        // Immediately run once on startup
        if let Err(e) = Self::sweep_once(&storage, &orchestrator_url, &client, Some(&metrics)).await
        {
            error!("❌ Initial WAL sweep failed: {}", e);
        }

        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    info!("🛑 Sweeper loop stopped by cancellation token");
                    break;
                }
                _ = interval.tick() => {
                    if let Err(e) = Self::sweep_once(&storage, &orchestrator_url, &client, Some(&metrics)).await {
                        error!("❌ WAL sweep failed: {}", e);
                    }
                }
            }
        }
    }

    pub async fn sweep_once(
        storage: &tessella_storage::SqliteStorage,
        orchestrator_url: &str,
        client: &reqwest::Client,
        metrics: Option<&tessella_core::telemetry::WorkerMetrics>,
    ) -> Result<()> {
        let pending = storage
            .get_pending_work_proofs()
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        if pending.is_empty() {
            return Ok(());
        }

        info!(
            "📤 Found {} pending work proofs. Sweeping...",
            pending.len()
        );

        let (ids, proofs): (Vec<i64>, Vec<_>) = pending.into_iter().unzip();

        let url = format!(
            "{}/api/v1/work_done_batch",
            orchestrator_url.trim_end_matches('/')
        );
        let start = std::time::Instant::now();
        let res = client.post(&url).json(&proofs).send().await;
        let duration = start.elapsed().as_secs_f64();

        if let Some(m) = metrics {
            m.record_batch_upload_duration(duration);
        }

        let res = match res {
            Ok(resp) => resp,
            Err(e) => {
                if let Some(m) = metrics {
                    m.inc_batch_uploads_failure();
                }
                return Err(e.into());
            }
        };

        if res.status().is_success() {
            info!(
                "✅ Successfully uploaded batch of {} work proofs",
                proofs.len()
            );
            if let Some(m) = metrics {
                m.inc_batch_uploads_success();
                m.inc_proofs_settled(proofs.len() as u64);
            }
            storage
                .delete_pending_work_proofs(ids)
                .await
                .map_err(|e| anyhow::anyhow!(e))?;
        } else {
            if let Some(m) = metrics {
                m.inc_batch_uploads_failure();
            }
            let status = res.status();
            let err_text = res.text().await.unwrap_or_default();
            anyhow::bail!(
                "Orchestrator rejected batch upload ({}): {}",
                status,
                err_text
            );
        }

        Ok(())
    }
}
