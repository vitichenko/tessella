use candle_core::Tensor;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tessella_protocol::models::QueuePriority;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;
use tracing::info;

use async_trait::async_trait;
use tessella_core::error::DomainError;
use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait LayerRunner: Send + Sync {
    async fn run_single(
        &self,
        session_key: &str,
        range: (usize, usize),
        seq_start_pos: usize,
        hidden_tensor: Tensor,
    ) -> Result<Tensor, DomainError>;
}

pub struct SchedulerCommand {
    pub request_id: String,
    /// Key of the K/V cache this step belongs to (`TASK-75`/`S82`): the client's continuity id when
    /// there is one, the work id otherwise. It is **not** `request_id`: the work id changes every turn
    /// of a conversation, so using it as the key of a session would open a new K/V per turn.
    pub session_key: String,
    /// The range of this node that serves this step (`TASK-96`/S102): the range that covers the
    /// layer the hop continues from, resolved by the engine before the command is queued. A node
    /// with two ranges has two backends and two K/V caches, so the step cannot be dispatched
    /// without saying which one it belongs to.
    pub range: (usize, usize),
    pub seq_start_pos: usize,
    pub hidden_tensor: Tensor,
    pub priority: QueuePriority,
    pub timestamp: Instant,
    pub response_tx: oneshot::Sender<Result<Tensor, DomainError>>,
}

pub struct ActiveSession {
    pub request_id: String,
    pub session_key: String,
    pub range: (usize, usize),
    pub seq_start_pos: usize,
    pub hidden_tensor: Tensor,
    pub priority: QueuePriority,
    pub timestamp: Instant,
    pub response_tx: Option<oneshot::Sender<Result<Tensor, DomainError>>>,
}

impl PartialEq for ActiveSession {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.timestamp == other.timestamp
    }
}

impl Eq for ActiveSession {}

impl PartialOrd for ActiveSession {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActiveSession {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => {
                // Older requests (smaller timestamp) have higher priority in the heap
                other.timestamp.cmp(&self.timestamp)
            }
            ord => ord,
        }
    }
}

pub struct Scheduler {
    runner: Arc<dyn LayerRunner>,
    rx: mpsc::Receiver<SchedulerCommand>,
    active_sessions: BinaryHeap<ActiveSession>,
}

impl Scheduler {
    pub fn new(runner: Arc<dyn LayerRunner>, rx: mpsc::Receiver<SchedulerCommand>) -> Self {
        Self {
            runner,
            rx,
            active_sessions: BinaryHeap::new(),
        }
    }

    /// Runs the priority-queue loop until the command channel closes or `cancel_token` fires.
    ///
    /// `TASK-10` Part A: the loop was started with `tokio::spawn` and neither its caller nor the
    /// worker's shutdown path could reach it (its `JoinHandle` was dropped and it ignored the
    /// token that `WorkerService::run` already created).
    pub async fn run(mut self, cancel_token: CancellationToken) {
        info!(
            "🚀 Priority-QoS Scheduler started: each hop applies its sessions one by one \
             (per-node batching is not implemented, see TASK-39/S45)"
        );

        loop {
            let wait_time = if self.active_sessions.is_empty() {
                Duration::from_secs(60) // wait indefinitely until a request comes
            } else {
                Duration::from_millis(5) // small timeout so concurrent arrivals travel in one pass
            };

            let received = tokio::select! {
                _ = cancel_token.cancelled() => {
                    info!("🛑 Scheduler stopped by cancellation token");
                    break;
                }
                received = timeout(wait_time, self.rx.recv()) => received,
            };

            match received {
                Ok(Some(cmd)) => {
                    self.active_sessions.push(ActiveSession {
                        request_id: cmd.request_id,
                        session_key: cmd.session_key,
                        range: cmd.range,
                        seq_start_pos: cmd.seq_start_pos,
                        hidden_tensor: cmd.hidden_tensor,
                        priority: cmd.priority,
                        timestamp: cmd.timestamp,
                        response_tx: Some(cmd.response_tx),
                    });

                    // Drain any immediately available
                    while let Ok(cmd) = self.rx.try_recv() {
                        self.active_sessions.push(ActiveSession {
                            request_id: cmd.request_id,
                            session_key: cmd.session_key,
                            range: cmd.range,
                            seq_start_pos: cmd.seq_start_pos,
                            hidden_tensor: cmd.hidden_tensor,
                            priority: cmd.priority,
                            timestamp: cmd.timestamp,
                            response_tx: Some(cmd.response_tx),
                        });
                    }
                }
                Ok(None) => break, // channel closed
                Err(_) => {}       // timeout: execute whatever is already queued
            }

            if self.active_sessions.is_empty() {
                continue;
            }

            self.process_batch().await;
        }
    }

    /// Runs every queued session, one after another, in the order [`execution_order`] decides.
    ///
    /// `TASK-39`/S45: this used to hand the decode steps to `run_batched`, which grouped them by
    /// `seq_start_pos`, built a `Tensor::cat` of their hidden states **and threw it away** before
    /// running the very same sessions one by one. Per-node batching is a feature, not a fix: it
    /// needs a KV cache per sequence and per-row positions, while `apply_layers` takes a single
    /// `seq_start_pos` and a session owns a single cache — so the concatenation was wasted work
    /// (one op plus VRAM proportional to `tokens × hidden` per group) *and* the seed of a
    /// numerically wrong answer had anyone ever consumed it. What is left is what the node really
    /// does: a priority queue of sessions, each of them applied by its own hop.
    async fn process_batch(&mut self) {
        let mut queued = Vec::with_capacity(self.active_sessions.len());
        while let Some(session) = self.active_sessions.pop() {
            queued.push(session);
        }

        for mut session in execution_order(queued) {
            let tx = session.response_tx.take();
            let res = self
                .runner
                .run_single(
                    &session.session_key,
                    session.range,
                    session.seq_start_pos,
                    session.hidden_tensor,
                )
                .await;
            if let Some(tx) = tx {
                let _ = tx.send(res);
            }
        }
    }
}

/// Order in which the sessions of one pass are executed: every **prefill** before any **decode
/// step**, keeping the priority order the heap handed them over in.
///
/// A session is a decode step when its hidden state carries a single sequence position
/// (`dim(1) == 1`); anything longer is a prefill, which is the compute-bound half of the work and
/// reaches the node once per prompt, so it goes first. Nothing is merged: each session is applied
/// by its own hop with its own KV cache and its own position (`TASK-39`/S45).
///
/// Pure on purpose: the decision is testable without a backend, a `Swarm` or a GPU.
pub fn execution_order(queued: Vec<ActiveSession>) -> Vec<ActiveSession> {
    let (prefills, decodes): (Vec<ActiveSession>, Vec<ActiveSession>) = queued
        .into_iter()
        .partition(|session| session.hidden_tensor.dim(1).unwrap_or(0) != 1);
    prefills.into_iter().chain(decodes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const HIDDEN: usize = 4;

    /// A session whose hidden state declares —through its sequence length— which kind it is:
    /// `dim(1) == 1` is a decode step, anything longer is a prefill.
    fn session(
        request_id: &str,
        seq_len: usize,
        priority: QueuePriority,
        timestamp: Instant,
    ) -> ActiveSession {
        ActiveSession {
            request_id: request_id.to_string(),
            // The fixtures do not distinguish the two ids: a client that sends no `session_id` ends up
            // with the work id as the key (`TASK-75`/`S82`), and the queue does not read either of them.
            session_key: request_id.to_string(),
            range: (0, 4),
            seq_start_pos: 0,
            hidden_tensor: Tensor::zeros((1, seq_len, HIDDEN), DType::F32, &Device::Cpu)
                .expect("a tiny CPU tensor"),
            priority,
            timestamp,
            response_tx: None,
        }
    }

    fn ids(sessions: &[ActiveSession]) -> Vec<&str> {
        sessions.iter().map(|s| s.request_id.as_str()).collect()
    }

    #[test]
    fn prefills_are_served_before_decodes_and_no_session_is_lost() {
        let now = Instant::now();
        let queued = vec![
            session("decode-a", 1, QueuePriority::Standard, now),
            session("prefill", 5, QueuePriority::Standard, now),
            session("decode-b", 1, QueuePriority::Standard, now),
        ];

        let order = execution_order(queued);

        assert_eq!(
            ids(&order),
            vec!["prefill", "decode-a", "decode-b"],
            "the compute-bound prefill goes first and the priority order of the heap survives"
        );
        assert_eq!(order.len(), 3, "every queued session must be executed");
    }

    /// The case the retired `run_batched` treated as a batch: two decode steps that share a
    /// `seq_start_pos`. It concatenated their hidden states with `Tensor::cat` and then threw the
    /// concatenation away to run them one by one (`TASK-39`/SP-06), so the regression this test
    /// locks is that nothing is merged any more: both sessions keep their own `(1, 1, hidden)`
    /// tensor, which is what `run_single` needs (one session owns one KV cache and one position).
    #[test]
    fn two_decodes_sharing_a_position_are_never_merged() {
        let mut a = session("decode-a", 1, QueuePriority::Standard, Instant::now());
        let mut b = session("decode-b", 1, QueuePriority::Standard, Instant::now());
        a.seq_start_pos = 7;
        b.seq_start_pos = 7;

        let order = execution_order(vec![a, b]);

        assert_eq!(ids(&order), vec!["decode-a", "decode-b"]);
        for session in &order {
            assert_eq!(
                session.hidden_tensor.dims(),
                &[1, 1, HIDDEN],
                "a merged (2, 1, hidden) tensor would mean the sessions were batched together"
            );
        }
    }

    /// The queue is what the node really has: the tier decides first and, inside a tier, the
    /// oldest request wins. This is the semantic the startup line announces.
    #[test]
    fn the_queue_serves_the_top_tier_and_then_the_oldest() {
        let now = Instant::now();
        let mut queue: BinaryHeap<ActiveSession> = BinaryHeap::new();
        queue.push(session("leech-new", 1, QueuePriority::Leech, now));
        queue.push(session("vip-old", 1, QueuePriority::Vip, now));
        queue.push(session("standard", 1, QueuePriority::Standard, now));
        queue.push(session(
            "vip-newer",
            1,
            QueuePriority::Vip,
            now + Duration::from_millis(5),
        ));

        let mut served = Vec::new();
        while let Some(session) = queue.pop() {
            served.push(session.request_id);
        }

        assert_eq!(
            served,
            vec!["vip-old", "vip-newer", "standard", "leech-new"]
        );
    }

    #[derive(Default)]
    struct MockLayerRunner {
        invocations: std::sync::atomic::AtomicUsize,
        should_fail: std::sync::atomic::AtomicBool,
    }

    impl MockLayerRunner {
        fn failing() -> Self {
            Self {
                invocations: std::sync::atomic::AtomicUsize::new(0),
                should_fail: std::sync::atomic::AtomicBool::new(true),
            }
        }
    }

    #[async_trait]
    impl LayerRunner for MockLayerRunner {
        async fn run_single(
            &self,
            _session_key: &str,
            _range: (usize, usize),
            _seq_start_pos: usize,
            hidden_tensor: Tensor,
        ) -> Result<Tensor, DomainError> {
            self.invocations
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.should_fail.load(std::sync::atomic::Ordering::SeqCst) {
                Err(DomainError::Inference("mock runner failure".into()))
            } else {
                Ok(hidden_tensor)
            }
        }
    }

    #[tokio::test]
    async fn scheduler_dispatches_to_runner_and_returns_response() {
        let runner = Arc::new(MockLayerRunner::default());
        let (tx, rx) = mpsc::channel(10);
        let scheduler = Scheduler::new(runner.clone(), rx);

        let cancel_token = CancellationToken::new();
        let cancel_clone = cancel_token.clone();
        let handle = tokio::spawn(async move {
            scheduler.run(cancel_clone).await;
        });

        let (resp_tx, resp_rx) = oneshot::channel();
        let tensor = Tensor::zeros((1, 1, HIDDEN), DType::F32, &Device::Cpu).unwrap();
        tx.send(SchedulerCommand {
            request_id: "req-1".into(),
            session_key: "sess-1".into(),
            range: (0, 4),
            seq_start_pos: 0,
            hidden_tensor: tensor.clone(),
            priority: QueuePriority::Standard,
            timestamp: Instant::now(),
            response_tx: resp_tx,
        })
        .await
        .unwrap();

        let res = resp_rx.await.unwrap().expect("inference succeeds");
        assert_eq!(res.dims(), tensor.dims());
        assert_eq!(
            runner.invocations.load(std::sync::atomic::Ordering::SeqCst),
            1
        );

        cancel_token.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn scheduler_propagates_runner_error() {
        let runner = Arc::new(MockLayerRunner::failing());
        let (tx, rx) = mpsc::channel(10);
        let scheduler = Scheduler::new(runner.clone(), rx);

        let cancel_token = CancellationToken::new();
        let cancel_clone = cancel_token.clone();
        let handle = tokio::spawn(async move {
            scheduler.run(cancel_clone).await;
        });

        let (resp_tx, resp_rx) = oneshot::channel();
        let tensor = Tensor::zeros((1, 1, HIDDEN), DType::F32, &Device::Cpu).unwrap();
        tx.send(SchedulerCommand {
            request_id: "req-2".into(),
            session_key: "sess-2".into(),
            range: (0, 4),
            seq_start_pos: 0,
            hidden_tensor: tensor,
            priority: QueuePriority::Vip,
            timestamp: Instant::now(),
            response_tx: resp_tx,
        })
        .await
        .unwrap();

        let res = resp_rx.await.unwrap();
        assert!(res.is_err(), "runner error must propagate to response_tx");
        assert_eq!(
            runner.invocations.load(std::sync::atomic::Ordering::SeqCst),
            1
        );

        cancel_token.cancel();
        handle.await.unwrap();
    }
}
