//! One vocabulary for stopping long-lived tasks, shared by both workspaces (H-08).
//!
//! The helper moved here in S35 (`TASK-29`) from `tessella-worker::shutdown` (S14, `TASK-10` Part
//! A), because Musivum needs the **identical** policy for its own loops (federation sync, its SSE
//! pusher, the autoscaler) and cannot depend on the worker crate. Before that, each owner of a
//! background task implemented its own variant, and the steps that fixed H-08 kept discovering new
//! detached `tokio::spawn`s. There is now one definition and one set of tests:
//!
//! * [`tick_or_cancel`] — how a periodic loop waits for its next tick.
//! * [`sleep_or_cancel`] — the same for one-off waits (a reconnect backoff, a poll interval).
//! * [`cancel_and_drain`] — cancel the token **first**, then join every handle inside a bounded
//!   grace; a task that ignores the token is aborted and counted instead of holding the process
//!   hostage.
//!
//! The worker's `crate::shutdown` re-exports these names, so its call sites and documentation did
//! not change when the implementation moved. Nothing here builds a `Swarm`, binds a socket or
//! touches the GPU, so every branch is unit tested offline.

use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

/// How long a process waits for its background tasks after cancelling them.
///
/// The `SIGTERM` path of both binaries is this drain plus whatever their own teardown costs, and
/// `scripts/e2e/run_tests.sh` gives a process a bounded grace before `SIGKILL`, so the budget is
/// deliberately small: every task it covers is parked on a timer, not on GPU work.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// What happened to the background tasks during shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrainOutcome {
    /// Tasks that finished before the deadline. Includes a task that had already finished and a
    /// task that ended by panicking: the shutdown path must survive both.
    pub completed: usize,
    /// Tasks still running when the deadline expired; they were aborted and forgotten, because a
    /// stuck task must never hold the process hostage.
    pub abandoned: usize,
}

impl DrainOutcome {
    /// `true` when every task stopped inside the grace period.
    pub fn all_completed(&self) -> bool {
        self.abandoned == 0
    }
}

/// Awaits the next tick of `interval` and reports whether it arrived before the cancellation.
///
/// `false` means "the token was cancelled, leave the loop". Both futures are cancel-safe and every
/// periodic loop of the stack waits through this helper, so cancelling the shared token stops the
/// worker's coordinator, batch processor and topology refresh — and Musivum's federation and
/// autoscaler loops — the same way.
pub async fn tick_or_cancel(
    interval: &mut tokio::time::Interval,
    cancel_token: &CancellationToken,
) -> bool {
    tokio::select! {
        _ = interval.tick() => true,
        _ = cancel_token.cancelled() => false,
    }
}

/// Sleeps for `duration` and reports whether the wait finished before the cancellation.
///
/// The one-off counterpart of [`tick_or_cancel`], for the waits that are not on a fixed cadence: a
/// reconnect backoff of up to 300 s would otherwise delay a shutdown by minutes.
pub async fn sleep_or_cancel(duration: Duration, cancel_token: &CancellationToken) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(duration) => true,
        _ = cancel_token.cancelled() => false,
    }
}

/// Cancels `cancel_token` and waits for every handle, bounded by [`SHUTDOWN_GRACE`].
pub async fn cancel_and_drain(
    cancel_token: &CancellationToken,
    handles: Vec<JoinHandle<()>>,
) -> DrainOutcome {
    cancel_and_drain_within(cancel_token, handles, SHUTDOWN_GRACE).await
}

/// [`cancel_and_drain`] with an explicit budget, so tests can assert the bound.
///
/// The token is cancelled **first** (that is what actually stops the loops; the handles are only
/// the confirmation), and every handle shares a single deadline, so N stuck tasks cost the same
/// wait as one.
pub async fn cancel_and_drain_within(
    cancel_token: &CancellationToken,
    handles: Vec<JoinHandle<()>>,
    grace: Duration,
) -> DrainOutcome {
    cancel_token.cancel();

    let deadline = tokio::time::Instant::now() + grace;
    let mut outcome = DrainOutcome::default();

    for mut handle in handles {
        match tokio::time::timeout_at(deadline, &mut handle).await {
            Ok(Ok(())) => outcome.completed += 1,
            Ok(Err(join_error)) => {
                warn!(
                    "⚠️ A background task did not stop cleanly during shutdown: {}",
                    join_error
                );
                outcome.completed += 1;
            }
            Err(_) => {
                warn!(
                    "⏳ A background task is still running after the {:?} shutdown grace; aborting it",
                    grace
                );
                handle.abort();
                outcome.abandoned += 1;
            }
        }
    }

    info!(
        "🧹 Shutdown: {} background task(s) joined, {} abandoned",
        outcome.completed, outcome.abandoned
    );
    outcome
}

#[cfg(test)]
mod tests {
    use super::{
        DrainOutcome, cancel_and_drain, cancel_and_drain_within, sleep_or_cancel, tick_or_cancel,
    };
    use std::time::{Duration, Instant};
    use tokio_util::sync::CancellationToken;

    /// Acceptance criterion of `TASK-10`: cancelling the token must stop a long-lived loop far
    /// earlier than the wait it is parked on (it used to have no cancellation path at all).
    ///
    /// The bound is **generous on purpose** (`S56`, §0.6 exception (ii), same treatment as the two
    /// flakes S49 and S67 fixed): 200 ms measured the machine, not the cancellation — the whole
    /// workspace's test binaries running in parallel made this assertion fail once while every single
    /// run in isolation passed (10/10 at ~10 ms). What must never pass is a 300 s wait that survives
    /// its cancellation, and no scheduling delay comes near 5 s.
    const BUDGET: Duration = Duration::from_secs(5);

    /// A far-away ticker, so only the cancellation can make the helper return.
    fn parked_interval() -> tokio::time::Interval {
        let mut interval = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_secs(600),
            Duration::from_secs(600),
        );
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        interval
    }

    #[tokio::test]
    async fn tick_or_cancel_keeps_a_live_loop_waiting() {
        let mut interval = tokio::time::interval(Duration::from_millis(5));
        interval.tick().await; // consume the immediate first tick
        let cancel_token = CancellationToken::new();

        assert!(
            tick_or_cancel(&mut interval, &cancel_token).await,
            "a live token must not stop the loop"
        );
        assert!(!cancel_token.is_cancelled());
    }

    /// `TASK-29`: un backoff de reconexión de minutos no puede retrasar el apagado. Mismo contrato
    /// que `tick_or_cancel`, sobre una espera de una sola vez.
    #[tokio::test]
    async fn sleep_or_cancel_returns_as_soon_as_the_token_is_cancelled() {
        let cancel_token = CancellationToken::new();
        let canceller = cancel_token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            canceller.cancel();
        });

        let started = Instant::now();
        assert!(
            !sleep_or_cancel(Duration::from_secs(300), &cancel_token).await,
            "a cancelled token must cut the wait short"
        );
        assert!(
            started.elapsed() < BUDGET,
            "300 s of backoff must not survive a cancellation"
        );

        // A wait that finishes on its own reports `true` and does not cancel anything.
        let live = CancellationToken::new();
        assert!(sleep_or_cancel(Duration::from_millis(1), &live).await);
        assert!(!live.is_cancelled());
    }

    #[tokio::test]
    async fn cancel_and_drain_stops_a_cancel_aware_task_within_the_budget() {
        let cancel_token = CancellationToken::new();
        let task_token = cancel_token.clone();
        let mut interval = parked_interval();
        let handle = tokio::spawn(async move {
            while tick_or_cancel(&mut interval, &task_token).await {
                // the topology refresh body would run here
            }
        });
        // Let the task park on its tick before cancelling it.
        tokio::time::sleep(Duration::from_millis(20)).await;

        let started = Instant::now();
        let outcome = cancel_and_drain(&cancel_token, vec![handle]).await;

        assert_eq!(
            outcome,
            DrainOutcome {
                completed: 1,
                abandoned: 0
            },
            "the task listens to the token, so it must be joined, not abandoned"
        );
        assert!(
            started.elapsed() < BUDGET,
            "the JoinHandle must resolve in <200 ms after cancel()"
        );
    }

    #[tokio::test]
    async fn cancel_and_drain_abandons_a_task_that_ignores_the_token() {
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async {
            // A task that never looks at the token must not hold the shutdown hostage.
            tokio::time::sleep(Duration::from_secs(600)).await;
        });

        let started = Instant::now();
        let outcome =
            cancel_and_drain_within(&cancel_token, vec![handle], Duration::from_millis(50)).await;

        assert_eq!(
            outcome,
            DrainOutcome {
                completed: 0,
                abandoned: 1
            }
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "the grace must bound how long the shutdown waits"
        );
        assert!(
            cancel_token.is_cancelled(),
            "draining must cancel the token"
        );
    }

    #[tokio::test]
    async fn cancel_and_drain_survives_a_panicking_task() {
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async { panic!("a background task blew up") });

        let outcome = cancel_and_drain(&cancel_token, vec![handle]).await;

        assert_eq!(
            outcome,
            DrainOutcome {
                completed: 1,
                abandoned: 0
            },
            "a panicking task must not take the shutdown path down with it"
        );
        assert!(outcome.all_completed());
    }

    #[tokio::test]
    async fn cancel_and_drain_cancels_the_token_even_without_handles() {
        let cancel_token = CancellationToken::new();

        let outcome = cancel_and_drain(&cancel_token, Vec::new()).await;

        assert_eq!(outcome, DrainOutcome::default());
        assert!(outcome.all_completed());
        assert!(
            cancel_token.is_cancelled(),
            "the loop that shares the token must be told to stop"
        );
    }
}
