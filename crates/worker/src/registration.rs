//! Bounded retry of the **boot** registration (`TASK-62`/S68, decision `D-25`).
//!
//! S21 promised that «the network survives the tracker going down» (D-4=(b)) and S58 measured with
//! real daemons that only half of it was true: the **topology** survives (DHT + `known_peers` +
//! routing from the cache, 0 `Pipeline timeout`), but **serving needs the control plane** (every
//! session asks for its `session_grant`, D-7/D-13) and a node that restarted during the outage **died
//! at boot**: `WorkerService::new` called `register_and_assign(…) ?` and a failed `POST /register`
//! propagated, so the process exited `1` and its persistent peer cache was never used.
//!
//! Decision D-25 keeps the refusal — the mesh does **not** serve without the orchestrator's
//! authorisation — and fixes the other half: a node that cannot reach its authority **waits** for it
//! with exponential backoff and a bounded budget instead of dying on the first attempt. Two rules,
//! both derived from D-25:
//!
//! 1. **Reachability is retried, a refusal is not.** A tracker that could not be reached has not
//!    spoken; a tracker that answered `403 worker_not_authorized` or `409 model_at_capacity` has, and
//!    waiting will not change its mind. It is the same asymmetry `fetch_session_grant` applies per
//!    session (a non-2xx is fatal; the network case is what a budget exists for).
//! 2. **The budget is finite and named.** `registration_grace_secs` (default
//!    [`DEFAULT_REGISTRATION_GRACE_SECS`], `0` = the pre-S68 fail-fast boot) is the operator-visible
//!    number, and exhausting it is a boot failure that says how long it waited.
use std::time::Duration;

/// How long a node waits for the control plane before giving up at boot, when the operator does not
/// say otherwise. It is the default of the `registration_grace_secs` key.
pub const DEFAULT_REGISTRATION_GRACE_SECS: u64 = 300;

/// Largest budget an operator may declare. A boot that hangs longer than an hour is a
/// misconfiguration (a wrong `orchestrator_url`, a tracker that will not come back), so it is refused
/// when the config is loaded instead of being accepted in silence.
pub const MAX_REGISTRATION_GRACE_SECS: u64 = 3600;

/// First wait between two registration attempts.
pub const REGISTRATION_INITIAL_BACKOFF: Duration = Duration::from_secs(2);

/// Ceiling of the backoff: after a few attempts the node probes every 30 s instead of doubling
/// forever, so a tracker that comes back ten minutes later is still found quickly.
pub const REGISTRATION_MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Why a registration attempt failed.
///
/// The distinction **is** the policy: only [`RegistrationError::Unreachable`] is retried, because
/// only that variant proves the authority never answered.
#[derive(Debug, thiserror::Error)]
pub enum RegistrationError {
    /// The tracker could not be reached at all (connection refused/reset/timeout/DNS).
    #[error("the tracker at {url} could not be reached: {source}")]
    Unreachable {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    /// The tracker answered and refused the registration (`4xx`/`5xx`). Waiting would not help.
    #[error("the tracker refused the registration (status {status}): {body}")]
    Refused { status: u16, body: String },
    /// The tracker answered something this node cannot use (an unreadable body, an unusable device
    /// or a signature that could not be produced). Retrying would repeat the same failure.
    #[error("the tracker answered something this node cannot use: {message}")]
    Malformed { message: String },
    /// The budget ran out: every attempt so far was a reachability failure.
    #[error(
        "gave up after {budget_secs} s of retries: the tracker at {url} never became reachable \
         (last error: {last})"
    )]
    GaveUp {
        url: String,
        budget_secs: u64,
        last: Box<RegistrationError>,
    },
}

impl RegistrationError {
    /// Whether waiting and asking again could give a different answer.
    ///
    /// `false` for a refusal (the authority spoke), for a malformed answer (the answer would be the
    /// same) and for an exhausted budget (it is already the final error).
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Unreachable { .. })
    }
}

/// The bounded-retry policy of the boot registration.
#[derive(Debug, Clone, Copy)]
pub struct RegistrationGrace {
    /// Total time the node may spend waiting for the tracker. `Duration::ZERO` = disabled
    /// (`registration_grace_secs = 0`), i.e. the pre-S68 behaviour: one attempt, fail fast.
    pub budget: Duration,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl RegistrationGrace {
    /// Builds the policy from the configured number of seconds.
    #[must_use]
    pub fn from_secs(budget_secs: u64) -> Self {
        Self {
            budget: Duration::from_secs(budget_secs),
            initial_backoff: REGISTRATION_INITIAL_BACKOFF,
            max_backoff: REGISTRATION_MAX_BACKOFF,
        }
    }

    /// `registration_grace_secs = 0`: no retries at all.
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.budget.is_zero()
    }

    /// How long to wait after `failed_attempts` consecutive reachability failures.
    ///
    /// Exponential from [`REGISTRATION_INITIAL_BACKOFF`] and capped at [`Self::max_backoff`]; the
    /// wait after the first failure is the initial one.
    #[must_use]
    pub fn backoff_after(&self, failed_attempts: u32) -> Duration {
        let doublings = failed_attempts.saturating_sub(1).min(16);
        let grown = self
            .initial_backoff
            .checked_mul(1u32.checked_shl(doublings).unwrap_or(u32::MAX))
            .unwrap_or(self.max_backoff);
        grown.min(self.max_backoff)
    }
}

impl Default for RegistrationGrace {
    fn default() -> Self {
        Self::from_secs(DEFAULT_REGISTRATION_GRACE_SECS)
    }
}

/// Runs `attempt` until it succeeds, is refused, or the budget runs out.
///
/// `attempt` receives the 1-based number of the attempt so the caller can log it. The function takes
/// the future from a closure so the policy is testable without a tracker: the tests drive it with a
/// counter and a zero backoff.
///
/// # Errors
///
/// The last error of `attempt` when it is not retryable, or [`RegistrationError::GaveUp`] when the
/// budget is exhausted by reachability failures.
pub async fn register_with_grace<T, F, Fut>(
    grace: RegistrationGrace,
    tracker_url: &str,
    mut attempt: F,
) -> Result<T, RegistrationError>
where
    F: FnMut(u32) -> Fut,
    Fut: std::future::Future<Output = Result<T, RegistrationError>>,
{
    let deadline = tokio::time::Instant::now() + grace.budget;
    let mut attempt_number = 0u32;
    loop {
        attempt_number += 1;
        match attempt(attempt_number).await {
            Ok(value) => {
                if attempt_number > 1 {
                    tracing::info!(
                        "✅ Registered with the tracker at {tracker_url} after {attempt_number} \
                         attempt(s): it was unreachable for a while at boot (TASK-62/S68)"
                    );
                }
                return Ok(value);
            }
            Err(error) if grace.is_disabled() || !error.is_retryable() => {
                // Either waiting is forbidden (`0`) or the authority answered: the original error is
                // what the operator has to see, not a retry story.
                tracing::error!("❌ {error}");
                return Err(error);
            }
            Err(error) => {
                let now = tokio::time::Instant::now();
                if now >= deadline {
                    tracing::error!(
                        "❌ Giving up on the registration after {}s of retries ({} attempts): \
                         {error}. The mesh serves nothing without the control plane (D-25); raise \
                         `registration_grace_secs` to wait longer at boot.",
                        grace.budget.as_secs(),
                        attempt_number
                    );
                    return Err(RegistrationError::GaveUp {
                        url: tracker_url.to_string(),
                        budget_secs: grace.budget.as_secs(),
                        last: Box::new(error),
                    });
                }
                // The last wait never overshoots the deadline: the node stops probing when the budget
                // is over instead of after it.
                let wait = grace.backoff_after(attempt_number).min(deadline - now);
                tracing::warn!(
                    "⚠️ Registration attempt {attempt_number} could not reach the tracker \
                     ({error}); retrying in {:.1}s (budget: {}s, TASK-62/S68)",
                    wait.as_secs_f32(),
                    grace.budget.as_secs()
                );
                tokio::time::sleep(wait).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// A `reqwest::Error` to build the retryable variant with: an unparseable port never opens a
    /// socket, so the test stays offline.
    fn transport_error() -> reqwest::Error {
        reqwest::Client::new()
            .get("http://127.0.0.1:99999/")
            .build()
            .expect_err("a port outside the u16 range must fail to build the request")
    }

    fn test_grace(budget_ms: u64) -> RegistrationGrace {
        RegistrationGrace {
            budget: Duration::from_millis(budget_ms),
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }

    /// The retryable variant is the one that proves the authority did not answer; everything else is
    /// terminal. The whole policy is that predicate.
    #[test]
    fn only_a_tracker_that_could_not_be_reached_is_retryable() {
        let unreachable = RegistrationError::Unreachable {
            url: "http://127.0.0.1:8080".to_string(),
            source: transport_error(),
        };
        assert!(unreachable.is_retryable());

        let refused = RegistrationError::Refused {
            status: 403,
            body: "{\"error\":\"worker_not_authorized\"}".to_string(),
        };
        assert!(!refused.is_retryable());
        assert!(refused.to_string().contains("403"));
        assert!(refused.to_string().contains("worker_not_authorized"));

        let malformed = RegistrationError::Malformed {
            message: "unreadable body".to_string(),
        };
        assert!(!malformed.is_retryable());

        let gave_up = RegistrationError::GaveUp {
            url: "http://127.0.0.1:8080".to_string(),
            budget_secs: 5,
            last: Box::new(RegistrationError::Malformed {
                message: "unreadable body".to_string(),
            }),
        };
        assert!(!gave_up.is_retryable());
        assert!(gave_up.to_string().contains("5 s"), "{gave_up}");
    }

    /// The wait doubles from the initial value and stops at the ceiling, so a tracker that is down for
    /// minutes is not probed thousands of times and one that comes back is still found quickly.
    #[test]
    fn the_backoff_grows_exponentially_and_is_capped() {
        let grace = RegistrationGrace::default();
        let waits: Vec<u64> = (1..=8).map(|n| grace.backoff_after(n).as_secs()).collect();
        assert_eq!(waits, vec![2, 4, 8, 16, 30, 30, 30, 30]);
        assert_eq!(grace.budget, Duration::from_secs(300));
        assert_eq!(
            grace.budget,
            Duration::from_secs(DEFAULT_REGISTRATION_GRACE_SECS)
        );
        assert!(!grace.is_disabled());
    }

    #[tokio::test]
    async fn an_unreachable_tracker_is_retried_until_the_first_success() {
        let attempts = Arc::new(AtomicU32::new(0));
        let seen = attempts.clone();
        let result: Result<&str, RegistrationError> = register_with_grace(
            test_grace(2000),
            "http://tracker:8080",
            move |attempt_number| {
                let seen = seen.clone();
                async move {
                    let done = seen.fetch_add(1, Ordering::SeqCst) + 1;
                    assert_eq!(done, attempt_number, "the attempt number must be 1-based");
                    if done < 3 {
                        Err(RegistrationError::Unreachable {
                            url: "http://tracker:8080".to_string(),
                            source: transport_error(),
                        })
                    } else {
                        Ok("assignment")
                    }
                }
            },
        )
        .await;

        assert_eq!(result.expect("the third attempt succeeds"), "assignment");
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "it must retry while the tracker is unreachable"
        );
    }

    /// A refusal is a decision: retrying it would keep a node in a boot loop for the whole budget and
    /// hide the real answer (`worker_not_authorized`, `model_at_capacity`) from the operator.
    #[tokio::test]
    async fn a_refused_registration_is_not_retried() {
        let attempts = Arc::new(AtomicU32::new(0));
        let seen = attempts.clone();
        let result: Result<(), RegistrationError> =
            register_with_grace(test_grace(2000), "http://tracker:8080", move |_| {
                let seen = seen.clone();
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Err(RegistrationError::Refused {
                        status: 409,
                        body: "model_at_capacity".to_string(),
                    })
                }
            })
            .await;

        match result {
            Err(RegistrationError::Refused { status, .. }) => assert_eq!(status, 409),
            other => panic!("a refusal must be returned as-is, got {other:?}"),
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "exactly one attempt");
    }

    /// `registration_grace_secs = 0` is the escape hatch: the pre-S68 boot, one attempt and the
    /// original error (not a `GaveUp` that would hide it).
    #[tokio::test]
    async fn a_disabled_grace_attempts_once_and_keeps_the_original_error() {
        let attempts = Arc::new(AtomicU32::new(0));
        let seen = attempts.clone();
        let result: Result<(), RegistrationError> = register_with_grace(
            RegistrationGrace::from_secs(0),
            "http://tracker:8080",
            move |_| {
                let seen = seen.clone();
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Err(RegistrationError::Malformed {
                        message: "unusable device".to_string(),
                    })
                }
            },
        )
        .await;

        assert!(matches!(result, Err(RegistrationError::Malformed { .. })));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(RegistrationGrace::from_secs(0).is_disabled());
    }

    /// Running out of budget says how long it waited and why it stopped: the operator has to be able
    /// to tell «the tracker was never reachable» from «the tracker said no».
    #[tokio::test]
    async fn an_exhausted_budget_gives_up_naming_the_wait() {
        let attempts = Arc::new(AtomicU32::new(0));
        let seen = attempts.clone();
        let result: Result<(), RegistrationError> =
            register_with_grace(test_grace(120), "http://tracker:8080", move |_| {
                let seen = seen.clone();
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Err(RegistrationError::Unreachable {
                        url: "http://tracker:8080".to_string(),
                        source: transport_error(),
                    })
                }
            })
            .await;

        match result {
            Err(RegistrationError::GaveUp { url, .. }) => {
                assert_eq!(url, "http://tracker:8080");
            }
            other => panic!("an unreachable tracker must end in GaveUp, got {other:?}"),
        }
        assert!(
            attempts.load(Ordering::SeqCst) >= 2,
            "it must keep trying until the budget runs out"
        );
    }
}
