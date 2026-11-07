//! The worker's periodic conversation with its authority, and what it says when it cannot have it
//! (`TASK-65`/S71, family of `D-25`).
//!
//! `TASK-62`/S68 closed one half of this family: the **boot** no longer dies in silence and tells a
//! tracker that cannot be *reached* (`RegistrationError::Unreachable`, retried with a bounded budget)
//! from one that *answers and refuses* (`403 worker_not_authorized`, `409 model_at_capacity`, fatal
//! and named). The **heartbeat** did the opposite in three places of the same block:
//!
//! 1. the answer of `POST /announce` was only inspected for `404`/`410` — a `401`/`403`/`5xx` and,
//!    above all, a **transport failure** (the tracker down) fell into `_ => {}`, so a node that had
//!    been unable to announce itself for an hour looked exactly like a healthy one;
//! 2. the re-registration was `if let Ok(…)`, so its failure said nothing;
//! 3. `notify_ready` was `let _ =`.
//!
//! This module is the decision that block needs, as pure functions and one small state machine, so
//! it can be tested without a tracker, a socket or a GPU. The same shape as
//! `musivum_api::failure_reports::failure_report_effect` (S28) and
//! `tessella_worker::scheduler::execution_order` (S45).
//!
//! Two rules, both derived from `D-25`:
//!
//! 1. **Only a lost state re-registers.** A refusal is the tracker's decision; answering it with a
//!    blind `POST /register` every 60 s is the useless loop D-25 refused at boot.
//! 2. **One line per transition.** The heartbeat is a periodic loop, so the operator-visible policy
//!    is «the change is announced once» (and so is the recovery), never «one line per attempt».

use std::fmt;

use crate::registration::RegistrationError;

/// Longest excerpt of a tracker body (or of a transport error) kept for a log line.
pub const MAX_DETAIL_CHARS: usize = 200;

/// What came back from one `POST /announce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnounceAnswer {
    /// The tracker answered with this HTTP status.
    Status(u16),
    /// The request produced no answer at all (connection refused/reset/timeout/DNS).
    NoAnswer,
}

/// What the heartbeat understands from that answer.
///
/// The variants are the outcomes the operator has to be able to tell apart (`D-25`), plus the
/// healthy one. `detail` is whatever the tracker said, already bounded by [`preview`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnounceOutcome {
    /// The tracker accepted the heartbeat.
    Healthy,
    /// The tracker answered that it does not know this node (`404`/`410`): re-register.
    LostState { status: u16 },
    /// The tracker answered and refused (`4xx`/`5xx`), or answered something this node cannot use.
    Refused {
        status: Option<u16>,
        detail: Option<String>,
    },
    /// The tracker could not be reached at all.
    Unreachable { detail: Option<String> },
}

/// What the heartbeat must do about an outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatAction {
    /// Nothing: the tracker accepted the heartbeat.
    Continue,
    /// The tracker lost this node's state: register again.
    ReRegister,
    /// Report it and change nothing else. Re-registering blindly against a refusal (or against a
    /// tracker that is down) is the loop `D-25` refused at boot.
    ReportOnly,
}

/// The health of this node's conversation with its authority, as far as the heartbeat knows.
///
/// It is deliberately coarser than [`AnnounceOutcome`]: the detail of a single answer does not change
/// the state, so an error message that varies slightly between two ticks (a timeout with a different
/// duration, a body with a new request id) does not become two log lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlPlaneState {
    /// The tracker accepts this node's heartbeats.
    Healthy,
    /// The tracker does not know this node (`404`/`410`).
    LostState,
    /// The tracker answers and refuses this node (or answers something unusable).
    Refused { status: Option<u16> },
    /// The request never reached an answer.
    Unreachable,
}

impl fmt::Display for ControlPlaneState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::LostState => write!(f, "the tracker lost our state"),
            Self::Refused {
                status: Some(status),
            } => write!(f, "refused with status {status}"),
            Self::Refused { status: None } => write!(f, "refused with an unusable answer"),
            Self::Unreachable => write!(f, "unreachable"),
        }
    }
}

/// The single log line the heartbeat owes the operator, or `None` when nothing changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeartbeatNotice {
    /// The node was announcing itself and cannot any more (or fails in a different way now).
    Lost {
        /// The state it came from; `None` on the very first observation (the boot registered).
        previous: Option<ControlPlaneState>,
    },
    /// The node announces itself again.
    Recovered { previous: ControlPlaneState },
}

impl HeartbeatNotice {
    /// The sentence to log, with the reason of the current outcome.
    #[must_use]
    pub fn message(&self, outcome: &AnnounceOutcome) -> String {
        match self {
            Self::Lost { previous } => format!(
                "cannot announce itself any more — {}. {}This node keeps routing from its cached \
                 mesh view, but the tracker will not count it alive until the announce succeeds \
                 again",
                outcome.describe(),
                previous
                    .as_ref()
                    .map(|previous| format!("Previous state: {previous}. "))
                    .unwrap_or_default()
            ),
            Self::Recovered { previous } => format!(
                "announcing itself again — {} (previous state: {previous})",
                outcome.describe()
            ),
        }
    }
}

impl AnnounceOutcome {
    /// The health state this outcome maps to.
    #[must_use]
    pub fn state(&self) -> ControlPlaneState {
        match self {
            Self::Healthy => ControlPlaneState::Healthy,
            Self::LostState { .. } => ControlPlaneState::LostState,
            Self::Refused { status, .. } => ControlPlaneState::Refused { status: *status },
            Self::Unreachable { .. } => ControlPlaneState::Unreachable,
        }
    }

    /// What to do about it.
    #[must_use]
    pub fn action(&self) -> HeartbeatAction {
        match self {
            Self::Healthy => HeartbeatAction::Continue,
            Self::LostState { .. } => HeartbeatAction::ReRegister,
            Self::Refused { .. } | Self::Unreachable { .. } => HeartbeatAction::ReportOnly,
        }
    }

    /// One sentence naming the outcome and, when there is one, what the tracker said.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Healthy => "the tracker accepted the heartbeat".to_string(),
            Self::LostState { status } => {
                format!("the tracker no longer knows this node (status {status})")
            }
            Self::Refused { status, detail } => {
                let base = match status {
                    Some(status) => format!("the tracker refused the heartbeat (status {status})"),
                    None => "the tracker answered something this node cannot use".to_string(),
                };
                match detail.as_deref().filter(|detail| !detail.is_empty()) {
                    Some(detail) => format!("{base}: {detail}"),
                    None => base,
                }
            }
            Self::Unreachable { detail } => {
                let base = "the tracker could not be reached".to_string();
                match detail.as_deref().filter(|detail| !detail.is_empty()) {
                    Some(detail) => format!("{base}: {detail}"),
                    None => base,
                }
            }
        }
    }
}

/// Remembers the last state so a tracker that is down for hours writes **one** line, not one per
/// tick.
#[derive(Debug, Default, Clone)]
pub struct HeartbeatHealth {
    last: Option<ControlPlaneState>,
}

impl HeartbeatHealth {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The state as of the last observation.
    #[must_use]
    pub fn state(&self) -> Option<&ControlPlaneState> {
        self.last.as_ref()
    }

    /// Records a new observation and returns the line worth printing, **only when something
    /// changed**.
    ///
    /// The first healthy observation is silent on purpose: the boot path already said the node
    /// registered (`✅ Registered with the tracker …`), so the first thing the heartbeat has to say
    /// is a problem.
    pub fn observe(&mut self, outcome: &AnnounceOutcome) -> Option<HeartbeatNotice> {
        let now = outcome.state();
        let previous = self.last.replace(now.clone());
        match previous {
            // The same state, however many ticks go by: nothing worth a line.
            Some(previous) if previous == now => None,
            // Back to normal: the recovery is the other half of the pair.
            Some(previous) if now == ControlPlaneState::Healthy => {
                Some(HeartbeatNotice::Recovered { previous })
            }
            // A different failure (or the first one) is a new fact for the operator.
            Some(previous) => Some(HeartbeatNotice::Lost {
                previous: Some(previous),
            }),
            // The boot path already announced the successful registration, so the heartbeat's first
            // healthy observation is silent.
            None if now == ControlPlaneState::Healthy => None,
            None => Some(HeartbeatNotice::Lost { previous: None }),
        }
    }
}

/// The decision for one answer of `POST /announce`.
///
/// `200`-series is health, `404`/`410` mean the tracker lost this node (re-register), and **everything
/// else that answers** is a refusal: a `401`/`403` (signature or admission), a `5xx`, a `3xx`. Only a
/// transport failure (`NoAnswer`) is the tracker being absent, which is a different problem for the
/// operator and a different (retryable) one for the node.
#[must_use]
pub fn announce_outcome(answer: AnnounceAnswer) -> AnnounceOutcome {
    match answer {
        AnnounceAnswer::NoAnswer => AnnounceOutcome::Unreachable { detail: None },
        AnnounceAnswer::Status(status) if (200..300).contains(&status) => AnnounceOutcome::Healthy,
        AnnounceAnswer::Status(lost @ (404 | 410)) => AnnounceOutcome::LostState { status: lost },
        AnnounceAnswer::Status(status) => AnnounceOutcome::Refused {
            status: Some(status),
            detail: None,
        },
    }
}

/// The same vocabulary as the boot path (`TASK-62`/S68), for the answers the **re-registration** gave.
///
/// An unreachable tracker is a reachability problem and one that answered (or answered something
/// unusable) is a decision — the asymmetry `RegistrationError::is_retryable` encodes at boot, applied
/// to the same states. The classification is exhaustive on purpose: a new variant of
/// `RegistrationError` breaks the build here instead of being classified in silence.
#[must_use]
pub fn registration_outcome(error: &RegistrationError) -> AnnounceOutcome {
    match error {
        RegistrationError::Refused { status, body } => AnnounceOutcome::Refused {
            status: Some(*status),
            detail: Some(preview(body)),
        },
        RegistrationError::Malformed { message } => AnnounceOutcome::Refused {
            status: None,
            detail: Some(preview(message)),
        },
        // The authority never spoke: it was unreachable, or its budget ran out on reachability
        // failures. Both are the same health state.
        RegistrationError::Unreachable { .. } | RegistrationError::GaveUp { .. } => {
            AnnounceOutcome::Unreachable {
                detail: Some(preview(&error.to_string())),
            }
        }
    }
}

/// Collapses the whitespace of a tracker answer/error and cuts it to [`MAX_DETAIL_CHARS`] chars
/// (never in the middle of a UTF-8 character) so a body cannot open a multi-line log entry.
#[must_use]
pub fn preview(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= MAX_DETAIL_CHARS {
        return collapsed;
    }
    let kept: String = collapsed.chars().take(MAX_DETAIL_CHARS).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(status: u16) -> AnnounceOutcome {
        announce_outcome(AnnounceAnswer::Status(status))
    }

    /// A `reqwest::Error` for the retryable variant: an unparseable port never opens a socket, so the
    /// test stays offline. Same trick as `registration.rs`.
    fn transport_error() -> reqwest::Error {
        reqwest::Client::new()
            .get("http://127.0.0.1:99999/")
            .build()
            .expect_err("a port outside the u16 range must fail to build the request")
    }

    /// What every answer of `/announce` means. The rows that used to fall into `_ => {}` are exactly
    /// the ones that now have a state of their own.
    #[test]
    fn the_answer_of_announce_is_classified_by_what_it_proves() {
        for healthy in [200u16, 201, 204, 299] {
            assert_eq!(status(healthy), AnnounceOutcome::Healthy);
        }
        for lost in [404u16, 410] {
            assert_eq!(status(lost), AnnounceOutcome::LostState { status: lost });
        }
        for refused in [400u16, 401, 403, 409, 429, 500, 503] {
            assert_eq!(
                status(refused),
                AnnounceOutcome::Refused {
                    status: Some(refused),
                    detail: None
                }
            );
        }
        // A 3xx is not an acceptance either.
        assert_eq!(
            status(302),
            AnnounceOutcome::Refused {
                status: Some(302),
                detail: None
            }
        );
        assert_eq!(
            announce_outcome(AnnounceAnswer::NoAnswer),
            AnnounceOutcome::Unreachable { detail: None }
        );
    }

    /// The rule `D-25` enforced at boot, now at every tick: only a lost state re-registers. A
    /// refusal answered with a blind `POST /register` every heartbeat is the useless loop it refused.
    #[test]
    fn only_a_lost_state_re_registers() {
        for lost in [404u16, 410] {
            assert_eq!(status(lost).action(), HeartbeatAction::ReRegister);
        }
        assert_eq!(status(200).action(), HeartbeatAction::Continue);
        for other in [302u16, 401, 403, 409, 500] {
            assert_eq!(status(other).action(), HeartbeatAction::ReportOnly);
        }
        assert_eq!(
            announce_outcome(AnnounceAnswer::NoAnswer).action(),
            HeartbeatAction::ReportOnly
        );
    }

    /// A tracker that is down for fifty heartbeats writes **one** line, and the recovery writes
    /// exactly one more. This is the whole anti-flood policy.
    #[test]
    fn a_tracker_down_for_many_heartbeats_writes_one_line() {
        let mut health = HeartbeatHealth::new();
        assert_eq!(
            health.observe(&AnnounceOutcome::Healthy),
            None,
            "the boot already said the node registered"
        );

        let down = AnnounceOutcome::Unreachable { detail: None };
        let reported = (0..50).filter(|_| health.observe(&down).is_some()).count();
        assert_eq!(reported, 1);
        assert_eq!(health.state(), Some(&ControlPlaneState::Unreachable));

        assert_eq!(
            health.observe(&AnnounceOutcome::Healthy),
            Some(HeartbeatNotice::Recovered {
                previous: ControlPlaneState::Unreachable
            })
        );
        assert_eq!(
            health.observe(&AnnounceOutcome::Healthy),
            None,
            "and the recovery is not repeated either"
        );
    }

    /// The first observation of a failure is reported even though nothing healthy was seen before,
    /// and a **change** of failure is a new line (the operator has to see the new reason).
    #[test]
    fn the_first_failure_and_a_change_of_failure_are_reported() {
        let mut health = HeartbeatHealth::new();
        assert_eq!(
            health.observe(&status(401)),
            Some(HeartbeatNotice::Lost { previous: None })
        );
        assert_eq!(
            health.observe(&status(403)),
            Some(HeartbeatNotice::Lost {
                previous: Some(ControlPlaneState::Refused { status: Some(401) })
            })
        );
        assert_eq!(
            health.observe(&status(403)),
            None,
            "the same failure is not repeated"
        );
    }

    /// A lost state is a state of its own: `404` and then `410` are the same problem, and the
    /// re-registration that follows does not turn into a line per tick.
    #[test]
    fn a_lost_state_is_reported_once() {
        let mut health = HeartbeatHealth::new();
        assert!(health.observe(&status(404)).is_some());
        assert_eq!(health.observe(&status(410)), None);
        assert_eq!(health.state(), Some(&ControlPlaneState::LostState));
    }

    /// The re-registration speaks the vocabulary of `TASK-62`/S68: a tracker that never answered is
    /// unreachable, one that refused (or answered something unusable) is a decision, and both end in
    /// the same state machine.
    #[test]
    fn a_failed_re_registration_maps_to_the_boot_vocabulary() {
        let unreachable = RegistrationError::Unreachable {
            url: "http://tracker:8080".to_string(),
            source: transport_error(),
        };
        let outcome = registration_outcome(&unreachable);
        assert_eq!(outcome.state(), ControlPlaneState::Unreachable);
        assert_eq!(outcome.action(), HeartbeatAction::ReportOnly);

        let refused = RegistrationError::Refused {
            status: 409,
            body: "model_at_capacity".to_string(),
        };
        let outcome = registration_outcome(&refused);
        assert_eq!(
            outcome.state(),
            ControlPlaneState::Refused { status: Some(409) }
        );
        assert_eq!(outcome.action(), HeartbeatAction::ReportOnly);
        assert!(outcome.describe().contains("model_at_capacity"));

        let malformed = RegistrationError::Malformed {
            message: "unusable device".to_string(),
        };
        assert_eq!(
            registration_outcome(&malformed).state(),
            ControlPlaneState::Refused { status: None }
        );

        // The budget of the boot path ran out on reachability failures: still unreachable.
        let gave_up = RegistrationError::GaveUp {
            url: "http://tracker:8080".to_string(),
            budget_secs: 5,
            last: Box::new(malformed),
        };
        assert_eq!(
            registration_outcome(&gave_up).state(),
            ControlPlaneState::Unreachable
        );
    }

    /// The line the operator reads names the state **and** what the tracker said, and never contains a
    /// newline (a body is quoted inside one entry).
    #[test]
    fn the_message_names_the_state_and_what_the_tracker_said() {
        let refused = AnnounceOutcome::Refused {
            status: Some(403),
            detail: Some("worker_not_authorized".to_string()),
        };
        let lost = HeartbeatNotice::Lost { previous: None }.message(&refused);
        assert!(lost.contains("403"), "{lost}");
        assert!(lost.contains("worker_not_authorized"), "{lost}");
        assert!(!lost.contains('\n'), "{lost}");

        let with_preview = AnnounceOutcome::Refused {
            status: Some(500),
            detail: Some(preview("boom\n internal detail \t here")),
        };
        let message = HeartbeatNotice::Lost { previous: None }.message(&with_preview);
        assert!(message.contains("boom internal detail here"), "{message}");
        assert!(!message.contains('\n'), "{message}");

        let recovered = HeartbeatNotice::Recovered {
            previous: ControlPlaneState::Unreachable,
        }
        .message(&AnnounceOutcome::Healthy);
        assert!(recovered.contains("again"), "{recovered}");
        assert!(recovered.contains("unreachable"), "{recovered}");
    }

    /// `preview` is the only thing standing between a tracker body and a log line: it collapses the
    /// whitespace and cuts by **characters**, never in the middle of one.
    #[test]
    fn preview_collapses_whitespace_and_never_breaks_a_character() {
        assert_eq!(preview("  a\n\n\t b  "), "a b");
        assert_eq!(preview(""), "");

        let long = "á".repeat(MAX_DETAIL_CHARS + 50);
        let cut = preview(&long);
        assert_eq!(cut.chars().count(), MAX_DETAIL_CHARS + 1);
        assert!(cut.ends_with('…'));
        assert!(cut.chars().all(|c| c == 'á' || c == '…'));
    }
}
