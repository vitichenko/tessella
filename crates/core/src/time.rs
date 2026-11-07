//! Unix-seconds clock shared by both workspaces (H-16 · `TASK-14`).
//!
//! Six copies of `SystemTime::now().duration_since(UNIX_EPOCH)` lived across the two workspaces,
//! four of them with `.unwrap()` **inside request handlers**. A `SystemTime` before the epoch (a
//! hand-set clock, a broken VM, a hostile NTP response) panicked the handler task, which turns a
//! clock problem into a request-level failure with no diagnostic. The single implementation
//! below logs the anomaly and returns `0`, which downstream code already treats as "not fresh"
//! (the `iat` window of H-09 rejects it) instead of crashing.

use std::time::{SystemTime, UNIX_EPOCH};

/// Current Unix time in seconds, or `0` if the system clock is before 1970.
///
/// The `Err` branch is unreachable in practice; it exists so that no caller has to choose between
/// a panic and a silent `unwrap_or_default()`. When it happens it is logged, because a wrong clock
/// breaks signature verification and session expiry in ways that are otherwise invisible.
pub fn unix_now_secs() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => elapsed.as_secs(),
        Err(err) => {
            // `error!` and not `warn!`: a backwards clock invalidates every signature window and
            // expires every ticket, so it must be as loud as the copies it replaced (the
            // `musivum-economy` one logged exactly this incident as an error).
            tracing::error!(
                "🔥 CRITICAL: the system clock is before the UNIX epoch ({err}); using 0 as the \
                 timestamp. Signature verification and ticket expiry will look stale until the \
                 clock is fixed."
            );
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_now_secs_matches_the_system_clock() {
        let reference = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the test machine has a sane clock")
            .as_secs();
        let now = unix_now_secs();
        // Same second, or at most one apart if the clock ticked between the two calls.
        assert!(
            now == reference || now + 1 == reference,
            "unix_now_secs() = {now} is not the system clock ({reference})"
        );
        assert!(now > 1_600_000_000, "expected a real timestamp, got {now}");
    }
}
