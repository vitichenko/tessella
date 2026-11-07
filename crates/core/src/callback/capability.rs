//! Capability auth for the token-stream callback (H-10 · `TASK-06` **mitigation 2**).
//!
//! The audit's H-10 had two halves. The first one (mitigation 1, S10) was **where** the final shard
//! is allowed to POST the generated tokens ([`super::validate_callback_url`]); the second one is that
//! `/v1/token_stream/{request_id}` had **no authentication at all** (`tessella/crates/api/src/lib.rs`,
//! `handle_token_stream`), so anyone who could guess a `request_id` — and the streaming path uses the
//! client-supplied `session_id` when there is one — could inject tokens into somebody else's SSE
//! stream.
//!
//! # Why the payload's signature *is* the capability
//!
//! The entry node seals **every** payload it emits with its own key (H-09 · `TASK-05`), and every
//! hop copies the signature fields verbatim, so:
//!
//! - the shard that holds a payload can compute a value nobody else can (ed25519 signatures are
//!   unforgeable, and the payload never leaves the mesh);
//! - the same value proves *which* payload the caller saw: the v1 preimage covers `session_id`,
//!   `request_id`, the tokens, `entry_node_url`, the karma tier, a one-shot `nonce` and `iat`, so a
//!   capability is bound to one request **and** to one decoding step;
//! - a hop that mutated it would break the signature verification of the next hop, which is the
//!   same property S11 pays for (that is why `TASK-05` was the natural pair of this task);
//! - it needs **no new cryptography, no new field and no extra round trip**: the alternative the
//!   card offered (a random 32-byte token published in the signed payload, or fetched from an
//!   authenticated endpoint) would have been a breaking change of `tessella-protocol`'s DTOs and of
//!   the v1 preimage, which is never touched outside a `BREAKING` step — S12 is not one.
//!
//! The entry node registers the capability of each payload it seals
//! ([`CallbackCapabilities::register`]) and the handler only accepts a callback that (a) carries
//! `Authorization: Bearer <capability>`, (b) matches a registered capability **of that
//! `request_id`**, and (c) has not been used before: a capability is **single-use**, because one
//! sealed payload produces at most one callback.
//!
//! # Why it lives in `tessella-core` (SOLID-89)
//!
//! The registry is **process state**, not a wire DTO, and it is shared by the engine that seals the
//! payloads and by the router that receives the callbacks — two halves of one process that must not
//! have to link the HTTP crate to agree. The HTTP half (reading `Authorization: Bearer …` out of the
//! headers, `authorize_callback`) stays in `tessella-api` as a thin adapter over this module.
//!
//! # Limits (documented, not silent)
//!
//! - A hop that legitimately processed a payload *can* replay its capability and inject content
//!   into that stream. It could already alter what it computes, so this is not a new trust
//!   boundary; the blast radius is bounded by the single-use rule and by the per-stream rate limit.
//! - A very long generation (more than [`MAX_PENDING_CAPABILITIES`] steps) or a callback delayed by
//!   more than [`CALLBACK_CAPABILITY_TTL`] loses the **fast path** (the tokens still reach the
//!   client through the synchronous call stack, which is the same degradation S10 chose).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use subtle::ConstantTimeEq;

/// How long a capability stays valid while the stream it belongs to is alive.
pub const CALLBACK_CAPABILITY_TTL: Duration = Duration::from_secs(300);

/// Most capabilities remembered for one stream. A capability is consumed on use, so the bound only
/// matters for a generation longer than this many decoding steps (the oldest are dropped, and a
/// callback that late only loses the fast path).
pub const MAX_PENDING_CAPABILITIES: usize = 256;

/// Accepted callbacks per stream and per [`CALLBACK_RATE_LIMIT_WINDOW`]. The limit exists to bound
/// a flood from a hop that holds several valid capabilities, **not** to shape normal traffic: one
/// request produces one callback per decoding step, and no single request decodes at 200 tok/s.
pub const CALLBACK_RATE_LIMIT_PER_SECOND: u32 = 200;

/// Length of the sliding window of the rate limit.
pub const CALLBACK_RATE_LIMIT_WINDOW: Duration = Duration::from_secs(1);

/// The capability a `/v1/token_stream/{request_id}` callback must present: the hex encoding of the
/// signature the entry node put on the payload, as it travels through the hops.
#[must_use]
pub fn callback_capability(signature: &[u8]) -> String {
    hex::encode(signature)
}

/// Outcome of the capability check of a callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackAuthz {
    /// The capability was issued for this stream, had not been used and is inside the rate limit.
    Accepted,
    /// Missing header, unknown capability, another stream's capability or a replay.
    Unauthorized,
    /// Too many accepted callbacks for this stream inside the current window.
    RateLimited,
}

/// Registry of the capabilities issued by **this** node, keyed by the `request_id` of the stream.
///
/// Only the entry node of a pipeline owns streams, so only it registers capabilities; every node
/// holds the same type (empty when it never seals a payload) because the engine that seals payloads
/// and the HTTP handler that receives callbacks are two halves of one process.
#[derive(Debug, Default)]
pub struct CallbackCapabilities {
    streams: DashMap<String, StreamState>,
}

impl CallbackCapabilities {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the capability of a payload the entry node just sealed for `request_id`.
    ///
    /// An empty signature (an unsealed or legacy-unsigned payload) issues nothing: the handler can
    /// then never be tricked into accepting a callback of unknown provenance.
    pub fn register(&self, request_id: &str, signature: &[u8], now: Instant) {
        if signature.is_empty() {
            return;
        }
        let capability = callback_capability(signature);
        if let Some(mut state) = self.streams.get_mut(request_id) {
            state.record(capability, now);
            return;
        }
        // A brand new request is the only moment at which the entries of *other* requests can be
        // swept, so a session without a stream (the non-streaming paths also seal payloads) cannot
        // leave one entry per request behind forever.
        self.sweep(now);
        self.streams
            .insert(request_id.to_string(), StreamState::new(capability, now));
    }

    /// Checks and **consumes** the capability of one callback (single use).
    pub fn authorize(&self, request_id: &str, capability: &str, now: Instant) -> CallbackAuthz {
        match self.streams.get_mut(request_id) {
            Some(mut state) => state.authorize(capability, now),
            None => CallbackAuthz::Unauthorized,
        }
    }

    /// Forgets a stream (called when the entry node stops accepting callbacks for it).
    pub fn remove(&self, request_id: &str) {
        self.streams.remove(request_id);
    }

    /// Streams currently remembering at least one capability (tests and observability).
    #[must_use]
    pub fn tracked_streams(&self) -> usize {
        self.streams.len()
    }

    /// Drops the streams that have been idle for longer than [`CALLBACK_CAPABILITY_TTL`].
    fn sweep(&self, now: Instant) {
        self.streams
            .retain(|_, state| now.duration_since(state.last_touch) <= CALLBACK_CAPABILITY_TTL);
    }
}

/// Capabilities of one stream, in issuance order.
#[derive(Debug)]
struct StreamState {
    pending: VecDeque<PendingCapability>,
    /// Start of the current rate-limit window.
    window_start: Instant,
    /// Accepted callbacks inside the current window.
    window_count: u32,
    /// Last registration or accepted callback, used to sweep idle streams.
    last_touch: Instant,
}

#[derive(Debug)]
struct PendingCapability {
    value: String,
    issued_at: Instant,
}

impl StreamState {
    fn new(capability: String, now: Instant) -> Self {
        let mut pending = VecDeque::with_capacity(1);
        pending.push_back(PendingCapability {
            value: capability,
            issued_at: now,
        });
        Self {
            pending,
            window_start: now,
            window_count: 0,
            last_touch: now,
        }
    }

    fn record(&mut self, capability: String, now: Instant) {
        self.drop_expired(now);
        self.pending.push_back(PendingCapability {
            value: capability,
            issued_at: now,
        });
        while self.pending.len() > MAX_PENDING_CAPABILITIES {
            self.pending.pop_front();
        }
        self.last_touch = now;
    }

    fn authorize(&mut self, capability: &str, now: Instant) -> CallbackAuthz {
        self.drop_expired(now);
        let Some(index) = self.position(capability) else {
            // A guess (or a foreign capability) never reaches the window: only callbacks the entry
            // node actually authorised may spend the rate-limit budget, or a third party could
            // starve a legitimate shard by flooding wrong tokens.
            return CallbackAuthz::Unauthorized;
        };
        if now.duration_since(self.window_start) >= CALLBACK_RATE_LIMIT_WINDOW {
            self.window_start = now;
            self.window_count = 0;
        }
        if self.window_count >= CALLBACK_RATE_LIMIT_PER_SECOND {
            // Not consumed: the capability stays pending, so the next decoding step (or a retry in
            // the next window) is not punished for this burst.
            return CallbackAuthz::RateLimited;
        }
        self.pending.remove(index);
        self.window_count += 1;
        self.last_touch = now;
        CallbackAuthz::Accepted
    }

    /// Capabilities are only ever appended, so the expired ones are a prefix of the queue.
    fn drop_expired(&mut self, now: Instant) {
        while let Some(front) = self.pending.front() {
            if now.duration_since(front.issued_at) > CALLBACK_CAPABILITY_TTL {
                self.pending.pop_front();
            } else {
                break;
            }
        }
    }

    /// Position of `capability`, comparing in constant time so the queue cannot be used as a timing
    /// oracle (the same primitive the Musivum `admin_token` uses for its bearer comparison).
    fn position(&self, capability: &str) -> Option<usize> {
        let mut found = None;
        for (index, pending) in self.pending.iter().enumerate() {
            let matches = bool::from(pending.value.as_bytes().ct_eq(capability.as_bytes()));
            if matches && found.is_none() {
                found = Some(index);
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature(byte: u8) -> Vec<u8> {
        vec![byte; 64]
    }

    /// Distinct 64-byte signatures for tests that need more than 256 of them.
    fn signature_at(step: usize) -> Vec<u8> {
        let mut signature = vec![0u8; 64];
        signature[0] = (step % 256) as u8;
        signature[1] = (step / 256) as u8;
        signature
    }

    #[test]
    fn a_capability_is_the_hex_of_the_signature() {
        assert_eq!(callback_capability(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
        assert_eq!(callback_capability(&signature(7)).len(), 128);
    }

    #[test]
    fn only_the_capability_issued_for_that_stream_is_accepted() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        capabilities.register("req-1", &signature(1), now);
        capabilities.register("req-2", &signature(2), now);

        assert_eq!(
            capabilities.authorize("req-1", &callback_capability(&signature(1)), now),
            CallbackAuthz::Accepted
        );
        // Another stream's capability (even though this node issued it) ...
        assert_eq!(
            capabilities.authorize("req-2", &callback_capability(&signature(1)), now),
            CallbackAuthz::Unauthorized
        );
        // ... a capability nobody issued ...
        assert_eq!(
            capabilities.authorize("req-2", &callback_capability(&signature(9)), now),
            CallbackAuthz::Unauthorized
        );
        // ... and a stream this node never heard of.
        assert_eq!(
            capabilities.authorize("chatcmpl-unknown", &callback_capability(&signature(2)), now),
            CallbackAuthz::Unauthorized
        );
    }

    #[test]
    fn a_capability_is_single_use() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        capabilities.register("req", &signature(3), now);
        let capability = callback_capability(&signature(3));

        assert_eq!(
            capabilities.authorize("req", &capability, now),
            CallbackAuthz::Accepted
        );
        assert_eq!(
            capabilities.authorize("req", &capability, now),
            CallbackAuthz::Unauthorized
        );
    }

    #[test]
    fn an_expired_capability_is_refused() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        capabilities.register("req", &signature(4), now);

        let capability = callback_capability(&signature(4));
        assert_eq!(
            capabilities.authorize("req", &capability, now + CALLBACK_CAPABILITY_TTL),
            CallbackAuthz::Accepted
        );

        capabilities.register("req", &signature(5), now);
        let stale = callback_capability(&signature(4));
        assert_eq!(
            capabilities.authorize(
                "req",
                &stale,
                now + CALLBACK_CAPABILITY_TTL + Duration::from_secs(1)
            ),
            CallbackAuthz::Unauthorized
        );
    }

    #[test]
    fn the_pending_queue_is_bounded() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        for step in 0..=MAX_PENDING_CAPABILITIES {
            capabilities.register("req", &signature_at(step), now);
        }

        // The oldest was dropped to keep the queue bounded ...
        assert_eq!(
            capabilities.authorize("req", &callback_capability(&signature_at(0)), now),
            CallbackAuthz::Unauthorized
        );
        // ... and the stream is still usable for the newest payload.
        assert_eq!(
            capabilities.authorize(
                "req",
                &callback_capability(&signature_at(MAX_PENDING_CAPABILITIES)),
                now
            ),
            CallbackAuthz::Accepted
        );
    }

    #[test]
    fn a_removed_stream_forgets_its_capabilities() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        capabilities.register("req", &signature(6), now);
        assert_eq!(capabilities.tracked_streams(), 1);

        capabilities.remove("req");
        assert_eq!(capabilities.tracked_streams(), 0);
        assert_eq!(
            capabilities.authorize("req", &callback_capability(&signature(6)), now),
            CallbackAuthz::Unauthorized
        );
    }

    #[test]
    fn idle_streams_are_swept_when_a_new_one_arrives() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        capabilities.register("req-old", &signature(7), now);
        capabilities.register(
            "req-new",
            &signature(8),
            now + CALLBACK_CAPABILITY_TTL + Duration::from_secs(1),
        );

        assert_eq!(capabilities.tracked_streams(), 1);
    }

    #[test]
    fn accepted_callbacks_are_rate_limited_per_stream_and_the_window_recovers() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        let budget = CALLBACK_RATE_LIMIT_PER_SECOND as usize;
        for step in 0..=budget {
            capabilities.register("req", &signature(step as u8), now);
        }

        let mut accepted = 0;
        let mut limited = 0;
        for step in 0..=budget {
            match capabilities.authorize("req", &callback_capability(&signature(step as u8)), now) {
                CallbackAuthz::Accepted => accepted += 1,
                CallbackAuthz::RateLimited => limited += 1,
                CallbackAuthz::Unauthorized => panic!("a registered capability was refused"),
            }
        }
        assert_eq!(accepted, budget);
        assert_eq!(limited, 1);

        // The throttled capability was not consumed, so the next window still accepts it.
        assert_eq!(
            capabilities.authorize(
                "req",
                &callback_capability(&signature(budget as u8)),
                now + CALLBACK_RATE_LIMIT_WINDOW
            ),
            CallbackAuthz::Accepted
        );
    }

    #[test]
    fn a_flood_of_wrong_capabilities_does_not_starve_the_stream() {
        let capabilities = CallbackCapabilities::new();
        let now = Instant::now();
        capabilities.register("req", &signature(11), now);

        for _ in 0..(CALLBACK_RATE_LIMIT_PER_SECOND * 4) {
            assert_eq!(
                capabilities.authorize("req", &callback_capability(&signature(12)), now),
                CallbackAuthz::Unauthorized
            );
        }
        // The real shard is unaffected: the guesses never spent the rate-limit budget.
        assert_eq!(
            capabilities.authorize("req", &callback_capability(&signature(11)), now),
            CallbackAuthz::Accepted
        );
    }

    #[test]
    fn an_empty_signature_issues_no_capability() {
        let capabilities = CallbackCapabilities::new();
        capabilities.register("req", &[], Instant::now());

        assert_eq!(capabilities.tracked_streams(), 0);
        assert_eq!(
            capabilities.authorize("req", "", Instant::now()),
            CallbackAuthz::Unauthorized
        );
    }
}
