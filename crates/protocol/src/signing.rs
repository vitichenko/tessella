//! Canonical signed preimage **v1** for inference requests (H-09 · `TASK-05`, decision D-2=(b)).
//!
//! H-09 was that `sign_message()` only signed `bincode(session_id)` (`protocol/src/network.rs`),
//! so `tokens`, `entry_node_url`, `karma_ticket`, … could be rewritten by any hop while the
//! signature still verified, and the same signed request could be replayed forever because there
//! was no nonce. D-2=(b) migrates through an explicit `signature_version` field:
//!
//! | version | preimage | time / replay |
//! |---|---|---|
//! | `1` | [`SignContext::v1_preimage`] | `iat` ± [`SIGNATURE_CLOCK_SKEW_SECS`] + one-shot `nonce` |
//! | `0` / absent | — | **rejected**: this is the retired legacy preimage (`bincode(session_id)`, `TASK-26`/S32) and, at the same time, the marker an unsealed DTO carries |
//! | anything else | — | **rejected** (never «passes by default») |
//!
//! # The v1 preimage is a wire contract — do not reorder it
//!
//! `DOMAIN_TAG | version(1) | len(session_id) ‖ session_id | len(request_id) ‖ request_id |
//! sha256(tokens) | presence+sha256(entry_node_url) | presence+code(karma_tier) | nonce | iat`,
//! with little-endian integers (`u32` lengths, `u64` nonces) and every variable-length string
//! length-prefixed, so `("ab","c")` cannot collide with `("a","bc")`. `tokens` and
//! `entry_node_url` travel as hashes to keep the preimage small. A fixed-vector test
//! (`test_v1_preimage_is_a_fixed_vector`) pins the bytes and
//!
//! The preimage deliberately contains **only fields that survive every hop**: the hidden state
//! (`data`/`shape`) is transformed by each shard, so a hop cannot re-sign it and a preimage that
//! covered it could never be verified downstream. `entry_node_url`, by contrast, is set once by
//! the entry node and must therefore be signed (H-10 already validates *where* the callback may
//! point; H-09 makes sure nobody rewrote the destination on the way).

use crate::error::DomainError;
use crate::models::QueuePriority;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;

/// `signature_version` of the **retired** legacy preimage (`bincode(session_id)`): it covered a
/// single field and carried no nonce, so it could not detect replays. The verifier no longer accepts
/// it — `TASK-26`/S32 closed the migration window D-2 had opened — and no code builds its preimage.
///
/// The constant survives because `0` is still a **wire value** in two senses: `#[serde(default)]`
/// and `Payload::from_tensor` leave the field at `0`, which is precisely the marker of an
/// *unsealed* request. An unsealed request is rejected, never verified.
pub const SIGNATURE_VERSION_LEGACY: u8 = 0;
/// `signature_version` of the canonical preimage: every routing field plus `nonce` and `iat`.
pub const SIGNATURE_VERSION_V1: u8 = 1;
/// Tolerance applied to `iat` in both directions (clocks are not synchronised to the second).
pub const SIGNATURE_CLOCK_SKEW_SECS: u64 = 120;
/// How long a `(session_id, nonce)` pair is remembered to catch replays.
pub const REPLAY_CACHE_TTL_SECS: u64 = 300;

/// Domain separator, so a v1 preimage can never be confused with another protocol's message.
const V1_DOMAIN_TAG: &[u8] = b"tessella-signed-request-v1";

/// Everything the v1 preimage signs, as owned by the DTO that travels on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignContext<'a> {
    /// Session this request belongs to (empty when the DTO has no session id at all).
    pub session_id: &'a str,
    /// Request/step this preimage belongs to (`Payload::request_id`).
    pub request_id: &'a str,
    /// Tokens of the request; empty for a hidden-state hop. Only [`hash_tokens`] is signed.
    pub tokens: &'a [u32],
    /// Token-stream callback of the entry node (`None` and `Some("")` are different contracts).
    pub entry_node_url: Option<&'a str>,
    /// Queue priority the request paid for. Only its [`QueuePriority::signature_code`] is signed.
    pub karma_tier: Option<QueuePriority>,
    /// One-shot value: a repeated `(session_id, nonce)` is a replay.
    pub nonce: u64,
    /// Unix seconds at which the emitter signed this preimage.
    pub iat: u64,
}

impl SignContext<'_> {
    /// Serializes the canonical preimage v1. **Reordering fields breaks every deployed node.**
    #[must_use]
    pub fn v1_preimage(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(V1_DOMAIN_TAG.len() + 160);
        out.extend_from_slice(V1_DOMAIN_TAG);
        out.push(SIGNATURE_VERSION_V1);
        push_len_prefixed(&mut out, self.session_id.as_bytes());
        push_len_prefixed(&mut out, self.request_id.as_bytes());
        out.extend_from_slice(&hash_tokens(self.tokens));
        push_optional_str(&mut out, self.entry_node_url);
        push_optional_tier(&mut out, self.karma_tier);
        out.extend_from_slice(&self.nonce.to_le_bytes());
        out.extend_from_slice(&self.iat.to_le_bytes());
        out
    }
}

fn push_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn push_optional_str(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(v) => {
            out.push(1);
            push_len_prefixed(out, v.as_bytes());
        }
        None => {
            out.push(0);
            push_len_prefixed(out, b"");
        }
    }
}

fn push_optional_tier(out: &mut Vec<u8>, tier: Option<QueuePriority>) {
    match tier {
        Some(t) => {
            out.push(1);
            out.push(t.signature_code());
        }
        None => {
            out.push(0);
            out.push(0);
        }
    }
}

/// `sha256` of the tokens as little-endian `u32`s. Signing the hash instead of the vector keeps
/// the preimage (and the copy performed per hop) small.
#[must_use]
pub fn hash_tokens(tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hasher.finalize().into()
}

/// A request whose integrity and freshness are protected by `signature` over
/// [`SignedRequest::signed_bytes`], authenticated by the key in `client_pubkey`.
///
/// Implemented by the two DTOs that carry inference work: `MoEForwardRequest`
/// (`tessella-protocol`) and `Payload` (`tessella-inference`). The control-plane DTOs of D-3 use
/// [`crate::PeerSignedRequest`] instead: they are one-shot commands, not a stream of payloads
/// transformed hop by hop, so a nonce window buys nothing there.
pub trait SignedRequest {
    /// `signature_version` declared by the request (`0` = legacy, `1` = canonical).
    fn signature_version(&self) -> u8;

    /// Unique value of this request, checked against [`ReplayCache`] for version `1`.
    fn nonce(&self) -> u64;

    /// Unix seconds at which the emitter signed the request (version `1` only).
    fn iat(&self) -> u64;

    /// Raw signature bytes (empty when unsigned).
    fn signature_bytes(&self) -> &[u8];

    /// Protobuf-encoded libp2p public key of the signer (empty when unsigned).
    fn client_pubkey(&self) -> &[u8];

    /// The signed fields of this request, as they travel on the wire.
    fn sign_context(&self) -> SignContext<'_>;

    /// Bytes covered by the signature. Only `signature_version = 1` has a preimage: the legacy one
    /// was retired (`TASK-26`/S32), so there is nothing to compute for the other versions.
    ///
    /// # Errors
    /// [`DomainError::Validation`] for any version other than `1`, including the retired legacy `0`
    /// (what an unsealed DTO carries). Failing here is what makes a fallback to «verify it like the
    /// old one» impossible for a caller.
    fn signed_bytes(&self) -> Result<Vec<u8>, DomainError> {
        match self.signature_version() {
            SIGNATURE_VERSION_V1 => Ok(self.sign_context().v1_preimage()),
            other => Err(DomainError::Validation(format!(
                "unsupported signature_version {other}: only 1 (canonical) is accepted; the legacy \
                 preimage was retired by TASK-26 (H-09)"
            ))),
        }
    }

    /// Writes the five signature fields at once, so a seal can never leave a half-updated
    /// `(version, nonce, signature)` triple behind.
    fn set_signature_fields(
        &mut self,
        version: u8,
        nonce: u64,
        iat: u64,
        signature: Vec<u8>,
        client_pubkey: Vec<u8>,
    );
}

/// Remembers the `(session_id, nonce)` pairs seen recently, so the same signed request cannot be
/// replayed inside [`REPLAY_CACHE_TTL_SECS`].
///
/// Scope: v1 requests, which since `TASK-26`/S32 is **every** accepted request — the legacy path had
/// no nonce, so this cache stops being optional for any live traffic.
///
/// The map stores the **local** arrival time (not the emitter's `iat`), so a peer with a wrong
/// clock cannot extend or shorten the window. Insertion prunes opportunistically once the map is
/// big enough and [`ReplayCache::prune`] is also driven by the worker's periodic cleanup task, so
/// memory stays bounded even with no traffic.
#[derive(Debug, Default)]
pub struct ReplayCache {
    seen: Mutex<HashMap<(String, u64), u64>>,
}

impl ReplayCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `(session_id, nonce)` as seen at `now` (local Unix seconds).
    ///
    /// Returns `false` when the pair was already recorded and is still inside the TTL — that is,
    /// when the request is a replay. An entry older than the TTL no longer counts as a replay and
    /// is refreshed in place, so the cache cannot be exhausted by guessing nonces either.
    pub fn check_and_record(&self, session_id: &str, nonce: u64, now: u64) -> bool {
        let mut seen = self.lock();
        if seen.len() > 256 {
            prune_locked(&mut seen, now);
        }
        let key = (session_id.to_string(), nonce);
        let is_fresh = match seen.get(&key) {
            Some(recorded_at) => now.saturating_sub(*recorded_at) > REPLAY_CACHE_TTL_SECS,
            None => true,
        };
        if is_fresh {
            // A rejected replay must not extend the window: the TTL is measured from the first
            // sighting of the pair, not from the last attempt.
            seen.insert(key, now);
        }
        is_fresh
    }

    /// Drops the entries older than [`REPLAY_CACHE_TTL_SECS`] and returns how many were removed.
    pub fn prune(&self, now: u64) -> usize {
        let mut seen = self.lock();
        prune_locked(&mut seen, now)
    }

    /// Number of remembered pairs (tests and the cleanup task's log line).
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(String, u64), u64>> {
        // The map has no invariant to break, so a poisoned lock is not worth a panic.
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn prune_locked(seen: &mut HashMap<(String, u64), u64>, now: u64) -> usize {
    let before = seen.len();
    seen.retain(|_, recorded_at| now.saturating_sub(*recorded_at) <= REPLAY_CACHE_TTL_SECS);
    before - seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::QueuePriority;

    const NOW: u64 = 1_700_000_000;

    fn context() -> SignContext<'static> {
        SignContext {
            session_id: "session-12345",
            request_id: "req-1",
            tokens: &[1, 2, 3],
            entry_node_url: Some("http://203.0.113.9:9001"),
            karma_tier: Some(QueuePriority::Vip),
            nonce: 0x0102_0304_0506_0708,
            iat: NOW,
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The preimage is a wire contract: a silent reorder or format change must break this test.
    #[test]
    fn test_v1_preimage_is_a_fixed_vector() {
        let bytes = context().v1_preimage();

        // Self-documenting layout checks (domain tag, version, length prefix) so the pinned hex
        // below can be read instead of trusted.
        assert_eq!(&bytes[..26], b"tessella-signed-request-v1");
        assert_eq!(bytes[26], SIGNATURE_VERSION_V1);
        assert_eq!(&bytes[27..31], &13u32.to_le_bytes(), "len(session_id)");
        assert_eq!(&bytes[31..44], b"session-12345");
        assert_eq!(&bytes[44..48], &5u32.to_le_bytes(), "len(request_id)");
        assert_eq!(&bytes[48..53], b"req-1");

        assert_eq!(
            hex(&bytes),
            "74657373656c6c612d7369676e65642d726571756573742d7631010d00000073657373696f6e2d3132333435050000007265712d314636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d0117000000687474703a2f2f3230332e302e3131332e393a393030310102080706050403020100f1536500000000"
        );
    }

    /// Every field of the context must change the preimage, or a signature could be reused
    /// for a mutated request (criterion 1 of `TASK-05`).
    #[test]
    fn test_v1_preimage_covers_every_signed_field() {
        let base = context().v1_preimage();
        let variants = [
            SignContext {
                session_id: "session-54321",
                ..context()
            },
            SignContext {
                request_id: "req-2",
                ..context()
            },
            SignContext {
                tokens: &[1, 2, 4],
                ..context()
            },
            SignContext {
                entry_node_url: Some("http://198.51.100.7:9002"),
                ..context()
            },
            SignContext {
                karma_tier: Some(QueuePriority::Leech),
                ..context()
            },
            SignContext {
                nonce: 9,
                ..context()
            },
            SignContext {
                iat: NOW + 1,
                ..context()
            },
        ];
        for variant in variants {
            assert_ne!(
                base,
                variant.v1_preimage(),
                "the preimage must change with every signed field: {variant:?}"
            );
        }
    }

    /// Length-prefixed concatenation: `("ab", "c")` must not collide with `("a", "bc")`.
    #[test]
    fn test_v1_preimage_is_unambiguous_across_field_boundaries() {
        let a = SignContext {
            session_id: "ab",
            request_id: "c",
            ..context()
        };
        let b = SignContext {
            session_id: "a",
            request_id: "bc",
            ..context()
        };
        assert_ne!(a.v1_preimage(), b.v1_preimage());
    }

    /// `None` and `Some("")` are different contracts, so the preimage carries a presence flag.
    #[test]
    fn test_v1_preimage_distinguishes_none_from_an_empty_entry_node_url() {
        let absent = SignContext {
            entry_node_url: None,
            ..context()
        };
        let empty = SignContext {
            entry_node_url: Some(""),
            ..context()
        };
        assert_ne!(absent.v1_preimage(), empty.v1_preimage());
        assert_ne!(absent.v1_preimage(), context().v1_preimage());
    }

    /// Only `signature_version = 1` has a preimage: the legacy one was retired (`TASK-26`/S32), so
    /// there is nothing to compute for `0` — which is exactly the marker an **unsealed** DTO carries
    /// (`#[serde(default)]`, `Payload::from_tensor`). Any other version is unknown.
    #[test]
    fn test_only_v1_has_a_preimage_and_the_retired_v0_does_not() {
        let mut req = crate::network::MoEForwardRequest {
            input_tensor: None,
            client_peer_id: None,
            session_id: Some("session-12345".to_string()),
            tokens: vec![1, 2, 3],
            speculative_count: 0,
            entry_node_url: None,
            signature: Vec::new(),
            client_pubkey: Vec::new(),
            karma_ticket: None,
            signature_version: SIGNATURE_VERSION_LEGACY,
            nonce: 0,
            iat: 0,
        };
        let err = req
            .signed_bytes()
            .expect_err("the retired version must not be signed as legacy any more");
        assert!(err.to_string().contains("version 0"), "{err}");

        req.signature_version = 7;
        let err = req
            .signed_bytes()
            .expect_err("an unknown version must not fall back to any path");
        assert!(err.to_string().contains('7'), "{err}");

        req.signature_version = SIGNATURE_VERSION_V1;
        assert!(
            req.signed_bytes().is_ok(),
            "v1 keeps the canonical preimage: it is the only signed version left"
        );
    }

    /// A reused `(session_id, nonce)` must be reported, and the entry must expire.
    #[test]
    fn test_replay_cache_detects_a_repeated_nonce_and_forgets_it_after_the_ttl() {
        let cache = ReplayCache::new();
        assert!(cache.check_and_record("s-1", 42, NOW), "first sight");
        assert!(
            !cache.check_and_record("s-1", 42, NOW + 1),
            "the same nonce of the same session is a replay"
        );
        assert!(
            cache.check_and_record("s-1", 43, NOW + 1),
            "a new nonce is fine"
        );
        assert!(
            cache.check_and_record("s-2", 42, NOW + 1),
            "another session is independent"
        );
        assert_eq!(cache.len(), 3);

        let later = NOW + REPLAY_CACHE_TTL_SECS + 1;
        assert!(
            cache.check_and_record("s-1", 42, later),
            "an expired entry is forgotten, so the pair may be reused"
        );
    }

    #[test]
    fn test_replay_cache_prune_drops_only_expired_entries() {
        let cache = ReplayCache::new();
        assert!(cache.check_and_record("s-1", 1, NOW));
        assert!(cache.check_and_record("s-1", 2, NOW + REPLAY_CACHE_TTL_SECS));
        assert_eq!(cache.prune(NOW + REPLAY_CACHE_TTL_SECS + 1), 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.prune(NOW), 0, "a fresh entry survives a prune");
    }

    #[test]
    fn test_signature_version_constants_are_the_wire_contract() {
        assert_eq!(SIGNATURE_VERSION_LEGACY, 0);
        assert_eq!(SIGNATURE_VERSION_V1, 1);
        assert_eq!(SIGNATURE_CLOCK_SKEW_SECS, 120);
    }

    #[test]
    fn test_queue_priority_signature_codes_are_stable() {
        assert_eq!(QueuePriority::Leech.signature_code(), 0);
        assert_eq!(QueuePriority::Standard.signature_code(), 1);
        assert_eq!(QueuePriority::Vip.signature_code(), 2);
    }
}
