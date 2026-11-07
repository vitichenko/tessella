use crate::models::{KarmaTicket, WorkTicket};
use crate::signing::{SignContext, SignedRequest};
use serde::{Deserialize, Serialize};

/// Serializable DType (candle::DType does not implement Serialize).
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum DTypeTag {
    F32,
    F16,
}

/// Serializable hidden state for P2P transmission.
///
/// `TASK-27`/S33 removed the `trace_route` field it used to carry: it was never in the signed
/// preimage and the two places that did fill it pushed a **URL** (while the doc claimed
/// `peer_id`s), so it was an unverifiable hint — and a hint that looked authoritative. The
/// authoritative route is what this hop can prove: the `layer_range` of its `WorkProof` plus the
/// tickets the orchestrator issued for the session.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Payload {
    pub request_id: String,
    pub session_id: Option<String>,
    pub seq_start_pos: u32,
    pub layer_idx: usize,  // last processed layer
    pub shape: Vec<usize>, // [batch, seq_len, hidden_dim]
    pub dtype: DTypeTag,
    pub data: Vec<u8>, // raw tensor bytes
    pub checksum: u32, // CRC32 of raw data
    /// HTTP base URL of the entry node holding the active SSE connection.
    /// Set once by the entry node; all hops pass it through unchanged.
    /// The final shard worker uses this to POST generated tokens directly
    /// to the entry node's `/v1/token_stream/{request_id}` endpoint.
    #[serde(default)]
    pub entry_node_url: Option<String>,
    /// Ed25519 signature of the emitter over the canonical v1 preimage (H-09): it covers
    /// `session_id`, `request_id`, the token digest, `entry_node_url`, the tier, `nonce` and `iat`,
    /// and it is what makes a payload of one hop accepted by the next one. It doubles as the
    /// capability of `/v1/token_stream/{request_id}` (H-10, mitigation 2).
    #[serde(default)]
    pub signature: Vec<u8>,
    #[serde(default)]
    pub client_pubkey: Vec<u8>,
    pub karma_ticket: Option<KarmaTicket>,
    /// One-shot authorisation minted by the orchestrator for this session (`TASK-07`, H-12).
    ///
    /// Set by the entry node from the `/api/v1/session_grant` response and copied verbatim by
    /// every hop, exactly like `karma_ticket`. Each hop attaches it to the `WorkProof` it stores
    /// in its WAL, so the orchestrator only credits work it authorised itself.
    #[serde(default)]
    pub work_ticket: Option<WorkTicket>,
    /// Version of the signed preimage (H-09 · `TASK-05`, D-2=(b)): `1` = canonical preimage v1. `0`
    /// is **not** a way in: it is the retired legacy preimage (`bincode(session_id)`, `TASK-26`/S32)
    /// and, at the same time, the value an **unsealed** payload carries, so a hop rejects it instead
    /// of verifying it. Anything else is unknown and rejected too.
    ///
    /// **Wire compatibility**: `Payload` travels as `bincode` (`to_bytes`), which is positional
    /// and has no notion of `#[serde(default)]`, so these three fields are a **breaking** change
    /// of the payload format — every node of the mesh must be updated together (announced in the
    /// changelog). `Input`/`Output` of the API keep accepting the old JSON payloads.
    #[serde(default)]
    pub signature_version: u8,
    /// One-shot value of the v1 preimage, renewed for **every** emitted payload (a decoding step
    /// is a new request, so reusing the session's nonce would look like a replay to the next hop).
    #[serde(default)]
    pub nonce: u64,
    /// Unix seconds at which this payload was sealed.
    #[serde(default)]
    pub iat: u64,
    /// When auditing the front (the entry node's first step), carries the prompt tokens
    /// so a replica can re-execute the front (`TASK-80`/S87, `D-34 = (C)`).
    /// Covered by the canonical v1 preimage (`SignContext::tokens`).
    #[serde(default)]
    pub tokens: Vec<u32>,
}

impl Payload {
    /// The **continuity key** of the session this payload belongs to (`TASK-75`/`S82`, `F-7`).
    ///
    /// It is `session_id` when the payload carries one — the client's continuity id, written once by
    /// the entry node and propagated verbatim by every hop — and falls back to `request_id` (the work
    /// id the orchestrator minted) when it does not. The fallback is not an invention: it is **exactly**
    /// the one the signed preimage applies ([`crate::signing::SignContext`]), so the key a node uses
    /// to find its K/V cache is the key that was signed and every hop of a pipeline agrees on it.
    #[must_use]
    pub fn session_key(&self) -> &str {
        self.session_id.as_deref().unwrap_or(&self.request_id)
    }

    /// Reads the continuity key of the conversation from serialized payload bytes without
    /// deserializing the tensor data (`TASK-81` · S88).
    ///
    /// The router calls this to look up or update the session's pinned route without paying the
    /// memory and CPU cost of deserializing multi-megabyte activation tensors.
    #[must_use]
    pub fn peek_session_key(bytes: &[u8]) -> Option<String> {
        #[derive(Deserialize)]
        struct SessionHeader {
            request_id: String,
            session_id: Option<String>,
        }
        let header: SessionHeader = bincode::deserialize_from(&mut &bytes[..]).ok()?;
        Some(header.session_id.unwrap_or(header.request_id))
    }

    /// How many tokens of sequence this hidden state carries (`TASK-76`/S83).
    ///
    /// The wire shape is `[batch, seq_len, hidden_dim]`, so the sequence is the **second**
    /// dimension; a MoE pipeline always transports `batch = 1` and the first hop carries the hidden
    /// state of the whole prompt.
    pub fn n_tokens(&self) -> Option<usize> {
        match self.shape.as_slice() {
            [_batch, seq_len, _hidden] => Some(*seq_len),
            _ => None,
        }
    }

    /// Verifies the data integrity using CRC32 checksum.
    pub fn verify_checksum(&self) -> bool {
        crc32fast::hash(&self.data) == self.checksum
    }

    /// Serializes to bytes for network transmission.
    ///
    /// # Errors
    ///
    /// Returns `anyhow::Error` if serialization fails.
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserializes from bytes received via network.
    ///
    /// # Errors
    ///
    /// Returns `anyhow::Error` if deserialization fails.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

impl SignedRequest for Payload {
    fn signature_version(&self) -> u8 {
        self.signature_version
    }

    fn nonce(&self) -> u64 {
        self.nonce
    }

    fn iat(&self) -> u64 {
        self.iat
    }

    fn signature_bytes(&self) -> &[u8] {
        &self.signature
    }

    fn client_pubkey(&self) -> &[u8] {
        &self.client_pubkey
    }

    fn sign_context(&self) -> SignContext<'_> {
        SignContext {
            session_id: self.session_id.as_deref().unwrap_or(&self.request_id),
            request_id: &self.request_id,
            tokens: &self.tokens,
            entry_node_url: self.entry_node_url.as_deref(),
            karma_tier: self.karma_ticket.as_ref().map(|ticket| ticket.tier),
            nonce: self.nonce,
            iat: self.iat,
        }
    }

    fn set_signature_fields(
        &mut self,
        version: u8,
        nonce: u64,
        iat: u64,
        signature: Vec<u8>,
        client_pubkey: Vec<u8>,
    ) {
        self.signature_version = version;
        self.nonce = nonce;
        self.iat = iat;
        self.signature = signature;
        self.client_pubkey = client_pubkey;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signing::SIGNATURE_VERSION_LEGACY;

    fn sample_payload() -> Payload {
        Payload {
            request_id: "req-1".to_string(),
            session_id: Some("session-12345".to_string()),
            seq_start_pos: 0,
            layer_idx: 3,
            shape: vec![1, 1, 4],
            dtype: DTypeTag::F32,
            data: vec![0u8; 16],
            checksum: crc32fast::hash(&[0u8; 16]),
            entry_node_url: Some("http://203.0.113.9:9001".to_string()),
            signature: Vec::new(),
            client_pubkey: Vec::new(),
            karma_ticket: None,
            work_ticket: None,
            signature_version: SIGNATURE_VERSION_LEGACY,
            nonce: 0,
            iat: 0,
            tokens: Vec::new(),
        }
    }

    #[test]
    fn test_payload_roundtrip_bytes() {
        let p = sample_payload();
        assert!(p.verify_checksum());
        let bytes = p.to_bytes().expect("serialize");
        let decoded = Payload::from_bytes(&bytes).expect("deserialize");
        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.session_id.as_deref(), Some("session-12345"));
        assert_eq!(decoded.layer_idx, 3);
        assert_eq!(decoded.dtype, DTypeTag::F32);
        assert!(decoded.verify_checksum());
    }

    #[test]
    fn test_session_key_and_peek() {
        let p = sample_payload();
        assert_eq!(p.session_key(), "session-12345");
        let bytes = p.to_bytes().unwrap();
        assert_eq!(
            Payload::peek_session_key(&bytes),
            Some("session-12345".to_string())
        );

        let mut fallback = sample_payload();
        fallback.session_id = None;
        assert_eq!(fallback.session_key(), "req-1");
        let bytes_fallback = fallback.to_bytes().unwrap();
        assert_eq!(
            Payload::peek_session_key(&bytes_fallback),
            Some("req-1".to_string())
        );
    }

    #[test]
    fn test_token_count_from_shape() {
        let mut p = sample_payload();
        p.shape = vec![1, 2048, 1536];
        assert_eq!(p.n_tokens(), Some(2048));

        p.shape = vec![4];
        assert_eq!(p.n_tokens(), None);
    }
}
