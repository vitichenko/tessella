use crate::layer_ranges::LayerRanges;
use crate::models::{KarmaTicket, ModelConfig, ModelManifest};
use crate::types::{ComputeDevice, PeerStatus, Role, TensorView};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
#[archive(check_bytes)]
pub struct MoEForwardRequest {
    pub input_tensor: Option<TensorView>,
    pub client_peer_id: Option<String>,
    pub session_id: Option<String>,
    pub tokens: Vec<u32>,
    pub speculative_count: u32,
    pub entry_node_url: Option<String>,
    #[serde(default)]
    pub signature: Vec<u8>,
    #[serde(default)]
    pub client_pubkey: Vec<u8>,
    pub karma_ticket: Option<KarmaTicket>,
    /// Version of the signed preimage (H-09 · `TASK-05`, D-2=(b)): `0`/absent = legacy
    /// `bincode(session_id)`, `1` = canonical [`crate::signing::SignContext::v1_preimage`],
    /// anything else is rejected. Additive, so a legacy emitter still deserializes.
    #[serde(default)]
    pub signature_version: u8,
    /// One-shot value of a v1 preimage; a repeated `(session_id, nonce)` is a replay.
    #[serde(default)]
    pub nonce: u64,
    /// Unix seconds at which the v1 preimage was signed.
    #[serde(default)]
    pub iat: u64,
}

impl crate::signing::SignedRequest for MoEForwardRequest {
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

    fn sign_context(&self) -> crate::signing::SignContext<'_> {
        // This DTO carries no `request_id`: its single identifier is the `session_id`, so both
        // preimage fields take it (documented, not accidental).
        let session = self.session_id.as_deref().unwrap_or("");
        crate::signing::SignContext {
            session_id: session,
            request_id: session,
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

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub enum WorkerGossipMsg {
    WorkerJoined {
        peer_id: String,
        http_addr: String,
        p2p_addr: String,
        /// The ranges the joining peer serves (`TASK-96`/S102). The gossip message is a routing
        /// input for every node that hears it, so it carries the whole shard — a peer announced with
        /// only its first chunk would be indexed under one stretch and invisible to the other.
        layer_ranges: LayerRanges,
    },
    WorkerLeft {
        peer_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct AnnounceRequest {
    pub peer_id: String,
    pub host: String,
    pub port: u16,
    pub vram_mb: u32,
    pub model_id: String,
    pub current_role: Role,
    pub p2p_address: String,
    pub http_api_url: Option<String>,
    pub relay_address: Option<String>,
    pub status: Option<PeerStatus>,
    pub queue_depth: usize,
    pub expert_usage: Option<std::collections::HashMap<u32, u64>>,
    pub parameter_contract_hash: String,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub public_api_port: Option<u16>,
    /// Protobuf-encoded libp2p public key of the peer that announces (D-3 · `TASK-03` Step B).
    ///
    /// Mandatory since `TASK-57`/S63: it used to be `#[serde(default)]` so that a pre-D-3 body
    /// deserialized and the policy decided what to do with it. An unsigned body is not a request of
    /// this protocol any more — the parser refuses it (`422`) before any policy could apply.
    pub public_key: Vec<u8>,
    /// Ed25519 signature over `peer_canonical_bytes()` (D-3 · `TASK-03` Step B).
    pub signature: Vec<u8>,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
#[serde(tag = "type")]
pub enum AnnounceResponse {
    Success {
        assigned_role: Role,
        active_workers: usize,
        seed_address: Option<String>,
    },
    AtCapacity {
        message: String,
    },
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub struct WorkerRegistration {
    pub peer_id: String,
    pub p2p_address: String,
    pub relay_address: Option<String>,
    pub api_addr: String,
    pub vram_mb: u32,
    pub device: ComputeDevice,
    pub model_id: String,
    pub total_layers: usize,
    pub force_full_model: Option<bool>,
    pub model_config: Option<ModelConfig>,
    pub manifest: ModelManifest,
    pub parameter_contract_hash: String,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub public_api_port: Option<u16>,
    /// Protobuf-encoded libp2p public key of the peer that registers (`TASK-49`/S55).
    ///
    /// `/register` was the last anonymous endpoint that **writes**: it overwrote
    /// `state.workers[peer_id]` — assignment included — with whatever body arrived, so anybody could
    /// hijack another node's assignment or take a slot in the mesh under a made-up peer id, and it
    /// is also what fills the P2P admission cache (`authorized_peers`), i.e. the door of the swarm.
    /// The admission policy of `private` needs a **proven** identity, and this is where it comes
    /// from.
    pub public_key: Vec<u8>,
    /// Ed25519 signature over [`PeerSignedRequest::peer_canonical_bytes`] (`TASK-49`/S55).
    ///
    /// Mandatory since `TASK-57`/S63: the two fields used to be `serde(default)` so that the parser
    /// accepted a pre-S55 body and the policy decided (a `private` deployment always refused an
    /// unsigned registration). A registration without them is not a body of this protocol, so the
    /// allow-list of `private` can never be reached with an unproven identity in the first place.
    pub signature: Vec<u8>,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub struct UpdateAssignmentRequest {
    pub assignment: WorkerAssignment,
    #[serde(default)]
    pub version: u64,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq, Eq,
)]
pub struct StaleVersionError {
    pub error: String,
    pub current_version: u64,
    pub rejected_version: u64,
}

impl StaleVersionError {
    pub fn new(current_version: u64, rejected_version: u64) -> Self {
        Self {
            error: "stale_version".to_string(),
            current_version,
            rejected_version,
        }
    }
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub struct WorkerAssignment {
    /// The layers this node serves (`TASK-96`/S102): disjoint, ascending, half-open ranges of the
    /// **same** replica group. One range is the common case; a second one exists because a node
    /// with surplus VRAM can take another chunk of the same chain — the budget already charges the
    /// edge tensors once per node (`tessella_core::assignment::layer_budget`).
    ///
    /// The orchestrator computes it (`calculate_assignment`); the node never chooses its shard.
    pub layer_ranges: LayerRanges,
    pub model_repo: String,
    pub replica_group_id: usize,
    pub expert_range: Option<(usize, usize)>,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub struct WorkerReady {
    pub peer_id: String,
    pub model_repo: String,
    /// What the node reports it serves (`TASK-96`/S102), read back from its assignment: readiness
    /// is a claim over a set of ranges now, not over one.
    pub layer_ranges: LayerRanges,
    pub replica_group_id: usize,
    pub expert_range: Option<(usize, usize)>,
    pub queue_depth: usize,
    /// Protobuf-encoded libp2p public key of the peer that reports readiness
    /// (D-3 · `TASK-03` Step B). Mandatory since `TASK-57`/S63.
    pub public_key: Vec<u8>,
    /// Ed25519 signature over `peer_canonical_bytes()` (D-3 · `TASK-03` Step B).
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FailureReport {
    pub peer_id: String,
    pub request_id: String,
    pub error: String,
    pub timestamp: u64,
    /// Protobuf-encoded libp2p public key of the peer that reports (D-3 · `TASK-03` Step B).
    /// Mandatory since `TASK-57`/S63.
    pub public_key: Vec<u8>,
    /// Ed25519 signature over `peer_canonical_bytes()` (D-3 · `TASK-03` Step B).
    pub signature: Vec<u8>,
}

/// A weight-poisoning accusation carrying the sender's (verifiable) transfer receipt.
///
/// It used to be an anonymous inline variant of the untagged `FailureReportPayload` enum, so
/// the `reporter_id` it declares was nobody in particular (H-11). D-3 · `TASK-03` Step B
/// promotes it to a real DTO with the same two authentication fields as the other signed
/// requests, and the orchestrator only accepts it when the signature belongs to `reporter_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightPoisoningReport {
    /// Peer that accuses. It must be the owner of the signing key.
    pub reporter_id: String,
    /// Peer accused of serving weights that do not match the manifest.
    pub target_peer_id: String,
    pub reason: String,
    pub model_id: String,
    pub role: String,
    #[serde(default)]
    pub transfer_receipt: Option<Box<crate::models::SignedTransferReceipt>>,
    /// Protobuf-encoded libp2p public key of the reporter (D-3 · `TASK-03` Step B).
    /// Mandatory since `TASK-57`/S63.
    pub public_key: Vec<u8>,
    /// Ed25519 signature over `peer_canonical_bytes()` (D-3 · `TASK-03` Step B).
    pub signature: Vec<u8>,
}

/// The swarm's standard sampling parameters (`TASK-40`/S46, SP-07 · D-16).
///
/// They live here — and not in `tessella-inference` — because they are read by **both** planes:
/// the data plane samples with them when a request does not ask for anything else, and the control
/// plane builds the parameter contract the mesh checks on `/register`, `/announce` and `/ready`
/// out of them. `tessella-protocol` is the only crate both depend on.
pub const SWARM_TEMPERATURE: f32 = 0.2;
pub const SWARM_TOP_P: f32 = 0.9;
pub const SWARM_TOP_K: u32 = 40;

/// The parameter contract of the mesh: the model **plus the swarm standard**.
///
/// It used to be fed by the worker's `[worker] temperature` (`network_manager.rs`), while the
/// sampling implementation hardcoded the standard anyway — so an operator who changed that key
/// made its node answer `Non-standard generation parameters` and left it out of the swarm
/// **without** changing a single byte of its output (SP-07). The canonical string is unchanged
/// (pinned by `test_the_swarm_contract_hash_is_a_fixed_vector`), so the wire does not move.
#[must_use]
pub fn swarm_contract_hash(model_name: &str) -> String {
    compute_parameter_contract_hash(model_name, SWARM_TEMPERATURE, SWARM_TOP_P, SWARM_TOP_K)
}

pub fn compute_parameter_contract_hash(
    model_name: &str,
    temperature: f32,
    top_p: f32,
    top_k: u32,
) -> String {
    use sha2::{Digest, Sha256};
    let canonical = format!("{}:{}:{}:{}", model_name, temperature, top_p, top_k);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let hash = hasher.finalize();
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_contract_hash_determinism() {
        let hash1 = compute_parameter_contract_hash("test-model", 0.2, 0.9, 40);
        let hash2 = compute_parameter_contract_hash("test-model", 0.2, 0.9, 40);
        assert_eq!(hash1, hash2);

        // Different temperature must result in different hash
        let hash3 = compute_parameter_contract_hash("test-model", 0.3, 0.9, 40);
        assert_ne!(hash1, hash3);

        // Different model name must result in different hash
        let hash4 = compute_parameter_contract_hash("other-model", 0.2, 0.9, 40);
        assert_ne!(hash1, hash4);
    }

    /// `TASK-40`/S46 (D-16): the swarm standard lives in **one** place and the contract the mesh
    /// checks is computed from it, so it can no longer be fed by a node's configuration (before,
    /// an operator changing `[worker] temperature` made `/register` and `/announce` answer
    /// `Non-standard generation parameters` and left an honest node out of the swarm **without**
    /// changing a single byte of its output).
    #[test]
    fn test_the_swarm_standard_is_one_definition_and_the_contract_is_built_from_it() {
        assert_eq!(SWARM_TEMPERATURE, 0.2);
        assert_eq!(SWARM_TOP_P, 0.9);
        assert_eq!(SWARM_TOP_K, 40);

        assert_eq!(
            swarm_contract_hash("test-model"),
            compute_parameter_contract_hash(
                "test-model",
                SWARM_TEMPERATURE,
                SWARM_TOP_P,
                SWARM_TOP_K
            )
        );
        assert_ne!(
            swarm_contract_hash("test-model"),
            swarm_contract_hash("other-model")
        );
    }

    /// The canonical string of the contract is part of the wire: pin it as a fixed vector so a
    /// change of format (which would expel every deployed worker) is a deliberate decision.
    #[test]
    fn test_the_swarm_contract_hash_is_a_fixed_vector() {
        assert_eq!(
            swarm_contract_hash("test-model"),
            "d767c918b3b07df21530925efa452497246d85e17f95eeafa1b5e2c6c69c7a25"
        );
    }
}
