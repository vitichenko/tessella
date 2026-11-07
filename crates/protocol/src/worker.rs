use crate::error::DomainError;
use crate::layer_ranges::LayerRanges;
use crate::types::{PeerStatus, QuantizationConfig, Role};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    Archive,
    RkyvDeserialize,
    RkyvSerialize,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
)]
#[archive(check_bytes)]
pub enum QueuePriority {
    Leech,
    Standard,
    Vip,
}

impl QueuePriority {
    /// Stable one-byte code of the tier inside the canonical v1 preimage (`TASK-05`).
    /// It is a wire contract: never derive it from the enum order.
    #[must_use]
    pub fn signature_code(self) -> u8 {
        match self {
            QueuePriority::Leech => 0,
            QueuePriority::Standard => 1,
            QueuePriority::Vip => 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
#[archive(check_bytes)]
pub struct KarmaTicket {
    pub client_pubkey: Vec<u8>,
    pub tier: QueuePriority,
    pub expiry: u64,
    pub orchestrator_signature: Vec<u8>,
}

impl KarmaTicket {
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, DomainError> {
        let mut cloned = self.clone();
        cloned.orchestrator_signature = vec![];
        serde_json::to_vec(&cloned).map_err(|e| DomainError::Serialization(e.to_string()))
    }
}

/// One peer's place in a [`RebalancePlan`].
///
/// The shard is a **set** of disjoint ascending ranges ([`crate::layer_ranges::LayerRanges`],
/// `TASK-96`/S102), the same shape `WorkerAssignment` and `Role::Worker` carry. Until
/// `TASK-100`/S106 it was a single `layer_start`/`layer_end` pair, so every writer of a plan
/// collapsed a multi-range shard — the one `TASK-99` repairs a hole with — back to a single range,
/// and a reader of the plan could not tell that two ranges of the same peer were two.
#[derive(
    Serialize, Deserialize, Clone, Debug, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub struct ShardAssignment {
    pub peer_id: String,
    pub layer_ranges: Vec<(usize, usize)>,
    pub replica_group_id: usize,
    pub expert_range: Option<(usize, usize)>,
}

#[derive(
    Serialize, Deserialize, Clone, Debug, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub struct RebalancePlan {
    pub version: u64,
    pub assignments: Vec<ShardAssignment>,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub enum WorkerStatus {
    Pending,
    PendingLoad,
    Ready,
    Offline,
}

#[derive(Serialize, Deserialize, Debug, Clone, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct DeregisterRequest {
    pub peer_id: String,
    pub reason: String,
    /// Protobuf-encoded libp2p public key of the peer that deregisters (D-3 · `TASK-03` Step B).
    ///
    /// Mandatory since `TASK-57`/S63: it used to be `#[serde(default)]` so that a pre-D-3 body still
    /// deserialized and the policy decided. A body without it is not a request of this protocol.
    pub public_key: Vec<u8>,
    /// Ed25519 signature over `peer_canonical_bytes()` (D-3 · `TASK-03` Step B).
    pub signature: Vec<u8>,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_id: String,

    #[serde(default)]
    pub quantization: QuantizationConfig,

    #[serde(default)]
    pub model_type: Option<String>,

    #[serde(default)]
    pub architectures: Vec<String>,

    #[serde(alias = "dim")]
    pub hidden_size: usize,

    #[serde(alias = "n_layers", alias = "num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(alias = "n_experts")]
    pub num_experts: Option<usize>,
    #[serde(alias = "top_k", alias = "num_experts_per_tok")]
    pub num_experts_per_tok: Option<usize>,

    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    pub moe_layer_freq: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,

    pub vocab_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,

    #[serde(alias = "hidden_dim")]
    pub intermediate_size: usize,

    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    pub max_position_embeddings: usize,

    #[serde(default)]
    pub rms_norm_eps: f64,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    pub eos_token_id: Option<u32>,
}

fn default_rope_theta() -> f64 {
    10000.0
}

impl ModelConfig {
    #[must_use]
    pub fn total_layers(&self) -> usize {
        self.num_hidden_layers
    }

    #[must_use]
    pub fn total_experts(&self) -> usize {
        self.num_experts.unwrap_or(0)
    }

    /// The layers this node serves, as declared by its assignment (`TASK-96`/S102).
    ///
    /// A node does **not** choose its shard — the orchestrator computes it
    /// (`musivum::calculate_assignment`) — so this is the assignment read back, in the one place the
    /// engine asks «what do I serve?». Everything downstream (the backend per range, the routing
    /// index, the hop the router sends) is derived from this answer, never from a second copy of
    /// the same fact.
    #[must_use]
    pub fn local_layer_ranges(&self, assignment: &crate::network::WorkerAssignment) -> LayerRanges {
        assignment.layer_ranges.clone()
    }

    pub fn validate_assignment(
        &self,
        assignment: &crate::network::WorkerAssignment,
    ) -> Result<(), DomainError> {
        crate::layer_ranges::validate(&assignment.layer_ranges, self.total_layers())
    }

    #[must_use]
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        match self.moe_layer_freq {
            Some(freq) => {
                if freq == 0 {
                    return false;
                }
                layer_idx.is_multiple_of(freq)
            }
            None => self.is_moe(),
        }
    }

    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some()
    }

    pub fn from_hf_json(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut cfg: Self = serde_json::from_str(&content)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;

        if json["num_key_value_heads"].is_null() {
            cfg.num_key_value_heads = cfg.num_attention_heads;
        }
        if json["rms_norm_eps"].is_null() {
            cfg.rms_norm_eps = 1e-6;
        }
        if let Some(0) = cfg.moe_layer_freq {
            return Err(DomainError::Config("moe_layer_freq cannot be zero".into()).into());
        }
        Ok(cfg)
    }

    /// A config that states **no shape**: every field unknown (`0`, `None`, empty), for a caller to
    /// fill in only the ones it can actually read (`TASK-82`/S89).
    #[must_use]
    pub fn no_shape() -> Self {
        Self {
            model_id: String::new(),
            quantization: QuantizationConfig::None,
            model_type: None,
            architectures: Vec::new(),
            hidden_size: 0,
            num_hidden_layers: 0,
            num_experts: None,
            num_experts_per_tok: None,
            shared_expert_intermediate_size: None,
            moe_layer_freq: None,
            norm_topk_prob: None,
            vocab_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            intermediate_size: 0,
            moe_intermediate_size: None,
            max_position_embeddings: 0,
            rms_norm_eps: 0.0,
            rope_theta: 0.0,
            eos_token_id: None,
        }
    }

    #[must_use]
    pub fn qwen2_5_0_5b() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            quantization: QuantizationConfig::None,
            model_type: Some("qwen2".to_string()),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hidden_size: 896,
            num_hidden_layers: 24,
            num_experts: None,
            num_experts_per_tok: None,
            shared_expert_intermediate_size: None,
            moe_layer_freq: None,
            norm_topk_prob: None,
            vocab_size: 151936,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            intermediate_size: 4864,
            moe_intermediate_size: None,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            eos_token_id: Some(151645),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::qwen2_5_0_5b()
    }
}

/// The peer record of the mesh: identity, both addresses, the shard it serves and its load.
///
/// `layer_ranges` is the **routing** view of the shard (`TASK-96`/S102): the routing table indexes
/// the peer under every layer of every range it declares, and the router asks
/// [`crate::layer_ranges::range_covering`] which of those ranges serves the step it is about to
/// send. A peer with two ranges is therefore one candidate for two disjoint stretches — and the
/// hop it receives names the range, so it never has to guess.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct LayerPeer {
    pub peer_id: String,
    pub http_addr: String,
    pub p2p_addr: String,
    pub layer_ranges: LayerRanges,
    pub expert_range: Option<(usize, usize)>,
    pub queue_depth: usize,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub public_api_port: Option<u16>,
    #[serde(default)]
    pub canary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
#[archive(check_bytes)]
pub struct NodeInfo {
    pub peer_id: String,
    pub host: String,
    pub port: u16,
    pub vram_mb: u32,
    pub role: Role,
    pub model_id: String,
    pub p2p_address: String,
    pub http_api_url: Option<String>,
    pub relay_address: Option<String>,
    pub status: PeerStatus,
    pub failed_requests: u32,
    pub successful_requests: u32,
    pub latency_p99_ms: u64,
    pub queue_depth: usize,
    pub last_error: Option<String>,
    pub slash_count: u32,
    pub blacklisted: bool,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub public_api_port: Option<u16>,
}

/// Minimum number of observed requests before the degraded heuristic is trusted.
pub const MIN_REQUESTS_TO_DEGRADE: u32 = 5;

impl NodeInfo {
    #[must_use]
    pub fn reputation_score(&self) -> f32 {
        self.successful_requests as f32
            / (self.successful_requests + self.failed_requests).max(1) as f32
    }

    #[must_use]
    pub fn is_degraded(&self, degraded_threshold: f64) -> bool {
        let total = self.successful_requests + self.failed_requests;
        if total < MIN_REQUESTS_TO_DEGRADE {
            return false;
        }
        f64::from(self.reputation_score()) < degraded_threshold
    }
}

/// One entry of `GET /peers?model_id=`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeersEntry {
    #[serde(flatten)]
    pub node: NodeInfo,
    #[serde(default)]
    pub canary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
#[archive(check_bytes)]
pub struct ModelManifest {
    pub model_id: String,
    pub num_hidden_layers: usize,
    pub num_experts: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
#[archive(check_bytes)]
pub struct GossipTopology {
    pub model_id: String,
    pub nodes: Vec<NodeInfo>,
    pub manifest: ModelManifest,
}

/// Event pushed from the final pipeline shard back to the entry node's SSE stream.
///
/// It travels **inside the mesh** (`POST /v1/token_stream/{request_id}`), so it is a wire payload of
/// the data plane and lives here with the rest of the node contracts — not in the HTTP crate that
/// happens to decode it (`SOLID-89`). The entry node holds an `mpsc::Sender<TokenStreamPayload>` per
/// active request, keyed by `request_id` in the API state; the engine that seals the payloads and the
/// handler that receives the callback share the type without the data plane having to link Axum.
///
/// `finish_reason` is `Some` only on the last event of a stream (`stop`, `length`, …); the empty
/// string is a legitimate token, so nothing here is optional for size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStreamPayload {
    pub request_id: String,
    pub token: String,
    pub finish_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TASK-82`/S89: the two "empty-ish" configs of this type are **not** the same thing.
    #[test]
    fn no_shape_states_nothing_and_is_not_the_default() {
        let empty = ModelConfig::no_shape();
        for (name, value) in [
            ("model_id", empty.model_id.is_empty()),
            ("model_type", empty.model_type.is_none()),
            ("architectures", empty.architectures.is_empty()),
            ("hidden_size", empty.hidden_size == 0),
            ("num_hidden_layers", empty.num_hidden_layers == 0),
            ("vocab_size", empty.vocab_size == 0),
            ("num_attention_heads", empty.num_attention_heads == 0),
            ("num_key_value_heads", empty.num_key_value_heads == 0),
            ("intermediate_size", empty.intermediate_size == 0),
            (
                "max_position_embeddings",
                empty.max_position_embeddings == 0,
            ),
            ("rms_norm_eps", empty.rms_norm_eps == 0.0),
            ("rope_theta", empty.rope_theta == 0.0),
            ("eos_token_id", empty.eos_token_id.is_none()),
            ("num_experts", empty.num_experts.is_none()),
        ] {
            assert!(
                value,
                "`no_shape()` states nothing, and `{name}` says otherwise"
            );
        }
        assert_eq!(ModelConfig::default().hidden_size, 896);
        assert_ne!(ModelConfig::default().hidden_size, empty.hidden_size);
    }

    #[test]
    fn test_gossip_topology_rkyv() {
        let topology = GossipTopology {
            model_id: "test-model".to_string(),
            nodes: vec![],
            manifest: ModelManifest {
                model_id: "test-model".to_string(),
                num_hidden_layers: 24,
                num_experts: None,
            },
        };
        let bytes = rkyv::to_bytes::<_, 256>(&topology).unwrap();
        let archived = rkyv::check_archived_root::<GossipTopology>(&bytes[..]).unwrap();
        let deserialized: GossipTopology =
            rkyv::Deserialize::<GossipTopology, _>::deserialize(archived, &mut rkyv::Infallible)
                .unwrap();
        assert_eq!(deserialized.model_id, "test-model");
    }

    fn node(successful_requests: u32, failed_requests: u32) -> NodeInfo {
        NodeInfo {
            peer_id: "peer".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8001,
            vram_mb: 6144,
            role: Role::Unassigned,
            model_id: "org/model".to_string(),
            p2p_address: "/ip4/127.0.0.1/tcp/9001".to_string(),
            http_api_url: Some("http://127.0.0.1:8001".to_string()),
            relay_address: None,
            status: PeerStatus::Healthy,
            failed_requests,
            successful_requests,
            latency_p99_ms: 0,
            queue_depth: 0,
            last_error: None,
            slash_count: 0,
            blacklisted: false,
            public_ip: None,
            public_api_port: None,
        }
    }

    #[test]
    fn is_degraded_honours_the_threshold_it_is_given() {
        let flaky = node(5, 5);

        assert!(flaky.is_degraded(0.7), "0.50 < 0.70 => degraded");
        assert!(!flaky.is_degraded(0.5), "0.50 is not < 0.50");
        assert!(!flaky.is_degraded(0.3));

        assert!(!node(1, 1).is_degraded(1.0));
        assert!(!node(0, 0).is_degraded(1.0));
    }

    #[test]
    fn peers_entry_and_layer_peer_canary_serde_compatibility() {
        let json_with_canary = serde_json::json!({
            "peer_id": "12D3KooWCanary",
            "host": "127.0.0.1",
            "port": 9001,
            "vram_mb": 4096,
            "role": {
                "type": "Worker",
                "layer_ranges": [[0, 8]],
                "expert_range": null
            },
            "model_id": "test-model",
            "p2p_address": "/ip4/127.0.0.1/tcp/9001",
            "http_api_url": "http://127.0.0.1:8001",
            "status": "Healthy",
            "failed_requests": 0,
            "successful_requests": 10,
            "latency_p99_ms": 1,
            "queue_depth": 0,
            "slash_count": 0,
            "blacklisted": false,
            "canary": true
        });

        let entry: PeersEntry = serde_json::from_value(json_with_canary.clone()).unwrap();
        assert!(entry.canary);
        assert_eq!(entry.node.peer_id, "12D3KooWCanary");

        let mut json_without_canary = json_with_canary;
        json_without_canary
            .as_object_mut()
            .unwrap()
            .remove("canary");
        let entry_no_canary: PeersEntry = serde_json::from_value(json_without_canary).unwrap();
        assert!(!entry_no_canary.canary);

        let lp_json = serde_json::json!({
            "peer_id": "12D3KooWPeer",
            "http_addr": "http://127.0.0.1:8001",
            "p2p_addr": "/ip4/127.0.0.1/tcp/9001",
            "layer_ranges": [[0, 8], [21, 28]],
            "expert_range": null,
            "queue_depth": 1
        });
        let lp: LayerPeer = serde_json::from_value(lp_json).unwrap();
        assert!(!lp.canary, "canary must default to false when omitted");
        assert_eq!(lp.layer_ranges, vec![(0, 8), (21, 28)]);

        // `TASK-96`/S102: the single-range shape is **retired**, not aliased. A body that still
        // carries a scalar `layer_range` is not a `LayerPeer` of this protocol any more — the
        // compatibility window closes with the change that opens it (Invariant 2 · `SKILL §3`),
        // instead of leaving two shapes alive and a reader that has to guess which one it got.
        let retired_shape = serde_json::json!({
            "peer_id": "12D3KooWPeer",
            "http_addr": "http://127.0.0.1:8001",
            "p2p_addr": "/ip4/127.0.0.1/tcp/9001",
            "layer_range": [0, 8],
            "expert_range": null,
            "queue_depth": 1
        });
        assert!(
            serde_json::from_value::<LayerPeer>(retired_shape).is_err(),
            "a peer body in the retired single-range shape must not parse"
        );
    }
}
