use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    Archive,
    RkyvDeserialize,
    RkyvSerialize,
    Default,
    PartialEq,
    Copy,
)]
/// How the weights of a shard are stored (`S51`/`TASK-45`).
///
/// `Q4K` used to be a third variant here and **nothing else**: no loader read it (the only hook is
/// `worker/src/hf_loader.rs`, which recognises `q8_0`), no consumer referenced it and no measurement
/// backed its four-bits-per-parameter promise — so it was retired instead of kept as a phantom option
/// that silently fell back to unquantized weights. It is a `tessella-protocol` type, so removing it is
/// a **breaking** change of the payload schema: declared as breaking, deployed with the whole
/// cluster at once, and a `worker.toml` that still says `"Q4K"` fails to parse naming the value.
pub enum QuantizationConfig {
    #[default]
    None,
    #[allow(non_camel_case_types)]
    Q8_0,
}

pub type ExpertId = u32;

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Archive,
    RkyvDeserialize,
    RkyvSerialize,
    Default,
)]
#[archive(check_bytes)]
#[serde(rename_all = "lowercase", tag = "type")]
pub enum InferenceMode {
    #[default]
    Chat,
    Fim {
        prefix: String,
        suffix: String,
    },
}

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub enum ComputeDevice {
    Cpu,
    Cuda { device_id: usize },
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
#[archive(check_bytes)]
pub struct TensorView {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub ttft_ms: f64,
    pub inter_token_latency_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
}

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    Default,
    PartialEq,
    Archive,
    RkyvDeserialize,
    RkyvSerialize,
)]
#[archive(check_bytes)]
#[serde(tag = "type")]
pub enum Role {
    #[default]
    Unassigned,
    Worker {
        #[serde(default)]
        layer_ranges: Vec<(usize, usize)>,
        #[serde(default)]
        replica_group_id: usize,
        #[serde(default)]
        expert_range: Option<(usize, usize)>,
    },
}

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Archive, RkyvDeserialize, RkyvSerialize,
)]
#[archive(check_bytes)]
pub enum PeerStatus {
    Healthy,
    Degraded,
    Slashed,
    Offline,
    Pending,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub in_flight: u32,
    pub success: u64,
    pub failure: u64,
    pub last_seen_secs_ago: u64,
    pub expert_activations: u64,
    pub expert_usage: HashMap<u32, u64>,
}
