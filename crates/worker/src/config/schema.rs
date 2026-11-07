use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tessella_core::config::NetworkConfig;
use tracing::warn;

use crate::config::derived::{
    default_api_host, default_api_port, default_device, default_oom_streak, default_p2p_host,
    default_p2p_inference_timeout_secs, default_p2p_port, default_registration_grace_secs,
    default_streak_window_secs, default_true, default_wal_checkpoint_interval_secs,
    quantization_name,
};
use crate::config::env::apply_env_overrides;
use crate::config::validate::{
    has_worker_table, removed_keys_in, retired_key_reason, validate_invariants,
};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct UnifiedConfigToml {
    pub orchestrator_url: String,
    pub keypair_path: String,
    pub default_model: String,
    pub worker: WorkerTomlSection,
    #[serde(default)]
    pub gpu: Option<GpuTomlSection>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct GpuTomlSection {
    #[serde(default)]
    pub oom_streak: Option<u32>,
    #[serde(default)]
    pub streak_window_secs: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WorkerTomlSection {
    pub vram_mb: u32,
    pub models_dir: String,
    pub api_host: Option<String>,
    pub api_port: u16,
    pub p2p_host: Option<String>,
    pub p2p_port: u16,
    pub public_ip: Option<String>,
    pub public_p2p_port: Option<u16>,
    pub public_api_port: Option<u16>,
    pub public_api_url: Option<String>,
    /// `"cuda" | "metal"` — validated by [`validated_device`] (Tessella is GPU-only).
    pub device: Option<String>,
    /// The prompt this node's VRAM can **prefill**, in tokens (`TASK-76`/S83); `0`/absent = undeclared.
    #[serde(default)]
    pub max_input_tokens: Option<usize>,
    /// How long a hop of this node's pipeline may take, in seconds (`TASK-76`/S83).
    #[serde(default)]
    pub p2p_inference_timeout_secs: Option<u64>,
    /// Opt-in for private/loopback `entry_node_url` targets (H-10).
    #[serde(default)]
    pub allow_private_addrs: Option<bool>,
    /// Host suffixes allowed as `entry_node_url` targets even when not in the topology (H-10).
    #[serde(default)]
    pub callback_allowed_hosts: Vec<String>,
    /// Cadence of the SQLite WAL maintenance task (`TASK-13`, H-21); `0` disables it.
    #[serde(default)]
    pub wal_checkpoint_interval_secs: Option<u64>,
    /// Join the Kademlia DHT (`TASK-15`). Absent ⇒ enabled.
    #[serde(default)]
    pub kademlia_enabled: Option<bool>,
    /// How long the boot may wait for the tracker before giving up (`TASK-62`/S68). Absent ⇒
    /// [`crate::registration::DEFAULT_REGISTRATION_GRACE_SECS`]; `0` = the pre-S68 fail-fast boot.
    #[serde(default)]
    pub registration_grace_secs: Option<u64>,
    /// How the weights of this node are stored (`S51`/`TASK-45`): `"None"` (default, unquantized) or
    /// `"Q8_0"`.
    #[serde(default)]
    pub quantization: Option<tessella_core::models::QuantizationConfig>,
    /// `"off" | "echo" | "scale"`: this node answers with a known-wrong tensor instead of computing
    /// its layers (`TASK-71` · S78 · `D-28`).
    #[serde(default)]
    pub canary_mode: Option<CanaryMode>,
    /// Whether this node prunes the shards its new assignment no longer needs (`S80`/`TASK-73`):
    /// `"off"` (default), `"dry-run"` or `"on"`.
    #[serde(default)]
    pub model_cache_prune: Option<CachePruneMode>,
    /// Consecutive OOMs required to declare saturation (`TASK-85`/S92); default: 3.
    #[serde(default)]
    pub oom_streak: Option<u32>,
    /// Time window in seconds for the OOM streak (`TASK-85`/S92); default: 30.
    #[serde(default)]
    pub streak_window_secs: Option<u64>,
}

/// What a **canary** node answers instead of computing its layers (`TASK-71` · S78 · decision `D-28`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CanaryMode {
    /// Not a canary (default).
    #[default]
    #[serde(alias = "off", alias = "none", alias = "disabled")]
    Off,
    /// Answer every step with its input unchanged.
    #[serde(alias = "identity")]
    Echo,
    /// Answer every step with its input multiplied by a constant factor.
    Scale,
}

impl CanaryMode {
    #[must_use]
    pub fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// The value a `TESSELLA_CANARY_MODE` spelling selects.
    pub fn parse(raw: &str) -> Result<Self, anyhow::Error> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" | "false" | "0" => Ok(Self::Off),
            "echo" | "identity" => Ok(Self::Echo),
            "scale" | "x2" => Ok(Self::Scale),
            other => anyhow::bail!(
                "TESSELLA_CANARY_MODE='{other}' is not a canary mode: accepted values are \
                 \"off\" (default), \"echo\" (answer with the input unchanged) and \"scale\" \
                 (answer with the input multiplied by {})",
                crate::canary::CANARY_SCALE_FACTOR
            ),
        }
    }
}

/// What a node does with the shards its cache holds and its **new** range no longer needs
/// (`TASK-73` · S80 · finding F-5 of the S74 spike).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CachePruneMode {
    /// Never touch the cache (default).
    #[default]
    #[serde(rename = "off", alias = "none", alias = "disabled", alias = "false")]
    Off,
    /// Log the plan, delete nothing.
    #[serde(rename = "dry-run", alias = "dry_run", alias = "dryrun")]
    DryRun,
    /// Delete the shards this node no longer serves.
    #[serde(rename = "on", alias = "true")]
    On,
}

impl CachePruneMode {
    #[must_use]
    pub fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// `dry-run` inspects the cache but deletes nothing.
    #[must_use]
    pub fn deletes(self) -> bool {
        matches!(self, Self::On)
    }

    /// The spelling used in the configuration file and in the logs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::DryRun => "dry-run",
            Self::On => "on",
        }
    }

    /// The value a `TESSELLA_MODEL_CACHE_PRUNE` spelling selects.
    pub fn parse(raw: &str) -> Result<Self, anyhow::Error> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" | "false" | "0" => Ok(Self::Off),
            "dry-run" | "dry_run" | "dryrun" => Ok(Self::DryRun),
            "on" | "true" | "1" => Ok(Self::On),
            other => anyhow::bail!(
                "TESSELLA_MODEL_CACHE_PRUNE='{other}' is not a cache-prune mode: accepted values \
                 are \"off\" (default, the cache is never touched), \"dry-run\" (log the plan and \
                 delete nothing) and \"on\" (delete the shards of this node's own model that its new \
                 assignment does not need)"
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerConfig {
    // Network
    pub listen_port: u16, // P2P port
    #[serde(default = "default_api_host")]
    pub api_host: String,
    #[serde(default = "default_api_port")]
    pub api_port: u16,
    #[serde(default = "default_p2p_host")]
    pub p2p_host: String,
    #[serde(default = "default_p2p_port")]
    pub p2p_port: u16,
    #[serde(alias = "musivum_url")]
    pub orchestrator_url: String,

    // Credentials
    pub hf_token: String, // HuggingFace token
    #[serde(alias = "model_id")]
    pub default_model: String,
    #[serde(default)]
    pub keypair_path: Option<String>,

    // Hardware (capabilities reported to the musivum)
    #[serde(alias = "vram_gb")]
    pub vram_budget_gb: f32,
    pub device: String, // "cuda" | "metal" (Tessella is GPU-only: see `validated_device`)

    /// The prompt this node's VRAM can **prefill**, in tokens (`TASK-76`/S83). `0` = undeclared.
    #[serde(default)]
    pub max_input_tokens: usize,

    pub models_dir: String,

    #[serde(default)]
    pub quantization: tessella_core::models::QuantizationConfig,

    /// Override: force full model assignment regardless of VRAM.
    #[serde(default)]
    pub force_full_model: bool,

    pub max_new_tokens: usize,
    pub p2p_inference_timeout_secs: u64,
    pub db_path: String,
    pub heartbeat_interval_secs: u64,
    pub orchestrator_public_key: Option<String>,
    #[serde(default)]
    pub admin_token: Option<String>,

    // mDNS
    #[serde(default = "default_true")]
    pub mdns_enabled: bool,

    /// Join the Kademlia DHT (`TASK-15`, H-06).
    #[serde(default = "default_true")]
    pub kademlia_enabled: bool,

    // Public IP
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub public_p2p_port: Option<u16>,
    #[serde(default)]
    pub public_api_port: Option<u16>,

    /// Full public HTTP URL announced to the swarm (scheme + host + optional port/path).
    #[serde(default)]
    pub public_api_url: Option<String>,

    /// Accept `entry_node_url` token-stream callbacks to private/loopback/link-local addresses
    /// (`ALLOW_PRIVATE_ADDRS`, H-10).
    #[serde(default)]
    pub allow_private_addrs: bool,

    /// Host suffixes accepted as `entry_node_url` targets even when the host is not one of the
    /// peers of this mesh.
    #[serde(default)]
    pub callback_allowed_hosts: Vec<String>,

    /// How often the worker consolidates and truncates its SQLite write-ahead log (`TASK-13`, H-21).
    #[serde(default = "default_wal_checkpoint_interval_secs")]
    pub wal_checkpoint_interval_secs: u64,

    /// How long this node may wait for the tracker **at boot** before giving up (`TASK-62`/S68).
    #[serde(default = "default_registration_grace_secs")]
    pub registration_grace_secs: u64,

    /// What this node answers **instead of computing its layers** (`TASK-71` · S78 · decision `D-28`).
    #[serde(default)]
    pub canary_mode: CanaryMode,

    /// Whether this node **prunes** the shards the cache holds and its new assignment no longer needs.
    #[serde(default)]
    pub model_cache_prune: CachePruneMode,

    /// Consecutive OOMs required to declare saturation (`TASK-85`/S92); default: 3.
    #[serde(default = "default_oom_streak")]
    pub oom_streak: u32,

    /// Time window in seconds for the OOM streak (`TASK-85`/S92); default: 30.
    #[serde(default = "default_streak_window_secs")]
    pub streak_window_secs: u64,

    // Network internal config
    #[serde(default)]
    pub network: NetworkConfig,

    #[serde(default)]
    pub _experts_per_node: u32,
    #[serde(default)]
    pub _model_shard: u8,
    #[serde(default)]
    pub _inference_timeout_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            listen_port: 9080,
            api_host: "0.0.0.0".to_string(),
            api_port: 8080,
            p2p_host: "0.0.0.0".to_string(),
            p2p_port: 9080,
            orchestrator_url: "http://127.0.0.1:8080".to_string(),
            hf_token: "".to_string(),
            default_model: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            vram_budget_gb: 8.0,
            device: "cuda".to_string(),
            max_input_tokens: 0,
            models_dir: "./model_cache".to_string(),
            quantization: tessella_core::models::QuantizationConfig::None,
            force_full_model: false,
            max_new_tokens: 256,
            p2p_inference_timeout_secs: default_p2p_inference_timeout_secs(),
            db_path: "worker.db".to_string(),
            heartbeat_interval_secs: 60,
            orchestrator_public_key: None,
            admin_token: None,
            mdns_enabled: true,
            kademlia_enabled: true,
            public_ip: None,
            public_p2p_port: None,
            network: NetworkConfig::default(),
            _experts_per_node: 1,
            _model_shard: 1,
            _inference_timeout_secs: 30,
            keypair_path: None,
            public_api_port: None,
            public_api_url: None,
            allow_private_addrs: false,
            callback_allowed_hosts: Vec::new(),
            wal_checkpoint_interval_secs: default_wal_checkpoint_interval_secs(),
            registration_grace_secs: default_registration_grace_secs(),
            canary_mode: CanaryMode::Off,
            model_cache_prune: CachePruneMode::Off,
            oom_streak: default_oom_streak(),
            streak_window_secs: default_streak_window_secs(),
        }
    }
}

impl WorkerConfig {
    /// Loads the configuration from a TOML file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed, or if `device` does not select
    /// a supported GPU backend (Tessella is GPU-only: `cuda` or `metal`).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let mut cfg = if !path.as_ref().exists() {
            Self::default()
        } else {
            let content = fs::read_to_string(&path)?;

            // S45/S46/S52/S56: a key that no longer exists is named — with what replaces it —,
            // never ignored in silence.
            for key in removed_keys_in(&content) {
                warn!(
                    "⚠️ '{key}' in {} was retired: {}",
                    path.as_ref().display(),
                    retired_key_reason(key)
                );
            }

            match toml::from_str::<UnifiedConfigToml>(&content) {
                Ok(unified) => Self {
                    orchestrator_url: unified.orchestrator_url,
                    keypair_path: Some(unified.keypair_path),
                    default_model: unified.default_model,
                    vram_budget_gb: unified.worker.vram_mb as f32 / 1024.0,
                    models_dir: unified.worker.models_dir,
                    api_host: unified.worker.api_host.unwrap_or_else(default_api_host),
                    api_port: unified.worker.api_port,
                    p2p_host: unified.worker.p2p_host.unwrap_or_else(default_p2p_host),
                    p2p_port: unified.worker.p2p_port,
                    listen_port: unified.worker.p2p_port,
                    public_ip: unified.worker.public_ip,
                    public_p2p_port: unified.worker.public_p2p_port,
                    public_api_port: unified.worker.public_api_port,
                    public_api_url: unified.worker.public_api_url,
                    device: unified.worker.device.unwrap_or_else(default_device),
                    // H-10: honoured, not dropped (the defect `device` had until S27).
                    allow_private_addrs: unified.worker.allow_private_addrs.unwrap_or_default(),
                    callback_allowed_hosts: unified.worker.callback_allowed_hosts,
                    // H-21: same rule — a key of the unified schema must not be dropped in silence.
                    wal_checkpoint_interval_secs: unified
                        .worker
                        .wal_checkpoint_interval_secs
                        .unwrap_or_else(default_wal_checkpoint_interval_secs),
                    // TASK-15: same rule again (the defect `device` had until S27).
                    kademlia_enabled: unified.worker.kademlia_enabled.unwrap_or(true),
                    // S68: same rule again.
                    registration_grace_secs: unified
                        .worker
                        .registration_grace_secs
                        .unwrap_or_else(default_registration_grace_secs),
                    // S51: `quantization` of the unified schema is honoured, not dropped.
                    quantization: unified.worker.quantization.unwrap_or_default(),
                    // S78: same rule again.
                    canary_mode: unified.worker.canary_mode.unwrap_or_default(),
                    // S80: same rule again (the cache-prune key of the unified schema is honoured).
                    model_cache_prune: unified.worker.model_cache_prune.unwrap_or_default(),
                    // S83: same rule again (the prefill ceiling of the unified schema is honoured).
                    max_input_tokens: unified.worker.max_input_tokens.unwrap_or_default(),
                    // S83: and the declared hop wait — the key that had no reader at all until this
                    // session, whichever schema declared it.
                    p2p_inference_timeout_secs: unified
                        .worker
                        .p2p_inference_timeout_secs
                        .unwrap_or_else(default_p2p_inference_timeout_secs),
                    // S92: saturation thresholds from [gpu] or [worker] section.
                    oom_streak: unified
                        .gpu
                        .as_ref()
                        .and_then(|g| g.oom_streak)
                        .or(unified.worker.oom_streak)
                        .unwrap_or_else(default_oom_streak),
                    streak_window_secs: unified
                        .gpu
                        .as_ref()
                        .and_then(|g| g.streak_window_secs)
                        .or(unified.worker.streak_window_secs)
                        .unwrap_or_else(default_streak_window_secs),
                    ..Self::default()
                },
                Err(err) => {
                    // `S51`/`TASK-45`: the fallback to the flat legacy schema exists for the old
                    // single-level `worker.toml`, and it silently ignores **every** key it does not
                    // know. Applied to a file that *has* a `[worker]` table, a typo (or a retired
                    // value such as `quantization = "Q4K"`) made the whole table disappear: the node
                    // booted with defaults, unquantized and with the wrong VRAM claim, and said
                    // nothing. A `[worker]` table that does not parse is an error now, with the
                    // parser's message.
                    if has_worker_table(&content) {
                        return Err(anyhow::anyhow!(
                            "the [worker] table of {} could not be parsed and it is not ignored in \
                             silence: {err}",
                            path.as_ref().display()
                        ));
                    }
                    let mut legacy_cfg: Self = toml::from_str(&content)?;
                    if !content.contains("api_port") {
                        legacy_cfg.api_port = legacy_cfg.listen_port
                            + tessella_core::constants::network::API_PORT_OFFSET;
                    }
                    legacy_cfg.p2p_port = legacy_cfg.listen_port;
                    legacy_cfg
                }
            }
        };

        apply_env_overrides(&mut cfg)?;
        validate_invariants(&mut cfg)?;

        Ok(cfg)
    }

    /// Saves the configuration to a TOML file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), anyhow::Error> {
        let parent = path
            .as_ref()
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid config path"))?;
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
        let mut worker_section = format!(
            r#"vram_mb = {}
models_dir = "{}"
api_host = "{}"
api_port = {}
p2p_host = "{}"
p2p_port = {}
device = "{}"
"#,
            (self.vram_budget_gb * 1024.0) as u32,
            self.models_dir,
            self.api_host,
            self.api_port,
            self.p2p_host,
            self.p2p_port,
            self.device
        );
        if let Some(ref ip) = self.public_ip {
            worker_section.push_str(&format!("public_ip = \"{}\"\n", ip));
        }
        if let Some(port) = self.public_p2p_port {
            worker_section.push_str(&format!("public_p2p_port = {}\n", port));
        }
        if let Some(port) = self.public_api_port {
            worker_section.push_str(&format!("public_api_port = {}\n", port));
        }
        if let Some(ref url) = self.public_api_url {
            worker_section.push_str(&format!("public_api_url = \"{}\"\n", url));
        }
        if self.allow_private_addrs {
            worker_section.push_str("allow_private_addrs = true\n");
        }
        if !self.callback_allowed_hosts.is_empty() {
            let hosts = self
                .callback_allowed_hosts
                .iter()
                .map(|host| format!("\"{host}\""))
                .collect::<Vec<_>>()
                .join(", ");
            worker_section.push_str(&format!("callback_allowed_hosts = [{hosts}]\n"));
        }
        if self.wal_checkpoint_interval_secs != default_wal_checkpoint_interval_secs() {
            worker_section.push_str(&format!(
                "wal_checkpoint_interval_secs = {}\n",
                self.wal_checkpoint_interval_secs
            ));
        }
        if !self.kademlia_enabled {
            worker_section.push_str("kademlia_enabled = false\n");
        }
        if self.registration_grace_secs != default_registration_grace_secs() {
            worker_section.push_str(&format!(
                "registration_grace_secs = {}\n",
                self.registration_grace_secs
            ));
        }
        if self.quantization != tessella_core::models::QuantizationConfig::None {
            worker_section.push_str(&format!(
                "quantization = \"{}\"\n",
                quantization_name(self.quantization)
            ));
        }
        if self.model_cache_prune != CachePruneMode::Off {
            worker_section.push_str(&format!(
                "model_cache_prune = \"{}\"\n",
                self.model_cache_prune.as_str()
            ));
        }
        if self.p2p_inference_timeout_secs != default_p2p_inference_timeout_secs() {
            worker_section.push_str(&format!(
                "p2p_inference_timeout_secs = {}\n",
                self.p2p_inference_timeout_secs
            ));
        }
        if self.oom_streak != default_oom_streak() {
            worker_section.push_str(&format!("oom_streak = {}\n", self.oom_streak));
        }
        if self.streak_window_secs != default_streak_window_secs() {
            worker_section.push_str(&format!(
                "streak_window_secs = {}\n",
                self.streak_window_secs
            ));
        }

        let config_content = format!(
            r#"# Global identity and orchestrator
orchestrator_url = "{}"
keypair_path = "{}"
default_model = "{}"

[worker]
{}"#,
            self.orchestrator_url,
            self.keypair_path.clone().unwrap_or_default(),
            self.default_model,
            worker_section
        );
        fs::write(path, config_content)?;
        Ok(())
    }
}
