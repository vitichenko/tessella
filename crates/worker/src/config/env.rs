//! Environment variable overrides and configuration precedence rules (SOLID-91).
//!
//! # Configuration Precedence Order (Per Field)
//!
//! For every configuration parameter, Tessella resolves the final value in strict precedence order:
//!
//! 1. **Environment Variables**: Highest priority. Container deployments and CLI invocations override
//!    any file-based settings.
//! 2. **Configuration File (`worker.toml`)**:
//!    - **Unified Schema** (`[worker]`, `[gpu]` sections): The canonical documented format.
//!    - **Flat Legacy Schema**: Root-level fallback for historical configurations.
//! 3. **Derived / Inbuilt Defaults**: Lowest priority. Applied when neither an environment variable
//!    nor a configuration file entry is provided.
//!
//! ### Field Precedence Reference Table
//!
//! | Field | Environment Variable | Config File Key | Default Value | Notes |
//! |---|---|---|---|---|
//! | `api_host` | `TESSELLA_API_HOST` | `api_host` | `"0.0.0.0"` | Bind IP for internal HTTP service |
//! | `api_port` | `TESSELLA_API_PORT` | `api_port` | `8080` (or `listen_port + 1` in legacy) | Service port |
//! | `p2p_host` | `TESSELLA_P2P_HOST` | `p2p_host` | `"0.0.0.0"` | Swarm listen address |
//! | `p2p_port` | `TESSELLA_P2P_PORT` | `p2p_port` / `listen_port` | `9080` | P2P swarm port |
//! | `public_ip` | `TESSELLA_PUBLIC_IP` / `TESSELLA_EXTERNAL_IP` | `public_ip` | `None` | Advertised external IP |
//! | `public_p2p_port` | `TESSELLA_PUBLIC_P2P_PORT` / `TESSELLA_EXTERNAL_PORT` | `public_p2p_port` | `None` | Advertised external P2P port |
//! | `public_api_port` | `TESSELLA_PUBLIC_API_PORT` | `public_api_port` | `None` | Advertised external HTTP port |
//! | `public_api_url` | `TESSELLA_PUBLIC_API_URL` | `public_api_url` | `None` | Takes precedence over IP/port |
//! | `vram_budget_gb` | `TESSELLA_VRAM_BUDGET` / `VRAM_LIMIT` / `TESSELLA_VRAM_MB` | `vram_mb` / `vram_gb` | `8.0` GB | VRAM budget for model allocation |
//! | `orchestrator_url` | `ORCHESTRATOR_URL` / `TRACKER_URL` | `orchestrator_url` / `tracker_url` | `"http://127.0.0.1:8080"` | Control plane tracker endpoint |
//! | `models_dir` | `TESSELLA_MODELS_DIR` | `models_dir` | `"./model_cache"` | Weight and cache storage directory |
//! | `hf_token` | `HF_TOKEN` | `hf_token` | `""` | HuggingFace authentication token |
//! | `keypair_path` | `TESSELLA_KEYPAIR_PATH` | `keypair_path` | `"node_<p2p_port>.key"` | Ed25519 identity key location |
//! | `allow_private_addrs` | `TESSELLA_ALLOW_PRIVATE_ADDRS` | `allow_private_addrs` | `false` | SSRF protection toggle |
//! | `callback_allowed_hosts`| `TESSELLA_CALLBACK_ALLOWED_HOSTS` | `callback_allowed_hosts` | `[]` | Comma-separated allowed host list |
//! | `wal_checkpoint_interval_secs` | `TESSELLA_WAL_CHECKPOINT_INTERVAL_SECS` | `wal_checkpoint_interval_secs` | `300` s | SQLite WAL consolidation cadence |
//! | `kademlia_enabled` | `TESSELLA_KADEMLIA_ENABLED` | `kademlia_enabled` | `true` | DHT participation switch |
//! | `quantization` | `TESSELLA_QUANTIZATION` | `quantization` | `None` (`"None"` / `"Q8_0"`) | Quantization format |
//! | `registration_grace_secs` | `TESSELLA_REGISTRATION_GRACE_SECS` | `registration_grace_secs` | `300` s | Max wait time for tracker boot |
//! | `canary_mode` | `TESSELLA_CANARY_MODE` | `canary_mode` | `Off` (`"off"`, `"echo"`, `"scale"`) | Operator canary replica test mode |
//! | `model_cache_prune` | `TESSELLA_MODEL_CACHE_PRUNE` | `model_cache_prune` | `Off` (`"off"`, `"dry-run"`, `"on"`) | Unused weight shard pruning |
//! | `max_input_tokens` | `TESSELLA_MAX_INPUT_TOKENS` | `max_input_tokens` | `0` (undeclared) | VRAM-measured prefill ceiling |
//! | `p2p_inference_timeout_secs` | `TESSELLA_P2P_INFERENCE_TIMEOUT_SECS` | `p2p_inference_timeout_secs` | `10` s | Hop inference execution timeout |
//! | `oom_streak` | `TESSELLA_OOM_STREAK` / `TESSELLA_GPU_OOM_STREAK` | `gpu.oom_streak` / `worker.oom_streak` | `3` | Consecutive OOM threshold |
//! | `streak_window_secs` | `TESSELLA_STREAK_WINDOW_SECS` / `TESSELLA_GPU_STREAK_WINDOW_SECS` | `gpu.streak_window_secs` / `worker.streak_window_secs` | `30` s | OOM streak observation window |

use crate::config::schema::{CachePruneMode, CanaryMode, WorkerConfig};

/// Applies all environment variable overrides on top of the loaded [`WorkerConfig`].
pub fn apply_env_overrides(cfg: &mut WorkerConfig) -> Result<(), anyhow::Error> {
    if let Ok(val) = std::env::var("TESSELLA_API_HOST") {
        cfg.api_host = val;
    }
    if let Ok(val_str) = std::env::var("TESSELLA_API_PORT") {
        if let Ok(val) = val_str.parse::<u16>() {
            cfg.api_port = val;
        }
    }
    if let Ok(val) = std::env::var("TESSELLA_P2P_HOST") {
        cfg.p2p_host = val;
    }
    if let Ok(val_str) = std::env::var("TESSELLA_P2P_PORT") {
        if let Ok(val) = val_str.parse::<u16>() {
            cfg.p2p_port = val;
            cfg.listen_port = val;
        }
    }
    if let Ok(ext_ip) =
        std::env::var("TESSELLA_PUBLIC_IP").or_else(|_| std::env::var("TESSELLA_EXTERNAL_IP"))
    {
        cfg.public_ip = Some(ext_ip);
    }
    if let Ok(ext_port_str) = std::env::var("TESSELLA_PUBLIC_P2P_PORT")
        .or_else(|_| std::env::var("TESSELLA_EXTERNAL_PORT"))
    {
        if let Ok(port) = ext_port_str.parse::<u16>() {
            cfg.public_p2p_port = Some(port);
        }
    }
    if let Ok(pub_api_port_str) = std::env::var("TESSELLA_PUBLIC_API_PORT") {
        if let Ok(port) = pub_api_port_str.parse::<u16>() {
            cfg.public_api_port = Some(port);
        }
    }
    if let Ok(pub_api_url) = std::env::var("TESSELLA_PUBLIC_API_URL") {
        if !pub_api_url.trim().is_empty() {
            cfg.public_api_url = Some(pub_api_url.trim().to_string());
        }
    }
    if let Ok(vram_str) =
        std::env::var("TESSELLA_VRAM_BUDGET").or_else(|_| std::env::var("VRAM_LIMIT"))
    {
        if let Ok(vram_gb) = vram_str.parse::<f32>() {
            cfg.vram_budget_gb = vram_gb;
        }
    }
    if let Ok(vram_mb_str) = std::env::var("TESSELLA_VRAM_MB") {
        if let Ok(vram_mb) = vram_mb_str.parse::<u32>() {
            cfg.vram_budget_gb = vram_mb as f32 / 1024.0;
        }
    }
    if let Ok(url) = std::env::var("ORCHESTRATOR_URL").or_else(|_| std::env::var("TRACKER_URL")) {
        cfg.orchestrator_url = url;
    }
    if let Ok(models_dir) = std::env::var("TESSELLA_MODELS_DIR") {
        cfg.models_dir = models_dir;
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        cfg.hf_token = token;
    }
    if let Ok(k_path) = std::env::var("TESSELLA_KEYPAIR_PATH") {
        cfg.keypair_path = Some(k_path);
    }
    if let Ok(raw) = std::env::var("TESSELLA_ALLOW_PRIVATE_ADDRS") {
        cfg.allow_private_addrs = parse_bool_env(&raw).ok_or_else(|| {
            anyhow::anyhow!(
                "TESSELLA_ALLOW_PRIVATE_ADDRS must be a boolean (1/true/yes/on or \
                 0/false/no/off), got '{raw}'"
            )
        })?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_CALLBACK_ALLOWED_HOSTS") {
        cfg.callback_allowed_hosts = parse_host_list(&raw);
    }
    if let Ok(raw) = std::env::var("TESSELLA_WAL_CHECKPOINT_INTERVAL_SECS") {
        cfg.wal_checkpoint_interval_secs =
            parse_secs_env("TESSELLA_WAL_CHECKPOINT_INTERVAL_SECS", &raw)?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_KADEMLIA_ENABLED") {
        cfg.kademlia_enabled = parse_bool_env(&raw).ok_or_else(|| {
            anyhow::anyhow!(
                "TESSELLA_KADEMLIA_ENABLED must be a boolean (1/true/yes/on or \
                 0/false/no/off), got '{raw}'"
            )
        })?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_QUANTIZATION") {
        cfg.quantization = parse_quantization_env(&raw)?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_REGISTRATION_GRACE_SECS") {
        cfg.registration_grace_secs = parse_secs_env("TESSELLA_REGISTRATION_GRACE_SECS", &raw)?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_CANARY_MODE") {
        cfg.canary_mode = CanaryMode::parse(&raw)?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_MODEL_CACHE_PRUNE") {
        cfg.model_cache_prune = CachePruneMode::parse(&raw)?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_MAX_INPUT_TOKENS") {
        cfg.max_input_tokens = raw.trim().parse::<usize>().map_err(|_| {
            anyhow::anyhow!(
                "TESSELLA_MAX_INPUT_TOKENS must be a non-negative number of tokens (0 = \
                 undeclared), got '{raw}'"
            )
        })?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_P2P_INFERENCE_TIMEOUT_SECS") {
        cfg.p2p_inference_timeout_secs = raw.trim().parse::<u64>().map_err(|_| {
            anyhow::anyhow!(
                "TESSELLA_P2P_INFERENCE_TIMEOUT_SECS must be a whole number of seconds (the wait \
                 for a hop of this node's pipeline), got '{raw}'"
            )
        })?;
    }
    if let Ok(raw) =
        std::env::var("TESSELLA_OOM_STREAK").or_else(|_| std::env::var("TESSELLA_GPU_OOM_STREAK"))
    {
        cfg.oom_streak = raw.trim().parse::<u32>().map_err(|_| {
            anyhow::anyhow!("TESSELLA_OOM_STREAK must be a positive integer, got '{raw}'")
        })?;
    }
    if let Ok(raw) = std::env::var("TESSELLA_STREAK_WINDOW_SECS")
        .or_else(|_| std::env::var("TESSELLA_GPU_STREAK_WINDOW_SECS"))
    {
        cfg.streak_window_secs = raw.trim().parse::<u64>().map_err(|_| {
            anyhow::anyhow!(
                "TESSELLA_STREAK_WINDOW_SECS must be a whole number of seconds, got '{raw}'"
            )
        })?;
    }

    if let Ok(val) = std::env::var("TESSELLA_ADMIN_TOKEN") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            cfg.admin_token = Some(trimmed.to_string());
        }
    }

    Ok(())
}

/// Parses the boolean spellings accepted by `TESSELLA_ALLOW_PRIVATE_ADDRS` and `TESSELLA_KADEMLIA_ENABLED`.
pub fn parse_bool_env(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Parses the values accepted by `TESSELLA_QUANTIZATION` (`S51`/`TASK-45`).
pub fn parse_quantization_env(
    raw: &str,
) -> Result<tessella_core::models::QuantizationConfig, anyhow::Error> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" | "f16" | "unquantized" => Ok(tessella_core::models::QuantizationConfig::None),
        "q8_0" | "q8" | "int8" => Ok(tessella_core::models::QuantizationConfig::Q8_0),
        "q4k" | "q4_k" => anyhow::bail!(
            "TESSELLA_QUANTIZATION='{raw}': Q4K was retired (`TASK-45`/S51) — it has no loader, so a \
             node that asked for it ran unquantized in silence. Use 'Q8_0' or 'None'."
        ),
        other => anyhow::bail!(
            "TESSELLA_QUANTIZATION='{other}' is not a quantization this node supports \
             (allowed values: 'None', 'Q8_0')"
        ),
    }
}

/// Parses the comma-separated allow-list accepted by `TESSELLA_CALLBACK_ALLOWED_HOSTS`.
pub fn parse_host_list(raw: &str) -> Vec<String> {
    tessella_core::callback::parse_host_list(raw)
}

/// Parses the seconds accepted by `TESSELLA_WAL_CHECKPOINT_INTERVAL_SECS` and `TESSELLA_REGISTRATION_GRACE_SECS`.
pub fn parse_secs_env(name: &str, raw: &str) -> Result<u64, anyhow::Error> {
    raw.trim().parse::<u64>().map_err(|_| {
        anyhow::anyhow!(
            "{name} must be a whole number of seconds (0 disables the WAL maintenance task), \
             got '{raw}'"
        )
    })
}
