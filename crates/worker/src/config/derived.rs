use crate::config::schema::WorkerConfig;
use crate::config::validate::validated_device;

pub(crate) fn default_device() -> String {
    "cuda".to_string()
}

pub(crate) fn default_oom_streak() -> u32 {
    tessella_core::telemetry::gpu_health::DEFAULT_OOM_STREAK
}

pub(crate) fn default_streak_window_secs() -> u64 {
    tessella_core::telemetry::gpu_health::DEFAULT_STREAK_WINDOW_SECS
}

pub(crate) fn default_wal_checkpoint_interval_secs() -> u64 {
    tessella_storage::WAL_CHECKPOINT_INTERVAL_SECS
}

pub(crate) fn default_registration_grace_secs() -> u64 {
    crate::registration::DEFAULT_REGISTRATION_GRACE_SECS
}

pub(crate) fn default_p2p_inference_timeout_secs() -> u64 {
    10
}

pub(crate) fn default_api_host() -> String {
    "0.0.0.0".to_string()
}

pub(crate) fn default_api_port() -> u16 {
    8080
}

pub(crate) fn default_p2p_host() -> String {
    "0.0.0.0".to_string()
}

pub(crate) fn default_p2p_port() -> u16 {
    9080
}

pub(crate) fn default_true() -> bool {
    true
}

pub(crate) fn quantization_name(
    quantization: tessella_core::models::QuantizationConfig,
) -> &'static str {
    match quantization {
        tessella_core::models::QuantizationConfig::None => "None",
        tessella_core::models::QuantizationConfig::Q8_0 => "Q8_0",
    }
}

/// Length of the accepted `orchestrator_public_key` value: the **raw** Ed25519 public key
/// (32 bytes) in hex, which is what `/p2p_id` publishes as `orchestrator_pubkey` (`TASK-19`).
pub const ORCHESTRATOR_PUBLIC_KEY_HEX_LEN: usize = 64;

/// Decodes an operator-provided `orchestrator_public_key`.
pub fn parse_orchestrator_public_key(raw: &str) -> Result<ed25519_dalek::VerifyingKey, String> {
    let trimmed = raw.trim();
    let bytes = hex::decode(trimmed).map_err(|e| {
        format!(
            "orchestrator_public_key is not hexadecimal ({e}): expected \
             {ORCHESTRATOR_PUBLIC_KEY_HEX_LEN} hex chars, the raw Ed25519 public key that GET \
             /p2p_id publishes as 'orchestrator_pubkey'"
        )
    })?;
    let key_bytes: [u8; 32] = bytes.try_into().map_err(|bytes: Vec<u8>| {
        format!(
            "orchestrator_public_key must be {ORCHESTRATOR_PUBLIC_KEY_HEX_LEN} hex chars (32 raw \
             Ed25519 bytes), got {} chars",
            bytes.len() * 2
        )
    })?;
    ed25519_dalek::VerifyingKey::from_bytes(&key_bytes).map_err(|e| {
        format!(
            "orchestrator_public_key is {ORCHESTRATOR_PUBLIC_KEY_HEX_LEN} hex chars but not a \
             valid Ed25519 public key ({e}): copy the 'orchestrator_pubkey' field of GET /p2p_id"
        )
    })
}

impl WorkerConfig {
    /// Returns the configured compute device, or an error when the configuration does not
    /// select a supported GPU backend.
    pub fn get_candle_device(&self) -> Result<tessella_core::models::ComputeDevice, anyhow::Error> {
        match validated_device(&self.device)?.as_str() {
            "cuda" => Ok(tessella_core::models::ComputeDevice::Cuda { device_id: 0 }),
            "metal" => Ok(tessella_core::models::ComputeDevice::Metal),
            other => anyhow::bail!("Unsupported device '{other}' in worker.toml"),
        }
    }

    /// The wait **this node** declares for a hop of its own pipeline (`TASK-76`/S83).
    pub fn hop_wait(&self) -> Result<std::time::Duration, anyhow::Error> {
        if self.p2p_inference_timeout_secs == 0 {
            anyhow::bail!(
                "`p2p_inference_timeout_secs = 0` in worker.toml: a hop that may take no time at all \
                 would lose every request — and `0` does not mean *wait forever* either, because a \
                 mute node has to be declared dead. Declare the seconds a hop of this node may take \
                 (default: 10)"
            );
        }
        Ok(std::time::Duration::from_secs(
            self.p2p_inference_timeout_secs,
        ))
    }

    /// Resolves the HTTP API URL that this worker announces to the swarm.
    ///
    /// Precedence (RunPod/NAT friendly):
    /// 1. `public_api_url` — a full URL (scheme + host + optional port/path), used verbatim.
    /// 2. `public_ip` (+ optional `public_api_port`) — classic `ip:port` form.
    /// 3. `bind_ip` + `api_port` — local/LAN fallback.
    #[must_use]
    pub fn resolve_http_api_url(&self, bind_ip: &str) -> String {
        if let Some(url) = self
            .public_api_url
            .as_deref()
            .map(str::trim)
            .filter(|u| !u.is_empty())
        {
            return url.trim_end_matches('/').to_string();
        }
        match (&self.public_ip, self.public_api_port) {
            (Some(ip), Some(port)) => format!("http://{}:{}", ip, port),
            (Some(ip), None) => format!("http://{}:{}", ip, self.api_port),
            (None, _) => format!("http://{}:{}", bind_ip, self.api_port),
        }
    }

    /// The pinned orchestrator key, decoded once (`TASK-19`).
    pub fn orchestrator_verifying_key(
        &self,
    ) -> Result<Option<ed25519_dalek::VerifyingKey>, anyhow::Error> {
        let Some(raw) = self
            .orchestrator_public_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        parse_orchestrator_public_key(raw)
            .map(Some)
            .map_err(anyhow::Error::msg)
    }

    /// Returns the resolved keypair path.
    #[must_use]
    pub fn get_keypair_path(&self) -> String {
        self.keypair_path
            .clone()
            .unwrap_or_else(|| format!("node_{}.key", self.listen_port))
    }
}
