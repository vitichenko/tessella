use crate::config::schema::WorkerConfig;

/// Validates and normalises `worker.toml`'s `device`.
///
/// Tessella is **GPU-only** (decision `D-6`, executed in `TASK-21`/S27 and still enforced at boot):
/// the only accepted values are `cuda` and `metal`, compared case-insensitively and with
/// surrounding whitespace trimmed. `cpu` — whatever its casing — aborts the boot.
pub fn validated_device(raw: &str) -> Result<String, anyhow::Error> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cuda" => Ok("cuda".to_string()),
        "metal" => Ok("metal".to_string()),
        "cpu" => anyhow::bail!(
            "CPU device is not allowed in Tessella. \
             Set device = \"cuda\" or device = \"metal\" in worker.toml"
        ),
        other => anyhow::bail!(
            "Unsupported device '{other}' in worker.toml: Tessella is GPU-only \
             (allowed values: \"cuda\", \"metal\")"
        ),
    }
}

/// Whether the file declares a `[worker]` table (`S51`/`TASK-45`).
pub fn has_worker_table(content: &str) -> bool {
    toml::from_str::<toml::Value>(content)
        .ok()
        .and_then(|parsed| parsed.get("worker").map(toml::Value::is_table))
        .unwrap_or(false)
}

/// Names the configuration keys this loader no longer understands, so a removed one is not
/// forgotten in silence.
pub fn removed_keys_in(content: &str) -> Vec<&'static str> {
    let Ok(parsed) = toml::from_str::<toml::Value>(content) else {
        return Vec::new();
    };
    let mentions_table = |name: &str, table: Option<&toml::Value>| {
        table
            .and_then(|table| table.get(name))
            .is_some_and(toml::Value::is_table)
    };
    let in_any_schema = |name: &str| {
        parsed.get(name).is_some() || parsed.get("worker").and_then(|w| w.get(name)).is_some()
    };

    let mut removed = Vec::new();
    if mentions_table("batching", Some(&parsed)) || mentions_table("batching", parsed.get("worker"))
    {
        removed.push("[batching]");
    }
    if in_any_schema("seed") {
        removed.push("seed");
    }
    if in_any_schema("temperature") {
        removed.push("temperature");
    }
    if in_any_schema("kv_cache_max_entries") {
        removed.push("kv_cache_max_entries");
    }
    if in_any_schema("kv_cache_ttl_secs") {
        removed.push("kv_cache_ttl_secs");
    }
    if in_any_schema("circuit_breaker_failure_threshold") {
        removed.push("circuit_breaker_failure_threshold");
    }
    if in_any_schema("circuit_breaker_cooldown_secs") {
        removed.push("circuit_breaker_cooldown_secs");
    }
    removed
}

/// What replaces a retired key, so the warning says the truth **per key**.
pub fn retired_key_reason(key: &str) -> &'static str {
    match key {
        "[batching]" | "seed" | "temperature" => {
            "this node does not batch (the scheduler applies its sessions one by one) and the \
             sampling parameters are the request's, with the mesh standard 0.2/0.9/40 as default \
             (S45/TASK-39, S46/TASK-40)"
        }
        "kv_cache_max_entries" | "kv_cache_ttl_secs" => {
            "the shared KV cache store was retired: the KV cache that exists is per session and \
             lives inside the backend (S52/TASK-46)"
        }
        "circuit_breaker_failure_threshold" | "circuit_breaker_cooldown_secs" => {
            "the breaker they tuned was never consulted by anything: the failed hops this node sees \
             are counted and published by `GET /status` (`peer_failures`) instead (S56/TASK-50)"
        }
        _ => "see the changelog of this node",
    }
}

/// Validates all system invariants on a loaded and environment-overridden [`WorkerConfig`].
pub fn validate_invariants(cfg: &mut WorkerConfig) -> Result<(), anyhow::Error> {
    // S92 (TASK-85): saturation thresholds cannot be 0.
    if cfg.oom_streak == 0 {
        anyhow::bail!("oom_streak (or gpu.oom_streak) cannot be 0 (must be >= 1)");
    }
    if cfg.streak_window_secs == 0 {
        anyhow::bail!("streak_window_secs (or gpu.streak_window_secs) cannot be 0 (must be >= 1)");
    }

    // S68: a boot that may hang for longer than an hour is a misconfiguration.
    if cfg.registration_grace_secs > crate::registration::MAX_REGISTRATION_GRACE_SECS {
        anyhow::bail!(
            "registration_grace_secs = {} is above the maximum ({} s): a node that waits \
             longer than an hour for the tracker at boot is misconfigured, not tolerant \
             (`0` disables the retry)",
            cfg.registration_grace_secs,
            crate::registration::MAX_REGISTRATION_GRACE_SECS
        );
    }

    // Normalise the allow-list once, whichever schema or env var provided it.
    cfg.callback_allowed_hosts = cfg
        .callback_allowed_hosts
        .iter()
        .map(|host| host.trim().trim_matches('.').to_string())
        .filter(|host| !host.is_empty())
        .collect();

    // GPU-only enforcement (D-6): validates *and* normalises the device.
    cfg.device = validated_device(&cfg.device)?;

    Ok(())
}
