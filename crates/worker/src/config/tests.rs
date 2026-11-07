use super::*;
use tessella_core::models::ComputeDevice;

/// Writes a temporary `worker.toml` and returns its path.
pub(crate) fn write_tmp_worker_toml(name: &str, body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tessella-worker-config-{name}-{}.toml",
        std::process::id()
    ));
    std::fs::write(&path, body).expect("write the temporary worker.toml");
    path
}

/// Minimal legacy-schema header (the schema used by `config/swarm/worker_*.toml`).
pub(crate) const LEGACY_PREFIX: &str =
    "tracker_url = \"http://127.0.0.1:8080\"\nlisten_port = 9010\n";

#[test]
fn device_accepts_cuda_and_metal_case_insensitively() {
    for (raw, expected) in [
        ("cuda", ComputeDevice::Cuda { device_id: 0 }),
        ("CUDA", ComputeDevice::Cuda { device_id: 0 }),
        ("metal", ComputeDevice::Metal),
        ("  Metal  ", ComputeDevice::Metal),
    ] {
        let path = write_tmp_worker_toml("ok", &format!("{LEGACY_PREFIX}device = \"{raw}\"\n"));
        let cfg = WorkerConfig::load(&path)
            .unwrap_or_else(|e| panic!("device={raw:?} must be accepted, got: {e}"));
        assert_eq!(
            cfg.get_candle_device().expect("device"),
            expected,
            "device={raw:?}"
        );
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn device_rejects_cpu_whatever_the_casing_or_padding() {
    for raw in ["cpu", "CPU", "Cpu", " cpu ", "cpu "] {
        let path = write_tmp_worker_toml("cpu", &format!("{LEGACY_PREFIX}device = \"{raw}\"\n"));
        let err = WorkerConfig::load(&path).expect_err("a CPU device must abort the boot");
        assert!(
            err.to_string().contains("CPU device is not allowed"),
            "device={raw:?} produced an unusable message: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn device_rejects_unknown_values_instead_of_silently_using_the_cpu() {
    // Regression guard: `get_candle_device()` mapped every unrecognised value to
    // `ComputeDevice::Cpu`, so a typo ("cuad") or another backend ("gpu") ran on CPU
    // even though the boot check was supposed to make Tessella GPU-only.
    for raw in ["cuad", "gpu", "tpu", "", "cuda0", "nvidia"] {
        let path =
            write_tmp_worker_toml("unknown", &format!("{LEGACY_PREFIX}device = \"{raw}\"\n"));
        let err = WorkerConfig::load(&path).expect_err("an unknown device must abort the boot");
        assert!(
            err.to_string().contains("Unsupported device"),
            "device={raw:?} produced an unusable message: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn get_candle_device_never_falls_back_to_the_cpu() {
    let cpu = WorkerConfig {
        device: "cpu".to_string(),
        ..Default::default()
    };
    let err = cpu
        .get_candle_device()
        .expect_err("cpu must not be selectable");
    assert!(err.to_string().contains("CPU device is not allowed"));

    let typo = WorkerConfig {
        device: "CUAD".to_string(),
        ..Default::default()
    };
    let err = typo
        .get_candle_device()
        .expect_err("an unknown device must not become cpu");
    assert!(err.to_string().contains("Unsupported device"));

    let default = WorkerConfig::default();
    assert_eq!(
        default.get_candle_device().expect("default device"),
        ComputeDevice::Cuda { device_id: 0 }
    );
}

#[test]
fn unified_config_honours_and_validates_the_worker_device() {
    let body = r#"orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "node.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 8192
models_dir = "./model_cache"
api_port = 10001
p2p_port = 9001
device = "metal"
"#;
    // The unified schema used to drop `device` in silence (always the `cuda` default).
    let path = write_tmp_worker_toml("unified-metal", body);
    let cfg = WorkerConfig::load(&path).expect("unified worker config");
    assert_eq!(
        cfg.get_candle_device().expect("device"),
        ComputeDevice::Metal
    );
    let _ = std::fs::remove_file(&path);

    // ...and a CPU requested there is rejected instead of being ignored.
    let path = write_tmp_worker_toml("unified-cpu", &body.replace("metal", "cpu"));
    let err = WorkerConfig::load(&path).expect_err("unified device=cpu must abort the boot");
    assert!(err.to_string().contains("CPU device is not allowed"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn saved_config_round_trips_the_device_and_the_callback_guard_keys() {
    let cfg = WorkerConfig {
        device: "metal".to_string(),
        allow_private_addrs: true,
        callback_allowed_hosts: vec!["proxy.runpod.net".to_string()],
        ..Default::default()
    };
    let path =
        std::env::temp_dir().join(format!("tessella-worker-save-{}.toml", std::process::id()));
    cfg.save(&path).expect("save the config");
    let reloaded = WorkerConfig::load(&path).expect("reload the saved config");
    assert_eq!(
        reloaded.get_candle_device().expect("device"),
        ComputeDevice::Metal
    );
    assert!(reloaded.allow_private_addrs);
    assert_eq!(
        reloaded.callback_allowed_hosts,
        vec!["proxy.runpod.net".to_string()]
    );
    let _ = std::fs::remove_file(&path);
}

/// `S51`/`TASK-45` (VR-01): `quantization` used to exist only in the flat legacy schema, so writing
/// it in the documented unified `[worker]` table was ignored in silence — and the planner could not
/// see it either.
#[test]
fn the_unified_schema_honours_the_quantization() {
    let body = r#"orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "node.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 8192
models_dir = "./model_cache"
api_host = "0.0.0.0"
api_port = 8080
p2p_host = "0.0.0.0"
p2p_port = 9080
device = "cuda"
quantization = "Q8_0"
"#;
    let path = write_tmp_worker_toml("quantization-unified", body);
    let cfg = WorkerConfig::load(&path).expect("the unified schema must load");
    assert_eq!(
        cfg.quantization,
        tessella_core::models::QuantizationConfig::Q8_0,
        "the key of the documented schema must reach the node, not be dropped"
    );
    let _ = std::fs::remove_file(&path);

    // The legacy flat schema keeps working, and `save()` writes the key back when it is not the
    // default (so a round-trip cannot lose it).
    let cfg = WorkerConfig {
        quantization: tessella_core::models::QuantizationConfig::Q8_0,
        ..Default::default()
    };
    let path = std::env::temp_dir().join(format!(
        "tessella-worker-quantization-{}.toml",
        std::process::id()
    ));
    cfg.save(&path).expect("save the config");
    assert!(
        std::fs::read_to_string(&path)
            .expect("read the saved config")
            .contains("quantization = \"Q8_0\""),
        "save() must write the quantization when it differs from the default"
    );
    let reloaded = WorkerConfig::load(&path).expect("reload the saved config");
    assert_eq!(reloaded.quantization, cfg.quantization);
    let _ = std::fs::remove_file(&path);
}

/// `S51`/`TASK-45`: `Q4K` was a phantom option (no loader, no consumer, no measurement), so it was
/// retired from `QuantizationConfig`. What must **not** happen is the old behaviour: a node that
/// asked for it silently running unquantized.
#[test]
fn q4k_is_not_a_quantization_option_any_more() {
    let body = r#"orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "node.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 8192
models_dir = "./model_cache"
api_host = "0.0.0.0"
api_port = 8080
p2p_host = "0.0.0.0"
p2p_port = 9080
device = "cuda"
quantization = "Q4K"
"#;
    let path = write_tmp_worker_toml("quantization-q4k", body);
    let err = WorkerConfig::load(&path).expect_err("Q4K must not be accepted");
    assert!(
        err.to_string().contains("Q4K"),
        "the error must name the retired value: {err}"
    );
    let _ = std::fs::remove_file(&path);

    let err = parse_quantization_env("Q4K").expect_err("Q4K must not be accepted by env either");
    assert!(err.to_string().contains("retired"), "{err}");
    assert_eq!(
        parse_quantization_env("q8_0").expect("q8_0 is accepted"),
        tessella_core::models::QuantizationConfig::Q8_0
    );
    assert_eq!(
        parse_quantization_env("none").expect("none is accepted"),
        tessella_core::models::QuantizationConfig::None
    );
    assert!(parse_quantization_env("qwhatever").is_err());

    // The retired KV-cache keys are named as well, not ignored in silence (`S52`/`TASK-46`).
    let removed =
        removed_keys_in("[worker]\nkv_cache_max_entries = 256\nkv_cache_ttl_secs = 300\n");
    assert!(removed.contains(&"kv_cache_max_entries"), "{removed:?}");
    assert!(removed.contains(&"kv_cache_ttl_secs"), "{removed:?}");
}

/// H-21: the WAL maintenance cadence is an ordinary key of both schemas, and its default is the
/// constant of `tessella-storage` — **not** `0`, which would silently disable the task.
#[test]
fn wal_checkpoint_interval_defaults_to_the_storage_cadence() {
    let cfg = WorkerConfig::default();
    assert_eq!(
        cfg.wal_checkpoint_interval_secs,
        tessella_storage::WAL_CHECKPOINT_INTERVAL_SECS
    );
    assert_eq!(
        cfg.wal_checkpoint_interval_secs, 300,
        "the documented default cadence"
    );
}

#[test]
fn the_configs_honour_the_wal_checkpoint_interval() {
    let body = r#"orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "node.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 8192
models_dir = "./model_cache"
api_port = 10001
p2p_port = 9001
wal_checkpoint_interval_secs = 45
"#;
    let path = write_tmp_worker_toml("unified-wal", body);
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("unified worker config")
            .wal_checkpoint_interval_secs,
        45,
        "the unified schema must not drop the key in silence"
    );
    let _ = std::fs::remove_file(&path);

    // The legacy schema (the one `run_tests.sh` writes) accepts it too, including 0 = disabled.
    let legacy = write_tmp_worker_toml(
        "legacy-wal",
        &format!("{LEGACY_PREFIX}wal_checkpoint_interval_secs = 0\n"),
    );
    assert_eq!(
        WorkerConfig::load(&legacy)
            .expect("legacy worker config")
            .wal_checkpoint_interval_secs,
        0
    );
    let _ = std::fs::remove_file(&legacy);
}

#[test]
fn saved_config_round_trips_the_wal_checkpoint_interval() {
    let path = std::env::temp_dir().join(format!(
        "tessella-worker-save-wal-{}.toml",
        std::process::id()
    ));
    let cfg = WorkerConfig {
        wal_checkpoint_interval_secs: 45,
        ..Default::default()
    };
    cfg.save(&path).expect("save the config");
    let body = std::fs::read_to_string(&path).expect("read the saved config");
    assert!(
        body.contains("wal_checkpoint_interval_secs = 45"),
        "save() must write a cadence that differs from the default:\n{body}"
    );
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("reload the saved config")
            .wal_checkpoint_interval_secs,
        45
    );

    // The default cadence is not written (the file stays as small as it was before S15).
    let default_path = std::env::temp_dir().join(format!(
        "tessella-worker-save-wal-default-{}.toml",
        std::process::id()
    ));
    WorkerConfig::default()
        .save(&default_path)
        .expect("save the default config");
    let body = std::fs::read_to_string(&default_path).expect("read the default config");
    assert!(
        !body.contains("wal_checkpoint_interval_secs"),
        "the default cadence must not be written:\n{body}"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&default_path);
}

/// The env override is parsed by this helper, so its accepted spellings and its error message are
/// pinned here (reading the process environment in a test would race with the rest of the suite).
#[test]
fn wal_checkpoint_env_parsing_accepts_seconds_and_rejects_typos() {
    const VAR: &str = "TESSELLA_WAL_CHECKPOINT_INTERVAL_SECS";
    assert_eq!(parse_secs_env(VAR, "45").unwrap(), 45);
    assert_eq!(parse_secs_env(VAR, " 45 ").unwrap(), 45);
    assert_eq!(
        parse_secs_env(VAR, "0").unwrap(),
        0,
        "0 is documented as \"disabled\", not as an error"
    );
    for raw in ["5m", "-1", "1.5", ""] {
        let err = parse_secs_env(VAR, raw).expect_err("a typo must abort the boot");
        assert!(
            err.to_string().contains("whole number of seconds"),
            "the message must name the problem for raw={raw:?}: {err}"
        );
    }
}

/// H-10: both knobs must be conservative by default — the worker accepts token-stream
/// callbacks only to hosts it knows, and never to a private address without this opt-in.
/// S68: the boot retry of `TASK-62` has a default that is the policy constant — not `0`, which
/// would silently restore the pre-S68 boot (a node that dies the first time the tracker is down).
#[test]
fn the_registration_grace_defaults_to_the_policy_constant() {
    let cfg = WorkerConfig::default();
    assert_eq!(
        cfg.registration_grace_secs,
        crate::registration::DEFAULT_REGISTRATION_GRACE_SECS
    );
    assert_eq!(cfg.registration_grace_secs, 300, "the documented budget");

    // A config that does not mention the key (the gate of §1bis writes one of these on every
    // run) must get the default, not zero.
    let path = write_tmp_worker_toml("grace-absent", LEGACY_PREFIX);
    let loaded = WorkerConfig::load(&path).expect("legacy worker config");
    assert_eq!(
        loaded.registration_grace_secs,
        crate::registration::DEFAULT_REGISTRATION_GRACE_SECS
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_configs_honour_the_registration_grace() {
    let body = r#"orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "node.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 8192
models_dir = "./model_cache"
api_port = 10001
p2p_port = 9001
registration_grace_secs = 45
"#;
    let path = write_tmp_worker_toml("unified-grace", body);
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("unified worker config")
            .registration_grace_secs,
        45,
        "the unified schema must not drop the key in silence"
    );
    let _ = std::fs::remove_file(&path);

    // The legacy schema (the one `run_tests.sh` writes) accepts it too, including 0 = no retry.
    let legacy = write_tmp_worker_toml(
        "legacy-grace",
        &format!("{LEGACY_PREFIX}registration_grace_secs = 0\n"),
    );
    let cfg = WorkerConfig::load(&legacy).expect("legacy worker config");
    assert_eq!(cfg.registration_grace_secs, 0);
    assert!(
        crate::registration::RegistrationGrace::from_secs(cfg.registration_grace_secs)
            .is_disabled()
    );
    let _ = std::fs::remove_file(&legacy);
}

/// A boot that may hang for hours is a misconfiguration: the key is refused **naming it**, so the
/// operator reads the reason instead of watching a node that never comes up.
#[test]
fn an_absurd_registration_grace_is_refused() {
    let too_long = crate::registration::MAX_REGISTRATION_GRACE_SECS + 1;
    let path = write_tmp_worker_toml(
        "grace-absurd",
        &format!("{LEGACY_PREFIX}registration_grace_secs = {too_long}\n"),
    );
    let err = WorkerConfig::load(&path).expect_err("a budget above the maximum must abort");
    let message = err.to_string();
    assert!(message.contains("registration_grace_secs"), "{message}");
    assert!(message.contains("misconfigured"), "{message}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn saved_config_round_trips_the_registration_grace() {
    let path = std::env::temp_dir().join(format!(
        "tessella-worker-save-grace-{}.toml",
        std::process::id()
    ));
    let cfg = WorkerConfig {
        registration_grace_secs: 45,
        ..Default::default()
    };
    cfg.save(&path).expect("save the config");
    let body = std::fs::read_to_string(&path).expect("read the saved config");
    assert!(
        body.contains("registration_grace_secs = 45"),
        "save() must write a budget that differs from the default:\n{body}"
    );
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("reload the saved config")
            .registration_grace_secs,
        45
    );

    // The default budget is not written (the file stays as small as it was before S68).
    let default_path = std::env::temp_dir().join(format!(
        "tessella-worker-save-grace-default-{}.toml",
        std::process::id()
    ));
    WorkerConfig::default()
        .save(&default_path)
        .expect("save the default config");
    let body = std::fs::read_to_string(&default_path).expect("read the default config");
    assert!(
        !body.contains("registration_grace_secs"),
        "the default budget must not be written:\n{body}"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&default_path);
}

#[test]
fn callback_guard_defaults_are_conservative() {
    let cfg = WorkerConfig::default();
    assert!(
        !cfg.allow_private_addrs,
        "private callback targets must be opted in"
    );
    assert!(cfg.callback_allowed_hosts.is_empty());
}

/// The unified schema must honour the two keys instead of dropping them in silence
/// (the same defect `device` had until S27), and the allow-list is trimmed once.
#[test]
fn unified_config_honours_the_callback_guard_keys() {
    let body = r#"orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "node.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 8192
models_dir = "./model_cache"
api_port = 10001
p2p_port = 9001
allow_private_addrs = true
callback_allowed_hosts = ["proxy.runpod.net", " gw.example.com ", ""]
"#;
    let path = write_tmp_worker_toml("unified-callback", body);
    let cfg = WorkerConfig::load(&path).expect("unified worker config");
    assert!(cfg.allow_private_addrs);
    assert_eq!(
        cfg.callback_allowed_hosts,
        vec!["proxy.runpod.net".to_string(), "gw.example.com".to_string()]
    );
    let _ = std::fs::remove_file(&path);
}

/// The env overrides are parsed by these helpers, so the accepted spellings are pinned here
/// (reading the process environment in a test would race with the rest of the suite).
#[test]
fn callback_guard_env_parsing_accepts_the_documented_spellings() {
    for raw in ["1", "true", "TRUE", " yes ", "on"] {
        assert_eq!(parse_bool_env(raw), Some(true), "raw={raw:?}");
    }
    for raw in ["0", "false", "FALSE", " no ", "off"] {
        assert_eq!(parse_bool_env(raw), Some(false), "raw={raw:?}");
    }
    assert_eq!(parse_bool_env("maybe"), None);

    assert_eq!(
        parse_host_list(" proxy.runpod.net , gw.example.com ,, "),
        vec!["proxy.runpod.net".to_string(), "gw.example.com".to_string()]
    );
    assert!(parse_host_list("   ").is_empty());
}
