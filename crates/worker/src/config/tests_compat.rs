use super::tests::{write_tmp_worker_toml, LEGACY_PREFIX};
use super::*;

#[test]
fn resolve_http_api_url_prefers_full_url() {
    let cfg = WorkerConfig {
        public_api_url: Some("https://pod-8080.proxy.runpod.net/".to_string()),
        public_ip: Some("203.0.113.5".to_string()),
        public_api_port: Some(54325),
        ..Default::default()
    };
    assert_eq!(
        cfg.resolve_http_api_url("10.0.0.2"),
        "https://pod-8080.proxy.runpod.net"
    );
}

#[test]
fn resolve_http_api_url_uses_public_ip_and_port() {
    let cfg = WorkerConfig {
        public_ip: Some("203.0.113.5".to_string()),
        public_api_port: Some(54325),
        ..Default::default()
    };
    assert_eq!(
        cfg.resolve_http_api_url("10.0.0.2"),
        "http://203.0.113.5:54325"
    );
}

#[test]
fn resolve_http_api_url_public_ip_without_port_uses_api_port() {
    let cfg = WorkerConfig {
        public_ip: Some("203.0.113.5".to_string()),
        api_port: 8080,
        ..Default::default()
    };
    assert_eq!(
        cfg.resolve_http_api_url("10.0.0.2"),
        "http://203.0.113.5:8080"
    );
}

#[test]
fn resolve_http_api_url_falls_back_to_bind_ip() {
    let cfg = WorkerConfig {
        api_port: 8080,
        ..Default::default()
    };
    assert_eq!(cfg.resolve_http_api_url("10.0.0.2"), "http://10.0.0.2:8080");
}

#[test]
fn resolve_http_api_url_ignores_blank_public_url() {
    let cfg = WorkerConfig {
        public_api_url: Some("   ".to_string()),
        public_ip: Some("203.0.113.5".to_string()),
        ..Default::default()
    };
    assert_eq!(
        cfg.resolve_http_api_url("10.0.0.2"),
        "http://203.0.113.5:8080"
    );
}
/// TASK-19 · criterio 2: `orchestrator_public_key` acepta exactamente lo que `/p2p_id` publica
/// (64 hex = 32 bytes crudos ed25519) y **explica** por qué rechaza lo demás, en vez de degradar
/// todo el tráfico a `Leech` sin decir nada (los `.ok()` encadenados de `service.rs`).
#[test]
fn orchestrator_verifying_key_accepts_the_published_format() {
    let keypair = tessella_network::generate_keypair();
    let published = tessella_network::ed25519_public_key_hex(&keypair).unwrap();
    let expected = ed25519_dalek::VerifyingKey::from_bytes(
        &tessella_network::ed25519_public_key_bytes(&keypair).unwrap(),
    )
    .unwrap();

    let cfg = WorkerConfig {
        orchestrator_public_key: Some(published.clone()),
        ..Default::default()
    };
    assert_eq!(
        cfg.orchestrator_verifying_key().unwrap(),
        Some(expected),
        "the value /p2p_id publishes must pin the tickets the orchestrator signs"
    );

    // Un valor pegado con mayúsculas o con espacios es ruido de copia-pega, no un error.
    let messy = WorkerConfig {
        orchestrator_public_key: Some(format!("  {}  ", published.to_uppercase())),
        ..Default::default()
    };
    assert_eq!(messy.orchestrator_verifying_key().unwrap(), Some(expected));

    // Sin clave no hay pinning, y eso no es un error de configuración.
    assert_eq!(
        WorkerConfig::default()
            .orchestrator_verifying_key()
            .unwrap(),
        None
    );
    let blank = WorkerConfig {
        orchestrator_public_key: Some("   ".into()),
        ..Default::default()
    };
    assert_eq!(blank.orchestrator_verifying_key().unwrap(), None);
}

/// Endurecimiento acordado en S25: un valor ilegible deja de ser silencioso. El mensaje nombra
/// el campo y el formato aceptado, porque el síntoma (todo el tráfico en `Leech`) no lo hace.
#[test]
fn an_unusable_orchestrator_public_key_reports_why() {
    for (label, value) in [
        ("protobuf (72 hex) instead of the raw key", "0".repeat(72)),
        ("not hex at all", "zz".repeat(32)),
        ("truncated", "a".repeat(62)),
        ("odd number of hex chars", "a".repeat(63)),
    ] {
        let cfg = WorkerConfig {
            orchestrator_public_key: Some(value),
            ..Default::default()
        };
        let err = cfg
            .orchestrator_verifying_key()
            .expect_err(label)
            .to_string();
        assert!(
            err.contains("orchestrator_public_key") && err.contains("64 hex"),
            "'{label}' must name the field and the expected format: {err}"
        );
    }
}

/// `TASK-39`/S45: la tabla `[batching]` ya no existe (el worker nunca batcheaba y el único
/// camino que podía hacerlo no tenía productor). `WorkerConfig` no usa `deny_unknown_fields`
/// —el esquema unificado y el legacy comparten `[worker]` y se parsean a mano—, así que sin
/// este aviso un operador seguiría creyendo que su nodo batchea. Es una función pura sobre el
/// fichero crudo: se prueba sin capturar logs.
#[test]
fn a_removed_batching_table_is_reported_in_both_schemas() {
    // Esquema legacy: el que escriben el harness y los `config/swarm/worker_*.toml`.
    assert_eq!(
        removed_keys_in("[batching]\nenabled = true\nmax_batch_size = 8\n"),
        vec!["[batching]"],
        "the legacy schema's `[batching]` table must be named"
    );
    // Esquema unificado, en su forma anidada…
    assert_eq!(
        removed_keys_in("[worker.batching]\nenabled = true\n"),
        vec!["[batching]"]
    );
    // …y en su forma en línea.
    assert_eq!(
        removed_keys_in(
            "[worker]\nvram_mb = 1024\nbatching = { enabled = true, max_wait_ms = 20 }\n"
        ),
        vec!["[batching]"]
    );
    // Un fichero que no menciona la clave no dice nada.
    assert!(removed_keys_in("listen_port = 9010\nvram_gb = 2.0\ndevice = \"cuda\"\n").is_empty());
}

/// `TASK-40`/S46: `seed` y `temperature` ya no son claves del worker (los parámetros de una
/// petición son del cliente y el estándar del mesh es una constante compartida con el contrato
/// de parámetros). Un TOML que las conserve —el harness escribía las dos— se nombra.
#[test]
fn the_retired_sampling_keys_are_reported_in_both_schemas() {
    assert_eq!(
        removed_keys_in("seed = 42\ntemperature = 0.2\n"),
        vec!["seed", "temperature"],
        "the legacy schema's sampling keys must be named"
    );
    assert_eq!(
        removed_keys_in("[worker]\nvram_mb = 1024\ntemperature = 0.7\n"),
        vec!["temperature"]
    );
    assert_eq!(removed_keys_in("[worker]\nseed = 1\n"), vec!["seed"]);
    assert!(removed_keys_in("max_new_tokens = 256\n").is_empty());
}

/// `TASK-50`/S56: las dos claves que ajustaban el `CircuitBreaker` retirado. Estaban en el
/// esquema legacy plano y solo gobernaban un estado que **nadie leía**, así que se nombran (con
/// lo que las sustituye) en vez de ignorarse.
#[test]
fn the_retired_breaker_keys_are_reported_with_what_replaces_them() {
    assert_eq!(
        removed_keys_in(
            "circuit_breaker_failure_threshold = 3\ncircuit_breaker_cooldown_secs = 30\n"
        ),
        vec![
            "circuit_breaker_failure_threshold",
            "circuit_breaker_cooldown_secs"
        ]
    );
    assert_eq!(
        removed_keys_in("[worker]\ncircuit_breaker_cooldown_secs = 30\n"),
        vec!["circuit_breaker_cooldown_secs"]
    );
    assert!(
        removed_keys_in("max_new_tokens = 256\n").is_empty(),
        "una clave viva no se reporta"
    );

    // El motivo es el de la clave, no una frase genérica (que mentiría para las demás familias).
    assert!(
        retired_key_reason("circuit_breaker_failure_threshold").contains("peer_failures"),
        "{}",
        retired_key_reason("circuit_breaker_failure_threshold")
    );
    assert!(retired_key_reason("kv_cache_ttl_secs").contains("KV cache"));
    assert!(retired_key_reason("temperature").contains("0.2/0.9/40"));
    assert!(retired_key_reason("[batching]").contains("does not batch"));
}

/// S80: the prune is opt-in, so **both** schemas must carry the key. A unified `worker.toml` that
/// dropped it in silence would leave an operator who asked for the disk back with a cache that is
/// never pruned (or, worse, a node pruning a shared cache with a mode nobody chose).
#[test]
fn the_configs_honour_the_model_cache_prune() {
    let body = r#"orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "node.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 8192
models_dir = "./model_cache"
api_port = 10001
p2p_port = 9001
model_cache_prune = "dry-run"
"#;
    let path = write_tmp_worker_toml("unified-prune", body);
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("unified worker config")
            .model_cache_prune,
        CachePruneMode::DryRun,
        "the unified schema must not drop the key in silence"
    );
    let _ = std::fs::remove_file(&path);

    // The legacy schema (the one `run_tests.sh` writes) accepts it too.
    let legacy = write_tmp_worker_toml(
        "legacy-prune",
        &format!("{LEGACY_PREFIX}model_cache_prune = \"on\"\n"),
    );
    assert_eq!(
        WorkerConfig::load(&legacy)
            .expect("legacy worker config")
            .model_cache_prune,
        CachePruneMode::On
    );
    let _ = std::fs::remove_file(&legacy);

    // Absent means `off`: no schema turns the prune on in silence.
    let absent = write_tmp_worker_toml("prune-absent", LEGACY_PREFIX);
    assert_eq!(
        WorkerConfig::load(&absent)
            .expect("legacy worker config")
            .model_cache_prune,
        CachePruneMode::Off,
        "the cache must be untouched unless the operator asks for the prune"
    );
    let _ = std::fs::remove_file(&absent);
}

/// S80: a round-trip may not turn the prune on, and it may not forget it either.
#[test]
fn saved_config_round_trips_the_model_cache_prune() {
    let path = std::env::temp_dir().join(format!(
        "tessella-worker-save-prune-{}.toml",
        std::process::id()
    ));
    let cfg = WorkerConfig {
        model_cache_prune: CachePruneMode::On,
        ..Default::default()
    };
    cfg.save(&path).expect("save the config");
    let body = std::fs::read_to_string(&path).expect("read the saved config");
    assert!(
        body.contains("model_cache_prune = \"on\""),
        "save() must write a mode that differs from the default:\n{body}"
    );
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("reload the saved config")
            .model_cache_prune,
        CachePruneMode::On
    );

    // The default mode is not written: a config that never mentioned the prune cannot turn it on.
    let default_path = std::env::temp_dir().join(format!(
        "tessella-worker-save-prune-default-{}.toml",
        std::process::id()
    ));
    WorkerConfig::default()
        .save(&default_path)
        .expect("save the default config");
    let body = std::fs::read_to_string(&default_path).expect("read the default config");
    assert!(
        !body.contains("model_cache_prune"),
        "the default mode must not be written:\n{body}"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&default_path);
}

/// S80: the env spelling is parsed by this helper, so its accepted values and its error message are
/// pinned here (reading the process environment in a test would race with the rest of the suite).
#[test]
fn model_cache_prune_env_parsing_accepts_the_documented_spellings() {
    for raw in ["off", "OFF", " none ", "disabled", "false", "0"] {
        assert_eq!(
            CachePruneMode::parse(raw).expect("a documented spelling"),
            CachePruneMode::Off,
            "raw={raw:?}"
        );
    }
    for raw in ["dry-run", "DRY_RUN", "DryRun"] {
        assert_eq!(
            CachePruneMode::parse(raw).expect("a documented spelling"),
            CachePruneMode::DryRun,
            "raw={raw:?}"
        );
    }
    for raw in ["on", "ON", " true ", "1"] {
        assert_eq!(
            CachePruneMode::parse(raw).expect("a documented spelling"),
            CachePruneMode::On,
            "raw={raw:?}"
        );
    }
    let err = CachePruneMode::parse("prune-everything")
        .expect_err("a typo must abort the boot, not silently leave the cache alone");
    assert!(
        err.to_string().contains("is not a cache-prune mode"),
        "the message must name the problem: {err}"
    );

    // The default is the safe direction, and the label written to the file parses back.
    assert_eq!(CachePruneMode::default(), CachePruneMode::Off);
    for mode in [
        CachePruneMode::Off,
        CachePruneMode::DryRun,
        CachePruneMode::On,
    ] {
        assert_eq!(
            CachePruneMode::parse(mode.as_str()).expect("its own label parses back"),
            mode
        );
    }
    assert!(!CachePruneMode::Off.is_on());
    assert!(
        CachePruneMode::DryRun.is_on(),
        "a dry run inspects the cache, so the node has to look at it"
    );
    assert!(!CachePruneMode::DryRun.deletes());
    assert!(CachePruneMode::On.deletes());
    assert!(!CachePruneMode::Off.deletes());
}

/// `TASK-76`/S83: the declared prefill ceiling is read from the schema the harness writes
/// (`config/swarm/worker_*.toml`, see `write_worker_config` of `scripts/e2e/swarm.sh`), and an
/// **absent** key means «this operator measured nothing» — `0`, never a silent default ceiling.
/// The difference is published: `0` makes `GET /status` say `max_input_tokens_source = "wire"`.
#[test]
fn the_declared_prefill_ceiling_is_read_from_the_legacy_schema_and_absent_means_undeclared() {
    let path = write_tmp_worker_toml(
        "ceiling",
        &format!("{LEGACY_PREFIX}max_input_tokens = 2048\n"),
    );
    let cfg = WorkerConfig::load(&path).expect("a declared ceiling loads");
    assert_eq!(
        cfg.max_input_tokens, 2_048,
        "the value written into worker.toml is the one the node declares"
    );
    let _ = std::fs::remove_file(&path);

    let path = write_tmp_worker_toml("ceiling-absent", LEGACY_PREFIX);
    let cfg = WorkerConfig::load(&path).expect("the key is optional");
    assert_eq!(
        cfg.max_input_tokens, 0,
        "0 = undeclared: only the wire binds and the node says so instead of inventing a ceiling"
    );
    let _ = std::fs::remove_file(&path);
}

/// `TASK-76`/S83: `p2p_inference_timeout_secs` is read in **both** schemas — it spent the whole
/// life of the project with **zero** readers while `http_pipeline_router` killed each forward after
/// a hard-coded 5 s, and the harness declares 60 s in the config it writes (the legacy schema of
/// `scripts/e2e/swarm.sh`, the one of `LEGACY_PREFIX`).
#[test]
fn the_declared_hop_wait_is_read_in_both_schemas_and_absent_means_the_documented_default() {
    // Legacy (flat) schema: what the e2e harness writes for every worker.
    let path = write_tmp_worker_toml(
        "hop-wait-legacy",
        &format!("{LEGACY_PREFIX}p2p_inference_timeout_secs = 60\n"),
    );
    let cfg = WorkerConfig::load(&path).expect("a declared wait loads");
    assert_eq!(
        cfg.p2p_inference_timeout_secs, 60,
        "the number of worker.toml is the one the router will wait"
    );
    assert_eq!(
        cfg.hop_wait().expect("a declared wait resolves"),
        std::time::Duration::from_secs(60),
        "the seconds of the config reach the router as a `Duration`"
    );
    let _ = std::fs::remove_file(&path);

    // Unified schema: a key of the `[worker]` table is honoured, never dropped in silence (the
    // defect `device` had until S27, `quantization` until S51, `canary_mode` until S78).
    let body = r#"
orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "/tmp/tessella-s83-hop-wait.key"
default_model = "Qwen/Qwen2.5-0.5B-Instruct"

[worker]
vram_mb = 1024
models_dir = "/tmp/tessella-s83-models"
api_port = 9001
p2p_port = 9001
p2p_inference_timeout_secs = 45
"#;
    let path = write_tmp_worker_toml("hop-wait-unified", body);
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("unified worker config")
            .hop_wait()
            .expect("a declared wait resolves"),
        std::time::Duration::from_secs(45),
        "the unified schema must not drop the key in silence"
    );
    let _ = std::fs::remove_file(&path);

    // Absent: the documented default, never an invented one.
    let path = write_tmp_worker_toml("hop-wait-absent", LEGACY_PREFIX);
    assert_eq!(
        WorkerConfig::load(&path)
            .expect("the key is optional")
            .hop_wait()
            .expect("the default resolves"),
        std::time::Duration::from_secs(default_p2p_inference_timeout_secs()),
        "absent means the named default, and it is the default of the struct too"
    );
    let _ = std::fs::remove_file(&path);
}

/// S83: `0` is neither «wait forever» nor «a hop has no time at all». The boot refuses it, naming
/// the key: `0` used to be the shape of a dead key, and a pipeline whose every hop loses the race
/// against its own timer is the symptom this session is about.
#[test]
fn a_zero_hop_wait_is_refused_instead_of_becoming_an_instant_timeout() {
    let path = write_tmp_worker_toml(
        "hop-wait-zero",
        &format!("{LEGACY_PREFIX}p2p_inference_timeout_secs = 0\n"),
    );
    // The *config* accepts it (it is a number, not a typo)…
    let cfg = WorkerConfig::load(&path).expect("`0` is a value, not a typo");
    assert_eq!(cfg.p2p_inference_timeout_secs, 0);
    // …and the boot refuses it with the key named.
    let err = cfg
        .hop_wait()
        .expect_err("`0` must not turn the pipeline into an instant timeout");
    let text = err.to_string();
    for expected in ["p2p_inference_timeout_secs = 0", "default: 10"] {
        assert!(text.contains(expected), "{expected:?} not in {text:?}");
    }
    let _ = std::fs::remove_file(&path);
}

/// S83: a round-trip neither loses the wait nor changes it.
#[test]
fn saved_config_round_trips_the_declared_hop_wait() {
    let path = std::env::temp_dir().join(format!(
        "tessella-worker-save-hop-wait-{}.toml",
        std::process::id()
    ));
    let cfg = WorkerConfig {
        p2p_inference_timeout_secs: 60,
        ..Default::default()
    };
    cfg.save(&path).expect("save the config");
    let body = std::fs::read_to_string(&path).expect("read the saved config");
    assert!(
        body.contains("p2p_inference_timeout_secs = 60"),
        "save() must write a wait that differs from the default:\n{body}"
    );
    let reloaded = WorkerConfig::load(&path).expect("reload the saved config");
    assert_eq!(reloaded.p2p_inference_timeout_secs, 60);
    assert_eq!(
        reloaded.hop_wait().expect("the reloaded wait resolves"),
        std::time::Duration::from_secs(60)
    );

    // The default is not written: a config that never mentioned the wait keeps the default of the
    // struct (and the default of the struct is the one the docs quote).
    let default_path = std::env::temp_dir().join(format!(
        "tessella-worker-save-hop-wait-default-{}.toml",
        std::process::id()
    ));
    WorkerConfig::default()
        .save(&default_path)
        .expect("save the default config");
    let body = std::fs::read_to_string(&default_path).expect("read the default config");
    assert!(
        !body.contains("p2p_inference_timeout_secs"),
        "the default wait must not be written:\n{body}"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&default_path);
}

#[test]
fn oom_streak_and_streak_window_secs_defaults_and_custom() {
    let default_toml = r#"
orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "keys/id_ed25519"
default_model = "test-model"

[worker]
vram_mb = 4096
models_dir = "./model_cache"
api_port = 8080
p2p_port = 9080
"#;
    let p_default = write_tmp_worker_toml("default-sat", default_toml);
    let cfg_default = WorkerConfig::load(&p_default).expect("load default config");
    assert_eq!(cfg_default.oom_streak, 3);
    assert_eq!(cfg_default.streak_window_secs, 30);
    let _ = std::fs::remove_file(&p_default);

    let custom_gpu_toml = r#"
orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "keys/id_ed25519"
default_model = "test-model"

[worker]
vram_mb = 4096
models_dir = "./model_cache"
api_port = 8080
p2p_port = 9080

[gpu]
oom_streak = 5
streak_window_secs = 45
"#;
    let p_custom = write_tmp_worker_toml("custom-gpu-sat", custom_gpu_toml);
    let cfg_custom = WorkerConfig::load(&p_custom).expect("load custom gpu config");
    assert_eq!(cfg_custom.oom_streak, 5);
    assert_eq!(cfg_custom.streak_window_secs, 45);
    let _ = std::fs::remove_file(&p_custom);
}

#[test]
fn oom_streak_and_streak_window_secs_zero_rejected() {
    let zero_streak_toml = r#"
orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "keys/id_ed25519"
default_model = "test-model"

[worker]
vram_mb = 4096
models_dir = "./model_cache"
api_port = 8080
p2p_port = 9080
oom_streak = 0
"#;
    let p_zero_streak = write_tmp_worker_toml("zero-streak", zero_streak_toml);
    let err = WorkerConfig::load(&p_zero_streak).expect_err("oom_streak=0 must fail");
    assert!(err.to_string().contains("oom_streak"));
    let _ = std::fs::remove_file(&p_zero_streak);

    let zero_window_toml = r#"
orchestrator_url = "http://127.0.0.1:8080"
keypair_path = "keys/id_ed25519"
default_model = "test-model"

[worker]
vram_mb = 4096
models_dir = "./model_cache"
api_port = 8080
p2p_port = 9080
streak_window_secs = 0
"#;
    let p_zero_window = write_tmp_worker_toml("zero-window", zero_window_toml);
    let err = WorkerConfig::load(&p_zero_window).expect_err("streak_window_secs=0 must fail");
    assert!(err.to_string().contains("streak_window_secs"));
    let _ = std::fs::remove_file(&p_zero_window);
}

#[test]
fn saved_config_round_trips_saturation_parameters() {
    let path = std::env::temp_dir().join(format!(
        "tessella-worker-save-sat-{}.toml",
        std::process::id()
    ));
    let cfg = WorkerConfig {
        oom_streak: 5,
        streak_window_secs: 60,
        ..Default::default()
    };
    cfg.save(&path).expect("save config");
    let body = std::fs::read_to_string(&path).expect("read saved config");
    assert!(body.contains("oom_streak = 5"));
    assert!(body.contains("streak_window_secs = 60"));

    let reloaded = WorkerConfig::load(&path).expect("reload saved config");
    assert_eq!(reloaded.oom_streak, 5);
    assert_eq!(reloaded.streak_window_secs, 60);

    let _ = std::fs::remove_file(&path);
}
