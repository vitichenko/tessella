use crate::config::WorkerConfig;
use crate::http_pipeline_router::HttpPipelineRouter;
use crate::inference_engine::InferenceEngine;
use crate::model_cache;
use crate::network_manager::{MusivumRelay, NetworkManager};
use crate::routing_table::RoutingTable;
use crate::runtime::WorkerRuntime;
use crate::shutdown::cancel_and_drain;
use crate::task_scheduler::TaskScheduler;
use crate::topology_cache::TopologyCache;

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tessella_core::{
    error::DomainError,
    traits::{DistributedInference, DistributedSession, PipelineRouter},
};
use tessella_protocol::WorkerAssignment;
use tokio::sync::{mpsc, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

pub struct WorkerService {
    pub config: WorkerConfig,
    pub engine: Arc<InferenceEngine>,
    pub routing_table: Arc<RwLock<RoutingTable>>,
    pub topology_cache: Arc<RwLock<TopologyCache>>,
    pub runtime: Arc<WorkerRuntime>,
    pub model_dir: std::path::PathBuf,
    pub tokenizer_path: Option<std::path::PathBuf>,
    pub topology_notification: Arc<tokio::sync::Notify>,
    pub storage: Arc<tessella_storage::SqliteStorage>,
    pub token_stream_senders: Arc<
        dashmap::DashMap<String, tokio::sync::mpsc::Sender<tessella_protocol::TokenStreamPayload>>,
    >,

    pub network_manager: Arc<NetworkManager>,
    pub task_scheduler: Arc<TaskScheduler>,
    pub scheduler_rx:
        tokio::sync::Mutex<Option<mpsc::Receiver<crate::scheduler::SchedulerCommand>>>,
}

/// Builds the `/api/v1/session_grant` request URL (`TASK-07`, H-12).
///
/// One round-trip returns the QoS ticket of H-13 **and** the `WorkTicket` that authorises this
/// session's work, so the authorisation costs no extra latency: the session already had to ask the
/// orchestrator for the Karma ticket.
fn session_grant_url(
    orchestrator_url: &str,
    peer_id: &str,
    pubkey_hex: &str,
    model_id: &str,
) -> String {
    format!(
        "{}/api/v1/session_grant?peer_id={}&pubkey={}&model_id={}",
        orchestrator_url.trim_end_matches('/'),
        peer_id,
        pubkey_hex,
        model_id
    )
}

/// Body of `GET /api/v1/session_grant` (`TASK-07`).
#[derive(Debug, serde::Deserialize)]
struct SessionGrant {
    karma_ticket: tessella_core::models::KarmaTicket,
    work_ticket: tessella_core::models::WorkTicket,
}

impl WorkerService {
    /// Boots the node: the port of the HTTP surface arrives from the composition root (`SOLID-89`),
    /// which is the only package that knows its concrete implementation.
    pub async fn new(
        config: WorkerConfig,
        api_host: Arc<dyn tessella_ingress::ApiHost>,
    ) -> Result<Self, DomainError> {
        let key_path = config.get_keypair_path();
        let (swarm, peer_id, keypair) = NetworkManager::init_p2p(&config, &key_path)
            .map_err(|e| DomainError::Config(e.to_string()))?;
        let (peer_network, cmd_rx) =
            tessella_network::peer_network::create_peer_network_handle(keypair.clone(), 128);

        let metrics = Arc::new(tessella_core::telemetry::WorkerMetrics::new());
        let storage = Arc::new(
            tessella_storage::SqliteStorage::new(&config.db_path, None)?
                .with_metrics(metrics.clone()),
        );

        // Model Metadata (moved earlier to support dynamic registration)
        // `TASK-82`/S89: the whole difference between a canary and a normal node at boot lives in this
        // call — a canary resolves nothing, reads no `config.json` and loads no tokenizer, and the
        // directory the operator declares for it need not even exist. What comes back has the same
        // shape either way, so no later step of the boot has to ask which kind of node this is.
        let crate::canary::BootModel {
            model_dir,
            tokenizer_path,
            model_config,
        } = crate::canary::boot_model(&config)
            .await
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        // Register with the musivum
        let bind_ip = if config.api_host != "0.0.0.0" {
            config.api_host.clone()
        } else if let Some(ref pub_ip) = config.public_ip {
            pub_ip.clone()
        } else {
            NetworkManager::get_first_non_loopback_ip().unwrap_or_else(|| "127.0.0.1".to_string())
        };

        // H-18: the address is built by the shared helper (and its host validated): an IP that is
        // not a literal used to be formatted into a multiaddr that only failed later, when it was
        // parsed two lines below.
        let p2p_addr_str = if let Some(ref pub_ip) = config.public_ip {
            let pub_port = config.public_p2p_port.unwrap_or(config.p2p_port);
            tessella_core::net::multiaddr_tcp(pub_ip, pub_port).ok_or_else(|| {
                DomainError::Config(format!("public_ip '{pub_ip}' is not a literal IP address"))
            })?
        } else {
            let bind_port = config.public_p2p_port.unwrap_or(config.p2p_port);
            tessella_core::net::multiaddr_tcp(&bind_ip, bind_port).ok_or_else(|| {
                DomainError::Config(format!(
                    "bind address '{bind_ip}' is not a literal IP address"
                ))
            })?
        };
        let p2p_addr = p2p_addr_str
            .parse::<tessella_network::NetworkAddress>()
            .map_err(|e| DomainError::Config(e.to_string()))?;

        let routing_table = Arc::new(RwLock::new(RoutingTable::new()));

        let mut cache_dir = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        cache_dir.push(".tessella");
        let _ = std::fs::create_dir_all(&cache_dir);
        let cache_path = cache_dir.join(format!("worker_topology_{}.json", config.listen_port));

        let topology_cache = Arc::new(RwLock::new(
            TopologyCache::new(Duration::from_secs(300)).with_persistence(cache_path),
        ));

        // H-13 + H-12 (and `TASK-62`/S68, decision D-25): the boot registration is the node asking its
        // authority to exist in the mesh. A tracker that **could not be reached** has not refused the
        // node, so the node waits for it with a bounded backoff instead of dying on the first attempt.
        // A tracker that **answered** and refused (`403`/`409`) is an authority's decision and is not
        // retried — `register_with_grace` is the only place that distinction is made.
        let grace =
            crate::registration::RegistrationGrace::from_secs(config.registration_grace_secs);
        if !grace.is_disabled() {
            info!(
                "🔁 Boot registration: up to {}s of retries if the tracker is unreachable \
                 (registration_grace_secs, TASK-62/S68). Serving still needs its authorisation.",
                grace.budget.as_secs()
            );
        }
        let (assignment, peers) =
            crate::registration::register_with_grace(grace, &config.orchestrator_url, |_attempt| {
                // The closure owns what each attempt needs: `register_and_assign` takes references and
                // the boot keeps using every one of these values afterwards.
                let config = config.clone();
                let peer_network = peer_network.clone();
                let peer_id = peer_id.clone();
                let p2p_addr = p2p_addr.clone();
                let bind_ip = bind_ip.clone();
                let model_config = model_config.clone();
                async move {
                    NetworkManager::register_and_assign(
                        &config,
                        peer_network.as_ref(),
                        &peer_id,
                        &p2p_addr,
                        &bind_ip,
                        Some(model_config),
                    )
                    .await
                }
            })
            .await
            .map_err(|e| {
                let reason = e.to_string();
                match e {
                    // A tracker that was never reachable (or whose budget ran out) is a network failure of
                    // the control plane, not an inference failure: the boot message has to say that.
                    crate::registration::RegistrationError::Unreachable { .. }
                    | crate::registration::RegistrationError::GaveUp { .. } => {
                        DomainError::Network(reason)
                    }
                    crate::registration::RegistrationError::Refused { .. }
                    | crate::registration::RegistrationError::Malformed { .. } => {
                        DomainError::Inference(reason)
                    }
                }
            })?;

        // `TASK-58`/S64: the cache takes the peers registration already returned (`LayerPeer`,
        // the same record the routing table uses). It used to build a second shape (`WorkerInfo`)
        // out of them just to store it and convert it back.

        // TASK-15 (H-06): what registration taught us is also the persistent cache, so the next
        // restart has a mesh view even if the tracker is down.
        if !peers.is_empty() {
            if let Err(e) = storage
                .remember_known_peers(&peers, tessella_core::time::unix_now_secs())
                .await
            {
                warn!(
                    "⚠️ Could not persist the peer cache from registration: {}",
                    e
                );
            }
        }

        routing_table.write().await.replace_all(peers.clone());
        topology_cache.write().await.update(peers);
        info!("🌐 Bootstrapped routing table and topology cache with peers from registration");

        // TASK-15 (H-06): if registration returned nobody (tracker with an empty mesh, or a tracker
        // that answered but knows nothing yet), the cache is what keeps this node reachable and
        // routable. It is a *fallback*: it never overwrites a view that came from the tracker.
        if routing_table.read().await.peers().is_empty() {
            match storage
                .load_known_peers(
                    tessella_core::time::unix_now_secs(),
                    tessella_storage::KNOWN_PEERS_TTL_SECS,
                )
                .await
            {
                Ok(cached) if !cached.is_empty() => {
                    info!(
                        "📂 Registration returned no peer; seeding the routing table with {} cached peer(s) (TASK-15)",
                        cached.len()
                    );
                    routing_table.write().await.replace_all(cached);
                }
                Ok(_) => {}
                Err(e) => warn!("⚠️ Could not read the peer cache: {}", e),
            }
        }

        // H-05: this is the boot-time attempt; the P2P loop re-discovers the
        // endpoint on every heartbeat while there is no dialable address, so a NAT
        // remap or an orchestrator restart no longer needs a worker restart. The
        // error is logged here instead of being discarded with `.ok()`.
        let musivum_relay = match NetworkManager::fetch_musivum_p2p_info(&config).await {
            Ok(relay) => relay,
            Err(e) => {
                warn!(
                    "⚠️ Could not fetch musivum P2P info from /p2p_id ({}); retrying on every heartbeat (HTTP routing remains active)",
                    e
                );
                MusivumRelay::default()
            }
        };
        // TASK-19: the boot-time identity check. If the operator pinned a key and the orchestrator
        // just observed is a different one, say so once instead of letting every request silently
        // fall back to `Leech`.
        crate::network_manager::warn_on_identity_conflict(
            config.orchestrator_public_key.as_deref(),
            musivum_relay.pubkey.as_deref(),
            &config.orchestrator_url,
        );
        if let Some(id) = musivum_relay.peer_id.as_deref() {
            info!(
                "🎯 Discovered musivum P2P ID: {} (dialable addr: {})",
                id,
                musivum_relay
                    .p2p_addr
                    .as_ref()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_else(|| "<none>".to_string())
            );
        }
        let musivum_relay = Arc::new(RwLock::new(musivum_relay));

        // Announce
        let real_ip = NetworkManager::get_real_ip(&swarm, &bind_ip);
        NetworkManager::announce_to_musivum(
            &config,
            &assignment,
            &peer_id,
            &real_ip,
            &bind_ip,
            peer_network.as_ref(),
        )
        .await
        .map_err(|e| DomainError::Inference(e.to_string()))?;

        // Backend Registry
        let mut registry = tessella_inference::backends::BackendRegistry::new();
        registry.register(tessella_inference::backends::BackendDescriptor {
            matches: |cfg| {
                let arch_match = cfg
                    .architectures
                    .iter()
                    .any(|a| a == "Qwen2ForCausalLM" || a == "Qwen2MoeForCausalLM");
                let type_match = cfg.model_type.as_deref() == Some("qwen2")
                    || cfg.model_type.as_deref() == Some("qwen2_moe")
                    || cfg.model_type.as_deref() == Some("qwen2_5");
                arch_match || type_match
            },
            factory: |dir, weights, device, tok, activations| {
                tessella_inference::backends::create_qwen_backend(
                    dir,
                    weights,
                    device,
                    tok,
                    activations,
                )
            },
            eos_token_ids: || vec![151643, 151645, 2],
        });
        let registry = Arc::new(registry);

        // Weights
        let candle_device =
            NetworkManager::select_device().map_err(|e| DomainError::Inference(e.to_string()))?;

        // Log diagnostics for heterogeneous GPU clusters and networks
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_dev) = candle_device.as_cuda_device() {
                let major = cuda_dev.attribute(candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0);
                let minor = cuda_dev.attribute(candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0);
                tracing::info!(
                    "Detected GPU Architecture: CUDA Compute Capability {}.{}",
                    major,
                    minor
                );
            } else {
                tracing::info!(
                    "Detected GPU Architecture: Non-CUDA Device (selected device is not CUDA)"
                );
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::info!("Detected GPU Architecture: Non-CUDA compilation");
        }

        match local_ip_address::list_afinet_netifas() {
            Ok(interfaces) => {
                let mut iface_list = Vec::new();
                for (name, ip) in interfaces {
                    iface_list.push(format!("{} ({})", name, ip));
                }
                tracing::info!("Available network interfaces: [{}]", iface_list.join(", "));
            }
            Err(e) => {
                tracing::warn!("Failed to retrieve network interfaces: {}", e);
            }
        }
        let (shards, served_shards) = NetworkManager::acquire_weights(
            &config,
            &assignment,
            &candle_device,
            model_config.total_layers(),
        )
        .await
        .map_err(|e| DomainError::Inference(e.to_string()))?;

        // `TASK-73`/S80: a node that **restarts** into a new range is the other half of the story
        // (`reload_shard` is the first one): the shards of yesterday's range are still on this disk, and
        // this is where they go. Same three guards as the reload path — nothing happens unless
        // `model_cache_prune` is on, nothing outside this model's own directory is touched, and nothing
        // is touched at all while another live process holds the model.
        if let Some(outcome) =
            model_cache::prune_after_reload(config.model_cache_prune, &served_shards)
        {
            model_cache::log_outcome(&outcome);
        }

        // Notify ready
        NetworkManager::notify_ready(&config, &assignment, &peer_id, 0, peer_network.as_ref())
            .await
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        let runtime = Arc::new(WorkerRuntime {
            config: config.clone(),
            peer_id: peer_id.clone(),
            orchestrator_url: config.orchestrator_url.clone(),
            bind_ip: bind_ip.clone(),
            p2p_addr: p2p_addr.to_string(),
            http_client: reqwest::Client::new(),
            // `TASK-76`/S83: the wait for a hop of this node's own pipeline. It was a dead key
            // (`p2p_inference_timeout_secs`, zero consumers) behind a 5 s literal in the router; here it
            // is validated **once** — a `0` that would lose every request is refused, naming the key —
            // and from here on it is the only timer a hop has.
            hop_wait: config
                .hop_wait()
                .map_err(|e| DomainError::Config(e.to_string()))?,
            keypair: Arc::new(keypair.clone()),
            // `TASK-50`/S56: what a worker keeps about hops that failed. A plain counter now — the
            // `CircuitBreaker` it replaces was never consulted by anybody — and `GET /status`
            // publishes it, which is the only consequence of a failed hop beyond the log line.
            peer_failures: Arc::new(tessella_core::telemetry::metrics::PeerFailureRegistry::new()),
            metrics: metrics.clone(),
        });

        // Reachable entry-node URL used for SSE token callbacks. Must be the
        // public address (proxy/NAT aware) so downstream workers can call back.
        let own_api_url = Some(config.resolve_http_api_url(&bind_ip));

        // H-10: the final shard only calls back hosts of this mesh (own URL + routing table),
        // and only reaches private ranges with an explicit opt-in.
        let callback_guard = crate::callback_guard::CallbackGuard::from_config(
            &config,
            own_api_url.clone(),
            routing_table.clone(),
        );

        let orchestrator_public_key = match config.orchestrator_verifying_key() {
            Ok(key) => key,
            Err(e) => {
                error!(
                    "❌ {}. The worker keeps running, but WITHOUT pinning: every KarmaTicket is \
                     ignored and every request is served as Leech until the key is fixed (copy the \
                     'orchestrator_pubkey' field of GET /p2p_id)",
                    e
                );
                None
            }
        };

        // H-10 (mitigation 2): the entry node issues one callback capability per sealed payload and
        // the HTTP handler of this same process is the only one able to accept it.
        let callback_capabilities =
            Arc::new(tessella_core::callback::capability::CallbackCapabilities::new());

        // S49/TASK-43: the expert-activation registry of **this** worker. It used to be a
        // process-wide `static`, so two engines in one process shared the counters and the drain of
        // one emptied the other. The engine owns it (the MoE hot path records there) and the
        // heartbeat drains the same `Arc` into `/announce.expert_usage`.
        let expert_activations =
            Arc::new(tessella_core::telemetry::metrics::ExpertActivationRegistry::new());

        let mut engine_inner = InferenceEngine::new(
            shards,
            model_config.clone(),
            assignment.clone(),
            candle_device,
            registry.clone(),
            own_api_url,
            Some(storage.clone()),
            Some(keypair.clone()),
            orchestrator_public_key,
            callback_guard,
            callback_capabilities,
            expert_activations,
            config.canary_mode,
            // TASK-76/S83: what this node's declared VRAM can prefill (`max_input_tokens` of
            // `worker.toml`, `0` = undeclared). The engine carries it to the guard of `Session::forward`
            // and to `GET /status`, which is why it is passed here and not re-read from the config.
            config.max_input_tokens,
            // TASK-77/S84: el compás con el que este nodo revalida su device cuando se declara caído. Es
            // el mismo del latido (`heartbeat_interval_secs`), y el engine lo publica en `/status` como
            // `gpu.revalidate_secs` para que el operador sepa cuánto puede tardar una vuelta sola.
            config.heartbeat_interval_secs,
            // TASK-85/S92: umbral y ventana para declarar saturación por racha consecutiva de OOMs.
            config.oom_streak,
            config.streak_window_secs,
        );

        let (scheduler_tx, scheduler_rx) = mpsc::channel(100);
        engine_inner.set_scheduler_tx(scheduler_tx);
        let engine = Arc::new(engine_inner);

        let topology_notification = Arc::new(tokio::sync::Notify::new());
        let token_stream_senders = Arc::new(dashmap::DashMap::new());

        let network_manager = Arc::new(NetworkManager::new(
            config.clone(),
            runtime.clone(),
            routing_table.clone(),
            topology_cache.clone(),
            engine.clone(),
            storage.clone(),
            token_stream_senders.clone(),
            topology_notification.clone(),
            musivum_relay.clone(),
            model_dir.clone(),
            tokenizer_path.clone(),
            api_host.clone(),
            peer_network.clone(),
            Some(cmd_rx),
        ));

        let task_scheduler = Arc::new(TaskScheduler::new(
            engine.clone(),
            storage.clone(),
            runtime.clone(),
        ));

        Ok(Self {
            config,
            engine,
            routing_table,
            topology_cache,
            runtime,
            model_dir,
            tokenizer_path,
            topology_notification,
            storage,
            token_stream_senders,
            network_manager,
            task_scheduler,
            scheduler_rx: tokio::sync::Mutex::new(Some(scheduler_rx)),
        })
    }

    pub async fn run(self) -> Result<(), DomainError> {
        let app_state = Arc::new(self);

        info!("🚀 Worker ready — starting P2P loop and API server");

        let config = app_state.config.clone();

        let key_path = config.get_keypair_path();
        let (swarm, _, _) = NetworkManager::init_p2p(&config, &key_path)
            .map_err(|e| DomainError::Config(e.to_string()))?;

        let cancel_token = CancellationToken::new();

        // H-21/TASK-13: a process that never closes its SQLite connection never reclaims the `-wal`
        // sidecar, so the worker runs the same maintenance task the orchestrator runs. It is not one
        // of the scheduler loops: it owns its own token and is stopped at the end of `run`.
        let wal_maintenance = tessella_storage::WalMaintenance::spawn(
            app_state.storage.clone(),
            std::time::Duration::from_secs(app_state.config.wal_checkpoint_interval_secs),
        );

        // 1. Start TaskScheduler (which manages the priority queue scheduler, WAL sweeper, and batch collector)
        //    TASK-10 Part A: the handles are kept so the shutdown path below can wait for them.
        let task_handles = if let Some(scheduler_rx) = app_state.scheduler_rx.lock().await.take() {
            app_state
                .task_scheduler
                .clone()
                .start(scheduler_rx, cancel_token.clone())
        } else {
            Vec::new()
        };

        // 2. Load tokenizer and set router on the engine. `TASK-82`/S89: a canary loads **no**
        // tokenizer — it is the piece that used to force a model directory into a node that has no
        // model — and `initialize_models` is where the requirement lives for every other node.
        let tokenizer =
            Self::load_tokenizer(&app_state).map_err(|e| DomainError::Inference(e.to_string()))?;

        let router = Arc::new(HttpPipelineRouter::new(
            app_state.routing_table.clone(),
            app_state.topology_cache.clone(),
            app_state.runtime.http_client.clone(),
            app_state.runtime.orchestrator_url.clone(),
            app_state.runtime.peer_id.clone(),
            app_state.engine.model_config.model_id.clone(),
            app_state.runtime.peer_failures.clone(),
            app_state.topology_notification.clone(),
            app_state.runtime.keypair.clone(),
            // `TASK-76`/S83: the declared wait, not the 5 s literal that used to kill a hop which was
            // still working (and counted it as a failure).
            app_state.runtime.hop_wait,
        ));
        app_state.engine.set_router(router).await;

        app_state
            .engine
            .initialize_models(&app_state.model_dir, tokenizer.clone())
            .await?;

        // Validate assignment. `TASK-82`/S89: a canary has no `config.json` to read, and the layer
        // count it validated at boot is the one it uses here — the same source its registration
        // manifest uses (`canary::canary_model_config`), so the two cannot disagree. The check itself
        // is unchanged and runs for both kinds of node: a range that does not fit the model is a
        // misconfiguration whether or not the node holds the weights.
        let (total_layers, layer_ranges) = {
            let total = if app_state.config.canary_mode.is_on() {
                app_state.engine.model_config.num_hidden_layers
            } else {
                let config_json_path = app_state.model_dir.join("config.json");
                let config_json_content =
                    std::fs::read_to_string(&config_json_path).map_err(|e| {
                        DomainError::Inference(format!("Failed to read config.json: {}", e))
                    })?;
                let config_json: serde_json::Value = serde_json::from_str(&config_json_content)
                    .map_err(|e| {
                        DomainError::Inference(format!("Failed to parse config.json: {}", e))
                    })?;

                config_json["num_hidden_layers"]
                    .as_u64()
                    .or_else(|| config_json["n_layers"].as_u64())
                    .ok_or_else(|| {
                        DomainError::Inference("Missing num_hidden_layers in config.json".into())
                    })? as usize
            };

            let assignment = app_state.engine.assignment.read().await;
            // Every range of the shard, not only its end: a node with a second chunk has to be
            // validated against the model it is about to load twice (`TASK-96`/S102). The refusal
            // is the one `layer_ranges::validate` gives, which names the offending range.
            tessella_protocol::layer_ranges::validate(&assignment.layer_ranges, total)
                .map_err(|e| DomainError::Inference(e.to_string()))?;
            (total, assignment.layer_ranges.clone())
        };

        info!(
            "✅ Model config validated: {} layers, shard {:?}",
            total_layers, layer_ranges,
        );

        // 3. Delegate to NetworkManager to execute P2P loop and HTTP API server
        let network_manager = app_state.network_manager.clone();
        let result = network_manager
            .run(app_state.clone(), swarm, cancel_token.clone())
            .await;

        // TASK-10 Part A: `run` has already cancelled the shared token (its own shutdown path
        // drains the tasks it spawned), so the scheduler tasks are on their way out. Waiting for
        // them here is what guarantees that no background task outlives this function.
        let outcome = cancel_and_drain(&cancel_token, task_handles).await;
        if !outcome.all_completed() {
            warn!("⚠️ The worker stopped with background tasks still running");
        }

        // H-21/TASK-13: the WAL maintenance task is not part of `task_handles` (it owns its own
        // token), so it is stopped here — after the shared token has already stopped everything
        // else. `shutdown()` cancels and joins, and never fails the caller.
        if let Some(task) = wal_maintenance {
            task.shutdown().await;
        }

        result
    }

    pub async fn reload_shard(&self, new_assignment: &WorkerAssignment) -> Result<(), DomainError> {
        self.network_manager.reload_shard(new_assignment).await
    }

    /// The tokenizer of this node, or `None` for a canary (`TASK-82`/S89).
    ///
    /// A canary has none to load: it never embeds a token (its entry-node path refuses by name), and
    /// forcing it to carry the files was exactly what made an empty `models_dir` impossible. A path
    /// that is missing stays an error like any other — only a canary is exempt, and only by being a
    /// canary.
    fn load_tokenizer(
        state: &WorkerService,
    ) -> Result<Option<Arc<tessella_inference::TokenizerWrapper>>, anyhow::Error> {
        if state.config.canary_mode.is_on() {
            return Ok(None);
        }
        let path = state
            .tokenizer_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer path"))?;
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
        Ok(Some(Arc::new(
            tessella_inference::TokenizerWrapper::from_file(path_str)?,
        )))
    }
}

#[async_trait::async_trait]
impl DistributedInference for WorkerService {
    async fn reset_distributed_cache(&self) {
        self.engine.reset_distributed_cache().await;
    }

    async fn apply_prompt_template(
        &self,
        mode: &tessella_core::models::InferenceMode,
        raw_prompt: &str,
    ) -> String {
        self.engine.apply_prompt_template(mode, raw_prompt).await
    }

    async fn begin_session(
        self: Arc<Self>,
        session_hint: Option<String>,
    ) -> Result<(Box<dyn DistributedSession>, String), DomainError> {
        let router = Arc::new(HttpPipelineRouter::new(
            self.routing_table.clone(),
            self.topology_cache.clone(),
            self.runtime.http_client.clone(),
            self.runtime.orchestrator_url.clone(),
            self.runtime.peer_id.clone(),
            self.engine.model_config.model_id.clone(),
            self.runtime.peer_failures.clone(),
            self.topology_notification.clone(),
            self.runtime.keypair.clone(),
            // `TASK-76`/S83: same declared wait as the router of `WorkerService::new` — this one serves
            // the requests of an existing session, and they are the long ones (a ceiling prompt).
            self.runtime.hop_wait,
        ));

        // Attempt to refresh the routing table to get the latest topology
        if let Err(e) = router.refresh().await {
            tracing::warn!("Failed to refresh routing table on begin_session: {}", e);
        }

        let model_id = self.engine.model_config.model_id.clone();
        let covers_layer_0 = {
            let assignment = self.engine.assignment.read().await;
            if tessella_protocol::layer_ranges::covers(&assignment.layer_ranges, 0) {
                true
            } else {
                let table = self.routing_table.read().await;
                !table.get_peers_for_layer(0).is_empty()
            }
        };

        if !covers_layer_0 {
            return Err(DomainError::Network(format!(
                "No healthy workers found for model {} covering Layer 0",
                model_id
            )));
        }

        // H-13 + H-12: a single round-trip asks the orchestrator for the QoS ticket **and** the
        // `WorkTicket` that authorises this session's work. Posing both together is what keeps the
        // credit authorisation from adding latency to the first token, because the session already
        // had to talk to the orchestrator for the Karma ticket.
        let (karma_ticket, work_ticket) = self.fetch_session_grant().await?;

        // H-09 (`TASK-05`, point 5): without a keypair the payloads emitted by this session
        // could not be signed, and an unsigned payload is rejected by every hop (and must not be
        // emitted at all). The old code silently sent `signature: []`, which verified nowhere.
        if self.engine.keypair.is_none() {
            return Err(DomainError::Config(
                "no local keypair: refusing to start an inference session whose payloads cannot \
                 be signed (H-09). Check the worker key file / permissions."
                    .to_string(),
            ));
        }

        let (session, request_id) = self
            .engine
            .clone()
            .begin_session_with_router(
                router,
                // `session_hint` is the **client's** `session_id`, and only a hint (`TASK-78`/S85,
                // `D-36`): the engine honours it when it names a session this node still holds and
                // issues a new id otherwise. It is never the work id — that one is minted by the
                // orchestrator (`TASK-07`) and comes back inside the work ticket, which is what the
                // caller routes and reports.
                session_hint,
                self.engine.own_api_url.clone(),
                Some(karma_ticket),
                Some(work_ticket),
            )
            .await?;

        Ok((session, request_id))
    }

    async fn forward_hidden(&self, bytes: &[u8]) -> Result<Vec<u8>, DomainError> {
        self.engine.forward_hidden(bytes).await
    }

    async fn update_assignment(&self, new_assignment: WorkerAssignment) -> Result<(), DomainError> {
        self.reload_shard(&new_assignment).await?;

        // Notify musivum that we are ready with new shards
        let queue_depth = self.engine.sessions.len();
        let mut ready_req = tessella_core::WorkerReady {
            peer_id: self.runtime.peer_id.clone(),
            model_repo: new_assignment.model_repo.clone(),
            layer_ranges: new_assignment.layer_ranges.clone(),
            replica_group_id: new_assignment.replica_group_id,
            expert_range: new_assignment.expert_range,
            queue_depth,
            public_key: Vec::new(),
            signature: Vec::new(),
        };
        // D-3: `/ready` is a destructive endpoint for the orchestrator (rebalance), so the
        // self-reported peer id must be backed by this worker's signature.
        tessella_network::sign_peer_request(&self.runtime.keypair, &mut ready_req)
            .map_err(|e| DomainError::Network(format!("Could not sign /ready: {e}")))?;

        let _ = self
            .runtime
            .http_client
            .post(format!("{}/ready", self.runtime.orchestrator_url))
            .json(&ready_req)
            .send()
            .await;

        Ok(())
    }

    async fn current_assignment(&self) -> Option<WorkerAssignment> {
        Some(self.engine.assignment.read().await.clone())
    }

    async fn rebalance(&self, layer_start: usize, layer_end: usize) -> Result<(), DomainError> {
        let replica_group_id = self.engine.assignment.read().await.replica_group_id;
        let new_assignment = WorkerAssignment {
            layer_ranges: vec![(layer_start, layer_end)],
            model_repo: self.engine.model_config.model_id.clone(),
            replica_group_id,
            expert_range: None,
        };
        self.update_assignment(new_assignment).await
    }
} // end impl WorkerService

impl WorkerService {
    /// The refusal a node gives when its **control plane** cannot be reached for the session grant
    /// (`TASK-62`/S68, decision `D-25`).
    ///
    /// The words are the contract: S58 measured that a request to the entry node with the tracker down
    /// comes back `500` in **0.0 s** while no hop was lost (0 `Pipeline timeout`, routing from the
    /// cached topology). The missing piece is the per-request authority of D-7/D-13, not the routing,
    /// and the message has to say so — otherwise the operator goes looking for a broken hop.
    ///
    /// Split out of `fetch_session_grant` (like `apply_session_metadata` in
    /// `inference_engine`) so the words can be pinned by a test with no tracker, no model and no GPU.
    fn session_grant_unreachable(
        orchestrator_url: &str,
        cause: &dyn std::fmt::Display,
    ) -> DomainError {
        DomainError::Network(format!(
            "the mesh does not serve a session without the orchestrator's authorisation (D-7/D-13, \
             decision D-25): the control plane at {orchestrator_url} could not be reached for the \
             session grant: {cause}. The topology is not the problem — this node routes from its \
             cached view — it just cannot serve work its authority did not authorise."
        ))
    }

    /// Asks the orchestrator for the **session grant** (`TASK-07`, H-12): the Karma ticket of H-13
    /// and the `WorkTicket` that makes this session's credit authoritative.
    ///
    /// Both are **required** since `TASK-57`/S63. Until then an orchestrator that answered `404`
    /// (a pre-`TASK-07` one) made the worker fall back to the classic `/api/v1/karma_ticket` and
    /// start the session without a work ticket, whose work an orchestrator still running with
    /// `require_work_ticket = false` credited from the worker's own `credits_delta`. That window is
    /// closed on both sides: there is no fleet to migrate, and a session that cannot be paid is not
    /// a session this node should serve — it fails here, before any GPU work, naming what is wrong
    /// instead of producing tokens nobody will be paid for.
    ///
    /// `TASK-62`/S68 (D-25) keeps that promise **explicit** for the unreachable case: a network
    /// failure is not a degraded mode, it is the declared limit of the mesh (the control plane
    /// authorises the work of every session) and the error says exactly that. What S68 fixed is the
    /// other half of the same finding: the **boot** no longer dies the first time the tracker is
    /// unreachable (`registration::register_with_grace`).
    async fn fetch_session_grant(
        &self,
    ) -> Result<
        (
            tessella_core::models::KarmaTicket,
            tessella_core::models::WorkTicket,
        ),
        DomainError,
    > {
        // H-13: the orchestrator derives the peer id from this public key and refuses to issue a
        // ticket that is not bound to it, so the request has to carry the same key the payload is
        // signed with below. Without a keypair the session could not be sealed either (H-09), so it
        // is a configuration error and not a degraded mode.
        let requester_pubkey_hex = match self.engine.keypair.as_ref() {
            Some(kp) => hex::encode(kp.public().encode_protobuf()),
            None => {
                return Err(DomainError::Config(
                    "no local keypair: cannot request the work ticket that authorises this session \
                     (H-13/D-7). Check the worker key file / permissions."
                        .to_string(),
                ));
            }
        };

        let url = session_grant_url(
            &self.runtime.orchestrator_url,
            &self.runtime.peer_id,
            &requester_pubkey_hex,
            &self.engine.model_config.model_id,
        );
        let resp = self
            .runtime
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| Self::session_grant_unreachable(&self.runtime.orchestrator_url, &e))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(DomainError::Network(format!(
                "Orchestrator does not serve /api/v1/session_grant (404): this worker cannot be paid \
                 for its work without a work ticket, so the session is refused instead of served for \
                 free. The orchestrator is older than TASK-07; orchestrator and workers deploy \
                 together (TASK-57/S63 closed the migration window). URL: {url}"
            )));
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(DomainError::Network(format!(
                "Orchestrator rejected the session grant (status {}): {}",
                status, err_body
            )));
        }

        let grant = resp.json::<SessionGrant>().await.map_err(|e| {
            DomainError::Serialization(format!(
                "Failed to parse the session grant from the orchestrator: {}",
                e
            ))
        })?;

        // `TASK-63`/S69: both tickets are returned **by value** and travel inside the session and
        // every payload it seals (see `apply_session_metadata` in this crate). Nothing else keeps a
        // copy: until S63 there was an `Arc<Mutex<Option<KarmaTicket>>>` written here and read
        // only by the `/api/v1/karma_ticket` fallback that S63 retired, and it was still being
        // cloned across three layers into an `ApiState` field no handler consulted.
        info!(
            "🎫 Session grant received: work ticket '{}' for model '{}' (karma tier {:?})",
            grant.work_ticket.request_id, grant.work_ticket.model_id, grant.karma_ticket.tier
        );
        Ok((grant.karma_ticket, grant.work_ticket))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TASK-62`/S68 (D-25): the words of the refusal **are** the contract. S58 measured that a
    /// request with the tracker down is refused in 0.0 s with no hop lost, so the message has to name
    /// the authority that is missing (and its decision) instead of letting an operator go looking for
    /// a broken hop. It also has to carry the URL and the cause, or it is not actionable.
    #[test]
    fn the_session_grant_refusal_names_the_authority_that_is_missing() {
        let error = WorkerService::session_grant_unreachable(
            "http://tracker:8080",
            &"error sending request for url (http://tracker:8080/api/v1/session_grant)",
        );
        let message = error.to_string();

        assert!(message.contains("orchestrator"), "{message}");
        assert!(message.contains("session grant"), "{message}");
        assert!(message.contains("D-7"), "{message}");
        assert!(message.contains("http://tracker:8080"), "{message}");
        assert!(
            message.contains("could not be reached"),
            "the cause has to be part of the message: {message}"
        );
        assert!(
            message.contains("topology is not the problem"),
            "it must not read like a routing failure: {message}"
        );
        assert!(
            matches!(error, DomainError::Network(_)),
            "the mesh has no degraded mode: it is a control-plane network failure"
        );
    }
}
