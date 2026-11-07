pub mod api_host;
pub mod membership;
pub mod swarm_loop;
pub mod token_stream;

#[cfg(test)]
mod tests;

pub use api_host::setup_api_state;
pub use membership::{
    announce_to_musivum, announce_via_gossip, handle_membership_gossip, notify_ready,
    perform_graceful_shutdown,
};
pub use swarm_loop::{
    fetch_musivum_p2p_info, fetch_musivum_p2p_info_from, get_first_non_loopback_ip, get_real_ip,
    init_p2p, orchestrator_key_conflict, real_ip_from_listeners, refresh_topology_cache,
    remember_peers, run_p2p_loop, run_refresh_loop, run_topology_refresh,
    warn_on_identity_conflict, MusivumRelay,
};
pub use token_stream::{new_token_stream_map, TokenStreamSenderMap};

use crate::config::WorkerConfig;
use crate::hf_loader;
use crate::inference_engine::InferenceEngine;
use crate::inference_engine::ShardSet;
use crate::model_cache;
use crate::registration::RegistrationError;
use crate::routing_table::RoutingTable;
use crate::runtime::WorkerRuntime;
use crate::shutdown::cancel_and_drain;
use crate::topology_cache::TopologyCache;

use anyhow::Result;
use candle_core::Device;
use libp2p;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use tessella_core::{
    error::DomainError,
    traits::{DistributedInference, PeerNetwork},
};
use tessella_inference::{load_sharded_weights, ShardedWeights};
use tessella_ingress::{ApiHost, ApiState};
use tessella_network::peer_network::SwarmCommand;
use tessella_network::{MoeBehaviour, NetworkKeypair};
use tessella_protocol::layer_ranges;
use tessella_protocol::{ModelManifest, WorkerAssignment, WorkerRegistration};
use tokio::signal;
use tokio::sync::{mpsc, RwLock};
use tokio_util::sync::CancellationToken;
use tracing as log;
use tracing::{error, info, warn};

/// Rewrites the host of an announced HTTP URL to the peer's Tailscale/CGNAT address when the
/// peer advertises one (`100.x`, the `100.64.0.0/10` range).
pub(crate) fn resolve_preferred_http_addr(http_addr: &str, p2p_addr: &str) -> String {
    let Some(IpAddr::V4(ip)) = tessella_core::net::multiaddr_ip(p2p_addr) else {
        return http_addr.to_string();
    };
    if ip.octets()[0] != 100 {
        return http_addr.to_string();
    }

    let Some((scheme, port)) = tessella_core::net::url_scheme_and_port(http_addr) else {
        warn!(
            "⚠️ Cannot value the Tailscale address {} over '{}': the announced URL is not \
             parseable, so it is left as it is",
            ip, http_addr
        );
        return http_addr.to_string();
    };

    match port {
        Some(port) => format!("{scheme}://{ip}:{port}"),
        None => format!("{scheme}://{ip}"),
    }
}

pub struct NetworkManager {
    pub config: WorkerConfig,
    pub runtime: Arc<WorkerRuntime>,
    pub routing_table: Arc<RwLock<RoutingTable>>,
    pub topology_cache: Arc<RwLock<TopologyCache>>,
    pub engine: Arc<InferenceEngine>,
    pub storage: Arc<tessella_storage::SqliteStorage>,
    pub token_stream_senders: TokenStreamSenderMap,
    pub topology_notification: Arc<tokio::sync::Notify>,
    pub musivum_relay: Arc<RwLock<MusivumRelay>>,
    pub model_dir: std::path::PathBuf,
    pub tokenizer_path: Option<std::path::PathBuf>,
    pub api_host: Arc<dyn ApiHost>,
    pub peer_network: Arc<dyn PeerNetwork>,
    pub cmd_rx: tokio::sync::Mutex<Option<mpsc::Receiver<SwarmCommand>>>,
}

impl NetworkManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: WorkerConfig,
        runtime: Arc<WorkerRuntime>,
        routing_table: Arc<RwLock<RoutingTable>>,
        topology_cache: Arc<RwLock<TopologyCache>>,
        engine: Arc<InferenceEngine>,
        storage: Arc<tessella_storage::SqliteStorage>,
        token_stream_senders: TokenStreamSenderMap,
        topology_notification: Arc<tokio::sync::Notify>,
        musivum_relay: Arc<RwLock<MusivumRelay>>,
        model_dir: std::path::PathBuf,
        tokenizer_path: Option<std::path::PathBuf>,
        api_host: Arc<dyn ApiHost>,
        peer_network: Arc<dyn PeerNetwork>,
        cmd_rx: Option<mpsc::Receiver<SwarmCommand>>,
    ) -> Self {
        Self {
            config,
            runtime,
            routing_table,
            topology_cache,
            engine,
            storage,
            token_stream_senders,
            topology_notification,
            musivum_relay,
            model_dir,
            tokenizer_path,
            api_host,
            peer_network,
            cmd_rx: tokio::sync::Mutex::new(cmd_rx),
        }
    }

    pub async fn run(
        self: Arc<Self>,
        service: Arc<dyn DistributedInference>,
        mut swarm: libp2p::Swarm<MoeBehaviour>,
        cancel_token: CancellationToken,
    ) -> Result<(), DomainError> {
        info!("P2P networking loop and HTTP API server starting via NetworkManager");

        let cmd_rx = self.cmd_rx.lock().await.take().ok_or_else(|| {
            DomainError::Network("swarm command receiver already consumed".to_string())
        })?;

        let config = self.config.clone();
        let assignment = self.engine.assignment.read().await.clone();
        let p2p_addr = self
            .runtime
            .p2p_addr
            .parse::<libp2p::Multiaddr>()
            .map_err(|e| DomainError::Config(e.to_string()))?;

        let gossip_topic = tessella_network::GossipTopic::new(tessella_network::WORKERS_TOPIC);
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&gossip_topic)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        let topology_topic = tessella_network::GossipTopic::new(format!(
            "/musivum/topology/{}",
            self.engine.model_config.model_id
        ));
        info!("Subscribing to topology topic: {}", topology_topic);
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&topology_topic)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        let bind_ip = self.runtime.bind_ip.clone();
        let peer_id = self.runtime.peer_id.clone();
        let real_ip = Self::get_real_ip(&swarm, &bind_ip);
        Self::announce_via_gossip(
            &mut swarm,
            &peer_id,
            &config,
            &assignment,
            &p2p_addr,
            &real_ip,
            &bind_ip,
        )
        .map_err(|e| DomainError::Inference(e.to_string()))?;

        self.run_api_and_p2p(service, swarm, cmd_rx, cancel_token, gossip_topic)
            .await
    }

    async fn run_api_and_p2p(
        self: Arc<Self>,
        service: Arc<dyn DistributedInference>,
        swarm: libp2p::Swarm<MoeBehaviour>,
        cmd_rx: mpsc::Receiver<SwarmCommand>,
        cancel_token: CancellationToken,
        _gossip_topic: tessella_network::GossipTopic,
    ) -> Result<(), DomainError> {
        let addr: SocketAddr = format!(
            "{}:{}",
            self.runtime.config.api_host, self.runtime.config.api_port
        )
        .parse()
        .map_err(|e| DomainError::Config(format!("Invalid API bind address: {}", e)))?;
        let api_state = self.setup_api_state(service);

        let mut handles = vec![self.engine.spawn_cleanup_task(cancel_token.clone())];
        handles.push(tokio::spawn({
            let network = self.clone();
            let cancel = cancel_token.clone();
            async move { network.run_topology_refresh(cancel).await }
        }));

        let self_cl = self.clone();
        let cancel_token_cl = cancel_token.clone();
        let api_host = self.api_host.clone();

        tokio::select! {
            _ = api_host::serve_api(api_host, addr, api_state) => {
                error!("❌ API Server halted");
            }
            _ = swarm_loop::run_p2p_loop(self_cl, swarm, cmd_rx, cancel_token_cl) => {
                error!("❌ P2P Loop halted");
            }
            _ = self.clone().run_refresh_loop(cancel_token.clone()) => {
                error!("❌ Refresh Loop halted");
            }
            _ = Self::shutdown_signal() => {
                info!("🛑 Shutdown signal received.");
            }
        }

        info!("🛑 Stopping background tasks...");
        let outcome = cancel_and_drain(&cancel_token, handles).await;
        if !outcome.all_completed() {
            warn!("⚠️ The worker is shutting down with background tasks still running");
        }

        self.perform_graceful_shutdown().await;
        Ok(())
    }

    pub async fn run_topology_refresh(self: Arc<Self>, cancel_token: CancellationToken) {
        swarm_loop::run_topology_refresh(self, cancel_token).await;
    }

    pub async fn run_refresh_loop(
        self: Arc<Self>,
        cancel_token: CancellationToken,
    ) -> Result<(), DomainError> {
        swarm_loop::run_refresh_loop(self, cancel_token).await
    }

    pub async fn refresh_topology_cache(&self) {
        swarm_loop::refresh_topology_cache(self).await;
    }

    pub fn select_device() -> Result<candle_core::Device> {
        #[cfg(feature = "cuda")]
        {
            match candle_core::Device::new_cuda(0) {
                Ok(d) => {
                    tracing::info!("Device: CUDA:0");
                    return Ok(d);
                }
                Err(e) => {
                    tracing::warn!("CUDA available but failed to initialize: {}. Using CPU.", e)
                }
            }
        }
        #[cfg(feature = "metal")]
        {
            match candle_core::Device::new_metal(0) {
                Ok(d) => {
                    tracing::info!("Device: Metal");
                    return Ok(d);
                }
                Err(e) => tracing::warn!(
                    "Metal available but failed to initialize: {}. Using CPU.",
                    e
                ),
            }
        }
        tracing::info!("Device: CPU (GPU not available)");
        Ok(candle_core::Device::Cpu)
    }

    pub fn init_p2p(
        config: &WorkerConfig,
        key_path: &str,
    ) -> Result<(libp2p::Swarm<MoeBehaviour>, String, NetworkKeypair)> {
        swarm_loop::init_p2p(config, key_path)
    }

    pub async fn register_and_assign(
        config: &WorkerConfig,
        peer_network: &dyn PeerNetwork,
        peer_id: &str,
        listen_addr: &libp2p::Multiaddr,
        bind_ip: &str,
        model_config: Option<tessella_core::models::ModelConfig>,
    ) -> Result<(WorkerAssignment, Vec<tessella_core::models::LayerPeer>), RegistrationError> {
        info!("📡 Registering with musivum at {}", config.orchestrator_url);

        let model_config = crate::canary::registration_model_config(config, model_config.as_ref());
        let total_layers = model_config.as_ref().map_or_else(
            || tessella_protocol::get_total_layers(&config.default_model).unwrap_or(28),
            tessella_core::models::ModelConfig::total_layers,
        );

        let manifest = if let Some(ref mc) = model_config {
            ModelManifest {
                model_id: mc.model_id.clone(),
                num_hidden_layers: mc.num_hidden_layers,
                num_experts: mc.num_experts,
            }
        } else {
            ModelManifest {
                model_id: config.default_model.clone(),
                num_hidden_layers: tessella_protocol::get_total_layers(&config.default_model)
                    .unwrap_or(28),
                num_experts: None,
            }
        };

        let api_addr = config.resolve_http_api_url(bind_ip);

        let vram_mb = if let Ok(val) = std::env::var("TESSELLA_VRAM_MB") {
            val.parse::<u32>()
                .unwrap_or_else(|_| tessella_core::vram::vram_mb_from_gb(config.vram_budget_gb))
        } else {
            tessella_core::vram::vram_mb_from_gb(config.vram_budget_gb)
        };

        let mut registration = WorkerRegistration {
            peer_id: peer_id.to_string(),
            p2p_address: listen_addr.to_string(),
            relay_address: None,
            api_addr,
            vram_mb,
            device: config
                .get_candle_device()
                .map_err(|e| RegistrationError::Malformed {
                    message: e.to_string(),
                })?,
            model_id: config.default_model.clone(),
            total_layers,
            force_full_model: Some(config.force_full_model),
            model_config,
            manifest,
            parameter_contract_hash: tessella_protocol::swarm_contract_hash(&config.default_model),
            public_ip: config.public_ip.clone(),
            public_api_port: config.public_api_port,
            public_key: Vec::new(),
            signature: Vec::new(),
        };

        peer_network
            .sign_peer_request(&mut registration)
            .map_err(|e| RegistrationError::Malformed {
                message: format!("could not sign the registration with this node's identity: {e}"),
            })?;
        tracing::info!("📡 Registering with vram_mb={}", registration.vram_mb);

        let client = reqwest::Client::new();
        let register_url = format!("{}/register", config.orchestrator_url);
        let res = client
            .post(&register_url)
            .json(&registration)
            .send()
            .await
            .map_err(|source| RegistrationError::Unreachable {
                url: register_url.clone(),
                source,
            })?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let body = res.text().await.unwrap_or_default();
            return Err(RegistrationError::Refused { status, body });
        }

        #[derive(serde::Deserialize)]
        struct RegisterResponse {
            assignment: WorkerAssignment,
            peers: Vec<tessella_core::models::LayerPeer>,
        }

        let resp: RegisterResponse =
            res.json()
                .await
                .map_err(|source| RegistrationError::Malformed {
                    message: format!(
                        "Malformed registration response from {register_url}: {source}"
                    ),
                })?;

        info!(
            "✅ Registration successful. Assigned model={}, layers={:?}, replica_group={}",
            resp.assignment.model_repo,
            resp.assignment.layer_ranges,
            resp.assignment.replica_group_id
        );

        Ok((resp.assignment, resp.peers))
    }

    pub async fn fetch_musivum_p2p_info(config: &WorkerConfig) -> Result<MusivumRelay> {
        swarm_loop::fetch_musivum_p2p_info(config).await
    }

    pub async fn fetch_musivum_p2p_info_from(
        client: &reqwest::Client,
        orchestrator_url: &str,
    ) -> Result<MusivumRelay> {
        swarm_loop::fetch_musivum_p2p_info_from(client, orchestrator_url).await
    }

    pub fn get_real_ip(swarm: &libp2p::Swarm<MoeBehaviour>, fallback_ip: &str) -> String {
        swarm_loop::get_real_ip(swarm, fallback_ip)
    }

    pub fn real_ip_from_listeners<'a>(
        listeners: impl IntoIterator<Item = &'a libp2p::Multiaddr>,
        fallback_ip: &str,
    ) -> String {
        swarm_loop::real_ip_from_listeners(listeners, fallback_ip)
    }

    pub fn get_first_non_loopback_ip() -> Option<String> {
        swarm_loop::get_first_non_loopback_ip()
    }

    pub async fn announce_to_musivum(
        config: &WorkerConfig,
        assignment: &WorkerAssignment,
        peer_id: &str,
        real_ip: &str,
        bind_ip: &str,
        peer_network: &dyn PeerNetwork,
    ) -> Result<()> {
        membership::announce_to_musivum(config, assignment, peer_id, real_ip, bind_ip, peer_network)
            .await
    }

    pub async fn prepare_model_metadata(
        model_id: &str,
        config: &WorkerConfig,
    ) -> Result<(
        std::path::PathBuf,
        std::path::PathBuf,
        tessella_core::models::ModelConfig,
    )> {
        let token = if config.hf_token.is_empty() {
            None
        } else {
            Some(config.hf_token.as_str())
        };
        let (config_path, tokenizer_path, _) =
            hf_loader::resolve_index(model_id, token, std::path::Path::new(&config.models_dir))
                .await?;
        let model_dir = config_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid config path"))?
            .to_path_buf();
        let config_json_content = std::fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_json_content)?;
        let num_hidden_layers = config_json["num_hidden_layers"]
            .as_u64()
            .or_else(|| config_json["n_layers"].as_u64())
            .ok_or_else(|| anyhow::anyhow!("Missing num_hidden_layers in config.json"))?
            as usize;

        let mut model_config = tessella_core::models::ModelConfig::from_hf_json(&config_path)?;
        model_config.model_id = model_id.to_string();
        model_config.num_hidden_layers = num_hidden_layers;
        model_config.quantization = config.quantization;
        Ok((model_dir, tokenizer_path, model_config))
    }

    /// Downloads what every range of `assignment` needs and loads one [`ShardedWeights`] per range
    /// (`TASK-96`/S102).
    ///
    /// One download, N loads: the union of the shards the ranges need is fetched once (a node with
    /// two chunks of one model may well need the same files twice) and then filtered once per range,
    /// which is exactly what `shard_loader` already does with the range it is handed. The edge
    /// tensors follow the same rule they always did — the range that starts at 0 gets the embedding
    /// table, the range that reaches `total_layers` gets the final norm and the LM head — so a node
    /// that holds both edges loads them once in total, which is the arithmetic `layer_budget`
    /// charges.
    pub async fn acquire_weights(
        config: &WorkerConfig,
        assignment: &WorkerAssignment,
        device: &Device,
        total_layers: usize,
    ) -> Result<(ShardSet, Vec<std::path::PathBuf>)> {
        if config.canary_mode.is_on() {
            info!(
                "🐤 Canary node ({:?}): no shard is downloaded and no weight is loaded for ranges \
                 {:?} (TASK-71/S78)",
                config.canary_mode, assignment.layer_ranges
            );
            let empty = assignment
                .layer_ranges
                .iter()
                .map(|range| {
                    (
                        *range,
                        Arc::new(ShardedWeights {
                            layers: std::collections::HashMap::new(),
                            global: std::collections::HashMap::new(),
                            layer_range: *range,
                            expert_range: assignment.expert_range,
                            quantization: config.quantization,
                        }),
                    )
                })
                .collect();
            return Ok((empty, Vec::new()));
        }

        let token = if config.hf_token.is_empty() {
            None
        } else {
            Some(config.hf_token.as_str())
        };
        let (_config_path, _, _index) = hf_loader::resolve_index(
            &assignment.model_repo,
            token,
            std::path::Path::new(&config.models_dir),
        )
        .await?;
        let is_entrypoint = layer_ranges::head_range(&assignment.layer_ranges).is_some();
        let is_final = layer_ranges::tail_range(&assignment.layer_ranges, total_layers).is_some();

        info!("📥 Downloading missing shards from HuggingFace Hub...");
        let artifacts = hf_loader::resolve_shards_for_layers(
            &assignment.model_repo,
            &assignment.layer_ranges,
            is_entrypoint,
            is_final,
            token,
            &config.quantization,
            std::path::Path::new(&config.models_dir),
        )
        .await?;

        for root in model_cache::hold_shards(&artifacts.shard_paths) {
            info!(
                model_dir = %root.display(),
                "🔒 holding the shared cache lock of this model (TASK-73/S80): a neighbour's prune \
                 will not touch the shards this node serves"
            );
        }

        let mut shards: ShardSet = Vec::with_capacity(assignment.layer_ranges.len());
        for range in &assignment.layer_ranges {
            let weights = load_sharded_weights(
                &artifacts.shard_paths,
                *range,
                assignment.expert_range,
                config.quantization,
                device,
            )
            .await?;
            shards.push((*range, Arc::new(weights)));
        }

        Ok((shards, artifacts.shard_paths))
    }

    pub async fn notify_ready(
        config: &WorkerConfig,
        assignment: &WorkerAssignment,
        peer_id: &str,
        queue_depth: usize,
        peer_network: &dyn PeerNetwork,
    ) -> Result<()> {
        membership::notify_ready(config, assignment, peer_id, queue_depth, peer_network).await
    }

    pub async fn reload_shard(&self, new_assignment: &WorkerAssignment) -> Result<(), DomainError> {
        let current = { self.engine.assignment.read().await.clone() };

        if !tessella_core::assignment::requires_weight_reload(&current, new_assignment) {
            *self.engine.assignment.write().await = new_assignment.clone();
            log::debug!(
                "🔁 Assignment re-adopted without reloading the shard (layers {:?}, experts {:?}, \
                 replica group {})",
                new_assignment.layer_ranges,
                new_assignment.expert_range,
                new_assignment.replica_group_id
            );
            return Ok(());
        }

        info!(
            "🔄 Musivum assigned new layers {:?} / experts {:?}, reloading model shards \
             (was {:?} / experts {:?})...",
            new_assignment.layer_ranges,
            new_assignment.expert_range,
            current.layer_ranges,
            current.expert_range
        );

        if !self.engine.drain_sessions(Duration::from_secs(5)).await {
            return Err(DomainError::Inference(
                "Drain timeout exceeded during reload".into(),
            ));
        }

        let (new_weights, new_shards) = Self::acquire_weights(
            &self.config,
            new_assignment,
            &self.engine.device,
            self.engine.model_config.total_layers(),
        )
        .await
        .map_err(|e| DomainError::Inference(format!("Failed to acquire weights: {}", e)))?;

        {
            let mut assignment_lock = self.engine.assignment.write().await;
            *assignment_lock = new_assignment.clone();

            let mut shards_lock = self.engine.shards.write().await;
            *shards_lock = new_weights;
        }

        let tokenizer =
            if self.config.canary_mode.is_on() {
                None
            } else {
                Some(self.load_tokenizer().map_err(|e| {
                    DomainError::Inference(format!("Failed to load tokenizer: {e}"))
                })?)
            };

        self.engine
            .initialize_models(&self.model_dir, tokenizer)
            .await?;

        if self
            .engine
            .gpu_health
            .mark_recovered(tessella_core::telemetry::gpu_health::GpuRecovery::ShardReload)
        {
            info!("✅ GPU: shard reloaded — this node is serving again and leaves the down state (TASK-77).");
        }

        if let Some(outcome) =
            model_cache::prune_after_reload(self.config.model_cache_prune, &new_shards)
        {
            model_cache::log_outcome(&outcome);
        }

        info!(
            "✅ Shard reload complete for ranges {:?}",
            new_assignment.layer_ranges
        );
        Ok(())
    }

    pub fn announce_via_gossip(
        swarm: &mut libp2p::Swarm<MoeBehaviour>,
        peer_id: &str,
        config: &WorkerConfig,
        assignment: &WorkerAssignment,
        listen_addr: &libp2p::Multiaddr,
        real_ip: &str,
        bind_ip: &str,
    ) -> Result<()> {
        membership::announce_via_gossip(
            swarm,
            peer_id,
            config,
            assignment,
            listen_addr,
            real_ip,
            bind_ip,
        )
    }

    pub fn load_tokenizer(&self) -> Result<Arc<tessella_inference::TokenizerWrapper>> {
        let path = self.tokenizer_path.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "[TESERROR-07] This node holds no tokenizer: entry nodes require a tokenizer; verify HF_TOKEN or model files"
            )
        })?;
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
        Ok(Arc::new(tessella_inference::TokenizerWrapper::from_file(
            path_str,
        )?))
    }

    pub fn setup_api_state(&self, service: Arc<dyn DistributedInference>) -> ApiState {
        api_host::setup_api_state(self, service)
    }

    pub async fn remember_peers(&self, peers: &[tessella_core::models::LayerPeer]) {
        swarm_loop::remember_peers(self, peers).await;
    }

    pub async fn perform_graceful_shutdown(&self) {
        membership::perform_graceful_shutdown(self).await;
    }

    async fn shutdown_signal() {
        let ctrl_c = async {
            match signal::ctrl_c().await {
                Ok(()) => {}
                Err(e) => {
                    warn!(
                        "⚠️ Could not install the Ctrl+C handler ({e}); the worker keeps running \
                         and will still stop on SIGTERM"
                    );
                    std::future::pending::<()>().await;
                }
            }
        };
        #[cfg(unix)]
        let terminate = async {
            match signal::unix::signal(signal::unix::SignalKind::terminate()) {
                Ok(mut sig) => {
                    sig.recv().await;
                }
                Err(e) => {
                    warn!(
                        "⚠️ Could not install the SIGTERM handler ({e}); the worker keeps running \
                         and will still stop on Ctrl+C"
                    );
                    std::future::pending::<()>().await;
                }
            }
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();
        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    }
}
