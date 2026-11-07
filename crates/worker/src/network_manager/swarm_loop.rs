//! Swarm event loop, Kademlia DHT and peer event pump collaborator for NetworkManager (`SOLID-90`).

use anyhow::Result;
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tessella_core::error::DomainError;
use tessella_core::traits::PipelineRouter;
use tessella_network::kademlia::{self, DiscoverySource};
use tessella_network::peer_network::SwarmCommand;
use tessella_network::{GossipTopic, MoeBehaviour, MoeBehaviourEvent, SwarmEvent};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing as log;
use tracing::{error, info, warn};

use super::NetworkManager;
use crate::heartbeat::{
    announce_outcome, preview, registration_outcome, AnnounceAnswer, AnnounceOutcome,
    HeartbeatAction, HeartbeatHealth, HeartbeatNotice,
};
use crate::membership::GossipMembership;
use crate::peer_discovery::{plan_seeds, select_dialable_providers};

/// DHT state kept by this worker node (`TASK-15` · H-06).
pub struct KadState {
    pub enabled: bool,
    pub bootstrapped: bool,
    pub provided: HashSet<String>,
    pub pending_providers: HashMap<libp2p::kad::QueryId, String>,
    pub last_bootstrap_attempt: Option<std::time::Instant>,
}

impl KadState {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            bootstrapped: false,
            provided: HashSet::new(),
            pending_providers: HashMap::new(),
            last_bootstrap_attempt: None,
        }
    }

    pub fn bootstrap_due(&self) -> bool {
        match self.last_bootstrap_attempt {
            None => true,
            Some(t) => t.elapsed() >= Duration::from_secs(60),
        }
    }

    pub fn note_bootstrap_attempt(&mut self) {
        self.last_bootstrap_attempt = Some(std::time::Instant::now());
    }
}

use std::sync::atomic::{AtomicBool, Ordering};
use tessella_network::{
    create_worker_swarm, generate_keypair, keypair_from_bytes, keypair_to_bytes,
    peer_id_from_keypair, NetworkAddress, NetworkKeypair,
};

/// Orchestrator relay endpoint, as advertised by `GET /p2p_id`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MusivumRelay {
    pub peer_id: Option<String>,
    pub p2p_addr: Option<libp2p::Multiaddr>,
    pub pubkey: Option<String>,
}

impl MusivumRelay {
    pub fn from_payload(json: &serde_json::Value) -> Result<Self> {
        let peer_id = json["peer_id"]
            .as_str()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Invalid musivum PeerId response"))?;

        let p2p_addr = match json["p2p_addr"].as_str().map(str::trim) {
            Some(addr) if !addr.is_empty() => match addr.parse::<libp2p::Multiaddr>() {
                Ok(parsed) => Some(parsed),
                Err(e) => {
                    warn!(
                        "⚠️ Invalid musivum P2P address '{}' from /p2p_id: {}",
                        addr, e
                    );
                    None
                }
            },
            _ => None,
        };

        Ok(Self {
            peer_id: Some(peer_id.to_string()),
            p2p_addr,
            pubkey: json["orchestrator_pubkey"]
                .as_str()
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .map(str::to_string),
        })
    }

    pub fn needs_addr(&self) -> bool {
        self.p2p_addr.is_none()
    }

    pub fn is_peer(&self, peer_id: &str) -> bool {
        self.peer_id.as_deref() == Some(peer_id)
    }

    pub fn update(&mut self, fresh: Self) -> Option<libp2p::Multiaddr> {
        let previous = self.p2p_addr.clone();
        if let Some(peer_id) = fresh.peer_id {
            self.peer_id = Some(peer_id);
        }
        if let Some(addr) = fresh.p2p_addr {
            self.p2p_addr = Some(addr);
        }
        if let Some(pubkey) = fresh.pubkey {
            self.pubkey = Some(pubkey);
        }
        if self.p2p_addr != previous {
            self.p2p_addr.clone()
        } else {
            None
        }
    }

    pub fn forget_addr(&mut self) -> bool {
        self.p2p_addr.take().is_some()
    }
}

static IDENTITY_CONFLICT_WARNED: AtomicBool = AtomicBool::new(false);

pub fn orchestrator_key_conflict(pinned: Option<&str>, observed: Option<&str>) -> bool {
    match (pinned, observed) {
        (Some(pinned), Some(observed)) => !pinned.trim().eq_ignore_ascii_case(observed.trim()),
        _ => false,
    }
}

pub fn warn_on_identity_conflict(
    pinned: Option<&str>,
    observed: Option<&str>,
    orchestrator_url: &str,
) {
    if !orchestrator_key_conflict(pinned, observed) {
        return;
    }
    if IDENTITY_CONFLICT_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    warn!(
        "🚨 The orchestrator at {} is NOT the one pinned in `orchestrator_public_key` (pinned {}, \
         observed {}). Possible impersonation, or an orchestrator whose identity changed. The \
         pinned key is kept, so KarmaTickets signed by the observed peer are ignored and every \
         request is served as Leech; update the configuration if the change was intentional \
         (copy the new 'orchestrator_pubkey' from GET /p2p_id)",
        orchestrator_url,
        pinned.unwrap_or("<unset>"),
        observed.unwrap_or("<not published>")
    );
}

pub async fn fetch_musivum_p2p_info(config: &crate::config::WorkerConfig) -> Result<MusivumRelay> {
    fetch_musivum_p2p_info_from(&reqwest::Client::new(), &config.orchestrator_url).await
}

pub async fn fetch_musivum_p2p_info_from(
    client: &reqwest::Client,
    orchestrator_url: &str,
) -> Result<MusivumRelay> {
    let res = client
        .get(format!("{}/p2p_id", orchestrator_url))
        .send()
        .await?;
    if !res.status().is_success() {
        anyhow::bail!("/p2p_id returned status {}", res.status());
    }
    let json: serde_json::Value = res.json().await?;
    MusivumRelay::from_payload(&json)
}

pub fn get_real_ip(swarm: &libp2p::Swarm<MoeBehaviour>, fallback_ip: &str) -> String {
    real_ip_from_listeners(swarm.listeners(), fallback_ip)
}

pub fn real_ip_from_listeners<'a>(
    listeners: impl IntoIterator<Item = &'a libp2p::Multiaddr>,
    fallback_ip: &str,
) -> String {
    listeners
        .into_iter()
        .filter_map(|addr| {
            for proto in addr.iter() {
                if let libp2p::multiaddr::Protocol::Ip4(ip) = proto {
                    if !ip.is_loopback() {
                        return Some(ip.to_string());
                    }
                }
            }
            None
        })
        .next()
        .unwrap_or_else(|| fallback_ip.to_string())
}

pub fn get_first_non_loopback_ip() -> Option<String> {
    local_ip_address::local_ip()
        .ok()
        .filter(|ip| !ip.is_loopback())
        .map(|ip| ip.to_string())
}

pub fn init_p2p(
    config: &crate::config::WorkerConfig,
    key_path: &str,
) -> Result<(libp2p::Swarm<MoeBehaviour>, String, NetworkKeypair)> {
    let keypair = if std::path::Path::new(key_path).exists() {
        let bytes = std::fs::read(key_path)?;
        keypair_from_bytes(&bytes).unwrap_or_else(|_| generate_keypair())
    } else {
        let kp = generate_keypair();
        let bytes = keypair_to_bytes(&kp)?;
        let _ = std::fs::write(key_path, bytes);
        kp
    };

    let peer_id = peer_id_from_keypair(&keypair);
    let mut swarm = create_worker_swarm(
        keypair.clone(),
        config.mdns_enabled,
        config.kademlia_enabled,
    )?;

    if let (Some(public_ip), Some(public_p2p_port)) = (&config.public_ip, config.public_p2p_port) {
        match tessella_core::net::multiaddr_tcp(public_ip, public_p2p_port) {
            Some(ext_addr_str) => {
                let ext_addr = ext_addr_str.parse::<libp2p::Multiaddr>()?;
                swarm.add_external_address(ext_addr.clone());
                log::info!(
                    "🌍 Explicitly announcing public external address to P2P network: {}",
                    ext_addr
                );
            }
            None => warn!(
                "⚠️ public_ip '{}' is not a literal IP address: not announcing an external \
                 P2P address",
                public_ip
            ),
        }
    }

    let listen_addr_str = if config.public_ip.is_some() && config.public_p2p_port.is_some() {
        tessella_core::net::multiaddr_tcp(&config.p2p_host, config.p2p_port).ok_or_else(|| {
            anyhow::anyhow!("p2p_host '{}' is not a literal IP address", config.p2p_host)
        })?
    } else {
        tessella_core::net::multiaddr_tcp("0.0.0.0", config.listen_port)
            .ok_or_else(|| anyhow::anyhow!("0.0.0.0 must be a literal IP address, but is not"))?
    };
    let listen_addr = listen_addr_str.parse::<NetworkAddress>()?;
    swarm.listen_on(listen_addr)?;

    Ok((swarm, peer_id, keypair))
}

pub fn dial_musivum_relay(swarm: &mut libp2p::Swarm<MoeBehaviour>, addr: &libp2p::Multiaddr) {
    info!("📡 Dialing musivum for relay: {}", addr);
    let _ = swarm.dial(addr.clone());

    let relay_listen_addr = addr.clone().with(libp2p::multiaddr::Protocol::P2pCircuit);
    info!("👂 Listening on relay circuit: {}", relay_listen_addr);
    if let Err(e) = swarm.listen_on(relay_listen_addr) {
        warn!("⚠️ Failed to listen on relay circuit: {}", e);
    }
}

pub async fn rediscover_musivum_relay(
    network: &NetworkManager,
    swarm: &mut libp2p::Swarm<MoeBehaviour>,
) {
    match NetworkManager::fetch_musivum_p2p_info(&network.config).await {
        Ok(fresh) => {
            let observed_key = fresh.pubkey.clone();
            let dialable = network.musivum_relay.write().await.update(fresh);
            super::warn_on_identity_conflict(
                network.config.orchestrator_public_key.as_deref(),
                observed_key.as_deref(),
                &network.config.orchestrator_url,
            );
            if let Some(addr) = dialable {
                info!("🔄 Re-discovered musivum relay: {}", addr);
                dial_musivum_relay(swarm, &addr);
            }
        }
        Err(e) => warn!(
            "⚠️ Musivum P2P re-discovery via /p2p_id failed ({}); retrying on the next heartbeat (HTTP routing remains active)",
            e
        ),
    }
}

pub async fn remember_peers(network: &NetworkManager, peers: &[tessella_core::models::LayerPeer]) {
    if peers.is_empty() {
        return;
    }
    match network
        .storage
        .remember_known_peers(peers, tessella_core::time::unix_now_secs())
        .await
    {
        Ok(stored) => log::debug!("🧭 Peer cache: {} address(es) remembered", stored),
        Err(e) => warn!("⚠️ Could not persist the peer cache: {}", e),
    }
}

pub async fn seed_kademlia(
    network: &NetworkManager,
    swarm: &mut libp2p::Swarm<MoeBehaviour>,
    kad_state: &mut KadState,
) {
    if !kad_state.bootstrap_due() {
        return;
    }
    let now = tessella_core::time::unix_now_secs();
    let cached = match network
        .storage
        .load_known_peers(now, tessella_storage::KNOWN_PEERS_TTL_SECS)
        .await
    {
        Ok(peers) => peers,
        Err(e) => {
            warn!(
                "⚠️ Could not read the peer cache ({}) — bootstrapping the DHT with the live mesh view only",
                e
            );
            Vec::new()
        }
    };
    if network.routing_table.read().await.peers().is_empty() && !cached.is_empty() {
        let peers = cached.clone();
        info!(
            "📂 Seeding the routing table with {} peer(s) from the persistent cache (the tracker has not answered)",
            peers.len()
        );
        network.routing_table.write().await.replace_all(peers);
    }
    let mesh: Vec<tessella_core::models::LayerPeer> =
        network.routing_table.read().await.peers().to_vec();

    let mut plan = plan_seeds(&network.runtime.peer_id, &mesh, &cached);
    let relay = network.musivum_relay.read().await.clone();
    let relay_seed = match (relay.p2p_addr.clone(), relay.peer_id.clone()) {
        (Some(addr), Some(relay_peer)) if relay_peer != network.runtime.peer_id => {
            relay_peer.parse().ok().map(|peer_id| kademlia::PeerSeed {
                source: DiscoverySource::Tracker,
                peer_id,
                addr,
            })
        }
        _ => None,
    };
    if let Some(seed) = relay_seed {
        plan.seeds.push(seed);
    }

    if plan.is_empty() {
        kad_state.note_bootstrap_attempt();
        warn!(
            "⚠️ Kademlia bootstrap: no seed available ({} unusable candidate(s)). The mesh view \
             is only whatever HTTP discovered; this will be retried on the next heartbeat",
            plan.unusable
        );
        return;
    }

    for seed in &plan.seeds {
        swarm
            .behaviour_mut()
            .kademlia
            .add_address(&seed.peer_id, seed.addr.clone());
    }
    kad_state.note_bootstrap_attempt();
    match swarm.behaviour_mut().kademlia.bootstrap() {
        Ok(_) => info!(
            "🧭 Kademlia bootstrap started with {} seed(s) ({} unusable candidate(s))",
            plan.seeds.len(),
            plan.unusable
        ),
        Err(e) => warn!("⚠️ Kademlia bootstrap refused: {}", e),
    }
}

pub fn provide_model(
    swarm: &mut libp2p::Swarm<MoeBehaviour>,
    kad_state: &mut KadState,
    model_id: &str,
) {
    if !kad_state.enabled || !kad_state.provided.insert(model_id.to_string()) {
        return;
    }
    let key = kademlia::model_record_key(model_id);
    match swarm.behaviour_mut().kademlia.start_providing(key) {
        Ok(_) => info!(
            "🧭 Kademlia: announcing this node as a provider of {} (record re-published every 15 min)",
            model_id
        ),
        Err(e) => {
            kad_state.provided.remove(model_id);
            warn!(
                "⚠️ Kademlia could not announce {} as a provided model: {}",
                model_id, e
            );
        }
    }
}

pub fn query_model_providers(
    swarm: &mut libp2p::Swarm<MoeBehaviour>,
    kad_state: &mut KadState,
    model_id: &str,
) {
    if !kad_state.enabled {
        return;
    }
    let key = kademlia::model_record_key(model_id);
    let query_id = swarm.behaviour_mut().kademlia.get_providers(key);
    kad_state
        .pending_providers
        .insert(query_id, model_id.to_string());
    info!(
        "🧭 Kademlia: asking the DHT who serves {} (HTTP discovery has nothing to offer)",
        model_id
    );
}

pub async fn run_p2p_loop(
    network: Arc<NetworkManager>,
    mut swarm: libp2p::Swarm<MoeBehaviour>,
    mut cmd_rx: mpsc::Receiver<SwarmCommand>,
    cancel_token: CancellationToken,
) -> Result<()> {
    let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(
        network.runtime.config.heartbeat_interval_secs,
    ));

    super::warn_on_identity_conflict(
        network.config.orchestrator_public_key.as_deref(),
        network.musivum_relay.read().await.pubkey.as_deref(),
        &network.config.orchestrator_url,
    );
    let advertised = network.musivum_relay.read().await.p2p_addr.clone();
    match advertised {
        Some(addr) => dial_musivum_relay(&mut swarm, &addr),
        None => info!(
            "ℹ️ No dialable musivum P2P address advertised; skipping relay (HTTP routing remains active)"
        ),
    }

    let mut kad_state = KadState::new(network.config.kademlia_enabled);
    let mut membership = GossipMembership::new();
    let mut heartbeat_health = HeartbeatHealth::new();
    if kad_state.enabled {
        seed_kademlia(&network, &mut swarm, &mut kad_state).await;
        let model_id = network.engine.model_config.model_id.clone();
        provide_model(&mut swarm, &mut kad_state, &model_id);
    } else {
        info!("ℹ️ Kademlia disabled (kademlia_enabled = false): HTTP discovery only");
    }

    loop {
        tokio::select! {
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(SwarmCommand::Publish { topic, data, respond_to }) => {
                        let gossip_topic = GossipTopic::new(topic);
                        let res = swarm.behaviour_mut().gossipsub.publish(gossip_topic, data)
                            .map(|_| ())
                            .map_err(|e| DomainError::Network(e.to_string()));
                        if let Some(resp) = respond_to {
                            let _ = resp.send(res);
                        }
                    }
                    Some(SwarmCommand::Subscribe { topic, respond_to }) => {
                        let gossip_topic = GossipTopic::new(topic);
                        let res = swarm.behaviour_mut().gossipsub.subscribe(&gossip_topic)
                            .map(|_| ())
                            .map_err(|e| DomainError::Network(e.to_string()));
                        if let Some(resp) = respond_to {
                            let _ = resp.send(res);
                        }
                    }
                    Some(SwarmCommand::Dial { addr, respond_to }) => {
                        let res = match addr.parse::<libp2p::Multiaddr>() {
                            Ok(multiaddr) => swarm.dial(multiaddr)
                                .map_err(|e| DomainError::Network(e.to_string())),
                            Err(e) => Err(DomainError::Network(format!("invalid multiaddr: {e}"))),
                        };
                        if let Some(resp) = respond_to {
                            let _ = resp.send(res);
                        }
                    }
                    None => {
                        log::debug!("Swarm command channel closed");
                    }
                }
            }
            _ = heartbeat_interval.tick() => {
                let now = std::time::Instant::now();
                let forgotten = membership.expire(now);
                if !forgotten.is_empty() {
                    log::debug!(
                        "📨 Membership: forgot {} observation(s) of unadmitted peer(s): {}",
                        forgotten.len(),
                        forgotten.join(", ")
                    );
                }
                let eligible_again = network.routing_table.write().await.expire_suppressions(now);
                if !eligible_again.is_empty() {
                    info!(
                        "▶️ Membership: {} peer(s) are eligible for routing again after the \
                         suppression window",
                        eligible_again.len()
                    );
                }
                let relay_needs_addr = network.musivum_relay.read().await.needs_addr();
                if relay_needs_addr {
                    rediscover_musivum_relay(&network, &mut swarm).await;
                }

                if kad_state.enabled {
                    if !kad_state.bootstrapped {
                        seed_kademlia(&network, &mut swarm, &mut kad_state).await;
                    }
                    if network.routing_table.read().await.peers().is_empty() {
                        let model_id = network.engine.model_config.model_id.clone();
                        query_model_providers(&mut swarm, &mut kad_state, &model_id);
                    }
                }

                if let Err(probe_error) = network.engine.revalidate_gpu() {
                    log::debug!(
                        "⚠️ GPU: this node is still down — the device does not answer its probe \
                         ({}). It revalidates on every heartbeat.",
                        tessella_core::telemetry::gpu_health::bounded(&probe_error)
                    );
                }

                let assignment = network.engine.assignment.read().await.clone();
                let queue_depth = network.engine.sessions.len();
                let mut announce_req = tessella_core::AnnounceRequest {
                    peer_id: network.runtime.peer_id.clone(),
                    host: network.config.public_ip.clone().unwrap_or_else(|| network.runtime.bind_ip.clone()),
                    port: network.config.public_p2p_port.unwrap_or(network.runtime.config.p2p_port),
                    vram_mb: if let Ok(val) = std::env::var("TESSELLA_VRAM_MB") {
                        val.parse::<u32>().unwrap_or((network.runtime.config.vram_budget_gb * 1024.0) as u32)
                    } else {
                        (network.runtime.config.vram_budget_gb * 1024.0) as u32
                    },
                    model_id: assignment.model_repo.clone(),
                    current_role: tessella_core::models::Role::Worker {
                        layer_ranges: assignment.layer_ranges.clone(),
                        replica_group_id: assignment.replica_group_id,
                        expert_range: assignment.expert_range,
                    },
                    p2p_address: network.runtime.p2p_addr.clone(),
                    http_api_url: Some(
                        network.config.resolve_http_api_url(&network.runtime.bind_ip),
                    ),
                    relay_address: None,
                    status: Some(if network.engine.gpu_health.is_healthy() && !network.engine.gpu_health.is_saturated() {
                        tessella_core::models::PeerStatus::Healthy
                    } else {
                        tessella_core::models::PeerStatus::Degraded
                    }),
                    queue_depth,
                    expert_usage: Some(network.engine.expert_activations.drain()),
                    parameter_contract_hash: tessella_protocol::swarm_contract_hash(
                        &assignment.model_repo,
                    ),
                    public_ip: network.config.public_ip.clone(),
                    public_api_port: network.config.public_api_port,
                    public_key: Vec::new(),
                    signature: Vec::new(),
                };

                if let Err(e) = network.peer_network.sign_peer_request(&mut announce_req) {
                    warn!("⚠️ Could not sign announce request: {e}");
                }
                let announce_res = network.runtime.http_client
                    .post(format!("{}/announce", network.runtime.orchestrator_url))
                    .json(&announce_req)
                    .send()
                    .await;

                let outcome = match announce_res {
                    Ok(resp) => {
                        let outcome =
                            announce_outcome(AnnounceAnswer::Status(resp.status().as_u16()));
                        match outcome {
                            AnnounceOutcome::Refused { status, .. } => AnnounceOutcome::Refused {
                                status,
                                detail: Some(preview(&resp.text().await.unwrap_or_default())),
                            },
                            other => other,
                        }
                    }
                    Err(e) => AnnounceOutcome::Unreachable {
                        detail: Some(preview(&e.to_string())),
                    },
                };

                if let Some(notice) = heartbeat_health.observe(&outcome) {
                    match notice {
                        HeartbeatNotice::Recovered { .. } => {
                            info!("✅ Heartbeat: {}", notice.message(&outcome))
                        }
                        HeartbeatNotice::Lost { .. } => {
                            warn!("⚠️ Heartbeat: {}", notice.message(&outcome))
                        }
                    }
                }

                match outcome.action() {
                    HeartbeatAction::Continue | HeartbeatAction::ReportOnly => {}
                    HeartbeatAction::ReRegister => {
                        info!("🔁 Re-registering this node: {}", outcome.describe());
                        let p2p_addr_str = if let (Some(pub_ip), Some(pub_port)) = (&network.config.public_ip, network.config.public_p2p_port) {
                            tessella_core::net::multiaddr_tcp(pub_ip, pub_port)
                        } else {
                            tessella_core::net::multiaddr_tcp(&network.runtime.bind_ip, network.runtime.config.p2p_port)
                        };
                        let p2p_addr = p2p_addr_str.and_then(|addr| addr.parse::<libp2p::Multiaddr>().ok());
                        match p2p_addr {
                            None => warn!(
                                "⚠️ Cannot re-register: this node has no usable P2P address to \
                                 announce (public_ip/public_p2p_port and the bind address must be \
                                 literal IPs). The next heartbeat will try again."
                            ),
                            Some(p2p_addr) => {
                                match NetworkManager::register_and_assign(&network.config, network.peer_network.as_ref(), &network.runtime.peer_id, &p2p_addr, &network.runtime.bind_ip, Some(network.engine.model_config.clone())).await {
                                    Ok((new_assignment, peers)) => {
                                        if let Err(e) = network.reload_shard(&new_assignment).await {
                                            error!("❌ Critical: Failed to reload shard after re-registration: {}", e);
                                            std::process::exit(1);
                                        }

                                        network.routing_table.write().await.replace_all(peers);

                                        let qd = network.engine.sessions.len();
                                        if let Err(e) = NetworkManager::notify_ready(&network.config, &new_assignment, &network.runtime.peer_id, qd, network.peer_network.as_ref()).await {
                                            warn!(
                                                "⚠️ Re-registered, but the tracker did not accept \
                                                 this node's /ready ({}): it stays out of the \
                                                 ready set until the next re-registration.",
                                                e
                                            );
                                        }
                                        info!("✅ Re-registration complete. Assignment: {:?}", new_assignment.layer_ranges);

                                        let healthy = AnnounceOutcome::Healthy;
                                        if let Some(notice) = heartbeat_health.observe(&healthy) {
                                            info!("✅ Heartbeat: {}", notice.message(&healthy));
                                        }
                                    }
                                    Err(e) => {
                                        let failure = registration_outcome(&e);
                                        if let Some(notice) = heartbeat_health.observe(&failure) {
                                            warn!("⚠️ Heartbeat: {}", notice.message(&failure));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ = cancel_token.cancelled() => break,
            event = swarm.select_next_some() => {
                handle_swarm_event(&network, event, &mut swarm, &mut kad_state, &mut membership).await;
            }
        }
    }

    let connected: Vec<libp2p::PeerId> = swarm.connected_peers().cloned().collect();
    if !connected.is_empty() {
        info!(
            "🛑 Closing {} P2P connection(s) before shutdown",
            connected.len()
        );
    }
    for peer in connected {
        let _ = swarm.disconnect_peer_id(peer);
    }
    drop(swarm);
    info!("🛑 P2P loop stopped");
    Ok(())
}

async fn handle_kademlia_event(
    network: &NetworkManager,
    event: libp2p::kad::Event,
    swarm: &mut libp2p::Swarm<MoeBehaviour>,
    kad_state: &mut KadState,
) {
    use libp2p::kad;
    match event {
        kad::Event::OutboundQueryProgressed { id, result, .. } => match result {
            kad::QueryResult::Bootstrap(Ok(outcome))
                if outcome.num_remaining == 0 && !kad_state.bootstrapped =>
            {
                kad_state.bootstrapped = true;
                info!(
                    "🧭 Kademlia: the DHT routing table is bootstrapped (last hop through {})",
                    outcome.peer
                );
            }
            kad::QueryResult::Bootstrap(Ok(_)) => {}
            kad::QueryResult::Bootstrap(Err(e)) => {
                warn!("⚠️ Kademlia bootstrap incomplete: {:?}", e)
            }
            kad::QueryResult::StartProviding(Err(e)) => {
                warn!("⚠️ Kademlia provider announcement failed: {:?}", e)
            }
            kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
                providers,
                ..
            })) => {
                let model_id = kad_state.pending_providers.remove(&id);
                let providers: Vec<String> =
                    providers.into_iter().map(|peer| peer.to_string()).collect();
                let mesh = network.routing_table.read().await.peers().to_vec();
                let orchestrator = network.musivum_relay.read().await.peer_id.clone();
                let plan = select_dialable_providers(
                    &providers,
                    &network.runtime.peer_id,
                    orchestrator.as_deref(),
                    &mesh,
                );
                if plan.unauthorized > 0 {
                    warn!(
                        "⚠️ Kademlia: {} provider record(s) refused — the mesh never admitted them, so the DHT does not get to onboard them (H-06 mitigation)",
                        plan.unauthorized
                    );
                }
                if plan.without_address > 0 {
                    log::debug!(
                        "🧭 Kademlia: {} admitted provider(s) have no dialable address yet",
                        plan.without_address
                    );
                }
                for candidate in &plan.dialable {
                    info!(
                        "🧭 Kademlia: dialing {} (announced as a provider of {})",
                        candidate.peer_id,
                        model_id.as_deref().unwrap_or("the model")
                    );
                    if let Err(e) = swarm.dial(candidate.addr.clone()) {
                        warn!("⚠️ Kademlia: could not dial {}: {}", candidate.peer_id, e);
                    }
                }
                if plan.is_empty() {
                    info!(
                        "🧭 Kademlia: the DHT named {} provider(s) for {} but none is dialable yet",
                        providers.len(),
                        model_id.as_deref().unwrap_or("the model")
                    );
                }
            }
            kad::QueryResult::GetProviders(Ok(
                kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
            )) => {
                let model_id = kad_state.pending_providers.remove(&id);
                info!(
                    "🧭 Kademlia: no provider of {} was found in the DHT",
                    model_id.as_deref().unwrap_or("the model")
                );
            }
            kad::QueryResult::GetProviders(Err(e)) => {
                kad_state.pending_providers.remove(&id);
                warn!("⚠️ Kademlia provider lookup failed: {:?}", e);
            }
            _ => {}
        },
        kad::Event::RoutingUpdated {
            peer, is_new_peer, ..
        } => log::debug!(
            "🧭 Kademlia: routing table updated ({}peer {})",
            if is_new_peer { "new " } else { "known " },
            peer
        ),
        kad::Event::InboundRequest { request } => {
            log::debug!("🧭 Kademlia: inbound request served ({:?})", request)
        }
        _ => {}
    }
}

async fn handle_swarm_event(
    network: &NetworkManager,
    event: SwarmEvent<MoeBehaviourEvent>,
    swarm: &mut libp2p::Swarm<MoeBehaviour>,
    kad_state: &mut KadState,
    membership: &mut GossipMembership,
) {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            info!("📡 Worker listening on {:?}", address)
        }
        SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
            warn!("🔌 Connection closed to {:?}: {:?}", peer_id, cause);
            network.runtime.peer_failures.record_failure(
                &peer_id.to_string(),
                tessella_core::telemetry::metrics::PeerFailureReason::ConnectionClosed,
            );
            let forgot_relay = {
                let mut relay = network.musivum_relay.write().await;
                relay.is_peer(&peer_id.to_string()) && relay.forget_addr()
            };
            if forgot_relay {
                info!(
                    "🔄 Lost the musivum relay connection; re-discovery scheduled for the next heartbeat tick"
                );
            }
        }
        SwarmEvent::OutgoingConnectionError {
            peer_id: Some(pid), ..
        } => {
            network.runtime.peer_failures.record_failure(
                &pid.to_string(),
                tessella_core::telemetry::metrics::PeerFailureReason::DialFailed,
            );
        }
        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
            let peer_str = peer_id.to_string();
            let in_routing_table = network.routing_table.read().await.contains_peer(&peer_str);
            let orchestrator = network.musivum_relay.read().await.peer_id.clone();
            let authorized = kademlia::peer_is_authorized(
                &peer_str,
                &network.runtime.peer_id,
                orchestrator.as_deref(),
                in_routing_table,
            );

            if authorized {
                info!("🔗 Peer authorized: {:?}", peer_id);
            } else {
                warn!("Rejected connection from unregistered peer: {}", peer_str);
                let _ = swarm.disconnect_peer_id(peer_id);
            }
        }
        SwarmEvent::Behaviour(MoeBehaviourEvent::Identify(
            tessella_network::events::IdentifyEvent::Received { peer_id, info },
        )) => {
            let learned = info.listen_addrs.len();
            for addr in info.listen_addrs {
                swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
            }
            if learned > 0 {
                log::debug!(
                    "🧭 Kademlia: {} address(es) learned for {} through identify",
                    learned,
                    peer_id
                );
            }
            if kad_state.enabled
                && !kad_state.bootstrapped
                && kad_state.bootstrap_due()
                && swarm.behaviour_mut().kademlia.bootstrap().is_ok()
            {
                kad_state.note_bootstrap_attempt();
                info!("🧭 Kademlia bootstrap started after identify {}", peer_id);
            }
        }
        SwarmEvent::Behaviour(MoeBehaviourEvent::Kademlia(event)) => {
            handle_kademlia_event(network, event, swarm, kad_state).await;
        }
        SwarmEvent::Behaviour(MoeBehaviourEvent::Gossipsub(
            tessella_network::events::GossipsubEvent::Message {
                propagation_source: _,
                message_id: _,
                message,
            },
        )) => {
            let expected_topic = GossipTopic::new(format!(
                "/musivum/topology/{}",
                network.engine.model_config.model_id
            ));
            if message.topic == expected_topic.hash() {
                info!("📩 Received Gossipsub topology update");
                match rkyv::check_archived_root::<tessella_protocol::GossipTopology>(&message.data)
                {
                    Ok(archived) => {
                        match rkyv::Deserialize::<tessella_protocol::GossipTopology, _>::deserialize(
                            archived,
                            &mut rkyv::Infallible,
                        ) {
                            Ok(topology) => {
                                let current_canaries: std::collections::HashSet<String> = network
                                    .routing_table
                                    .read()
                                    .await
                                    .peers()
                                    .iter()
                                    .filter(|p| p.canary)
                                    .map(|p| p.peer_id.clone())
                                    .collect();

                                let mut layer_peers: Vec<tessella_core::models::LayerPeer> =
                                    topology
                                        .nodes
                                        .iter()
                                        .map(crate::peer_discovery::layer_peer_from_node_info)
                                        .collect();

                                for peer in &mut layer_peers {
                                    if current_canaries.contains(&peer.peer_id) {
                                        peer.canary = true;
                                    }
                                }

                                network
                                    .topology_cache
                                    .write()
                                    .await
                                    .update(layer_peers.clone());

                                remember_peers(network, &layer_peers).await;
                                network.routing_table.write().await.replace_all(layer_peers);

                                network.topology_notification.notify_waiters();
                                info!("🔄 Local routing table and topology cache dynamically updated from Gossipsub");
                            }
                            Err(e) => {
                                error!("Failed to deserialize GossipTopology: {:?}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed check_archived_root for GossipTopology: {:?}", e);
                    }
                }
            } else if message.topic == GossipTopic::new(tessella_network::WORKERS_TOPIC).hash() {
                super::membership::handle_membership_gossip(
                    network,
                    message,
                    membership,
                    std::time::Instant::now(),
                )
                .await;
            }
        }
        _ => {}
    }
}

pub async fn run_topology_refresh(network: Arc<NetworkManager>, cancel_token: CancellationToken) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    while crate::shutdown::tick_or_cancel(&mut interval, &cancel_token).await {
        let model_id = network.engine.model_config.model_id.clone();
        info!("🔄 Background topology refresh for {}...", model_id);

        let res = network
            .runtime
            .http_client
            .get(format!(
                "{}/peers?model_id={}",
                network.runtime.orchestrator_url, model_id
            ))
            .send()
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<Vec<tessella_core::models::PeersEntry>>().await {
                    Ok(entries) => {
                        let peers: Vec<tessella_core::models::LayerPeer> = entries
                            .iter()
                            .map(crate::peer_discovery::layer_peer_from_peers_entry)
                            .collect();
                        remember_peers(&network, &peers).await;
                        network.topology_cache.write().await.update(peers.clone());
                        network.routing_table.write().await.replace_all(peers);

                        info!(
                            "✅ Background topology refresh successful ({} peers)",
                            entries.len()
                        );
                    }
                    Err(e) => {
                        warn!("⚠️ Failed to deserialize peers from /peers response: {}", e);
                    }
                }
            }
            Ok(resp) => {
                warn!(
                    "⚠️ Background topology refresh failed: /peers returned {}",
                    resp.status()
                );
            }
            Err(e) => {
                warn!("⚠️ Background topology refresh request error: {}", e);
            }
        }
    }
}

pub async fn run_refresh_loop(
    network: Arc<NetworkManager>,
    cancel_token: CancellationToken,
) -> Result<(), DomainError> {
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    let mut topology_interval = tokio::time::interval(Duration::from_secs(60));

    let router = crate::http_pipeline_router::HttpPipelineRouter::new(
        network.routing_table.clone(),
        network.topology_cache.clone(),
        network.runtime.http_client.clone(),
        network.runtime.orchestrator_url.clone(),
        network.runtime.peer_id.clone(),
        network.engine.model_config.model_id.clone(),
        network.runtime.peer_failures.clone(),
        network.topology_notification.clone(),
        network.runtime.keypair.clone(),
        network.runtime.hop_wait,
    );

    loop {
        tokio::select! {
            _ = interval.tick() => {
                if let Err(e) = router.refresh().await {
                    tracing::warn!("Failed to refresh routing table: {}", e);
                }
            }
            _ = topology_interval.tick() => {
                refresh_topology_cache(&network).await;
            }
            _ = cancel_token.cancelled() => {
                break;
            }
        }
    }
    Ok(())
}

pub async fn refresh_topology_cache(network: &NetworkManager) {
    let model_id = network.engine.model_config.model_id.clone();
    let url = format!(
        "{}/peers?model_id={}",
        network.runtime.orchestrator_url, model_id
    );
    match network.runtime.http_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Vec<tessella_core::models::PeersEntry>>().await {
                Ok(entries) => {
                    let peers: Vec<tessella_core::models::LayerPeer> = entries
                        .iter()
                        .map(crate::peer_discovery::layer_peer_from_peers_entry)
                        .collect();
                    let count = peers.len();
                    remember_peers(network, &peers).await;
                    network.topology_cache.write().await.update(peers);
                    tracing::debug!("🔄 topology cache refreshed ({} peers)", count);
                }
                Err(e) => {
                    tracing::warn!(
                        "⚠️ Topology cache refresh: /peers answered 200 but its body did not \
                         parse ({}). Serving from the stale cache.",
                        e
                    );
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                "Failed to refresh topology cache: {}. Serving from stale cache.",
                e
            );
        }
        Ok(resp) if resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE => {
            tracing::debug!("⏳ pipeline not yet complete, will retry topology refresh");
        }
        _ => {
            tracing::warn!(
                "Failed to refresh topology cache: Musivum returned non-success status."
            );
        }
    }
}
