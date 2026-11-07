//! Membership gossip handling and graceful leave collaborator for NetworkManager (`SOLID-90`).

use std::time::Instant;
use tessella_protocol::{DeregisterRequest, WorkerAssignment};
use tracing::{info, warn};

use super::NetworkManager;
use crate::membership::{
    Applied, GossipMembership, MembershipAction, MembershipChannels, MembershipDrop,
};

/// Handles one message from the peer membership channel (`TASK-16` · S22).
pub async fn handle_membership_gossip(
    network: &NetworkManager,
    message: tessella_network::GossipsubMessage,
    membership: &mut GossipMembership,
    now: Instant,
) {
    let Ok(msg) = serde_json::from_slice::<tessella_core::models::WorkerGossipMsg>(&message.data)
    else {
        warn!(
            "⚠️ Membership gossip: unreadable payload ({} bytes) from {:?}",
            message.data.len(),
            message.source
        );
        return;
    };

    let source = message.source.as_ref().map(|peer| peer.to_string());
    let orchestrator = network.musivum_relay.read().await.peer_id.clone();
    let action = {
        let table = network.routing_table.read().await;
        let channels = MembershipChannels {
            own_peer_id: &network.runtime.peer_id,
            orchestrator_peer_id: orchestrator.as_deref(),
            routing_table: &table,
        };
        match &msg {
            tessella_core::models::WorkerGossipMsg::WorkerJoined { .. } => {
                membership.observe_joined(&msg, source.as_deref(), &channels, now)
            }
            tessella_core::models::WorkerGossipMsg::WorkerLeft { .. } => {
                membership.observe_left(&msg, source.as_deref(), &channels, now)
            }
        }
    };

    match &action {
        MembershipAction::Refresh { peer_id } => {
            let applied = {
                let mut table = network.routing_table.write().await;
                crate::membership::apply_to_routing_table(&action, &mut table, now)
            };
            match applied {
                Applied::EligibilityRestored {
                    was_suppressed: true,
                    ..
                } => info!(
                    "🔄 Membership: {} announced itself again — eligible for routing once more",
                    peer_id
                ),
                _ => tracing::debug!(
                    "📨 Membership: {} is a peer of this mesh (announcement confirmed)",
                    peer_id
                ),
            }
        }
        MembershipAction::Suppress { peer_id, window } => {
            let applied = {
                let mut table = network.routing_table.write().await;
                crate::membership::apply_to_routing_table(&action, &mut table, now)
            };
            if matches!(applied, Applied::EligibilityRemoved { .. }) {
                warn!(
                    "⏸️ Membership: {} announced it is leaving — out of the routing decisions \
                     for {}s (its entry is kept until then)",
                    peer_id,
                    window.as_secs()
                );
            }
        }
        MembershipAction::Dropped { peer_id, reason } => match reason {
            MembershipDrop::UnauthorizedPeer => warn!(
                "⚠️ Membership: dropped the announcement of {}: this mesh has not admitted it \
                 (`/register` + `/peers` are the only door; gossip never onboards)",
                peer_id
            ),
            MembershipDrop::OwnAnnouncement => {
                tracing::debug!("📨 Membership: our own announcement came back")
            }
            MembershipDrop::UnknownPeer => info!(
                "📨 Membership: a leave for the peer {} was ignored — it is not in the routing \
                 table any more (the tracker's `/peers` already dropped it, S58)",
                peer_id
            ),
            MembershipDrop::RateLimited { window_secs, limit } => warn!(
                "⚠️ Membership: {} exceeded {} messages in {}s — dropping the rest of the window",
                peer_id, limit, window_secs
            ),
            other => warn!(
                "⚠️ Membership: dropped the message about {} ({:?}), source {:?}",
                peer_id, other, source
            ),
        },
    }
}

pub async fn perform_graceful_shutdown(network: &NetworkManager) {
    info!("👋 Gracefully deregistering from musivum...");
    let mut deregister_req = DeregisterRequest {
        peer_id: network.runtime.peer_id.clone(),
        reason: "shutdown".to_string(),
        public_key: Vec::new(),
        signature: Vec::new(),
    };
    if let Err(e) = network.peer_network.sign_peer_request(&mut deregister_req) {
        warn!("⚠️ Could not sign deregistration request: {}", e);
        return;
    }

    let url = format!("{}/deregister", network.runtime.orchestrator_url);
    match network
        .runtime
        .http_client
        .post(&url)
        .json(&deregister_req)
        .send()
        .await
    {
        Ok(res) if res.status().is_success() => {
            info!("✅ Successfully deregistered from musivum");
        }
        Ok(res) => {
            warn!("⚠️ Deregistration returned status: {}", res.status());
        }
        Err(e) => {
            warn!("⚠️ Failed to send deregistration: {}", e);
        }
    }
}

pub async fn announce_to_musivum(
    config: &crate::config::WorkerConfig,
    assignment: &WorkerAssignment,
    peer_id: &str,
    real_ip: &str,
    bind_ip: &str,
    peer_network: &dyn tessella_core::traits::PeerNetwork,
) -> anyhow::Result<()> {
    let http_api_url = config.resolve_http_api_url(bind_ip);
    let real_p2p_ip = config.public_ip.as_deref().unwrap_or(real_ip);
    let real_p2p_port = config.public_p2p_port.unwrap_or(config.p2p_port);
    let p2p_addr =
        tessella_core::net::multiaddr_tcp(real_p2p_ip, real_p2p_port).ok_or_else(|| {
            anyhow::anyhow!(
                "public_ip '{}' is not a literal IP address: cannot advertise a P2P address",
                real_p2p_ip
            )
        })?;

    let vram_mb = if let Ok(val) = std::env::var("TESSELLA_VRAM_MB") {
        val.parse::<u32>()
            .unwrap_or_else(|_| tessella_core::vram::vram_mb_from_gb(config.vram_budget_gb))
    } else {
        tessella_core::vram::vram_mb_from_gb(config.vram_budget_gb)
    };

    let mut announce_req = tessella_core::AnnounceRequest {
        peer_id: peer_id.to_string(),
        host: real_p2p_ip.to_string(),
        port: real_p2p_port,
        #[allow(clippy::cast_possible_truncation)]
        vram_mb,
        model_id: assignment.model_repo.clone(),
        current_role: tessella_core::Role::Worker {
            layer_ranges: assignment.layer_ranges.clone(),
            replica_group_id: assignment.replica_group_id,
            expert_range: assignment.expert_range,
        },
        p2p_address: p2p_addr,
        http_api_url: Some(http_api_url),
        relay_address: None,
        status: Some(tessella_core::models::PeerStatus::Healthy),
        queue_depth: 0,
        expert_usage: None,
        parameter_contract_hash: tessella_protocol::swarm_contract_hash(&assignment.model_repo),
        public_ip: config.public_ip.clone(),
        public_api_port: config.public_api_port,
        public_key: Vec::new(),
        signature: Vec::new(),
    };
    peer_network.sign_peer_request(&mut announce_req)?;
    tracing::info!(
        "📡 Announcing to musivum with vram_mb={}",
        announce_req.vram_mb
    );

    let client = reqwest::Client::new();
    let _ = client
        .post(format!("{}/announce", config.orchestrator_url))
        .json(&announce_req)
        .send()
        .await;
    Ok(())
}

pub async fn notify_ready(
    config: &crate::config::WorkerConfig,
    assignment: &WorkerAssignment,
    peer_id: &str,
    queue_depth: usize,
    peer_network: &dyn tessella_core::traits::PeerNetwork,
) -> anyhow::Result<()> {
    let mut ready_req = tessella_core::WorkerReady {
        peer_id: peer_id.to_string(),
        model_repo: assignment.model_repo.clone(),
        layer_ranges: assignment.layer_ranges.clone(),
        replica_group_id: assignment.replica_group_id,
        expert_range: assignment.expert_range,
        queue_depth,
        public_key: Vec::new(),
        signature: Vec::new(),
    };
    peer_network.sign_peer_request(&mut ready_req)?;
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{}/ready", config.orchestrator_url))
        .json(&ready_req)
        .send()
        .await?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        anyhow::bail!("Musivum rejected /ready (status={}): {}", status, body);
    }
    Ok(())
}

pub fn announce_via_gossip(
    swarm: &mut libp2p::Swarm<tessella_network::MoeBehaviour>,
    peer_id: &str,
    config: &crate::config::WorkerConfig,
    assignment: &WorkerAssignment,
    listen_addr: &libp2p::Multiaddr,
    _real_ip: &str,
    bind_ip: &str,
) -> anyhow::Result<()> {
    let http_api_url = config.resolve_http_api_url(bind_ip);
    let gossip_msg = tessella_core::models::WorkerGossipMsg::WorkerJoined {
        peer_id: peer_id.to_string(),
        http_addr: http_api_url,
        p2p_addr: listen_addr.to_string(),
        layer_ranges: assignment.layer_ranges.clone(),
    };
    if let Ok(bytes) = serde_json::to_vec(&gossip_msg) {
        let topic = tessella_network::GossipTopic::new(tessella_network::WORKERS_TOPIC);
        let _ = swarm.behaviour_mut().gossipsub.publish(topic, bytes);
    }
    Ok(())
}
