//! Pure pieces of the Kademlia wiring of the worker (`TASK-15` · H-06 · D-4=(b)).
//!
//! The plan forbids opening sockets inside a step's verification, so everything that can be decided
//! from data lives here as a pure function with its own test:
//!
//! * [`plan_seeds`] turns the two places a worker learns addresses from — the **persistent cache**
//!   (`known_peers` in SQLite) and the **live mesh view** (the routing table, fed by `/peers` and the
//!   topology gossip) — into the ordered bootstrap plan that `tessella-network` computes. Precedence
//!   is the one the card fixes: persistent cache → tracker → mDNS, first usable observation wins.
//! * [`select_dialable_providers`] decides which of the peers a **DHT provider answer** named may be
//!   dialed, and it is the enforcement point of the card's «no unauthorized peer reaches `dial`
//!   through a Kademlia answer»: a provider record is unauthenticated data, so the DHT may only
//!   refresh the address of a peer the mesh already admitted (`/register` + `/peers` + gossip).
//!
//! The precedence of *discovery sources* is **tracker (authoritative) > gossipsub > Kademlia >
//! persistent cache** (`known_peers`), and this module is the code half of that rule: the DHT may
//! refresh an already-admitted peer, it may never admit one.

use tessella_core::models::LayerPeer;
use tessella_network::kademlia::{
    peer_is_authorized, plan_bootstrap, BootstrapPlan, DiscoverySource,
};
use tessella_network::NetworkAddress;

/// A provider of a model that passed the admission filter and has a dialable address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialCandidate {
    /// The provider that was announced through the DHT.
    pub peer_id: String,
    /// Where to dial it (from the routing table, never from the record itself).
    pub addr: NetworkAddress,
}

/// What a provider answer left behind, with the reasons a candidate was dropped (for the log).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DialPlan {
    /// Providers that may be dialed.
    pub dialable: Vec<DialCandidate>,
    /// Providers the mesh never admitted: the DHT does not get to onboard a peer.
    pub unauthorized: usize,
    /// Admitted providers whose address we do not know (nothing to dial).
    pub without_address: usize,
}

impl DialPlan {
    /// `true` when there is nobody to dial.
    pub fn is_empty(&self) -> bool {
        self.dialable.is_empty()
    }
}

/// Addresses of the peers that announced one (`p2p_addr` empty means the peer cannot be dialed).
fn candidates(peers: &[LayerPeer]) -> Vec<(String, String)> {
    peers
        .iter()
        .filter(|peer| !peer.p2p_addr.trim().is_empty())
        .map(|peer| (peer.peer_id.clone(), peer.p2p_addr.clone()))
        .collect()
}

/// The sources of a bootstrap attempt, in precedence order.
///
/// mDNS is the third source and it is empty on purpose: its peers arrive through libp2p's mDNS
/// behaviour (which stores them in the swarm), not through [`LayerPeer`], so there is nothing to
/// list here. The variant is still returned so the precedence is explicit in one place.
pub fn seed_sources(
    mesh: &[LayerPeer],
    cached: &[LayerPeer],
) -> Vec<(DiscoverySource, Vec<(String, String)>)> {
    vec![
        (DiscoverySource::PersistentCache, candidates(cached)),
        (DiscoverySource::Tracker, candidates(mesh)),
        (DiscoverySource::Mdns, Vec::new()),
    ]
}

/// The bootstrap plan of this worker: which peers to add to the DHT and how many candidates were
/// unusable (bad peer id, bad address, or our own id).
pub fn plan_seeds(own_peer_id: &str, mesh: &[LayerPeer], cached: &[LayerPeer]) -> BootstrapPlan {
    plan_bootstrap(own_peer_id, &seed_sources(mesh, cached))
}

/// The peer record the routing table stores, from one entry of `GET /peers?model_id=`
/// (`TASK-41`/S47 · `TASK-87`/S94).
///
/// `/peers` is **the** routing contract of the worker: the orchestrator filters it by model and
/// answers `PeersEntry` (wrapping `NodeInfo` and the `canary` mark).
///
/// Propagates `canary` from the entry so the worker's router knows never to pick a canary as
/// the primary of a client session (`TASK-87`/S94).
pub fn layer_peer_from_peers_entry(entry: &tessella_core::models::PeersEntry) -> LayerPeer {
    layer_peer_from_node_info_with_canary(&entry.node, entry.canary)
}

pub fn layer_peer_from_node_info(info: &tessella_core::models::NodeInfo) -> LayerPeer {
    layer_peer_from_node_info_with_canary(info, false)
}

pub fn layer_peer_from_node_info_with_canary(
    info: &tessella_core::models::NodeInfo,
    canary: bool,
) -> LayerPeer {
    let http_addr = info.http_api_url.clone().unwrap_or_default();
    let resolved_addr =
        crate::network_manager::resolve_preferred_http_addr(&http_addr, &info.p2p_address);

    LayerPeer {
        peer_id: info.peer_id.clone(),
        http_addr: resolved_addr,
        p2p_addr: info.p2p_address.clone(),
        // `TASK-96`/S102: **all** the ranges the role publishes, not the first one. Dropping the
        // rest here was what made one range per node true in practice: the routing table indexed
        // one stretch, so a node holding two was only ever asked for the first — the second chunk
        // of VRAM served nothing and the mesh could not see it.
        layer_ranges: match &info.role {
            tessella_core::models::Role::Worker { layer_ranges, .. } => layer_ranges.clone(),
            _ => Vec::new(),
        },
        expert_range: match &info.role {
            tessella_core::models::Role::Worker { expert_range, .. } => *expert_range,
            _ => None,
        },
        queue_depth: info.queue_depth,
        public_ip: info.public_ip.clone(),
        public_api_port: info.public_api_port,
        canary,
    }
}

/// Filters the providers of a DHT answer down to the ones this worker may dial.
///
/// `mesh` is the routing table at the moment of the answer: it is both the source of the address and
/// the membership list, so a provider the tracker/gossip never mentioned is **refused** even if the
/// DHT swears it serves the model. That is deliberate: records are unauthenticated, and the
/// onboarding path is `/register` (plus `TASK-16`'s authenticated membership channel), not the DHT.
pub fn select_dialable_providers(
    providers: &[String],
    own_peer_id: &str,
    orchestrator_peer_id: Option<&str>,
    mesh: &[LayerPeer],
) -> DialPlan {
    let mut plan = DialPlan::default();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for provider in providers {
        if !seen.insert(provider.as_str()) {
            continue;
        }
        let known = mesh.iter().find(|peer| &peer.peer_id == provider);
        if !peer_is_authorized(provider, own_peer_id, orchestrator_peer_id, known.is_some()) {
            plan.unauthorized += 1;
            continue;
        }
        match known
            .map(|peer| peer.p2p_addr.trim())
            .filter(|addr| !addr.is_empty())
            .and_then(|addr| addr.parse::<NetworkAddress>().ok())
        {
            Some(addr) => plan.dialable.push(DialCandidate {
                peer_id: provider.clone(),
                addr,
            }),
            None => plan.without_address += 1,
        }
    }

    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(peer_id: &str, p2p_addr: &str) -> LayerPeer {
        LayerPeer {
            peer_id: peer_id.to_string(),
            http_addr: format!("http://{peer_id}:10001"),
            p2p_addr: p2p_addr.to_string(),
            layer_ranges: vec![(0, 12)],
            expert_range: None,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        }
    }

    #[test]
    fn seed_sources_promise_the_cache_before_the_tracker() {
        let cached = vec![peer("peer-a", "/ip4/10.0.0.1/tcp/9001")];
        let mesh = vec![
            peer("peer-b", "/ip4/10.0.0.2/tcp/9001"),
            // A peer that never announced a P2P address cannot be a DHT seed.
            peer("peer-c", ""),
        ];

        let sources = seed_sources(&mesh, &cached);
        assert_eq!(sources.len(), 3, "cache, tracker, mdns");
        assert_eq!(sources[0].0, DiscoverySource::PersistentCache);
        assert_eq!(
            sources[0].1,
            vec![("peer-a".to_string(), "/ip4/10.0.0.1/tcp/9001".to_string())]
        );
        assert_eq!(sources[1].0, DiscoverySource::Tracker);
        assert_eq!(
            sources[1].1,
            vec![("peer-b".to_string(), "/ip4/10.0.0.2/tcp/9001".to_string())]
        );
        assert_eq!(sources[2].0, DiscoverySource::Mdns);
        assert!(sources[2].1.is_empty());
    }

    #[test]
    fn plan_seeds_keeps_one_seed_per_peer_and_reports_candidates() {
        let own = tessella_network::generate_keypair();
        let own_id = tessella_network::peer_id_from_keypair(&own);
        let other = tessella_network::generate_keypair();
        let other_id = tessella_network::peer_id_from_keypair(&other);

        let cached = vec![peer(&other_id, "/ip4/10.0.0.1/tcp/9001")];
        // The tracker now announces the same peer with another address; the cache wins.
        let mesh = vec![
            peer(&other_id, "/ip4/10.0.0.2/tcp/9001"),
            peer(&own_id, "/ip4/10.0.0.3/tcp/9001"),
        ];

        let plan = plan_seeds(&own_id, &mesh, &cached);
        assert_eq!(
            plan.seeds.len(),
            1,
            "one seed per peer; we are never our own seed"
        );
        assert_eq!(plan.seeds[0].peer_id.to_string(), other_id);
        assert_eq!(plan.seeds[0].addr.to_string(), "/ip4/10.0.0.1/tcp/9001");
        assert!(plan.unusable == 0, "no candidate was malformed");
    }

    #[test]
    fn a_provider_answer_never_dials_an_unauthorized_peer() {
        // The card's criterion 7: the DHT cannot onboard a peer. `stranger` is what a poisoned
        // provider record looks like, and it must not reach `dial`.
        let mesh = vec![peer("admitted", "/ip4/10.0.0.1/tcp/9001")];
        let providers = vec![
            "admitted".to_string(),
            "stranger".to_string(),
            "the-orchestrator".to_string(),
        ];

        let plan = select_dialable_providers(&providers, "us", Some("the-orchestrator"), &mesh);

        assert_eq!(plan.unauthorized, 1, "only the stranger is refused");
        let dialed: Vec<&str> = plan
            .dialable
            .iter()
            .map(|candidate| candidate.peer_id.as_str())
            .collect();
        assert_eq!(dialed, vec!["admitted"]);
        assert_eq!(plan.dialable[0].addr.to_string(), "/ip4/10.0.0.1/tcp/9001");
        // The orchestrator passed the admission filter but the routing table does not carry its
        // address (that comes from `/p2p_id`), so it is reported instead of dialed blindly.
        assert_eq!(plan.without_address, 1);
    }

    /// `TASK-41`/S47: `GET /peers?model_id=` is **the** routing contract of the worker, and what it
    /// answers is a `NodeInfo` whose `role` carries the layer range. The heartbeat used to query the
    /// observability endpoint (`/workers?model_id=`), whose body is a different shape, and parse it
    /// as `Vec<LayerPeer>`: it could never work and the failure was invisible. This test pins the
    /// shared conversion so a shape change is a red test and not a silent routing gap.
    #[test]
    fn a_peers_entry_becomes_the_layer_peer_the_routing_table_stores() {
        let node = node_info(
            "12D3KooWPeerA",
            "http://10.0.0.7:10001",
            "/ip4/10.0.0.7/tcp/9001",
            Some((0, 12)),
        );

        let peer = layer_peer_from_node_info(&node);

        assert_eq!(peer.peer_id, "12D3KooWPeerA");
        assert_eq!(peer.http_addr, "http://10.0.0.7:10001");
        assert_eq!(peer.p2p_addr, "/ip4/10.0.0.7/tcp/9001");
        assert_eq!(peer.layer_ranges, vec![(0, 12)]);
        assert_eq!(peer.expert_range, Some((0, 4)));
        assert_eq!(peer.queue_depth, 2);
        assert!(!peer.canary);
    }

    /// `TASK-87`/S94: when `/peers` provides a `PeersEntry` with `canary: true`,
    /// `layer_peer_from_peers_entry` sets `canary: true` on `LayerPeer`.
    #[test]
    fn a_peers_entry_with_canary_mark_sets_canary_flag_on_layer_peer() {
        let node = node_info(
            "12D3KooWCanary",
            "http://10.0.0.9:10001",
            "/ip4/10.0.0.9/tcp/9001",
            Some((0, 12)),
        );
        let entry = tessella_core::models::PeersEntry { node, canary: true };

        let peer = layer_peer_from_peers_entry(&entry);

        assert_eq!(peer.peer_id, "12D3KooWCanary");
        assert!(peer.canary);
    }

    /// A node without a layer range (`Role::Unassigned`) is not a routing candidate: it declares an
    /// **empty** set of ranges, so the routing index never mentions it — it used to declare `(0, 0)`,
    /// a range that covers nothing but still looked like a shard.
    #[test]
    fn a_non_worker_role_declares_no_layer_range() {
        let mut node = node_info("12D3KooWPeerB", "", "", None);
        node.role = tessella_core::models::Role::Unassigned;

        let peer = layer_peer_from_node_info(&node);

        assert!(
            peer.layer_ranges.is_empty(),
            "an unassigned node serves no layer: {:?}",
            peer.layer_ranges
        );
        assert_eq!(peer.expert_range, None);
    }

    fn node_info(
        peer_id: &str,
        http_api_url: &str,
        p2p_address: &str,
        layer_range: Option<(usize, usize)>,
    ) -> tessella_core::models::NodeInfo {
        tessella_core::models::NodeInfo {
            peer_id: peer_id.to_string(),
            host: "10.0.0.7".to_string(),
            port: 9001,
            vram_mb: 2048,
            role: match layer_range {
                Some(range) => tessella_core::models::Role::Worker {
                    layer_ranges: vec![range],
                    replica_group_id: 0,
                    expert_range: Some((0, 4)),
                },
                None => tessella_core::models::Role::Unassigned,
            },
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            p2p_address: p2p_address.to_string(),
            http_api_url: Some(http_api_url.to_string()).filter(|url| !url.is_empty()),
            relay_address: None,
            status: tessella_core::models::PeerStatus::Healthy,
            failed_requests: 0,
            successful_requests: 3,
            latency_p99_ms: 5,
            queue_depth: 2,
            last_error: None,
            slash_count: 0,
            blacklisted: false,
            public_ip: None,
            public_api_port: None,
        }
    }

    #[test]
    fn an_authorized_provider_without_an_address_is_counted_but_not_dialed() {
        let mesh = vec![peer("no-addr", ""), peer("bad-addr", "not-a-multiaddr")];
        let providers = vec![
            "no-addr".to_string(),
            "bad-addr".to_string(),
            "no-addr".to_string(),
        ];

        let plan = select_dialable_providers(&providers, "us", None, &mesh);
        assert!(plan.is_empty(), "nothing dialable");
        assert_eq!(
            plan.without_address, 2,
            "repeated answers count once per peer"
        );
        assert_eq!(plan.unauthorized, 0, "they are in the routing table");
    }
}
