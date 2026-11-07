//! Real Kademlia discovery: the DHT the mesh falls back to when the tracker is unreachable.
//!
//! H-06 (audit of 2026-09-13) found Kademlia **declared in four places with zero
//! queries**, so discovery was hub-and-spoke in practice: `/register`, `/announce`, `/peers` and
//! the orchestrator's topology gossip. Decision **D-4=(b)**, registered in S1, requires the
//! product promise to hold — «the network must stay usable if the
//! tracker disappears» — so `TASK-15`/S21 wires the DHT for real.
//!
//! Everything that can be decided *without a network* lives here as a pure function, because the
//! plan forbids opening sockets inside a step's verification:
//!
//! * [`kademlia_config`] — one explicit `kad::Config` for both roles (query timeout for a small
//!   mesh, four-hour record/provider TTLs, provider re-publication every 15 min and **our own
//!   protocol name**, so a Tessella DHT can never be confused with the public IPFS one).
//! * [`model_record_key`] — the single key function of the provider records
//!   (`tessella/model/<model_id>`), so a provider and a consumer cannot look at different keys.
//! * [`plan_bootstrap`] — which peers seed the routing table, in which order of precedence, and how
//!   many candidates had to be dropped (unparseable address, or our own peer id).
//! * [`peer_is_authorized`] — the admission predicate the DHT must share with the connection
//!   handler: a Kademlia record is unauthenticated data, so it may **never** onboard a peer the
//!   HTTP plane (`/register` + `/peers`) or the orchestrator has not already admitted.
//!
//! The precedence of discovery **sources** is tracker (authoritative) > gossipsub
//! (topology) > Kademlia (DHT) > persistent peer cache (`known_peers`). The *bootstrap* order of
//! [`plan_bootstrap`] is a different question — which address do we dial first — and it is the one
//! the card fixes: persistent cache → tracker → mDNS.

use libp2p::{Multiaddr, PeerId, StreamProtocol, kad};

/// Namespace of the provider records published through the DHT.
///
/// It is a prefix instead of a bare `model_id` so a record of this project cannot be confused with
/// another application's, and so the *record key* function is a single place to change.
pub const MODEL_KEY_PREFIX: &str = "tessella/model/";

/// Protocol name of our Kademlia network.
///
/// Kademlia nodes only talk to nodes announcing the same protocol (`Config::set_protocol_names`),
/// so this line is what keeps a Tessella mesh out of the public IPFS DHT even if a worker is some
/// day fed a public bootstrap address by mistake. The library default (`/ipfs/kad/1.0.0`) is
/// deliberately not used.
pub const KAD_PROTOCOL_NAME: &str = "/tessella/kad/1.0.0";

/// TTL of a *record* (not a provider record) in the DHT.
///
/// Four hours: long enough for a tracker outage, short enough that a peer which left the mesh stops
/// being advertised in the same working session. The card asks for an explicit TTL instead of the
/// library default (36 h), which is tuned for a public DHT with thousands of nodes.
const RECORD_TTL: std::time::Duration = std::time::Duration::from_secs(4 * 60 * 60);

/// How often a node re-publishes its own records, so they do not expire under a 4 h TTL.
const RECORD_REPUBLICATION: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// TTL of a provider record (`tessella/model/<model_id>` → peer id).
const PROVIDER_TTL: std::time::Duration = std::time::Duration::from_secs(4 * 60 * 60);

/// How often a node re-announces that it serves a model.
///
/// This is the knob that makes «provide by model» survive restarts and topology changes: the
/// publication repeats itself, so a worker that reconnects is announced again within minutes
/// without anybody calling `start_providing` by hand.
const PROVIDER_REPUBLICATION: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// Time budget of a single DHT query.
///
/// 30 s: this mesh is small (a handful of nodes per model) and a query that cannot finish in half a
/// minute is not worth blocking a discovery attempt on. The library default is 60 s.
const QUERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Replication factor: to how many closest peers a record is published.
///
/// 3 instead of the library's `K_VALUE` (20): with fewer than 20 nodes, asking 20 of them means
/// asking everybody — and timing out for the ones that are not there. The card's risk section is
/// explicit that a small mesh is what this DHT is for.
const REPLICATION_FACTOR: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(3) {
    Some(value) => value,
    None => panic!("REPLICATION_FACTOR must not be zero"),
};

/// The DHT key of a model's provider record.
///
/// Deterministic by construction (no hashing of our own, no random component): the provider and
/// every consumer that looks the model up compute the same key, which is what makes
/// `start_providing`/`get_providers` agree. The key is a plain UTF-8 string inside the record-key
/// type, which keeps the DHT inspectable while [`KAD_PROTOCOL_NAME`] keeps it private to Tessella.
pub fn model_record_key(model_id: &str) -> kad::RecordKey {
    kad::RecordKey::new(&format!("{MODEL_KEY_PREFIX}{model_id}"))
}

/// The `kad::Config` both roles use.
///
/// One function for the two builders, exactly like [`crate::gossipsub_config`]: two roles with two
/// different Kademlia configurations would be a protocol divergence (two nodes that cannot answer
/// each other's queries) and it would not be noticed until a tracker outage.
pub fn kademlia_config() -> kad::Config {
    let mut config = kad::Config::default();
    config
        .set_protocol_names(vec![StreamProtocol::new(KAD_PROTOCOL_NAME)])
        .set_query_timeout(QUERY_TIMEOUT)
        .set_replication_factor(REPLICATION_FACTOR)
        .set_record_ttl(Some(RECORD_TTL))
        .set_publication_interval(Some(RECORD_REPUBLICATION))
        .set_provider_record_ttl(Some(PROVIDER_TTL))
        .set_provider_publication_interval(Some(PROVIDER_REPUBLICATION));
    config
}

/// Builds the Kademlia behaviour of a node.
///
/// `server_mode` is what the card asks for in both roles: `Some(Mode::Server)` fixes the mode
/// unconditionally. Without it the behaviour starts as an auto-mode client that only becomes a
/// server once it confirms an external address — precisely the state a worker behind NAT never
/// reaches, so it would answer nobody's queries and the DHT would be inert (the defect H-06
/// describes behind «0 queries»).
///
/// The store is the library `MemoryStore`: the peer cache that must survive a restart lives in
/// SQLite (`tessella-storage`, table `known_peers`), not here.
pub fn kademlia_behaviour(
    local_peer_id: PeerId,
    server_mode: bool,
) -> kad::Behaviour<kad::store::MemoryStore> {
    let mut behaviour = kad::Behaviour::with_config(
        local_peer_id,
        kad::store::MemoryStore::new(local_peer_id),
        kademlia_config(),
    );
    if server_mode {
        behaviour.set_mode(Some(kad::Mode::Server));
    }
    behaviour
}

/// Where a candidate seed address comes from.
///
/// The order of the variants is the precedence order of [`plan_bootstrap`]: the caller passes its
/// sources already sorted this way, and a source that comes first wins the peers it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoverySource {
    /// The `known_peers` table of SQLite: the addresses this node actually used last time.
    PersistentCache,
    /// `GET /peers` (authoritative while it answers).
    Tracker,
    /// mDNS, LAN-only.
    Mdns,
}

impl DiscoverySource {
    /// Stable label used in logs and tests.
    pub fn label(self) -> &'static str {
        match self {
            Self::PersistentCache => "persistent-cache",
            Self::Tracker => "tracker",
            Self::Mdns => "mdns",
        }
    }
}

/// A peer we can dial to join the DHT.
///
/// Both fields are **validated**: a candidate whose peer id or address does not parse never becomes
/// a seed (it is counted in [`BootstrapPlan::unusable`]), so the caller can hand the values straight
/// to `kademlia.add_address` without another round of parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerSeed {
    /// Where the address was learned.
    pub source: DiscoverySource,
    /// The peer the address belongs to.
    pub peer_id: PeerId,
    /// The parsed, dialable address.
    pub addr: Multiaddr,
}

/// Result of [`plan_bootstrap`]: the seeds to add and how many candidates were unusable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BootstrapPlan {
    /// One seed per peer, in the order the sources named them.
    pub seeds: Vec<PeerSeed>,
    /// Candidates dropped because the peer id or the address did not parse (or was empty).
    pub unusable: usize,
}

impl BootstrapPlan {
    /// `true` when there is nothing to bootstrap with, i.e. when the caller must say so out loud
    /// instead of calling `bootstrap()` and swallowing `NoKnownPeers`.
    pub fn is_empty(&self) -> bool {
        self.seeds.is_empty()
    }
}

/// Decides which peers seed the DHT, from `sources` **in precedence order**.
///
/// Rules (each one pinned by a test):
///
/// * our own `peer_id` is never a seed (dialing ourselves is not discovery);
/// * the first source that names a peer with a **parseable** peer id *and* address wins that peer; a
///   source whose observation of that peer is unusable does **not** block a later source from
///   providing a good one (it only counts in [`BootstrapPlan::unusable`]);
/// * an empty plan is a legitimate outcome — the caller logs it, it is not an error.
pub fn plan_bootstrap(
    own_peer_id: &str,
    sources: &[(DiscoverySource, Vec<(String, String)>)],
) -> BootstrapPlan {
    let mut plan = BootstrapPlan::default();
    let mut resolved: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for (source, candidates) in sources {
        for (peer_id_str, addr) in candidates {
            if peer_id_str == own_peer_id || resolved.contains(peer_id_str.as_str()) {
                continue;
            }
            let parsed_id = peer_id_str.parse::<PeerId>();
            let parsed_addr = addr.parse::<Multiaddr>();
            match (parsed_id, parsed_addr) {
                (Ok(peer_id), Ok(addr)) => {
                    resolved.insert(peer_id_str.as_str());
                    plan.seeds.push(PeerSeed {
                        source: *source,
                        peer_id,
                        addr,
                    });
                }
                _ => plan.unusable += 1,
            }
        }
    }

    plan
}

/// The **single** admission predicate of the mesh.
///
/// A peer is allowed when it is us, when it is the orchestrator discovered through `/p2p_id`, or
/// when `/peers`/gossip has already put it in the routing table. It is used by the connection
/// handler (`SwarmEvent::ConnectionEstablished`) **and** by the Kademlia dial path, and that is the
/// point: a DHT record is unauthenticated data, so a provider answer may only *refresh the address*
/// of a peer this predicate already accepts — never onboard a new one. Onboarding is the job of
/// `/register` (and of `TASK-16`'s authenticated membership channel).
pub fn peer_is_authorized(
    peer_id: &str,
    own_peer_id: &str,
    orchestrator_peer_id: Option<&str>,
    in_routing_table: bool,
) -> bool {
    peer_id == own_peer_id
        || orchestrator_peer_id.is_some_and(|id| id == peer_id)
        || in_routing_table
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::swarm::NetworkBehaviour;
    use std::task::Poll;

    #[test]
    fn model_record_key_is_deterministic_and_namespaced() {
        let key = model_record_key("Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(
            key,
            model_record_key("Qwen/Qwen2.5-0.5B-Instruct"),
            "the provider and the consumer must build the same key"
        );
        assert_eq!(
            std::str::from_utf8(key.as_ref()).expect("utf8 key"),
            "tessella/model/Qwen/Qwen2.5-0.5B-Instruct"
        );
    }

    #[test]
    fn model_record_keys_do_not_collide() {
        // The namespace is what keeps two different model ids apart even when one is a prefix of
        // the other, and what keeps a Tessella record from colliding with another application's.
        assert_ne!(model_record_key("a"), model_record_key("a/b"));
        assert_ne!(
            model_record_key("Qwen/Qwen2.5-0.5B-Instruct"),
            model_record_key("Qwen/Qwen2.5-1.5B-Instruct")
        );
    }

    #[test]
    fn kademlia_behaviour_uses_our_own_protocol() {
        // The protocol name is the segregation between this DHT and the public IPFS one: two nodes
        // only talk when they announce the same name, so this assertion is what keeps a Tessella
        // mesh out of a network it does not belong to.
        let behaviour = kademlia_behaviour(PeerId::random(), true);
        let names: Vec<String> = behaviour
            .protocol_names()
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        assert_eq!(names, vec![KAD_PROTOCOL_NAME.to_string()]);
    }

    /// The offline half of criterion 3 of `TASK-15`: `start_providing` + `get_providers` agree on
    /// the key, so a node knows its own model record before any network exists. The cross-node half
    /// (a provider answer arriving through a real connection) is `tests/kademlia_discovery.rs`,
    /// which uses the in-memory transport.
    #[test]
    fn a_server_answers_its_own_provider_query() {
        let local = PeerId::random();
        let mut behaviour = kademlia_behaviour(local, true);
        let key = model_record_key("Qwen/Qwen2.5-0.5B-Instruct");

        behaviour
            .start_providing(key.clone())
            .expect("the local provider record must be stored");

        let query_id = behaviour.get_providers(key.clone());

        let waker = futures::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        let mut found: Option<Vec<PeerId>> = None;
        for _ in 0..16 {
            match NetworkBehaviour::poll(&mut behaviour, &mut cx) {
                Poll::Ready(libp2p::swarm::ToSwarm::GenerateEvent(
                    kad::Event::OutboundQueryProgressed {
                        id,
                        result:
                            kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
                                key: answered,
                                providers,
                            })),
                        ..
                    },
                )) if id == query_id && answered == key => {
                    found = Some(providers.into_iter().collect());
                    break;
                }
                // `start_providing` emits its own events too; anything that is not the answer we
                // are waiting for is not an error, the loop just keeps draining.
                Poll::Ready(_) => continue,
                Poll::Pending => break,
            }
        }

        let providers = found.expect("the local provider query must be answered");
        assert_eq!(providers, vec![local]);
    }

    #[test]
    fn plan_bootstrap_keeps_the_first_usable_address_per_peer() {
        let peer_a = PeerId::random().to_string();
        let peer_b = PeerId::random().to_string();
        let us = PeerId::random().to_string();

        let plan = plan_bootstrap(
            &us,
            &[
                (
                    DiscoverySource::PersistentCache,
                    vec![
                        (peer_a.clone(), "/ip4/10.0.0.1/tcp/9001".to_string()),
                        (peer_b.clone(), "not-a-multiaddr".to_string()),
                        (us.clone(), "/ip4/10.0.0.9/tcp/9001".to_string()),
                        (
                            "not-a-peer-id".to_string(),
                            "/ip4/10.0.0.8/tcp/9001".to_string(),
                        ),
                    ],
                ),
                (
                    DiscoverySource::Tracker,
                    vec![
                        // Same peer as the cache: the first source wins, no duplicate dial.
                        (peer_a.clone(), "/ip4/10.0.0.2/tcp/9001".to_string()),
                        // Unusable in the cache, usable here: `unusable` does not poison the peer.
                        (peer_b.clone(), "/ip4/10.0.0.3/tcp/9001".to_string()),
                    ],
                ),
            ],
        );

        assert_eq!(
            plan.unusable, 2,
            "the malformed address and the malformed id"
        );
        assert_eq!(
            plan.seeds.len(),
            2,
            "one seed per peer, not per observation"
        );
        assert_eq!(plan.seeds[0].peer_id.to_string(), peer_a);
        assert_eq!(plan.seeds[0].source, DiscoverySource::PersistentCache);
        assert_eq!(plan.seeds[0].addr.to_string(), "/ip4/10.0.0.1/tcp/9001");
        assert_eq!(plan.seeds[1].peer_id.to_string(), peer_b);
        assert_eq!(plan.seeds[1].source, DiscoverySource::Tracker);
        assert_eq!(plan.seeds[1].addr.to_string(), "/ip4/10.0.0.3/tcp/9001");
    }

    #[test]
    fn plan_bootstrap_without_sources_is_empty() {
        // The caller must warn about this instead of calling `bootstrap()` and swallowing
        // `NoKnownPeers`.
        let empty = plan_bootstrap("us", &[]);
        assert!(empty.is_empty());
        assert_eq!(empty.unusable, 0);

        let us = PeerId::random().to_string();
        let only_us = plan_bootstrap(
            &us,
            &[(
                DiscoverySource::Mdns,
                vec![(us.clone(), "/ip4/10.0.0.9/tcp/9001".to_string())],
            )],
        );
        assert!(only_us.is_empty(), "dialing ourselves is not discovery");
    }

    #[test]
    fn peer_is_authorized_matches_the_three_admission_paths() {
        // The DHT must use exactly this predicate: a provider record may refresh the address of an
        // admitted peer, never onboard a stranger.
        assert!(peer_is_authorized("us", "us", None, false));
        assert!(peer_is_authorized(
            "orchestrator",
            "us",
            Some("orchestrator"),
            false
        ));
        assert!(peer_is_authorized("peer-a", "us", None, true));
        assert!(!peer_is_authorized("stranger", "us", None, false));
        assert!(
            !peer_is_authorized("stranger", "us", Some("orchestrator"), false),
            "an orchestrator id must not authorize anybody else"
        );
    }
}
