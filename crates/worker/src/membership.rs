//! The worker side of the P2P membership channel `tessella/workers/v1` (`TASK-16` · S22).
//!
//! Until S22 the topic was **dead in both directions**: the worker subscribed to it but its
//! gossipsub handler only looked at the topology topic and dropped everything else in a `_ => {}`,
//! so a `WorkerJoined`/`WorkerLeft` from another node was received and silently discarded.
//!
//! This module is the *decision* half of the fix, and it is pure on purpose: the plan forbids
//! opening sockets inside a step's verification, so building a `Swarm` just to test the handler is
//! not an option. The rules it implements:
//!
//! 1. **Gossip never onboards.** A peer only enters the routing table through `/register` +
//!    `/announce` + `/peers` (the tracker) or through the topology gossip Musivum publishes. An
//!    announcement of a peer the mesh has not admitted is dropped and remembered only as an
//!    *observation*, never as a route. The shared admission predicate
//!    ([`tessella_network::kademlia::peer_is_authorized`]) is the only door, exactly as in the
//!    Kademlia dial path (H-06 mitigation).
//! 2. **The declared identity must be the emitter's.** libp2p gossipsub signs the `source` of a
//!    message (Noise + `MessageAuthenticity::Signed`), so a payload that claims another `peer_id`
//!    is rejected. That is the «as a minimum, only already-authorized peer ids» rule of the card
//!    done with the identity the transport already authenticated, so `WorkerGossipMsg` keeps
//!    working byte-for-byte (no DTO change: `tessella-protocol` is never touched outside a
//!    BREAKING step and S22 is not one).
//!    **One emitter is the exception, and only for a leave** (`S58`): the orchestrator announces
//!    the departure of a *third* node — it is the node that sees `ConnectionClosed` and the only
//!    source of `WorkerLeft` in the system — so `observe_left` accepts `source == peer_id` **or**
//!    `source == orchestrator`. Before S58 the strict check dropped the tracker's leave as
//!    `SourceMismatch`, which is exactly what a real two-node run measured: the suppression path
//!    that S22 wrote could never run. An *announcement* keeps the strict rule: letting the tracker
//!    onboard by gossip would contradict rule 1.
//! 3. **A leave is not a delete.** `WorkerLeft` marks the peer as *not eligible* with a TTL
//!    ([`GOSSIP_SUPPRESSION_TTL`]): the router stops choosing it at once and the mark is only
//!    purged when the window expires, so an out-of-order message cannot lose a peer forever.
//! 4. **A join stops the suppression.** Seeing the peer again is the newest observation, so it
//!    clears the mark (last observation wins, which is what makes the reducer tolerant to the
//!    unordered delivery of gossipsub).

use crate::routing_table::RoutingTable;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tessella_core::models::WorkerGossipMsg;
use tessella_network::kademlia::peer_is_authorized;

/// How long a `WorkerLeft` keeps a peer out of the routing decisions.
///
/// Long enough to survive the tracker's own `/peers` refresh (60 s) without sending hops to a peer
/// that is leaving, and short enough that a message which arrived out of order — or a single node
/// that lied — cannot keep a live peer out of the mesh for the rest of the session.
pub const GOSSIP_SUPPRESSION_TTL: Duration = Duration::from_secs(300);

/// How long the observation of a peer the mesh has not admitted is remembered (for the log and for
/// diagnostics). It is **not** a route and it never becomes one.
pub const GOSSIP_OBSERVATION_TTL: Duration = Duration::from_secs(600);

/// Maximum number of membership messages accepted from one peer inside [`MEMBERSHIP_RATE_WINDOW`].
pub const MEMBERSHIP_MESSAGES_PER_WINDOW: u32 = 30;

/// Sliding window of the per-peer membership budget.
pub const MEMBERSHIP_RATE_WINDOW: Duration = Duration::from_secs(60);

/// Everything the reducer needs to know about this node and its mesh view.
pub struct MembershipChannels<'a> {
    /// This worker's peer id (announcing ourselves teaches nothing).
    pub own_peer_id: &'a str,
    /// The orchestrator discovered through `/p2p_id`, if any.
    pub orchestrator_peer_id: Option<&'a str>,
    /// The mesh view `/peers` and the topology gossip built.
    pub routing_table: &'a RoutingTable,
}

impl MembershipChannels<'_> {
    /// The shared admission predicate of the mesh (H-06 mitigation): us, the orchestrator, or a
    /// peer the tracker already admitted.
    fn admits(&self, peer_id: &str) -> bool {
        peer_is_authorized(
            peer_id,
            self.own_peer_id,
            self.orchestrator_peer_id,
            self.routing_table.contains_peer(peer_id),
        )
    }
}

/// What the worker does with one membership message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MembershipAction {
    /// The peer belongs to the mesh: its membership is refreshed (a previous `WorkerLeft` for it
    /// stops suppressing it).
    Refresh {
        /// The peer that announced itself.
        peer_id: String,
    },
    /// A `WorkerLeft` for a peer of the mesh: it stops being eligible for routing until
    /// [`GOSSIP_SUPPRESSION_TTL`] expires.
    Suppress {
        /// The peer that left.
        peer_id: String,
        /// How long it stays not eligible.
        window: Duration,
    },
    /// The message is not acted upon. `reason` is what the caller logs.
    Dropped {
        /// The peer the message was about.
        peer_id: String,
        /// Why it was dropped.
        reason: MembershipDrop,
    },
}

/// Why a membership message was dropped. Every variant maps to one line in the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MembershipDrop {
    /// The payload declares a `peer_id` that is not the identity gossipsub authenticated.
    SourceMismatch {
        /// The authenticated emitter, when there was one.
        source: Option<String>,
    },
    /// `peer_id`, `p2p_addr`, `http_addr` or `layer_range` are not usable (the same shape the H-04
    /// fix applies to an advertised address: a literal IP with a real TCP port and an `http(s)`
    /// URL, with no DNS resolution).
    MalformedPayload(String),
    /// The message talks about this very node.
    OwnAnnouncement,
    /// Not us, not the orchestrator, not a peer of the routing table: the mesh onboards through
    /// `/register` + `/announce`, never through gossip.
    UnauthorizedPeer,
    /// A `WorkerLeft` for a peer nobody knows: there is nothing to un-schedule.
    UnknownPeer,
    /// Too many messages from this emitter inside [`MEMBERSHIP_RATE_WINDOW`].
    RateLimited {
        /// Length of the window, in seconds.
        window_secs: u64,
        /// Messages accepted per window.
        limit: u32,
    },
}

/// What was recorded about a peer the mesh has not admitted (observability only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateObservation {
    /// The announced peer.
    pub peer_id: String,
    /// The layer ranges it claims (`TASK-96`/S102: the whole shard, not its first chunk).
    pub layer_ranges: Vec<(usize, usize)>,
    /// When it was seen.
    pub seen_at: Instant,
}

/// Per-emitter sliding-window accounting.
#[derive(Debug, Clone, Copy)]
struct RateWindow {
    messages: u32,
    started: Instant,
}

/// The effect of a [`MembershipAction`] on the routing table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Applied {
    /// The peer is eligible again (the mark was there or it was not).
    EligibilityRestored {
        /// The peer that announced itself.
        peer_id: String,
        /// `true` when a `WorkerLeft` mark was cleared.
        was_suppressed: bool,
    },
    /// The peer is not eligible until the window expires.
    EligibilityRemoved {
        /// The peer that left.
        peer_id: String,
        /// How long it stays not eligible.
        window: Duration,
    },
    /// Nothing to do: the action was a drop, or it targeted a peer the table does not know.
    Nothing,
}

/// The membership reducer of the worker. Pure: the clock and the mesh view come in as parameters.
#[derive(Debug, Default)]
pub struct GossipMembership {
    /// Peers announced by gossip that the mesh has not admitted.
    observations: HashMap<String, CandidateObservation>,
    /// Messages seen per emitter inside the current window.
    budget: HashMap<String, RateWindow>,
}

impl GossipMembership {
    /// An empty reducer.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many peers are currently recorded as observations (never as routes).
    pub fn candidate_count(&self) -> usize {
        self.observations.len()
    }

    /// The observation of `peer_id`, when it is still remembered (diagnostics).
    pub fn observation_of(&self, peer_id: &str) -> Option<&CandidateObservation> {
        self.observations.get(peer_id)
    }

    /// Decides what a `WorkerJoined` means. See the module rules for the order of the checks.
    pub fn observe_joined(
        &mut self,
        msg: &WorkerGossipMsg,
        source: Option<&str>,
        channels: &MembershipChannels<'_>,
        now: Instant,
    ) -> MembershipAction {
        let WorkerGossipMsg::WorkerJoined {
            peer_id,
            http_addr,
            p2p_addr,
            layer_ranges,
        } = msg
        else {
            return MembershipAction::Dropped {
                peer_id: message_peer_id(msg),
                reason: MembershipDrop::MalformedPayload(
                    "a leave is not an announcement".to_string(),
                ),
            };
        };

        if let Some(dropped) = self.check_flood(source.unwrap_or(peer_id), now) {
            return dropped;
        }
        if let Some(dropped) = check_source(peer_id, source) {
            return dropped;
        }
        if let Err(reason) = validate_announcement(peer_id, http_addr, p2p_addr, layer_ranges) {
            return MembershipAction::Dropped {
                peer_id: peer_id.clone(),
                reason,
            };
        }
        if peer_id == channels.own_peer_id {
            return MembershipAction::Dropped {
                peer_id: peer_id.clone(),
                reason: MembershipDrop::OwnAnnouncement,
            };
        }

        if channels.admits(peer_id) {
            self.observations.remove(peer_id);
            return MembershipAction::Refresh {
                peer_id: peer_id.clone(),
            };
        }

        self.observations.insert(
            peer_id.clone(),
            CandidateObservation {
                peer_id: peer_id.clone(),
                layer_ranges: layer_ranges.clone(),
                seen_at: now,
            },
        );
        MembershipAction::Dropped {
            peer_id: peer_id.clone(),
            reason: MembershipDrop::UnauthorizedPeer,
        }
    }

    /// Decides what a `WorkerLeft` means: only an admitted peer can be un-scheduled.
    pub fn observe_left(
        &mut self,
        msg: &WorkerGossipMsg,
        source: Option<&str>,
        channels: &MembershipChannels<'_>,
        now: Instant,
    ) -> MembershipAction {
        let WorkerGossipMsg::WorkerLeft { peer_id } = msg else {
            return MembershipAction::Dropped {
                peer_id: message_peer_id(msg),
                reason: MembershipDrop::MalformedPayload(
                    "an announcement is not a leave".to_string(),
                ),
            };
        };

        if let Some(dropped) = self.check_flood(source.unwrap_or(peer_id), now) {
            return dropped;
        }
        // S58: a leave may be announced by the peer itself **or** by the orchestrator (the only
        // entity that sees `ConnectionClosed`). `check_source` stays strict for announcements.
        if let Some(dropped) = check_left_source(peer_id, source, channels.orchestrator_peer_id) {
            return dropped;
        }
        if peer_id.parse::<tessella_network::NetworkPeerId>().is_err() {
            return MembershipAction::Dropped {
                peer_id: peer_id.clone(),
                reason: MembershipDrop::MalformedPayload("not a libp2p peer id".to_string()),
            };
        }
        if peer_id == channels.own_peer_id {
            return MembershipAction::Dropped {
                peer_id: peer_id.clone(),
                reason: MembershipDrop::OwnAnnouncement,
            };
        }

        if channels.admits(peer_id) {
            self.observations.remove(peer_id);
            return MembershipAction::Suppress {
                peer_id: peer_id.clone(),
                window: GOSSIP_SUPPRESSION_TTL,
            };
        }

        MembershipAction::Dropped {
            peer_id: peer_id.clone(),
            reason: MembershipDrop::UnknownPeer,
        }
    }

    /// Drops the observations whose TTL passed and returns the peers that were forgotten.
    ///
    /// The P2P loop calls it on its heartbeat tick (the same tick that expires the routing table's
    /// suppressions), so the memory of this reducer is bounded by construction.
    pub fn expire(&mut self, now: Instant) -> Vec<String> {
        let expired: Vec<String> = self
            .observations
            .values()
            .filter(|observation| {
                now.saturating_duration_since(observation.seen_at) >= GOSSIP_OBSERVATION_TTL
            })
            .map(|observation| observation.peer_id.clone())
            .collect();
        for peer_id in &expired {
            self.observations.remove(peer_id);
        }
        self.budget.retain(|_, window| {
            now.saturating_duration_since(window.started) < MEMBERSHIP_RATE_WINDOW
        });
        expired
    }

    /// Charges one message to `emitter`'s budget, or returns the drop that says the window is full.
    fn check_flood(&mut self, emitter: &str, now: Instant) -> Option<MembershipAction> {
        let window = self
            .budget
            .entry(emitter.to_string())
            .or_insert(RateWindow {
                messages: 0,
                started: now,
            });
        if now.saturating_duration_since(window.started) >= MEMBERSHIP_RATE_WINDOW {
            window.messages = 0;
            window.started = now;
        }
        window.messages += 1;
        if window.messages > MEMBERSHIP_MESSAGES_PER_WINDOW {
            return Some(MembershipAction::Dropped {
                peer_id: emitter.to_string(),
                reason: MembershipDrop::RateLimited {
                    window_secs: MEMBERSHIP_RATE_WINDOW.as_secs(),
                    limit: MEMBERSHIP_MESSAGES_PER_WINDOW,
                },
            });
        }
        None
    }
}

/// The peer id a message is about, whatever its variant (used to fill the drop reason).
fn message_peer_id(msg: &WorkerGossipMsg) -> String {
    match msg {
        WorkerGossipMsg::WorkerJoined { peer_id, .. } => peer_id.clone(),
        WorkerGossipMsg::WorkerLeft { peer_id } => peer_id.clone(),
    }
}

/// The declared identity must be the one gossipsub authenticated for the message.
fn check_source(declared: &str, source: Option<&str>) -> Option<MembershipAction> {
    match source {
        Some(source) if source == declared => None,
        other => Some(MembershipAction::Dropped {
            peer_id: declared.to_string(),
            reason: MembershipDrop::SourceMismatch {
                source: other.map(|value| value.to_string()),
            },
        }),
    }
}

/// Who may announce that `declared` **left** (`S58`).
///
/// The emitter must be the peer itself or the **orchestrator**: Musivum publishes `WorkerLeft` on
/// `ConnectionClosed` (`musivum/crates/musivum/src/p2p_manager.rs`) about a third node, and that is
/// the only leave signal the system produces. Requiring `source == declared` for it — the rule of
/// [`check_source`], which is right for a self-announcement — made every departure from the tracker
/// unprocessable: measured in S58 as
/// `⚠️ Membership: dropped the message about <peer> (SourceMismatch { source: Some(<tracker>) })`,
/// so the suppression of `TASK-16`/S22 never ran in a real mesh.
///
/// A peer still cannot announce a departure on behalf of another peer: only these two identities
/// pass, and the orchestrator is the one the whole plane already trusts as the eviction authority.
fn check_left_source(
    declared: &str,
    source: Option<&str>,
    orchestrator: Option<&str>,
) -> Option<MembershipAction> {
    match source {
        Some(source) if source == declared => None,
        Some(source) if orchestrator == Some(source) => None,
        other => Some(MembershipAction::Dropped {
            peer_id: declared.to_string(),
            reason: MembershipDrop::SourceMismatch {
                source: other.map(|value| value.to_string()),
            },
        }),
    }
}

/// The address/identity validation of one announcement, without DNS and without inventing values.
fn validate_announcement(
    peer_id: &str,
    http_addr: &str,
    p2p_addr: &str,
    layer_ranges: &[(usize, usize)],
) -> Result<(), MembershipDrop> {
    if peer_id.parse::<tessella_network::NetworkPeerId>().is_err() {
        return Err(MembershipDrop::MalformedPayload(format!(
            "'{peer_id}' is not a libp2p peer id"
        )));
    }
    if p2p_addr
        .parse::<tessella_network::NetworkAddress>()
        .is_err()
        || tessella_core::net::multiaddr_ip(p2p_addr).is_none()
    {
        return Err(MembershipDrop::MalformedPayload(format!(
            "'{p2p_addr}' is not a dialable multiaddr"
        )));
    }
    match tessella_core::net::tcp_port_of(p2p_addr) {
        None | Some(0) => {
            return Err(MembershipDrop::MalformedPayload(format!(
                "'{p2p_addr}' has no usable TCP port"
            )))
        }
        Some(_) => {}
    }
    match tessella_core::net::url_scheme_and_port(http_addr) {
        Some((scheme, _)) if scheme == "http" || scheme == "https" => {}
        _ => {
            return Err(MembershipDrop::MalformedPayload(format!(
                "'{http_addr}' is not an http(s) URL"
            )))
        }
    }
    // `TASK-96`/S102: every range of the announced shard is checked, not only the first. A gossip
    // message is a routing input, and half a checked shard is a peer indexed under a range nobody
    // validated.
    if layer_ranges.is_empty() {
        return Err(MembershipDrop::MalformedPayload(
            "the announcement declares no layer range".to_string(),
        ));
    }
    for &(start, end) in layer_ranges {
        if start >= end {
            return Err(MembershipDrop::MalformedPayload(format!(
                "layer range ({start}, {end}) is empty or inverted"
            )));
        }
    }
    Ok(())
}

/// Applies an action to the routing table. Kept separate from the reducer so both halves can be
/// tested without a `Swarm`: the handler is then decode → observe → apply → log.
pub fn apply_to_routing_table(
    action: &MembershipAction,
    table: &mut RoutingTable,
    now: Instant,
) -> Applied {
    match action {
        MembershipAction::Refresh { peer_id } => {
            if !table.contains_peer(peer_id) {
                return Applied::Nothing;
            }
            let was_suppressed = table.unsuppress(peer_id);
            Applied::EligibilityRestored {
                peer_id: peer_id.clone(),
                was_suppressed,
            }
        }
        MembershipAction::Suppress { peer_id, window } => {
            if !table.contains_peer(peer_id) {
                return Applied::Nothing;
            }
            table.suppress(peer_id, now, *window);
            Applied::EligibilityRemoved {
                peer_id: peer_id.clone(),
                window: *window,
            }
        }
        MembershipAction::Dropped { .. } => Applied::Nothing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing_table::RoutingTable;
    use std::time::{Duration, Instant};
    use tessella_core::models::{LayerPeer, WorkerGossipMsg};

    const PEER_A: &str = "12D3KooWCPEF1tkntHNUkJXAWx1j1Jz4ZoWJDMvLETmsZPtBxwax";
    const PEER_B: &str = "12D3KooWSvvoYMBy7s3iGSqxAj4sdjQDZaoPgAWMCesEvZURhAPQ";
    const STRANGER: &str = "12D3KooWSZBii7nmGoNdWBDjxpDFQukc1yjxsPeD1FNksrsNRq8h";
    const US: &str = "12D3KooWGoXe3Ax4UFNnsQ7f7ABnyj65xF7ovMNnsPft382YGWBx";
    const ORCHESTRATOR: &str = "12D3KooWJFrTi78KyNv6kAyVZ5JvXQprVY4wNynJLoZwx5VE9ojm";

    fn layer_peer(peer_id: &str, layer_range: (usize, usize)) -> LayerPeer {
        LayerPeer {
            peer_id: peer_id.to_string(),
            http_addr: format!("http://10.0.0.7:{}", 9000 + layer_range.0),
            p2p_addr: format!("/ip4/10.0.0.7/tcp/{}", 9000 + layer_range.0),
            layer_ranges: vec![layer_range],
            expert_range: None,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        }
    }

    /// A routing table holding `peers`, i.e. the peers the tracker admitted (`/register` + `/peers`).
    fn table_with(peers: &[LayerPeer]) -> RoutingTable {
        let mut table = RoutingTable::new();
        table.replace_all(peers.to_vec());
        table
    }

    fn channels<'a>(
        own: &'a str,
        orchestrator: Option<&'a str>,
        table: &'a RoutingTable,
    ) -> MembershipChannels<'a> {
        MembershipChannels {
            own_peer_id: own,
            orchestrator_peer_id: orchestrator,
            routing_table: table,
        }
    }

    fn joined(
        peer_id: &str,
        http_addr: &str,
        p2p_addr: &str,
        layer_range: (usize, usize),
    ) -> WorkerGossipMsg {
        joined_with_ranges(peer_id, http_addr, p2p_addr, vec![layer_range])
    }

    /// A joining peer that serves more than one range (`TASK-96`/S102).
    fn joined_with_ranges(
        peer_id: &str,
        http_addr: &str,
        p2p_addr: &str,
        layer_ranges: Vec<(usize, usize)>,
    ) -> WorkerGossipMsg {
        WorkerGossipMsg::WorkerJoined {
            peer_id: peer_id.to_string(),
            http_addr: http_addr.to_string(),
            p2p_addr: p2p_addr.to_string(),
            layer_ranges,
        }
    }

    fn joined_ok(peer_id: &str) -> WorkerGossipMsg {
        joined(
            peer_id,
            "http://10.0.0.7:10001",
            "/ip4/10.0.0.7/tcp/9001",
            (0, 12),
        )
    }

    fn left(peer_id: &str) -> WorkerGossipMsg {
        WorkerGossipMsg::WorkerLeft {
            peer_id: peer_id.to_string(),
        }
    }

    #[test]
    fn an_announcement_from_an_admitted_peer_refreshes_its_membership() {
        let table = table_with(&[layer_peer(PEER_B, (0, 12))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();

        let action =
            membership.observe_joined(&joined_ok(PEER_B), Some(PEER_B), &channels, Instant::now());

        assert_eq!(
            action,
            MembershipAction::Refresh {
                peer_id: PEER_B.to_string()
            }
        );
        assert_eq!(membership.candidate_count(), 0);
    }

    #[test]
    fn a_leave_marks_the_peer_not_eligible_without_deleting_it() {
        let mut table = table_with(&[layer_peer(PEER_B, (0, 12))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();
        let now = Instant::now();
        assert!(
            table.find_target_for_layer(3).selected.is_some(),
            "eligible to start with"
        );

        let action = membership.observe_left(&left(PEER_B), Some(PEER_B), &channels, now);

        assert_eq!(
            action,
            MembershipAction::Suppress {
                peer_id: PEER_B.to_string(),
                window: GOSSIP_SUPPRESSION_TTL,
            }
        );
        assert_eq!(
            apply_to_routing_table(&action, &mut table, now),
            Applied::EligibilityRemoved {
                peer_id: PEER_B.to_string(),
                window: GOSSIP_SUPPRESSION_TTL,
            }
        );
        assert!(
            table.contains_peer(PEER_B),
            "the entry is kept: an out-of-order leave must not lose a peer"
        );
        assert!(table.is_suppressed(PEER_B));
        assert!(
            table.find_target_for_layer(3).is_empty(),
            "the router stops choosing it at once"
        );
    }

    #[test]
    fn a_join_right_after_a_leave_stops_suppressing_the_peer() {
        let mut table = table_with(&[layer_peer(PEER_B, (0, 12))]);
        let mut membership = GossipMembership::new();
        let now = Instant::now();

        let leave = {
            let channels = channels(US, Some(ORCHESTRATOR), &table);
            membership.observe_left(&left(PEER_B), Some(PEER_B), &channels, now)
        };
        apply_to_routing_table(&leave, &mut table, now);
        assert!(table.is_suppressed(PEER_B));

        let join = {
            let channels = channels(US, Some(ORCHESTRATOR), &table);
            membership.observe_joined(&joined_ok(PEER_B), Some(PEER_B), &channels, now)
        };

        assert!(matches!(leave, MembershipAction::Suppress { .. }));
        assert_eq!(
            join,
            MembershipAction::Refresh {
                peer_id: PEER_B.to_string()
            },
            "the newest observation wins"
        );
        assert_eq!(
            apply_to_routing_table(&join, &mut table, now),
            Applied::EligibilityRestored {
                peer_id: PEER_B.to_string(),
                was_suppressed: true,
            }
        );
        assert!(!table.is_suppressed(PEER_B));
        assert!(
            table.find_target_for_layer(3).selected.is_some(),
            "eligible again without waiting for the TTL"
        );
    }

    #[test]
    fn an_announcement_of_a_stranger_is_dropped_and_is_never_a_route() {
        let mut table = table_with(&[]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();

        let action = membership.observe_joined(
            &joined_ok(STRANGER),
            Some(STRANGER),
            &channels,
            Instant::now(),
        );

        assert_eq!(
            action,
            MembershipAction::Dropped {
                peer_id: STRANGER.to_string(),
                reason: MembershipDrop::UnauthorizedPeer,
            }
        );
        assert_eq!(
            apply_to_routing_table(&action, &mut table, Instant::now()),
            Applied::Nothing
        );
        assert!(table.peers().is_empty(), "gossip never onboards");
        assert_eq!(
            membership.candidate_count(),
            1,
            "recorded as an observation only"
        );
        assert_eq!(
            membership
                .observation_of(STRANGER)
                .map(|o| o.layer_ranges.clone()),
            Some(vec![(0, 12)]),
            "the observation carries what the peer claimed, for the log"
        );
    }

    #[test]
    fn an_announcement_without_an_authenticated_source_is_dropped() {
        let table = table_with(&[layer_peer(PEER_A, (0, 12))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();

        let action = membership.observe_joined(&joined_ok(PEER_A), None, &channels, Instant::now());

        assert!(matches!(
            action,
            MembershipAction::Dropped {
                reason: MembershipDrop::SourceMismatch { source: None },
                ..
            }
        ));
    }

    #[test]
    fn announcements_with_unusable_addresses_or_ranges_are_dropped() {
        let table = table_with(&[layer_peer(PEER_A, (0, 12))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();
        let now = Instant::now();

        // (message, authenticated source): the source matches the declared id in every case, so what
        // is under test is the payload validation and nothing else.
        let cases = [
            (
                joined(PEER_A, "http://10.0.0.7:10001", "not-a-multiaddr", (0, 12)),
                PEER_A,
            ),
            (
                joined(
                    PEER_A,
                    "http://10.0.0.7:10001",
                    "/ip4/10.0.0.7/tcp/0",
                    (0, 12),
                ),
                PEER_A,
            ),
            (
                joined(
                    PEER_A,
                    "http://10.0.0.7:10001",
                    "/ip4/10.0.0.7/udp/9001",
                    (0, 12),
                ),
                PEER_A,
            ),
            (
                joined(
                    PEER_A,
                    "http://10.0.0.7:10001",
                    "/ip4/musivum.dev/tcp/9001",
                    (0, 12),
                ),
                PEER_A,
            ),
            (
                joined(PEER_A, "not-a-url", "/ip4/10.0.0.7/tcp/9001", (0, 12)),
                PEER_A,
            ),
            (
                joined(
                    PEER_A,
                    "ftp://10.0.0.7:21",
                    "/ip4/10.0.0.7/tcp/9001",
                    (0, 12),
                ),
                PEER_A,
            ),
            (
                joined(
                    PEER_A,
                    "http://10.0.0.7:10001",
                    "/ip4/10.0.0.7/tcp/9001",
                    (12, 12),
                ),
                PEER_A,
            ),
            (
                joined(
                    PEER_A,
                    "http://10.0.0.7:10001",
                    "/ip4/10.0.0.7/tcp/9001",
                    (12, 3),
                ),
                PEER_A,
            ),
            (
                joined(
                    "",
                    "http://10.0.0.7:10001",
                    "/ip4/10.0.0.7/tcp/9001",
                    (0, 12),
                ),
                "",
            ),
            (
                joined(
                    "not-a-peer-id",
                    "http://10.0.0.7:10001",
                    "/ip4/10.0.0.7/tcp/9001",
                    (0, 12),
                ),
                "not-a-peer-id",
            ),
        ];

        for (index, (msg, source)) in cases.iter().enumerate() {
            let action = membership.observe_joined(msg, Some(source), &channels, now);
            assert!(
                matches!(
                    action,
                    MembershipAction::Dropped {
                        reason: MembershipDrop::MalformedPayload(_),
                        ..
                    }
                ),
                "case {index} must be dropped as malformed, got {action:?}"
            );
        }
        assert_eq!(
            membership.candidate_count(),
            0,
            "a malformed payload is not even an observation"
        );
    }

    #[test]
    fn our_own_announcement_is_ignored() {
        let table = table_with(&[]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();

        let action = membership.observe_joined(&joined_ok(US), Some(US), &channels, Instant::now());

        assert_eq!(
            action,
            MembershipAction::Dropped {
                peer_id: US.to_string(),
                reason: MembershipDrop::OwnAnnouncement,
            }
        );
    }

    #[test]
    fn the_orchestrator_is_admitted_without_a_routing_table_entry() {
        let table = table_with(&[]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();
        let now = Instant::now();

        let join =
            membership.observe_joined(&joined_ok(ORCHESTRATOR), Some(ORCHESTRATOR), &channels, now);
        let leave =
            membership.observe_left(&left(ORCHESTRATOR), Some(ORCHESTRATOR), &channels, now);

        assert!(matches!(join, MembershipAction::Refresh { .. }));
        assert!(matches!(leave, MembershipAction::Suppress { .. }));
    }

    #[test]
    fn a_leave_from_an_unknown_peer_is_ignored() {
        let table = table_with(&[layer_peer(PEER_A, (0, 12))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();

        let action =
            membership.observe_left(&left(STRANGER), Some(STRANGER), &channels, Instant::now());

        assert_eq!(
            action,
            MembershipAction::Dropped {
                peer_id: STRANGER.to_string(),
                reason: MembershipDrop::UnknownPeer,
            }
        );
    }

    #[test]
    fn a_peer_cannot_announce_another_peer() {
        let table = table_with(&[layer_peer(PEER_A, (0, 12))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();

        // `attacker` (PEER_B) claims to be the admitted PEER_A.
        let action =
            membership.observe_joined(&joined_ok(PEER_A), Some(PEER_B), &channels, Instant::now());

        assert_eq!(
            action,
            MembershipAction::Dropped {
                peer_id: PEER_A.to_string(),
                reason: MembershipDrop::SourceMismatch {
                    source: Some(PEER_B.to_string()),
                },
            }
        );
    }

    /// `S58`: the orchestrator is the node that sees `ConnectionClosed`, so its `WorkerLeft` about a
    /// **third** node is the only departure signal the system produces. Before the fix it was
    /// dropped as `SourceMismatch` and the suppression path of `TASK-16`/S22 never ran in a real
    /// mesh (measured with two workers in `scripts/e2e/swarm.sh --scenario membership_two_nodes`).
    #[test]
    fn the_orchestrator_can_announce_the_departure_of_a_third_peer() {
        let mut table = table_with(&[layer_peer(PEER_B, (0, 12))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();
        let now = Instant::now();

        let action = membership.observe_left(&left(PEER_B), Some(ORCHESTRATOR), &channels, now);

        assert_eq!(
            action,
            MembershipAction::Suppress {
                peer_id: PEER_B.to_string(),
                window: GOSSIP_SUPPRESSION_TTL,
            }
        );
        assert!(matches!(
            apply_to_routing_table(&action, &mut table, now),
            Applied::EligibilityRemoved { .. }
        ));
        assert!(table.contains_peer(PEER_B), "still not a delete");
    }

    /// The relaxation of the leave is not a door for peers: a node still cannot announce that
    /// *another* node left.
    #[test]
    fn a_peer_still_cannot_announce_the_departure_of_another_peer() {
        let table = table_with(&[layer_peer(PEER_A, (0, 12)), layer_peer(PEER_B, (12, 24))]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();

        let action =
            membership.observe_left(&left(PEER_B), Some(PEER_A), &channels, Instant::now());

        assert_eq!(
            action,
            MembershipAction::Dropped {
                peer_id: PEER_B.to_string(),
                reason: MembershipDrop::SourceMismatch {
                    source: Some(PEER_A.to_string()),
                },
            }
        );
    }

    /// Until `/p2p_id` has told us who the orchestrator is, nobody is allowed to play it.
    #[test]
    fn without_a_discovered_orchestrator_a_third_party_leave_is_dropped() {
        let table = table_with(&[layer_peer(PEER_B, (0, 12))]);
        let channels = channels(US, None, &table);
        let mut membership = GossipMembership::new();

        let action =
            membership.observe_left(&left(PEER_B), Some(ORCHESTRATOR), &channels, Instant::now());

        assert!(matches!(
            action,
            MembershipAction::Dropped {
                reason: MembershipDrop::SourceMismatch { .. },
                ..
            }
        ));
    }

    #[test]
    fn a_flood_from_one_peer_is_rate_limited() {
        let table = table_with(&[]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();
        let now = Instant::now();

        for _ in 0..MEMBERSHIP_MESSAGES_PER_WINDOW {
            let action =
                membership.observe_joined(&joined_ok(STRANGER), Some(STRANGER), &channels, now);
            assert!(matches!(
                action,
                MembershipAction::Dropped {
                    reason: MembershipDrop::UnauthorizedPeer,
                    ..
                }
            ));
        }

        let flooded =
            membership.observe_joined(&joined_ok(STRANGER), Some(STRANGER), &channels, now);

        assert_eq!(
            flooded,
            MembershipAction::Dropped {
                peer_id: STRANGER.to_string(),
                reason: MembershipDrop::RateLimited {
                    window_secs: MEMBERSHIP_RATE_WINDOW.as_secs(),
                    limit: MEMBERSHIP_MESSAGES_PER_WINDOW,
                },
            }
        );

        // The window slides: a message after it is observed again.
        let later = now + MEMBERSHIP_RATE_WINDOW + Duration::from_secs(1);
        let action =
            membership.observe_joined(&joined_ok(STRANGER), Some(STRANGER), &channels, later);
        assert!(matches!(
            action,
            MembershipAction::Dropped {
                reason: MembershipDrop::UnauthorizedPeer,
                ..
            }
        ));
    }

    #[test]
    fn observations_of_unadmitted_peers_expire() {
        let table = table_with(&[]);
        let channels = channels(US, Some(ORCHESTRATOR), &table);
        let mut membership = GossipMembership::new();
        let now = Instant::now();

        membership.observe_joined(&joined_ok(STRANGER), Some(STRANGER), &channels, now);
        assert_eq!(membership.candidate_count(), 1);

        let expired = membership.expire(now + GOSSIP_OBSERVATION_TTL - Duration::from_secs(1));
        assert!(expired.is_empty(), "not yet");
        assert_eq!(membership.candidate_count(), 1);

        let expired = membership.expire(now + GOSSIP_OBSERVATION_TTL + Duration::from_secs(1));
        assert_eq!(expired, vec![STRANGER.to_string()]);
        assert_eq!(membership.candidate_count(), 0);
    }
}
