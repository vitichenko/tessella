use std::collections::HashMap;
use std::time::Instant;
use tessella_core::models::LayerPeer;
pub use tessella_core::routing::{
    select_route_with, RouteSelection, NO_EXPLORATION, UNMEASURED_LATENCY,
};

pub struct RoutingTable {
    /// layer_id → peers that serve it (lookup O(1))
    layers: HashMap<usize, Vec<LayerPeer>>,
    /// flat list for ordered iteration
    peers: Vec<LayerPeer>,
    updated_at: Instant,
    /// peer_id -> Exponential Weighted Moving Average (EWMA) latency
    latencies: HashMap<String, std::time::Duration>,
    /// Peers that must **not** be chosen for routing until this instant (`TASK-16` · S22).
    ///
    /// It is how a `WorkerLeft` seen on the membership channel is honoured: the peer stops being
    /// eligible *now* and its entry is kept, so an out-of-order message (or a lie from a single
    /// node) cannot delete a peer that the tracker still lists. The mark expires on its own
    /// ([`RoutingTable::expire_suppressions`]) and a later `WorkerJoined` clears it
    /// ([`RoutingTable::unsuppress`]): the newest observation wins.
    suppressed: HashMap<String, Instant>,
}

impl RoutingTable {
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            peers: Vec::new(),
            updated_at: Instant::now(),
            latencies: HashMap::new(),
            suppressed: HashMap::new(),
        }
    }

    /// Rebuilds both views from a whole topology (`TASK-96`/S102: every range of every peer).
    pub fn replace_all(&mut self, peers: Vec<LayerPeer>) {
        self.layers.clear();
        for peer in &peers {
            for &(start, end) in &peer.layer_ranges {
                for i in start..end {
                    self.layers.entry(i).or_default().push(peer.clone());
                }
            }
        }
        self.peers = peers;
        self.updated_at = Instant::now();
    }

    /// Inserts or updates one peer. It lands under **every** layer of every range it declares, so a
    /// peer holding two disjoint chunks is a candidate for both stretches (`TASK-96`/S102) — which
    /// is the whole point of the field being a set.
    ///
    /// The refresh is in place: a peer whose ranges shrank keeps the entries of the layers it no
    /// longer serves until the next `replace_all` (the topology refresh), exactly as it did when a
    /// single range moved. That is safe because a hop sent to a range the peer does not serve is
    /// **refused** by the receiver (invariant 17) rather than silently answered by another range.
    pub fn insert_peer(&mut self, peer: LayerPeer) {
        for &(start, end) in &peer.layer_ranges {
            for i in start..end {
                let entry = self.layers.entry(i).or_default();
                // replace existing entry for this peer_id if present, otherwise push
                if let Some(pos) = entry.iter().position(|p| p.peer_id == peer.peer_id) {
                    entry[pos] = peer.clone();
                } else {
                    entry.push(peer.clone());
                }
            }
        }
        // The flat view is refreshed **in place** too (`TASK-72` · S79). The loop above already
        // replaced the record inside every layer this peer covers, and leaving this one behind makes
        // the two views disagree: the router inserts peers through this path when it falls back to
        // the cached topology (`http_pipeline_router.rs`), so a `queue_depth` read from `self.peers`
        // would be the one from *before* the fallback — precisely the kind of stale field the
        // routing policy now consumes. Replacing in place rather than pushing keeps the candidate
        // order of the flat view (which `peers()` hands out) untouched.
        if let Some(pos) = self.peers.iter().position(|p| p.peer_id == peer.peer_id) {
            self.peers[pos] = peer;
        } else {
            self.peers.push(peer);
        }
        self.updated_at = std::time::Instant::now();
    }

    pub fn update_latency(&mut self, peer_id: &str, latency: std::time::Duration) {
        let alpha = 0.3;
        let entry = self.latencies.entry(peer_id.to_string()).or_insert(latency);
        *entry = std::time::Duration::from_secs_f64(
            alpha * latency.as_secs_f64() + (1.0 - alpha) * entry.as_secs_f64(),
        );
    }

    /// The latency this node publishes for `peer_id`: the EWMA if it has measured it, and
    /// [`UNMEASURED_LATENCY`] if it never has.
    ///
    /// The number does not change in `S79` — what changes is that the *selection* no longer reads it:
    /// it reads [`RoutingTable::measured_latency`], where "never measured" is a fact (`None`) instead
    /// of a value nobody can tell apart from a real 200 ms.
    pub fn get_latency(&self, peer_id: &str) -> std::time::Duration {
        self.measured_latency(peer_id).unwrap_or(UNMEASURED_LATENCY)
    }

    /// The EWMA of `peer_id` **if this node ever measured it**, `None` otherwise (`TASK-72` · S79).
    ///
    /// It is the latency signal the routing policy ranks with: a peer nobody has timed enters with
    /// [`UNMEASURED_LATENCY`] as a prior and is reachable through the exploration share, whereas a
    /// peer that failed carries its penalty as any other measurement (10 s).
    pub fn measured_latency(&self, peer_id: &str) -> Option<std::time::Duration> {
        self.latencies.get(peer_id).copied()
    }

    /// The depth of the queue of the **live** peer: the field the mesh publishes (`TASK-72` · S79).
    ///
    /// It is read from `self.peers` and not from the candidate of the moment on purpose: candidates
    /// are clones of a lookup and a `/peers` refresh replaces the entries, so the live record is the
    /// newest view this node has of that peer.
    fn queue_depth_of(&self, peer_id: &str) -> usize {
        self.peers
            .iter()
            .find(|peer| peer.peer_id == peer_id)
            .map(|peer| peer.queue_depth)
            .unwrap_or(0)
    }

    pub fn penalize_peer(&mut self, peer_id: &str) {
        self.latencies
            .insert(peer_id.to_string(), std::time::Duration::from_secs(10));
    }

    /// Marks `peer_id` as **not eligible for routing** until `now + window` (`TASK-16` · S22).
    ///
    /// Called when the membership channel reports a `WorkerLeft`. The peer stays in the table on
    /// purpose: only un-scheduling changes, so a message that arrives out of order (or from a
    /// single lying node) cannot remove a peer the tracker still lists. Suppressing a peer that is
    /// not in the table is harmless and creates no entry.
    pub fn suppress(&mut self, peer_id: &str, now: Instant, window: std::time::Duration) {
        self.suppressed.insert(peer_id.to_string(), now + window);
    }

    /// Clears the mark of `peer_id`. Returns `true` when there was one.
    ///
    /// Called when the peer announces itself again: a join is newer evidence than a leave.
    pub fn unsuppress(&mut self, peer_id: &str) -> bool {
        self.suppressed.remove(peer_id).is_some()
    }

    /// `true` while the peer is marked as not eligible.
    pub fn is_suppressed(&self, peer_id: &str) -> bool {
        self.suppressed.contains_key(peer_id)
    }

    /// Drops the marks whose window has passed and returns the peers that became eligible again.
    ///
    /// The P2P loop calls it on its heartbeat tick: the mark is bounded, so a peer nobody ever
    /// re-announces comes back on its own (the tracker's `/peers` remains the source of truth about
    /// who is really in the mesh).
    pub fn expire_suppressions(&mut self, now: Instant) -> Vec<String> {
        let expired: Vec<String> = self
            .suppressed
            .iter()
            .filter(|(_, until)| **until <= now)
            .map(|(peer_id, _)| peer_id.clone())
            .collect();
        for peer_id in &expired {
            self.suppressed.remove(peer_id);
        }
        expired
    }

    /// Returns a peer for the given layer prioritizing the best weight (latency and queue).
    ///
    /// Same policy as the route selectors (`S43`): a peer that announces an `expert_range` is not a
    /// candidate, because it cannot serve the whole layer. No exploration share is drawn here: the
    /// helper is used by assertions, and the router is the caller that owns the draw.
    #[cfg(test)]
    pub fn find_peer_for_layer(&self, layer_idx: usize) -> Option<LayerPeer> {
        let mut candidates = self.layers.get(&layer_idx)?.clone();
        candidates.retain(|peer| !self.is_suppressed(&peer.peer_id));
        select_route_with(
            &candidates,
            |peer_id| self.measured_latency(peer_id),
            |peer_id| self.queue_depth_of(peer_id),
            NO_EXPLORATION,
        )
        .selected
    }

    /// Returns the route for `layer_idx`, prioritizing the best weight (latency and queue).
    ///
    /// No exploration share either: this entry point answers «who covers this layer», and its callers
    /// (membership) must not depend on a draw. The router calls
    /// [`RoutingTable::find_target_for_layer_excluding`] with the draw it has made for the hop.
    pub fn find_target_for_layer(&self, layer_idx: usize) -> RouteSelection {
        self.find_target_for_layer_excluding(
            layer_idx,
            &std::collections::HashSet::new(),
            NO_EXPLORATION,
        )
    }

    /// Returns the route for `layer_idx`, excluding given peer IDs.
    ///
    /// Peers marked as not eligible (`WorkerLeft` seen on the membership channel) are never
    /// candidates: the router prefers to wait for another peer instead of sending the hop to one
    /// that just announced it is leaving. Peers that announce an `expert_range` are not candidates
    /// either, and come back in [`RouteSelection::unsupported_expert_shards`] so the caller can say
    /// why instead of waiting for a peer that can never serve the layer (`S43`/`TASK-37`).
    ///
    /// `draw ∈ [0, 1)` is the caller's exploration draw (`TASK-72` · S79). It is passed in instead of
    /// generated here because the share belongs to the *policy* ([`select_route_with`]) and the
    /// filtering belongs to the table: a suppressed peer is removed **before** the policy sees it, so
    /// the share can never be a back door to a node that announced it is leaving.
    pub fn find_target_for_layer_excluding(
        &self,
        layer_idx: usize,
        excluded: &std::collections::HashSet<String>,
        draw: f64,
    ) -> RouteSelection {
        let Some(candidates) = self.layers.get(&layer_idx) else {
            return RouteSelection::default();
        };
        let candidates: Vec<_> = candidates
            .iter()
            .filter(|p| {
                !p.canary && !excluded.contains(&p.peer_id) && !self.is_suppressed(&p.peer_id)
            })
            .cloned()
            .collect();
        select_route_with(
            &candidates,
            |peer_id| self.measured_latency(peer_id),
            |peer_id| self.queue_depth_of(peer_id),
            draw,
        )
    }

    /// Returns all available peers for a specific layer.
    ///
    /// This is a raw **view of the topology** (used to answer «is there anybody for this layer?»),
    /// not a route selector: eligibility is enforced by the `find_*` methods above.
    pub fn get_peers_for_layer(&self, layer_idx: usize) -> Vec<LayerPeer> {
        self.layers.get(&layer_idx).cloned().unwrap_or_default()
    }

    /// Every peer currently known to this table, whatever its layer range.
    ///
    /// Used by the callback guard (H-10) to answer «is this destination a node of this mesh?»:
    /// `LayerPeer::http_addr` is the URL the peer announced, and `/peers` is what fills it.
    pub fn peers(&self) -> &[LayerPeer] {
        &self.peers
    }

    pub fn contains_peer(&self, peer_id: &str) -> bool {
        self.peers.iter().any(|p| p.peer_id == peer_id)
    }

    /// Returns how long ago the table was last updated.
    pub fn age(&self) -> std::time::Duration {
        self.updated_at.elapsed()
    }
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_routing_and_ewma() {
        let mut table = RoutingTable::new();
        let peer1 = LayerPeer {
            peer_id: "peer1".to_string(),
            http_addr: "http://peer1".to_string(),
            p2p_addr: "/ip4/1.1.1.1/tcp/9000".to_string(),
            layer_ranges: vec![(0, 10)],
            expert_range: None,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        };
        let peer2 = LayerPeer {
            peer_id: "peer2".to_string(),
            http_addr: "http://peer2".to_string(),
            p2p_addr: "/ip4/1.1.1.2/tcp/9000".to_string(),
            layer_ranges: vec![(0, 10)],
            expert_range: None,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        };

        table.insert_peer(peer1);
        table.insert_peer(peer2);

        // Initially both get default latency 200ms
        assert_eq!(
            table.get_latency("peer1"),
            std::time::Duration::from_millis(200)
        );
        assert_eq!(
            table.get_latency("peer2"),
            std::time::Duration::from_millis(200)
        );

        // Update peer1 latency to 100ms
        table.update_latency("peer1", std::time::Duration::from_millis(100));
        // First measurement is set directly to 100ms
        assert_eq!(
            table.get_latency("peer1"),
            std::time::Duration::from_millis(100)
        );

        // Update peer2 latency to 50ms
        table.update_latency("peer2", std::time::Duration::from_millis(50));
        // First measurement is set directly to 50ms
        assert_eq!(
            table.get_latency("peer2"),
            std::time::Duration::from_millis(50)
        );

        // Sorting check: peer2 (50ms) should be prioritized over peer1 (100ms)
        let chosen = table.find_peer_for_layer(5).unwrap();
        assert_eq!(chosen.peer_id, "peer2");

        // Now blend: update peer1 with 200ms
        table.update_latency("peer1", std::time::Duration::from_millis(200));
        // EWMA: 0.3 * 200 + 0.7 * 100 = 60 + 70 = 130ms
        assert_eq!(
            table.get_latency("peer1"),
            std::time::Duration::from_millis(130)
        );

        // Penalize peer2
        table.penalize_peer("peer2");
        assert_eq!(
            table.get_latency("peer2"),
            std::time::Duration::from_secs(10)
        );

        // Sorting check after penalization: peer1 (130ms) should be prioritized over peer2 (10s)
        let chosen_after = table.find_peer_for_layer(5).unwrap();
        assert_eq!(chosen_after.peer_id, "peer1");
    }

    /// A `WorkerLeft` seen on the membership channel marks the peer as *not eligible* without
    /// deleting it (`TASK-16`): the router stops choosing it, the entry stays.
    #[test]
    fn a_suppressed_peer_is_not_chosen_for_routing_but_is_kept_in_the_table() {
        let mut table = RoutingTable::new();
        table.insert_peer(peer("peer1", "/ip4/1.1.1.1/tcp/9000"));
        table.insert_peer(peer("peer2", "/ip4/1.1.1.2/tcp/9000"));
        table.update_latency("peer1", std::time::Duration::from_millis(50));
        table.update_latency("peer2", std::time::Duration::from_millis(120));
        assert_eq!(
            table.find_peer_for_layer(5).unwrap().peer_id,
            "peer1",
            "lowest latency wins to start with"
        );

        let now = std::time::Instant::now();
        table.suppress("peer1", now, std::time::Duration::from_secs(300));

        assert!(table.is_suppressed("peer1"));
        assert!(
            table.contains_peer("peer1"),
            "the entry is kept: only the eligibility changes"
        );
        assert_eq!(table.find_peer_for_layer(5).unwrap().peer_id, "peer2");
        // The exclusion path (`HttpPipelineRouter` retries) respects it too: with both peers
        // excluded there is nothing left to route to.
        let excluded: std::collections::HashSet<String> =
            ["peer2".to_string()].into_iter().collect();
        assert!(
            table
                .find_target_for_layer_excluding(5, &excluded, NO_EXPLORATION)
                .is_empty(),
            "peer1 is suppressed and peer2 is excluded"
        );
        assert_eq!(
            table
                .find_target_for_layer_excluding(
                    5,
                    &std::collections::HashSet::new(),
                    NO_EXPLORATION,
                )
                .selected
                .map(|peer| peer.peer_id),
            Some("peer2".to_string())
        );
    }

    /// `S43`/`TASK-37`: a peer that announces a partial `expert_range` cannot serve a whole layer, so
    /// the table never routes to it — and it reports it, so the router can say why it stopped instead
    /// of waiting 30 s for a peer that can never answer.
    #[test]
    fn a_peer_with_an_expert_range_is_never_selected_for_a_layer() {
        let mut table = RoutingTable::new();
        table.insert_peer(peer("shard", "/ip4/1.1.1.1/tcp/9000"));
        table.insert_peer(LayerPeer {
            expert_range: Some((0, 4)),
            ..peer("half", "/ip4/1.1.1.2/tcp/9000")
        });

        let selection = table.find_target_for_layer(5);
        assert_eq!(
            selection.selected.as_ref().map(|p| p.peer_id.as_str()),
            Some("shard"),
            "only the whole-layer peer is routable"
        );
        assert_eq!(
            selection.unsupported_expert_shards,
            vec!["half".to_string()]
        );

        // With the whole-layer peer gone, there is no route left — and the shard is still named.
        let mut only_shards = RoutingTable::new();
        only_shards.insert_peer(LayerPeer {
            expert_range: Some((0, 4)),
            ..peer("half", "/ip4/1.1.1.2/tcp/9000")
        });
        let selection = only_shards.find_target_for_layer(5);
        assert!(selection.is_empty());
        assert_eq!(
            selection.unsupported_expert_shards,
            vec!["half".to_string()]
        );
        assert!(
            only_shards.find_peer_for_layer(5).is_none(),
            "the latency-based selector applies the same policy"
        );
    }

    #[test]
    fn a_suppression_expires_and_the_peer_becomes_eligible_again() {
        let mut table = RoutingTable::new();
        table.insert_peer(peer("peer1", "/ip4/1.1.1.1/tcp/9000"));
        let now = std::time::Instant::now();
        let window = std::time::Duration::from_secs(300);
        table.suppress("peer1", now, window);

        assert!(
            table
                .expire_suppressions(now + window - std::time::Duration::from_secs(1))
                .is_empty(),
            "inside the window"
        );
        assert!(table.is_suppressed("peer1"));

        let expired = table.expire_suppressions(now + window + std::time::Duration::from_secs(1));
        assert_eq!(expired, vec!["peer1".to_string()]);
        assert!(!table.is_suppressed("peer1"));
        assert_eq!(table.find_peer_for_layer(5).unwrap().peer_id, "peer1");
    }

    #[test]
    fn unsuppress_reports_whether_it_changed_anything() {
        let mut table = RoutingTable::new();
        table.insert_peer(peer("peer1", "/ip4/1.1.1.1/tcp/9000"));
        let now = std::time::Instant::now();

        assert!(!table.unsuppress("peer1"), "nothing to clear");
        table.suppress("peer1", now, std::time::Duration::from_secs(300));
        assert!(table.unsuppress("peer1"), "the mark is gone now");
        assert!(!table.is_suppressed("peer1"));
    }

    #[test]
    fn suppressing_an_unknown_peer_is_harmless() {
        // A `WorkerLeft` of a peer that is not in the table never reaches this call (the reducer
        // drops it), but the table must not grow an entry for a stranger if it ever did.
        let mut table = RoutingTable::new();
        let now = std::time::Instant::now();
        table.suppress("stranger", now, std::time::Duration::from_secs(300));

        assert!(table.peers().is_empty());
        assert!(table.find_target_for_layer(5).is_empty());
    }

    /// `TASK-72`/S79: the table can now tell «never measured» apart from «measured 200 ms», which the
    /// published number cannot. The published number is left as it was, so no other reader changes.
    #[test]
    fn measured_latency_tells_never_measured_apart_from_a_real_measurement() {
        let mut table = RoutingTable::new();
        table.insert_peer(peer("peer1", "/ip4/1.1.1.1/tcp/9000"));

        assert_eq!(table.measured_latency("peer1"), None, "nobody timed it yet");
        assert_eq!(
            table.get_latency("peer1"),
            UNMEASURED_LATENCY,
            "the published value is the prior, and it does not change"
        );

        table.update_latency("peer1", std::time::Duration::from_millis(33));
        assert_eq!(
            table.measured_latency("peer1"),
            Some(std::time::Duration::from_millis(33))
        );
    }

    /// The second half of the weight: the depth is read from the peer the table holds **now**, so a
    /// `/peers` refresh is what the policy ranks with, and a stranger has no queue.
    #[test]
    fn the_queue_depth_signal_is_read_from_the_live_peer() {
        let mut table = RoutingTable::new();
        table.insert_peer(LayerPeer {
            queue_depth: 3,
            ..peer("busy", "/ip4/1.1.1.2/tcp/9000")
        });

        assert_eq!(table.queue_depth_of("busy"), 3);
        assert_eq!(
            table.queue_depth_of("stranger"),
            0,
            "unknown peer, no queue"
        );

        table.insert_peer(LayerPeer {
            queue_depth: 9,
            ..peer("busy", "/ip4/1.1.1.2/tcp/9000")
        });
        assert_eq!(table.queue_depth_of("busy"), 9, "the newest view wins");
    }

    /// The share is a door, not a master key: it picks among the candidates the table hands over, and
    /// a suppressed peer was already filtered out. The contrast at the end is the point — same draw,
    /// same peers, and the only thing that changed is the `WorkerLeft` mark.
    #[test]
    fn a_suppressed_peer_is_filtered_out_before_the_exploration_share() {
        let mut table = RoutingTable::new();
        table.insert_peer(peer("measured", "/ip4/1.1.1.1/tcp/9000"));
        table.insert_peer(peer("gone", "/ip4/1.1.1.2/tcp/9000"));
        table.update_latency("measured", std::time::Duration::from_millis(50));
        // `gone` was never measured, so it is exactly the candidate the share reaches for.
        table.suppress(
            "gone",
            std::time::Instant::now(),
            std::time::Duration::from_secs(300),
        );

        let selection =
            table.find_target_for_layer_excluding(5, &std::collections::HashSet::new(), 0.0);
        assert_eq!(
            selection.selected.map(|peer| peer.peer_id),
            Some("measured".to_string()),
            "the share cannot resurrect a peer that a WorkerLeft took out"
        );

        table.unsuppress("gone");
        let selection =
            table.find_target_for_layer_excluding(5, &std::collections::HashSet::new(), 0.0);
        assert_eq!(
            selection.selected.map(|peer| peer.peer_id),
            Some("gone".to_string()),
            "an unmeasured candidate is what the share is for"
        );
    }

    fn peer(peer_id: &str, p2p_addr: &str) -> LayerPeer {
        LayerPeer {
            peer_id: peer_id.to_string(),
            http_addr: format!("http://{peer_id}"),
            p2p_addr: p2p_addr.to_string(),
            layer_ranges: vec![(0, 10)],
            expert_range: None,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        }
    }

    /// `TASK-96`/S102: a peer that declares two disjoint ranges is a candidate for **both** stretches,
    /// and the range that serves a step is the one covering it — never «the peer's range», which does
    /// not exist any more (a node can own two).
    #[test]
    fn a_dual_range_peer_is_indexed_under_every_range_it_declares() {
        let mut table = RoutingTable::new();
        table.replace_all(vec![LayerPeer {
            layer_ranges: vec![(0, 4), (8, 12)],
            ..peer("dual", "/ip4/1.1.1.1/tcp/9000")
        }]);

        for layer in [0, 3, 8, 11] {
            assert_eq!(
                table
                    .get_peers_for_layer(layer)
                    .iter()
                    .map(|p| p.peer_id.as_str())
                    .collect::<Vec<_>>(),
                vec!["dual"],
                "layer {layer} is served by the dual-range peer"
            );
        }
        for layer in [4, 6, 7, 12] {
            assert!(
                table.get_peers_for_layer(layer).is_empty(),
                "layer {layer} is the hole between the ranges: nobody is indexed for it"
            );
        }

        let owners = table.get_peers_for_layer(9);
        assert_eq!(
            tessella_protocol::layer_ranges::range_covering(&owners[0].layer_ranges, 9),
            Some((8, 12)),
            "the step is served by the range that covers it, and the router sends exactly that one"
        );
    }

    /// `TASK-87`/S94: a canary is never selected by find_target_for_layer_excluding.
    #[test]
    fn canary_peer_is_never_returned_by_routing_table() {
        let mut table = RoutingTable::new();
        let mut canary_p = peer("canary-node", "/ip4/1.1.1.1/tcp/9001");
        canary_p.canary = true;
        let honest_p = peer("honest-node", "/ip4/1.1.1.2/tcp/9002");

        table.replace_all(vec![canary_p, honest_p]);

        let target =
            table.find_target_for_layer_excluding(5, &std::collections::HashSet::new(), 0.0);
        assert_eq!(
            target.selected.map(|p| p.peer_id),
            Some("honest-node".to_string()),
            "only honest node may be selected"
        );

        // When only a canary covers the layer, selection is None
        let mut table_only_canary = RoutingTable::new();
        let mut sole_canary = peer("sole-canary", "/ip4/1.1.1.1/tcp/9001");
        sole_canary.canary = true;
        table_only_canary.replace_all(vec![sole_canary]);

        let target = table_only_canary.find_target_for_layer_excluding(
            5,
            &std::collections::HashSet::new(),
            0.0,
        );
        assert!(
            target.selected.is_none(),
            "mesh with only a canary must return None rather than routing to canary"
        );
    }
}
