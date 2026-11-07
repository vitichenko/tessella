use crate::models::LayerPeer;
use dashmap::DashMap;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The route this node should take to advance one layer, plus the peers it must **not** use.
///
/// `S43`/`TASK-37` **retired expert parallelism as a routing mode**. The old
/// `RoutingTarget::Collective` scattered the same hidden state to every peer that announced an
/// `expert_range` and summed the answers, but each of those peers executes the **whole layer**
/// (attention, residual and shared expert included), so the gather produced
/// `N·(h + attn + shared) + Σ moe_i` instead of `h + attn + shared + Σ moe_i`: wrong logits,
/// silently. Nothing in the payload asked a hop for *only* its experts' contribution, and no
/// assignment guaranteed that the union of the announced ranges covers every expert the gate may
/// pick (an expert nobody holds contributes zeros without a word).
///
/// A peer that announces a partial `expert_range` is therefore **never** selected; it is reported in
/// [`RouteSelection::unsupported_expert_shards`] so the caller can say what happened instead of
/// waiting for a peer that cannot serve the layer correctly.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RouteSelection {
    /// Lowest-latency peer that can serve the whole layer, if any.
    pub selected: Option<LayerPeer>,
    /// `peer_id`s of the peers that announced an expert range, in candidate order.
    pub unsupported_expert_shards: Vec<String>,
}

impl RouteSelection {
    /// `true` when there is nobody to route to (whatever the reason).
    pub fn is_empty(&self) -> bool {
        self.selected.is_none()
    }
}

/// Latency a candidate this node has **never measured** enters the ranking with (`TASK-72` · S79).
///
/// It is the very number this node published before `S79` (200 ms, `RoutingTable::get_latency`), so
/// the policy keeps a meaning for a peer nobody has timed. What changes is *how* it is used: with a
/// strict `<` and equal values the first candidate of the topology kept the hop for ever, so a peer
/// that never received traffic could never be measured — and therefore could neither earn a place
/// nor lose it. Now it is a prior of the weight, and [`EXPLORATION_SHARE`] is the door that gives an
/// unmeasured peer the traffic that measures it.
pub const UNMEASURED_LATENCY: Duration = Duration::from_millis(200);

/// Exploration share ε of the routing policy (`TASK-72` · S79): 5 % of the hop decisions.
///
/// Reserved for a candidate this node has **never measured**. Without it the winner is measured
/// every time it serves and the loser never is, so `<` alone is a lock: the fastest *measured* peer
/// keeps the traffic even when somebody else would be faster, and a peer that recovers from a
/// penalty cannot come back without somebody else failing. It is deliberately small because sending
/// a hop to a bad peer costs a real latency (`TASK-72`, risks); and it never reaches a suppressed or
/// excluded peer, because the table filters those out *before* the policy runs.
pub const EXPLORATION_SHARE: f64 = 0.05;

/// `draw` value that never explores: what a caller that must be deterministic passes.
pub const NO_EXPLORATION: f64 = 1.0;

/// Picks the next hop for a layer: the candidate with the best weight, plus an exploration share.
///
/// ```text
/// weight = 1 / (latency_ms × (1 + queue_depth))
/// ```
///
/// The policy in full:
///
/// 1. **Eligibility**: a peer that announces an `expert_range` is not routable (see
///    [`RouteSelection`]). A `LayerPeer` does not carry the model's expert count, so this is the
///    conservative form of the same rule that [`crate::assignment::serves_whole_layers`] applies
///    where the count is known — and every announcement the orchestrator of `S43` produces has
///    `expert_range: None`, so this filter never drops a peer that could have served the layer.
/// 2. **Ranking**: the best `weight`, where a candidate with no measurement enters with
///    [`UNMEASURED_LATENCY`] and a penalised one with its measured penalty (10 s). A peer that just
///    failed therefore loses to a candidate nobody has tried, which is how the mesh recovers.
/// 3. **Exploration**: when `draw` falls inside `[0, EXPLORATION_SHARE)` **and** at least one
///    candidate has never been measured, the hop goes to the first such candidate in candidate
///    order, whatever the weights say. With no unmeasured candidate the share has nobody to explore
///    (ε is not a random loser): the ranking stands.
/// 4. **Ties** keep the candidate order of the topology, exactly as the strict `<` of the retired
///    `select_route` did. Reordering them by `peer_id` looks harmless and is not: the `S75`/`F-1`
///    regression (`tessella-worker/tests/http_pipeline_two_hops.rs`,
///    `a_hop_that_refuses_a_decode_step_aborts_the_request_instead_of_burning_the_next_candidate`)
///    relies on it — for layer 1 the first candidate is the one that refuses the step, and picking
///    the other one silently returns `F-1`'s plausible `200` built on an empty KV.
///
/// The signals are read through closures, so the caller owns where they come from: the router
/// passes the `queue_depth` of the live peer — the field the mesh publishes — and the EWMA latency
/// it has measured, or `None` when it has never measured that peer.
///
/// `draw` is an **input** (`[0, 1)`), never generated here: nothing inside the function is random,
/// which is what makes every case below deterministic. [`NO_EXPLORATION`] disables the share.
pub fn select_route_with(
    candidates: &[LayerPeer],
    mut get_latency: impl FnMut(&str) -> Option<Duration>,
    mut get_queue_depth: impl FnMut(&str) -> usize,
    draw: f64,
) -> RouteSelection {
    let mut selection = RouteSelection::default();

    // 1. Eligibility — `S43`/`TASK-37`: only a peer that serves the **whole** layer can receive the
    //    hop, and the shards are named for the caller's log instead of silently discarded.
    let mut routable = Vec::with_capacity(candidates.len());
    for peer in candidates {
        if peer.expert_range.is_some() {
            selection
                .unsupported_expert_shards
                .push(peer.peer_id.clone());
        } else if !peer.canary {
            routable.push(peer);
        }
    }

    // 2. Exploration, decided **before** the ranking because it is the one case where the ranking
    //    must be ignored: a peer this node has never measured is invisible to it (it only ever
    //    carries the prior), so without this door the winner is measured for ever and the loser
    //    never is. `draw` decides *whether* to explore, never *whom*: the hop goes to the first
    //    unmeasured candidate of the topology, so the behaviour is reproducible from the draw plus
    //    the candidate order. With every candidate measured there is nobody to explore and the share
    //    is a no-op — ε is not a lottery, it is a rehearsal.
    if draw < EXPLORATION_SHARE
        && let Some(peer) = routable
            .iter()
            .find(|peer| get_latency(&peer.peer_id).is_none())
    {
        selection.selected = Some((*peer).clone());
        return selection;
    }

    // 3. Ranking by `weight = 1 / (latency_ms × (1 + queue_depth))`. Bigger is better, and the
    //    candidate order decides ties (`≤` keeps the incumbent), which is the guarantee `S75`/`F-1`
    //    depends on. A candidate this node never measured enters with the prior; a penalised one
    //    enters with its penalty (10 s), so it loses to the prior and the mesh can recover.
    //
    //    The floor keeps a measurement of 0 — a hop that answered inside the timer's resolution —
    //    from dividing by zero; for any real latency the ordering is untouched.
    const MIN_LATENCY_MS: f64 = 0.001;
    let mut best: Option<(&LayerPeer, f64)> = None;
    for peer in routable {
        let latency = get_latency(&peer.peer_id).unwrap_or(UNMEASURED_LATENCY);
        let latency_ms = (latency.as_secs_f64() * 1000.0).max(MIN_LATENCY_MS);
        let queue_depth = get_queue_depth(&peer.peer_id) as f64;
        let weight = 1.0 / (latency_ms * (1.0 + queue_depth));

        match best {
            Some((_, best_weight)) if weight <= best_weight => {}
            _ => best = Some((peer, weight)),
        }
    }

    selection.selected = best.map(|(peer, _)| peer.clone());
    selection
}

/// Default idle time before a pinned session route expires (`TASK-81` · S88).
///
/// Matches the worker's `SESSION_TTL_SECS` (300 s), so a session that is idle long enough for
/// downstream workers to evict its KV cache is unpinned here too.
pub const DEFAULT_PIN_TTL: Duration = Duration::from_secs(300);

/// Maximum number of sessions tracked by [`SessionPinRegistry`] concurrently (`TASK-81` · S88).
///
/// Bounded on purpose (like [`crate::telemetry::metrics::MAX_TRACKED_PEERS`]): an active worker
/// with many clients cannot leak memory. When full, expired sessions are swept first, followed
/// by eviction of the oldest session.
pub const MAX_PINNED_SESSIONS: usize = 1024;

/// A pinned downstream route for a specific layer range of an active session (`TASK-81` · S88).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedRoute {
    pub peer_id: String,
    pub layer_range: (usize, usize),
    pub last_used: Instant,
}

/// Remembers which peer served each layer stretch of an active session (`TASK-81` · S88).
///
/// In a mesh with replicas (`D-20`/S57), two or more peers can serve the same layer range. Because
/// KV cache is held locally by the peer that processed the prompt's prefill, subsequent decode steps
/// of that session MUST continue to reach that exact peer. Without this pin, routing table refreshes,
/// latency jitter or the exploration share (ε) can select the twin replica mid-generation — and the
/// twin, lacking the KV cache, refuses with `409 missing_prefill`, terminating the entire request.
#[derive(Debug)]
pub struct SessionPinRegistry {
    /// session_key -> (layer_start -> PinnedRoute)
    sessions: DashMap<String, HashMap<usize, PinnedRoute>>,
    ttl: Duration,
    max_sessions: usize,
}

impl Default for SessionPinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionPinRegistry {
    /// Creates a new registry with default TTL (300 s) and capacity (1024 sessions).
    #[must_use]
    pub fn new() -> Self {
        Self::with_ttl_and_capacity(DEFAULT_PIN_TTL, MAX_PINNED_SESSIONS)
    }

    /// Creates a new registry with custom TTL and maximum session capacity.
    #[must_use]
    pub fn with_ttl_and_capacity(ttl: Duration, max_sessions: usize) -> Self {
        Self {
            sessions: DashMap::new(),
            ttl,
            max_sessions,
        }
    }

    /// Returns the pinned `peer_id` for `layer` within `session_key`, if an active unexpired pin exists.
    ///
    /// Updates `last_used` on hit. If the pin is expired, it is removed and `None` is returned.
    pub fn get_pinned_peer(&self, session_key: &str, layer: usize) -> Option<String> {
        self.get_pinned_peer_at(session_key, layer, Instant::now())
    }

    /// [`Self::get_pinned_peer`] with the instant supplied by the caller (`TASK-101` · S107).
    ///
    /// The clock is an **input** because the TTL is only observable through it: a caller that has to
    /// tell «still pinned» from «expired» — or that wants two reads to share one instant — passes
    /// the instant instead of sleeping between the reads, so the answer cannot depend on when the
    /// thread was scheduled. Production reaches this through [`Self::get_pinned_peer`], which is
    /// this function with `Instant::now()`, so the contract does not move: the pin is dropped **on
    /// read** as soon as `now - last_used > ttl`, and a hit refreshes `last_used` to `now`.
    pub fn get_pinned_peer_at(
        &self,
        session_key: &str,
        layer: usize,
        now: Instant,
    ) -> Option<String> {
        let mut entry = self.sessions.get_mut(session_key)?;
        let ttl = self.ttl;

        // Clean up expired routes for this session:
        entry.retain(|_, route| now.duration_since(route.last_used) <= ttl);
        if entry.is_empty() {
            drop(entry);
            self.sessions.remove(session_key);
            return None;
        }

        let route = entry
            .values_mut()
            .find(|r| r.layer_range.0 <= layer && layer < r.layer_range.1)?;
        route.last_used = now;
        Some(route.peer_id.clone())
    }

    /// Pins or updates `peer_id` as the handler for `layer_range` of `session_key`.
    pub fn pin_peer(&self, session_key: &str, layer_range: (usize, usize), peer_id: &str) {
        self.pin_peer_at(session_key, layer_range, peer_id, Instant::now());
    }

    /// [`Self::pin_peer`] with the instant supplied by the caller (`TASK-101` · S107).
    ///
    /// `now` is the pin's `last_used`, so it is what decides when the pin expires: read it back at
    /// `now + ttl` and the pin is still there, one millisecond later it is not. It is also the
    /// instant the capacity eviction below ages the other pins with, so a caller that wants a
    /// deterministic eviction order passes increasing instants instead of sleeping between pins.
    pub fn pin_peer_at(
        &self,
        session_key: &str,
        layer_range: (usize, usize),
        peer_id: &str,
        now: Instant,
    ) {
        if !self.sessions.contains_key(session_key) && self.sessions.len() >= self.max_sessions {
            self.evict_expired_or_oldest(now);
        }
        let mut entry = self.sessions.entry(session_key.to_string()).or_default();
        entry.insert(
            layer_range.0,
            PinnedRoute {
                peer_id: peer_id.to_string(),
                layer_range,
                last_used: now,
            },
        );
    }

    /// Unpins any route covering `layer` for `session_key`.
    pub fn unpin_peer(&self, session_key: &str, layer: usize) {
        if let Some(mut entry) = self.sessions.get_mut(session_key) {
            entry.retain(|_, r| !(r.layer_range.0 <= layer && layer < r.layer_range.1));
            if entry.is_empty() {
                drop(entry);
                self.sessions.remove(session_key);
            }
        }
    }

    /// Removes all pinned routes for `session_key` (e.g. when session ends or is discarded on `missing_prefill`).
    pub fn remove_session(&self, session_key: &str) -> bool {
        self.sessions.remove(session_key).is_some()
    }

    /// Number of active tracked sessions.
    #[must_use]
    pub fn tracked_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// `true` when no sessions are pinned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Sweeps expired sessions and, if still at or over capacity, evicts the oldest session.
    fn evict_expired_or_oldest(&self, now: Instant) {
        let ttl = self.ttl;
        self.sessions.retain(|_, routes| {
            routes.retain(|_, r| now.duration_since(r.last_used) <= ttl);
            !routes.is_empty()
        });

        if self.sessions.len() >= self.max_sessions {
            let oldest_key = self
                .sessions
                .iter()
                .map(|entry| {
                    let most_recent = entry
                        .value()
                        .values()
                        .map(|r| r.last_used)
                        .max()
                        .unwrap_or(now);
                    (entry.key().clone(), most_recent)
                })
                .min_by_key(|(_, time)| *time)
                .map(|(key, _)| key);

            if let Some(key) = oldest_key {
                self.sessions.remove(&key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn peer(peer_id: &str, expert_range: Option<(usize, usize)>) -> LayerPeer {
        LayerPeer {
            peer_id: peer_id.to_string(),
            http_addr: format!("http://{peer_id}"),
            p2p_addr: "/ip4/1.1.1.1/tcp/9000".to_string(),
            layer_ranges: vec![(0, 10)],
            expert_range,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        }
    }

    /// The same peer, as a node whose queue the mesh published (`TASK-72`/S79).
    fn queued(peer_id: &str, queue_depth: usize) -> LayerPeer {
        LayerPeer {
            queue_depth,
            ..peer(peer_id, None)
        }
    }

    fn latencies(pairs: &[(&str, u64)]) -> HashMap<String, Duration> {
        pairs
            .iter()
            .map(|(id, ms)| ((*id).to_string(), Duration::from_millis(*ms)))
            .collect()
    }

    /// Runs the policy with the queue depth read from the candidate itself, which is where the
    /// router gets it from: `LayerPeer::queue_depth` is the field the mesh publishes. A peer that is
    /// **not** in `lat` has never been measured (`None`) — that is the whole point of the share.
    fn select_with(
        peers: &[LayerPeer],
        lat: &HashMap<String, Duration>,
        draw: f64,
    ) -> RouteSelection {
        select_route_with(
            peers,
            |id| lat.get(id).copied(),
            |id| {
                peers
                    .iter()
                    .find(|candidate| candidate.peer_id == id)
                    .map(|candidate| candidate.queue_depth)
                    .unwrap_or(0)
            },
            draw,
        )
    }

    #[test]
    fn an_empty_candidate_pool_selects_nothing() {
        let selection = select_with(&[], &latencies(&[]), 0.0);

        assert_eq!(selection, RouteSelection::default());
        assert!(selection.is_empty());
        assert!(
            selection.unsupported_expert_shards.is_empty(),
            "an empty pool cannot report a shard"
        );
    }

    #[test]
    fn the_lowest_latency_whole_layer_peer_wins() {
        let lat = latencies(&[("peer1", 100), ("peer2", 50)]);
        let selection = select_with(
            &[peer("peer1", None), peer("peer2", None)],
            &lat,
            NO_EXPLORATION,
        );

        assert_eq!(selection.selected, Some(peer("peer2", None)));
        assert!(selection.unsupported_expert_shards.is_empty());
    }

    /// `TASK-72`/S79, premise (i): the queue is part of the weight, and at equal latency it decides.
    #[test]
    fn a_deeper_queue_loses_at_equal_latency() {
        let peers = [queued("idle", 0), queued("busy", 4)];
        let lat = latencies(&[("idle", 50), ("busy", 50)]);

        let selection = select_with(&peers, &lat, NO_EXPLORATION);

        assert_eq!(
            selection.selected,
            Some(queued("idle", 0)),
            "weight = 1/(latency × (1 + queue)): the queue makes `busy` five times as expensive"
        );
    }

    /// The formula is not decorative: a peer twice as slow with a free queue beats a fast one that is
    /// already saturated. `1/(50×5) = 0.004 < 1/(100×1) = 0.01`.
    #[test]
    fn a_free_queue_beats_a_faster_saturated_peer() {
        let peers = [queued("saturated", 4), queued("free", 0)];
        let lat = latencies(&[("saturated", 50), ("free", 100)]);

        assert_eq!(
            select_with(&peers, &lat, NO_EXPLORATION)
                .selected
                .map(|peer| peer.peer_id),
            Some("free".to_string())
        );
    }

    /// `TASK-72`/S79, premise (ii): the ε share is the **only** door to a candidate this node has
    /// never measured.
    #[test]
    fn an_unmeasured_candidate_enters_only_through_the_exploration_share() {
        let peers = [peer("measured", None), peer("untried", None)];
        let lat = latencies(&[("measured", 5)]);

        assert_eq!(
            select_with(&peers, &lat, NO_EXPLORATION)
                .selected
                .map(|peer| peer.peer_id),
            Some("measured".to_string()),
            "outside the share the measured 5 ms wins over the 200 ms prior"
        );
        assert_eq!(
            select_with(&peers, &lat, EXPLORATION_SHARE / 2.0)
                .selected
                .map(|peer| peer.peer_id),
            Some("untried".to_string()),
            "inside the share the hop goes to the peer nobody has timed, whatever the weights say"
        );
        assert_eq!(
            select_with(&peers, &lat, EXPLORATION_SHARE)
                .selected
                .map(|peer| peer.peer_id),
            Some("measured".to_string()),
            "the share is half-open: draw = ε does not explore"
        );
    }

    /// The other half of the lock: a peer that just failed (penalised at 10 s) does not keep the hop
    /// against a candidate nobody has tried. That is how a node that came back becomes usable again.
    #[test]
    fn a_penalised_peer_loses_to_one_that_was_never_measured() {
        let peers = [peer("penalised", None), peer("untried", None)];
        let lat = latencies(&[("penalised", 10_000)]);

        assert_eq!(
            select_with(&peers, &lat, NO_EXPLORATION)
                .selected
                .map(|peer| peer.peer_id),
            Some("untried".to_string()),
            "a 10 s penalty is a measurement, and it loses to the 200 ms prior"
        );
    }

    /// ε is not a blind draw: with every candidate measured there is nobody to explore, and the
    /// slower one must not receive the hop 5 % of the time.
    #[test]
    fn the_share_never_reaches_a_measured_candidate() {
        let peers = [peer("fast", None), peer("slow", None)];
        let lat = latencies(&[("fast", 5), ("slow", 400)]);

        assert_eq!(
            select_with(&peers, &lat, 0.0)
                .selected
                .map(|peer| peer.peer_id),
            Some("fast".to_string()),
            "both are measured: the weights decide"
        );
    }

    /// With one candidate there is nothing to explore, and in a pool where nobody has been measured
    /// the share picks the same peer the ranking does — so no pool is ever left without a hop.
    #[test]
    fn the_share_never_leaves_a_pool_without_a_hop() {
        let single = [peer("only", None)];
        let unmeasured = [peer("first", None), peer("second", None)];
        let none_measured = latencies(&[]);

        for draw in [
            0.0,
            EXPLORATION_SHARE / 2.0,
            EXPLORATION_SHARE,
            NO_EXPLORATION,
        ] {
            assert_eq!(
                select_with(&single, &none_measured, draw)
                    .selected
                    .map(|peer| peer.peer_id),
                Some("only".to_string()),
                "draw = {draw}"
            );
            assert_eq!(
                select_with(&unmeasured, &none_measured, draw)
                    .selected
                    .map(|peer| peer.peer_id),
                Some("first".to_string()),
                "nobody measured: the share cannot disagree with the ranking (draw = {draw})"
            );
        }
    }

    /// `TASK-68`/`S75`: a tie keeps the **first** candidate of the topology, exactly as the retired
    /// strict `<` did. It is not cosmetic: the regression of `F-1` sends the decode step to the
    /// candidate that refuses it *because* it is the first one of layer 1, and picking the other one
    /// would return `F-1`'s plausible `200` built on an empty KV.
    #[test]
    fn a_tie_keeps_the_candidate_order_and_not_the_peer_id() {
        let none_measured = latencies(&[]);
        let straight = select_with(
            &[peer("peer-a", None), peer("peer-b", None)],
            &none_measured,
            NO_EXPLORATION,
        );
        let flipped = select_with(
            &[peer("peer-b", None), peer("peer-a", None)],
            &none_measured,
            NO_EXPLORATION,
        );

        assert_eq!(
            straight.selected.map(|peer| peer.peer_id),
            Some("peer-a".to_string())
        );
        assert_eq!(
            flipped.selected.map(|peer| peer.peer_id),
            Some("peer-b".to_string()),
            "the first candidate of the topology wins the tie, whatever its peer_id is"
        );
    }

    /// The behaviour `S43`/`TASK-37` retired: two peers that each hold **part** of the experts used
    /// to be returned as a `Collective` and their answers summed, which counted attention, the
    /// residual and the shared expert once per peer. They are not routable any more, and the caller
    /// is told which ones they were so it can say *why* instead of waiting for a peer that can never
    /// serve the layer correctly.
    #[test]
    fn a_partial_expert_shard_is_never_selected_and_is_reported() {
        let selection = select_with(
            &[peer("shard-a", Some((0, 4))), peer("shard-b", Some((4, 8)))],
            &latencies(&[]),
            0.0,
        );

        assert!(
            selection.selected.is_none(),
            "a partial expert range cannot serve a whole layer"
        );
        assert_eq!(
            selection.unsupported_expert_shards,
            vec!["shard-a".to_string(), "shard-b".to_string()]
        );
    }

    #[test]
    fn a_whole_layer_peer_wins_over_shards_and_the_shards_are_still_reported() {
        let lat = latencies(&[("shard", 5), ("full", 500)]);
        let selection = select_with(
            &[peer("shard", Some((0, 4))), peer("full", None)],
            &lat,
            NO_EXPLORATION,
        );

        assert_eq!(
            selection.selected,
            Some(peer("full", None)),
            "latency does not make a shard routable"
        );
        assert_eq!(
            selection.unsupported_expert_shards,
            vec!["shard".to_string()]
        );
    }

    #[test]
    fn a_pinned_peer_is_returned_for_any_layer_in_its_range() {
        let registry = SessionPinRegistry::new();
        registry.pin_peer("sess-1", (0, 8), "peer-entry");
        registry.pin_peer("sess-1", (8, 16), "peer-mid");
        registry.pin_peer("sess-1", (16, 24), "peer-tail");

        assert_eq!(
            registry.get_pinned_peer("sess-1", 0),
            Some("peer-entry".to_string())
        );
        assert_eq!(
            registry.get_pinned_peer("sess-1", 7),
            Some("peer-entry".to_string())
        );
        assert_eq!(
            registry.get_pinned_peer("sess-1", 8),
            Some("peer-mid".to_string())
        );
        assert_eq!(
            registry.get_pinned_peer("sess-1", 15),
            Some("peer-mid".to_string())
        );
        assert_eq!(
            registry.get_pinned_peer("sess-1", 16),
            Some("peer-tail".to_string())
        );
        assert_eq!(
            registry.get_pinned_peer("sess-1", 23),
            Some("peer-tail".to_string())
        );
        assert_eq!(registry.get_pinned_peer("sess-1", 24), None);
        assert_eq!(registry.get_pinned_peer("sess-unknown", 8), None);
    }

    #[test]
    fn re_pinning_updates_the_peer() {
        let registry = SessionPinRegistry::new();
        registry.pin_peer("sess-1", (8, 16), "peer-a");
        assert_eq!(
            registry.get_pinned_peer("sess-1", 8),
            Some("peer-a".to_string())
        );

        registry.pin_peer("sess-1", (8, 16), "peer-b");
        assert_eq!(
            registry.get_pinned_peer("sess-1", 8),
            Some("peer-b".to_string())
        );
    }

    /// `TASK-101`/S107: the clock is an input, so the TTL is measured in **explicit instants** and
    /// this test contains no `sleep`. The version it replaces built a 10 ms TTL and asserted
    /// «present» before sleeping 20 ms: on the loaded machine of a `cargo test --workspace` run the
    /// thread could return *after* the TTL, and the first assertion failed with `left: None` — a red
    /// that said nothing about the TTL contract, and that was re-run until it passed.
    ///
    /// With the clock as an input the edge the `sleep` could only approach is asserted exactly, and
    /// that matters: the contract is inclusive (`duration_since <= ttl`), so a rewrite to `<` would
    /// be a silent one-millisecond change to every pin. `sess-2` is what holds that edge.
    #[test]
    fn an_expired_pin_is_dropped_and_returns_none() {
        const TTL: Duration = Duration::from_millis(10);
        let registry = SessionPinRegistry::with_ttl_and_capacity(TTL, 100);
        // The only clock read of the test: everything below is derived from it by arithmetic, so
        // nothing depends on how long the machine takes between two statements.
        let pinned_at = Instant::now();
        let boundary = pinned_at + TTL;
        registry.pin_peer_at("sess-1", (8, 16), "peer-a", pinned_at);
        registry.pin_peer_at("sess-2", (8, 16), "peer-b", pinned_at);

        // One millisecond before the TTL the pin is there, and the hit moves its deadline.
        let last_hit = boundary - Duration::from_millis(1);
        assert_eq!(
            registry.get_pinned_peer_at("sess-1", 8, last_hit),
            Some("peer-a".to_string())
        );

        // Exactly at the TTL goes the other way: still alive.
        assert_eq!(
            registry.get_pinned_peer_at("sess-2", 8, boundary),
            Some("peer-b".to_string())
        );

        // One millisecond past its *refreshed* deadline, `sess-1` is dropped on read and the
        // now-empty session is swept.
        assert_eq!(
            registry.get_pinned_peer_at("sess-1", 8, last_hit + TTL + Duration::from_millis(1)),
            None
        );
        assert_eq!(
            registry.tracked_sessions(),
            1,
            "only the boundary session survives"
        );

        // `sess-2` follows one TTL after *its* refreshing read, not after the pin.
        assert_eq!(
            registry.get_pinned_peer_at("sess-2", 8, boundary + TTL + Duration::from_millis(1)),
            None
        );
        assert_eq!(registry.tracked_sessions(), 0);
    }

    /// The seam is one contract, not a parameter that is passed and ignored: the instant handed to
    /// `pin_peer_at` **is** the pin's `last_used`, so a pin created with an instant a minute in the
    /// past is already expired although the call itself happened now. The read is explicit — one
    /// millisecond past that pin's deadline, and still a minute in the past for the real clock —
    /// which is what tells «the age is the instant you passed» from «the age is when you called».
    #[test]
    fn the_ttl_is_measured_from_the_instant_that_was_passed() {
        const TTL: Duration = Duration::from_millis(10);
        let registry = SessionPinRegistry::with_ttl_and_capacity(TTL, 100);
        let long_ago = Instant::now() - Duration::from_secs(60);
        registry.pin_peer_at("sess-1", (8, 16), "peer-a", long_ago);

        assert_eq!(
            registry.get_pinned_peer_at("sess-1", 8, long_ago + TTL + Duration::from_millis(1)),
            None
        );
        assert_eq!(registry.tracked_sessions(), 0);
    }

    #[test]
    fn removing_a_session_drops_all_its_layer_pins() {
        let registry = SessionPinRegistry::new();
        registry.pin_peer("sess-1", (0, 8), "peer-a");
        registry.pin_peer("sess-1", (8, 16), "peer-b");
        assert_eq!(registry.tracked_sessions(), 1);

        assert!(registry.remove_session("sess-1"));
        assert_eq!(registry.get_pinned_peer("sess-1", 0), None);
        assert_eq!(registry.get_pinned_peer("sess-1", 8), None);
        assert_eq!(registry.tracked_sessions(), 0);
    }

    #[test]
    fn unpinning_a_layer_drops_only_that_stretch() {
        let registry = SessionPinRegistry::new();
        registry.pin_peer("sess-1", (0, 8), "peer-a");
        registry.pin_peer("sess-1", (8, 16), "peer-b");

        registry.unpin_peer("sess-1", 10);
        assert_eq!(
            registry.get_pinned_peer("sess-1", 0),
            Some("peer-a".to_string())
        );
        assert_eq!(registry.get_pinned_peer("sess-1", 8), None);
        assert_eq!(registry.tracked_sessions(), 1);

        registry.unpin_peer("sess-1", 0);
        assert_eq!(registry.tracked_sessions(), 0);
    }

    /// `TASK-103`/S110: the eviction order is stated as an input, not waited for.
    ///
    /// The version this replaces separated the two pins with two 10 ms `std::thread::sleep` calls.
    /// They were not a TTL wait — the TTL here is 60 s, so a stall could only *widen* the gap — they
    /// were the ordering that `min_by_key` compares, and the seam makes that worse than useless:
    /// `pin_peer` reads `Instant::now()` itself, so two consecutive calls can stamp the **same**
    /// instant, and on a tie the eviction falls back to `HashMap` iteration order (`RandomState`,
    /// randomized per map). Measured with the sleeps deleted and the two pins stamped by hand at one
    /// instant: the tie evicted the **newest** session 987 times out of 2000, so a tie would have
    /// failed this test about half the times it happened — intermittently, for a reason that has
    /// nothing to do with eviction.
    ///
    /// With the clock as an input (`TASK-101`'s seam) the tie is not made rare, it is removed:
    /// `sess-1` and `sess-2` are 1 ms apart by construction and every read passes the instant it
    /// means.
    #[test]
    fn capacity_eviction_drops_oldest_when_full() {
        let registry = SessionPinRegistry::with_ttl_and_capacity(Duration::from_secs(60), 2);
        // The only clock read of the test: both pins are stamped through it, 1 ms apart.
        let t0 = Instant::now();
        registry.pin_peer_at("sess-1", (0, 8), "peer-1", t0);
        registry.pin_peer_at("sess-2", (0, 8), "peer-2", t0 + Duration::from_millis(1));

        assert_eq!(registry.tracked_sessions(), 2);

        // sess-3 triggers eviction of sess-1 (oldest)
        let newcomer = t0 + Duration::from_millis(2);
        registry.pin_peer_at("sess-3", (0, 8), "peer-3", newcomer);
        assert_eq!(registry.tracked_sessions(), 2);
        assert_eq!(registry.get_pinned_peer_at("sess-1", 0, newcomer), None);
        assert_eq!(
            registry.get_pinned_peer_at("sess-2", 0, newcomer),
            Some("peer-2".to_string())
        );
        assert_eq!(
            registry.get_pinned_peer_at("sess-3", 0, newcomer),
            Some("peer-3".to_string())
        );
    }

    /// `TASK-103`/S110: the TTL sweep inside `evict_expired_or_oldest` stops being a branch no test
    /// enters — the test above cannot reach it, because with a 60 s TTL nothing is expired by the
    /// time the ordering runs.
    ///
    /// Here the newcomer arrives *past* the TTL of both pins (two milliseconds past the first one's,
    /// so one past the second's — the boundary is inclusive, `duration_since <= ttl`), and it finds
    /// the map full of sessions that are already dead. An expired session must not keep a capacity
    /// slot, and it must not push a **live** session out to make room for the newcomer.
    ///
    /// `tracked_sessions() == 1` is what states it: the ordering branch removes exactly one session
    /// (`min_by_key`), and the one it would have kept is `sess-2` (the newest of the two), so a `1`
    /// can only be read as «both expired sessions were swept **before** the ordering decision».
    #[test]
    fn capacity_eviction_sweeps_expired_sessions_before_it_orders() {
        const TTL: Duration = Duration::from_secs(60);
        let registry = SessionPinRegistry::with_ttl_and_capacity(TTL, 2);
        let t0 = Instant::now();
        registry.pin_peer_at("sess-1", (0, 8), "peer-1", t0);
        registry.pin_peer_at("sess-2", (0, 8), "peer-2", t0 + Duration::from_millis(1));

        assert_eq!(
            registry.tracked_sessions(),
            2,
            "the registry is at capacity"
        );

        let after_ttl = t0 + TTL + Duration::from_millis(2);
        registry.pin_peer_at("sess-3", (0, 8), "peer-3", after_ttl);

        assert_eq!(
            registry.tracked_sessions(),
            1,
            "both expired sessions are swept, so no session is evicted by the ordering branch"
        );
        assert_eq!(registry.get_pinned_peer_at("sess-1", 0, after_ttl), None);
        assert_eq!(registry.get_pinned_peer_at("sess-2", 0, after_ttl), None);
        assert_eq!(
            registry.get_pinned_peer_at("sess-3", 0, after_ttl),
            Some("peer-3".to_string())
        );
    }

    /// `TASK-87`/S94: a canary node is never picked as the primary of a route,
    /// even if its measured latency is microseconds (fastest) and queue is 0.
    #[test]
    fn a_canary_is_never_selected_as_route_primary() {
        let mut canary_peer = peer("canary-node", None);
        canary_peer.canary = true;
        let honest_peer = peer("honest-node", None);

        let candidates = vec![canary_peer, honest_peer.clone()];
        let lat = latencies(&[("canary-node", 1), ("honest-node", 100)]);

        let selection = select_route_with(
            &candidates,
            |id| lat.get(id).copied(),
            |_| 0,
            NO_EXPLORATION,
        );

        assert_eq!(
            selection.selected.map(|p| p.peer_id),
            Some("honest-node".to_string()),
            "honest node must win despite canary having 1ms vs 100ms latency"
        );
    }
}
