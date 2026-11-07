use dashmap::DashMap;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

pub use super::worker_metrics::{BATCH_UPLOAD_LATENCY_BUCKETS, WorkerMetrics};

/// Why a hop to a peer did not complete (`TASK-50`/S56).
///
/// The reason reaches the log line, `GET /status` (`peer_failures`) and the operator reading it:
/// `timeout` is a slow node, `http_error` a node that answers but cannot serve, `network_error` a
/// transport failure, and the two `connection_*` ones come from the P2P plane (a peer that left, a
/// dial that never connected). `divergent_audit` comes from the compute audit (`TASK-53`/S59): the
/// peer answered the hop, but with a tensor a second execution did not reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerFailureReason {
    NetworkError,
    Timeout,
    HttpError,
    ConnectionClosed,
    DialFailed,
    DivergentAudit,
}

/// Number of [`PeerFailureReason`] variants. It is also the length of the per-peer counter array,
/// which is why it is a named constant: `PeerFailureReason::ALL.len()` is not usable in a
/// const-generic position.
pub const REASON_COUNT: usize = 6;

impl PeerFailureReason {
    pub const ALL: [Self; REASON_COUNT] = [
        Self::NetworkError,
        Self::Timeout,
        Self::HttpError,
        Self::ConnectionClosed,
        Self::DialFailed,
        Self::DivergentAudit,
    ];

    /// The name published by `GET /status`. Stable: it is part of the operator-facing surface.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NetworkError => "network_error",
            Self::Timeout => "timeout",
            Self::HttpError => "http_error",
            Self::ConnectionClosed => "connection_closed",
            Self::DialFailed => "dial_failed",
            Self::DivergentAudit => "divergent_audit",
        }
    }

    /// Index in the per-peer counters. The enum has no explicit discriminants, so this is the
    /// declaration order and cannot disagree with [`Self::ALL`].
    fn index(self) -> usize {
        self as usize
    }
}

/// One counter per peer (`TASK-50`/S56). Serialised by `GET /status`.
///
/// `total_failures` is cumulative for the life of the process (what a dashboard plots) and
/// `consecutive_failures` goes back to zero as soon as a hop to that peer completes (what says
/// "this hop is broken **now**").
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PeerFailureCounts {
    pub peer_id: String,
    pub total_failures: u64,
    pub consecutive_failures: u64,
    pub last_failure_unix_secs: u64,
    /// `PeerFailureReason::as_str` of the most recent failure of this peer.
    pub last_reason: &'static str,
    /// Only the reasons that actually happened (zeros are omitted, like the string keys of a
    /// label-based metric).
    pub by_reason: BTreeMap<&'static str, u64>,
}

/// What one node has seen fail towards its peers, as `GET /status` publishes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PeerFailureReport {
    pub total_failures: u64,
    pub tracked_peers: usize,
    /// Failures of peers this node could not track because [`MAX_TRACKED_PEERS`] was already full.
    /// Published so the bound is never a silent drop.
    pub untracked_failures: u64,
    /// Sorted by `total_failures` (descending), tie-broken by `peer_id`.
    pub peers: Vec<PeerFailureCounts>,
}

/// How many times each MoE expert was routed on **one** node (`TASK-43`/S49).
///
/// It used to be a process-wide `static` (`EXPERT_ACTIVATIONS`), so two engines in the same process
/// —the fixtures, two workers, a future multi-cluster run— shared it: `drain()` of one emptied the
/// counters of the other and the telemetry of both was mixed. Each engine owns one and the **same**
/// `Arc` reaches the only two places that matter:
///
/// - the hot path ([`ExpertActivationRegistry::record`], called from
///   `tessella_inference::moe::MoeBlock::forward` once per routed expert per token), and
/// - the heartbeat, which drains it into the `expert_usage` field of `/announce`, the input of the
///   provider's Karma (`SEC-07`, and the live consumer that `TASK-60`/S66 pinned by test).
///
/// If those two ever got **different** registries the telemetry would be lost in silence, which is
/// exactly the defect of `TASK-41`: the same `Arc` is injected in `InferenceEngine::new` and read by
/// the heartbeat from the engine, so there is only one.
///
/// The data structure is the one the `static` had (`DashMap` + `AtomicU64`, no global lock), so the
/// contention profile of the hot path does not change. **Prohibido** afirmar rendimiento sin medir:
/// aquí solo se conserva la estructura.
#[derive(Debug, Default)]
pub struct ExpertActivationRegistry {
    counts: DashMap<u32, AtomicU64>,
}

impl ExpertActivationRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments the activation count of one expert (global or shard-local index).
    pub fn record(&self, expert_id: u32) {
        let entry = self
            .counts
            .entry(expert_id)
            .or_insert_with(|| AtomicU64::new(0));
        entry.fetch_add(1, Ordering::Relaxed);
    }

    /// Drains the counts of **this** registry into a map and resets them.
    ///
    /// `swap(0)` (not a read followed by a store) so a `record` that lands between the two cannot be
    /// lost. Experts with zero activations are not reported, which is what `/announce` sent before.
    pub fn drain(&self) -> HashMap<u32, u64> {
        let mut usage = HashMap::new();
        for entry in self.counts.iter() {
            let count = entry.value().swap(0, Ordering::Relaxed);
            if count > 0 {
                usage.insert(*entry.key(), count);
            }
        }
        usage
    }

    /// How many activations this registry has recorded for `expert_id` since the last drain.
    ///
    /// Read-only helper: it exists so a test can ask the hot path what it recorded without draining
    /// (draining to inspect would hide the reset behaviour behind the assertion).
    #[must_use]
    pub fn recorded(&self, expert_id: u32) -> u64 {
        self.counts
            .get(&expert_id)
            .map(|entry| entry.value().load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Experts with a **pending** (non-zero) activation count, i.e. what the next `drain()` would
    /// report. After a drain the entries stay in the map with a zero count (bounded by the number of
    /// experts of the model), so counting entries would be misleading.
    #[must_use]
    pub fn experts(&self) -> usize {
        self.counts
            .iter()
            .filter(|entry| entry.value().load(Ordering::Relaxed) > 0)
            .count()
    }
}

/// Peers one node tracks failures for (`TASK-50`/S56). Same order of magnitude as the persisted
/// `known_peers` table (256 rows): a node only records peers it tried to reach, so the real bound is
/// the mesh it belongs to. Reaching the cap is **published** (`untracked_failures`), never silent.
pub const MAX_TRACKED_PEERS: usize = 256;

/// How often a hop to a peer failed, split by reason (`TASK-50`/S56).
///
/// This replaces the `tessella_network::CircuitBreaker`, which was **write-only**: `is_allowed`,
/// `state_of`, `get_all_infos` and `record_success` had zero callers in the whole workspace, so the
/// state machine it computed governed nothing while the two config keys that shaped it
/// (`circuit_breaker_failure_threshold` / `_cooldown_secs`) changed no behaviour. What is kept is
/// the honest half — the count — and it now has a reader: `GET /status`.
///
/// It is the local half of the answer to a hop that does not answer. There is deliberately **no**
/// third-party report channel (`TASK-50`/S56, decision `b`): `/report_failure` authenticates a
/// single identity (D-3) and accusing another peer without proof is the griefing S8 and S28 closed,
/// so the authority stays where it already is — the tracker (P2P `ConnectionClosed` + `/announce`)
/// for liveness, and the sampled compute audit (`TASK-27`/S33) for the work a peer claims.
#[derive(Debug, Default)]
pub struct PeerFailureRegistry {
    peers: DashMap<String, PeerFailureEntry>,
    untracked: AtomicU64,
}

#[derive(Debug, Default)]
struct PeerFailureEntry {
    total: AtomicU64,
    consecutive: AtomicU64,
    by_reason: [AtomicU64; REASON_COUNT],
    last_failure_unix_secs: AtomicU64,
    /// `PeerFailureReason::index()` of the most recent failure (`0` = `NetworkError`, the initial
    /// value of an entry that exists and was never updated — which cannot happen today, because an
    /// entry is only created by a failure).
    last_reason: AtomicU8,
}

impl PeerFailureRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one failed hop towards `peer_id`.
    ///
    /// The instant comes from [`crate::time::unix_now_secs`], the shared clock of the workspace.
    pub fn record_failure(&self, peer_id: &str, reason: PeerFailureReason) {
        let now = crate::time::unix_now_secs();
        if let Some(entry) = self.peers.get(peer_id) {
            entry.record(reason, now);
            return;
        }
        if self.peers.len() >= MAX_TRACKED_PEERS {
            // Bounded on purpose (a public mesh has churn) and counted, so the reader sees that
            // something was left out instead of a number that quietly stops growing.
            self.untracked.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                "📉 Peer failure registry is full ({MAX_TRACKED_PEERS} peers): '{peer_id}' is not \
                 tracked (untracked failures are published by /status)"
            );
            return;
        }
        self.peers
            .entry(peer_id.to_string())
            .or_default()
            .record(reason, now);
    }

    /// A hop to `peer_id` completed or the peer came back: the consecutive counter goes to zero.
    ///
    /// A no-op for a peer with no recorded failures: reading a success must not create state (the
    /// `is_allowed` of the old breaker did create an entry for every peer it was asked about).
    pub fn record_success(&self, peer_id: &str) {
        if let Some(entry) = self.peers.get(peer_id) {
            entry.consecutive.store(0, Ordering::Relaxed);
        }
    }

    /// Peers with at least one recorded failure.
    #[must_use]
    pub fn tracked_peers(&self) -> usize {
        self.peers.len()
    }

    /// The whole registry as `GET /status` publishes it.
    #[must_use]
    pub fn report(&self) -> PeerFailureReport {
        let mut peers: Vec<PeerFailureCounts> = self
            .peers
            .iter()
            .map(|entry| entry.value().counts(entry.key()))
            .collect();
        peers.sort_by(|a, b| {
            b.total_failures
                .cmp(&a.total_failures)
                .then_with(|| a.peer_id.cmp(&b.peer_id))
        });

        PeerFailureReport {
            total_failures: peers.iter().map(|counts| counts.total_failures).sum(),
            tracked_peers: peers.len(),
            untracked_failures: self.untracked.load(Ordering::Relaxed),
            peers,
        }
    }
}

impl PeerFailureEntry {
    fn record(&self, reason: PeerFailureReason, now: u64) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.consecutive.fetch_add(1, Ordering::Relaxed);
        self.by_reason[reason.index()].fetch_add(1, Ordering::Relaxed);
        self.last_failure_unix_secs.store(now, Ordering::Relaxed);
        self.last_reason
            .store(u8::try_from(reason.index()).unwrap_or(0), Ordering::Relaxed);
    }

    fn counts(&self, peer_id: &str) -> PeerFailureCounts {
        let mut by_reason = BTreeMap::new();
        for reason in PeerFailureReason::ALL {
            let count = self.by_reason[reason.index()].load(Ordering::Relaxed);
            if count > 0 {
                by_reason.insert(reason.as_str(), count);
            }
        }

        PeerFailureCounts {
            peer_id: peer_id.to_string(),
            total_failures: self.total.load(Ordering::Relaxed),
            consecutive_failures: self.consecutive.load(Ordering::Relaxed),
            last_failure_unix_secs: self.last_failure_unix_secs.load(Ordering::Relaxed),
            last_reason: PeerFailureReason::ALL
                [usize::from(self.last_reason.load(Ordering::Relaxed))]
            .as_str(),
            by_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Criterio 2 de `TASK-43`: dos registros no comparten nada. Antes el estado era un `static`, así
    /// que vaciar uno vaciaba el otro y los contadores de dos motores se mezclaban.
    #[test]
    fn two_registries_do_not_share_state() {
        let first = ExpertActivationRegistry::new();
        let second = ExpertActivationRegistry::new();

        first.record(0);
        first.record(0);
        second.record(7);

        assert_eq!(first.recorded(0), 2);
        assert_eq!(second.recorded(0), 0, "el segundo registro está intacto");

        let drained_first = first.drain();
        assert_eq!(drained_first.get(&0), Some(&2));
        assert_eq!(
            second.recorded(7),
            1,
            "vaciar un registro no puede vaciar el otro"
        );

        let drained_second = second.drain();
        assert_eq!(drained_second.get(&7), Some(&1));
        assert_eq!(first.experts(), 0, "el drain del primero ya lo reseteó");
    }

    /// El `drain` **resetea**: lo que se anuncia una vez no se vuelve a anunciar (el heartbeat manda
    /// el delta, no el acumulado).
    #[test]
    fn draining_resets_the_counters_of_that_registry() {
        let registry = ExpertActivationRegistry::new();
        registry.record(3);
        registry.record(3);
        registry.record(9);

        let usage = registry.drain();
        assert_eq!(usage.get(&3), Some(&2));
        assert_eq!(usage.get(&9), Some(&1));
        assert_eq!(registry.recorded(3), 0);
        assert_eq!(registry.experts(), 0, "un experto en 0 no se reporta");

        let again = registry.drain();
        assert!(again.is_empty(), "el segundo drain no tiene nada que dar");

        registry.record(9);
        let after_a_new_activation = registry.drain();
        assert_eq!(after_a_new_activation.get(&9), Some(&1));
    }

    /// Un registro que nadie tocó anuncia un mapa **vacío** (no `None`): la semántica del anuncio no
    /// cambia, el heartbeat sigue envolviendo el resultado en `Some(..)`.
    #[test]
    fn a_registry_without_activations_drains_to_an_empty_map() {
        let registry = ExpertActivationRegistry::new();
        assert!(registry.drain().is_empty());
    }

    /// Un clon de `Arc` es el **mismo** registro: es lo que comparten el motor y el heartbeat.
    #[test]
    fn a_cloned_handle_observes_the_same_counts() {
        use std::sync::Arc;

        let registry = Arc::new(ExpertActivationRegistry::new());
        let shared = Arc::clone(&registry);

        registry.record(5);
        assert_eq!(shared.recorded(5), 1);
        assert_eq!(shared.drain().get(&5), Some(&1));
        assert_eq!(
            registry.recorded(5),
            0,
            "el clon ve el mismo estado, incluido el reset"
        );
    }
}

#[cfg(test)]
mod peer_failure_tests {
    use super::*;

    /// The entry of one peer, or a panic naming what the report actually had.
    fn counts_for(report: &PeerFailureReport, peer_id: &str) -> PeerFailureCounts {
        report
            .peers
            .iter()
            .find(|counts| counts.peer_id == peer_id)
            .unwrap_or_else(|| panic!("{peer_id} is not in {:?}", report.peers))
            .clone()
    }

    /// `TASK-50`/S56: the datum is per **peer** and per **reason**, because the failure modes of a
    /// hop mean different things to the operator (a timeout is a slow node, an HTTP error is a node
    /// that answers but cannot serve, a refused connection is a node that is not there).
    #[test]
    fn failures_are_counted_per_peer_and_per_reason() {
        let registry = PeerFailureRegistry::new();

        registry.record_failure("12D3KooWSlow", PeerFailureReason::Timeout);
        registry.record_failure("12D3KooWSlow", PeerFailureReason::Timeout);
        registry.record_failure("12D3KooWSlow", PeerFailureReason::NetworkError);
        registry.record_failure("12D3KooWGone", PeerFailureReason::DialFailed);

        let report = registry.report();
        assert_eq!(report.tracked_peers, 2);
        assert_eq!(report.total_failures, 4);
        assert_eq!(report.untracked_failures, 0);

        let slow = counts_for(&report, "12D3KooWSlow");
        assert_eq!(slow.total_failures, 3);
        assert_eq!(slow.consecutive_failures, 3);
        assert_eq!(slow.by_reason.get("timeout"), Some(&2));
        assert_eq!(slow.by_reason.get("network_error"), Some(&1));
        assert_eq!(slow.last_reason, "network_error");

        let gone = counts_for(&report, "12D3KooWGone");
        assert_eq!(gone.total_failures, 1);
        assert_eq!(gone.last_reason, "dial_failed");
    }

    /// A hop that completes proves the peer is alive again: the consecutive count goes back to zero
    /// while the cumulative total — what the operator looks at for a long-running node — does not.
    #[test]
    fn a_successful_hop_resets_the_consecutive_counter_not_the_total() {
        let registry = PeerFailureRegistry::new();
        registry.record_failure("12D3KooWFlaky", PeerFailureReason::Timeout);
        registry.record_failure("12D3KooWFlaky", PeerFailureReason::Timeout);

        registry.record_success("12D3KooWFlaky");

        let counts = counts_for(&registry.report(), "12D3KooWFlaky");
        assert_eq!(counts.consecutive_failures, 0);
        assert_eq!(counts.total_failures, 2, "lo que ya falló no se olvida");
        assert_eq!(counts.by_reason.get("timeout"), Some(&2));
    }

    /// Reading a success must not create state. The `CircuitBreaker` this replaces did exactly the
    /// opposite (`is_allowed` inserted an entry for every peer it was asked about), which is how a
    /// "metric" ends up owning topology.
    #[test]
    fn recording_a_success_for_an_unknown_peer_creates_nothing() {
        let registry = PeerFailureRegistry::new();
        registry.record_success("12D3KooWNeverFailed");

        assert_eq!(registry.tracked_peers(), 0);
        assert!(registry.report().peers.is_empty());
    }

    /// The report is what `GET /status` publishes, so it must be readable at a glance: the peers
    /// with more failures come first and a reason with zero failures is not listed.
    #[test]
    fn the_report_is_sorted_and_omits_reasons_at_zero() {
        let registry = PeerFailureRegistry::new();
        registry.record_failure("12D3KooWOne", PeerFailureReason::HttpError);
        registry.record_failure("12D3KooWMany", PeerFailureReason::Timeout);
        registry.record_failure("12D3KooWMany", PeerFailureReason::Timeout);

        let report = registry.report();
        assert_eq!(report.peers[0].peer_id, "12D3KooWMany");
        assert_eq!(report.peers[1].peer_id, "12D3KooWOne");

        let one = counts_for(&report, "12D3KooWOne");
        assert_eq!(
            one.by_reason.len(),
            1,
            "solo el motivo que ocurrió: {:?}",
            one.by_reason
        );
        assert!(!one.by_reason.contains_key("timeout"));
    }

    /// Nothing silent: an untouched registry reports zeros, so the reader of `/status` never has to
    /// distinguish "no failures" from "no field".
    #[test]
    fn an_empty_registry_reports_zeros() {
        let report = PeerFailureRegistry::new().report();
        assert_eq!(report.total_failures, 0);
        assert_eq!(report.tracked_peers, 0);
        assert_eq!(report.untracked_failures, 0);
        assert!(report.peers.is_empty());
    }

    /// Every entry carries an instant from the shared clock helper, so a failure published by
    /// `/status` can be compared with the tracker's log.
    #[test]
    fn every_entry_carries_the_instant_of_its_last_failure() {
        let before = crate::time::unix_now_secs();
        let registry = PeerFailureRegistry::new();
        registry.record_failure("12D3KooWWhen", PeerFailureReason::HttpError);
        let after = crate::time::unix_now_secs();

        let counts = counts_for(&registry.report(), "12D3KooWWhen");
        assert!(
            counts.last_failure_unix_secs >= before && counts.last_failure_unix_secs <= after,
            "{} is not in [{before}, {after}]",
            counts.last_failure_unix_secs
        );
    }

    /// The map is bounded (a public mesh has churn) and what it cannot hold it **counts** instead of
    /// dropping in silence.
    #[test]
    fn the_cap_counts_what_it_cannot_track() {
        let registry = PeerFailureRegistry::new();
        for index in 0..MAX_TRACKED_PEERS {
            registry.record_failure(&format!("12D3KooWPeer{index}"), PeerFailureReason::Timeout);
        }
        assert_eq!(registry.tracked_peers(), MAX_TRACKED_PEERS);

        registry.record_failure("12D3KooWOneTooMany", PeerFailureReason::Timeout);
        registry.record_failure("12D3KooWOneTooMany", PeerFailureReason::Timeout);

        let report = registry.report();
        assert_eq!(report.tracked_peers, MAX_TRACKED_PEERS);
        assert_eq!(report.untracked_failures, 2);
        assert!(
            report
                .peers
                .iter()
                .all(|counts| counts.peer_id != "12D3KooWOneTooMany")
        );
    }

    /// The names are the ones `/status` publishes and the ones a future consumer would have to
    /// understand: distinct, stable and lowercase.
    #[test]
    fn every_reason_has_a_distinct_name() {
        assert_eq!(PeerFailureReason::ALL.len(), REASON_COUNT);
        let mut names: Vec<&str> = PeerFailureReason::ALL
            .iter()
            .map(|reason| reason.as_str())
            .collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), REASON_COUNT, "dos motivos comparten nombre");
        assert!(
            names
                .iter()
                .all(|name| name.chars().all(|c| c.is_ascii_lowercase() || c == '_')),
            "{names:?}"
        );
    }
}
