use crate::network_manager::resolve_preferred_http_addr;
use crate::routing_table::RoutingTable;
use crate::topology_cache::TopologyCache;
use async_trait::async_trait;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tessella_core::error::DomainError;
use tessella_core::telemetry::metrics::{PeerFailureReason, PeerFailureRegistry};
use tessella_core::traits::{AuditOrder, PipelineRouter};
use tessella_network::NetworkKeypair;
use tessella_protocol::models::{majority_digest, AuditExecution, AuditReport};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

pub struct HttpPipelineRouter {
    routing_table: Arc<RwLock<RoutingTable>>,
    topology_cache: Arc<RwLock<TopologyCache>>,
    http_client: Client,
    orchestrator_url: String,
    peer_id: String,
    model_id: String,
    peer_failures: Arc<PeerFailureRegistry>,
    notify: Arc<tokio::sync::Notify>,
    /// How long **this node** waits for a hop of its own pipeline to answer
    /// (`p2p_inference_timeout_secs`, `TASK-76`/S83).
    ///
    /// It used to be a literal in the middle of the loop — `tokio::time::timeout(Duration::from_secs(5),
    /// …)` around the request —, which **overrode** both the 30 s declared in this same call
    /// (`.timeout(Duration::from_secs(30))` on the `RequestBuilder`) and `p2p_inference_timeout_secs`, a
    /// key with **zero** consumers in the whole workspace. That literal is the second half of the
    /// defect: a hop that spends more than five seconds applying a prompt of the published ceiling
    /// (measured in `S83`: 2 037 tokens, 3.5 MiB of hidden state) is a hop that is **working**, and the
    /// literal declared it dead — counted the failure, burned the next candidate and, when there was
    /// none left, suspended 30 s to end in the generic `500 … Pipeline timeout` of `S76`. The `413` of
    /// axum was gone; the client's symptom was not.
    ///
    /// It is **declared** and not derived from the payload on purpose: what a hop costs depends on its
    /// GPU, its shard and its load, and the operator is who measured them. It is the only timer of the
    /// hop: with two (a client timeout and this wrapper) which reason gets counted — `timeout`, which
    /// replaces the candidate, or `network_error` — would be a race.
    hop_timeout: Duration,
    /// Identity of this node, used to sign the [`AuditReport`] of a sampled session (`TASK-53`/S59).
    ///
    /// S8 injected the keypair into the callers that report to the orchestrator (`/announce`,
    /// `/ready`, `/report_failure`); the audit is the first report the **router** itself sends, and
    /// only the entry node of a session may send it, so the router needs the key.
    keypair: Arc<NetworkKeypair>,
    /// Pinned downstream peers for active sessions (`TASK-81` · S88).
    ///
    /// Ensures that mid-generation decode steps stay on the same peer that executed prefill (and
    /// holds the KV cache), avoiding false `409 missing_prefill` errors when twin replicas exist.
    session_pins: Arc<tessella_core::routing::SessionPinRegistry>,
}

impl HttpPipelineRouter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        routing_table: Arc<RwLock<RoutingTable>>,
        topology_cache: Arc<RwLock<TopologyCache>>,
        http_client: Client,
        orchestrator_url: String,
        peer_id: String,
        model_id: String,
        peer_failures: Arc<PeerFailureRegistry>,
        notify: Arc<tokio::sync::Notify>,
        keypair: Arc<NetworkKeypair>,
        hop_timeout: Duration,
    ) -> Self {
        Self::with_session_pins(
            routing_table,
            topology_cache,
            http_client,
            orchestrator_url,
            peer_id,
            model_id,
            peer_failures,
            notify,
            keypair,
            hop_timeout,
            Arc::new(tessella_core::routing::SessionPinRegistry::new()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_session_pins(
        routing_table: Arc<RwLock<RoutingTable>>,
        topology_cache: Arc<RwLock<TopologyCache>>,
        http_client: Client,
        orchestrator_url: String,
        peer_id: String,
        model_id: String,
        peer_failures: Arc<PeerFailureRegistry>,
        notify: Arc<tokio::sync::Notify>,
        keypair: Arc<NetworkKeypair>,
        hop_timeout: Duration,
        session_pins: Arc<tessella_core::routing::SessionPinRegistry>,
    ) -> Self {
        Self {
            routing_table,
            topology_cache,
            http_client,
            orchestrator_url,
            peer_id,
            model_id,
            peer_failures,
            notify,
            hop_timeout,
            keypair,
            session_pins,
        }
    }

    pub fn session_pins(&self) -> &Arc<tessella_core::routing::SessionPinRegistry> {
        &self.session_pins
    }

    /// `TASK-27`/S33 · `TASK-53`/S59 · `TASK-70`/S77: the **active** half of capa 3 — executes the
    /// **marked step** on up to `replicas` extra peers of the same layer range, compares the tensors
    /// and reports what it saw.
    ///
    /// S33 compared the answer of the primary with **one** replica. S59 turned that into a majority:
    /// the orchestrator sends the number of extra executions signed inside the ticket, and with three
    /// executions a 2-1 split names the dissident, which is what lets the orchestrator pay the two
    /// that agree and refuse only the third. S77 fixed **what** is compared: the marked step (a signed
    /// `audit_layer`, not «the shard that ends the pipeline»), which the marked hop executes in
    /// isolation — so the digest attributed to a peer is the digest *of that peer's work* — and the
    /// replicas must hold the **same** range, because a peer that owns another range computes another
    /// step and would look divergent while being honest.
    ///
    /// The comparison happens here, over `Payload::data` as this node receives it, so nothing has to be
    /// trusted: the same input travels to every execution and the answers must be the same tensor.
    ///
    /// The verdict travels back as a signed [`AuditReport`] (`POST /api/v1/audit_report`). It is what
    /// makes the credit of that range follow the *evidence* instead of the order of arrival: the claims
    /// reach the orchestrator through each worker's WAL sweeper (up to 60 s later), while this report
    /// is sent while the pipeline is still running, so the orchestrator can refuse the **first** claim
    /// of a diverging work — the hole S33 left open.
    ///
    /// Failure modes, deliberately asymmetric:
    ///
    /// * **No second replica of that range** (the common case: a two-node pipeline replicates nothing):
    ///   `warn!` and continue. The audit is a sample, not a gate.
    /// * **A replica does not answer** (down, timeout, HTTP error): `warn!`, count it and continue.
    ///   With fewer than two executions there is nothing to compare, so no report is sent and the
    ///   claim-based rule of S33 applies.
    /// * **The executions disagree**: `Err`. This is the whole point of sampling — an answer two
    ///   executions do not corroborate is not served, so a client never receives output the mesh
    ///   cannot reproduce. The peers that did not agree with the majority are the dissidents, and
    ///   they are the ones counted locally (`peer_failures`) and penalised in the routing table.
    async fn audit_against_replicas(
        &self,
        step: (usize, usize),
        request_bytes: &[u8],
        primary_peer: &str,
        primary_response: &[u8],
        replicas: usize,
        canary: Option<&str>,
    ) -> Result<(), DomainError> {
        let step_start = step.0;
        let primary = tensor_evidence(primary_response)?;
        let mut executions = vec![AuditExecution {
            peer_id: primary_peer.to_string(),
            output_hash: primary.digest,
        }];
        let mut fingerprints: Vec<(String, String)> =
            vec![(primary_peer.to_string(), primary.fingerprint)];

        let candidates = {
            let table = self.routing_table.read().await;
            let mut candidates = table
                .get_peers_for_layer(step_start)
                .into_iter()
                // `TASK-70`/S77 · `TASK-96`/S102: same **range** or nothing — the range of the peer
                // that covers this step, which with more than one range per node is no longer «the
                // peer's range». The replica has to compute *this* step to be comparable: any other
                // range is other work, and taking it as a replica would turn an honest node into a
                // false dissident.
                .filter(|peer| {
                    tessella_protocol::layer_ranges::range_covering(&peer.layer_ranges, step_start)
                        == Some(step)
                })
                .filter(|peer| peer.peer_id != primary_peer && !table.is_suppressed(&peer.peer_id))
                // `TASK-83`/S90 (`R2`): the canary is **not** one of the `replicas` the session paid
                // for. It used to be inserted *before* the truncation below, so declaring a canary
                // silently took the slot of the honest replica the comparison needs.
                // `TASK-87`/S94: any canary is excluded from honest replicas (`!peer.canary`).
                .filter(|peer| canary != Some(peer.peer_id.as_str()) && !peer.canary)
                .collect::<Vec<_>>();
            candidates.truncate(replicas);
            // `TASK-71`/S78: the operator's canary is **always** one of the executions when the signed
            // ticket names one, and S90 keeps that where it belongs — **on top of** the replicas the
            // session bought. The reward of the audit depends on the report showing its divergence (the
            // orchestrator knows the canary's answer is wrong without executing the model), so leaving
            // the choice to the latency-based pick would make the check depend on which peer happened to
            // be fastest — and a lazy auditor could drop the canary by simply not calling it. It never
            // votes, though: see the majority below (`R1`).
            if let Some(canary) = canary {
                if !executions
                    .iter()
                    .any(|execution| execution.peer_id == canary)
                {
                    if let Some(peer) =
                        table
                            .get_peers_for_layer(step_start)
                            .into_iter()
                            .find(|peer| {
                                peer.peer_id == canary
                                    && tessella_protocol::layer_ranges::range_covering(
                                        &peer.layer_ranges,
                                        step_start,
                                    ) == Some(step)
                            })
                    {
                        candidates.push(peer);
                    }
                }
            }
            candidates
        };

        if candidates.is_empty() {
            warn!(
                "🧾 Audited session: no replica covers layer {step_start} besides {primary_peer}, so \
                 this step cannot be audited (TASK-27: the audit is a sample, never a gate)"
            );
            return Ok(());
        }

        for replica in candidates {
            let Some(url) = hop_url(&replica) else {
                warn!(
                    "🧾 Audited session: replica {} exposes no reachable HTTP address, so layer {} \
                     stays unaudited on it (TASK-27/S59)",
                    replica.peer_id, step_start
                );
                continue;
            };

            let response = self
                .http_client
                .post(&url)
                .header("Content-Type", "application/octet-stream")
                // `TASK-84`/S91 (b′): the honest lane, the same one the hop below declares (see the
                // long comment there) — the audit re-sends a body of the same size, so it deserves the
                // same chance of being refused *before* the bytes travel.
                .header("Expect", "100-continue")
                .body(request_bytes.to_vec())
                .timeout(std::time::Duration::from_secs(30))
                .send()
                .await;

            let audited_bytes = match response {
                Ok(resp) if resp.status().is_success() => match resp.bytes().await {
                    Ok(bytes) => bytes.to_vec(),
                    Err(e) => {
                        warn!(
                            "🧾 Audited session: could not read the audit response of {} for layer \
                             {} ({e}); the primary answer stands",
                            replica.peer_id, step_start
                        );
                        continue;
                    }
                },
                Ok(resp) => {
                    warn!(
                        "🧾 Audited session: replica {} answered {} for layer {}; the primary answer \
                         stands",
                        replica.peer_id,
                        resp.status(),
                        step_start
                    );
                    continue;
                }
                Err(e) => {
                    warn!(
                        "🧾 Audited session: replica {} did not answer for layer {} ({e}); the primary \
                         answer stands",
                        replica.peer_id, step_start
                    );
                    continue;
                }
            };

            let audited = tensor_evidence(&audited_bytes)?;
            fingerprints.push((replica.peer_id.clone(), audited.fingerprint));
            executions.push(AuditExecution {
                peer_id: replica.peer_id.clone(),
                output_hash: audited.digest,
            });
        }

        // `TASK-83`/S90 (`R1`): **the canary does not vote**. It is the control the ticket named — the one
        // execution whose answer the orchestrator already knows is wrong — so it belongs in the report
        // (that is exactly what `qualify_report` checks) and never in the majority. Until S90 it was one
        // more execution, and since it dissents by construction it decided the outcome of every sampled
        // session it took part in: with two executions it produced a tie (no credit for the honest hop,
        // a `DivergentAudit` for it and an aborted request) and with three it aborted the request even
        // when the honest majority was clear.
        let voters: Vec<AuditExecution> = executions
            .iter()
            .filter(|execution| canary != Some(execution.peer_id.as_str()))
            .cloned()
            .collect();

        // A comparison needs **two honest executions**. With fewer there is nothing reproducible to
        // attribute, and the session falls back to the claim rule of S33: a `warn!`, and **no penalty**
        // for anybody — no node can be blamed for a mesh that gave the marked step no second honest
        // owner. Sending a report from here would be worse than useless: it would be a report that names
        // a canary and no honest majority, which is what `audit_credit` refuses to pay.
        if voters.len() < 2 {
            warn!(
                "🧾 Audited session: layer {step_start} has {} honest execution(s) besides \
                 {primary_peer}, and the canary the ticket named is a control rather than a second \
                 opinion: there is nothing to compare, so no report is sent and its claims are judged by \
                 the rule of S33 (TASK-53 · TASK-83/S90)",
                voters.len()
            );
            return Ok(());
        }

        self.send_audit_report(request_bytes, step, &executions)
            .await;

        let rendered = fingerprints
            .iter()
            .map(|(peer, fingerprint)| format!("{peer} → {fingerprint}"))
            .collect::<Vec<_>>()
            .join(", ");

        if let Some(agreed) = majority_digest(&voters) {
            if voters
                .iter()
                .all(|execution| execution.output_hash == agreed)
            {
                info!(
                    "🧾 Audited session: layer {step_start} of request confirmed by {} honest \
                     execution(s) ({rendered}){}",
                    voters.len(),
                    match canary {
                        Some(canary) =>
                            format!("; the canary {canary} dissented as expected (TASK-71/S78)"),
                        None => String::new(),
                    }
                );
                return Ok(());
            }

            // A strict majority **among the honest executions**: the ones that did not agree with it are
            // the dissidents, and they are the only ones penalised locally. The majority keeps its
            // route, and the canary is never part of this count (`voters` excludes it by construction).
            let dissidents = voters
                .iter()
                .filter(|execution| execution.output_hash != agreed)
                .map(|execution| execution.peer_id.clone())
                .collect::<Vec<_>>();
            error!(
                "🚨 AUDIT DIVERGENCE on layer {step_start}: {dissidents:?} did not reproduce the tensor \
                 the majority agreed on ({}). The request is not served and the orchestrator decides \
                 the credit from the report (TASK-53)",
                hex::encode(agreed)
            );
            for peer in &dissidents {
                self.record_hop_failure(peer, PeerFailureReason::DivergentAudit)
                    .await;
            }
            return Err(DomainError::Network(format!(
                "audit divergence on layer {step_start}: {dissidents:?} disagree with the majority of \
                 the executions about the tensor they produced for the same input; the request was \
                 not served (TASK-53)"
            )));
        }

        // No honest majority (two honest executions that disagree, or three different answers):
        // nobody can be attributed, so every honest execution of this step is penalised locally and the
        // request fails. Paying such a step would reward the dissident half the time. The canary is not
        // among them: it is the control, and it dissenting is what the ticket asked for.
        let peers = voters
            .iter()
            .map(|execution| execution.peer_id.clone())
            .collect::<Vec<_>>();
        error!(
            "🚨 AUDIT DIVERGENCE on layer {step_start}: {} honest executions produced different tensors \
             ({rendered}) and there is no majority, so neither the request nor any claim of it is \
             served (TASK-53)",
            voters.len()
        );
        for peer in &peers {
            self.record_hop_failure(peer, PeerFailureReason::DivergentAudit)
                .await;
        }
        Err(DomainError::Network(format!(
            "audit divergence on layer {step_start}: {peers:?} disagree and there is no majority; the \
             request was not served (TASK-53)"
        )))
    }

    /// Signs and delivers the [`AuditReport`] of a sampled step (`TASK-53`/S59).
    ///
    /// Losing it is **not** a failure of the request: the report is what decides the credit of the
    /// marked range, and without it the orchestrator falls back to the claim-based rule of S33 (the
    /// first claim is paid). So every failure is a `warn!` that says what is lost, and the report
    /// never changes what the client gets.
    ///
    /// No report is sent when the payload carries no ticket: without it the orchestrator cannot tell
    /// that the session exists, that it was sampled and who its entry node is, and it would have to
    /// trust the report's own word for all three.
    async fn send_audit_report(
        &self,
        request_bytes: &[u8],
        layer_range: (usize, usize),
        executions: &[AuditExecution],
    ) {
        let payload = match tessella_inference::HiddenStatePayload::from_bytes(request_bytes) {
            Ok(payload) => payload,
            Err(e) => {
                warn!("🧾 Could not read the audited payload back to report it ({e})");
                return;
            }
        };
        let Some(ticket) = payload.work_ticket.clone() else {
            warn!(
                "🧾 Audited session '{}' carries no work ticket, so its audit cannot be reported: the \
                 orchestrator will judge its claims as before (TASK-53)",
                payload.request_id
            );
            return;
        };

        let mut report = AuditReport {
            request_id: payload.request_id.clone(),
            layer_range,
            input_hash: if layer_range.0 == 0 {
                [0u8; 32]
            } else {
                tensor_digest(&payload.data)
            },
            executions: executions.to_vec(),
            work_ticket: ticket,
            auditor_peer_id: self.peer_id.clone(),
            timestamp: tessella_core::time::unix_now_secs(),
            public_key: Vec::new(),
            signature: Vec::new(),
        };

        if let Err(e) = tessella_network::sign_peer_request(&self.keypair, &mut report) {
            warn!(
                "🧾 Could not sign the audit report of '{}': {e}",
                report.request_id
            );
            return;
        }

        let url = format!(
            "{}/api/v1/audit_report",
            self.orchestrator_url.trim_end_matches('/')
        );
        match self
            .http_client
            .post(&url)
            .json(&report)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => info!(
                "🧾 Audit report of session '{}' layers {layer_range:?} delivered to the \
                 orchestrator ({} executions)",
                report.request_id,
                report.executions.len()
            ),
            Ok(response) => warn!(
                "🧾 The orchestrator answered {} to the audit report of '{}': the credit of layers \
                 {layer_range:?} falls back to the claim-based rule (TASK-53)",
                response.status(),
                report.request_id
            ),
            Err(e) => warn!(
                "🧾 Could not deliver the audit report of '{}' ({e}): the credit of layers \
                 {layer_range:?} falls back to the claim-based rule (TASK-53)",
                report.request_id
            ),
        }
    }
}

/// Explains why a failed next hop is only punished **locally**.
///
/// The router used to POST `{reporter_id, target_id, reason}` to the orchestrator's
/// `/report_failure`. That payload matched **neither** variant of the untagged
/// `FailureReportPayload` enum (`FailureReport` needs `peer_id`/`request_id`/`error`/`timestamp`,
/// `WeightPoisoning` needs `target_peer_id`/`model_id`/`role`), so axum answered `422` and the
/// report was silently dropped. Since D-3 (`TASK-03` Step B) that endpoint additionally only
/// accepts a request signed by the identity it carries, so a peer can no longer report
/// **another** peer without a cryptographic receipt (`WeightPoisoning`). Rather than keep a
/// broken call, the router keeps the local penalty (circuit breaker + routing table) and says so
/// here; third-party failure reports with corroboration are `TASK-22`.
impl HttpPipelineRouter {
    /// A hop to `peer_id` did not complete: the failure is counted, published by `GET /status`
    /// (`peer_failures`) and penalised in this node's routing table, so the next layer prefers
    /// somebody else.
    ///
    /// This is deliberately the **whole** consequence (`TASK-50`/S56, decision `b`). The router used
    /// to POST `{reporter_id, target_id, reason}` to the orchestrator's `/report_failure`, a payload
    /// that matched neither variant of `FailureReportPayload` (axum answered `422` and the report
    /// was dropped in silence). Since D-3 (`TASK-03` Step B) that endpoint only accepts a request
    /// signed by the identity it carries, so a peer cannot accuse **another** peer without a
    /// cryptographic receipt — and opening a two-identity channel for this would buy the tracker a
    /// signal it already has (P2P `ConnectionClosed` plus the `/announce` heartbeat) at the price of
    /// a breaking DTO and a griefing surface whose only mitigation — three distinct reporters — is
    /// *counted*, not proven (S28). The authority on the work a peer claims stays where it is: the
    /// sampled compute audit (`TASK-27`/S33).
    async fn record_hop_failure(&self, peer_id: &str, reason: PeerFailureReason) {
        self.peer_failures.record_failure(peer_id, reason);
        self.routing_table.write().await.penalize_peer(peer_id);
        warn!(
            "⚠️ Hop to {peer_id} failed ({}): counted locally and published by `GET /status` \
             (`peer_failures`), and {reporter} does not report another peer to the orchestrator \
             (third-party accusations need proof: D-3 · S8/S28)",
            reason.as_str(),
            reporter = self.peer_id,
        );
    }
}

/// What two honest executions of the same shard step must agree on: the **tensor** inside a
/// hidden-state payload.
///
/// The envelope is deliberately excluded. Its seal carries a fresh nonce, `iat` and signature per
/// execution, so comparing the serialized payloads would make two honest replicas look divergent —
/// the same trap that made the old `output_hash` (taken over the sealed payload) useless for the
/// audit. `shape` and `dtype` are part of the fingerprint so a replica that answers with the right
/// bytes but the wrong geometry is a disagreement too.
struct TensorEvidence {
    /// `sha256` of the tensor bytes: the digest that travels in the [`AuditReport`] and in every
    /// `WorkProof` of the session.
    digest: [u8; 32],
    /// `shape` + `dtype` + the digest in hex, for the operator's log.
    fingerprint: String,
}

fn tensor_evidence(payload_bytes: &[u8]) -> Result<TensorEvidence, DomainError> {
    let payload = tessella_inference::HiddenStatePayload::from_bytes(payload_bytes)
        .map_err(|e| DomainError::Inference(e.to_string()))?;
    let digest = tensor_digest(&payload.data);

    Ok(TensorEvidence {
        digest,
        fingerprint: format!(
            "shape={:?} dtype={:?} sha256={}",
            payload.shape,
            payload.dtype,
            hex::encode(digest)
        ),
    })
}

/// `sha256` of a bare tensor (`Payload::data`), which is the unit both planes compare.
fn tensor_digest(data: &[u8]) -> [u8; 32] {
    use sha2::Digest;

    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// The range of `peer` that serves `step`, or a refusal (`TASK-96`/S102).
///
/// The routing index answers by **layer** (`RoutingTable::get_peers_for_layer`), so any peer that
/// comes out of it for `step` declares a range covering it — that is what the index is built from.
/// This turns that agreement into something the code checks: when the two views disagree, the hop
/// stops here, naming the peer, instead of travelling to a range the node does not serve
/// (invariant 17 — explicit failure over silent degradation).
///
/// It is also what keeps the hop contract of `TASK-70`/`S77` intact with more than one range: the
/// step being served is *the range that covers it*, never «the peer's range», because a node can now
/// own two.
fn step_range_of(
    peer: &tessella_core::models::LayerPeer,
    step: usize,
) -> Result<(usize, usize), DomainError> {
    tessella_protocol::layer_ranges::range_covering(&peer.layer_ranges, step).ok_or_else(|| {
        DomainError::Inference(format!(
            "Peer {} is a candidate for layer {step} but declares no range covering it ({:?}): \
             refusing to send a hop it cannot serve (TASK-96/S102)",
            peer.peer_id, peer.layer_ranges
        ))
    })
}

/// Resolves the `/v1/forward` URL of a peer, preferring the fully-resolved HTTP address (it already
/// carries any public proxy URL) over the legacy `ip:port` pair. `None` when the peer exposes no
/// reachable address at all, which is what the audit reports before giving up.
fn hop_url(peer: &tessella_core::models::LayerPeer) -> Option<String> {
    let resolved = resolve_preferred_http_addr(&peer.http_addr, &peer.p2p_addr);
    if !resolved.trim().is_empty() {
        return Some(format!("{}/v1/forward", resolved.trim_end_matches('/')));
    }
    match (&peer.public_ip, peer.public_api_port) {
        (Some(ip), Some(port)) => Some(format!("http://{ip}:{port}/v1/forward")),
        _ => None,
    }
}

/// Which **named refusal** a hop's answer carries, if any (`TASK-68`/S75, extended by `TASK-76`/S83
/// and by `TASK-84`/S91).
///
/// The machine-readable half is the `X-Tessella-Refusal` header the HTTP edge publishes
/// (`tessella_core::error::REFUSAL_HEADER`); the body is the fallback, and the code is its first word.
/// Accepting both is deliberate: a proxy that drops unknown headers must not be able to turn a refusal
/// into the silent retry that produces plausible garbage.
///
/// A bare `413` counts as [`tessella_core::error::PAYLOAD_TOO_LARGE_CODE`] even when nothing names it.
/// In this mesh that status has one meaning — a hop body over `MAX_HIDDEN_STATE_BYTES` — and it is the
/// status `S76` measured coming from a limit **no node had published**: the router retried it, marked
/// the honest nodes as failures, ran out of candidates, waited the 30 s of the suspension and answered
/// `500`. Deciding from the status is strictly better than that cascade, and it costs nothing when the
/// header is there.
///
/// There is deliberately **no** such fallback for the quota of `TASK-84`/S91: a bare `503` in this mesh
/// also means «this node is restarting», which is a failure worth replacing. The code is required, and
/// a node that answers one without naming it gets the old behaviour (retry) instead of a wrong abort.
fn refusal_of_hop(
    header: Option<&str>,
    status: reqwest::StatusCode,
    body: &str,
) -> Option<&'static str> {
    use tessella_core::error::{
        MISSING_PREFILL_CODE, PAYLOAD_TOO_LARGE_CODE, PROMPT_TOO_LONG_CODE,
        TOO_MANY_HOP_BODIES_CODE,
    };
    const CODES: [&str; 4] = [
        MISSING_PREFILL_CODE,
        PROMPT_TOO_LONG_CODE,
        PAYLOAD_TOO_LARGE_CODE,
        TOO_MANY_HOP_BODIES_CODE,
    ];
    let code_of_header = header
        .map(str::trim)
        .and_then(|header| CODES.into_iter().find(|code| *code == header));
    if let Some(code) = code_of_header {
        return Some(code);
    }
    if let Some(code) = CODES
        .into_iter()
        .find(|code| body.trim_start().starts_with(code))
    {
        return Some(code);
    }
    (status == reqwest::StatusCode::PAYLOAD_TOO_LARGE).then_some(PAYLOAD_TOO_LARGE_CODE)
}

/// The refusal re-issued by the node that was **routing** the step, when the hop's refusal is about
/// the **bytes** of this request (`TASK-76`/S83: `prompt_too_long`, `payload_too_large`).
///
/// It keeps the code as its first word, so the entry node's API maps it to the same `400`/`413` the
/// hop answered with, and a router upstream of this one aborts too. It adds the two facts only this
/// node knows: **which layer** was in flight and **which hop** refused it. The body of the refusal
/// (which names the tokens, the bytes and the limit that binds) travels with it.
fn refusal_on_hop(code: &'static str, hop: &str, layer: usize, body: &str) -> DomainError {
    DomainError::Validation(format!(
        "{code}: hop {hop} refused layer {layer} of this pipeline because the bytes of this request \
         do not fit in one hop, and they are the same bytes for every other candidate: the request is \
         aborted instead of retried (TASK-76/S83). The hop said: {}",
        body.trim()
    ))
}

/// The refusal re-issued by the node that was **routing** the step, when the hop it chose is **at its
/// quota of hop bodies** (`TASK-84`/S91).
///
/// It needs its own emission — and not [`refusal_on_hop`] — because the prose of the size refusals is
/// **false** here: those bytes *do* fit in that node (as soon as it is free) and they *do* fit in
/// another candidate right now. What makes this refusal terminal is not the size of the request but the
/// cost of the retry: the same body is 70-80 MiB, measured in `S91`, and pushing it at every candidate
/// in turn is the amplification the quota exists to stop. So the message says what is true — this node
/// is full, not broken — and tells the client to retry, which is what its `503` means.
///
/// It adds the two facts only this node knows: **which layer** was in flight and **which hop** was full.
fn busy_on_hop(code: &'static str, hop: &str, layer: usize, body: &str) -> DomainError {
    DomainError::Validation(format!(
        "{code}: hop {hop} refused layer {layer} of this pipeline because it was already reading its \
         quota of hop bodies. The node is full, not broken, and the refusal is **terminal** here: \
         this node does not re-send the body to another candidate, because the body is the same 70-80 \
         MiB for every one of them and pushing it at each in turn is exactly the amplification a quota \
         exists to stop (TASK-84/S91). The retry belongs to the client: the same request will be \
         served by that hop, or by another, as soon as one of them is free. The hop said: {}",
        body.trim()
    ))
}

/// The refusal re-issued by the node that was **routing** the step (decisión (a), `TASK-68`/S75).
///
/// It keeps the code as its first word, so the entry node's API maps it to the same `409` the hop
/// answered with (`tessella_core::error::is_missing_prefill`), and it adds the two facts only this
/// node knows: **which layer** was in flight and **which hop** refused it. The body of the refusal
/// (which names the session, the prefix and the layers of the refusing node) travels with it.
fn missing_prefill_on_hop(hop: &str, layer: usize, body: &str) -> DomainError {
    DomainError::Validation(format!(
        "{}: hop {hop} refused layer {layer} of this pipeline because the session's prefix never \
         passed through it, and the same bytes would produce plausible garbage in any other \
         candidate: the request is aborted instead of retried (TASK-68/S75, decision (a)). The hop \
         said: {}",
        tessella_core::error::MISSING_PREFILL_CODE,
        body.trim()
    ))
}

#[async_trait]
impl PipelineRouter for HttpPipelineRouter {
    async fn route(
        &self,
        layer_start: usize,
        total_layers: usize,
        payload_bytes: Vec<u8>,
        model_id: &str,
        audit: Option<AuditOrder>,
    ) -> Result<Vec<u8>, DomainError> {
        let mut current_layer_start = layer_start;
        let mut current_payload_bytes = payload_bytes;
        // `TASK-81`/S88: session continuity key used to maintain primary peer stability.
        let session_key =
            tessella_inference::HiddenStatePayload::peek_session_key(&current_payload_bytes);
        // `S43`/`TASK-37`: at most one `warn!` per request about peers that announce an expert range.
        let mut warned_about_expert_shards = false;

        // 1. Refresh if stale (> 300s). No eager refresh on empty because table
        // is now pre-populated at registration.
        {
            let table = self.routing_table.read().await;
            if table.age() > std::time::Duration::from_secs(30) {
                drop(table);
                let _ = self.refresh().await;
            }
        }

        while current_layer_start < total_layers {
            let mut tried_peers = std::collections::HashSet::new();
            // `TASK-72`/S79: the exploration share is drawn **once per layer step**, not once per
            // attempt — a retry is the same hop with one candidate fewer (`tried_peers`), and drawing
            // again would turn it into a second lottery instead of the first one minus a loser.
            let exploration_draw: f64 = rand::random();

            // `TASK-83`/S90 (`R6`): «a canary never serves a client». The step the ticket marked is
            // exactly the one the canary replicates — that is why the orchestrator named it — so it is
            // excluded from the **primary** pick of that hop. The audit asks it explicitly
            // (`audit_against_replicas`) and never through the latency policy, which would otherwise
            // hand the client the known-wrong tensor of the control with a `200` on top. Excluding it
            // here also covers the cached-topology fallback below, which honours `tried_peers` too, and
            // it is safe because `pick_audit_layer` only marks a step with **two owners that are not
            // canaries** (`R4`): there is always an honest node for this hop.
            if let Some(order) = audit.as_ref() {
                if order.marks(current_layer_start) {
                    if let Some(canary) = order.canary.as_ref() {
                        tried_peers.insert(canary.clone());
                    }
                }
            }

            loop {
                let mut target_peer_id = None;
                let mut target_http_addr = None;
                let mut target_public_ip = None;
                let mut target_public_api_port = None;
                // `TASK-70`/S77: the range owned by the peer this hop is about to call. It is the step
                // the audit compares (`AuditOrder::marks`), and it is read from the topology — never
                // from the payload — so a hop cannot claim a range it does not serve.
                let mut target_layer_range: Option<(usize, usize)> = None;

                // 2. Try pinned route first (`TASK-81`/S88).
                // If this session previously prefilled this layer stretch on a peer that is still
                // alive and eligible (not tried/failed in this step, not suppressed, and still covering
                // this layer), prefer that peer. This prevents twin replicas without KV cache from
                // being picked mid-generation on table refresh, latency jitter or exploration share.
                let pinned_peer = if let Some(ref s_key) = session_key {
                    self.session_pins
                        .get_pinned_peer(s_key, current_layer_start)
                } else {
                    None
                };

                let mut target_from_pin = None;
                if let Some(ref pin_id) = pinned_peer {
                    if !tried_peers.contains(pin_id) {
                        let table = self.routing_table.read().await;
                        if !table.is_suppressed(pin_id) {
                            if let Some(peer) = table
                                .get_peers_for_layer(current_layer_start)
                                .into_iter()
                                .find(|p| {
                                    p.peer_id == *pin_id && p.expert_range.is_none() && !p.canary
                                })
                            {
                                debug!(
                                    "📌 Using pinned peer {pin_id} for layer {current_layer_start} of session {}",
                                    session_key.as_deref().unwrap_or("?")
                                );
                                target_from_pin = Some(peer);
                            }
                        }
                    }
                }

                if let Some(peer) = target_from_pin {
                    target_peer_id = Some(peer.peer_id.clone());
                    target_http_addr = Some(peer.http_addr.clone());
                    target_public_ip = peer.public_ip.clone();
                    target_public_api_port = peer.public_api_port;
                    target_layer_range = Some(step_range_of(&peer, current_layer_start)?);
                } else {
                    // Try local cache
                    let routing_target = {
                        let table = self.routing_table.read().await;
                        table.find_target_for_layer_excluding(
                            current_layer_start,
                            &tried_peers,
                            exploration_draw,
                        )
                    };

                    // `S43`/`TASK-37`: the expert-parallel scatter-gather used to live here. It sent the
                    // same hidden state to every peer that announced an `expert_range` and **summed** the
                    // answers, but each of those peers executes the whole layer (attention, residual and
                    // shared expert included), so the gather produced `N·(h + attn + shared) + Σ moe_i`
                    // instead of `h + attn + shared + Σ moe_i`: wrong logits, with no warning. Nothing in
                    // the payload asked a hop for only its experts' contribution, and no assignment
                    // guarantees that the union of the announced ranges covers every expert the gate may
                    // pick. Expert parallelism is not implemented, so a layer whose only candidates are
                    // such shards **cannot** be served: failing now, naming them, is the whole point —
                    // the router used to wait 30 s for a peer that can never answer correctly.
                    if let Some(peer) = routing_target.selected {
                        target_peer_id = Some(peer.peer_id.clone());
                        target_http_addr = Some(peer.http_addr.clone());
                        target_public_ip = peer.public_ip.clone();
                        target_public_api_port = peer.public_api_port;
                        // `TASK-70`/S77: the range of the peer that covers this step *is* the step this
                        // hop serves (`TASK-96`/S102: for a peer with two ranges, the one that covers
                        // it), and it is what identifies the work the audit compares.
                        target_layer_range = Some(step_range_of(&peer, current_layer_start)?);
                    } else if !routing_target.unsupported_expert_shards.is_empty() {
                        error!(
                            "❌ Layer {} of '{}' can only be served by expert-parallel shards ({:?}), and \
                             expert parallelism is not implemented (`TASK-37`/S43): the gather would sum \
                             whole layers. Refusing instead of serving wrong logits.",
                            current_layer_start, model_id, routing_target.unsupported_expert_shards
                        );
                        return Err(DomainError::Validation(format!(
                            "layer {} of '{}' can only be served by expert-parallel shards ({:?}) and \
                             expert parallelism is not implemented (`TASK-37`/S43): the assignment must \
                             give whole layers to the nodes",
                            current_layer_start, model_id, routing_target.unsupported_expert_shards
                        )));
                    }

                    if !routing_target.unsupported_expert_shards.is_empty()
                        && target_peer_id.is_some()
                    {
                        // A usable peer exists, so the layer is served; but a peer that announces an
                        // expert range is a misconfiguration worth saying out loud. Once per `route()`
                        // call, so a long generation does not turn the log into noise.
                        if !warned_about_expert_shards {
                            warned_about_expert_shards = true;
                            warn!(
                                "⚠️ Ignoring expert-parallel shards for layer {} of '{}': {:?} announce an \
                                 expert_range and cannot serve a whole layer (`TASK-37`/S43). Routing to \
                                 the whole-layer peers instead.",
                                current_layer_start,
                                model_id,
                                routing_target.unsupported_expert_shards
                            );
                        }
                    }
                }

                // 3. Fallback to the **cached topology** (`TASK-36`/S42).
                //
                // This branch used to be an HTTP call to `GET /substitute?…`, an endpoint that
                // `create_router` never served: it answered 404, the `_ =>` arm below swallowed the
                // status, and the whole thing was reported as «musivum unreachable, routing from
                // cache» — a lie about a perfectly healthy orchestrator that also hid the fact that
                // the only half which ever worked was the cached lookup. The dead call is gone and
                // the lookup is the branch itself now, with a log that says what really happened.
                if target_peer_id.is_none() {
                    let cache = self.topology_cache.read().await;
                    let cache_age = cache.age_secs();
                    let cached = cache.get().and_then(|peers| {
                        peers.into_iter().find(|peer| {
                            tessella_protocol::layer_ranges::covers(
                                &peer.layer_ranges,
                                current_layer_start,
                            ) && !tried_peers.contains(&peer.peer_id)
                                && !peer.canary
                        })
                    });
                    drop(cache);

                    match cached {
                        Some(peer_info) => {
                            let resolved_addr = resolve_preferred_http_addr(
                                &peer_info.http_addr,
                                &peer_info.p2p_addr,
                            );
                            debug!(
                                "🧭 No candidate in the routing table for layer {} of '{}': \
                                     using the cached topology (age: {}s) → {}",
                                current_layer_start, model_id, cache_age, peer_info.peer_id
                            );
                            target_layer_range =
                                Some(step_range_of(&peer_info, current_layer_start)?);
                            target_peer_id = Some(peer_info.peer_id.clone());
                            target_http_addr = Some(resolved_addr.clone());
                            target_public_ip = peer_info.public_ip.clone();
                            target_public_api_port = peer_info.public_api_port;
                            // Keep it in the routing table: the cached view is one observation
                            // behind, so the next layer should not pay the lookup again.
                            self.routing_table.write().await.insert_peer(
                                tessella_core::models::LayerPeer {
                                    peer_id: peer_info.peer_id.clone(),
                                    http_addr: resolved_addr,
                                    p2p_addr: peer_info.p2p_addr.clone(),
                                    layer_ranges: peer_info.layer_ranges.clone(),
                                    expert_range: peer_info.expert_range,
                                    queue_depth: peer_info.queue_depth,
                                    public_ip: peer_info.public_ip.clone(),
                                    public_api_port: peer_info.public_api_port,
                                    canary: peer_info.canary,
                                },
                            );
                        }
                        None => {
                            // Nothing local and nothing cached. `TASK-36` removed the last
                            // fallback (an endpoint that never existed), so the only recovery
                            // left is a topology update: the `warn!` + suspend below is the
                            // whole - and now explicit - path.
                            debug!(
                                "🧭 No cached candidate for layer {} of '{}' either \
                                     (cache age: {}s)",
                                current_layer_start, model_id, cache_age
                            );
                        }
                    }
                }

                let target_peer = match target_peer_id {
                    Some(id) => id,
                    None => {
                        warn!(
                            "⏳ No peers available for layer {}. Suspending task in memory...",
                            current_layer_start
                        );
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(30),
                            self.notify.notified(),
                        )
                        .await
                        {
                            Ok(_) => {
                                info!(
                                    "🔔 Resume signaled. Rechecking routing table for layer {}...",
                                    current_layer_start
                                );
                                tried_peers.clear();
                                continue;
                            }
                            Err(_) => {
                                error!(
                                    "❌ Global suspend timeout: waited > 30s for layer {} topology update. Dropping tensor.",
                                    current_layer_start
                                );
                                return Err(DomainError::Network(format!(
                                    "Pipeline timeout: no peer found for layer {} within 30 seconds",
                                    current_layer_start
                                )));
                            }
                        }
                    }
                };

                tried_peers.insert(target_peer.clone());

                // Prefer the resolved HTTP address (carries `public_api_url` when set);
                // only fall back to the legacy ip:port pair when it is empty.
                let resolved_addr = target_http_addr
                    .as_deref()
                    .map(str::trim)
                    .filter(|a| !a.is_empty());
                let target_url = if let Some(addr) = resolved_addr {
                    format!("{}/v1/forward", addr.trim_end_matches('/'))
                } else if let (Some(ref pub_ip), Some(pub_port)) =
                    (&target_public_ip, target_public_api_port)
                {
                    format!("http://{}:{}/v1/forward", pub_ip, pub_port)
                } else {
                    return Err(DomainError::Network(
                        "No reachable http_addr for next worker".into(),
                    ));
                };

                if tried_peers.len() <= 1 {
                    debug!("→ layer {} → {}", current_layer_start, target_url);
                } else {
                    warn!(
                        "→ Retrying layer {} → {} (tried: {})",
                        current_layer_start,
                        target_url,
                        tried_peers.len()
                    );
                }

                let start_time = std::time::Instant::now();
                // `TASK-76`/S83: the request carries **no** timeout of its own. The wait is the one
                // declared by the operator (`p2p_inference_timeout_secs`, `self.hop_timeout`) and it is
                // applied once, below: the `.timeout(Duration::from_secs(30))` that used to be here was
                // unreachable — the 5 s literal wrapped it —, and with two live timers the reason that
                // gets counted (`timeout`, which replaces the candidate, or `network_error`) would be a
                // race instead of a decision.
                let mut req = self
                    .http_client
                    .post(&target_url)
                    .header("Content-Type", "application/octet-stream")
                    // `TASK-84`/S91 · **(b′)**, the **honest lane**. Declaring `Expect: 100-continue`
                    // tells the receiver that this body is worth a round trip before it is pushed, so
                    // a node that would refuse it — a hop at its quota, a declared length over the
                    // limit — can answer first and the 70-80 MiB never travel.
                    //
                    // It is an **optimisation and not a control**, and the distinction is the whole
                    // lesson of `S91` (b): the header is optional by construction. This very client
                    // (`reqwest` 0.12 over hyper 1.x) does **not** wait for the `100` and writes the
                    // body anyway — hyper's client role sets `expect_continue = false` —, and a
                    // hostile sender simply omits it. That is why the RAM of a node is protected by
                    // its quota of bodies and not by this: whoever wants to occupy a node's memory can
                    // keep doing it, and now they occupy eight bodies instead of a hundred.
                    .header("Expect", "100-continue")
                    .body(current_payload_bytes.clone());

                if tried_peers.len() > 1 {
                    req = req.header("X-Retry", "1");
                }

                let send_future = req.send();
                let send_res = tokio::time::timeout(self.hop_timeout, send_future).await;

                let resp = match send_res {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => {
                        error!("HTTP request to {} failed: {}", target_url, e);
                        self.record_hop_failure(&target_peer, PeerFailureReason::NetworkError)
                            .await;
                        // Immediately retry with a different candidate
                        continue;
                    }
                    Err(_) => {
                        error!(
                            "HTTP request to {} timed out after {}s (the declared wait, \
                             `p2p_inference_timeout_secs`): the hop is counted as `timeout` and another \
                             candidate is tried (TASK-76/S83)",
                            target_url,
                            self.hop_timeout.as_secs()
                        );
                        self.record_hop_failure(&target_peer, PeerFailureReason::Timeout)
                            .await;
                        // Immediately retry with a different candidate
                        continue;
                    }
                };

                let latency = start_time.elapsed();
                let status = resp.status();
                if !status.is_success() {
                    // `TASK-68`/S75 · decisión **(a)**, extended by `TASK-76`/S83 and `TASK-84`/S91: a
                    // **named refusal** is a terminal answer — the hop lost the prefix of a
                    // conversation, the prompt does not fit in one hop of this pipeline, the body is
                    // over the wire limit, or the hop is already reading its quota of bodies. Retrying
                    // them can only reproduce the refusal (with `missing_prefill` it used to produce
                    // plausible garbage with `200`, `S74`, `F-1`) or, for the quota, push the same
                    // 70-80 MiB body at every candidate in turn. The request **aborts** here, naming
                    // the layer and the hop, whatever the code is: `missing_prefill` (`409`),
                    // `prompt_too_long` (`400`), `payload_too_large` (`413`) or `too_many_hop_bodies`
                    // (`503`).
                    //
                    // Every other failure keeps the old behaviour (try a different candidate): a dead
                    // or overloaded node *is* worth replacing, and a `503` that names nothing keeps
                    // meaning exactly that.
                    let refusal = resp
                        .headers()
                        .get(tessella_core::error::REFUSAL_HEADER)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_owned);
                    let err_text = resp.text().await.unwrap_or_default();
                    error!(
                        "HTTP request to {} returned status: {} (latency={:?}) body: {}",
                        target_url, status, latency, err_text
                    );

                    if let Some(code) = refusal_of_hop(refusal.as_deref(), status, &err_text) {
                        // A refusal is a decision, not a fault of this hop: it is **not** counted as a
                        // failure. That counter is what marks a node degraded, and counting honest
                        // refusals is what turned the `S76` measurement — 2048 tokens answered with a
                        // `413` nobody had published — into a cascade of evictions that ended in a
                        // `500` after the 30 s suspension.
                        if code == tessella_core::error::MISSING_PREFILL_CODE {
                            if let Some(ref s_key) = session_key {
                                self.session_pins.remove_session(s_key);
                            }
                            return Err(missing_prefill_on_hop(
                                &target_peer,
                                current_layer_start,
                                &err_text,
                            ));
                        }
                        // `TASK-84`/S91: a full hop is the one refusal whose prose is not about the
                        // bytes, so it gets its own re-issue instead of inheriting the false one.
                        if code == tessella_core::error::TOO_MANY_HOP_BODIES_CODE {
                            return Err(busy_on_hop(
                                code,
                                &target_peer,
                                current_layer_start,
                                &err_text,
                            ));
                        }
                        return Err(refusal_on_hop(
                            code,
                            &target_peer,
                            current_layer_start,
                            &err_text,
                        ));
                    }

                    self.record_hop_failure(&target_peer, PeerFailureReason::HttpError)
                        .await;
                    // Immediately retry with a different candidate
                    continue;
                }

                // Succeeded: update peer latency
                self.routing_table
                    .write()
                    .await
                    .update_latency(&target_peer, latency);
                // `TASK-50`/S56: the peer answered a hop, so it is alive for this node again: the
                // consecutive counter of `peer_failures` goes back to zero (the cumulative total
                // does not — what already failed is history the operator may still want).
                self.peer_failures.record_success(&target_peer);

                // `TASK-81`/S88: pin this layer range to the peer that served it so future decode
                // steps stay on the same node holding the KV cache.
                if let Some(ref s_key) = session_key {
                    if let Some(range) = target_layer_range {
                        self.session_pins.pin_peer(s_key, range, &target_peer);
                    }
                }

                let resp_bytes = match resp.bytes().await {
                    Ok(b) => b.to_vec(),
                    Err(e) => {
                        error!("Failed to read response body from {}: {}", target_url, e);
                        // `TASK-50`/S56: this fourth case is why the metric is complete instead of
                        // "the three failures that used to call the warn". A hop whose body never
                        // arrives did not serve this node, so it is counted and penalised like any
                        // other: the penalty is a latency estimate that the next successful hop
                        // restores (`update_latency`), not an exclusion.
                        self.record_hop_failure(&target_peer, PeerFailureReason::NetworkError)
                            .await;
                        // Immediately retry with a different candidate
                        continue;
                    }
                };

                // Successful hop — debug only, not info (one per token is too verbose)
                debug!(
                    "✓ layer {} forwarded: status={}, bytes={}, latency={:.1}ms",
                    current_layer_start,
                    status,
                    resp_bytes.len(),
                    latency.as_secs_f64() * 1000.0
                );

                let payload = tessella_inference::HiddenStatePayload::from_bytes(&resp_bytes)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;

                // `TASK-70`/S77 · the audit compares the **marked step**: the one whose range starts at
                // the `audit_layer` the orchestrator signed into the ticket, and only when this hop is
                // the one that served it (`target_layer_range.0`). `audit` is `None` for every
                // unaudited session, which is the default of the product.
                //
                // It used to fire on `payload.layer_idx + 1 >= total_layers` — «the shard that ends the
                // pipeline» — and that condition had two measured problems (`S77`): the digest attributed
                // to a peer could be the digest of a whole *tail* (dead evidence: the report's range
                // matched no assignment, so `audit_credit` never ran) and, in a chain of three or more
                // nodes, the replica's own onward routing re-sent the session nonce, which H-09 refused
                // (`409 replay detected` + 30 s of suspension). The marked step is isolated, so neither
                // can happen.
                if let Some(order) = audit.as_ref() {
                    if let Some(step) = target_layer_range.filter(|range| order.marks(range.0)) {
                        self.audit_against_replicas(
                            step,
                            &current_payload_bytes,
                            &target_peer,
                            &resp_bytes,
                            order.replicas,
                            order.canary.as_deref(),
                        )
                        .await?;
                    }
                }

                current_layer_start = payload.layer_idx + 1;
                current_payload_bytes = resp_bytes;
                break;
            }
        }

        Ok(current_payload_bytes)
    }

    async fn refresh(&self) -> Result<(), DomainError> {
        let url = format!("{}/peers?model_id={}", self.orchestrator_url, self.model_id);
        let res = self.http_client.get(url).send().await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<Vec<tessella_core::models::PeersEntry>>().await {
                    Ok(entries) => {
                        let peers: Vec<tessella_core::models::LayerPeer> = entries
                            .iter()
                            .map(crate::peer_discovery::layer_peer_from_peers_entry)
                            .collect();
                        self.routing_table.write().await.replace_all(peers);
                        Ok(())
                    }
                    Err(e) => {
                        // `TASK-41`/S47: a change in the `/peers` contract must be visible here
                        // instead of silently leaving the routing table untouched.
                        warn!(
                            "⚠️ Could not parse the /peers answer of {}: {}",
                            self.orchestrator_url, e
                        );
                        Err(DomainError::Network(format!(
                            "Failed to parse peers JSON: {e}"
                        )))
                    }
                }
            }
            Ok(resp) if resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                debug!("⏳ pipeline not yet complete, will retry topology refresh");
                Ok(()) // Treat as transient success to avoid caller warning
            }
            Ok(resp) => Err(DomainError::Network(format!(
                "Musivum returned non-success status: {}",
                resp.status()
            ))),
            Err(e) => Err(DomainError::Network(format!("Network error: {}", e))),
        }
    }

    async fn audit_front(
        &self,
        step: (usize, usize),
        request_bytes: &[u8],
        primary_peer: &str,
        primary_response: &[u8],
        audit: &tessella_core::traits::AuditOrder,
    ) -> Result<(), DomainError> {
        self.audit_against_replicas(
            step,
            request_bytes,
            primary_peer,
            primary_response,
            audit.replicas,
            audit.canary.as_deref(),
        )
        .await
    }
}
