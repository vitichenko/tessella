use crate::error::DomainError;
use ed25519_dalek::Verifier;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Archive, RkyvDeserialize, RkyvSerialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ModelHealth {
    Incomplete,
    Healthy,
    Degraded,
}

/// One-shot authorisation for **one unit of work** of one session (H-12 · `TASK-07`).
///
/// H-12 was that `/work_done` and `/api/v1/work_done_batch` only verified the *worker's own*
/// signature, so `request_id`, `credits_delta` and `output_hash` were whatever the caller decided
/// to write and inventing unique `request_id`s was enough to mint Karma (the only brake,
/// `check_conflicting_proof`, punishes collisions, not fabrications).
///
/// The orchestrator now mints the `request_id` of a session, the entry node adopts it and every
/// hop attaches this ticket to its [`WorkProof`]; the orchestrator credits only a ticket it
/// issued itself, exactly once per worker.
///
/// The ticket deliberately carries **no `worker_id`**: the entry node does not know the final
/// route (the topology is refreshed per session) and `replica_group_id` exists precisely so that
/// a failing shard can be replaced by another node of its replica group. The identity that gets
/// paid is the one that signs the proof, and the orchestrator checks that it is a registered
/// worker with an assignment for `model_id`.
#[derive(
    Debug, Clone, PartialEq, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct WorkTicket {
    /// Identifier of the work, minted by the orchestrator (never by a worker).
    pub request_id: String,
    /// Model whose session this ticket authorises.
    pub model_id: String,
    /// Peer that asked for the ticket, i.e. the entry node of the session.
    pub entry_peer_id: String,
    pub issued_at: u64,
    /// Unix second after which the ticket is refused (work still in flight loses its credit).
    pub expires_at: u64,
    /// The **step** this sampled session asks the mesh to compare — the first layer of the range a
    /// hop owns — or `None` when the session is **not** sampled (`TASK-70` · S77 · decisión **D-31**).
    pub audit_layer: Option<u32>,
    /// How many **extra** executions of the marked shard the orchestrator wants when it audits this
    /// session (`TASK-53` · S59).
    pub audit_replicas: u8,
    /// Peer id of the operator's **canary** that has to show up as a dissident in this session's
    /// `AuditReport` (`TASK-71` · S78 · decisión **D-28**), or `None` when the mesh runs no canary.
    pub audit_canary: Option<String>,
    /// Credits the orchestrator pays the **auditor** for a valid report of this session (`TASK-71` ·
    /// S78 · decisión **D-28**: a fixed, small reward minted by the mesh, never charged to the client).
    pub audit_reward: f32,
    /// Ed25519 signature of the orchestrator over [`WorkTicket::to_canonical_bytes`].
    pub orchestrator_signature: Vec<u8>,
}

impl WorkTicket {
    /// Canonical bytes the orchestrator signs: the ticket **without** the signature itself, so
    /// the signature never covers itself (same trap `KarmaTicket` avoids).
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, DomainError> {
        let mut cloned = self.clone();
        cloned.orchestrator_signature = vec![];
        serde_json::to_vec(&cloned).map_err(|e| DomainError::Serialization(e.to_string()))
    }

    /// A ticket is valid **until** the second it expires, which is the semantics the workers
    /// already apply to `KarmaTicket::expiry`.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        now > self.expires_at
    }
}

/// What the orchestrator decided about the authorisation behind an incoming [`WorkProof`]
/// (`TASK-07`, H-12).
#[derive(Debug, Clone, PartialEq)]
pub enum WorkTicketOutcome {
    /// First time this worker consumes the ticket: credit `credits`.
    Accepted { credits: f32 },
    /// The same worker already consumed this ticket (sweeper retry): idempotent, no double credit.
    Duplicate { credits: f32 },
    /// No such ticket: it was never issued here (forged, or minted by another orchestrator).
    Unknown,
    /// The ticket expired before the proof arrived.
    Expired { expires_at: u64 },
    /// The ticket authorises a different model than the one of the proof.
    ModelMismatch { expected: String, got: String },
}

/// Ceiling of [`WorkTicket::audit_replicas`] (`TASK-53` · S59, re-scoped by `TASK-70`/S77).
pub const MAX_AUDIT_REPLICAS: u8 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct WorkProof {
    pub request_id: String,
    pub worker_id: String,
    pub credits_delta: f32,
    /// `sha256` of the **hidden state this hop emitted** (`Payload::data`, i.e. the tensor bytes,
    /// not the serialized envelope).
    pub output_hash: [u8; 32],
    pub timestamp: u64,
    /// `sha256` of the hidden-state payload this hop **received** (`TASK-07`, layer 2).
    pub input_hash: [u8; 32],
    /// Layers this hop computed, as `[start, end)`, taken from its own assignment.
    pub layer_range: (usize, usize),
    /// Authorisation this work was done under (H-12).
    pub work_ticket: WorkTicket,
}

impl WorkProof {
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, DomainError> {
        serde_json::to_vec(self).map_err(|e| DomainError::Serialization(e.to_string()))
    }
}

/// One execution of a sampled shard step, as its auditor saw it (`TASK-53` · S59).
#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct AuditExecution {
    /// Worker that ran the step (the primary the router chose or one of its replicas).
    pub peer_id: String,
    /// `sha256` of the tensor bytes **it answered with** (`Payload::data`).
    pub output_hash: [u8; 32],
}

/// What the auditor of a sampled session saw when it executed the marked shard more than once
/// (`TASK-53` · S59).
#[derive(
    Debug, Clone, PartialEq, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct AuditReport {
    /// The session's work id, as minted by the orchestrator (never by a worker).
    pub request_id: String,
    /// Layers the auditor executed more than once, as `[start, end)`.
    pub layer_range: (usize, usize),
    /// `sha256` of the tensor **every** execution received (`Payload::data` of the request).
    pub input_hash: [u8; 32],
    /// Every execution the auditor compared, in the order it ran them. **At least two**, and no peer twice.
    pub executions: Vec<AuditExecution>,
    /// The authorisation of the session, carried exactly like in [`WorkProof`].
    pub work_ticket: WorkTicket,
    /// Peer id of the auditor — the entry node — declared here and bound to the signature below.
    pub auditor_peer_id: String,
    pub timestamp: u64,
    /// Protobuf-encoded libp2p public key of the auditor (D-3).
    pub public_key: Vec<u8>,
    /// Ed25519 signature of the auditor over [`PeerSignedRequest::peer_canonical_bytes`] (D-3).
    pub signature: Vec<u8>,
}

impl AuditReport {
    /// The digest every execution should have answered with, or `None` when they did not all agree.
    #[must_use]
    pub fn agreed_output(&self) -> Option<[u8; 32]> {
        let mut executions = self.executions.iter();
        let first = executions.next()?.output_hash;
        if executions.all(|execution| execution.output_hash == first) {
            Some(first)
        } else {
            None
        }
    }
}

/// The digest the **majority** of the executions agreed on, if there is one (`TASK-53` · S59).
#[must_use]
pub fn majority_digest(executions: &[AuditExecution]) -> Option<[u8; 32]> {
    let mut groups: Vec<([u8; 32], usize)> = Vec::new();
    for execution in executions {
        match groups
            .iter_mut()
            .find(|(digest, _)| *digest == execution.output_hash)
        {
            Some((_, count)) => *count += 1,
            None => groups.push((execution.output_hash, 1)),
        }
    }

    let mut counts: Vec<usize> = groups.iter().map(|(_, count)| *count).collect();
    counts.sort_unstable_by(|left, right| right.cmp(left));
    let top = counts.first().copied().unwrap_or(0);
    let runner_up = counts.get(1).copied().unwrap_or(0);

    if top >= 2 && top > runner_up {
        return groups
            .iter()
            .find(|(_, count)| *count == top)
            .map(|(digest, _)| *digest);
    }
    None
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct SignedWorkProof {
    pub proof: WorkProof,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
}

impl SignedWorkProof {
    pub fn verify(&self) -> Result<(), DomainError> {
        let pubkey = ed25519_dalek::VerifyingKey::from_bytes(
            &self.public_key[4..]
                .try_into()
                .map_err(|_| DomainError::Validation("Invalid public key length".into()))?,
        )
        .map_err(|e| DomainError::Validation(e.to_string()))?;

        let sig = ed25519_dalek::Signature::from_slice(&self.signature)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        pubkey
            .verify(&self.proof.to_canonical_bytes()?, &sig)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work_ticket() -> WorkTicket {
        WorkTicket {
            request_id: "work-1".to_string(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            entry_peer_id: "12D3KooWEntry".to_string(),
            issued_at: 1_700_000_000,
            expires_at: 1_700_000_600,
            audit_layer: None,
            audit_replicas: 2,
            audit_canary: None,
            audit_reward: 0.0,
            orchestrator_signature: vec![7; 64],
        }
    }

    fn work_proof(input_hash: [u8; 32], ticket: WorkTicket) -> WorkProof {
        WorkProof {
            request_id: "req".to_string(),
            worker_id: "peer".to_string(),
            credits_delta: 8.0,
            output_hash: [1u8; 32],
            timestamp: 42,
            input_hash,
            layer_range: (4, 12),
            work_ticket: ticket,
        }
    }

    /// `TASK-57`/S63: the canonical of a `WorkProof` has **no optional fields** any more.
    #[test]
    fn work_proof_without_the_session_authorisation_is_not_a_valid_canonical() {
        let legacy = format!(
            r#"{{"request_id":"req","worker_id":"peer","credits_delta":1.0,"output_hash":{},"timestamp":42}}"#,
            serde_json::to_string(&[1u8; 32]).unwrap()
        );

        assert!(
            serde_json::from_str::<WorkProof>(&legacy).is_err(),
            "a proof without input_hash, layer_range and work_ticket must not deserialize"
        );
    }

    /// The canonical carries **every** evidence field, with no `skip_serializing_if`.
    #[test]
    fn the_work_proof_canonical_carries_every_evidence_field() {
        let proof = work_proof([3u8; 32], work_ticket());
        let json: serde_json::Value =
            serde_json::from_slice(&proof.to_canonical_bytes().unwrap()).unwrap();

        assert_eq!(json["input_hash"], serde_json::to_value([3u8; 32]).unwrap());
        assert_eq!(json["layer_range"], serde_json::json!([4, 12]));
        assert_eq!(json["work_ticket"]["request_id"], "work-1");
    }

    #[test]
    fn work_proof_with_the_work_ticket_of_task07_round_trips_through_json() {
        let proof = work_proof([3u8; 32], work_ticket());
        let bytes = proof.to_canonical_bytes().unwrap();

        let parsed: WorkProof = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.input_hash, [3u8; 32]);
        assert_eq!(parsed.work_ticket, work_ticket());
        assert_eq!(parsed.layer_range, (4, 12));

        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("\"work_ticket\""));
        assert!(text.contains("\"input_hash\""));
        assert!(text.contains("\"layer_range\""));
    }

    /// The signature cannot cover itself.
    #[test]
    fn work_ticket_canonical_bytes_exclude_the_signature() {
        let mut ticket = work_ticket();
        let without = ticket.to_canonical_bytes().unwrap();
        ticket.orchestrator_signature = vec![9; 64];

        assert_eq!(without, ticket.to_canonical_bytes().unwrap());
        let json: serde_json::Value = serde_json::from_slice(&without).unwrap();
        assert_eq!(json["request_id"], "work-1");
        assert_eq!(json["model_id"], "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(json["entry_peer_id"], "12D3KooWEntry");
        assert_eq!(json["issued_at"], 1_700_000_000u64);
        assert_eq!(json["expires_at"], 1_700_000_600u64);
        assert_eq!(json["orchestrator_signature"], serde_json::json!([]));
    }

    /// Valid *until* the expiry second is over.
    #[test]
    fn work_ticket_expiry_is_inclusive() {
        let ticket = work_ticket();
        assert!(!ticket.is_expired(1_700_000_000));
        assert!(!ticket.is_expired(1_700_000_600));
        assert!(ticket.is_expired(1_700_000_601));
    }

    /// `audit` and `audit_replicas` are mandatory fields of the canonical.
    #[test]
    fn the_ticket_carries_its_audit_flags_in_every_canonical() {
        let legacy = r#"{"request_id":"work-1","model_id":"Qwen/Qwen2.5-0.5B-Instruct","entry_peer_id":"12D3KooWEntry","issued_at":1700000000,"expires_at":1700000600,"orchestrator_signature":[]}"#;
        assert!(
            serde_json::from_str::<WorkTicket>(legacy).is_err(),
            "a canonical without `audit`/`audit_replicas` must not deserialize"
        );

        let canonical = work_ticket().to_canonical_bytes().unwrap();
        let json: serde_json::Value = serde_json::from_slice(&canonical).unwrap();
        assert_eq!(
            json["audit_layer"],
            serde_json::Value::Null,
            "the marked step is always serialized, and `null` means «not sampled»"
        );
        assert_eq!(json["audit_replicas"], 2, "and so is the cost of the audit");
        assert_eq!(json["request_id"], "work-1");
    }

    /// The marked step is part of the preimage.
    #[test]
    fn work_ticket_audit_is_inside_the_signed_preimage() {
        let plain = work_ticket();
        let mut audited = work_ticket();
        audited.audit_layer = Some(8);

        assert_ne!(
            plain.to_canonical_bytes().unwrap(),
            audited.to_canonical_bytes().unwrap(),
            "a marked step must change the signed bytes"
        );
        let json: serde_json::Value =
            serde_json::from_slice(&audited.to_canonical_bytes().unwrap()).unwrap();
        assert_eq!(json["audit_layer"], 8);

        let mut cleared = audited.clone();
        cleared.audit_layer = None;
        assert_ne!(
            audited.to_canonical_bytes().unwrap(),
            cleared.to_canonical_bytes().unwrap()
        );

        let mut moved = audited.clone();
        moved.audit_layer = Some(16);
        assert_ne!(
            audited.to_canonical_bytes().unwrap(),
            moved.to_canonical_bytes().unwrap()
        );
    }

    /// `layer_range` is part of every canonical.
    #[test]
    fn the_layer_range_is_part_of_every_canonical() {
        let proof = work_proof([0u8; 32], work_ticket());
        let bytes = proof.to_canonical_bytes().unwrap();
        let parsed: WorkProof = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.layer_range, (4, 12));
        assert!(std::str::from_utf8(&bytes).unwrap().contains("[4,12]"));

        let mut other_range = proof.clone();
        other_range.layer_range = (0, 4);
        assert_ne!(
            proof.to_canonical_bytes().unwrap(),
            other_range.to_canonical_bytes().unwrap()
        );
    }
}
