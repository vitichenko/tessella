use ed25519_dalek::Verifier;
use sha2::Digest;
use std::sync::Arc;
use tessella_core::error::DomainError;
use tessella_core::models::{KarmaTicket, QueuePriority, SignedWorkProof, WorkProof, WorkTicket};
use tessella_core::traits::AuditOrder;
use tessella_inference::backends::{InferenceBackend, InferenceSession};
use tracing::{debug, error, info, warn};

use crate::canary::CanarySession;
use crate::config::CanaryMode;

/// Local Unix seconds.
pub fn unix_now() -> u64 {
    tessella_core::time::unix_now_secs()
}

/// Resolves the queue priority a payload earns from its `KarmaTicket` (H-13).
pub fn resolve_ticket_priority(
    ticket: &KarmaTicket,
    requester_pubkey: &[u8],
    request_id: &str,
    now: u64,
    orchestrator_public_key: Option<&ed25519_dalek::VerifyingKey>,
) -> QueuePriority {
    if ticket.expiry <= now {
        warn!("🎟️ EXPIRED KarmaTicket for req {}", request_id);
        return QueuePriority::Leech;
    }

    let Some(musivum_pubkey) = orchestrator_public_key else {
        debug!(
            "🎟️ No orchestrator public key configured: ignoring the KarmaTicket of req {} (orchestrator_public_key is unset)",
            request_id
        );
        return QueuePriority::Leech;
    };

    let Ok(canonical) = ticket.to_canonical_bytes() else {
        warn!("🎟️ Unserializable KarmaTicket for req {}", request_id);
        return QueuePriority::Leech;
    };
    let Ok(sig) = ed25519_dalek::Signature::from_slice(&ticket.orchestrator_signature) else {
        warn!("🎟️ INVALID KarmaTicket signature for req {}", request_id);
        return QueuePriority::Leech;
    };
    if musivum_pubkey.verify(&canonical, &sig).is_err() {
        warn!("🎟️ INVALID KarmaTicket signature for req {}", request_id);
        return QueuePriority::Leech;
    }

    if ticket.client_pubkey.is_empty() {
        warn!(
            "🎟️ Unbound KarmaTicket (no client_pubkey) for req {} ⇒ Leech",
            request_id
        );
        return QueuePriority::Leech;
    }
    if ticket.client_pubkey != requester_pubkey {
        warn!(
            "🎟️ KarmaTicket bound to another peer (ticket={} bytes, requester={} bytes) for req {} ⇒ Leech",
            ticket.client_pubkey.len(),
            requester_pubkey.len(),
            request_id
        );
        return QueuePriority::Leech;
    }

    debug!("🎟️ Valid KarmaTicket found: tier={:?}", ticket.tier);
    ticket.tier
}

/// Extracts the audit order from a work ticket, if present.
pub fn audit_order_of(ticket: Option<&WorkTicket>) -> Option<AuditOrder> {
    let ticket = ticket?;
    AuditOrder::of(
        ticket.audit_layer,
        ticket.audit_replicas,
        ticket.audit_canary.as_deref(),
    )
}

/// Checks whether the step starting at `layer_start` is marked for isolated execution
/// by the compute audit.
pub fn is_marked_step(audit_order: Option<&AuditOrder>, layer_start: usize) -> bool {
    audit_order
        .as_ref()
        .is_some_and(|order| order.marks(layer_start))
}

/// The session this node uses for a step whose key is not in sessions yet.
///
/// An ordinary node asks its backend for a shard session. A canary hands a CanarySession instead.
pub fn new_session_for_step(
    canary: CanaryMode,
    backend: Option<&Arc<dyn InferenceBackend>>,
) -> Result<Box<dyn InferenceSession>, DomainError> {
    if canary.is_on() {
        return Ok(Box::new(CanarySession::new(canary)));
    }
    match backend {
        Some(backend) => backend.begin_session(),
        None => Err(DomainError::Inference("Backend not loaded".into())),
    }
}

/// Synthesizes one cryptographic [`WorkProof`] for credit verification.
pub fn build_hop_work_proof(
    request_id: &str,
    worker_id: String,
    layer_range: (usize, usize),
    input: Option<&[u8]>,
    output: &[u8],
    work_ticket: WorkTicket,
    timestamp: u64,
) -> WorkProof {
    let digest = |bytes: &[u8]| -> [u8; 32] {
        let mut hasher = sha2::Sha256::new();
        hasher.update(bytes);
        hasher.finalize().into()
    };

    WorkProof {
        request_id: request_id.to_string(),
        worker_id,
        credits_delta: layer_range.1.saturating_sub(layer_range.0) as f32,
        output_hash: digest(output),
        timestamp,
        input_hash: input.map(digest).unwrap_or([0u8; 32]),
        layer_range,
        work_ticket,
    }
}

/// Signs one [`WorkProof`] with the local key and stores it in the WAL, best-effort.
pub async fn sign_and_store_work_proof(
    storage: &tessella_storage::SqliteStorage,
    keypair: &tessella_network::NetworkKeypair,
    request_id: &str,
    layer_range: (usize, usize),
    input: Option<&[u8]>,
    output: &[u8],
    work_ticket: &WorkTicket,
) {
    let proof = build_hop_work_proof(
        request_id,
        tessella_network::peer_id_from_keypair(keypair),
        layer_range,
        input,
        output,
        work_ticket.clone(),
        unix_now(),
    );

    let canonical = match proof.to_canonical_bytes() {
        Ok(canonical) => canonical,
        Err(e) => {
            error!("❌ Could not serialize the work proof of req {request_id}: {e}");
            return;
        }
    };
    let signature = match keypair.sign(&canonical) {
        Ok(signature) => signature,
        Err(e) => {
            error!("❌ Could not sign the work proof of req {request_id}: {e}");
            return;
        }
    };
    let signed = SignedWorkProof {
        proof,
        signature,
        public_key: keypair.public().encode_protobuf(),
    };

    match storage.save_pending_work_proof(&signed).await {
        Ok(_) => info!(
            "💾 Saved pending work proof to WAL (request_id={request_id}, layers=[{}, {}))",
            layer_range.0, layer_range.1
        ),
        Err(e) => error!("❌ Failed to save pending work proof to WAL: {e}"),
    }
}
