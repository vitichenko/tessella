//! Peer-signed HTTP requests (decision **D-3 = (a)** · `TASK-03` Step B).
//!
//! The destructive worker endpoints (`/report_failure`, `/deregister`, `/announce`, `/ready`)
//! used to be anonymous, so anybody could degrade a third party's reputation or mark a peer
//! offline (H-11 / H-14). D-3 authenticates them with the libp2p Ed25519 key every worker
//! **already** generates and persists (`worker/src/main.rs`, `network_manager.rs:346-352`), so
//! the identity of a request is `peer_id(public_key)` instead of a self-declared string.
//!
//! Every signed DTO carries the same two fields — `public_key` (protobuf-encoded, exactly what
//! `tessella_network::verify_signature` and `peer_id_from_public_key_bytes` expect: 36 bytes
//! / 72 hex for ed25519, *not* the 32 raw key bytes) and `signature` over
//! [`PeerSignedRequest::peer_canonical_bytes`] — and implements this trait, so both the signing
//! (`tessella_network::sign_peer_request`) and the verification
//! (`musivum_api::auth::require_peer_signature`) live in exactly one place each.
//!
//! The canonical preimage is the JSON serialization of the request **without** the two
//! authentication keys, which is exactly the request a pre-D-3 client would have sent (the
//! same "drop the signature, then serialize" idea used by `KarmaTicket`, `WeightManifest` and
//! `TransferReceipt`, implemented once for all five DTOs). A single signed preimage per request
//! is what makes "fill in a field after signing" impossible; the versioned preimage with a nonce
//! (`TASK-05`, S11) replaces this one for the inference payload, not for these control-plane
//! DTOs.

use crate::error::DomainError;

/// A request whose body is authenticated by the libp2p key of the peer that sends it.
pub trait PeerSignedRequest {
    /// Bytes covered by the signature: the semantic content of the request **without** the
    /// authentication fields, so a signer can compute it before filling them in and a verifier
    /// can recompute it after they travelled on the wire.
    ///
    /// # Errors
    /// Returns [`DomainError::Serialization`] when the request cannot be serialized.
    fn peer_canonical_bytes(&self) -> Result<Vec<u8>, DomainError>;

    /// Protobuf-encoded libp2p public key of the signer (empty when unsigned).
    fn peer_public_key(&self) -> &[u8];

    /// Raw signature bytes (empty when unsigned).
    fn peer_signature(&self) -> &[u8];

    /// Stores the authentication fields (used by the signer and by the canonicalizer).
    fn set_peer_auth(&mut self, public_key: Vec<u8>, signature: Vec<u8>);

    /// `true` when the request carries **both** authentication fields.
    #[must_use]
    fn has_peer_auth(&self) -> bool {
        !self.peer_public_key().is_empty() && !self.peer_signature().is_empty()
    }
}

macro_rules! impl_peer_signed_request {
    ($ty:ty) => {
        impl PeerSignedRequest for $ty {
            fn peer_canonical_bytes(&self) -> Result<Vec<u8>, DomainError> {
                // Value-based canonicalization: serialize the request (whatever order the
                // fields happen to have in this struct definition) and drop the two
                // authentication keys, so the preimage is exactly the request a pre-D-3
                // client would have sent. Both signer and verifier run *this* function, so
                // the definition of the preimage lives in a single place.
                let mut value = serde_json::to_value(self)
                    .map_err(|e| DomainError::Serialization(e.to_string()))?;
                if let Some(map) = value.as_object_mut() {
                    map.remove("public_key");
                    map.remove("signature");
                }
                serde_json::to_vec(&value).map_err(|e| DomainError::Serialization(e.to_string()))
            }

            fn peer_public_key(&self) -> &[u8] {
                &self.public_key
            }

            fn peer_signature(&self) -> &[u8] {
                &self.signature
            }

            fn set_peer_auth(&mut self, public_key: Vec<u8>, signature: Vec<u8>) {
                self.public_key = public_key;
                self.signature = signature;
            }
        }
    };
}

impl_peer_signed_request!(crate::network::AnnounceRequest);
impl_peer_signed_request!(crate::network::WorkerReady);
impl_peer_signed_request!(crate::network::FailureReport);
impl_peer_signed_request!(crate::network::WeightPoisoningReport);
impl_peer_signed_request!(crate::models::DeregisterRequest);
// `TASK-49`/S55: `/register` was the last anonymous endpoint that writes state (it overwrites the
// assignment of the peer id it claims and fills the swarm's admission cache), so the admission
// policy of a `private` deployment needs its identity to be proven the same way.
impl_peer_signed_request!(crate::network::WorkerRegistration);
// `TASK-53`/S59: the auditor's report. It is not an accusation but a *fact* (the two or three
// digests it compared for one input), and it is signed by the node that may report it — the entry
// node of the session, which the ticket inside the report names.
impl_peer_signed_request!(crate::models::AuditReport);

#[cfg(test)]
mod tests {
    use super::*;

    fn failure_report() -> crate::network::FailureReport {
        crate::network::FailureReport {
            peer_id: "12D3KooWReporter".to_string(),
            request_id: "req-1".to_string(),
            error: "timeout".to_string(),
            timestamp: 1234,
            public_key: vec![1, 2, 3],
            signature: vec![9, 9],
        }
    }

    /// The preimage must not include the authentication fields: otherwise a signer could never
    /// produce it (the signature does not exist yet) and a verifier could not recompute it.
    #[test]
    fn test_canonical_bytes_ignore_the_authentication_fields() {
        let signed = failure_report();
        let mut unsigned = signed.clone();
        unsigned.set_peer_auth(Vec::new(), Vec::new());

        let canonical_signed = signed.peer_canonical_bytes().unwrap();
        let canonical_unsigned = unsigned.peer_canonical_bytes().unwrap();

        assert_eq!(
            canonical_signed, canonical_unsigned,
            "public_key/signature must not be part of the signed preimage"
        );
        let as_text = String::from_utf8(canonical_signed).unwrap();
        assert!(
            !as_text.contains("\"signature\""),
            "the preimage must carry no signature field: {as_text}"
        );
        assert!(as_text.contains("\"request_id\":\"req-1\""), "{as_text}");
    }

    /// Any other field must change the preimage, or a signature could be replayed onto a
    /// modified request.
    #[test]
    fn test_canonical_bytes_change_with_the_payload() {
        let report = failure_report();
        let mut tampered = report.clone();
        tampered.error = "something else".to_string();

        assert_ne!(
            report.peer_canonical_bytes().unwrap(),
            tampered.peer_canonical_bytes().unwrap()
        );
    }

    #[test]
    fn test_has_peer_auth_requires_both_fields() {
        let mut req = crate::network::WorkerReady {
            peer_id: "12D3KooWPeer".to_string(),
            model_repo: "model".to_string(),
            layer_ranges: vec![(0, 10)],
            replica_group_id: 0,
            expert_range: None,
            queue_depth: 0,
            public_key: Vec::new(),
            signature: Vec::new(),
        };
        assert!(!req.has_peer_auth());

        req.set_peer_auth(vec![7; 36], Vec::new());
        assert!(!req.has_peer_auth(), "a missing signature is not signed");

        req.set_peer_auth(Vec::new(), vec![8; 64]);
        assert!(!req.has_peer_auth(), "a missing public key is not signed");

        req.set_peer_auth(vec![7; 36], vec![8; 64]);
        assert!(req.has_peer_auth());
    }

    /// `DeregisterRequest` is also rkyv-archived, so its new fields must not break the schema.
    #[test]
    fn test_deregister_request_canonical_bytes() {
        let req = crate::models::DeregisterRequest {
            peer_id: "12D3KooWPeer".to_string(),
            reason: "shutdown".to_string(),
            public_key: vec![1],
            signature: vec![2],
        };
        let canonical = String::from_utf8(req.peer_canonical_bytes().unwrap()).unwrap();
        assert!(canonical.contains("\"reason\":\"shutdown\""), "{canonical}");
        assert!(!canonical.contains("\"signature\""), "{canonical}");
    }

    /// `TASK-49`/S55: a registration is signed too, and the preimage covers **everything that
    /// decides the assignment** (the claim, the model, the manifest, the address), so a signature
    /// cannot be lifted onto a different offer.
    #[test]
    fn test_worker_registration_canonical_bytes() {
        fn registration() -> crate::network::WorkerRegistration {
            crate::network::WorkerRegistration {
                peer_id: "12D3KooWWorker".to_string(),
                p2p_address: "/ip4/1.1.1.1/tcp/9000".to_string(),
                relay_address: None,
                api_addr: "http://1.1.1.1:9001".to_string(),
                vram_mb: 4096,
                device: crate::types::ComputeDevice::Cuda { device_id: 0 },
                model_id: "test-model".to_string(),
                total_layers: 24,
                force_full_model: None,
                model_config: None,
                manifest: crate::ModelManifest {
                    model_id: "test-model".to_string(),
                    num_hidden_layers: 24,
                    num_experts: None,
                },
                parameter_contract_hash: "abc".to_string(),
                public_ip: None,
                public_api_port: None,
                public_key: vec![7; 36],
                signature: vec![8; 64],
            }
        }

        let signed = registration();
        assert!(signed.has_peer_auth());

        let mut unsigned = signed.clone();
        unsigned.set_peer_auth(Vec::new(), Vec::new());
        assert_eq!(
            signed.peer_canonical_bytes().unwrap(),
            unsigned.peer_canonical_bytes().unwrap(),
            "the authentication fields are not part of the preimage"
        );

        let mut richer = registration();
        richer.vram_mb = 8192;
        assert_ne!(
            signed.peer_canonical_bytes().unwrap(),
            richer.peer_canonical_bytes().unwrap(),
            "the claim decides the assignment, so it must be signed"
        );

        let mut elsewhere = registration();
        elsewhere.api_addr = "http://2.2.2.2:9001".to_string();
        assert_ne!(
            signed.peer_canonical_bytes().unwrap(),
            elsewhere.peer_canonical_bytes().unwrap(),
            "the address the worker asks to be reached at is signed"
        );
    }

    /// The two authentication fields are part of the wire contract, not an option
    /// (`TASK-57`/S63 closed the one-release window of D-3).
    ///
    /// A body without them is not a request this version can decide anything about: it fails in the
    /// **parser** (`422` at the HTTP edge) instead of reaching a handler that answers `401`. That is
    /// the difference between «the policy rejects you» and «this is not a body of this protocol»,
    /// and it is what makes the retry story unambiguous for a worker: what it must fix is its
    /// payload, not a configuration key.
    #[test]
    fn test_an_announce_request_without_auth_is_not_a_valid_body() {
        let legacy = serde_json::json!({
            "peer_id": "12D3KooWPeer",
            "host": "1.2.3.4",
            "port": 9000,
            "vram_mb": 4096,
            "model_id": "model",
            "current_role": {"type": "Unassigned"},
            "p2p_address": "/ip4/1.2.3.4/tcp/9000",
            "relay_address": null,
            "http_api_url": null,
            "status": null,
            "queue_depth": 0,
            "parameter_contract_hash": "abc"
        });
        assert!(
            serde_json::from_value::<crate::network::AnnounceRequest>(legacy).is_err(),
            "an unsigned body must not deserialize once the migration window is closed"
        );
    }
    /// `TASK-53`/S59: the auditor's report is signed like the other control-plane DTOs and its
    /// preimage covers **exactly what the orchestrator will judge** — the session, the range, the
    /// input and every execution. A digest edited on the way (or a session swapped for another one)
    /// cannot pass verification, so the credit of the marked range cannot be redirected.
    #[test]
    fn test_audit_report_canonical_bytes() {
        fn report() -> crate::models::AuditReport {
            crate::models::AuditReport {
                request_id: "work-1".to_string(),
                layer_range: (12, 24),
                input_hash: [3; 32],
                executions: vec![
                    crate::models::AuditExecution {
                        peer_id: "12D3KooWPrimary".to_string(),
                        output_hash: [7; 32],
                    },
                    crate::models::AuditExecution {
                        peer_id: "12D3KooWReplica".to_string(),
                        output_hash: [7; 32],
                    },
                ],
                work_ticket: crate::models::WorkTicket {
                    request_id: "work-1".to_string(),
                    model_id: "test-model".to_string(),
                    entry_peer_id: "12D3KooWAuditor".to_string(),
                    issued_at: 1,
                    expires_at: u64::MAX,
                    audit_layer: Some(8),
                    audit_replicas: 2,
                    audit_canary: None,
                    audit_reward: 0.0,
                    orchestrator_signature: vec![5; 64],
                },
                auditor_peer_id: "12D3KooWAuditor".to_string(),
                timestamp: 1_700_000_000,
                public_key: vec![7; 36],
                signature: vec![8; 64],
            }
        }

        let signed = report();
        assert!(signed.has_peer_auth());

        let mut unsigned = signed.clone();
        unsigned.set_peer_auth(Vec::new(), Vec::new());
        assert_eq!(
            signed.peer_canonical_bytes().unwrap(),
            unsigned.peer_canonical_bytes().unwrap(),
            "the authentication fields are not part of the preimage"
        );

        let mut tampered = report();
        tampered.executions[1].output_hash = [42; 32];
        assert_ne!(
            signed.peer_canonical_bytes().unwrap(),
            tampered.peer_canonical_bytes().unwrap(),
            "the digests the orchestrator judges are signed"
        );

        let mut other_range = report();
        other_range.layer_range = (0, 12);
        assert_ne!(
            signed.peer_canonical_bytes().unwrap(),
            other_range.peer_canonical_bytes().unwrap(),
            "the range decides which claims the report governs"
        );

        let mut other_ticket = report();
        other_ticket.work_ticket.request_id = "work-2".to_string();
        assert_ne!(
            signed.peer_canonical_bytes().unwrap(),
            other_ticket.peer_canonical_bytes().unwrap(),
            "the authorisation of the session is part of the signed evidence"
        );
    }

    /// `TASK-57`/S63: the replica count is part of the canonical, so a ticket minted before S59 — the
    /// one that has no `audit_replicas` — is not a canonical this version accepts. The window S59
    /// opened existed so that a ticket in flight kept meaning «primary plus one replica»; with no
    /// fleet to migrate there was nothing left to keep, and a count that is **read** is better than
    /// one that is assumed.
    #[test]
    fn test_a_ticket_without_the_replica_count_is_not_a_valid_canonical() {
        let legacy = serde_json::json!({
            "request_id": "work-legacy",
            "model_id": "test-model",
            "entry_peer_id": "12D3KooWEntry",
            "issued_at": 1,
            "expires_at": u64::MAX,
            "audit": true,
            "orchestrator_signature": []
        });
        assert!(
            serde_json::from_value::<crate::models::WorkTicket>(legacy).is_err(),
            "a canonical without the replica count must not deserialize"
        );
    }
}
