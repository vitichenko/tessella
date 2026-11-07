use crate::ApiState;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};

/// `GET /status` — liveness, the transport ceiling of this node (`TASK-47`/S53), what it has seen
/// fail towards its peers (`TASK-50`/S56) and whether it is serving at all (`TASK-77`/S84).
///
/// The body still carries the literal `OK` every harness greps for (`scripts/e2e/swarm.sh` waits for
/// `curl …/status | grep -qi ok`), and adds the numbers a client needs to size a prompt before
/// sending it: the model width and how many tokens fit in **one hop**.
///
/// Since `TASK-76`/S83 that ceiling is the `min` of the two limits that bind the node — what the wire
/// can carry and what the node's own VRAM declares it can prefill — and it is published **with its
/// reason** (`max_input_tokens_source`: `wire`, `vram` or `wire_and_vram`). Without the reason the
/// number is not actionable: it tells the operator whether to send a shorter prompt or to shard the
/// model differently. `max_input_tokens` is `null` when neither limit is known yet (the model config
/// is not readable and nothing is declared), in which case the check is skipped, never turned into a
/// `0`-token ceiling.
///
/// Since `TASK-77`/S84 the body also carries `gpu`: the health of **this** node, which is a different
/// question from `peer_failures` (what this node has seen fail *towards others*). `gpu.verdict` is
/// `serving` or `down`, with `kind`/`context`/`reason`/`since_unix_secs` when it is down, and the
/// counters that tell the two failure kinds apart: `out_of_memory` counts requests that did not fit
/// (the node keeps serving) and `device_failures` counts device errors the probe did **not** confirm.
/// `revalidate_secs` is the bound of a self-recovery — the same value as the heartbeat — and
/// `last_recovery` says whether the node came back alone (`device_probe`) or because a shard reloaded.
/// A client that got a `503` from a node can read here whether it is worth retrying anywhere soon.
pub async fn handle_status(State(state): State<ApiState>) -> impl IntoResponse {
    let current_assignment = if let Some(dist) = &state.dist_inference {
        dist.current_assignment().await
    } else {
        None
    };
    let layer_ranges = current_assignment.as_ref().map(|a| a.layer_ranges.clone());

    Json(serde_json::json!({
        "status": "OK",
        "model": state.default_model,
        "layer_ranges": layer_ranges,
        "assignment": current_assignment,
        "hidden_size": state.model_config.as_ref().map(|cfg| cfg.hidden_size),
        "max_input_tokens": state.max_input_tokens,
        "max_input_tokens_source": state.max_input_tokens_source,
        // `TASK-76`/S83: the other half of the deal. The ceiling says *how much* this node accepts in
        // one hop; this says *until when* it waits for the hop that carries it. A client that reads
        // only the first one sizes a prompt the node will then declare dead.
        "hop_timeout_secs": state.hop_timeout_secs,
        // `TASK-84`/S91: el tercer número del mismo trato, y el único que no se declara sino que se
        // **mide**. El techo dice *cuánto* acepta este nodo en un hop y la espera dice *hasta cuándo*
        // aguanta el hop que lo trae; esto dice *cuántos cuerpos a la vez* lee, y el contador vivo es lo
        // que desde fuera distingue «vacío» de «al tope»: con `hop_bodies_in_flight ==
        // max_concurrent_hop_bodies` el nodo está negando todo con un `503 too_many_hop_bodies`, que es
        // un nodo **lleno**, no uno roto. El deadline es el otro extremo del mismo control: corta al
        // emisor que deja de mandar a mitad de cuerpo.
        "max_concurrent_hop_bodies": crate::hop_body::hop_body_quota().capacity(),
        "hop_bodies_in_flight": crate::hop_body::hop_body_quota().in_flight(),
        "hop_body_idle_timeout_secs": crate::hop_body::hop_body_quota().idle().as_secs(),
        "peer_failures": state.peer_failures.as_ref().map(|registry| registry.report()),
        // `TASK-77`/S84: salud **de este nodo**. `peer_failures` es lo que este nodo vio fallar hacia
        // otros; esto es lo que le pasa a él. Se publica aunque esté sano —`verdict: "serving"` con los
        // contadores— porque el número que interesa (`out_of_memory`) crece sin que el nodo se caiga.
        "gpu": state.gpu_health.as_ref().map(|health| health.report()),
    }))
}

pub async fn handle_update_assignment(
    State(state): State<ApiState>,
    Json(req): Json<tessella_protocol::UpdateAssignmentRequest>,
) -> impl IntoResponse {
    let Some(dist_inference) = &state.dist_inference else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Inference service not available",
        )
            .into_response();
    };

    let current_version = state
        .current_assignment_version
        .load(std::sync::atomic::Ordering::SeqCst);
    if req.version < current_version {
        tracing::warn!(
            "⚠️ Stale assignment update rejected: incoming version {} < current version {}",
            req.version,
            current_version
        );
        let error_payload = tessella_protocol::StaleVersionError::new(current_version, req.version);
        return (StatusCode::CONFLICT, axum::Json(error_payload)).into_response();
    }

    tracing::info!(
        "📥 Received assignment update (v{}): layers {:?}, replica_group {}",
        req.version,
        req.assignment.layer_ranges,
        req.assignment.replica_group_id
    );

    let assignment = req.assignment;
    match dist_inference.update_assignment(assignment.clone()).await {
        Ok(_) => {
            state
                .current_assignment_version
                .fetch_max(req.version, std::sync::atomic::Ordering::SeqCst);
            if let Some(storage) = &state.storage
                && let Err(e) = storage
                    .set_active_assignment(req.version, &assignment)
                    .await
            {
                tracing::error!(
                    "Failed to persist active assignment (v{}): {}",
                    req.version,
                    e
                );
            }
            StatusCode::OK.into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tessella_core::error::DomainError;
    use tessella_core::models::InferenceMode;
    use tessella_core::traits::{DistributedInference, DistributedSession};
    use tessella_protocol::{UpdateAssignmentRequest, WorkerAssignment};
    use tokio::sync::RwLock;

    struct MockInferenceEngine {
        assignment: RwLock<Option<WorkerAssignment>>,
    }

    #[async_trait::async_trait]
    impl DistributedInference for MockInferenceEngine {
        async fn reset_distributed_cache(&self) {}
        async fn apply_prompt_template(&self, _mode: &InferenceMode, raw: &str) -> String {
            raw.to_string()
        }
        async fn begin_session(
            self: Arc<Self>,
            _hint: Option<String>,
        ) -> Result<(Box<dyn DistributedSession>, String), DomainError> {
            Err(DomainError::Inference("not implemented".into()))
        }
        async fn forward_hidden(&self, bytes: &[u8]) -> Result<Vec<u8>, DomainError> {
            Ok(bytes.to_vec())
        }
        async fn update_assignment(
            &self,
            new_assignment: WorkerAssignment,
        ) -> Result<(), DomainError> {
            *self.assignment.write().await = Some(new_assignment);
            Ok(())
        }
        async fn current_assignment(&self) -> Option<WorkerAssignment> {
            self.assignment.read().await.clone()
        }
    }

    fn test_api_state(engine: Option<Arc<dyn DistributedInference>>) -> ApiState {
        ApiState {
            dist_inference: engine,
            storage: None,
            credits: None,
            discovery: None,
            slashing: None,
            peer_slashing: None,
            musivum_state: None,
            max_new_tokens: 512,
            max_input_tokens: Some(2048),
            max_input_tokens_source: Some("wire"),
            hop_timeout_secs: 15,
            musivum_keypair: None,
            admin_token: None,
            default_model: "test-model".to_string(),
            model_config: None,
            peer_failures: None,
            gpu_health: None,
            tokenizer: None,
            token_stream_senders: None,
            orchestrator_url: None,
            own_api_url: None,
            replay_cache: Arc::new(tessella_protocol::signing::ReplayCache::new()),
            callback_capabilities: None,
            metrics: None,
            current_assignment_version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    #[tokio::test]
    async fn test_status_declares_layer_range_and_assignment() {
        let engine = Arc::new(MockInferenceEngine {
            assignment: RwLock::new(Some(WorkerAssignment {
                layer_ranges: vec![(0, 12)],
                model_repo: "test-model".to_string(),
                replica_group_id: 0,
                expert_range: None,
            })),
        });

        let state = test_api_state(Some(engine));
        let response = handle_status(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(json["status"], "OK");
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["layer_ranges"], serde_json::json!([[0, 12]]));
        assert_eq!(json["assignment"]["replica_group_id"], 0);
    }

    #[tokio::test]
    async fn test_update_assignment_reloads_and_updates_status() {
        let engine = Arc::new(MockInferenceEngine {
            assignment: RwLock::new(Some(WorkerAssignment {
                layer_ranges: vec![(0, 8)],
                model_repo: "test-model".to_string(),
                replica_group_id: 0,
                expert_range: None,
            })),
        });

        let state = test_api_state(Some(engine.clone()));

        // 1. Initial status has layers 0-8
        let status_resp = handle_status(State(state.clone())).await.into_response();
        let bytes = axum::body::to_bytes(status_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["layer_ranges"], serde_json::json!([[0, 8]]));

        // 2. Musivum sends assignment update to layers 8-16 with version 1
        let update_req = UpdateAssignmentRequest {
            assignment: WorkerAssignment {
                layer_ranges: vec![(8, 16)],
                model_repo: "test-model".to_string(),
                replica_group_id: 1,
                expert_range: None,
            },
            version: 1,
        };

        let update_resp = handle_update_assignment(State(state.clone()), Json(update_req))
            .await
            .into_response();
        assert_eq!(update_resp.status(), StatusCode::OK);

        // 3. Status now dynamically declares layers 8-16 and replica_group_id 1
        let updated_status_resp = handle_status(State(state)).await.into_response();
        let updated_bytes = axum::body::to_bytes(updated_status_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let updated_json: serde_json::Value = serde_json::from_slice(&updated_bytes).unwrap();
        assert_eq!(updated_json["layer_ranges"], serde_json::json!([[8, 16]]));
        assert_eq!(updated_json["assignment"]["replica_group_id"], 1);
    }

    #[tokio::test]
    async fn test_update_assignment_without_engine_returns_503() {
        let state = test_api_state(None);
        let update_req = UpdateAssignmentRequest {
            assignment: WorkerAssignment {
                layer_ranges: vec![(0, 4)],
                model_repo: "test-model".to_string(),
                replica_group_id: 0,
                expert_range: None,
            },
            version: 1,
        };

        let resp = handle_update_assignment(State(state), Json(update_req))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_update_assignment_rejects_stale_version_with_409() {
        let engine = Arc::new(MockInferenceEngine {
            assignment: RwLock::new(None),
        });
        let state = test_api_state(Some(engine.clone()));

        // Apply version 5 -> OK
        let req_v5 = UpdateAssignmentRequest {
            assignment: WorkerAssignment {
                layer_ranges: vec![(0, 8)],
                model_repo: "test-model".to_string(),
                replica_group_id: 0,
                expert_range: None,
            },
            version: 5,
        };
        let resp = handle_update_assignment(State(state.clone()), Json(req_v5))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            state
                .current_assignment_version
                .load(std::sync::atomic::Ordering::SeqCst),
            5
        );

        // Stale version 4 (< 5) -> 409 Conflict
        let req_v4 = UpdateAssignmentRequest {
            assignment: WorkerAssignment {
                layer_ranges: vec![(0, 4)],
                model_repo: "test-model".to_string(),
                replica_group_id: 0,
                expert_range: None,
            },
            version: 4,
        };
        let resp = handle_update_assignment(State(state.clone()), Json(req_v4))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let err: tessella_protocol::StaleVersionError =
            serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(err.error, "stale_version");
        assert_eq!(err.current_version, 5);
        assert_eq!(err.rejected_version, 4);

        // Same version 5 (retry/idempotent) -> OK
        let req_v5_retry = UpdateAssignmentRequest {
            assignment: WorkerAssignment {
                layer_ranges: vec![(0, 8)],
                model_repo: "test-model".to_string(),
                replica_group_id: 0,
                expert_range: None,
            },
            version: 5,
        };
        let resp = handle_update_assignment(State(state.clone()), Json(req_v5_retry))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);

        // Newer version 6 (> 5) -> OK
        let req_v6 = UpdateAssignmentRequest {
            assignment: WorkerAssignment {
                layer_ranges: vec![(8, 16)],
                model_repo: "test-model".to_string(),
                replica_group_id: 1,
                expert_range: None,
            },
            version: 6,
        };
        let resp = handle_update_assignment(State(state.clone()), Json(req_v6))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            state
                .current_assignment_version
                .load(std::sync::atomic::Ordering::SeqCst),
            6
        );
    }

    struct MockStorage {
        persisted_version: std::sync::atomic::AtomicU64,
        persisted_assignment: RwLock<Option<WorkerAssignment>>,
    }

    #[async_trait::async_trait]
    impl tessella_core::traits::StorageService for MockStorage {
        async fn save_node(
            &self,
            _node: tessella_core::models::NodeInfo,
        ) -> Result<(), DomainError> {
            Ok(())
        }
        async fn get_nodes(&self) -> Result<Vec<tessella_core::models::NodeInfo>, DomainError> {
            Ok(vec![])
        }
        async fn get_all_nodes(&self) -> Result<Vec<tessella_core::models::NodeInfo>, DomainError> {
            Ok(vec![])
        }
        async fn deregister_node(&self, _peer_id: &str) -> Result<(), DomainError> {
            Ok(())
        }
        async fn save_manifest(
            &self,
            _manifest: tessella_core::models::WeightManifest,
        ) -> Result<(), DomainError> {
            Ok(())
        }
        async fn get_manifest(
            &self,
            _model_id: &str,
            _role_json: &str,
        ) -> Result<Option<tessella_core::models::WeightManifest>, DomainError> {
            Ok(None)
        }
        async fn get_manifests_for_model(
            &self,
            _model_id: &str,
        ) -> Result<Vec<tessella_core::models::WeightManifest>, DomainError> {
            Ok(vec![])
        }
        async fn list_manifests(
            &self,
        ) -> Result<Vec<tessella_core::models::WeightManifest>, DomainError> {
            Ok(vec![])
        }
        async fn save_receipt(
            &self,
            _receipt: &tessella_core::models::SignedTransferReceipt,
        ) -> Result<(), DomainError> {
            Ok(())
        }
        async fn get_receipt(
            &self,
            _s: &str,
            _r: &str,
            _m: &str,
            _role: &str,
        ) -> Result<Option<tessella_core::models::SignedTransferReceipt>, DomainError> {
            Ok(None)
        }
        async fn list_receipts(
            &self,
            _peer_id: &str,
        ) -> Result<Vec<tessella_core::models::TransferReceipt>, DomainError> {
            Ok(vec![])
        }
        async fn update_karma(&self, _peer_id: &str, _balance: i64) -> Result<(), DomainError> {
            Ok(())
        }
        async fn get_all_karma(&self) -> Result<Vec<(String, i64)>, DomainError> {
            Ok(vec![])
        }
        async fn set_assignment_version(&self, version: u64) -> Result<(), DomainError> {
            self.persisted_version
                .store(version, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        async fn get_assignment_version(&self) -> Result<u64, DomainError> {
            Ok(self
                .persisted_version
                .load(std::sync::atomic::Ordering::SeqCst))
        }
        async fn set_active_assignment(
            &self,
            version: u64,
            assignment: &WorkerAssignment,
        ) -> Result<(), DomainError> {
            self.persisted_version
                .store(version, std::sync::atomic::Ordering::SeqCst);
            *self.persisted_assignment.write().await = Some(assignment.clone());
            Ok(())
        }
        async fn get_active_assignment(&self) -> Result<Option<WorkerAssignment>, DomainError> {
            Ok(self.persisted_assignment.read().await.clone())
        }
    }

    #[tokio::test]
    async fn test_update_assignment_persists_version_to_storage() {
        let engine = Arc::new(MockInferenceEngine {
            assignment: RwLock::new(None),
        });
        let storage = Arc::new(MockStorage {
            persisted_version: std::sync::atomic::AtomicU64::new(0),
            persisted_assignment: RwLock::new(None),
        });
        let mut state = test_api_state(Some(engine.clone()));
        state.storage = Some(storage.clone());

        let req = UpdateAssignmentRequest {
            assignment: WorkerAssignment {
                layer_ranges: vec![(0, 8)],
                model_repo: "test-model".to_string(),
                replica_group_id: 0,
                expert_range: None,
            },
            version: 42,
        };

        let resp = handle_update_assignment(State(state.clone()), Json(req.clone()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            storage
                .persisted_version
                .load(std::sync::atomic::Ordering::SeqCst),
            42
        );
        assert_eq!(
            storage.persisted_assignment.read().await.as_ref(),
            Some(&req.assignment)
        );
    }
}
