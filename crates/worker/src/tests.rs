#[cfg(test)]
mod worker_tests {
    use tessella_core::models::TensorView;

    #[test]
    fn weighted_combination_is_normalized() {
        let output1 = TensorView {
            data: vec![1.0, 1.0],
            shape: vec![1, 2],
        };
        let output2 = TensorView {
            data: vec![1.0, 1.0],
            shape: vec![1, 2],
        };
        let w1 = 0.3f32;
        let w2 = 0.7f32;

        let mut combined_data = vec![0.0f32; 2];
        let mut total_weight = 0.0f32;

        // Simulating Step 4 combination
        for (acc, val) in combined_data.iter_mut().zip(output1.data.iter()) {
            *acc += val * w1;
        }
        total_weight += w1;

        for (acc, val) in combined_data.iter_mut().zip(output2.data.iter()) {
            *acc += val * w2;
        }
        total_weight += w2;

        if total_weight > 0.0 {
            for v in combined_data.iter_mut() {
                *v /= total_weight;
            }
        }

        assert_eq!(total_weight, 1.0);
        assert_eq!(combined_data, vec![1.0, 1.0]);
    }

    #[tokio::test]
    async fn test_pending_work_proofs_wal_lifecycle() {
        use tessella_core::models::{SignedWorkProof, WorkProof};
        use tessella_storage::SqliteStorage;

        let db = SqliteStorage::new(":memory:", None).unwrap();

        let proof = SignedWorkProof {
            proof: WorkProof {
                request_id: "test-req".to_string(),
                worker_id: "test-worker".to_string(),
                credits_delta: 5.0,
                output_hash: [0u8; 32],
                timestamp: 123456789,
                input_hash: [0u8; 32],
                layer_range: (0, 4),
                work_ticket: tessella_core::models::WorkTicket {
                    request_id: "test-req".to_string(),
                    model_id: "test-model".to_string(),
                    entry_peer_id: "test-worker".to_string(),
                    issued_at: 123_456_000,
                    expires_at: 999_999_999,
                    audit_layer: None,
                    audit_replicas: 2,
                    audit_canary: None,
                    audit_reward: 0.0,
                    orchestrator_signature: vec![7; 64],
                },
            },
            signature: vec![1, 2, 3],
            public_key: vec![4, 5, 6],
        };

        // Save
        let id = db.save_pending_work_proof(&proof).await.unwrap();

        // Get
        let pending = db.get_pending_work_proofs().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, id);
        assert_eq!(pending[0].1.proof.request_id, "test-req");
        assert_eq!(pending[0].1.signature, vec![1, 2, 3]);

        // Delete
        db.delete_pending_work_proofs(vec![id]).await.unwrap();

        // Verify empty
        let pending_after = db.get_pending_work_proofs().await.unwrap();
        assert!(pending_after.is_empty());
    }

    #[test]
    fn test_config_env_overrides() {
        std::env::set_var("TESSELLA_API_HOST", "127.0.0.9");
        std::env::set_var("TESSELLA_API_PORT", "18080");
        std::env::set_var("TESSELLA_P2P_HOST", "127.0.0.10");
        std::env::set_var("TESSELLA_P2P_PORT", "19080");
        std::env::set_var("TESSELLA_PUBLIC_IP", "1.2.3.4");
        std::env::set_var("TESSELLA_PUBLIC_P2P_PORT", "29080");

        let cfg = crate::config::WorkerConfig::load("non_existent_file.toml").unwrap();

        assert_eq!(cfg.api_host, "127.0.0.9");
        assert_eq!(cfg.api_port, 18080);
        assert_eq!(cfg.p2p_host, "127.0.0.10");
        assert_eq!(cfg.p2p_port, 19080);
        assert_eq!(cfg.public_ip, Some("1.2.3.4".to_string()));
        assert_eq!(cfg.public_p2p_port, Some(29080));

        // Clean up env
        std::env::remove_var("TESSELLA_API_HOST");
        std::env::remove_var("TESSELLA_API_PORT");
        std::env::remove_var("TESSELLA_P2P_HOST");
        std::env::remove_var("TESSELLA_P2P_PORT");
        std::env::remove_var("TESSELLA_PUBLIC_IP");
        std::env::remove_var("TESSELLA_PUBLIC_P2P_PORT");
    }

    /// El retirado `is_cuda_gpu_error` (`TASK-77`/S84) devolvía un `bool` para estos mismos mensajes, y
    /// con él el llamante no podía separar «esta petición no cupo» —un juicio sobre la petición— de
    /// «este device murió» —un juicio sobre el nodo—: los dos sacaban al nodo de la malla para siempre.
    /// Lo que este test fija es lo que ocupó su lugar: la **clase** del fallo, la sonda como oráculo
    /// (sólo se paga cuando el veredicto depende de ella) y que un device sano contesta que sí.
    #[test]
    fn test_the_device_probe_is_the_oracle_of_a_gpu_verdict() {
        use crate::inference_engine::device_probe;
        use tessella_core::telemetry::gpu_health::{
            classify_gpu_failure, probe_needed, GpuFailureKind,
        };

        // Un `OOM` es un juicio sobre la petición: ni se le pregunta al device.
        let oom = classify_gpu_failure("CUDA driver error: Out of Memory").expect("clasificado");
        assert_eq!(oom, GpuFailureKind::OutOfMemory);
        assert_eq!(
            classify_gpu_failure("OOM error in kernel execution"),
            Some(GpuFailureKind::OutOfMemory)
        );
        assert!(!probe_needed(oom));

        // Un device perdido sí, y sobre un device vivo la sonda contesta que sí.
        let lost = classify_gpu_failure("Illegal Memory Access on device").expect("clasificado");
        assert_eq!(lost, GpuFailureKind::DeviceLost);
        assert_eq!(
            classify_gpu_failure("Invalid PTX version"),
            Some(GpuFailureKind::DeviceLost)
        );
        assert!(probe_needed(lost));
        assert!(device_probe(&candle_core::Device::Cpu).is_ok());

        assert_eq!(classify_gpu_failure("Some normal validation error"), None);
    }

    #[tokio::test]
    async fn test_worker_metrics_on_proof_and_sweeper_lifecycle() {
        use crate::task_scheduler::TaskScheduler;
        use axum::routing::post;
        use std::sync::atomic::Ordering;
        use std::sync::Arc;
        use tessella_core::models::{SignedWorkProof, WorkProof, WorkTicket};
        use tessella_core::telemetry::WorkerMetrics;
        use tessella_storage::SqliteStorage;

        let metrics = Arc::new(WorkerMetrics::new());
        let db = SqliteStorage::new(":memory:", None)
            .unwrap()
            .with_metrics(metrics.clone());
        let client = reqwest::Client::new();

        // 1. Initial assertion
        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 0);
        assert_eq!(metrics.proofs_queued.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.proofs_settled.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.batch_uploads_success.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.batch_uploads_failure.load(Ordering::Relaxed), 0);

        let make_proof = |req_id: &str| SignedWorkProof {
            proof: WorkProof {
                request_id: req_id.to_string(),
                worker_id: "test-worker".to_string(),
                credits_delta: 2.0,
                output_hash: [0u8; 32],
                timestamp: 100,
                input_hash: [0u8; 32],
                layer_range: (0, 4),
                work_ticket: WorkTicket {
                    request_id: req_id.to_string(),
                    model_id: "test-model".to_string(),
                    entry_peer_id: "test-worker".to_string(),
                    issued_at: 100,
                    expires_at: 999_999,
                    audit_layer: None,
                    audit_replicas: 2,
                    audit_canary: None,
                    audit_reward: 0.0,
                    orchestrator_signature: vec![1; 64],
                },
            },
            signature: vec![1, 2, 3],
            public_key: vec![4, 5, 6],
        };

        // 2. Proof generation and WAL storage
        db.save_pending_work_proof(&make_proof("req-1"))
            .await
            .unwrap();
        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 1);
        assert_eq!(metrics.proofs_queued.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.proofs_settled.load(Ordering::Relaxed), 0);

        // 3. Mock orchestrator server for successful batch upload
        let mock_app = axum::Router::new().route(
            "/api/v1/work_done_batch",
            post(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mock_addr = listener.local_addr().unwrap();
        let mock_url = format!("http://{}", mock_addr);
        tokio::spawn(async move {
            axum::serve(listener, mock_app).await.unwrap();
        });

        // 4. Run sweeper against mock orchestrator
        TaskScheduler::sweep_once(&db, &mock_url, &client, Some(&metrics))
            .await
            .expect("sweeper must succeed");

        // 5. Assert post-sweep state
        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 0);
        assert_eq!(metrics.proofs_queued.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.proofs_settled.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.batch_uploads_success.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.batch_uploads_failure.load(Ordering::Relaxed), 0);
        assert_eq!(
            metrics.batch_upload_latency_count.load(Ordering::Relaxed),
            1
        );

        // 6. Save a second proof
        db.save_pending_work_proof(&make_proof("req-2"))
            .await
            .unwrap();
        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 1);
        assert_eq!(metrics.proofs_queued.load(Ordering::Relaxed), 2);

        // 7. Run sweeper against a failing endpoint (unreachable port)
        let failing_url = "http://127.0.0.1:1";
        let sweep_err = TaskScheduler::sweep_once(&db, failing_url, &client, Some(&metrics)).await;
        assert!(sweep_err.is_err());

        // 8. Assert error state: proof remains in WAL, failure counter incremented, settled unchanged
        assert_eq!(db.count_pending_work_proofs().await.unwrap(), 1);
        assert_eq!(metrics.proofs_queued.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.proofs_settled.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.batch_uploads_success.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.batch_uploads_failure.load(Ordering::Relaxed), 1);
        assert_eq!(
            metrics.batch_upload_latency_count.load(Ordering::Relaxed),
            2
        );

        // 9. Verify Prometheus exposition output
        let rendered = metrics.render_prometheus(db.count_pending_work_proofs().await.unwrap());
        assert!(rendered.contains("tessella_wal_pending_proofs_depth 1"));
        assert!(rendered.contains("tessella_proofs_queued_total 2"));
        assert!(rendered.contains("tessella_proofs_settled_total 1"));
        assert!(rendered.contains("tessella_proof_batch_uploads_total{status=\"success\"} 1"));
        assert!(rendered.contains("tessella_proof_batch_uploads_total{status=\"failure\"} 1"));
        assert!(rendered.contains("tessella_proof_batch_upload_duration_seconds_count 2"));
    }
}
