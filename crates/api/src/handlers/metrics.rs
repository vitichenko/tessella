use axum::{
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode, header::CONTENT_TYPE},
    response::IntoResponse,
};
use std::net::SocketAddr;
use tessella_ingress::ApiState;

/// Serves Prometheus text exposition format (0.0.4) for worker metrics (`TASK-94`).
///
/// Protected endpoint:
/// - If `admin_token` is configured on the worker: requires `Authorization: Bearer <token>`.
/// - If `admin_token` is not configured: only permits local loopback requests (`127.0.0.1` / `::1`).
///   External non-loopback requests without an admin token are rejected with `403 Forbidden`
///   to prevent third-party probing of economic and session activity.
pub async fn handle_metrics(
    State(state): State<ApiState>,
    addr: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    // 1. Authenticate / Authorize
    if let Some(expected_token) = &state.admin_token {
        let auth_header = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok());
        let provided_token = auth_header.and_then(|h| {
            h.strip_prefix("Bearer ")
                .or_else(|| h.strip_prefix("bearer "))
        });

        match provided_token {
            Some(token) if token == expected_token => {}
            Some(_) => return Err((StatusCode::UNAUTHORIZED, "Invalid admin token\n")),
            None => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    "Missing Authorization: Bearer <admin_token>\n",
                ));
            }
        }
    } else {
        let is_loopback = addr.is_none_or(|ConnectInfo(a)| a.ip().is_loopback());
        if !is_loopback {
            return Err((
                StatusCode::FORBIDDEN,
                "Forbidden: /metrics without admin_token is restricted to loopback\n",
            ));
        }
    }

    // 2. Query actual WAL depth from SQLite storage if available
    let depth = if let Some(storage) = &state.storage {
        storage.count_pending_work_proofs().await.unwrap_or(0)
    } else {
        0
    };

    // 3. Render Prometheus text format
    let body = if let Some(metrics) = &state.metrics {
        metrics.render_prometheus(depth)
    } else {
        String::new()
    };

    Ok((
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use std::sync::Arc;
    use tessella_core::telemetry::WorkerMetrics;

    fn mock_api_state(admin_token: Option<String>, metrics: Arc<WorkerMetrics>) -> ApiState {
        ApiState {
            dist_inference: None,
            storage: None,
            credits: None,
            discovery: None,
            slashing: None,
            peer_slashing: None,
            musivum_state: None,
            max_new_tokens: 0,
            max_input_tokens: None,
            max_input_tokens_source: None,
            hop_timeout_secs: 10,
            musivum_keypair: None,
            admin_token,
            default_model: String::new(),
            model_config: None,
            peer_failures: None,
            gpu_health: None,
            tokenizer: None,
            token_stream_senders: None,
            orchestrator_url: None,
            own_api_url: None,
            replay_cache: Arc::new(tessella_protocol::signing::ReplayCache::new()),
            callback_capabilities: None,
            metrics: Some(metrics),
            current_assignment_version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    #[tokio::test]
    async fn loopback_allowed_without_token() {
        let metrics = Arc::new(WorkerMetrics::new());
        let state = mock_api_state(None, metrics);
        let addr = "127.0.0.1:40000".parse::<SocketAddr>().unwrap();

        let resp = handle_metrics(State(state), Some(ConnectInfo(addr)), HeaderMap::new())
            .await
            .unwrap()
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("tessella_wal_pending_proofs_depth 0"));
        assert!(text.contains("tessella_proofs_queued_total 0"));
    }

    #[tokio::test]
    async fn non_loopback_forbidden_without_token() {
        let metrics = Arc::new(WorkerMetrics::new());
        let state = mock_api_state(None, metrics);
        let addr = "192.168.1.100:40000".parse::<SocketAddr>().unwrap();

        let result = handle_metrics(State(state), Some(ConnectInfo(addr)), HeaderMap::new()).await;

        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::FORBIDDEN),
            Ok(_) => panic!("Expected 403 Forbidden"),
        }
    }

    #[tokio::test]
    async fn admin_token_protection() {
        let metrics = Arc::new(WorkerMetrics::new());
        let secret = "secret_metrics_token_123456789012".to_string();
        let state = mock_api_state(Some(secret.clone()), metrics);
        let addr = "192.168.1.100:40000".parse::<SocketAddr>().unwrap();

        // 1. Missing header -> 401
        match handle_metrics(
            State(state.clone()),
            Some(ConnectInfo(addr)),
            HeaderMap::new(),
        )
        .await
        {
            Err((status, _)) => assert_eq!(status, StatusCode::UNAUTHORIZED),
            Ok(_) => panic!("Expected 401 Unauthorized"),
        }

        // 2. Wrong token -> 401
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong_token"),
        );
        match handle_metrics(State(state.clone()), Some(ConnectInfo(addr)), headers).await {
            Err((status, _)) => assert_eq!(status, StatusCode::UNAUTHORIZED),
            Ok(_) => panic!("Expected 401 Unauthorized"),
        }

        // 3. Valid token -> 200
        let mut valid_headers = HeaderMap::new();
        valid_headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {secret}")).unwrap(),
        );
        let ok = handle_metrics(State(state), Some(ConnectInfo(addr)), valid_headers)
            .await
            .unwrap()
            .into_response();
        assert_eq!(ok.status(), StatusCode::OK);
    }
}
