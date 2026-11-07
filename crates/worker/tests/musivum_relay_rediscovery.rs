//! H-05 · TASK-08: the worker must re-discover the orchestrator's relay endpoint
//! instead of using the single `/p2p_id` fetch done at boot (whose error was
//! discarded with `.ok()`).
//!
//! The mock orchestrator answers `503` twice and only then advertises an address,
//! which is exactly the case the old boot-only fetch got stuck on (orchestrator
//! restarting, NAT remap). Everything runs on an ephemeral loopback port: no
//! product port and no daemon is involved.
//!
//! The test drives the same loop the heartbeat tick runs: *while there is no
//! dialable address, fetch `/p2p_id` and dial whatever comes back as new.*

use axum::response::IntoResponse;
use axum::{http::StatusCode, routing::get, Json, Router};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tessella_worker::network_manager::{MusivumRelay, NetworkManager};

#[tokio::test]
async fn worker_rediscovers_the_musivum_relay_after_a_failed_first_fetch() {
    let peer_id = tessella_network::peer_id_from_keypair(&tessella_network::generate_keypair());
    let advertised = format!("/ip4/203.0.113.5/tcp/54326/p2p/{}", peer_id);

    let attempts = Arc::new(AtomicUsize::new(0));
    let state = (attempts.clone(), peer_id.clone(), advertised.clone());
    let app = Router::new().route(
        "/p2p_id",
        get(move || {
            let (attempts, peer_id, advertised) = state.clone();
            async move {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                if attempt < 2 {
                    // Still starting (or the address is not known yet).
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({ "error": "still starting" })),
                    )
                        .into_response();
                }
                Json(serde_json::json!({
                    "peer_id": peer_id,
                    "p2p_port": 54326,
                    "p2p_addr": advertised,
                }))
                .into_response()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let base_url = format!("http://{}", addr);
    let mut relay = MusivumRelay::default();
    assert!(relay.needs_addr());

    let mut dialed: Vec<String> = Vec::new();
    for _ in 0..5 {
        if relay.needs_addr() {
            match NetworkManager::fetch_musivum_p2p_info_from(&client, &base_url).await {
                Ok(fresh) => {
                    if let Some(address) = relay.update(fresh) {
                        dialed.push(address.to_string());
                    }
                }
                // The heartbeat tick must log and retry, never give up silently.
                Err(e) => eprintln!("retrying, /p2p_id failed: {}", e),
            }
        }
    }

    assert_eq!(
        attempts.load(Ordering::SeqCst),
        3,
        "two failures + one success"
    );
    assert_eq!(
        dialed,
        vec![advertised.clone()],
        "the relay must be dialed exactly once, after the transient failures"
    );
    assert!(!relay.needs_addr());

    // Same advertisement once more → nothing new to dial (no dial storm per tick).
    let fresh = NetworkManager::fetch_musivum_p2p_info_from(&client, &base_url)
        .await
        .expect("the third attempt advertises the relay");
    assert!(relay.update(fresh).is_none());

    // Orchestrator connection closed → the address is forgotten, so the next
    // heartbeat tick re-discovers it (a restarted orchestrator may live elsewhere).
    assert!(relay.forget_addr());
    assert!(relay.needs_addr());

    server.abort();
}
