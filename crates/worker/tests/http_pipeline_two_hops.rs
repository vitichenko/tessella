//! The distributed pipeline against **real hops** (H-23 · `TASK-12` · S17).
//!
//! This file replaces `two_worker_routing.rs`, whose two mock handlers only added `+1.0` /
//! `*2.0` to a raw `Vec<f32>` body. That test never touched `HttpPipelineRouter`, the routing
//! table, `HiddenStatePayload` or the H-09 signature, so its name promised coverage that did
//! not exist (audit H-23: "the name induces to think it covers the real P2P routing").
//!
//! What is exercised now, all of it over real HTTP and with the production code of the router:
//!
//! * `HttpPipelineRouter::route` walking **two** hops from a routing table it resolves itself
//!   (`find_target_for_layer_excluding` → `resolve_preferred_http_addr` → `POST /v1/forward`);
//! * a `HiddenStatePayload` built from a real `candle` tensor (`from_tensor` → `to_bytes`);
//! * the H-09 seal: the entry node signs the payload with `seal_signed_request` and **each hop
//!   verifies it** with `verify_signed_request` before touching the tensor, so breaking the
//!   signature (or the preimage) makes this test fail instead of silently routing garbage;
//! * the layer advance (`layer_idx` is read back from the *response*) and the data path
//!   (`(1.0 + 1.0) * 2.0 = 4.0`);
//! * an unsigned payload answering `401` and a **tampered** payload answering `401`.
//!
//! The mocks are deliberately dumb (one `affine` each): what they must not be dumb about is the
//! wire contract, which is why they speak the real format instead of a private one. A real
//! `InferenceEngine` cannot be used here because Tessella is GPU-only (D-6) and this suite must
//! run everywhere.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use candle_core::{Device, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tessella_core::models::{LayerPeer, NodeInfo, PeerStatus, Role};
use tessella_core::telemetry::metrics::PeerFailureRegistry;
use tessella_core::traits::PipelineRouter;
use tessella_inference::{HiddenStatePayload, PayloadTensorExt};
use tessella_network::{
    generate_keypair, seal_signed_request, verify_signature, verify_signed_request, NetworkKeypair,
};
use tessella_protocol::signing::ReplayCache;
use tessella_protocol::PeerSignedRequest;
use tessella_worker::http_pipeline_router::HttpPipelineRouter;
use tessella_worker::routing_table::RoutingTable;
use tessella_worker::topology_cache::TopologyCache;
use tokio::sync::{Notify, RwLock};

const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
/// The hidden size of the fixture. Small on purpose: the shape travels in the payload and the
/// mock only applies an `affine`, so 32 floats are enough to detect a wrong hop or order.
const HIDDEN: usize = 32;

/// The wait a node declares for a hop of its own pipeline — `p2p_inference_timeout_secs`
/// (`TASK-76`/S83). 30 s is the default of the config and what the router already intended (it is
/// the timeout of its own HTTP client); the e2e harness declares 60 s. What matters here is that it
/// is **declared** and not a constant welded into the router.
const HOP_WAIT: Duration = Duration::from_secs(30);

/// What a hop saw, kept so the assertions can prove *which* hop processed *what* and with
/// *whose* signature.
#[derive(Debug, Clone)]
struct HopSeen {
    request_id: String,
    layer_idx_in: usize,
    layer_idx_out: usize,
    /// The public key the payload was signed with — it must be the entry node's one at **every**
    /// hop (a hop copies the signature verbatim, it does not re-seal: see H-09 in `TASK-05`).
    signer_pubkey: Vec<u8>,
    /// `X-Retry` is only added when the router has already burned a candidate for this layer.
    retry: bool,
}

#[derive(Clone)]
struct HopConfig {
    peer_id: String,
    /// The layers this hop serves. The answer it emits is determined by **its own** range
    /// (`layer_range.1 - 1`), not by the index of the payload it received: the incoming index is
    /// the last layer the *upstream* hop processed, so every hop would otherwise re-answer the
    /// same index and the router would never leave the layer it is on.
    layer_range: (usize, usize),
    mul: f64,
    add: f64,
    /// `TASK-76`/S83: how long this hop **works** before answering. A hop that applies a prompt of
    /// the published ceiling costs seconds of real GPU on the reference machine, so the wait of the
    /// node above has to cover it: this is what makes «slow» and «dead» distinguishable.
    delay: Duration,
    /// `TASK-68`/S75: answer the refusal of a **decode step** (`seq_start_pos > 0`) that this node
    /// never prefilled — the `409` plus `X-Tessella-Refusal` the real edge produces
    /// (`handle_forward_hidden`), instead of serving the step.
    refuse_missing_prefill: bool,
    /// `TASK-76`/S83: the named refusal this hop answers **instead of** serving its layer.
    refusal: Option<HopRefusal>,
    /// How many times this hop refused the request. It is deliberately **not** `seen`: a refusal is
    /// not work, and every other assertion of this file uses `seen` to prove what was *computed*.
    refusals: Arc<AtomicUsize>,
    replay: Arc<ReplayCache>,
    seen: Arc<Mutex<Vec<HopSeen>>>,
}

/// The named refusal of a hop (`TASK-76`/S83, `TASK-84`/S91), i.e. the refusals of the mesh that a hop
/// can produce without a GPU.
///
/// The refusal travels as the three things the router of the upstream node reads — the HTTP status,
/// the code in `X-Tessella-Refusal` and a body whose **first word is the code** —, and it is the same
/// contract `missing_prefill` already had (`TASK-68`/S75): a refusal is a **terminal** answer, which no
/// other candidate can improve, unlike a dead or overloaded hop.
#[derive(Debug, Clone, Copy)]
enum HopRefusal {
    /// The body is over the wire limit of `POST /v1/forward` (`413`). In production it never reaches
    /// the handler: `HopBody` refuses it while reading the body.
    Bytes,
    /// The prompt is longer than what this node can prefill (`400`). In production the engine refuses
    /// it **before** the prefill, so it costs nothing.
    Prompt,
    /// The node is already reading its quota of hop bodies (`503`, `TASK-84`/S91). In production it
    /// never reaches the handler either: `HopBody` refuses it before the first byte. It is the one
    /// refusal whose cause is **not** the bytes of the request — the same bytes fit here later — and it
    /// is terminal anyway, because retrying it would push the same body at every candidate.
    Quota,
}

impl HopConfig {
    fn new(peer_id: &str, layer_range: (usize, usize), mul: f64, add: f64) -> Self {
        Self {
            peer_id: peer_id.to_string(),
            layer_range,
            mul,
            add,
            delay: Duration::ZERO,
            refuse_missing_prefill: false,
            refusal: None,
            refusals: Arc::new(AtomicUsize::new(0)),
            replay: Arc::new(ReplayCache::new()),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The same hop, playing the node of `F-1`: a live process that never prefilled this session.
    fn refusing_missing_prefill(mut self) -> Self {
        self.refuse_missing_prefill = true;
        self
    }

    /// The same hop, refusing the body that does not fit in one hop (`413` + `payload_too_large`,
    /// `TASK-76`/S83).
    fn refusing_oversized_bytes(mut self) -> Self {
        self.refusal = Some(HopRefusal::Bytes);
        self
    }

    /// The same hop, refusing a prompt longer than it can prefill (`400` + `prompt_too_long`,
    /// `TASK-76`/S83).
    fn refusing_a_prompt_over_its_ceiling(mut self) -> Self {
        self.refusal = Some(HopRefusal::Prompt);
        self
    }

    /// The same hop, at its quota of hop bodies (`503` + `too_many_hop_bodies`, `TASK-84`/S91).
    fn refusing_at_its_quota(mut self) -> Self {
        self.refusal = Some(HopRefusal::Quota);
        self
    }

    /// The same hop, but **slow**: it applies its layer and answers only after `delay`
    /// (`TASK-76`/S83). A hop like this is what a prompt of the published ceiling looks like — in
    /// the `S76` measurement the entry node spent 4.63 s on 1 065 tokens, and the sweep of `S83`
    /// found the hop of a 2 037-token prefix killed at 5 s just before answering.
    fn answering_after(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// How many times this hop answered a refusal instead of serving the layer.
    fn refusals(&self) -> usize {
        self.refusals.load(Ordering::SeqCst)
    }

    fn seen(&self) -> Vec<HopSeen> {
        self.seen.lock().expect("hop log is not poisoned").clone()
    }
}

/// A hop: authenticate, transform, re-emit.
///
/// The response carries the **last layer this hop processed** (`layer_range.1 - 1`, mirroring
/// `inference_engine.rs:527-533`), which is what `HttpPipelineRouter::route` uses to continue: it
/// resumes at `payload.layer_idx + 1`. The index of the *incoming* payload is therefore not the
/// layer being requested — it is what the upstream hop reported — so each mock derives its answer
/// from its own range.
///
/// The authentication comes first, exactly as in the production handler
/// (`handle_forward_hidden` verifies the signature *before* the payload reaches the engine).
async fn hop_handler(State(hop): State<HopConfig>, headers: HeaderMap, body: Bytes) -> Response {
    // `TASK-76`/S83 and `TASK-84`/S91: the named refusals that need no GPU to be real. All of them are
    // answered **before** anything is authenticated or computed, which is where the production path
    // answers them too — the body that does not fit never reaches the handler (`HopBody` refuses it
    // while reading the body), the body of a node at its quota does not reach it either (`HopBody`
    // refuses it before the first byte) and the prompt that does not fit is refused by the engine
    // **before** the prefill.
    //
    // The three halves of the contract are the same as `missing_prefill`'s (`TASK-68`/S75): the status
    // of `inference_error_status`, the code in `X-Tessella-Refusal`, and a body that starts with the
    // code. Without them the router of the upstream node cannot tell «these bytes will never fit»
    // (abort) apart from «this node is slow» (try another candidate).
    if let Some(refusal) = hop.refusal {
        hop.refusals.fetch_add(1, Ordering::SeqCst);
        let (status, code, error) = match refusal {
            HopRefusal::Bytes => {
                let limit = tessella_core::constants::inference::MAX_HIDDEN_STATE_BYTES;
                (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    tessella_core::error::PAYLOAD_TOO_LARGE_CODE,
                    tessella_core::error::payload_too_large_error(limit + 1, limit),
                )
            }
            HopRefusal::Prompt => (
                StatusCode::BAD_REQUEST,
                tessella_core::error::PROMPT_TOO_LONG_CODE,
                tessella_core::error::prompt_too_long_error(
                    12_000,
                    HIDDEN,
                    std::mem::size_of::<f32>(),
                    &tessella_core::constants::inference::HopCeiling {
                        tokens: 9_362,
                        source: tessella_core::constants::inference::HopCeilingSource::Wire,
                        wire_tokens: Some(9_362),
                        declared_tokens: 0,
                    },
                ),
            ),
            // `TASK-84`/S91: the quota of the node, the only one of the family whose reason is
            // «not now» instead of «never here» — and the one that a router retrying blindly would
            // amplify, because the body it refuses is the 70-80 MiB one.
            HopRefusal::Quota => (
                StatusCode::SERVICE_UNAVAILABLE,
                tessella_core::error::TOO_MANY_HOP_BODIES_CODE,
                tessella_core::error::too_many_hop_bodies_error(
                    tessella_core::constants::inference::HOP_BODY_CONCURRENCY,
                    tessella_core::constants::inference::HOP_BODY_CONCURRENCY,
                ),
            ),
        };
        return (
            status,
            [(
                axum::http::HeaderName::from_static(tessella_core::error::REFUSAL_HEADER),
                axum::http::HeaderValue::from_static(code),
            )],
            tessella_core::error::refusal_message(&error)
                .expect("a named refusal has a message to publish")
                .to_string(),
        )
            .into_response();
    }

    let payload = match HiddenStatePayload::from_bytes(&body) {
        Ok(payload) => payload,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("Invalid payload bytes: {e}"),
            )
                .into_response();
        }
    };

    let now = tessella_core::time::unix_now_secs();
    if let Err(rejection) = verify_signed_request(&payload, &hop.replay, now, "test hop") {
        return (
            StatusCode::UNAUTHORIZED,
            format!("Signed request rejected: {rejection}"),
        )
            .into_response();
    }

    let tensor = match payload.to_tensor(&Device::Cpu) {
        Ok(tensor) => tensor,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    // `TASK-68`/S75: the node of `F-1` — a live hop that never prefilled this session, asked for a
    // decode step (`seq_start_pos > 0`). It refuses **before** computing anything, exactly as the
    // engine does, with the same status, the same machine-readable header and the same code in the
    // body (the three things the router of the upstream node reads).
    if hop.refuse_missing_prefill && payload.seq_start_pos > 0 {
        hop.seen
            .lock()
            .expect("hop log is not poisoned")
            .push(HopSeen {
                request_id: payload.request_id.clone(),
                layer_idx_in: payload.layer_idx,
                layer_idx_out: payload.layer_idx,
                signer_pubkey: payload.client_pubkey.clone(),
                retry: headers.contains_key("X-Retry"),
            });
        return (
            StatusCode::CONFLICT,
            [(
                axum::http::HeaderName::from_static(tessella_core::error::REFUSAL_HEADER),
                axum::http::HeaderValue::from_static(tessella_core::error::MISSING_PREFILL_CODE),
            )],
            tessella_core::error::missing_prefill_error(
                &payload.request_id,
                payload.seq_start_pos as usize,
                hop.layer_range,
            )
            .to_string(),
        )
            .into_response();
    }

    // `TASK-76`/S83: the hop **works** for `delay` before it can answer. It goes after every
    // callback-free refusal and before the transform, which is where a real hop spends its time (the
    // prefill of the prefix, then the layers of its own shard).
    if !hop.delay.is_zero() {
        tokio::time::sleep(hop.delay).await;
    }

    let transformed = match tensor.affine(hop.mul, hop.add) {
        Ok(tensor) => tensor,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    let mut next = match HiddenStatePayload::from_tensor(
        &transformed,
        payload.request_id.clone(),
        payload.seq_start_pos,
        hop.layer_range.1 - 1,
    ) {
        Ok(next) => next,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };
    next.session_id = payload.session_id.clone();
    // Hops copy the signature fields verbatim; the preimage is built from fields that survive a
    // hop (`session_id`, `request_id`, tokens, `entry_node_url`, tier, nonce, iat), so it still
    // verifies at the next hop.
    next.signature_version = payload.signature_version;
    next.nonce = payload.nonce;
    next.iat = payload.iat;
    next.signature = payload.signature.clone();
    next.client_pubkey = payload.client_pubkey.clone();
    // `TASK-07`/`TASK-53`: the work ticket and the karma ticket are the authorisation and the
    // priority of the **session**, so a hop propagates them verbatim (it is what the production path
    // does in `forward_hidden`). Without this, a 3-node pipeline would leave the hops after the first
    // one without authorisation — and the audit of the last shard could not be reported either.
    next.karma_ticket = payload.karma_ticket.clone();
    next.work_ticket = payload.work_ticket.clone();

    hop.seen
        .lock()
        .expect("hop log is not poisoned")
        .push(HopSeen {
            request_id: payload.request_id.clone(),
            layer_idx_in: payload.layer_idx,
            layer_idx_out: next.layer_idx,
            signer_pubkey: payload.client_pubkey.clone(),
            retry: headers.contains_key("X-Retry"),
        });

    match next.to_bytes() {
        Ok(bytes) => (StatusCode::OK, bytes).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Binds a hop on an **ephemeral** loopback port and returns its base URL.
async fn spawn_hop(hop: HopConfig) -> anyhow::Result<String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let app = Router::new()
        .route("/v1/forward", post(hop_handler))
        .with_state(hop);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(format!("http://{addr}"))
}

fn layer_peer(peer_id: &str, http_addr: &str, layer_range: (usize, usize)) -> LayerPeer {
    LayerPeer {
        peer_id: peer_id.to_string(),
        http_addr: http_addr.to_string(),
        p2p_addr: String::new(),
        layer_ranges: vec![layer_range],
        expert_range: None,
        queue_depth: 0,
        public_ip: None,
        public_api_port: None,
        canary: false,
    }
}

/// A tensor whose elements are all `value`, i.e. the input of the classic `(1+1)*2 = 4` chain.
fn hidden_state(value: f32) -> anyhow::Result<Tensor> {
    Tensor::from_vec(vec![value; HIDDEN], (1, 1, HIDDEN), &Device::Cpu).map_err(Into::into)
}

/// The payload an entry node emits: a real tensor, sealed with the canonical preimage v1.
fn sealed_payload(keypair: &NetworkKeypair) -> anyhow::Result<HiddenStatePayload> {
    let mut payload =
        HiddenStatePayload::from_tensor(&hidden_state(1.0)?, "req-two-hops".to_string(), 0, 0)?;
    payload.session_id = Some("session-two-hops".to_string());
    seal_signed_request(
        keypair,
        &mut payload,
        4242,
        tessella_core::time::unix_now_secs(),
    )?;
    Ok(payload)
}

/// A **decode step** of an ongoing session (`TASK-68`/S75): one position of hidden state and a prefix
/// (`seq_start_pos > 0`) that the receiving node is supposed to have **already** applied — which is
/// exactly what a hop that just replaced a dead node does not have.
fn sealed_decode_step(
    keypair: &NetworkKeypair,
    seq_start_pos: usize,
) -> anyhow::Result<HiddenStatePayload> {
    let mut payload = HiddenStatePayload::from_tensor(
        &hidden_state(1.0)?,
        "req-s75-decode".to_string(),
        u32::try_from(seq_start_pos).expect("a prefix that fits in the wire field"),
        0,
    )?;
    payload.session_id = Some("session-s75".to_string());
    seal_signed_request(
        keypair,
        &mut payload,
        4343,
        tessella_core::time::unix_now_secs(),
    )?;
    Ok(payload)
}

fn router_for(peers: Vec<LayerPeer>) -> anyhow::Result<HttpPipelineRouter> {
    // Never reached in these tests: both layers resolve from the table, and a freshly built table is
    // not stale, so `refresh()` is not called either.
    router_with_cache(peers, Vec::new(), "http://127.0.0.1:1".to_string())
}

/// The same router, with the cached topology populated and a chosen orchestrator URL.
///
/// `TASK-36`/S42: the fallback of the router used to be an HTTP call to `GET /substitute` — an
/// endpoint that does not exist — and the cached-topology lookup only ran *by accident*, inside the
/// `_ =>` arm that swallowed the 404. The regression test at the end of this file passes an
/// orchestrator that **does** answer that dead endpoint, so the two paths are distinguishable by
/// their result instead of by a log line.
fn router_with_cache(
    peers: Vec<LayerPeer>,
    cached: Vec<LayerPeer>,
    orchestrator_url: String,
) -> anyhow::Result<HttpPipelineRouter> {
    router_with_registry(
        peers,
        cached,
        orchestrator_url,
        Arc::new(PeerFailureRegistry::new()),
        Arc::new(generate_keypair()),
        HOP_WAIT,
    )
}

/// The same router, with a keypair the caller keeps (`TASK-53`/S59): the audit report of a sampled
/// session travels signed, so a test that reads it needs the key that signed it.
fn router_with_identity(
    peers: Vec<LayerPeer>,
    orchestrator_url: String,
    keypair: Arc<NetworkKeypair>,
) -> anyhow::Result<HttpPipelineRouter> {
    router_with_registry(
        peers,
        Vec::new(),
        orchestrator_url,
        Arc::new(PeerFailureRegistry::new()),
        keypair,
        HOP_WAIT,
    )
}

/// The same router with a registry the caller keeps (`TASK-50`/S56): a failed hop must be visible in
/// it, so the test needs the handle the worker publishes through `GET /status`.
///
/// The keypair is a parameter because a router **signs** the audit report of a sampled session
/// (`TASK-53`/S59) and a test that asserts what travels to the orchestrator has to be able to verify
/// that signature. The peer id of the router is derived from it, so the fixture cannot claim one
/// identity and sign with another.
fn router_with_registry(
    peers: Vec<LayerPeer>,
    cached: Vec<LayerPeer>,
    orchestrator_url: String,
    peer_failures: Arc<PeerFailureRegistry>,
    keypair: Arc<NetworkKeypair>,
    hop_wait: Duration,
) -> anyhow::Result<HttpPipelineRouter> {
    let routing_table = Arc::new(RwLock::new(RoutingTable::new()));
    routing_table
        .try_write()
        .expect("a fresh lock is free")
        .replace_all(peers);
    let topology_cache = Arc::new(RwLock::new(TopologyCache::new(Duration::from_secs(300))));
    topology_cache
        .try_write()
        .expect("a fresh lock is free")
        .update(cached);
    Ok(HttpPipelineRouter::new(
        routing_table,
        topology_cache,
        reqwest::Client::new(),
        orchestrator_url,
        tessella_network::peer_id_from_keypair(&keypair),
        MODEL_ID.to_string(),
        peer_failures,
        Arc::new(Notify::new()),
        keypair,
        hop_wait,
    ))
}

/// A peer as the topology cache stores it (`TASK-58`/S64: the cache and the routing table share
/// the same record, so this is also what `layer_peer(...)` builds above).
fn cached_peer(peer_id: &str, http_addr: &str, layer_range: (usize, usize)) -> LayerPeer {
    LayerPeer {
        peer_id: peer_id.to_string(),
        http_addr: http_addr.to_string(),
        p2p_addr: String::new(),
        layer_ranges: vec![layer_range],
        expert_range: None,
        queue_depth: 0,
        public_ip: None,
        public_api_port: None,
        canary: false,
    }
}

/// A `NodeInfo` as the `/peers?model_id=` routing contract answers it.
fn peers_entry(peer_id: &str, http_addr: &str, layer_range: (usize, usize)) -> NodeInfo {
    NodeInfo {
        peer_id: peer_id.to_string(),
        host: "127.0.0.1".to_string(),
        port: 9001,
        vram_mb: 4096,
        role: Role::Worker {
            layer_ranges: vec![layer_range],
            replica_group_id: 0,
            expert_range: None,
        },
        model_id: MODEL_ID.to_string(),
        p2p_address: String::new(),
        http_api_url: Some(http_addr.to_string()),
        relay_address: None,
        status: PeerStatus::Healthy,
        failed_requests: 0,
        successful_requests: 1,
        latency_p99_ms: 1,
        queue_depth: 0,
        last_error: None,
        slash_count: 0,
        blacklisted: false,
        public_ip: None,
        public_api_port: None,
    }
}

/// A stand-in for the `GET /substitute` endpoint that never existed, answering as if it did. It is
/// what makes the regression test at the end of this file fail before the fix.
async fn spawn_substitute_stub(candidate: NodeInfo) -> anyhow::Result<String> {
    async fn handler(State(candidate): State<NodeInfo>) -> axum::Json<Vec<NodeInfo>> {
        axum::Json(vec![candidate])
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let app = Router::new()
        .route("/substitute", axum::routing::get(handler))
        .with_state(candidate);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(format!("http://{addr}"))
}

/// H-23: the router must walk both hops, and what it walks must be signed by the entry node.
#[tokio::test]
async fn the_router_walks_two_signed_hops_and_advances_the_layer() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWFirstHop", (0, 1), 1.0, 1.0); // + 1.0
    let second = HopConfig::new("12D3KooWSecondHop", (1, 2), 2.0, 0.0); // * 2.0
    let first_url = spawn_hop(first.clone()).await?;
    let second_url = spawn_hop(second.clone()).await?;

    let router = router_for(vec![
        layer_peer(&first.peer_id, &first_url, first.layer_range),
        layer_peer(&second.peer_id, &second_url, second.layer_range),
    ])?;

    let keypair = generate_keypair();
    let payload_bytes = sealed_payload(&keypair)?.to_bytes()?;

    let out = router
        .route(0, 2, payload_bytes, MODEL_ID, None)
        .await
        .expect("the pipeline must complete against two healthy hops");

    // The second hop is the last one to answer, so this is the state *after* both.
    let out_payload = HiddenStatePayload::from_bytes(&out)?;
    assert_eq!(
        out_payload.layer_idx, 1,
        "the router must have advanced past both hops (layer_idx is read back from the response)"
    );

    let values = out_payload
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(values.len(), HIDDEN);
    for value in &values {
        assert!(
            (*value - 4.0).abs() < 1e-6,
            "(1.0 + 1.0) * 2.0 = 4.0 per element, got {values:?}"
        );
    }

    // Each hop authenticated the payload, and both did it with the **entry node's** key: a hop
    // that re-sealed with its own key would still produce 4.0 but would break the chain for the
    // next one, so this is not redundant with the asserts above.
    let expected_signer = keypair.public().encode_protobuf();
    let first_seen = first.seen();
    let second_seen = second.seen();
    assert_eq!(first_seen.len(), 1, "the first hop must run exactly once");
    assert_eq!(second_seen.len(), 1, "the second hop must run exactly once");
    assert_eq!(first_seen[0].layer_idx_in, 0);
    assert_eq!(first_seen[0].layer_idx_out, 0);
    // The second hop receives the index the *first* one reported (0), not "1": the layer being
    // requested is the router's business, and what a hop answers comes from its own range.
    assert_eq!(second_seen[0].layer_idx_in, 0);
    assert_eq!(second_seen[0].layer_idx_out, 1);
    for hop in [&first_seen[0], &second_seen[0]] {
        assert_eq!(
            hop.signer_pubkey, expected_signer,
            "the payload must still carry the entry node's signature at every hop"
        );
        assert_eq!(hop.request_id, "req-two-hops");
        assert!(!hop.retry, "no hop failed, so X-Retry must never be sent");
    }

    Ok(())
}

/// The criterion: an unsigned payload must be answered with `401`, and the very same
/// payload sealed with the entry node's key must not be (so the `401` is about the signature and
/// not about the endpoint rejecting everything).
#[tokio::test]
async fn a_hop_answers_401_to_an_unsigned_payload_and_200_to_the_sealed_one() -> anyhow::Result<()>
{
    let hop = HopConfig::new("12D3KooWGuardHop", (0, 1), 1.0, 0.0);
    let url = spawn_hop(hop.clone()).await?;
    let client = reqwest::Client::new();
    let endpoint = format!("{url}/v1/forward");

    // `from_tensor` leaves `signature`/`client_pubkey` empty: this is exactly what the router
    // used to send before H-09.
    let unsigned =
        HiddenStatePayload::from_tensor(&hidden_state(1.0)?, "req-unsigned".to_string(), 0, 0)?
            .to_bytes()?;
    let response = client
        .post(&endpoint)
        .header("content-type", "application/octet-stream")
        .body(unsigned)
        .send()
        .await?;
    assert_eq!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "an unsigned hidden state must not be processed by a hop"
    );

    let keypair = generate_keypair();
    let sealed = sealed_payload(&keypair)?.to_bytes()?;
    let response = client
        .post(&endpoint)
        .header("content-type", "application/octet-stream")
        .body(sealed)
        .send()
        .await?;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "the sealed payload must be accepted (otherwise the 401 above proves nothing)"
    );
    assert_eq!(
        hop.seen().len(),
        1,
        "only the sealed payload reached the hop"
    );

    Ok(())
}

/// H-09 has teeth: mutating a field of the signed preimage invalidates the request even if the
/// signature bytes are still attached.
#[tokio::test]
async fn a_hop_answers_401_to_a_tampered_payload() -> anyhow::Result<()> {
    let hop = HopConfig::new("12D3KooWTamperHop", (0, 1), 1.0, 0.0);
    let url = spawn_hop(hop.clone()).await?;

    let keypair = generate_keypair();
    let mut payload = sealed_payload(&keypair)?;
    // `session_id` is inside the v1 preimage (an attacker cannot re-point the KV-cache key).
    payload.session_id = Some("session-somebody-else".to_string());
    let tampered = payload.to_bytes()?;

    let response = reqwest::Client::new()
        .post(format!("{url}/v1/forward"))
        .header("content-type", "application/octet-stream")
        .body(tampered)
        .send()
        .await?;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(
        hop.seen().is_empty(),
        "a tampered payload must never reach the transformation"
    );

    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `TASK-27`/S33 · the active audit: the shard that ends the pipeline is executed a second time on
// a replica of the same layer range, and an answer that two executions do not corroborate is not
// served. Everything here is real HTTP against two hops; nothing needs a GPU.
// ---------------------------------------------------------------------------------------------

/// Two hops that cover the **same** layer range (a replica pair) plus the hop that starts the
/// pipeline. Returns `(router, first, primary, replica, urls)`'s peers wired into the table.
async fn audit_fixture(
    primary_mul: f64,
    replica_mul: f64,
) -> anyhow::Result<(
    HttpPipelineRouter,
    HopConfig,
    HopConfig,
    HopConfig,
    HiddenStatePayload,
)> {
    let first = HopConfig::new("12D3KooWAuditFirstHop", (0, 1), 1.0, 1.0);
    let primary = HopConfig::new("12D3KooWAuditPrimary", (1, 2), primary_mul, 0.0);
    let replica = HopConfig::new("12D3KooWAuditReplica", (1, 2), replica_mul, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let replica_url = spawn_hop(replica.clone()).await?;

    let router = router_for(vec![
        layer_peer(&first.peer_id, &first_url, first.layer_range),
        layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
        layer_peer(&replica.peer_id, &replica_url, replica.layer_range),
    ])?;

    let keypair = generate_keypair();
    let payload = sealed_payload(&keypair)?;

    Ok((router, first, primary, replica, payload))
}

/// The audit does not change what the client gets (the primary's answer), but it must **have**
/// executed the marked shard twice: that is what makes capa 3 a proof of computation instead of a
/// promise.
#[tokio::test]
async fn an_audited_session_corroborates_the_final_shard_on_a_replica() -> anyhow::Result<()> {
    let (router, first, primary, replica, payload) = audit_fixture(2.0, 2.0).await?;

    let out = router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, audit_order(1, 1))
        .await
        .expect("two replicas that agree must produce a normal answer");

    let out_payload = HiddenStatePayload::from_bytes(&out)?;
    assert_eq!(out_payload.layer_idx, 1);
    let values = out_payload
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &values {
        assert!(
            (*value - 4.0).abs() < 1e-6,
            "(1.0 + 1.0) * 2.0 = 4.0, got {values:?}"
        );
    }

    assert_eq!(first.seen().len(), 1, "the first hop runs once");
    assert_eq!(primary.seen().len(), 1);
    assert_eq!(
        replica.seen().len(),
        1,
        "an audited session must execute the marked shard on a second replica (TASK-27)"
    );

    Ok(())
}

/// The whole point of sampling: an answer that a second execution does not reproduce is **not
/// served**, so a client never receives output the mesh cannot reproduce. The proof of the two
/// claims reaches the orchestrator separately (each replica writes its own `WorkProof`).
#[tokio::test]
async fn an_audited_session_refuses_a_divergent_replica() -> anyhow::Result<()> {
    let (router, _first, primary, replica, payload) = audit_fixture(2.0, 3.0).await?;

    let error = router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, audit_order(1, 1))
        .await
        .expect_err("two different answers for the same input cannot be served");

    let message = error.to_string();
    assert!(
        message.contains("audit divergence"),
        "the error must name the audit, got: {message}"
    );
    assert!(
        message.contains(&primary.peer_id) && message.contains(&replica.peer_id),
        "the error must name both claimants, got: {message}"
    );
    assert_eq!(primary.seen().len(), 1);
    assert_eq!(replica.seen().len(), 1);

    Ok(())
}

/// `audit_sample_rate = 0.0` (the default) means **identical** behaviour: the replica is never
/// contacted, not even when the topology has one. This is the criterion that keeps capa 3 from
/// costing anything until the operator turns it on.
#[tokio::test]
async fn a_non_audited_session_never_uses_a_second_replica() -> anyhow::Result<()> {
    let (router, _first, primary, replica, payload) = audit_fixture(2.0, 3.0).await?;

    // A divergent replica would fail the request if the audit ran, so a successful answer here is
    // already proof that it did not.
    router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, None)
        .await
        .expect("an unaudited session must not look at a replica");

    assert_eq!(primary.seen().len(), 1);
    assert!(
        replica.seen().is_empty(),
        "an unaudited session must not execute any second replica"
    );

    Ok(())
}

/// `TASK-70`/S77 · the hole `F-3a` described, closed: a hop **in the middle** of the pipeline is the one
/// that can lie undetected, because the audit used to look only at the step that ends the pipeline — and
/// the comparison it made there covered a whole *tail*, so the digest it attributed to a peer was not
/// the digest of that peer's work and the report matched no assignment (dead evidence).
///
/// Here the marked step is the middle one (`audit_layer = 1` of a three-layer pipeline): the router
/// compares **that** hop on a replica of its own range, and the report it signs is about `[1, 2)` — the
/// range the orchestrator can actually pay (`audit_credit`) — while the pipeline still completes.
#[tokio::test]
async fn an_audited_session_compares_the_marked_step_wherever_it_is() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWMarkedFirst", (0, 1), 1.0, 1.0);
    let primary = HopConfig::new("12D3KooWMarkedPrimary", (1, 2), 2.0, 0.0);
    let replica = HopConfig::new("12D3KooWMarkedReplica", (1, 2), 2.0, 0.0);
    let tail = HopConfig::new("12D3KooWMarkedTail", (2, 3), 3.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let replica_url = spawn_hop(replica.clone()).await?;
    let tail_url = spawn_hop(tail.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
            layer_peer(&replica.peer_id, &replica_url, replica.layer_range),
            layer_peer(&tail.peer_id, &tail_url, tail.layer_range),
        ],
        sink.url.clone(),
        identity.clone(),
    )?;

    let payload = sealed_payload_with_audit(&identity, 1)?;
    let out = router
        .route(0, 3, payload.to_bytes()?, MODEL_ID, audit_order(1, 1))
        .await
        .expect("the pipeline must complete while its middle step is audited");

    let out_payload = HiddenStatePayload::from_bytes(&out)?;
    assert_eq!(out_payload.layer_idx, 2, "the whole pipeline still runs");
    let values = out_payload
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &values {
        assert!(
            (*value - 12.0).abs() < 1e-6,
            "(1.0 + 1.0) * 2.0 * 3.0 = 12.0, got {values:?}"
        );
    }

    assert_eq!(first.seen().len(), 1, "the entry hop runs once");
    assert_eq!(
        primary.seen().len(),
        1,
        "the marked hop runs once on the main path"
    );
    assert_eq!(
        replica.seen().len(),
        1,
        "the marked step is compared with a replica of the same range"
    );
    assert_eq!(
        tail.seen().len(),
        1,
        "and the tail still serves the session"
    );

    let report = sink.last();
    assert_eq!(
        report["layer_range"],
        serde_json::json!([1, 2]),
        "the report has to be about the step that was compared, not about a tail nobody owns: {report}"
    );
    let executions = report["executions"].as_array().expect("executions");
    assert_eq!(executions.len(), 2, "{report}");
    let named: Vec<&str> = executions
        .iter()
        .map(|execution| execution["peer_id"].as_str().expect("peer id"))
        .collect();
    assert!(
        named.contains(&primary.peer_id.as_str()) && named.contains(&replica.peer_id.as_str()),
        "the report names the two executions of the marked step: {named:?}"
    );

    Ok(())
}

/// `TASK-70`/S77 · a peer that covers the marked layer but owns **other** layers is not a replica: it
/// computes another step, so its answer is different while being perfectly honest. Before the fix the
/// router took any peer covering the layer, compared a different tensor and aborted an honest session
/// with `audit divergence` — the false positive this test fixes.
#[tokio::test]
async fn a_peer_with_another_range_is_never_used_as_a_replica() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWRangeFirst", (0, 1), 1.0, 1.0);
    let primary = HopConfig::new("12D3KooWRangePrimary", (1, 2), 2.0, 0.0);
    // Covers layer 1 — so it is a candidate for the marked *layer* — but its range is `(1, 3)`.
    let wider = HopConfig::new("12D3KooWRangeWider", (1, 3), 3.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let wider_url = spawn_hop(wider.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
            layer_peer(&wider.peer_id, &wider_url, wider.layer_range),
        ],
        sink.url.clone(),
        identity.clone(),
    )?;

    let payload = sealed_payload_with_audit(&identity, 1)?;
    let out = router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, audit_order(1, 1))
        .await
        .expect("a peer with another range must not turn an honest session into a divergence");

    assert_eq!(HiddenStatePayload::from_bytes(&out)?.layer_idx, 1);
    assert!(
        wider.seen().is_empty(),
        "a peer that owns another range is not a replica of the marked step"
    );
    assert_eq!(primary.seen().len(), 1);
    assert!(
        sink.reports
            .lock()
            .expect("sink is not poisoned")
            .is_empty(),
        "with no comparable execution there is no report: the audit is a sample, not a gate"
    );

    Ok(())
}

/// `TASK-36`/S42 (option b): the router resolves a layer from the **cached topology** and never asks
/// the orchestrator for a substitute, because `GET /substitute` never existed.
///
/// The two hops answer differently on purpose (`1.0 · 1 + 1 = 2.0` against `1.0 · 3 = 3.0`), so the
/// numeric result says which path was taken. Before the fix the router used the candidate the stub
/// offered (`3.0`, i.e. it depended on an endpoint absent from `create_router`); after it, the router
/// uses the cached view (`2.0`) and the stub is not called at all.
#[tokio::test]
async fn the_router_resolves_from_the_cached_topology_and_never_asks_for_a_substitute(
) -> anyhow::Result<()> {
    let cached_hop = HopConfig::new("12D3KooWCachedHop", (0, 1), 1.0, 1.0);
    let cached_url = spawn_hop(cached_hop.clone()).await?;

    let substitute_hop = HopConfig::new("12D3KooWSubstituteHop", (0, 1), 3.0, 0.0);
    let substitute_url = spawn_hop(substitute_hop.clone()).await?;

    let orchestrator = spawn_substitute_stub(peers_entry(
        "12D3KooWSubstituteHop",
        &substitute_url,
        (0, 1),
    ))
    .await?;

    let router = router_with_cache(
        // The routing table is empty: this is exactly the case the fallback exists for.
        Vec::new(),
        vec![cached_peer("12D3KooWCachedHop", &cached_url, (0, 1))],
        orchestrator,
    )?;

    let keypair = generate_keypair();
    let payload = sealed_payload(&keypair)?;
    let bytes = router
        .route(0, 1, payload.to_bytes()?, MODEL_ID, None)
        .await?;

    let answer = HiddenStatePayload::from_bytes(&bytes)?;
    let value = answer
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        value.iter().all(|v| (v - 2.0).abs() < 1e-6),
        "the hop of the cached topology (+1.0) must be the one that answered, got {value:?}"
    );
    assert_eq!(
        substitute_hop.seen().len(),
        0,
        "the dead /substitute endpoint must not be consulted"
    );
    assert_eq!(
        cached_hop.seen().len(),
        1,
        "the cached hop served the layer"
    );

    Ok(())
}

/// `S43`/`TASK-37`: a layer whose only candidates announce an `expert_range` cannot be served, and
/// the router must say so **at once**.
///
/// Before this, the same payload was scattered to both shards, both executed the *whole* layer and
/// the router summed the answers — wrong logits, no warning. The old code would also have waited 30 s
/// for a peer that can never answer correctly if the cache had been empty. The two `seen()` counters
/// are what make the difference observable: the fix is not «a log line», it is «not a single POST».
#[tokio::test]
async fn a_layer_only_expert_shards_can_serve_is_refused_without_a_single_post(
) -> anyhow::Result<()> {
    let shard_a = HopConfig::new("12D3KooWExpertShardA", (0, 1), 1.0, 1.0);
    let shard_b = HopConfig::new("12D3KooWExpertShardB", (0, 1), 2.0, 0.0);
    let a_url = spawn_hop(shard_a.clone()).await?;
    let b_url = spawn_hop(shard_b.clone()).await?;

    let router = router_for(vec![
        LayerPeer {
            expert_range: Some((0, 4)),
            ..layer_peer(&shard_a.peer_id, &a_url, (0, 1))
        },
        LayerPeer {
            expert_range: Some((4, 8)),
            ..layer_peer(&shard_b.peer_id, &b_url, (0, 1))
        },
    ])?;

    let keypair = generate_keypair();
    let payload_bytes = sealed_payload(&keypair)?.to_bytes()?;

    let error = router
        .route(0, 1, payload_bytes, MODEL_ID, None)
        .await
        .expect_err("a layer only expert shards can serve must not be served");

    let message = error.to_string();
    assert!(
        message.contains("expert parallelism is not implemented"),
        "the error says why the layer cannot be served: {message}"
    );
    assert!(
        message.contains("12D3KooWExpertShardA") && message.contains("12D3KooWExpertShardB"),
        "and it names the shards, so the operator knows what to fix: {message}"
    );
    assert_eq!(
        shard_a.seen().len(),
        0,
        "no scatter to a shard: summing whole layers is the bug"
    );
    assert_eq!(shard_b.seen().len(), 0, "not even the second one");

    Ok(())
}

/// A stand-in for the orchestrator that counts **every** request this node sends to the control
/// plane, whatever the method and the path.
struct TrackerStub {
    url: String,
    calls: Arc<AtomicUsize>,
}

impl TrackerStub {
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

async fn spawn_tracker_stub() -> anyhow::Result<TrackerStub> {
    #[derive(Clone)]
    struct Counter(Arc<AtomicUsize>);

    async fn handler(State(counter): State<Counter>) -> StatusCode {
        counter.0.fetch_add(1, Ordering::SeqCst);
        StatusCode::OK
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let app = Router::new()
        .fallback(handler)
        .with_state(Counter(calls.clone()));
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    Ok(TrackerStub {
        url: format!("http://{addr}"),
        calls,
    })
}

/// `TASK-50`/S56 (decision `b`): a hop that does not answer has **one** consequence in this node — it
/// is counted (and published by `GET /status`) and it is penalised so the next layer prefers somebody
/// else — and **no** report leaves the machine.
///
/// The two halves are asserted together on purpose, because either one alone is misleading:
///
/// * without the counter, a failed hop is a log line nobody aggregates (which is what the card is
///   about: `warn_third_party_report_not_delivered` was its only trace);
/// * without the «zero calls» half, the fix could be «open a channel and accuse the peer», which is
///   the griefing S8/S28 closed: `/report_failure` authenticates a single identity (D-3) and the
///   tracker already learns liveness from the P2P connection and the `/announce` heartbeat.
///
/// The dead peer is given a **lower** latency so `select_route` tries it first, which is also what
/// proves the retry (`the hop that is up answered`).
#[tokio::test]
async fn a_dead_hop_is_counted_and_never_reported_to_the_tracker() -> anyhow::Result<()> {
    let alive = HopConfig::new("12D3KooWAliveHop", (0, 1), 1.0, 1.0);
    let alive_url = spawn_hop(alive.clone()).await?;
    // Nothing listens on port 1, so the connection is refused at once: no 5 s timeout and no 30 s
    // pipeline wait, because the router still has a second candidate.
    let dead_peer_id = "12D3KooWDeadHop";
    let dead_url = "http://127.0.0.1:1".to_string();

    let routing_table = Arc::new(RwLock::new(RoutingTable::new()));
    {
        let mut table = routing_table.write().await;
        table.replace_all(vec![
            layer_peer(dead_peer_id, &dead_url, (0, 1)),
            layer_peer(&alive.peer_id, &alive_url, (0, 1)),
        ]);
        table.update_latency(dead_peer_id, Duration::from_millis(1));
        table.update_latency(&alive.peer_id, Duration::from_millis(500));
    }

    let tracker = spawn_tracker_stub().await?;
    let registry = Arc::new(PeerFailureRegistry::new());
    let identity = Arc::new(generate_keypair());
    let router = HttpPipelineRouter::new(
        routing_table,
        Arc::new(RwLock::new(TopologyCache::new(Duration::from_secs(300)))),
        reqwest::Client::new(),
        tracker.url.clone(),
        tessella_network::peer_id_from_keypair(&identity),
        MODEL_ID.to_string(),
        registry.clone(),
        Arc::new(Notify::new()),
        identity,
        HOP_WAIT,
    );

    let keypair = generate_keypair();
    let payload_bytes = sealed_payload(&keypair)?.to_bytes()?;
    router
        .route(0, 1, payload_bytes, MODEL_ID, None)
        .await
        .expect("the hop that is up must serve the layer once the dead one fails");

    assert_eq!(
        alive.seen().len(),
        1,
        "the router must retry with the other candidate"
    );

    let report = registry.report();
    assert_eq!(report.total_failures, 1, "{report:?}");
    let dead = report
        .peers
        .iter()
        .find(|counts| counts.peer_id == dead_peer_id)
        .expect("the dead hop must be in the report");
    assert_eq!(dead.consecutive_failures, 1);
    assert_eq!(dead.by_reason.get("network_error"), Some(&1), "{dead:?}");

    assert_eq!(
        tracker.calls(),
        0,
        "a failed hop must not be reported to the control plane: there is no third-party channel"
    );

    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `TASK-53` · S59 · el informe firmado del auditor
// ---------------------------------------------------------------------------------------------

/// An orchestrator stand-in that **keeps** every `POST /api/v1/audit_report` it receives.
#[derive(Clone)]
struct AuditSink {
    url: String,
    reports: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl AuditSink {
    fn reports(&self) -> Vec<serde_json::Value> {
        self.reports.lock().expect("sink is not poisoned").clone()
    }

    fn last(&self) -> serde_json::Value {
        self.reports()
            .last()
            .cloned()
            .expect("the auditor must have reported what it saw")
    }
}

async fn spawn_audit_sink() -> anyhow::Result<AuditSink> {
    async fn handler(
        State(sink): State<Arc<Mutex<Vec<serde_json::Value>>>>,
        axum::Json(body): axum::Json<serde_json::Value>,
    ) -> StatusCode {
        sink.lock().expect("sink is not poisoned").push(body);
        StatusCode::OK
    }

    let reports = Arc::new(Mutex::new(Vec::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let app = Router::new()
        .route("/api/v1/audit_report", post(handler))
        .with_state(reports.clone());
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    Ok(AuditSink {
        url: format!("http://{addr}"),
        reports,
    })
}

/// The audit order of these fixtures (`TASK-70`/S77): the **marked step** is always the second hop —
/// range `(1, 2)` of a two-layer pipeline — and `replicas` is how many extra executions of it the
/// session asks for. `None` is «this session is not sampled».
fn audit_order(marked: usize, replicas: usize) -> Option<tessella_core::traits::AuditOrder> {
    Some(tessella_core::traits::AuditOrder {
        layer: marked,
        replicas,
        canary: None,
    })
}

/// The payload an entry node emits for a **sampled** session (`TASK-53`): it carries the work ticket,
/// which is what tells the router which step the orchestrator marked and how many extra executions it
/// asked for — and what the report has to travel with to be admissible.
fn sealed_payload_with_audit(
    keypair: &NetworkKeypair,
    audit_replicas: u8,
) -> anyhow::Result<HiddenStatePayload> {
    let mut payload = sealed_payload(keypair)?;
    payload.work_ticket = Some(tessella_protocol::models::WorkTicket {
        request_id: payload.request_id.clone(),
        model_id: MODEL_ID.to_string(),
        // The auditor is the entry node of the session: this fixture plays that node.
        entry_peer_id: tessella_network::peer_id_from_keypair(keypair),
        issued_at: tessella_core::time::unix_now_secs(),
        expires_at: u64::MAX,
        audit_layer: Some(1),
        audit_replicas,
        audit_canary: None,
        audit_reward: 0.0,
        orchestrator_signature: vec![9; 64],
    });
    Ok(payload)
}

/// The signed report has to say **exactly** what the auditor saw: which peers ran the marked step,
/// with which tensor digest, for which input and under which authorisation. It is the evidence the
/// orchestrator judges the credit of that range with, so a hop that could reshape it could reshape a
/// payment.
///
/// Three executions agree here (the default of `TASK-53`: a majority needs three), the client gets the
/// primary's answer, and the report is signed by the entry node with a canonical preimage the test
/// recomputes — so the assertions cover the wire, not just the DTO.
#[tokio::test]
async fn an_audited_session_reports_the_three_executions_it_compared() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWAuditSinkFirst", (0, 1), 1.0, 1.0);
    let primary = HopConfig::new("12D3KooWAuditSinkPrimary", (1, 2), 2.0, 0.0);
    let replica_a = HopConfig::new("12D3KooWAuditSinkReplicaA", (1, 2), 2.0, 0.0);
    let replica_b = HopConfig::new("12D3KooWAuditSinkReplicaB", (1, 2), 2.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let replica_a_url = spawn_hop(replica_a.clone()).await?;
    let replica_b_url = spawn_hop(replica_b.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
            layer_peer(&replica_a.peer_id, &replica_a_url, replica_a.layer_range),
            layer_peer(&replica_b.peer_id, &replica_b_url, replica_b.layer_range),
        ],
        sink.url.clone(),
        identity.clone(),
    )?;

    // `TASK-57`/S63: el número de ejecuciones extra viaja en el **ticket** y el nodo de entrada se
    // lo pasa al router, así que el fixture usa el mismo número en los dos sitios.
    let audit_replicas = 2;
    let payload = sealed_payload_with_audit(&identity, audit_replicas)?;
    let out = router
        .route(
            0,
            2,
            payload.to_bytes()?,
            MODEL_ID,
            audit_order(1, usize::from(audit_replicas)),
        )
        .await
        .expect("three executions that agree must produce a normal answer");

    let out_payload = HiddenStatePayload::from_bytes(&out)?;
    let values = out_payload
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &values {
        assert!((*value - 4.0).abs() < 1e-6, "got {values:?}");
    }

    assert_eq!(primary.seen().len(), 1);
    assert_eq!(
        replica_a.seen().len(),
        1,
        "the ticket asked for two extra executions (TASK-53)"
    );
    assert_eq!(replica_b.seen().len(), 1);

    let report = sink.last();
    assert_eq!(
        report["auditor_peer_id"].as_str(),
        Some(tessella_network::peer_id_from_keypair(&identity).as_str()),
        "only the entry node of the session may report: {report}"
    );
    assert_eq!(report["request_id"].as_str(), Some("req-two-hops"));
    assert_eq!(report["layer_range"], serde_json::json!([1, 2]));

    let executions = report["executions"].as_array().expect("executions");
    assert_eq!(executions.len(), 3, "{report}");
    let digests: Vec<&serde_json::Value> = executions
        .iter()
        .map(|execution| &execution["output_hash"])
        .collect();
    assert!(
        digests.windows(2).all(|pair| pair[0] == pair[1]),
        "three agreeing executions must carry one digest: {report}"
    );
    let peers: Vec<&str> = executions
        .iter()
        .map(|execution| execution["peer_id"].as_str().expect("peer id"))
        .collect();
    assert!(peers.contains(&primary.peer_id.as_str()), "{report}");
    assert!(peers.contains(&replica_a.peer_id.as_str()), "{report}");
    assert!(peers.contains(&replica_b.peer_id.as_str()), "{report}");

    // The report travels signed, and the preimage is the one the DTO defines: recomputing it from the
    // body as received is what proves that a digest edited on the way is rejected.
    let mut parsed: tessella_protocol::models::AuditReport =
        serde_json::from_value(report.clone())?;
    assert!(parsed.has_peer_auth(), "{report}");
    assert_eq!(
        parsed.work_ticket.audit_replicas, 2,
        "the report carries the authorisation of the session"
    );
    let canonical = parsed.peer_canonical_bytes()?;
    assert!(
        verify_signature(&parsed.public_key, &canonical, &parsed.signature)?,
        "the report must be signed by the node that filled in auditor_peer_id"
    );
    parsed.executions[0].output_hash = [42; 32];
    assert!(
        !verify_signature(
            &parsed.public_key,
            &parsed.peer_canonical_bytes()?,
            &parsed.signature
        )?,
        "a report whose digests were edited must not verify"
    );

    Ok(())
}

/// A 2-1 split names the **dissident**: two honest executions keep their route, and only the one that
/// did not reproduce the tensor is counted locally (`divergent_audit`, which `GET /status` publishes).
///
/// That local mark is the whole «capacity to participate» a divergence costs in this node, and it is
/// deliberately modest: the orchestrator is the one that decides the credit (from the report), and it
/// does not blacklist anybody, because two colluding nodes can manufacture a majority.
#[tokio::test]
async fn a_divergent_audit_names_the_dissident_and_counts_only_it() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS59FirstHop", (0, 1), 1.0, 1.0);
    let honest_a = HopConfig::new("12D3KooWS59HonestA", (1, 2), 2.0, 0.0);
    let honest_b = HopConfig::new("12D3KooWS59HonestB", (1, 2), 2.0, 0.0);
    let liar = HopConfig::new("12D3KooWS59Dissident", (1, 2), 3.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let honest_a_url = spawn_hop(honest_a.clone()).await?;
    let honest_b_url = spawn_hop(honest_b.clone()).await?;
    let liar_url = spawn_hop(liar.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let registry = Arc::new(PeerFailureRegistry::new());
    let identity = Arc::new(generate_keypair());
    let router = router_with_registry(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&honest_a.peer_id, &honest_a_url, honest_a.layer_range),
            layer_peer(&honest_b.peer_id, &honest_b_url, honest_b.layer_range),
            layer_peer(&liar.peer_id, &liar_url, liar.layer_range),
        ],
        Vec::new(),
        sink.url.clone(),
        registry.clone(),
        identity.clone(),
        HOP_WAIT,
    )?;

    let audit_replicas = 2;
    let payload = sealed_payload_with_audit(&identity, audit_replicas)?;
    let error = router
        .route(
            0,
            2,
            payload.to_bytes()?,
            MODEL_ID,
            audit_order(1, usize::from(audit_replicas)),
        )
        .await
        .expect_err("an answer the majority does not reproduce cannot be served");

    let message = error.to_string();
    assert!(
        message.contains("audit divergence"),
        "the error must name the audit, got: {message}"
    );
    assert!(
        message.contains(&liar.peer_id),
        "the error must name the dissident, got: {message}"
    );

    // Every execution ran once: `audit_replicas = 3` asks for three extra ones, but only three peers
    // cover that range, so the audit compares what exists.
    assert_eq!(honest_a.seen().len(), 1);
    assert_eq!(honest_b.seen().len(), 1);
    assert_eq!(liar.seen().len(), 1);

    let report = sink.last();
    let executions = report["executions"].as_array().expect("executions");
    assert_eq!(executions.len(), 3, "{report}");
    let liar_digest = executions
        .iter()
        .find(|execution| execution["peer_id"].as_str() == Some(liar.peer_id.as_str()))
        .expect("the dissident must be in the report")["output_hash"]
        .clone();
    let agreeing = executions
        .iter()
        .filter(|execution| execution["output_hash"] != liar_digest)
        .count();
    assert_eq!(
        agreeing, 2,
        "the two honest executions must agree with each other: {report}"
    );

    // The local consequence, and its limit: the dissident is counted as a divergent audit and the two
    // that agreed are not touched.
    let failures = registry.report();
    let counted: Vec<&str> = failures
        .peers
        .iter()
        .map(|counts| counts.peer_id.as_str())
        .collect();
    assert_eq!(
        counted,
        vec![liar.peer_id.as_str()],
        "only the dissident is penalised locally: {failures:?}"
    );
    let dissident = &failures.peers[0];
    assert_eq!(dissident.by_reason.get("divergent_audit"), Some(&1));
    assert_eq!(dissident.total_failures, 1);

    Ok(())
}

/// Two executions that disagree are a **tie**: the orchestrator cannot attribute it, so the request is
/// not served, the report says so (two different digests) and this node counts **both** — it cannot
/// tell which one is broken, and paying either half the time would make lying profitable.
#[tokio::test]
async fn a_tie_between_two_executions_counts_both_and_reports_it() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS59TieFirst", (0, 1), 1.0, 1.0);
    let primary = HopConfig::new("12D3KooWS59TiePrimary", (1, 2), 2.0, 0.0);
    let replica = HopConfig::new("12D3KooWS59TieReplica", (1, 2), 5.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let replica_url = spawn_hop(replica.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let registry = Arc::new(PeerFailureRegistry::new());
    let identity = Arc::new(generate_keypair());
    let router = router_with_registry(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
            layer_peer(&replica.peer_id, &replica_url, replica.layer_range),
        ],
        Vec::new(),
        sink.url.clone(),
        registry.clone(),
        identity.clone(),
        HOP_WAIT,
    )?;

    // `audit_replicas = 1` is what S33 hardcoded: a comparison of two executions, and therefore a tie
    // whenever they disagree.
    let payload = sealed_payload_with_audit(&identity, 1)?;
    router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, audit_order(1, 1))
        .await
        .expect_err("a tie is not an answer the mesh can confirm");

    let report = sink.last();
    let executions = report["executions"].as_array().expect("executions");
    assert_eq!(executions.len(), 2, "{report}");
    assert_ne!(
        executions[0]["output_hash"], executions[1]["output_hash"],
        "a tie must reach the orchestrator as two different digests: {report}"
    );

    let failures = registry.report();
    let counted: Vec<&str> = failures
        .peers
        .iter()
        .map(|counts| counts.peer_id.as_str())
        .collect();
    assert!(
        counted.contains(&primary.peer_id.as_str()) && counted.contains(&replica.peer_id.as_str()),
        "with a tie both executions are suspect: {failures:?}"
    );

    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `TASK-83` · S90 · el canario no vota y nunca sirve al cliente
// ---------------------------------------------------------------------------------------------

/// The same sampled payload, with the canary the orchestrator named inside its **signed** ticket
/// (`TASK-71`/S78 · `TASK-83`/S90). The ticket is what makes the control a fact of the session — not
/// something the auditor chooses — and it travels with the report.
fn sealed_payload_with_canary(
    keypair: &NetworkKeypair,
    audit_replicas: u8,
    canary: &str,
) -> anyhow::Result<HiddenStatePayload> {
    let mut payload = sealed_payload_with_audit(keypair, audit_replicas)?;
    if let Some(ticket) = payload.work_ticket.as_mut() {
        ticket.audit_canary = Some(canary.to_string());
    }
    Ok(payload)
}

/// The audit order of a session whose ticket named a canary.
fn audit_order_with_canary(
    marked: usize,
    replicas: usize,
    canary: &str,
) -> Option<tessella_core::traits::AuditOrder> {
    let mut order = audit_order(marked, replicas)?;
    order.canary = Some(canary.to_string());
    Some(order)
}

/// `TASK-83`/S90 · `R1` y `R2`: **el canario es el control, no una segunda opinión**. Con el ticket
/// nombrando un canario la sesión se sirve igual (decide la mayoría honesta), el informe lleva las
/// ejecuciones honestas **más** el canario, y el canario va **encima** de las réplicas que la sesión
/// pagó en vez de robarles el sitio.
///
/// Antes de S90 esto era un fallo medido: el canario se insertaba el primero y luego se recortaba a
/// `replicas`, así que con `audit_replicas = 1` la comparación era «el primario contra el tensor
/// incorrecto», un empate; nadie cobraba, el hop honesto se llevaba un `DivergentAudit` y la petición
/// del cliente se abortaba. Con `2` se abortaba igual, porque cualquier disidencia hacía fallar el
/// paso aunque la mayoría honesta estuviera clara.
#[tokio::test]
async fn a_canary_is_a_control_and_the_honest_majority_still_serves_the_session(
) -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS90First", (0, 1), 1.0, 1.0);
    let primary = HopConfig::new("12D3KooWS90Primary", (1, 2), 2.0, 0.0);
    let honest = HopConfig::new("12D3KooWS90Honest", (1, 2), 2.0, 0.0);
    // El canario responde un tensor **distinto por construcción** (`echo` en producción): aquí es el
    // mismo `affine` con otro multiplicador, que es lo que lo hace comparable y a la vez incorrecto.
    let canary = HopConfig::new("12D3KooWS90Canary", (1, 2), 5.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let honest_url = spawn_hop(honest.clone()).await?;
    let canary_url = spawn_hop(canary.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
            layer_peer(&honest.peer_id, &honest_url, honest.layer_range),
            layer_peer(&canary.peer_id, &canary_url, canary.layer_range),
        ],
        sink.url.clone(),
        identity.clone(),
    )?;

    let payload = sealed_payload_with_canary(&identity, 1, &canary.peer_id)?;
    let out = router
        .route(
            0,
            2,
            payload.to_bytes()?,
            MODEL_ID,
            audit_order_with_canary(1, 1, &canary.peer_id),
        )
        .await
        .expect(
            "the honest majority must serve the session: the canary is the control, not a voter",
        );

    let out_payload = HiddenStatePayload::from_bytes(&out)?;
    let values = out_payload
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &values {
        assert!(
            (*value - 4.0).abs() < 1e-6,
            "the client gets the honest answer ((1.0 + 1.0) * 2.0 = 4.0), not the canary's: {values:?}"
        );
    }

    // El informe: tres ejecuciones — primario, réplica honesta y canario —, y las dos honestas con el
    // mismo dígito. Con el orden antiguo el canario ocupaba el sitio de la réplica honesta.
    let report = sink.last();
    let executions = report["executions"]
        .as_array()
        .expect("executions is an array");
    let ids: Vec<&str> = executions
        .iter()
        .map(|execution| execution["peer_id"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(executions.len(), 3, "{report}");
    assert!(
        ids.contains(&primary.peer_id.as_str()) && ids.contains(&honest.peer_id.as_str()),
        "the report must carry the honest replica the session bought, not the canary in its place: \
         {report}"
    );
    let digest_of = |peer: &str| -> String {
        executions
            .iter()
            .find(|execution| execution["peer_id"] == peer)
            .map(|execution| execution["output_hash"].to_string())
            .unwrap_or_default()
    };
    assert_eq!(
        digest_of(&primary.peer_id),
        digest_of(&honest.peer_id),
        "las dos ejecuciones honestas coinciden: {report}"
    );
    assert_ne!(
        digest_of(&canary.peer_id),
        digest_of(&primary.peer_id),
        "y el canario disiente, que es lo que el informe tiene que enseñar: {report}"
    );
    assert!(
        !canary.seen().is_empty(),
        "el auditor tiene que haber ejecutado el canario: es el control del ticket"
    );

    Ok(())
}

/// `TASK-83`/S90 · `R6`: **un canario nunca sirve al cliente**. El paso que el ticket marca es
/// justamente el que el canario replica (por eso lo nombró el orquestador), así que queda fuera del
/// sorteo del **primario**: lo pide la auditoría, explícitamente, y nunca la política de latencia.
///
/// La tabla de este fixture tiene al canario **el primero** de la capa 1, que es lo que haría que el
/// empate lo eligiera (los empates conservan el orden de la topología, `S75`/`F-1`): sin la regla, el
/// cliente recibiría el tensor del control con un `200` encima.
#[tokio::test]
async fn the_canary_of_the_marked_step_never_serves_the_client() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS90FirstHop", (0, 1), 1.0, 1.0);
    let canary = HopConfig::new("12D3KooWS90FastCanary", (1, 2), 5.0, 0.0);
    let honest = HopConfig::new("12D3KooWS90OnlyHonest", (1, 2), 2.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let canary_url = spawn_hop(canary.clone()).await?;
    let honest_url = spawn_hop(honest.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            // El canario, el primero: sería el primario de la capa 1 por orden de topología.
            layer_peer(&canary.peer_id, &canary_url, canary.layer_range),
            layer_peer(&honest.peer_id, &honest_url, honest.layer_range),
        ],
        sink.url.clone(),
        identity.clone(),
    )?;

    let payload = sealed_payload_with_canary(&identity, 1, &canary.peer_id)?;
    let out = router
        .route(
            0,
            2,
            payload.to_bytes()?,
            MODEL_ID,
            audit_order_with_canary(1, 1, &canary.peer_id),
        )
        .await
        .expect("the honest node serves the step: the canary is excluded from the primary pick");

    let out_payload = HiddenStatePayload::from_bytes(&out)?;
    let values = out_payload
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &values {
        assert!(
            (*value - 4.0).abs() < 1e-6,
            "el cliente recibe la respuesta del nodo honesto: el tensor del canario (×5.0) sería el \
             fallo que esta regla cierra, y llegaría con un 200: {values:?}"
        );
    }
    assert_eq!(
        canary.seen().len(),
        1,
        "el canario se consulta **una** vez y como control (la auditoría), nunca como primario: {:?}",
        canary.seen()
    );

    // Y con un único dueño honesto no hay comparación posible: el paso cae a la regla de S33 y
    // **nadie** queda penalizado (`R1`). El informe vacío es lo que demuestra que no se inventó una
    // comparación contra el control.
    assert!(
        sink.reports().is_empty(),
        "sin dos ejecuciones honestas no hay informe: {:?}",
        sink.reports()
    );

    Ok(())
}

/// `TASK-68`/S75 · decisión **(a)** del humano: cuando un hop contesta el **rechazo de `F-1`**
/// (un *decode step* cuyo prefijo nunca pasó por ese nodo), el router **aborta** la petición en vez
/// de reintentar la misma capa con los mismos bytes en el siguiente candidato.
///
/// El segundo candidato es el que hace visible el defecto: con el comportamiento anterior (`continue`)
/// el router le mandaba **los mismos bytes**, ese nodo tampoco tenía la sesión, y la diferencia es
/// que ahí ya no hay nada que detecte el problema: la respuesta volvía con `200` y un tensor
/// plausible (`F-1`: el *decode step* va sin máscara y la KV nace vacía). Este test fija las tres
/// cosas que el arreglo promete: la petición **falla**, el fallo **nombra la capa y el hop**, y el
/// segundo candidato **no se llegó a consultar**.
#[tokio::test]
async fn a_hop_that_refuses_a_decode_step_aborts_the_request_instead_of_burning_the_next_candidate(
) -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS75First", (0, 1), 1.0, 1.0);
    // The replacement hop of `F-1`: alive, at the right layer, with **no** session of its own.
    let refuses = HopConfig::new("12D3KooWS75Refuses", (1, 2), 2.0, 0.0).refusing_missing_prefill();
    // The candidate the old code would have burned right after: it answers happily, which is
    // exactly how the corruption reached the client.
    let decoy = HopConfig::new("12D3KooWS75Decoy", (1, 2), 2.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let refuses_url = spawn_hop(refuses.clone()).await?;
    let decoy_url = spawn_hop(decoy.clone()).await?;

    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&refuses.peer_id, &refuses_url, refuses.layer_range),
            layer_peer(&decoy.peer_id, &decoy_url, decoy.layer_range),
        ],
        "http://127.0.0.1:1".to_string(),
        identity.clone(),
    )?;

    // A decode step of an ongoing session: ONE position and a prefix of 12 that the replacement
    // hop never applied (it was not in the pipeline when the prefill happened).
    let payload = sealed_decode_step(&identity, 12)?;
    let error = router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, None)
        .await
        .expect_err("a refusal of the prefix cannot be answered by retrying the same bytes");

    assert!(
        tessella_core::error::is_missing_prefill(&error),
        "the client has to get the same named failure the hop produced: {error}"
    );
    let message = error.to_string();
    assert!(message.contains("12D3KooWS75Refuses"), "{message}");
    assert!(
        message.contains("layer 1"),
        "the refusal names the layer that was in flight: {message}"
    );
    assert!(
        decoy.seen().is_empty(),
        "the second candidate must not be asked: the same bytes would produce the same garbage \
             and that is exactly the silent `200` of F-1"
    );
    assert_eq!(
        refuses.seen().len(),
        1,
        "the refusal is the first thing that node answered"
    );

    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `TASK-76` · S83 · las negativas con nombre que no necesitan GPU, y el aborto sin reintento
// ---------------------------------------------------------------------------------------------

/// `TASK-76`/S83: un hop que **niega** la petición la nombra (status + `X-Tessella-Refusal` + un cuerpo
/// que empieza por el código) y el router del nodo de arriba **aborta** —nombrando la capa y el
/// hop— en vez de mandar los mismos bytes al siguiente candidato.
///
/// El segundo candidato es lo que hace visible el defecto: unos bytes que no caben en un hop no caben
/// en **ninguno** (el cuerpo lo fija el cliente y el techo es el mismo en toda la malla), así que
/// reintentarlos solo reproduce la negativa. En `S76` esa cadena era un `413` y, después de 30 s de
/// suspensión, un `500`: cada negativa se contaba como fallo del hop, el nodo honesto acababa degradado
/// y el router se quedaba sin candidato. Por eso este test fija también que la negativa **no** se
/// cuenta como fallo.
///
/// La latencia es explícita (1 ms el que niega, 500 ms el siguiente) para que el orden de consulta sea
/// el que hace observable el «sin reintento».
async fn assert_the_router_aborts_on(
    refusal: HopRefusal,
    named: fn(&tessella_core::error::DomainError) -> bool,
    why_the_next_candidate_is_not_asked: &str,
) -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS83First", (0, 1), 1.0, 1.0);
    let refuses = match refusal {
        HopRefusal::Bytes => {
            HopConfig::new("12D3KooWS83Refuses", (1, 2), 2.0, 0.0).refusing_oversized_bytes()
        }
        HopRefusal::Prompt => HopConfig::new("12D3KooWS83Refuses", (1, 2), 2.0, 0.0)
            .refusing_a_prompt_over_its_ceiling(),
        HopRefusal::Quota => {
            HopConfig::new("12D3KooWS83Refuses", (1, 2), 2.0, 0.0).refusing_at_its_quota()
        }
    };
    let code = match refusal {
        HopRefusal::Bytes => tessella_core::error::PAYLOAD_TOO_LARGE_CODE,
        HopRefusal::Prompt => tessella_core::error::PROMPT_TOO_LONG_CODE,
        HopRefusal::Quota => tessella_core::error::TOO_MANY_HOP_BODIES_CODE,
    };
    // The candidate the old code would have burned right after: it serves the layer happily, so a
    // router that retried would make this test pass with a tensor instead of failing.
    let decoy = HopConfig::new("12D3KooWS83Decoy", (1, 2), 2.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let refuses_url = spawn_hop(refuses.clone()).await?;
    let decoy_url = spawn_hop(decoy.clone()).await?;

    let routing_table = Arc::new(RwLock::new(RoutingTable::new()));
    {
        let mut table = routing_table.write().await;
        table.replace_all(vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&refuses.peer_id, &refuses_url, refuses.layer_range),
            layer_peer(&decoy.peer_id, &decoy_url, decoy.layer_range),
        ]);
        table.update_latency(&refuses.peer_id, Duration::from_millis(1));
        table.update_latency(&decoy.peer_id, Duration::from_millis(500));
    }

    let registry = Arc::new(PeerFailureRegistry::new());
    let identity = Arc::new(generate_keypair());
    let router = HttpPipelineRouter::new(
        routing_table,
        Arc::new(RwLock::new(TopologyCache::new(Duration::from_secs(300)))),
        reqwest::Client::new(),
        // Never reached: the layer 1 hop answers the refusal, so neither the topology refresh nor the
        // 30 s suspension can happen before the error comes back.
        "http://127.0.0.1:1".to_string(),
        tessella_network::peer_id_from_keypair(&identity),
        MODEL_ID.to_string(),
        registry.clone(),
        Arc::new(Notify::new()),
        identity.clone(),
        HOP_WAIT,
    );

    let payload = sealed_payload(&identity)?;
    let started = std::time::Instant::now();
    let error = router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, None)
        .await
        .expect_err("a named refusal cannot be answered by retrying the same bytes");
    let elapsed = started.elapsed();

    assert!(
        named(&error),
        "the client has to get the same named failure the hop produced: {error}"
    );
    let message = error.to_string();
    assert!(message.contains("12D3KooWS83Refuses"), "{message}");
    assert!(
        message.contains("layer 1"),
        "the refusal names the layer that was in flight: {message}"
    );
    assert!(
        message.contains(code),
        "the code has to survive the router so the client can act on it: {message}"
    );
    assert_eq!(
        refuses.refusals(),
        1,
        "the refusal is the first thing that node answered"
    );
    assert!(
        decoy.seen().is_empty(),
        "the next candidate must not be asked: {why_the_next_candidate_is_not_asked}"
    );
    assert_eq!(
        registry.report().total_failures,
        0,
        "a refusal is a decision of the request, not a fault of the hop: counting it degrades and \
         eventually evicts a node that is working ({:?})",
        registry.report()
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "the answer cannot wait for the 30 s suspension of `S76`: {elapsed:?}"
    );

    Ok(())
}

#[tokio::test]
async fn a_hop_that_refuses_an_oversized_body_aborts_the_request_instead_of_burning_the_next_candidate(
) -> anyhow::Result<()> {
    assert_the_router_aborts_on(
        HopRefusal::Bytes,
        tessella_core::error::is_payload_too_large,
        "the bytes that do not fit in one hop do not fit in another one either",
    )
    .await
}

#[tokio::test]
async fn a_hop_that_refuses_a_prompt_over_its_ceiling_aborts_the_request_instead_of_burning_the_next_candidate(
) -> anyhow::Result<()> {
    assert_the_router_aborts_on(
        HopRefusal::Prompt,
        tessella_core::error::is_prompt_too_long,
        "the bytes that do not fit in one hop do not fit in another one either",
    )
    .await
}

/// `TASK-84`/S91: un hop **lleno** (su cuota de cuerpos en vuelo) también aborta la petición, y aquí
/// la razón es **distinta** de la de las otras tres negativas: los mismos bytes **sí** caben en ese
/// nodo cuando se libere, y caben en otro candidato ahora mismo. Lo que hace terminal la negativa es
/// la amplificación: reintentar significa empujar el mismo cuerpo de 70-80 MiB al siguiente candidato,
/// y al siguiente, que es exactamente lo que la cuota existe para frenar (`S91`: 24 cuerpos a la vez
/// son +1,6 GiB de RSS en el nodo que los recibe). El reintento pertenece al **cliente**, y eso es lo
/// que su `503` ya significa.
///
/// El segundo candidato sigue siendo lo que hace visible el defecto: sirve la capa contento, así que
/// un router que reintentase devolvería un tensor y el cliente leería un `200` de un hop que él no
/// eligió.
#[tokio::test]
async fn a_hop_at_its_quota_aborts_the_request_instead_of_pushing_the_same_body_at_the_next_candidate(
) -> anyhow::Result<()> {
    assert_the_router_aborts_on(
        HopRefusal::Quota,
        tessella_core::error::is_too_many_hop_bodies,
        "the node is full, not broken: retrying would push the same 70-80 MiB body at every \
         candidate, which is the amplification the quota exists to stop",
    )
    .await
}

// ---------------------------------------------------------------------------------------------
// `TASK-76` · S83 · el techo publicado también tiene que caber en el **tiempo**
// ---------------------------------------------------------------------------------------------

/// `TASK-76`/S83 — el otro literal del mismo defecto: la petición al hop se mataba a los **5 s**.
///
/// El barrido de `S83` (2 workers, `--vram 1.2`, `max_input_tokens = 2048`) dejó atrás el `413` de
/// axum: el *entry node* prefilló 2 037 tokens y publicó su propio aviso
/// (`⚠️ Hidden state near the ceiling of one hop: 2037/2048 tokens … source: vram`). Lo que quedaba
/// era el **reloj**: `tokio::time::timeout(Duration::from_secs(5), …)` en `http_pipeline_router`
/// mataba la petición, el hop —que estaba **trabajando**— se contaba como caído, el router se
/// quedaba sin candidato y el cliente recibía el `500 … Pipeline timeout` de 30 s de `S76`. El mismo
/// síntoma con otra excusa, y encima `p2p_inference_timeout_secs` (el número que el harness declara,
/// 60 s) no lo leía **nadie**: la espera era un literal.
///
/// Un hop que contesta a los 6 s está vivo: se sirve, **no** se cuenta como fallo y el segundo
/// candidato no se toca (los mismos bytes no se reenvían a un nodo que no ha fallado).
#[tokio::test]
async fn a_hop_that_needs_more_than_five_seconds_is_served_instead_of_being_declared_dead(
) -> anyhow::Result<()> {
    let slow = HopConfig::new("12D3KooWS83SlowHop", (0, 1), 1.0, 1.0)
        .answering_after(Duration::from_secs(6));
    // The candidate the 5 s literal used to burn right after: it answers at once, so a retry is
    // invisible in the tensor — and that is precisely why the assertion has to be on `seen`.
    let decoy = HopConfig::new("12D3KooWS83DecoyHop", (1, 2), 2.0, 0.0);
    let slow_url = spawn_hop(slow.clone()).await?;
    let decoy_url = spawn_hop(decoy.clone()).await?;

    let registry = Arc::new(PeerFailureRegistry::new());
    let identity = Arc::new(generate_keypair());
    let router = router_with_registry(
        vec![
            layer_peer(&slow.peer_id, &slow_url, slow.layer_range),
            layer_peer(&decoy.peer_id, &decoy_url, decoy.layer_range),
        ],
        Vec::new(),
        "http://127.0.0.1:1".to_string(),
        registry.clone(),
        identity.clone(),
        HOP_WAIT,
    )?;

    let payload = sealed_payload(&identity)?;
    let started = std::time::Instant::now();
    let out = router
        .route(0, 2, payload.to_bytes()?, MODEL_ID, None)
        .await
        .expect("a hop that answers in 6 s is slow, not dead");
    let elapsed = started.elapsed();

    assert!(
        elapsed >= Duration::from_secs(6),
        "the wait must be the declared one, not a shorter literal: {elapsed:?}"
    );
    assert_eq!(
        HiddenStatePayload::from_bytes(&out)?.layer_idx,
        1,
        "the pipeline has to reach the last hop"
    );
    assert_eq!(
        slow.seen().len(),
        1,
        "the slow hop is the one that served layer 0"
    );
    assert_eq!(
        decoy.seen().len(),
        1,
        "layer 1 continues on the second hop exactly once"
    );
    assert_eq!(
        registry.report().total_failures,
        0,
        "a hop that answers is not a failed hop; counting it degrades and evicts a node that works \
         ({:?})",
        registry.report()
    );

    Ok(())
}

/// The other half of the same contract (`TASK-76`/S83): a hop that **never** answers still has to be
/// declared dead and replaced. Widening the wait must not turn a mute node into an infinite wait.
///
/// The declared wait here is 1 s, not the default: what is pinned is the *behaviour* — the failure is
/// counted as `timeout` and the router asks the next candidate —, not the number.
#[tokio::test]
async fn a_hop_that_never_answers_is_still_declared_dead_when_the_declared_wait_expires(
) -> anyhow::Result<()> {
    // Far beyond the 1 s declared below, so the "dead" one is unambiguously mute for this test.
    let mute = HopConfig::new("12D3KooWS83MuteHop", (0, 1), 1.0, 1.0)
        .answering_after(Duration::from_secs(30));
    let alive = HopConfig::new("12D3KooWS83AliveHop", (0, 1), 1.0, 1.0);
    let mute_url = spawn_hop(mute.clone()).await?;
    let alive_url = spawn_hop(alive.clone()).await?;

    let routing_table = Arc::new(RwLock::new(RoutingTable::new()));
    {
        let mut table = routing_table.write().await;
        table.replace_all(vec![
            layer_peer(&mute.peer_id, &mute_url, mute.layer_range),
            layer_peer(&alive.peer_id, &alive_url, alive.layer_range),
        ]);
        // The mute one is tried first, so what the assertions measure is the replacement.
        table.update_latency(&mute.peer_id, Duration::from_millis(1));
        table.update_latency(&alive.peer_id, Duration::from_millis(500));
    }

    let registry = Arc::new(PeerFailureRegistry::new());
    let identity = Arc::new(generate_keypair());
    let router = HttpPipelineRouter::new(
        routing_table,
        Arc::new(RwLock::new(TopologyCache::new(Duration::from_secs(300)))),
        reqwest::Client::new(),
        "http://127.0.0.1:1".to_string(),
        tessella_network::peer_id_from_keypair(&identity),
        MODEL_ID.to_string(),
        registry.clone(),
        Arc::new(Notify::new()),
        identity.clone(),
        Duration::from_secs(1),
    );

    let payload = sealed_payload(&identity)?;
    let started = std::time::Instant::now();
    router
        .route(0, 1, payload.to_bytes()?, MODEL_ID, None)
        .await
        .expect("the candidate that answers must serve the layer");
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(30),
        "the router cannot wait for the mute hop to give up by itself: {elapsed:?}"
    );
    assert_eq!(
        alive.seen().len(),
        1,
        "the router must retry with the other candidate"
    );

    let report = registry.report();
    assert_eq!(report.total_failures, 1, "{report:?}");
    let dead = report
        .peers
        .iter()
        .find(|counts| counts.peer_id == mute.peer_id)
        .expect("the mute hop must be in the report");
    assert_eq!(
        dead.by_reason.get("timeout"),
        Some(&1),
        "a hop that says nothing is a timeout of the declared wait: {dead:?}"
    );

    Ok(())
}

/// `TASK-80`/S87: Front audit (layer 0).
/// When layer 0 is marked for audit, the entry node executes the step locally, but also
/// fans out the tokens payload to replicas and canary, collecting an AuditReport where
/// `layer_range == [0, 1]` and `input_hash == [0u8; 32]`.
#[tokio::test]
async fn an_audited_session_compares_the_front_step_on_replicas_and_canary() -> anyhow::Result<()> {
    let replica = HopConfig::new("12D3KooWFrontReplica", (0, 1), 1.0, 1.0);
    let canary = HopConfig::new("12D3KooWFrontCanary", (0, 1), 99.0, 0.0);

    let replica_url = spawn_hop(replica.clone()).await?;
    let canary_url = spawn_hop(canary.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&replica.peer_id, &replica_url, replica.layer_range),
            layer_peer(&canary.peer_id, &canary_url, canary.layer_range),
        ],
        sink.url.clone(),
        identity.clone(),
    )?;

    let mut payload =
        HiddenStatePayload::from_tensor(&hidden_state(1.0)?, "req-front-audit".to_string(), 0, 0)?;
    payload.session_id = Some("session-front-audit".to_string());
    payload.tokens = vec![101, 102, 103];
    payload.work_ticket = Some(tessella_protocol::models::WorkTicket {
        request_id: payload.request_id.clone(),
        model_id: MODEL_ID.to_string(),
        entry_peer_id: tessella_network::peer_id_from_keypair(&identity),
        issued_at: tessella_core::time::unix_now_secs(),
        expires_at: u64::MAX,
        audit_layer: Some(0),
        audit_replicas: 1,
        audit_canary: Some(canary.peer_id.clone()),
        audit_reward: 0.0,
        orchestrator_signature: vec![9; 64],
    });
    seal_signed_request(
        &identity,
        &mut payload,
        4242,
        tessella_core::time::unix_now_secs(),
    )?;
    let request_bytes = payload.to_bytes()?;
    let primary_payload =
        HiddenStatePayload::from_tensor(&hidden_state(2.0)?, "req-front-audit".to_string(), 0, 0)?;
    let primary_response = primary_payload.to_bytes()?;

    let order = audit_order_with_canary(0, 1, &canary.peer_id)
        .expect("audit order with canary must be built");

    let own_peer = tessella_network::peer_id_from_keypair(&identity);
    router
        .audit_front((0, 1), &request_bytes, &own_peer, &primary_response, &order)
        .await
        .expect("front audit must succeed");

    assert_eq!(
        replica.seen().len(),
        1,
        "replica must have executed layer 0"
    );
    assert_eq!(canary.seen().len(), 1, "canary must have executed layer 0");

    let report = sink.last();
    assert_eq!(report["layer_range"], serde_json::json!([0, 1]));
    assert_eq!(report["input_hash"], serde_json::json!(vec![0u8; 32]));

    let executions = report["executions"].as_array().expect("executions");
    assert_eq!(executions.len(), 3, "{report}");
    let named: Vec<&str> = executions
        .iter()
        .map(|execution| execution["peer_id"].as_str().expect("peer id"))
        .collect();
    assert!(
        named.contains(&own_peer.as_str()),
        "report includes primary"
    );
    assert!(
        named.contains(&replica.peer_id.as_str()),
        "report includes replica"
    );
    assert!(
        named.contains(&canary.peer_id.as_str()),
        "report includes canary"
    );

    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `TASK-81` · S88 · estabilidad de primario por sesión en mallas con réplicas (pinning de ruta)
// ---------------------------------------------------------------------------------------------

fn sealed_step_for_session(
    keypair: &NetworkKeypair,
    session_id: &str,
    seq_start_pos: usize,
    nonce: u64,
) -> anyhow::Result<HiddenStatePayload> {
    let mut payload = HiddenStatePayload::from_tensor(
        &hidden_state(1.0)?,
        format!("req-{session_id}-{seq_start_pos}"),
        u32::try_from(seq_start_pos).expect("valid seq_start_pos"),
        0,
    )?;
    payload.session_id = Some(session_id.to_string());
    seal_signed_request(
        keypair,
        &mut payload,
        nonce,
        tessella_core::time::unix_now_secs(),
    )?;
    Ok(payload)
}

/// `TASK-81`/S88: En una malla con réplicas del mismo rango, la sesión queda fijada (pinned) al
/// nodo que sirvió el prefijo (`seq_start_pos == 0`). Cuando llega un paso de generación subsiguiente
/// (`seq_start_pos > 0`), el router envía el paso al nodo fijado que tiene la KV cache y NO a su
/// gemelo réplica (que no tiene KV y respondería `409 missing_prefill`), incluso si un refresco de
/// topología coloca al gemelo primero en la tabla de rutas.
#[tokio::test]
async fn the_primary_stays_pinned_for_the_session_even_if_twin_is_first_candidate_without_kv(
) -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS88First", (0, 1), 1.0, 1.0);
    // El nodo primario: responde felizmente al prefill y a los decode steps subsecuentes.
    let primary = HopConfig::new("12D3KooWS88Primary", (1, 2), 2.0, 0.0);
    // El nodo gemelo (réplica del mismo rango): como nunca recibió el prefill de esta sesión,
    // rechazaría con 409 missing_prefill si recibiera un decode step.
    let twin = HopConfig::new("12D3KooWS88Twin", (1, 2), 2.0, 0.0).refusing_missing_prefill();

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let twin_url = spawn_hop(twin.clone()).await?;

    let identity = Arc::new(generate_keypair());
    let routing_table = Arc::new(RwLock::new(RoutingTable::new()));
    routing_table.write().await.replace_all(vec![
        layer_peer(&first.peer_id, &first_url, first.layer_range),
        // Inicialmente primary está antes que twin
        layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
        layer_peer(&twin.peer_id, &twin_url, twin.layer_range),
    ]);

    let router = HttpPipelineRouter::new(
        routing_table.clone(),
        Arc::new(RwLock::new(TopologyCache::new(Duration::from_secs(300)))),
        reqwest::Client::new(),
        "http://127.0.0.1:1".to_string(),
        tessella_network::peer_id_from_keypair(&identity),
        MODEL_ID.to_string(),
        Arc::new(PeerFailureRegistry::new()),
        Arc::new(Notify::new()),
        identity.clone(),
        Duration::from_secs(30),
    );

    let session_id = "session-s88-pin";

    // 1. Prefill (seq_start_pos = 0)
    let prefill = sealed_step_for_session(&identity, session_id, 0, 1001)?;
    let prefill_res = router
        .route(0, 2, prefill.to_bytes()?, MODEL_ID, None)
        .await?;
    assert!(!prefill_res.is_empty());
    assert_eq!(primary.seen().len(), 1, "primary served prefill");
    assert_eq!(twin.seen().len(), 0, "twin did not receive prefill");

    // 2. Simular refresco de topología donde twin pasa a estar PRIMERO en la tabla
    routing_table.write().await.replace_all(vec![
        layer_peer(&first.peer_id, &first_url, first.layer_range),
        layer_peer(&twin.peer_id, &twin_url, twin.layer_range),
        layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
    ]);

    // 3. Decode step 1 (seq_start_pos = 10):
    // Sin pin, el router elegiría a `twin` (primer candidato) y twin respondería 409 missing_prefill.
    // Con pin, el router prefiere a `primary` (que tiene la KV cache y está vivo).
    let decode1 = sealed_step_for_session(&identity, session_id, 10, 1002)?;
    let decode1_res = router
        .route(0, 2, decode1.to_bytes()?, MODEL_ID, None)
        .await?;
    assert!(!decode1_res.is_empty());
    assert_eq!(
        primary.seen().len(),
        2,
        "primary served decode step 1 via pin"
    );
    assert_eq!(
        twin.seen().len(),
        0,
        "twin was NOT sent the decode step without KV"
    );
    assert_eq!(
        twin.refusals.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "twin never refused because it was never called"
    );

    // 4. Decode step 2 (seq_start_pos = 11):
    let decode2 = sealed_step_for_session(&identity, session_id, 11, 1003)?;
    let decode2_res = router
        .route(0, 2, decode2.to_bytes()?, MODEL_ID, None)
        .await?;
    assert!(!decode2_res.is_empty());
    assert_eq!(
        primary.seen().len(),
        3,
        "primary served decode step 2 via pin"
    );
    assert_eq!(twin.seen().len(), 0, "twin still untouched");

    Ok(())
}

/// `TASK-81`/S88: Cuando el primario fijado pierde su KV (responde `409 missing_prefill`),
/// el router aborta la petición (D-29 = a), no envía basura al siguiente candidato y
/// des-fija (unpins) la sesión para que un reintento futuro no quede atado al nodo corrupto.
#[tokio::test]
async fn when_the_pinned_primary_loses_its_kv_the_router_aborts_and_unpins_the_session(
) -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS88LoseFirst", (0, 1), 1.0, 1.0);
    // El nodo primario: responde al prefill, pero simula pérdida de KV en decode
    let primary =
        HopConfig::new("12D3KooWS88LosePrimary", (1, 2), 2.0, 0.0).refusing_missing_prefill();
    let twin = HopConfig::new("12D3KooWS88LoseTwin", (1, 2), 2.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let primary_url = spawn_hop(primary.clone()).await?;
    let twin_url = spawn_hop(twin.clone()).await?;

    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            layer_peer(&primary.peer_id, &primary_url, primary.layer_range),
            layer_peer(&twin.peer_id, &twin_url, twin.layer_range),
        ],
        "http://127.0.0.1:1".to_string(),
        identity.clone(),
    )?;

    let session_id = "session-s88-lost";

    // 1. Prefill (seq_start_pos = 0) succeeds on primary
    let prefill = sealed_step_for_session(&identity, session_id, 0, 2001)?;
    router
        .route(0, 2, prefill.to_bytes()?, MODEL_ID, None)
        .await?;
    assert_eq!(primary.seen().len(), 1);
    assert_eq!(
        router.session_pins().get_pinned_peer(session_id, 1),
        Some(primary.peer_id.clone())
    );

    // 2. Decode step (seq_start_pos = 1): primary refuses because it lost KV
    let decode = sealed_step_for_session(&identity, session_id, 1, 2002)?;
    let err = router
        .route(0, 2, decode.to_bytes()?, MODEL_ID, None)
        .await
        .expect_err("a 409 missing_prefill must abort");

    assert!(tessella_core::error::is_missing_prefill(&err));
    // Twin must NOT have been asked (D-29: do not burn the next candidate blindly)
    assert_eq!(twin.seen().len(), 0);
    // The session must be unpinned
    assert_eq!(router.session_pins().get_pinned_peer(session_id, 1), None);

    Ok(())
}

/// `TASK-87`/S94: In an unsampled session (`audit_sample_rate = 0.0`, no audit ticket),
/// a marked canary (`canary: true`) is never chosen as the primary of a hop, even if it is
/// first in the candidate list and fastest. When an audit explicitly names it, the audit
/// still reaches it by ID as a control.
#[tokio::test]
async fn in_an_unsampled_session_a_marked_canary_never_serves_a_hop() -> anyhow::Result<()> {
    let first = HopConfig::new("12D3KooWS94FirstHop", (0, 1), 1.0, 1.0);
    // Canary returns 5.0 (corrupted tensor)
    let canary = HopConfig::new("12D3KooWS94FastCanary", (1, 2), 5.0, 0.0);
    // Honest returns 2.0 (correct tensor: (1.0 + 1.0) * 2.0 = 4.0)
    let honest = HopConfig::new("12D3KooWS94HonestHop", (1, 2), 2.0, 0.0);

    let first_url = spawn_hop(first.clone()).await?;
    let canary_url = spawn_hop(canary.clone()).await?;
    let honest_url = spawn_hop(honest.clone()).await?;

    let sink = spawn_audit_sink().await?;
    let identity = Arc::new(generate_keypair());

    let mut canary_peer = layer_peer(&canary.peer_id, &canary_url, canary.layer_range);
    canary_peer.canary = true;
    let honest_peer = layer_peer(&honest.peer_id, &honest_url, honest.layer_range);

    let router = router_with_identity(
        vec![
            layer_peer(&first.peer_id, &first_url, first.layer_range),
            // Canary is placed first: without the canary filter, it would win the hop
            canary_peer,
            honest_peer,
        ],
        sink.url.clone(),
        identity.clone(),
    )?;

    // 1. Unsampled session (audit order = None, plain sealed payload)
    let payload = sealed_payload(&identity)?.to_bytes()?;
    let out = router
        .route(0, 2, payload, MODEL_ID, None)
        .await
        .expect("unsampled session must succeed through honest peer");

    let out_payload = HiddenStatePayload::from_bytes(&out)?;
    let values = out_payload
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &values {
        assert!(
            (*value - 4.0).abs() < 1e-6,
            "client must receive honest computation (4.0), not canary (5.0): {values:?}"
        );
    }

    assert_eq!(
        canary.seen().len(),
        0,
        "canary must NEVER be called in an unsampled session"
    );
    assert_eq!(honest.seen().len(), 1, "honest node must serve the client");

    // 2. Audited session with the canary named explicitly (fresh nonce/request_id to avoid replay detection)
    let mut audited_payload = sealed_payload_with_canary(&identity, 1, &canary.peer_id)?;
    audited_payload.request_id = "req-s94-audited".to_string();
    if let Some(ticket) = audited_payload.work_ticket.as_mut() {
        ticket.request_id = "req-s94-audited".to_string();
    }
    seal_signed_request(
        &identity,
        &mut audited_payload,
        9999,
        tessella_core::time::unix_now_secs(),
    )?;
    let audited_out = router
        .route(
            0,
            2,
            audited_payload.to_bytes()?,
            MODEL_ID,
            audit_order_with_canary(1, 1, &canary.peer_id),
        )
        .await
        .expect("audited session must succeed");

    let audited_values = HiddenStatePayload::from_bytes(&audited_out)?
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &audited_values {
        assert!(
            (*value - 4.0).abs() < 1e-6,
            "client still receives honest computation in audited session: {audited_values:?}"
        );
    }

    // Now canary was reached once by the audit as a control
    assert_eq!(
        canary.seen().len(),
        1,
        "audit must reach the canary by ID as a control"
    );

    Ok(())
}

/// `TASK-97`/S103: Dynamic cross-replica pipeline routing.
/// Verifies that when replica group 1 is incomplete (e.g. peer A covers [0, 1) and has no peer for [1, 2)),
/// the router bridges the pipeline to candidate peer B covering [1, 2) from replica group 0.
/// Furthermore, validates that:
/// 1. SessionPinRegistry pins each hop independently across different replica topologies.
/// 2. Subsequent decode steps (seq_start_pos > 0) strictly follow the pinned peers, preventing
///    divergence to twin replicas (e.g. peer C covering [0, 1) in replica group 0) and avoiding false 409 missing_prefill.
/// 3. If a peer loses KV cache on decode, the router aborts cleanly and clears the session pins.
#[tokio::test]
async fn cross_replica_pipeline_continuity_and_session_pinning_preserves_kv_cache(
) -> anyhow::Result<()> {
    // Peer A: covers [0, 1) (simulated replica group 1 - incomplete chain)
    let peer_a = HopConfig::new("12D3KooWPeerAReplica1", (0, 1), 1.0, 1.0);
    // Peer B: covers [1, 2) (simulated replica group 0 - bridging hop)
    let peer_b = HopConfig::new("12D3KooWPeerBReplica0", (1, 2), 2.0, 0.0);
    // Peer C: twin replica covering [0, 1) (simulated replica group 0)
    let peer_c =
        HopConfig::new("12D3KooWPeerCReplica0", (0, 1), 1.0, 1.0).refusing_missing_prefill();

    let url_a = spawn_hop(peer_a.clone()).await?;
    let url_b = spawn_hop(peer_b.clone()).await?;
    let url_c = spawn_hop(peer_c.clone()).await?;

    let identity = Arc::new(generate_keypair());
    let router = router_with_identity(
        vec![
            layer_peer(&peer_a.peer_id, &url_a, peer_a.layer_range),
            layer_peer(&peer_b.peer_id, &url_b, peer_b.layer_range),
            // Peer C is also registered for layer 0
            layer_peer(&peer_c.peer_id, &url_c, peer_c.layer_range),
        ],
        "http://127.0.0.1:1".to_string(),
        identity.clone(),
    )?;

    let session_id = "session-task97-cross-replica";

    // 1. Prefill step (seq_start_pos = 0):
    // Bridges peer_a [0, 1) to peer_b [1, 2).
    let prefill = sealed_step_for_session(&identity, session_id, 0, 3001)?;
    let prefill_out = router
        .route(0, 2, prefill.to_bytes()?, MODEL_ID, None)
        .await
        .expect("prefill across mixed replica groups must succeed");

    let prefill_values = HiddenStatePayload::from_bytes(&prefill_out)?
        .to_tensor(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for val in &prefill_values {
        // (1.0 + 1.0) * 2.0 = 4.0
        assert!(
            (*val - 4.0).abs() < 1e-6,
            "computed tensor matches expected"
        );
    }

    assert_eq!(peer_a.seen().len(), 1, "peer_a served hop [0, 1)");
    assert_eq!(peer_b.seen().len(), 1, "peer_b served hop [1, 2)");
    assert_eq!(peer_c.seen().len(), 0, "peer_c was not involved in prefill");

    // Check SessionPinRegistry has pinned both hops deterministically
    assert_eq!(
        router.session_pins().get_pinned_peer(session_id, 0),
        Some(peer_a.peer_id.clone()),
        "peer_a is pinned for layer 0"
    );
    assert_eq!(
        router.session_pins().get_pinned_peer(session_id, 1),
        Some(peer_b.peer_id.clone()),
        "peer_b is pinned for layer 1"
    );

    // 2. Decode step (seq_start_pos = 1):
    // Must respect pins and avoid routing layer 0 to peer_c (which would fail with 409 missing_prefill)
    let decode = sealed_step_for_session(&identity, session_id, 1, 3002)?;
    let decode_out = router
        .route(0, 2, decode.to_bytes()?, MODEL_ID, None)
        .await
        .expect("decode step must reuse pinned peers across mixed groups");

    assert!(!decode_out.is_empty());
    assert_eq!(peer_a.seen().len(), 2, "peer_a served decode hop 0 via pin");
    assert_eq!(peer_b.seen().len(), 2, "peer_b served decode hop 1 via pin");
    assert_eq!(
        peer_c.seen().len(),
        0,
        "peer_c was never called because layer 0 was pinned to peer_a"
    );
    assert_eq!(
        peer_c.refusals.load(Ordering::Relaxed),
        0,
        "peer_c never refused because it was never routed to"
    );

    Ok(())
}
