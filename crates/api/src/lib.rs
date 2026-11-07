#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::significant_drop_tightening)]

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;

pub mod axum_host;
pub mod callback_auth;
pub mod handlers;
pub mod hop_body;
use handlers::*;

// Domain models
pub use tessella_protocol::{
    ModelConfig, RebalancePlan, ShardAssignment, WorkerAssignment, WorkerReady, WorkerRegistration,
    WorkerStatus,
};

/// The state the HTTP surface publishes, the callback registry it authenticates against and the
/// worker records of the `Musivum` role moved to `tessella-ingress` (`SOLID-89`): the data plane can
/// depend on that crate without linking Axum, and these re-exports keep every handler path — and its
/// tests — naming them exactly as before.
pub use tessella_ingress::{ApiState, MusivumState, RegisteredWorker};

/// Event pushed from the final pipeline shard back to the entry node's SSE stream.
///
/// `SOLID-89`: it is a wire payload between data-plane nodes, so it lives in `tessella-protocol`;
/// this re-export keeps the paths that already name it from this crate working.
pub use tessella_protocol::TokenStreamPayload;

pub fn get_total_layers(state: &ApiState, model_repo: &str) -> usize {
    if let Some(cfg) = state
        .model_config
        .as_ref()
        .filter(|c| c.model_id == model_repo)
    {
        return cfg.total_layers();
    }
    tessella_protocol::get_total_layers(model_repo).unwrap_or(32)
}

#[derive(serde::Deserialize, Clone)]
pub struct ChatCompletionRequest {
    /// Accepted for compatibility with OpenAI-shaped clients (`"default"`, `"gpt-4"`, the real repo
    /// id…) and **not** honoured (`TASK-56`/S62): a node serves one model — its shard — so every
    /// answer names the model it really served, never the one asked for. Sending a different value is
    /// not an error; it is logged at `debug!`.
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<tessella_core::models::ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Upper bound of the answer. Omitted ⇒ the node's `[worker] max_new_tokens` (`TASK-40`/S46:
    /// the key used to be dead — the handler hardcoded 512 — and now it is what it says).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling of this request (`TASK-40`/S46, D-16). Omitted ⇒ the mesh standard (0.2/0.9/40);
    /// `0.0` is greedy; a value that cannot be sampled (negative, NaN, `top_p` outside `(0, 1]`)
    /// is answered with `400` instead of being ignored.
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Makes the answer reproducible: the same request with the same seed produces the same text.
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// A **hint**, not an id (`TASK-78`/S85, `D-36`): the id of a conversation is issued by the node
    /// that holds its K/V cache and it travels back in the answer
    /// ([`ChatCompletionResponse::session_id`] and the
    /// [`handlers::inference::SESSION_ID_HEADER`] header). This field is honoured only when it names a
    /// session **this** node issued and still holds; any other value —invented, copied from another
    /// client, or belonging to a node whose sweeper already dropped it— opens a new conversation
    /// instead of continuing somebody else's, which is what it did while it was the K/V key.
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(serde::Deserialize, Clone)]
pub struct FimRequest {
    pub prefix: String,
    pub suffix: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// The same **hint** as [`ChatCompletionRequest::session_id`] (`TASK-78`/S85, `D-36`): the id of a
    /// FIM conversation is issued by the node and it travels back in the
    /// [`handlers::inference::SESSION_ID_HEADER`] header of this stream.
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub mode: Option<tessella_protocol::InferenceMode>,
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub metrics: Option<tessella_core::models::GenerationMetrics>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    /// The **canonical** id of the session (`TASK-78`/S85, `D-36`), issued by the node that holds its
    /// K/V cache: it is the value a client sends back in `session_id` so the next turn resumes this
    /// conversation. It is here **and** in the [`handlers::inference::SESSION_ID_HEADER`] header, and the
    /// answer always carries it —also when the client sent no hint at all— because without it a client
    /// that only ever made one request would have no way to start resuming.
    pub session_id: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: tessella_core::models::ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, serde::Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct ChatDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

#[derive(Debug, serde::Serialize)]
pub struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

pub async fn start_api_server(addr: SocketAddr, state: ApiState) -> Result<()> {
    // `TASK-76`/S83: the inference surface accepts a body as long as the ceiling **this node
    // publishes** (`HOP_BODY_LIMIT` = `MAX_HIDDEN_STATE_BYTES` = 64 MiB) instead of axum's 2 MiB
    // default — the mismatch `S76` measured: 2048 tokens (1,8 MiB) were refused with a bare `413`
    // after the entry node had paid for the prefill. The layer is local to the extractors that apply
    // it (`Json`, and `Bytes` — which `HopBody` replaces), so it lands exactly on these four routes
    // and leaves the callback surface (`/v1/token_stream`, reachable with a single capability) at the
    // 2 MiB default, where it belongs.
    // `TASK-84`/S91: the same mount carries the **quota of hop bodies**. It goes on the four routes
    // for the same reason the limit does — they are the surface that reads a body of the wire — and it
    // travels in the request (`Extension`) instead of in the state so the extractor stays generic over
    // it: a test can put its own quota (of one body, of a millisecond) in a request and prove the
    // refusal without a node, and a route mounted without this layer is still bounded by the quota of
    // the process, which is the safe default.
    let inference_routes = Router::new()
        .route("/v1/forward", post(handle_forward_hidden))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/generate", post(handle_generate))
        .route("/v1/fim/completions", post(handle_fim_completions))
        .layer(axum::extract::DefaultBodyLimit::max(
            hop_body::HOP_BODY_LIMIT,
        ))
        .layer(axum::Extension(hop_body::hop_body_quota().clone()));

    let app = Router::new()
        .merge(inference_routes)
        .route("/api/v1/update_assignment", post(handle_update_assignment))
        .route("/status", get(handle_status))
        .route("/metrics", get(handle_metrics))
        .route("/v1/token_stream/:request_id", post(handle_token_stream))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("🌐 Worker API listening on http://{}", addr);
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
