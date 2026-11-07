use crate::callback_auth::{self, CallbackAuthz};
use crate::{
    ApiState, ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, ChatUsage, FimRequest, GenerateRequest, GenerateResponse,
    OpenAIError, OpenAIErrorResponse, TokenStreamPayload,
};
use axum::{
    extract::{Json, Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    response::sse::{Event, KeepAlive, Sse},
};
use std::time::Instant;
use tokio::sync::mpsc;

use tessella_core::chat::{apply_chat_template_raw, apply_fim_template};
use tessella_core::generation::{GenerationConfig, run_generation_loop};
use tessella_core::sampling::SamplingParams;
use tessella_core::{error::DomainError, models::ChatMessage, time::unix_now_secs};

/// Maps a signature rejection to the status the contract of `TASK-05` promises: `409` for a
/// replayed nonce, `400` for a request whose **version** is not supported (since `TASK-26`/S32 that
/// includes the retired legacy `0`, i.e. an emitter built before v1: the reason is in the body) or
/// whose preimage cannot be built, and `401` for everything that fails authentication.
/// Maps a generation failure to the status the OpenAI surface promises (`TASK-47`/S53): a
/// **validation** failure is about the request the client sent — a prompt whose hidden state does not
/// fit in one hop of this pipeline arrives here as `DomainError::Validation("prompt_too_long: …")` —
/// so it is a `400` with the message intact, not a `500`. Everything else (a dead backend, a GPU
/// failure, a lost session) is ours, so it stays `500`.
/// `TASK-68`/S75 adds one case to that rule: the refusal of a step whose **prefix never passed
/// through this node** (`F-1`) is a `409` **named** `missing_prefill`. It is not a `400`, because the
/// request is perfectly valid — it is this node's state that cannot serve it — and it is not a `500`,
/// because the client can act on it (restart the generation) and the node that was routing the step
/// has to recognise it to abort instead of retrying the same bytes (`http_pipeline_router.rs`).
///
/// `TASK-76`/S83 completes the family with the two refusals whose cause is the **size** of the
/// request: a prompt that does not fit in one hop is a `400` named `prompt_too_long` (it could fit in
/// a shorter form anywhere in the mesh) and a body over the published wire limit is a `413` named
/// `payload_too_large` (it does not fit in **any** node). Both keep their message, and both share the
/// header of `missing_prefill`: what the router needs to know is the same in the three cases — the
/// bytes will not fit, so aborting beats retrying.
///
/// `TASK-84`/S91 adds the fourth and the only one whose reason is **not** the bytes: a node already
/// reading its quota of hop bodies answers a `503` named `too_many_hop_bodies`. It is a `503` and not
/// a `4xx` because the request is fine and the node is the one that is full — it will serve it later —
/// and it is named because retrying it pushes the same 70-80 MiB body at every candidate in turn, so
/// the router stops and the retry belongs to the client.
///
/// This function is the **single** place where the `(status, type)` of a failure is decided, and
/// `hop_body::refusal_response` reuses it.
pub(crate) fn inference_error_status(error: &DomainError) -> (StatusCode, &'static str) {
    match error {
        DomainError::Validation(_) if tessella_core::error::is_missing_prefill(error) => (
            StatusCode::CONFLICT,
            tessella_core::error::MISSING_PREFILL_CODE,
        ),
        DomainError::Validation(_) if tessella_core::error::is_prompt_too_long(error) => (
            StatusCode::BAD_REQUEST,
            tessella_core::error::PROMPT_TOO_LONG_CODE,
        ),
        DomainError::Validation(_) if tessella_core::error::is_payload_too_large(error) => (
            StatusCode::PAYLOAD_TOO_LARGE,
            tessella_core::error::PAYLOAD_TOO_LARGE_CODE,
        ),
        DomainError::Validation(_) if tessella_core::error::is_too_many_hop_bodies(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            tessella_core::error::TOO_MANY_HOP_BODIES_CODE,
        ),
        DomainError::Validation(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "generation_error"),
    }
}

fn signed_request_rejection(
    rejection: tessella_network::SignedRequestError,
) -> axum::response::Response {
    use tessella_network::SignedRequestError as E;
    let status = match rejection {
        E::Replay { .. } => StatusCode::CONFLICT,
        E::UnsupportedVersion(_) | E::Malformed(_) => StatusCode::BAD_REQUEST,
        E::MissingSignature | E::InvalidSignature | E::NotYetValid { .. } | E::Expired { .. } => {
            StatusCode::UNAUTHORIZED
        }
    };
    tracing::warn!("🔏 Rejecting signed request: {rejection} (H-09 · TASK-05)");
    (status, format!("Signed request rejected: {rejection}")).into_response()
}

/// The **only** place the `model` field of an answer is decided (`TASK-56`/S62).
///
/// The request's `model` is accepted — the field is part of the OpenAI shape and clients send
/// whatever they like: the §1bis gate and `test_concurrent.py` send `"default"`, `tessella-cli` sends
/// the real repo id — and it is deliberately **not** echoed. This node serves exactly one model (its
/// shard) and the answer may only name that one; echoing the request made a `200` attribute the
/// tokens to `"gpt-4"`.
///
/// Rejecting the field with `model_not_found` was the alternative and it would have broken every
/// client that sends a placeholder, starting with the project's own close gate (the same trap that
/// D-16 measured for `temperature`). The mismatch is logged at `debug!` — the gate sends `"default"`
/// on every request, so a `warn!` would be noise.
#[must_use]
fn answer_model(served: &str, requested: Option<&str>) -> String {
    if let Some(requested) = requested.filter(|requested| *requested != served) {
        tracing::debug!(
            "🔤 The request asked for model '{requested}' and this node serves '{served}': the answer \
             names the model it really served"
        );
    }
    served.to_string()
}

/// The header with which **every** flow publishes the canonical id of its session (`TASK-78`/S85,
/// `D-36`).
///
/// The id travels in two places on purpose: in the JSON body of the non-streaming answer (where a
/// client reads it without looking at headers) and in the header of **every** flow —a stream's body is
/// SSE and its last event is already defined by the OpenAI shape, while a header is available before the
/// first byte, so a client can keep the id even if the stream is cut short—. The name carries the
/// `x-tessella-` prefix so it cannot collide with anything of the OpenAI surface that is being imitated.
pub const SESSION_ID_HEADER: &str = "x-tessella-session-id";

/// Publishes the canonical id of a session on **every** answer whose request can carry a hint
/// (`TASK-78`/S85, `D-36`).
///
/// The id the client sent is not repeated: what a caller needs back is the value that will resume its
/// conversation, and that one is issued by the node that holds the K/V cache
/// ([`DistributedSession::session_id`]), never the hint it sent —the hint is exactly what another
/// client could have sent too—. The header is added to the response of this request and to nothing
/// else: there is no per-node registry of «the session of the last caller», which would be wrong the
/// moment two clients talk to the same node.
fn name_session(response: axum::response::Response, session_id: &str) -> axum::response::Response {
    match axum::http::HeaderValue::from_str(session_id) {
        Ok(value) => {
            let mut response = response;
            response.headers_mut().insert(SESSION_ID_HEADER, value);
            response
        }
        // The ids are hex with a fixed prefix, so this arm is unreachable in practice; it is here
        // because the alternative is a `panic` in a request path, and an answer without the header is
        // better than no answer.
        Err(_) => {
            tracing::warn!("🔑 The issued session id cannot be a header value: not publishing it");
            response
        }
    }
}

/// Builds the generation configuration of **one** request (`TASK-40`/S46, D-16).
///
/// The sampling parameters are the client's and this is the only translation between its body and
/// the sampler: omission means the mesh standard (`0.2/0.9/40`), `0.0` temperature means greedy,
/// and a value that cannot be sampled is **named** (the caller answers `400`) instead of being
/// silently replaced — which is what the stack did until this step, when every request sampled at
/// `0.2/0.9/40` whatever it asked for.
///
/// `default_max_tokens` is the node's `[worker] max_new_tokens`: it used to be dead (the handler
/// hardcoded 512) and is now what it promises — the answer's upper bound when the client omits
/// `max_tokens`.
fn build_generation_config(
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stop_sequences: Vec<String>,
    default_max_tokens: usize,
) -> Result<GenerationConfig, String> {
    let params = SamplingParams::from_request(temperature, top_p, top_k)?;
    Ok(GenerationConfig {
        max_new_tokens: max_tokens.unwrap_or(default_max_tokens),
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        seed,
        stop_sequences,
    })
}

/// The `400` of a sampling parameter that cannot be honoured, in the OpenAI-shaped body the rest of
/// this surface uses.
fn invalid_sampling_response(reason: String) -> axum::response::Response {
    tracing::warn!(
        "🚫 Rejecting a request with sampling parameters that cannot be honoured: {reason}"
    );
    (
        StatusCode::BAD_REQUEST,
        Json(OpenAIErrorResponse {
            error: OpenAIError {
                message: format!("invalid_request: {reason}"),
                error_type: "invalid_request".to_string(),
                code: Some("400".to_string()),
            },
        }),
    )
        .into_response()
}

/// `POST /v1/forward` — one hop of a pipeline: a hidden state in, a hidden state out.
///
/// The body is capped by [`crate::hop_body::HopBody`] (`HOP_BODY_LIMIT`, 64 MiB, `TASK-76`/S83), the
/// extractor that refuses an oversized one with a **named** `413` **before** buffering it and warns
/// when what is arriving is close to the limit (`TASK-47`/S53). Both halves used to live in this
/// function, on a `Bytes` that axum had already capped at 2 MiB: the check could only ever see bodies
/// below the published ceiling, and the ones above died with an unnamed `413` of the framework.
///
/// The answer to a failure of the engine goes through [`crate::hop_body::refusal_response`], which is
/// what makes it actionable for the node that sent the bytes: status, `X-Tessella-Refusal` code and
/// message travel together, so a refusal is recognised and not retried.
pub async fn handle_forward_hidden(
    State(state): State<ApiState>,
    body: crate::hop_body::HopBody,
) -> impl IntoResponse {
    let payload = match tessella_protocol::hidden_state::Payload::from_bytes(body.as_bytes()) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("Invalid payload bytes: {}", e),
            )
                .into_response();
        }
    };

    // H-09: this edge verifies version + signature + `iat` window + nonce before the payload
    // reaches the inference engine (which authenticates again — same policy, its own cache — so
    // a replay that slipped past a buggy caller is still rejected). Both caches count a frame
    // once, so a legitimate payload is never replayed *locally*.
    if let Err(rejection) = tessella_network::verify_signed_request(
        &payload,
        &state.replay_cache,
        unix_now_secs(),
        "POST /v1/forward",
    ) {
        return signed_request_rejection(rejection);
    }

    let Some(dist) = &state.dist_inference else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "distributed inference not ready",
        )
            .into_response();
    };

    match dist.forward_hidden(body.as_bytes()).await {
        Ok(resp_bytes) => (StatusCode::OK, resp_bytes).into_response(),
        Err(e) => {
            let err_str = e.to_string();
            // `TASK-68`/S75, `TASK-76`/S83 and `TASK-84`/S91: the refusals of the mesh are **named** —
            // status, `X-Tessella-Refusal` and a message that starts with the code — so the node that
            // was routing the step recognises them and **aborts** instead of retrying bytes that would
            // not fit in another candidate either: `missing_prefill` (`409`, the prefix never passed
            // through this node), `prompt_too_long` (`400`, the prompt does not fit in one hop of this
            // pipeline), `payload_too_large` (`413`, the body was over the published wire limit) and
            // `too_many_hop_bodies` (`503`, this node is already reading its quota of bodies: the same
            // bytes fit here later, and retrying them elsewhere would push 70-80 MiB at every
            // candidate). Without the machine-readable halves the router would have to guess from
            // prose, and guessing wrong is the silent corruption of `F-1`.
            if let Some(response) = crate::hop_body::refusal_response(&e) {
                return response;
            }
            if err_str.contains("unhealthy")
                || err_str.contains("degraded")
                || err_str.contains("GPU")
                || err_str.contains("CUDA")
            {
                (StatusCode::SERVICE_UNAVAILABLE, err_str).into_response()
            } else {
                (StatusCode::INTERNAL_SERVER_ERROR, err_str).into_response()
            }
        }
    }
}

pub async fn handle_token_stream(
    Path(request_id): Path<String>,
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(payload): Json<TokenStreamPayload>,
) -> impl IntoResponse {
    // H-10 (mitigation 2): the callback is a *capability*, not a guess. The capability is the
    // signature the entry node put on the payload this shard processed, so only a node that saw
    // this pipeline's traffic can write into the stream — and only once per payload, because a
    // capability is single-use (`crate::callback_auth`).
    match callback_auth::authorize_callback(
        state.callback_capabilities.as_deref(),
        &request_id,
        &headers,
        Instant::now(),
    ) {
        CallbackAuthz::Accepted => {}
        CallbackAuthz::Unauthorized => {
            // Deliberately no `warn!`: the endpoint is reachable without credentials, so a log line
            // per attempt would be a log-flood vector.
            tracing::debug!(
                "🔒 token_stream callback for '{}' refused: no valid capability",
                request_id
            );
            return StatusCode::UNAUTHORIZED.into_response();
        }
        CallbackAuthz::RateLimited => {
            tracing::warn!(
                "🐢 token_stream callback for '{}' throttled (more than {} per second)",
                request_id,
                callback_auth::CALLBACK_RATE_LIMIT_PER_SECOND
            );
            return StatusCode::TOO_MANY_REQUESTS.into_response();
        }
    }

    let senders = match &state.token_stream_senders {
        Some(s) => s,
        None => return StatusCode::NOT_FOUND.into_response(),
    };

    match senders.get(&request_id) {
        Some(tx) => {
            if tx.send(payload).await.is_err() {
                drop(tx);
                state.forget_stream(&request_id);
                return StatusCode::GONE.into_response();
            }
            StatusCode::OK.into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn handle_chat_completions(
    State(state): State<ApiState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let tokenizer = match &state.tokenizer {
        Some(t) => t.clone(),
        None => return (StatusCode::SERVICE_UNAVAILABLE, "Tokenizer not loaded").into_response(),
    };

    let mut history_tuples = Vec::new();
    for msg in &req.messages {
        history_tuples.push((msg.role.as_str(), msg.content.as_str()));
    }
    let prompt = tokenizer.apply_chat_template(&history_tuples, true);

    if req.stream.unwrap_or(false) {
        handle_chat_streaming(state, req, prompt).await
    } else {
        handle_chat_non_streaming(state, req, prompt).await
    }
}

pub async fn handle_chat_streaming(
    state: ApiState,
    req: ChatCompletionRequest,
    prompt: String,
) -> axum::response::Response {
    let tokenizer = match &state.tokenizer {
        Some(t) => t.clone(),
        None => return (StatusCode::SERVICE_UNAVAILABLE, "Tokenizer not loaded").into_response(),
    };
    let _model_cfg = match &state.model_config {
        Some(mc) => mc.clone(),
        None => {
            return (StatusCode::SERVICE_UNAVAILABLE, "Model config not loaded").into_response();
        }
    };
    // `TASK-78`/S85 (`D-36`): what the client sent is a **hint**. What comes back —and what resumes the
    // conversation in the next turn— is the id the session issues (published by `name_session`).
    let session_hint = req.session_id.clone();

    let config = match build_generation_config(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        req.stop.clone().unwrap_or_default(),
        state.max_new_tokens,
    ) {
        Ok(config) => config,
        Err(reason) => return invalid_sampling_response(reason),
    };

    // `TASK-56`/S62: the answer may only name the model this node serves, whatever the client asked
    // for (`answer_model` is the single definition of that rule).
    let model = answer_model(&state.default_model, req.model.as_deref());

    let dist_inference = match &state.dist_inference {
        Some(di) => di.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Distributed inference not ready",
            )
                .into_response();
        }
    };

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(128);
    let (token_tx, mut token_rx) = mpsc::channel::<TokenStreamPayload>(128);

    // Relay of the tokens the final shard pushes into `/v1/token_stream` (the SSE fast path). It is
    // started **before** the session so a session error can still be delivered as an SSE event,
    // which is the contract this endpoint had before `TASK-07`.
    {
        let tx_relay = tx.clone();
        let id_relay = id.clone();
        let model_relay = model.clone();
        let created_timestamp = unix_now_secs();
        tokio::spawn(async move {
            while let Some(event) = token_rx.recv().await {
                let is_done = event.finish_reason.is_some();
                let chunk = ChatCompletionChunk {
                    id: id_relay.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created_timestamp,
                    model: model_relay.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: if is_done { None } else { Some(event.token) },
                        },
                        finish_reason: event.finish_reason,
                    }],
                };
                if let Ok(data) = serde_json::to_string(&chunk)
                    && tx_relay
                        .send(Ok(Event::default().data(data)))
                        .await
                        .is_err()
                {
                    break;
                }
                if is_done {
                    let _ = tx_relay.send(Ok(Event::default().data("[DONE]"))).await;
                    break;
                }
            }
        });
    }

    // H-12 (`TASK-07`): the session is created **before** the callback channel is registered,
    // because the work `request_id` is minted by the orchestrator. Registering the channel under a
    // locally invented id would silently drop every token of the fast path, since the final shard
    // pushes them under the authorised id.
    let (session, request_id) = match dist_inference.clone().begin_session(session_hint).await {
        Ok(session) => session,
        Err(e) => {
            let err_chunk = OpenAIErrorResponse {
                error: OpenAIError {
                    message: format!("Failed to start session: {e}"),
                    error_type: "session_error".to_string(),
                    code: Some("500".to_string()),
                },
            };
            if let Ok(data) = serde_json::to_string(&err_chunk) {
                let _ = tx.send(Ok(Event::default().data(data))).await;
            }
            return Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
                .keep_alive(KeepAlive::default())
                .into_response();
        }
    };
    // Read **before** the session is moved into the generation task: this is the id this answer
    // publishes, and it is the one that resumes the conversation (`TASK-78`/S85, `D-36`).
    let session_id = session.session_id().to_string();

    if let Some(senders) = &state.token_stream_senders {
        senders.insert(request_id.clone(), token_tx);
    }
    // H-10 (mitigation 2): the registry knows how to drop both halves of a stream (the channel and
    // the capability that authorised it), so every exit path of this function calls `forget_stream`.
    let stream_registry = state.clone();
    let rid_cleanup = request_id.clone();

    tokio::spawn(async move {
        let mut session = session;
        let id_clone = id.clone();
        let model_clone = model.clone();
        let created_timestamp = unix_now_secs();

        let initial_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: created_timestamp,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        if let Ok(data) = serde_json::to_string(&initial_chunk) {
            let _ = tx.send(Ok(Event::default().data(data))).await;
        }

        let tx_clone = tx.clone();
        let eos_ids = tokenizer.eos_token_ids();
        let result = run_generation_loop(
            session.as_mut(),
            tokenizer.as_ref(),
            &config,
            &prompt,
            |token| {
                let tx = tx_clone.clone();
                let chunk = ChatCompletionChunk {
                    id: id_clone.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created_timestamp,
                    model: model_clone.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(token),
                        },
                        finish_reason: None,
                    }],
                };
                async move {
                    if let Ok(data) = serde_json::to_string(&chunk)
                        && tx.send(Ok(Event::default().data(data))).await.is_err()
                    {
                        return false;
                    }
                    true
                }
            },
            &eos_ids,
        )
        .await;

        if let Err(ref e) = result {
            tracing::error!("Generation loop error: {:?}", e);
        }

        if let Err(error) = &result {
            // TASK-47/S53: a stream that dies has to say **why**. Before this change a rejected
            // prefill (or any other failure) closed the SSE with no event at all, so the client saw
            // a truncated answer; now the same OpenAI-shaped error the non-streaming path returns
            // travels as one last `data:` event, followed by the `[DONE]` sentinel.
            let (status, error_type) = inference_error_status(error);
            let err_chunk = OpenAIErrorResponse {
                error: OpenAIError {
                    message: error.to_string(),
                    error_type: error_type.to_string(),
                    code: Some(status.as_u16().to_string()),
                },
            };
            if let Ok(data) = serde_json::to_string(&err_chunk) {
                let _ = tx.send(Ok(Event::default().data(data))).await;
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        }

        if result.is_ok() {
            let final_chunk = ChatCompletionChunk {
                id: id_clone,
                object: "chat.completion.chunk".to_string(),
                created: created_timestamp,
                model: model_clone,
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            if let Ok(data) = serde_json::to_string(&final_chunk) {
                let _ = tx.send(Ok(Event::default().data(data))).await;
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        }

        stream_registry.forget_stream(&rid_cleanup);
    });

    name_session(
        Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response(),
        &session_id,
    )
}

pub async fn handle_chat_non_streaming(
    state: ApiState,
    req: ChatCompletionRequest,
    prompt: String,
) -> axum::response::Response {
    let tokenizer = match &state.tokenizer {
        Some(t) => t.clone(),
        None => return (StatusCode::SERVICE_UNAVAILABLE, "Tokenizer not loaded").into_response(),
    };
    let _model_cfg = match &state.model_config {
        Some(mc) => mc.clone(),
        None => {
            return (StatusCode::SERVICE_UNAVAILABLE, "Model config not loaded").into_response();
        }
    };
    // `TASK-78`/S85 (`D-36`): the client's `session_id` is a **hint**, and this answer carries the id
    // the node issued for the session it actually opened (in the body *and* in the header).
    let session_hint = req.session_id.clone();

    let config = match build_generation_config(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        req.stop.clone().unwrap_or_default(),
        state.max_new_tokens,
    ) {
        Ok(config) => config,
        Err(reason) => return invalid_sampling_response(reason),
    };

    // `TASK-56`/S62: same rule as the streaming path — the answer names the model this node serves.
    let model = answer_model(&state.default_model, req.model.as_deref());

    let dist_inference = match &state.dist_inference {
        Some(di) => di.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Distributed inference not ready",
            )
                .into_response();
        }
    };

    // `TASK-07`: the work id is minted by the orchestrator and is not needed once the answer is built
    // (this handler has no token-stream callback). `TASK-78`/S85: the session's own id **is** needed,
    // because the answer has to publish it; it is read here, before the session moves into the lock.
    let (session, _request_id) = match dist_inference.clone().begin_session(session_hint).await {
        Ok(session) => session,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAIErrorResponse {
                    error: OpenAIError {
                        message: format!("Failed to start session: {e}"),
                        error_type: "session_error".to_string(),
                        code: Some("500".to_string()),
                    },
                }),
            )
                .into_response();
        }
    };
    let session_id = session.session_id().to_string();

    let mut session = session;
    let start_time = unix_now_secs();
    let eos_ids = tokenizer.eos_token_ids();

    let result = run_generation_loop(
        session.as_mut(),
        tokenizer.as_ref(),
        &config,
        &prompt,
        |_| async { true },
        &eos_ids,
    )
    .await;

    match result {
        Ok((text, _metrics)) => {
            let tok = tokenizer.clone();
            let p_str = prompt.clone();
            let t_str = text.clone();
            let (prompt_tokens, completion_tokens) = tokio::task::spawn_blocking(move || {
                let p_len = tok.encode(&p_str).unwrap_or_default().len();
                let t_len = tok.encode(&t_str).unwrap_or_default().len();
                (p_len, t_len)
            })
            .await
            .unwrap_or((0, 0));

            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".to_string(),
                created: start_time,
                model,
                // `TASK-78`/S85 (`D-36`): the id of the session this answer was produced in — the one
                // the client sends back to continue, and not the hint it sent.
                session_id: session_id.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: ChatUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            name_session(Json(response).into_response(), &session_id)
        }
        Err(e) => {
            // TASK-47/S53: a prompt that does not fit in one hop is a `400` that names the numbers,
            // not a `500` with a prose message.
            let (status, error_type) = inference_error_status(&e);
            if status == StatusCode::BAD_REQUEST {
                tracing::warn!("🚫 Rejecting a chat request: {}", e);
            }
            (
                status,
                Json(OpenAIErrorResponse {
                    error: OpenAIError {
                        message: e.to_string(),
                        error_type: error_type.to_string(),
                        code: Some(status.as_u16().to_string()),
                    },
                }),
            )
                .into_response()
        }
    }
}

pub async fn handle_generate(
    State(state): State<ApiState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let tokenizer = match &state.tokenizer {
        Some(t) => t.clone(),
        None => return (StatusCode::SERVICE_UNAVAILABLE, "Tokenizer not loaded").into_response(),
    };
    let _model_cfg = match &state.model_config {
        Some(mc) => mc.clone(),
        None => {
            return (StatusCode::SERVICE_UNAVAILABLE, "Model config not loaded").into_response();
        }
    };
    let dist_inference = match &state.dist_inference {
        Some(di) => di.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Distributed inference not ready",
            )
                .into_response();
        }
    };

    let mode = req.mode.clone().unwrap_or_default();
    let formatted = match mode {
        tessella_protocol::InferenceMode::Chat => {
            apply_chat_template_raw(&[("user", &req.prompt)], true)
        }
        tessella_protocol::InferenceMode::Fim { prefix, suffix } => {
            apply_fim_template(&prefix, &suffix)
        }
    };

    let config = match build_generation_config(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        req.stop.clone(),
        state.max_new_tokens,
    ) {
        Ok(config) => config,
        Err(reason) => return invalid_sampling_response(reason),
    };

    // `TASK-07`: this path has no client session id, so the orchestrator mints the work id. That is
    // also why the answer publishes no `SESSION_ID_HEADER` (`TASK-78`/S85): `GenerateRequest` has no
    // `session_id` to hint with, so an id the client could never send back would be noise. A client
    // that wants a resumable conversation uses `/v1/chat/completions` or the FIM endpoint.
    let (session, _request_id) = match dist_inference.clone().begin_session(None).await {
        Ok(session) => session,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to start session: {e}"),
            )
                .into_response();
        }
    };

    let mut session = session;
    let eos_ids = tokenizer.eos_token_ids();

    let result = run_generation_loop(
        session.as_mut(),
        tokenizer.as_ref(),
        &config,
        &formatted,
        |_| async { true },
        &eos_ids,
    )
    .await;

    match result {
        Ok((text, metrics)) => {
            let tok = tokenizer.clone();
            let f_str = formatted.clone();
            let t_str = text.clone();
            let (prompt_tokens, tokens_generated) = tokio::task::spawn_blocking(move || {
                let p_len = tok.encode(&f_str).unwrap_or_default().len();
                let t_len = tok.encode(&t_str).unwrap_or_default().len();
                (p_len, t_len)
            })
            .await
            .unwrap_or((0, 0));

            Json(GenerateResponse {
                text,
                tokens_generated,
                finish_reason: "stop".to_string(),
                prompt_tokens,
                metrics: Some(metrics),
            })
            .into_response()
        }
        Err(e) => {
            let (status, _) = inference_error_status(&e);
            (
                status,
                Json(serde_json::json!({"error": format!("generation error: {e}")})),
            )
                .into_response()
        }
    }
}

pub async fn handle_fim_completions(
    State(state): State<ApiState>,
    Json(req): Json<FimRequest>,
) -> impl IntoResponse {
    let tokenizer = match &state.tokenizer {
        Some(t) => t.clone(),
        None => return (StatusCode::SERVICE_UNAVAILABLE, "Tokenizer not loaded").into_response(),
    };
    let _model_cfg = match &state.model_config {
        Some(mc) => mc.clone(),
        None => {
            return (StatusCode::SERVICE_UNAVAILABLE, "Model config not loaded").into_response();
        }
    };
    let dist_inference = match &state.dist_inference {
        Some(di) => di.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Distributed inference not ready",
            )
                .into_response();
        }
    };

    let prompt = apply_fim_template(&req.prefix, &req.suffix);

    let config = match build_generation_config(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        vec![],
        state.max_new_tokens,
    ) {
        Ok(config) => config,
        Err(reason) => return invalid_sampling_response(reason),
    };

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(128);
    let id = format!("fimcmpl-{}", uuid::Uuid::new_v4());
    // `TASK-56`/S62: the FIM request has no `model` field, and the answer still goes through the
    // single definition of what an answer may name.
    let model = answer_model(&state.default_model, None);
    // `TASK-07`: the work id is minted by the orchestrator, so the client's `session_id` is only a hint
    // for the K/V cache (`TASK-78`/S85, `D-36`). This handler has no token-stream callback, so a session
    // error can be answered as a plain HTTP error; the id of the session **is** needed, because this
    // answer publishes it in its header.
    let session_hint = req.session_id.clone();
    let (session, _request_id) = match dist_inference.clone().begin_session(session_hint).await {
        Ok(session) => session,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAIErrorResponse {
                    error: OpenAIError {
                        message: format!("Failed to start session: {e}"),
                        error_type: "session_error".to_string(),
                        code: Some("500".to_string()),
                    },
                }),
            )
                .into_response();
        }
    };
    let session_id = session.session_id().to_string();

    tokio::spawn(async move {
        let mut session = session;
        let id_clone = id.clone();
        let model_clone = model.clone();
        let created_timestamp = unix_now_secs();

        let initial_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: created_timestamp,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        if let Ok(data) = serde_json::to_string(&initial_chunk) {
            let _ = tx.send(Ok(Event::default().data(data))).await;
        }

        let tx_clone = tx.clone();
        let eos_ids = tokenizer.eos_token_ids();
        let result = run_generation_loop(
            session.as_mut(),
            tokenizer.as_ref(),
            &config,
            &prompt,
            |token| {
                let tx = tx_clone.clone();
                let chunk = ChatCompletionChunk {
                    id: id_clone.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created_timestamp,
                    model: model_clone.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(token),
                        },
                        finish_reason: None,
                    }],
                };
                async move {
                    if let Ok(data) = serde_json::to_string(&chunk)
                        && tx.send(Ok(Event::default().data(data))).await.is_err()
                    {
                        return false;
                    }
                    true
                }
            },
            &eos_ids,
        )
        .await;

        if result.is_ok() {
            let final_chunk = ChatCompletionChunk {
                id: id_clone,
                object: "chat.completion.chunk".to_string(),
                created: created_timestamp,
                model: model_clone,
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            if let Ok(data) = serde_json::to_string(&final_chunk) {
                let _ = tx.send(Ok(Event::default().data(data))).await;
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        }

        if let Err(error) = &result {
            // TASK-47/S53: same contract as the chat stream — a truncated FIM stream says why.
            let (status, error_type) = inference_error_status(error);
            let err_chunk = OpenAIErrorResponse {
                error: OpenAIError {
                    message: error.to_string(),
                    error_type: error_type.to_string(),
                    code: Some(status.as_u16().to_string()),
                },
            };
            if let Ok(data) = serde_json::to_string(&err_chunk) {
                let _ = tx.send(Ok(Event::default().data(data))).await;
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        }
    });

    // `TASK-78`/S85 (`D-36`): the id of the session this stream was generated in, published before the
    // first chunk so a client can keep it even if the stream is cut short.
    name_session(
        Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response(),
        &session_id,
    )
}

#[cfg(test)]
mod token_stream_auth_tests {
    use super::*;
    use crate::callback_auth::{CallbackCapabilities, callback_capability};
    use axum::http::{HeaderMap, HeaderValue, header};
    use std::sync::Arc;
    use std::time::Instant;
    use tokio::sync::mpsc;

    /// Minimal `ApiState` for `/v1/token_stream/{request_id}`: only the two fields the callback path
    /// reads are populated (the stream registry and the capability registry of the entry node).
    fn state_for(
        request_id: &str,
        capabilities: Arc<CallbackCapabilities>,
    ) -> (ApiState, mpsc::Receiver<TokenStreamPayload>) {
        let (token_tx, token_rx) = mpsc::channel(8);
        let senders = Arc::new(dashmap::DashMap::new());
        senders.insert(request_id.to_string(), token_tx);
        let state = ApiState {
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
            admin_token: None,
            default_model: String::new(),
            model_config: None,
            peer_failures: None,
            gpu_health: None,
            tokenizer: None,
            token_stream_senders: Some(senders),
            orchestrator_url: None,
            own_api_url: None,
            replay_cache: Arc::new(tessella_protocol::signing::ReplayCache::new()),
            callback_capabilities: Some(capabilities),
            metrics: None,
            current_assignment_version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };
        (state, token_rx)
    }

    fn headers(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, HeaderValue::from_str(value).unwrap());
        headers
    }

    fn payload(request_id: &str) -> TokenStreamPayload {
        TokenStreamPayload {
            request_id: request_id.to_string(),
            token: String::new(),
            finish_reason: None,
        }
    }

    /// The audit's H-10 (mitigation 2): guessing a `request_id` must not be enough to write into
    /// somebody else's SSE stream.
    #[tokio::test]
    async fn a_callback_without_a_capability_is_rejected_and_never_reaches_the_stream() {
        let capabilities = Arc::new(CallbackCapabilities::new());
        capabilities.register("req-1", &[7u8; 64], Instant::now());
        let (state, mut tokens) = state_for("req-1", capabilities);

        let response = handle_token_stream(
            Path("req-1".to_string()),
            State(state),
            HeaderMap::new(),
            Json(payload("req-1")),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(
            tokens.try_recv().is_err(),
            "a rejected callback must not push anything into the stream"
        );
    }

    #[tokio::test]
    async fn a_callback_with_a_foreign_or_unknown_capability_is_rejected() {
        let capabilities = Arc::new(CallbackCapabilities::new());
        capabilities.register("req-1", &[7u8; 64], Instant::now());
        capabilities.register("req-2", &[8u8; 64], Instant::now());
        let (state, mut tokens) = state_for("req-2", capabilities);

        // A capability this node *did* issue, but for another stream.
        let response = handle_token_stream(
            Path("req-2".to_string()),
            State(state.clone()),
            headers(&format!("Bearer {}", callback_capability(&[7u8; 64]))),
            Json(payload("req-2")),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        // A capability nobody issued (what a third party guessing the id can produce).
        let response = handle_token_stream(
            Path("req-2".to_string()),
            State(state),
            headers(&format!("Bearer {}", callback_capability(&[9u8; 64]))),
            Json(payload("req-2")),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(tokens.try_recv().is_err());
    }

    #[tokio::test]
    async fn a_callback_with_the_entry_nodes_capability_reaches_the_stream() {
        let capabilities = Arc::new(CallbackCapabilities::new());
        let signature = [9u8; 64];
        capabilities.register("req-1", &signature, Instant::now());
        let (state, mut tokens) = state_for("req-1", capabilities);

        let response = handle_token_stream(
            Path("req-1".to_string()),
            State(state),
            headers(&format!("Bearer {}", callback_capability(&signature))),
            Json(payload("req-1")),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        let forwarded = tokens
            .recv()
            .await
            .expect("the callback reached the stream");
        assert_eq!(forwarded.request_id, "req-1");
    }

    /// The capability is single-use: a hop that replays the callback of a payload it already
    /// delivered cannot inject a second time.
    #[tokio::test]
    async fn a_replayed_callback_is_rejected() {
        let capabilities = Arc::new(CallbackCapabilities::new());
        let signature = [5u8; 64];
        capabilities.register("req-1", &signature, Instant::now());
        let (state, mut tokens) = state_for("req-1", capabilities);
        let bearer = headers(&format!("Bearer {}", callback_capability(&signature)));

        let first = handle_token_stream(
            Path("req-1".to_string()),
            State(state.clone()),
            bearer.clone(),
            Json(payload("req-1")),
        )
        .await
        .into_response();
        let second = handle_token_stream(
            Path("req-1".to_string()),
            State(state),
            bearer,
            Json(payload("req-1")),
        )
        .await
        .into_response();

        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::UNAUTHORIZED);
        assert!(tokens.recv().await.is_some());
        assert!(tokens.try_recv().is_err());
    }

    // -------------------------------------------------------------------------------------
    // TASK-47 / S53 · declaring the transport ceiling instead of dying with a `413`
    // -------------------------------------------------------------------------------------

    /// A prompt that does not fit in one hop arrives as `DomainError::Validation` from the engine, so
    /// it must leave as `400` with its message (the numbers) intact — not as a `500` that tells the
    /// client nothing it can act on. Since `TASK-76`/S83 it also leaves **named**: the type is the code
    /// the router reads to abort instead of retrying the same bytes. Everything else keeps its `500`.
    #[test]
    fn a_validation_failure_is_a_client_error_and_everything_else_is_ours() {
        let too_long = DomainError::Validation(
            "prompt_too_long: the hidden state of 12000 tokens (3584-wide, 2 bytes per element = \
             86016000 bytes) does not fit in one hop of this pipeline: the wire carries at most \
             67108864 bytes per body (9362 tokens)"
                .to_string(),
        );
        assert_eq!(
            inference_error_status(&too_long),
            (
                StatusCode::BAD_REQUEST,
                tessella_core::error::PROMPT_TOO_LONG_CODE
            )
        );

        // A validation failure that is not one of the refusals keeps the generic type.
        let other = DomainError::Validation("a temperature of 3 is not a distribution".into());
        assert_eq!(
            inference_error_status(&other),
            (StatusCode::BAD_REQUEST, "invalid_request")
        );

        // `TASK-84`/S91: the node that is already reading its quota of hop bodies is the only one of
        // the family that answers a `5xx`, because the request is fine and the node will serve it
        // later. It is named anyway: the router stops, and the retry belongs to the client.
        assert_eq!(
            inference_error_status(&tessella_core::error::too_many_hop_bodies_error(8, 8)),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                tessella_core::error::TOO_MANY_HOP_BODIES_CODE
            )
        );

        assert_eq!(
            inference_error_status(&DomainError::Inference("GPU exploded".into())),
            (StatusCode::INTERNAL_SERVER_ERROR, "generation_error")
        );
        assert_eq!(
            inference_error_status(&DomainError::Network("router down".into())),
            (StatusCode::INTERNAL_SERVER_ERROR, "generation_error")
        );
    }

    /// `TASK-68`/S75: the refusal of a step whose prefix never passed through this node is **not** the
    /// generic `400` of a validation failure, and it is not a `500` either. It leaves as a `409` whose
    /// own code (`missing_prefill`) is what the router of the upstream node reads to abort instead of
    /// retrying the same bytes — the decision **(a)** of the human for this step.
    #[test]
    fn the_missing_prefill_refusal_is_a_named_conflict_and_not_a_generic_error() {
        let refusal = tessella_core::error::missing_prefill_error("work-7", 12, (8, 24));

        assert_eq!(
            inference_error_status(&refusal),
            (
                StatusCode::CONFLICT,
                tessella_core::error::MISSING_PREFILL_CODE
            )
        );
        assert!(
            tessella_core::error::is_missing_prefill(&refusal),
            "the code has to be detectable by the hop that was routing the step: {refusal}"
        );

        // The neighbouring `Validation` failures keep their `400` **and** their generic type: the new
        // arms are *narrower* ones, and confusing a refusal with a plain validation failure would make
        // the router abort a request that another candidate could have served.
        let other_validation = DomainError::Validation("the request has no session_id".to_string());
        assert_eq!(
            inference_error_status(&other_validation),
            (StatusCode::BAD_REQUEST, "invalid_request")
        );
    }

    /// `TASK-76`/S83: the two refusals whose cause is the **size** of the request are named like the one
    /// of `TASK-68`/S75 — the router has to abort for all three (`refusal_code`) — and each keeps the
    /// status that says who can act on it: `400` when a shorter prompt would fit anywhere (`the wire` or
    /// this node's VRAM is what binds) and `413` when the body itself is over the published wire limit.
    #[test]
    fn the_size_refusals_are_named_and_keep_their_own_status() {
        use tessella_core::constants::inference::{HopCeiling, HopCeilingSource};

        let too_long = tessella_core::error::prompt_too_long_error(
            12_000,
            3_584,
            2,
            &HopCeiling {
                tokens: 1_170,
                source: HopCeilingSource::Vram,
                wire_tokens: Some(9_362),
                declared_tokens: 1_170,
            },
        );
        assert_eq!(
            inference_error_status(&too_long),
            (
                StatusCode::BAD_REQUEST,
                tessella_core::error::PROMPT_TOO_LONG_CODE
            )
        );
        assert!(tessella_core::error::is_prompt_too_long(&too_long));

        let too_big = tessella_core::error::payload_too_large_error(
            70_000_000,
            crate::hop_body::HOP_BODY_LIMIT,
        );
        assert_eq!(
            inference_error_status(&too_big),
            (
                StatusCode::PAYLOAD_TOO_LARGE,
                tessella_core::error::PAYLOAD_TOO_LARGE_CODE
            )
        );
        assert_eq!(
            tessella_core::error::refusal_code(&too_big),
            Some(tessella_core::error::PAYLOAD_TOO_LARGE_CODE)
        );
    }

    /// `GET /status` keeps the literal `OK` the harnesses grep for (`scripts/e2e/swarm.sh`) **and**
    /// publishes the per-hop ceiling a client needs to size its prompt (`TASK-47`/S53) **with its
    /// reason** (`TASK-76`/S83): the number alone does not tell an operator whether to send a shorter
    /// prompt or to shard the model differently. Since `S83` it also publishes **the wait** the node
    /// resolved at boot: a client that sizes a prompt of the ceiling and does not know how long the
    /// hop may take has half the pair (the ceiling and the wait were both fixed in the same session).
    #[tokio::test]
    async fn status_keeps_ok_and_publishes_the_per_hop_ceiling() {
        let (mut state, _tokens) = state_for("req-status", Arc::new(CallbackCapabilities::new()));
        state.max_input_tokens = Some(21_845);
        state.max_input_tokens_source =
            Some(tessella_core::constants::inference::HopCeilingSource::Wire.as_str());
        state.hop_timeout_secs = 60;
        state.default_model = "Qwen/Qwen2.5-1.5B-Instruct".to_string();

        let response = crate::handlers::worker::handle_status(State(state))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the status body must be readable");
        let text = String::from_utf8(body.to_vec()).expect("the status body is UTF-8");
        let json: serde_json::Value = serde_json::from_str(&text).expect("the body is JSON");

        assert_eq!(json["status"], "OK");
        assert_eq!(json["max_input_tokens"], 21_845);
        assert_eq!(json["max_input_tokens_source"], "wire");
        assert_eq!(json["model"], "Qwen/Qwen2.5-1.5B-Instruct");
        assert_eq!(
            json["hop_timeout_secs"], 60,
            "the wait the node resolved must be published next to the ceiling: {text}"
        );
        // `TASK-84`/S91: the third number of the same deal — how many hop bodies this node reads at
        // once, how many it is reading right now and how long it waits between two frames. The counter
        // is the only observable of the policy from outside: `in_flight == quota` is what a node that
        // is refusing everything looks like, and it is measured, not declared.
        assert_eq!(
            json["max_concurrent_hop_bodies"],
            crate::hop_body::HOP_BODY_CONCURRENCY,
            "the quota of the node has to be published: {text}"
        );
        assert_eq!(
            json["hop_body_idle_timeout_secs"],
            crate::hop_body::HOP_BODY_IDLE_TIMEOUT.as_secs(),
            "the idle deadline of a hop body has to be published: {text}"
        );
        assert!(
            json["hop_bodies_in_flight"].is_u64(),
            "how many bodies are in flight right now is the live half of the quota: {text}"
        );
        assert!(
            text.to_lowercase().contains("ok"),
            "the harness greps the body for `ok`: {text}"
        );
    }

    /// `TASK-76`/S83: a node that knows **no** ceiling publishes `null` in the number *and* in its
    /// reason — the pair is what a client keys on, and a reason without a number (or the other way
    /// round) would be a lie the harness could not detect.
    #[tokio::test]
    async fn status_publishes_no_reason_when_there_is_no_ceiling() {
        let (state, _tokens) = state_for("req-status", Arc::new(CallbackCapabilities::new()));
        assert_eq!(state.max_input_tokens, None);
        assert_eq!(state.max_input_tokens_source, None);

        let response = crate::handlers::worker::handle_status(State(state))
            .await
            .into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the status body must be readable");
        let json: serde_json::Value =
            serde_json::from_slice(&body).expect("the status body is JSON");

        assert_eq!(json["max_input_tokens"], serde_json::Value::Null);
        assert_eq!(json["max_input_tokens_source"], serde_json::Value::Null);
    }

    /// `TASK-50`/S56: the failures this node has seen towards its peers are published, which is the
    /// only consequence a failed hop has beyond the log line — there is deliberately **no**
    /// third-party report channel (`/report_failure` authenticates a single identity, D-3).
    #[tokio::test]
    async fn status_publishes_the_peer_failures_this_node_saw() {
        use tessella_core::telemetry::metrics::{PeerFailureReason, PeerFailureRegistry};

        let (mut state, _tokens) = state_for("req-status", Arc::new(CallbackCapabilities::new()));
        let registry = Arc::new(PeerFailureRegistry::new());
        registry.record_failure("12D3KooWDeadHop", PeerFailureReason::Timeout);
        registry.record_failure("12D3KooWDeadHop", PeerFailureReason::Timeout);
        state.peer_failures = Some(registry);

        let response = crate::handlers::worker::handle_status(State(state))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the status body must be readable");
        let json: serde_json::Value =
            serde_json::from_slice(body.as_ref()).expect("the status body is JSON");

        assert_eq!(json["peer_failures"]["total_failures"], 2);
        assert_eq!(json["peer_failures"]["tracked_peers"], 1);
        assert_eq!(
            json["peer_failures"]["peers"][0]["peer_id"],
            "12D3KooWDeadHop"
        );
        assert_eq!(json["peer_failures"]["peers"][0]["by_reason"]["timeout"], 2);
        assert_eq!(json["peer_failures"]["peers"][0]["last_reason"], "timeout");
    }

    /// Without a registry (never in production: the worker always builds one) the field is `null`
    /// instead of a fabricated empty report.
    #[tokio::test]
    async fn status_publishes_a_missing_registry_as_null() {
        let (state, _tokens) = state_for("req-status", Arc::new(CallbackCapabilities::new()));
        assert!(
            state.peer_failures.is_none(),
            "the fixture has no registry, which is what this test checks"
        );

        let response = crate::handlers::worker::handle_status(State(state))
            .await
            .into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the status body must be readable");
        let json: serde_json::Value =
            serde_json::from_slice(body.as_ref()).expect("the status body is JSON");

        assert!(json["peer_failures"].is_null(), "{json}");
    }

    // -------------------------------------------------------------------------------------
    // H-09 at the HTTP edge of the worker (`POST /v1/forward`): the hidden state of every hop
    // is authenticated **before** it reaches the engine, so this is the edge the mesh actually
    // protects (the router's own client side is covered by `http_pipeline_two_hops` in
    // `tessella-worker`).
    // -------------------------------------------------------------------------------------

    /// The payload an entry node emits: a real tensor plus the canonical seal of `Session::forward`
    /// (`seal_signed_request` with a fresh nonce and the current clock).
    fn hidden_state_bytes(keypair: &tessella_network::NetworkKeypair, seal: bool) -> Vec<u8> {
        let data = vec![0u8; 16];
        let checksum = crc32fast::hash(&data);
        let mut payload = tessella_protocol::hidden_state::Payload {
            request_id: "req-forward".into(),
            session_id: Some("session-forward".to_string()),
            seq_start_pos: 0,
            layer_idx: 0,
            shape: vec![1, 1, 4],
            dtype: tessella_protocol::hidden_state::DTypeTag::F32,
            data,
            checksum,
            entry_node_url: None,
            signature: Vec::new(),
            client_pubkey: Vec::new(),
            karma_ticket: None,
            work_ticket: None,
            signature_version: tessella_protocol::signing::SIGNATURE_VERSION_LEGACY,
            nonce: 0,
            iat: 0,
            tokens: Vec::new(),
        };
        if seal {
            tessella_network::seal_signed_request(keypair, &mut payload, 7, unix_now_secs())
                .expect("the fixture payload must be sealable");
        }
        payload.to_bytes().expect("the fixture payload must encode")
    }

    /// An unsigned hidden state is rejected with `401` and never reaches the engine.
    #[tokio::test]
    async fn a_hidden_state_without_a_signature_is_rejected_with_401() {
        let (state, _tokens) = state_for("req-forward", Arc::new(CallbackCapabilities::new()));

        let response = handle_forward_hidden(
            State(state),
            crate::hop_body::HopBody::unadmitted(axum::body::Bytes::from(hidden_state_bytes(
                &tessella_network::generate_keypair(),
                false,
            ))),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// The sealed payload passes authentication, so what stops it is only the missing engine of
    /// this fixture (`503`): without this test the `401` above could be an endpoint that rejects
    /// everything.
    #[tokio::test]
    async fn a_sealed_hidden_state_passes_authentication_and_only_then_misses_the_engine() {
        let (state, _tokens) = state_for("req-forward", Arc::new(CallbackCapabilities::new()));

        let response = handle_forward_hidden(
            State(state),
            crate::hop_body::HopBody::unadmitted(axum::body::Bytes::from(hidden_state_bytes(
                &tessella_network::generate_keypair(),
                true,
            ))),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A hidden state reused verbatim is a replay: `409`, and the first use is not.
    #[tokio::test]
    async fn a_replayed_hidden_state_is_rejected_with_409() {
        let (state, _tokens) = state_for("req-forward", Arc::new(CallbackCapabilities::new()));
        let bytes = hidden_state_bytes(&tessella_network::generate_keypair(), true);

        let first = handle_forward_hidden(
            State(state.clone()),
            crate::hop_body::HopBody::unadmitted(axum::body::Bytes::from(bytes.clone())),
        )
        .await
        .into_response();
        let second = handle_forward_hidden(
            State(state),
            crate::hop_body::HopBody::unadmitted(axum::body::Bytes::from(bytes)),
        )
        .await
        .into_response();

        assert_eq!(first.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(second.status(), StatusCode::CONFLICT);
    }
}

/// `TASK-56`/S62: the `model` field of an answer may only name the model this node **serves**.
///
/// An OpenAI-shaped request carries a `model` and clients send whatever they like: the §1bis gate and
/// `test_concurrent.py` send `"default"` (measured), `tessella-cli` sends the real repo. The mesh
/// serves one model per shard, so the field cannot be honoured — and it must not be echoed either,
/// which is what the handlers did. The decision (S62) is to accept any value and answer with the
/// truth, because rejecting it would break every client that sends a placeholder, the project's own
/// close gate included (the same trap D-16 measured for `temperature`).
///
/// The fake engine and the temporary tokenizer exist so the assertion is on a **real handler
/// response** and not on a helper nobody calls: the echo lived in the handlers, and only they can
/// pin it.
#[cfg(test)]
mod served_model_tests {
    use super::*;
    use crate::ChatCompletionRequest;
    use std::sync::Arc;
    use tessella_core::traits::{DistributedInference, DistributedSession, ForwardOutput};

    /// What this node serves. Every request below asks for something else.
    const SERVED_MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
    /// What a client with a habit of its own sends.
    const REQUESTED_MODEL: &str = "gpt-4";
    /// The id the fake engine issues for every session it opens (`TASK-78`/S85): the handlers have to
    /// publish **this** one, never the hint the client sent.
    const ISSUED_SESSION_ID: &str = "sess-0f1e2d3c4b5a69788796a5b4c3d2e1f0";
    /// A hint that names no session of this node — `TASK-78` is precisely the defect of an answer that
    /// echoed it (or, worse, resumed a conversation under it).
    const CLIENT_HINT: &str = "session-of-another-client";

    struct FakeTokenizer;

    impl tessella_core::traits::TokenizerService for FakeTokenizer {
        fn encode(&self, _text: &str) -> Result<Vec<u32>, DomainError> {
            Ok(vec![1])
        }

        fn decode(&self, tokens: &[u32]) -> Result<String, DomainError> {
            if tokens.contains(&0) {
                Ok(String::new())
            } else {
                Ok("a".to_string())
            }
        }

        fn apply_chat_template(
            &self,
            messages: &[(&str, &str)],
            add_generation_prompt: bool,
        ) -> String {
            tessella_core::chat::apply_chat_template_raw(messages, add_generation_prompt)
        }

        fn apply_fim_template(&self, prefix: &str, suffix: &str) -> String {
            tessella_core::chat::apply_fim_template(prefix, suffix)
        }

        fn eos_token_ids(&self) -> Vec<u32> {
            vec![0]
        }
    }

    /// One token, always `a`, without a GPU and without a network.
    struct FakeSession {
        session_id: String,
    }

    #[async_trait::async_trait]
    impl DistributedSession for FakeSession {
        fn session_id(&self) -> &str {
            &self.session_id
        }

        async fn forward(&mut self, _tokens: Vec<u32>) -> Result<ForwardOutput, DomainError> {
            let mut logits = vec![-10.0f32; 4];
            logits[1] = 5.0; // `a` wins: it is not EOS, so the loop decodes one token and stops
            Ok(ForwardOutput {
                logits: Some(logits),
                seq_start_pos: 0,
            })
        }
    }

    struct FakeEngine;

    #[async_trait::async_trait]
    impl DistributedInference for FakeEngine {
        async fn reset_distributed_cache(&self) {}

        async fn apply_prompt_template(
            &self,
            _mode: &tessella_core::models::InferenceMode,
            raw_prompt: &str,
        ) -> String {
            raw_prompt.to_string()
        }

        async fn begin_session(
            self: Arc<Self>,
            // The hint is **not** what comes back (`TASK-78`/S85): the engine issues the id of the
            // session, exactly as the real one does.
            _session_hint: Option<String>,
        ) -> Result<(Box<dyn DistributedSession>, String), DomainError> {
            Ok((
                Box::new(FakeSession {
                    session_id: ISSUED_SESSION_ID.to_string(),
                }),
                "work-served-model".to_string(),
            ))
        }
    }

    /// An `ApiState` with everything the three generation handlers read, and nothing else.
    fn state_serving_one_model() -> ApiState {
        ApiState {
            dist_inference: Some(Arc::new(FakeEngine)),
            storage: None,
            credits: None,
            discovery: None,
            slashing: None,
            peer_slashing: None,
            musivum_state: None,
            max_new_tokens: 1,
            max_input_tokens: None,
            max_input_tokens_source: None,
            hop_timeout_secs: 10,
            musivum_keypair: None,
            admin_token: None,
            default_model: SERVED_MODEL.to_string(),
            model_config: Some(crate::ModelConfig {
                model_id: SERVED_MODEL.to_string(),
                ..Default::default()
            }),
            peer_failures: None,
            gpu_health: None,
            tokenizer: Some(Arc::new(FakeTokenizer)),
            token_stream_senders: Some(Arc::new(dashmap::DashMap::new())),
            orchestrator_url: None,
            own_api_url: None,
            replay_cache: Arc::new(tessella_protocol::signing::ReplayCache::new()),
            callback_capabilities: Some(
                Arc::new(crate::callback_auth::CallbackCapabilities::new()),
            ),
            metrics: None,
            current_assignment_version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    fn chat_request(model: Option<&str>, stream: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.map(str::to_string),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            stream: Some(stream),
            max_tokens: Some(1),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            stop: None,
            session_id: None,
        }
    }

    async fn body_text(response: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the response body must be readable");
        String::from_utf8(bytes.to_vec()).expect("the response body is UTF-8")
    }

    fn json_of(text: &str) -> serde_json::Value {
        serde_json::from_str(text).expect("the answer must be JSON")
    }

    /// The defect of S62: the answer carried `"model": "gpt-4"` (what the client sent) while the
    /// tokens came from `Qwen/Qwen2.5-0.5B-Instruct`.
    #[tokio::test]
    async fn the_chat_answer_names_the_served_model_not_the_requested_one() {
        let response = handle_chat_completions(
            State(state_serving_one_model()),
            Json(chat_request(Some(REQUESTED_MODEL), false)),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        let json = json_of(&text);
        assert_eq!(json["model"], SERVED_MODEL, "{text}");
        assert!(
            !text.contains(REQUESTED_MODEL),
            "the answer must not name a model this node does not serve: {text}"
        );
        // …and it is a real answer: the fake engine produced the token.
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .expect("the answer carries a message");
        assert!(
            content.contains('a'),
            "the generation must have run: {text}"
        );
    }

    /// The streaming path builds its chunks in another function, with its own `model` variable: the
    /// same rule has to hold there or the SSE client is told a different lie.
    #[tokio::test]
    async fn the_streaming_chunks_name_the_served_model() {
        let response = handle_chat_completions(
            State(state_serving_one_model()),
            Json(chat_request(Some(REQUESTED_MODEL), true)),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        assert!(
            text.contains(&format!("\"model\":\"{SERVED_MODEL}\"")),
            "the chunks must name the served model: {text}"
        );
        assert!(
            !text.contains(REQUESTED_MODEL),
            "no chunk may name a model this node does not serve: {text}"
        );
    }

    /// The FIM handler already answered with the served model; the test pins it so a future change
    /// cannot make the third answer of this node lie as well.
    #[tokio::test]
    async fn the_fim_answer_names_the_served_model() {
        let response = handle_fim_completions(
            State(state_serving_one_model()),
            Json(FimRequest {
                prefix: "fn main() {".to_string(),
                suffix: "}".to_string(),
                max_tokens: Some(1),
                temperature: Some(0.0),
                top_p: None,
                top_k: None,
                seed: None,
                session_id: None,
            }),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        assert!(
            text.contains(&format!("\"model\":\"{SERVED_MODEL}\"")),
            "the chunks must name the served model: {text}"
        );
    }

    /// A client that omits `model` (allowed: the field is optional) is answered with the served one,
    /// never with an empty string.
    #[tokio::test]
    async fn omitting_the_model_is_answered_with_the_served_one() {
        let response = handle_chat_completions(
            State(state_serving_one_model()),
            Json(chat_request(None, false)),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        assert_eq!(json_of(&text)["model"], SERVED_MODEL, "{text}");
    }

    /// The `session_id` of an answer, as the client reads it — the header every flow publishes
    /// (`TASK-78`/S85). `None` means the response did not name any session.
    fn session_header(response: &axum::response::Response) -> Option<String> {
        response
            .headers()
            .get(SESSION_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    }

    /// `TASK-78`/S85 (`D-36`): the answer names the session the **node** issued, in the body and in the
    /// header, and it never repeats what the client sent. Echoing the hint was the smaller half of the
    /// defect; the bigger one is that the hint *was* the key of the K/V cache, so two clients that chose
    /// the same value shared a conversation (the engine test pins that half).
    #[tokio::test]
    async fn the_answer_publishes_the_session_the_node_issued_and_not_the_hint() {
        let mut request = chat_request(None, false);
        request.session_id = Some(CLIENT_HINT.to_string());
        let response = handle_chat_completions(State(state_serving_one_model()), Json(request))
            .await
            .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            session_header(&response).as_deref(),
            Some(ISSUED_SESSION_ID),
            "the header carries the canonical id, not the hint"
        );
        let text = body_text(response).await;
        assert_eq!(
            json_of(&text)["session_id"],
            ISSUED_SESSION_ID,
            "and so does the body: {text}"
        );
        assert!(
            !text.contains(CLIENT_HINT),
            "the hint a client sent must not come back as if it named a session: {text}"
        );
    }

    /// The other half of the same rule: a client that never sent a hint still gets an id, or it could
    /// never start resuming (its second request would be a brand-new conversation for ever).
    #[tokio::test]
    async fn a_request_without_a_hint_still_gets_a_session_to_resume() {
        let response = handle_chat_completions(
            State(state_serving_one_model()),
            Json(chat_request(None, false)),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            session_header(&response).as_deref(),
            Some(ISSUED_SESSION_ID),
            "the answer of a first, hint-less request names its session too"
        );
        let text = body_text(response).await;
        assert_eq!(json_of(&text)["session_id"], ISSUED_SESSION_ID, "{text}");
    }

    /// The streaming and FIM answers have no JSON body to put a field in (the body is SSE and its events
    /// are already defined by the OpenAI shape), so the header is the whole contract there: if it were
    /// missing, a client could stream for ever and never learn what to send back.
    #[tokio::test]
    async fn the_streaming_and_fim_answers_publish_the_session_in_their_header() {
        let mut streaming = chat_request(None, true);
        streaming.session_id = Some(CLIENT_HINT.to_string());
        let streamed = handle_chat_completions(State(state_serving_one_model()), Json(streaming))
            .await
            .into_response();

        assert_eq!(streamed.status(), StatusCode::OK);
        assert_eq!(
            session_header(&streamed).as_deref(),
            Some(ISSUED_SESSION_ID),
            "the SSE answer publishes the canonical id in a header"
        );

        let mut fim = FimRequest {
            prefix: "fn main() {".to_string(),
            suffix: "}".to_string(),
            max_tokens: Some(1),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: None,
            session_id: None,
        };
        fim.session_id = Some(CLIENT_HINT.to_string());
        let completed = handle_fim_completions(State(state_serving_one_model()), Json(fim))
            .await
            .into_response();

        assert_eq!(completed.status(), StatusCode::OK);
        assert_eq!(
            session_header(&completed).as_deref(),
            Some(ISSUED_SESSION_ID),
            "and so does the FIM stream"
        );
    }
}
