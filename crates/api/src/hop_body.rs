//! The wire contract of the inference edge: the body limit of `POST /v1/forward`, the **named
//! refusals** that go with it (`TASK-76`/S83) and the **quota** of bodies one node reads at once
//! (`TASK-84`/S91).
//!
//! Three things live here because they are the same contract seen from its ends:
//!
//! * [`HopBody`], the extractor that buffers the body of a hop and refuses — **before** buffering it —
//!   anything over [`HOP_BODY_LIMIT`]. `MAX_HIDDEN_STATE_BYTES` is 64 MiB and it was published by
//!   every node while the real limit was axum's 2 MiB default: the mismatch is what `S76` measured
//!   (2048 tokens answered with a `413` no node had promised, followed by a `500` after the router
//!   suspended the request);
//! * [`HopBodyQuota`], the admission control that bounds how many of those bodies a node has **in
//!   flight**: one unauthenticated hop body costs 70-80 MiB of host RAM *before* any signature is
//!   verified (`S91`: 24 bodies at once = +1.6 GiB of RSS with `/status` still green, and the host
//!   runs out around 100), so a node with no quota is emptied from outside by whoever opens sockets.
//!   The quota is **terminal** at the router: retrying it would push the same body at every candidate;
//! * [`named_refusal`], the only place where a refusal becomes an HTTP answer, so the status, the
//!   `X-Tessella-Refusal` header and the body of the refusals of the mesh cannot drift apart.
//!
//! A refusal is *named* because of what the router does with it: the request **aborts** instead of
//! being retried (`TASK-68`/S75 put the rule in place for `missing_prefill`; `S83` extended it to the
//! two transport refusals and `S91` to the quota, which is the only one whose reason is «not now»
//! instead of «never here»).

use async_trait::async_trait;
use axum::{
    body::Bytes,
    extract::{FromRequest, Request},
    http::{StatusCode, header::CONTENT_LENGTH},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};
use tessella_core::{
    constants::inference::{MAX_HIDDEN_STATE_BYTES, is_near_the_limit},
    error::{self, DomainError, PAYLOAD_TOO_LARGE_CODE, REFUSAL_HEADER, TOO_MANY_HOP_BODIES_CODE},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// The largest body a hop accepts: **64 MiB**, the same number every node publishes
/// ([`tessella_core::constants::inference::MAX_HIDDEN_STATE_BYTES`]).
///
/// Applied to the inference routes with [`axum::extract::DefaultBodyLimit`] so the JSON surface
/// (`/v1/chat/completions`, `/v1/generate`, `/v1/fim/completions`) also carries a prompt as long as
/// the published ceiling says, and to [`HopBody`] explicitly (the layer is local: an extractor that
/// reads the body itself does not inherit it).
pub const HOP_BODY_LIMIT: usize = MAX_HIDDEN_STATE_BYTES;

/// How many hop bodies one node reads **at the same time** (`TASK-84`/S91) and how long it waits
/// between two frames of the same body.
///
/// Both moved to [`tessella_core::constants::inference`] with `SOLID-89` (they are properties of the
/// wire contract, and the data plane's integration tests must be able to name them without linking
/// this crate) and are re-exported here, where the quota that enforces them lives.
pub use tessella_core::constants::inference::{HOP_BODY_CONCURRENCY, HOP_BODY_IDLE_TIMEOUT};

/// The admission control of the hop bodies of **this node** (`TASK-84`/S91).
///
/// It is a [`Semaphore`] of [`HOP_BODY_CONCURRENCY`] permits and its whole point is what a permit is
/// **held by**: the [`HopBody`] value, not the read. A hop body is 70-80 MiB of host RAM that the
/// handler keeps while it works — the tensor is copied out of it, but the buffer is alive until the
/// answer is built —, so a quota that released the permit when the last frame arrived would bound the
/// *reading* and not the memory (`S91`: the RSS of a node does not come back after a burst).
///
/// It is built once per node ([`hop_body_quota`]) and read from the request by [`HopBody`], which is
/// what keeps the extractor free of state: the four inference routes mount it with
/// [`axum::Extension`] and everything else — the tests of a limit of a few bytes, a node with another
/// policy — can put its own in the request (or none, and get the one of the process).
pub struct HopBodyQuota {
    permits: Arc<Semaphore>,
    capacity: usize,
    idle: Duration,
}

/// The admission of **one** hop body, held for as long as its bytes live (`TASK-84`/S91).
///
/// Dropping it is what gives the permit back, so it travels inside the [`HopBody`] the handler
/// receives and it is released when that body is dropped — after the answer, not after the read.
pub struct HopBodyPermit(#[allow(dead_code)] OwnedSemaphorePermit);

impl std::fmt::Debug for HopBodyPermit {
    /// It prints its name and nothing else: what a permit *is* is the fact that it is held.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HopBodyPermit")
    }
}

impl HopBodyQuota {
    /// A quota of `capacity` bodies in flight, with the idle deadline of the node.
    pub fn new(capacity: usize) -> Self {
        Self::with_idle(capacity, HOP_BODY_IDLE_TIMEOUT)
    }

    /// The same quota with a declared idle deadline, which is what a test needs to exercise the
    /// deadline without waiting five seconds: the deadline is a property of the node's policy, and the
    /// policy is not a constant either.
    pub fn with_idle(capacity: usize, idle: Duration) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(capacity)),
            capacity,
            idle,
        }
    }

    /// How many bodies this node admits at once.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How long a body may go without sending anything before this node cuts it.
    pub fn idle(&self) -> Duration {
        self.idle
    }

    /// How many bodies are in flight right now, i.e. how much of the quota is taken. It is what
    /// `GET /status` publishes, and the only observable of the policy from outside.
    pub fn in_flight(&self) -> usize {
        self.capacity - self.permits.available_permits()
    }

    /// Admits one more body, or answers the **named refusal** of a node that is at its quota.
    ///
    /// The permit is taken *before* the first byte of the body is read — that is the whole bound: a
    /// refusal costs nothing (the message says so), and the body is never buffered. A refusal does
    /// **not** consume a permit (`in_flight` does not move when this node says no).
    fn admit(&self) -> Result<HopBodyPermit, Box<Response>> {
        match self.permits.clone().try_acquire_owned() {
            Ok(permit) => Ok(HopBodyPermit(permit)),
            Err(_) => {
                tracing::warn!(
                    "🚫 Refusing a hop body: this node is already reading {} of them, its quota \
                     (TASK-84/S91)",
                    self.in_flight()
                );
                Err(Box::new(named_refusal(
                    StatusCode::SERVICE_UNAVAILABLE,
                    TOO_MANY_HOP_BODIES_CODE,
                    &error::too_many_hop_bodies_error(self.in_flight(), self.capacity),
                )))
            }
        }
    }
}

/// The quota of **this process**, created on first use (`TASK-84`/S91).
///
/// It is one per node and not one per route: what it bounds is the RAM of the machine, which no route
/// owns. The four inference routes mount it (`api/src/lib.rs`) so a request carries it explicitly, and
/// [`HopBody`] falls back to it, so a route mounted *without* the layer is still bounded — the safe
/// default is the policy of the node, not the absence of one.
pub fn hop_body_quota() -> &'static Arc<HopBodyQuota> {
    static QUOTA: OnceLock<Arc<HopBodyQuota>> = OnceLock::new();
    QUOTA.get_or_init(|| Arc::new(HopBodyQuota::new(HOP_BODY_CONCURRENCY)))
}

/// The answer to a **named refusal** ([`error::refusal_code`]), or `None` when the failure is not one
/// of the three of the family — a dead backend or a CUDA failure keeps the answer the caller already
/// had.
pub fn refusal_response(error: &DomainError) -> Option<Response> {
    let code = error::refusal_code(error)?;
    let (status, _) = crate::handlers::inference::inference_error_status(error);
    Some(named_refusal(status, code, error))
}

/// The response of a named refusal: the status of the API edge, the code in [`REFUSAL_HEADER`] and
/// the message as the body.
///
/// The body **starts** with the code (the messages of `core::error` are built that way), so a caller
/// that reads only the body still gets it right — and the header is there for the ones that do not
/// want to parse prose.
pub fn named_refusal(status: StatusCode, code: &'static str, error: &DomainError) -> Response {
    tracing::warn!("🚫 Refusing a hop with `{code}` ({status}): {error}");
    // The message of the refusal, not `Display`: the body has to **start** with the code so a caller
    // that reads only the body (a log, a `curl`, the harness) sees the same thing as one that reads the
    // header. `Display` would prefix it with the name of the variant (`Validation error: …`).
    let body = error::refusal_message(error)
        .map(str::to_string)
        .unwrap_or_else(|| error.to_string());
    let mut response = (status, body).into_response();
    response.headers_mut().insert(
        axum::http::HeaderName::from_static(REFUSAL_HEADER),
        axum::http::HeaderValue::from_static(code),
    );
    response
}

/// The raw body of `POST /v1/forward`, capped at [`HOP_BODY_LIMIT`] **while** it is read.
///
/// It replaces the plain `axum::body::Bytes` the handler used, because that one answered with axum's
/// default rejection and a bare, unnamed `413`: the router could not tell "these bytes will never fit"
/// (abort) from "this node is in trouble" (retry another candidate). The two halves of the limit are
/// checked in the order that spends the least work:
///
/// 1. the `Content-Length` the client declared — a hop always declares it, so the common case is
///    refused without reading a single byte;
/// 2. the stream itself, counted frame by frame, for a body that arrives without a length: it is
///    refused at the exact byte the limit is crossed and **never** buffered beyond it.
///
/// Since `TASK-84`/S91 it is also the place where a body is **admitted**: the quota of the node
/// ([`HopBodyQuota`]) is asked *after* the cheaper refusal of the declared length and *before* the
/// first byte, and the permit it grants is held by the value itself, so the RAM of a body and its
/// admission have exactly the same lifetime.
pub struct HopBody {
    bytes: Bytes,
    /// The admission of the quota of this node, held for as long as these bytes live
    /// (`TASK-84`/S91): the memory of a hop body is the memory of the value that owns it, so this is
    /// what makes the quota a bound of RAM instead of a bound of connections. `None` only for a body
    /// built by a test.
    #[allow(dead_code)]
    // Held, not read: this field **is** the admission, and dropping the body is what gives the permit
    // back (`TASK-84`/S91).
    permit: Option<HopBodyPermit>,
}

impl HopBody {
    /// The bytes of the hop, whole: the ceiling refuses, it does not truncate.
    ///
    /// They live as long as this value, and so does the permit that admitted them.
    pub fn as_bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// How many bytes the hop sent.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the hop sent nothing (a hop never does: the payload has a header).
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn admitted(bytes: Bytes, permit: Option<HopBodyPermit>) -> Self {
        Self { bytes, permit }
    }

    /// The body of a hop whose admission was already decided (`TASK-84`/S91): the tests of the handler
    /// build one directly, because the quota is not what they are measuring.
    #[cfg(test)]
    pub(crate) fn unadmitted(bytes: Bytes) -> Self {
        Self::admitted(bytes, None)
    }
}

/// Reads the frames of a hop's body, refusing it the moment it crosses `limit` and cutting it when it
/// stops arriving for `idle` (`TASK-84`/S91).
///
/// It is a free function taking its two numbers as arguments (and not a method reading
/// [`HOP_BODY_LIMIT`] and [`HOP_BODY_IDLE_TIMEOUT`]) so the semantics — *refused at the crossing,
/// never buffered beyond it, a broken stream is not a size problem, a drip is not a size problem
/// either* — can be tested with a limit of a few bytes and a deadline of a few milliseconds instead of
/// allocating 64 MiB and waiting five seconds. The extractor always passes the numbers of the quota of
/// the node.
///
/// The deadline is **between two frames**: a body of the ceiling takes as long as it takes (its total
/// is bounded by the `hop_timeout_secs` of the node that sends it), and what this cuts is the socket
/// that sends nothing.
///
/// `Err` is boxed because [`Response`] is much larger than the `Vec<u8>` of the success case, and
/// the error is already being moved into a `Response` by the caller.
pub(crate) async fn read_hop_body<S>(
    mut stream: S,
    limit: usize,
    idle: Duration,
) -> Result<Vec<u8>, Box<Response>>
where
    S: futures::Stream<Item = Result<Bytes, axum::Error>> + Unpin,
{
    let mut buffer: Vec<u8> = Vec::new();
    loop {
        let next = if idle.is_zero() {
            stream.next().await
        } else {
            match tokio::time::timeout(idle, stream.next()).await {
                Ok(next) => next,
                Err(_) => {
                    // Not a size problem and not one of the named refusals: the body simply stopped
                    // arriving. The same bytes sent as a stream are served by this node, so the router
                    // of the upstream node may try another candidate — which is why this leaves as an
                    // unnamed `408` and not with the refusal header (`TASK-84`/S91).
                    tracing::warn!(
                        "⏱️ A hop body stopped arriving: {} bytes and then nothing for {}ms \
                         (TASK-84/S91)",
                        buffer.len(),
                        idle.as_millis()
                    );
                    return Err(Box::new(
                        (
                            StatusCode::REQUEST_TIMEOUT,
                            format!(
                                "request_timeout: {} bytes of the body of this hop arrived and then \
                                 nothing for {}ms, so this node stopped reading it and did not buffer \
                                 what is still missing. A hop streams its frames back to back; a socket \
                                 that goes quiet holds the RAM and the quota of the node for nothing. \
                                 Send the body as a stream, or route this hop elsewhere \
                                 (TASK-84/S91).",
                                buffer.len(),
                                idle.as_millis()
                            ),
                        )
                            .into_response(),
                    ));
                }
            }
        };
        let Some(chunk) = next else { break };
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => {
                // Not a size problem: the body itself did not arrive. That is the caller's, so a `400`
                // — and **not** one of the three named refusals, which the router has to be able to
                // tell apart.
                tracing::warn!("🚫 A hop's body could not be read: {e}");
                return Err(Box::new(
                    (
                        StatusCode::BAD_REQUEST,
                        format!("invalid_request: the body of the hop could not be read: {e}"),
                    )
                        .into_response(),
                ));
            }
        };
        if buffer.len() + chunk.len() > limit {
            return Err(Box::new(named_refusal(
                StatusCode::PAYLOAD_TOO_LARGE,
                PAYLOAD_TOO_LARGE_CODE,
                &error::payload_too_large_error(buffer.len() + chunk.len(), limit),
            )));
        }
        buffer.extend_from_slice(&chunk);
    }

    // `TASK-47`/S53: warn *before* the ceiling rejects anything, so a pipeline that is about to hit it
    // is visible in the log of the node that receives the payload (the same warning band the emitting
    // node uses in `Session::forward`). It is measured on what actually arrived, not on what the client
    // declared.
    if is_near_the_limit(buffer.len(), limit) {
        tracing::warn!(
            "⚠️ Hidden state close to the wire limit on 'POST /v1/forward': {} of {} bytes \
             (TASK-47/S53)",
            buffer.len(),
            limit
        );
    }

    Ok(buffer)
}

/// The extractor needs **no** state, so it is generic over it: that is what lets it be tested without
/// building an [`ApiState`] (`crates/api/src/hop_body.rs`, `mod tests`). The policy it applies is the
/// **quota** the request carries ([`HopBodyQuota`]) — the four inference routes mount it
/// (`api/src/lib.rs`) — and the quota of the process when it carries none, so a route mounted without
/// the layer is bounded too.
#[async_trait]
impl<S: Send + Sync> FromRequest<S> for HopBody {
    type Rejection = Response;

    async fn from_request(req: Request, _state: &S) -> Result<Self, Self::Rejection> {
        // Fast path: a hop always declares its length, so a body the client itself says is over the
        // limit is refused without reading a single byte of it.
        let declared = req
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<usize>().ok());
        if let Some(declared) = declared
            && declared > HOP_BODY_LIMIT
        {
            return Err(named_refusal(
                StatusCode::PAYLOAD_TOO_LARGE,
                PAYLOAD_TOO_LARGE_CODE,
                &error::payload_too_large_error(declared, HOP_BODY_LIMIT),
            ));
        }

        // `TASK-84`/S91: the admission control, after the cheapest refusal (a body that is over the
        // limit is refused without spending a permit) and before the first byte is read. It is what
        // bounds the RAM of the host against bodies that are never authenticable in the first place.
        let quota = req
            .extensions()
            .get::<Arc<HopBodyQuota>>()
            .cloned()
            .unwrap_or_else(|| Arc::clone(hop_body_quota()));
        let permit = quota.admit().map_err(|boxed| *boxed)?;

        let body = read_hop_body(
            req.into_body().into_data_stream(),
            HOP_BODY_LIMIT,
            quota.idle(),
        )
        .await
        .map_err(|boxed| *boxed)?;
        Ok(HopBody::admitted(Bytes::from(body), Some(permit)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, extract::Request, http::HeaderValue};

    /// The refusal a response carries, if it is one: the code in [`REFUSAL_HEADER`], the body (which
    /// starts with the same code) and the status.
    async fn refusal_of(response: Response) -> Option<(&'static str, String, StatusCode)> {
        let status = response.status();
        let code = response
            .headers()
            .get(REFUSAL_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| {
                [
                    error::MISSING_PREFILL_CODE,
                    error::PROMPT_TOO_LONG_CODE,
                    error::PAYLOAD_TOO_LARGE_CODE,
                    error::TOO_MANY_HOP_BODIES_CODE,
                ]
                .into_iter()
                .find(|code| *code == value)
            });
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the body of a refusal is readable");
        let text = String::from_utf8(body.to_vec()).expect("the message is UTF-8");
        code.map(|code| (code, text, status))
    }

    /// `TASK-76`/S83: the body limit of a hop is `MAX_HIDDEN_STATE_BYTES` — the number every node
    /// publishes — and not the 2 MiB of axum, which is what `S76` measured (2048 tokens refused).
    #[test]
    fn the_wire_limit_is_what_the_node_publishes() {
        assert_eq!(HOP_BODY_LIMIT, MAX_HIDDEN_STATE_BYTES);
        assert_eq!(HOP_BODY_LIMIT, 64 * 1024 * 1024);
    }

    /// A hop declares its length, so a body over the limit is refused **without reading a byte**: the
    /// number in the refusal is the declared one and not the size of what arrived (16 bytes here).
    #[tokio::test]
    async fn a_declared_body_over_the_limit_is_refused_without_being_read() {
        let declared = HOP_BODY_LIMIT + 1;
        let request = Request::builder()
            .header(CONTENT_LENGTH, declared)
            .body(Body::from(vec![0u8; 16]))
            .expect("the request builds");

        let response = HopBody::from_request(request, &())
            .await
            .err()
            .expect("a body over the limit cannot be extracted");
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response.headers().get(REFUSAL_HEADER),
            Some(&HeaderValue::from_static(PAYLOAD_TOO_LARGE_CODE)),
            "the refusal has to be named for the router to abort instead of retrying"
        );

        let (code, text, _) = refusal_of(response).await.expect("it is a named refusal");
        assert_eq!(code, PAYLOAD_TOO_LARGE_CODE);
        assert!(
            text.starts_with(PAYLOAD_TOO_LARGE_CODE),
            "the body has to start with its code: {text}"
        );
        assert!(
            text.contains(&declared.to_string()) && text.contains(&HOP_BODY_LIMIT.to_string()),
            "the refusal names both numbers ({declared} and {HOP_BODY_LIMIT}): {text}"
        );
    }

    /// The other half of the limit: a body that arrives **without** a declared length is counted frame
    /// by frame and refused at the byte the limit is crossed. The test uses a limit of 4 bytes instead
    /// of allocating 64 MiB, which is exactly why the counting loop is its own function.
    #[tokio::test]
    async fn a_stream_over_the_limit_is_refused_at_the_byte_the_limit_is_crossed() {
        let frames = futures::stream::iter([Ok(Bytes::from_static(&[7u8; 8]))]);
        let response = *read_hop_body(frames, 4, HOP_BODY_IDLE_TIMEOUT)
            .await
            .expect_err("8 bytes do not fit in 4");
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let (code, text, _) = refusal_of(response).await.expect("it is a named refusal");
        assert_eq!(code, PAYLOAD_TOO_LARGE_CODE);
        assert!(
            text.contains('8') && text.contains('4'),
            "what arrived (8) and the limit (4) are both named: {text}"
        );
    }

    /// A body that fits is returned whole, frame by frame and in order: the ceiling refuses, it does
    /// not truncate.
    #[tokio::test]
    async fn a_body_within_the_limit_is_read_whole() {
        let frames = futures::stream::iter([
            Ok(Bytes::from_static(&[1u8, 2])),
            Ok(Bytes::from_static(&[3u8, 4])),
        ]);
        assert_eq!(
            read_hop_body(frames, 4, HOP_BODY_IDLE_TIMEOUT)
                .await
                .expect("4 bytes fit in 4"),
            vec![1, 2, 3, 4]
        );
    }

    /// A stream that breaks is **not** a size problem: it leaves as a `400` without the header, so the
    /// node that was routing the step still tells "these bytes will not fit" (abort) apart from "this
    /// hop is broken" (try another candidate).
    #[tokio::test]
    async fn a_broken_stream_is_a_client_error_and_not_a_named_refusal() {
        let broken = std::io::Error::other("the peer hung up");
        let frames = futures::stream::iter([Err(axum::Error::new(broken))]);

        let response = *read_hop_body(frames, 8, HOP_BODY_IDLE_TIMEOUT)
            .await
            .expect_err("a broken stream cannot be read");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers().get(REFUSAL_HEADER), None);
    }

    // -------------------------------------------------------------------------------------
    // TASK-84 / S91 · la cuota de cuerpos en vuelo y el reloj del goteo
    // -------------------------------------------------------------------------------------

    /// A node admits its quota of hop bodies and refuses the one over it **by name**: `503`, the code
    /// in the header and a body that starts with it. A refusal is not an admission: the counter does
    /// not move when the node says no.
    ///
    /// It is tested with a quota of two and not with the eight of the node, so the test fails if the
    /// number is ever wired into the comparison instead of taken from the quota it was built with
    /// (`S91` derives the eight from the RAM of the reference machine: `8 × ~80 MiB` is ~8 % of what
    /// it has free).
    #[tokio::test]
    async fn the_quota_refuses_the_body_over_it_by_name() {
        let quota = HopBodyQuota::new(2);
        let _first = quota.admit().expect("one of two");
        let second = quota.admit().expect("two of two");
        assert_eq!(quota.in_flight(), 2);

        let response = *quota
            .admit()
            .expect_err("a third body over a quota of two cannot be admitted");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(quota.in_flight(), 2, "a refusal must not take a permit");

        let (code, text, _) = refusal_of(response).await.expect("it is a named refusal");
        assert_eq!(code, error::TOO_MANY_HOP_BODIES_CODE);
        assert!(text.starts_with(error::TOO_MANY_HOP_BODIES_CODE), "{text}");
        assert!(
            text.contains("2"),
            "the refusal names the quota of the node that answered: {text}"
        );

        // The permit is RAII: what makes this a bound of **memory** and not of connections is that it
        // goes back exactly when the bytes it protects are freed.
        drop(second);
        assert_eq!(quota.in_flight(), 1);
        let _third = quota
            .admit()
            .expect("the permit of a body that was dropped is reusable");
    }

    /// The permit lives inside the [`HopBody`], not inside the read: a hop body keeps its 70-80 MiB of
    /// host RAM while the handler works, so a quota that released the permit when the last frame
    /// arrived would bound nothing (`S91`: the RSS of a node is still up long after the burst, the
    /// glibc arenas do not give it back).
    #[tokio::test]
    async fn a_hop_body_keeps_its_permit_until_its_bytes_are_dropped() {
        let quota = Arc::new(HopBodyQuota::new(1));

        let frames =
            futures::stream::iter([Ok::<_, axum::Error>(Bytes::from_static(&[1u8, 2, 3]))]);
        let mut first = Request::builder()
            .body(Body::from_stream(frames))
            .expect("the request builds");
        first.extensions_mut().insert(Arc::clone(&quota));

        let body = HopBody::from_request(first, &())
            .await
            .expect("the first body of a quota of one is admitted");
        assert_eq!(body.as_bytes().as_ref(), &[1u8, 2, 3]);
        assert_eq!(
            quota.in_flight(),
            1,
            "the body was read whole and its bytes are still held by the handler"
        );

        // A second request while the first body is alive: refused, named, and its body never read.
        let mut other = Request::builder()
            .body(Body::from(vec![0u8; 4]))
            .expect("the request builds");
        other.extensions_mut().insert(Arc::clone(&quota));
        let response = HopBody::from_request(other, &())
            .await
            .err()
            .expect("a second body over a quota of one cannot be admitted");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let (code, _, _) = refusal_of(response).await.expect("it is a named refusal");
        assert_eq!(code, error::TOO_MANY_HOP_BODIES_CODE);

        drop(body);
        assert_eq!(quota.in_flight(), 0, "dropping the bytes frees the permit");
    }

    /// A body that **stops arriving** is cut by the idle deadline, and the answer is a plain `408`
    /// **without** the refusal header: it is not about the bytes of the request (the same body sent as
    /// a stream is served), so the node that was routing the step may try another candidate.
    ///
    /// A hop is not a browser: the frames of one hidden state arrive back to back, so a socket that
    /// sends nothing for five seconds is not slow, it is a request held open to keep 70-80 MiB of the
    /// host waiting (`S91`).
    #[tokio::test]
    async fn a_body_that_stops_arriving_is_cut_by_the_idle_deadline() {
        let frames = futures::stream::iter([Ok::<_, axum::Error>(Bytes::from_static(&[1u8, 2]))])
            .chain(futures::stream::pending::<Result<Bytes, axum::Error>>());

        let started = std::time::Instant::now();
        let response = *read_hop_body(frames, HOP_BODY_LIMIT, Duration::from_millis(50))
            .await
            .expect_err("a body that stops arriving cannot be read");
        let elapsed = started.elapsed();

        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(
            response.headers().get(REFUSAL_HEADER),
            None,
            "a drip is not one of the named refusals: the router may try another candidate"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the node cuts the socket itself instead of waiting for whoever holds it: {elapsed:?}"
        );

        let text = String::from_utf8(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("the body of a failure is readable")
                .to_vec(),
        )
        .expect("the message is UTF-8");
        assert!(text.starts_with("request_timeout"), "{text}");
        assert!(
            text.contains("50ms") && text.contains('2'),
            "the message names the deadline and what had arrived: {text}"
        );
    }

    /// The other half of the deadline: a body whose frames **keep arriving** within the declared idle
    /// gap is read whole, however long it takes in total. The deadline is between two reads and not
    /// over the body — cutting a hop that is streaming would be the `413` of `S76` with another excuse.
    #[tokio::test]
    async fn a_slow_but_alive_body_is_read_whole() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let feeder = tokio::spawn(async move {
            for _ in 0..3 {
                tokio::time::sleep(Duration::from_millis(20)).await;
                if tx
                    .send(Ok::<_, axum::Error>(Bytes::from_static(&[9u8; 2])))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        });

        let body = read_hop_body(
            tokio_stream::wrappers::ReceiverStream::new(rx),
            HOP_BODY_LIMIT,
            Duration::from_millis(200),
        )
        .await
        .expect("six bytes that arrive in three frames are not a drip");
        feeder.await.expect("the feeder does not panic");
        assert_eq!(body, vec![9u8; 6]);
    }

    /// [`refusal_response`] is the single detector the handler uses: it names the refusals of the mesh
    /// with the status `inference_error_status` gives them, and leaves every other failure alone.
    #[tokio::test]
    async fn only_the_named_refusals_leave_named() {
        use tessella_core::constants::inference::{HopCeiling, HopCeilingSource};

        let too_long = error::prompt_too_long_error(
            12_000,
            3_584,
            2,
            &HopCeiling {
                tokens: 9_362,
                source: HopCeilingSource::Wire,
                wire_tokens: Some(9_362),
                declared_tokens: 0,
            },
        );
        let response =
            refusal_response(&too_long).expect("a prompt that does not fit is a refusal");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let (code, text, _) = refusal_of(response).await.expect("it is a named refusal");
        assert_eq!(code, error::PROMPT_TOO_LONG_CODE);
        assert!(text.starts_with(error::PROMPT_TOO_LONG_CODE), "{text}");

        let missing = error::missing_prefill_error("work-7", 12, (8, 24));
        let response = refusal_response(&missing).expect("a prefix that never passed is a refusal");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let (code, _, _) = refusal_of(response).await.expect("it is a named refusal");
        assert_eq!(code, error::MISSING_PREFILL_CODE);

        // `TASK-84`/S91: the quota of hop bodies is a refusal of the same family and the only one
        // whose `503` means «later» instead of «never». The router stops either way, because retrying
        // it pushes the same 70-80 MiB body at the next candidate.
        let quota = refusal_response(&error::too_many_hop_bodies_error(8, 8))
            .expect("a full node refuses by name");
        assert_eq!(quota.status(), StatusCode::SERVICE_UNAVAILABLE);
        let (code, text, _) = refusal_of(quota).await.expect("it is a named refusal");
        assert_eq!(code, error::TOO_MANY_HOP_BODIES_CODE);
        assert!(text.starts_with(error::TOO_MANY_HOP_BODIES_CODE), "{text}");

        // The neighbouring failures keep the answer they already had: they are ours and not the
        // request's, so the router has to keep retrying them against another candidate.
        assert!(refusal_response(&DomainError::Inference("CUDA out of memory".into())).is_none());
        assert!(refusal_response(&DomainError::Network("router down".into())).is_none());
        assert!(
            refusal_response(&DomainError::Validation(
                "a temperature of 3 is not a distribution".into()
            ))
            .is_none()
        );
        assert!(
            refusal_response(&DomainError::Validation(
                "invalid_request: the request has no session_id".into()
            ))
            .is_none()
        );
    }

    /// The limit as a **hop** meets it, over a real TCP connection and through the composition
    /// `start_api_server` mounts (`DefaultBodyLimit` + the extractor + the route).
    ///
    /// `S76` measured a 2048-token payload (≈1,8–3,5 MiB depending on the shard) being answered with a
    /// bare `413` — the 2 MiB default of axum — while every node published 64 MiB. This test walks that
    /// case: a body of 3 MiB is **served whole** by the route, and one byte over the published limit is
    /// refused with the named contract instead of axum's bare rejection.
    ///
    /// The oversized case speaks raw HTTP on purpose: it sends the headers with
    /// `Content-Length: 67108865` and then 16 bytes, so getting an answer proves the body was **never
    /// read** (nothing of 64 MiB is allocated, neither here nor in the node).
    #[tokio::test]
    async fn the_route_serves_the_3_mib_body_of_s76_and_names_only_what_is_over_the_published_limit()
     {
        use axum::{Router, routing::post};
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        /// The handler of the fixture: it only has to prove **what arrived**, whole and in order.
        async fn echo_hop(body: HopBody) -> String {
            format!("{}", body.len())
        }

        let app = Router::new()
            .route("/v1/forward", post(echo_hop))
            // The layer the real router mounts. It does not limit a custom extractor — the reason
            // `HopBody` counts its own body — so the test is also what proves the two agree.
            .layer(axum::extract::DefaultBodyLimit::max(HOP_BODY_LIMIT));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("an ephemeral port is available");
        let addr = listener.local_addr().expect("the bound address is known");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        // `S76`, first case: the body a 2048-token hop really sends (12 layers, 896-wide, F16: ~3,5 MiB
        // at the end of the pipeline). It used to be answered with a bare `413` *after* the entry node
        // had paid for the prefill.
        let client = reqwest::Client::new();
        let served = vec![7u8; 3 * 1024 * 1024];
        let response = client
            .post(format!("http://{addr}/v1/forward"))
            .body(served.clone())
            .send()
            .await
            .expect("the route answers");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "3 MiB is what the wire promises now, not what the 2 MiB default refused"
        );
        assert_eq!(
            response.text().await.expect("the answer is readable"),
            served.len().to_string(),
            "the hop has to receive the body whole: the limit refuses, it does not truncate"
        );

        // `S76`, second case: one byte over the published limit, and the headers alone are enough.
        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("the route is listening");
        let head = format!(
            "POST /v1/forward HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/octet-stream\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            HOP_BODY_LIMIT + 1
        );
        stream
            .write_all(head.as_bytes())
            .await
            .expect("the headers are written");
        stream
            .write_all(&[0u8; 16])
            .await
            .expect("the first bytes are written");
        // Half-close so the node sees the end of what this hop sent: the rest is never written, and the
        // refusal has to arrive anyway.
        stream.shutdown().await.expect("the write half closes");
        let mut raw = String::new();
        tokio::time::timeout(Duration::from_secs(10), stream.read_to_string(&mut raw))
            .await
            .expect("the refusal cannot wait for the rest of the body")
            .expect("the refusal is readable");

        assert!(raw.starts_with("HTTP/1.1 413"), "{raw}");
        assert!(
            raw.contains(REFUSAL_HEADER),
            "the router of the upstream node reads the code from this header: {raw}"
        );
        assert!(
            raw.contains(PAYLOAD_TOO_LARGE_CODE),
            "the header and the body have to name the same refusal: {raw}"
        );
        assert!(
            raw.contains(&(HOP_BODY_LIMIT + 1).to_string())
                && raw.contains(&HOP_BODY_LIMIT.to_string()),
            "the refusal names both numbers: {raw}"
        );
        assert!(
            raw.contains("before buffering"),
            "and it says the body was never read: {raw}"
        );
    }

    /// The quota as a **hop** meets it: over real TCP and through the composition `start_api_server`
    /// mounts (`Extension(quota)` next to `DefaultBodyLimit`). The layer is what puts the quota of the
    /// process in the request, so this test is also what proves it is mounted — a route without it
    /// still has a quota (the extractor falls back to the one of the process), but a *different* one.
    ///
    /// What it fixes is the difference between a body **in flight** and a body that arrived: the first
    /// request declares 256 bytes and sends one, and the second — complete and small — is the one
    /// refused. That is the case `S91` measured as 70-80 MiB of host RAM per body: a node that only
    /// counts *finished* bodies has no bound at all.
    #[tokio::test]
    async fn the_route_refuses_the_body_over_the_quota_of_the_node_and_serves_the_one_in_flight() {
        use axum::{Router, routing::post};
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        /// The handler of the fixture: the answer names the size it was given, so the test can tell a
        /// served body from a refused one.
        async fn echo_hop(body: HopBody) -> String {
            format!("{}", body.len())
        }

        let quota = Arc::new(HopBodyQuota::new(1));
        let app = Router::new()
            .route("/v1/forward", post(echo_hop))
            .layer(axum::extract::DefaultBodyLimit::max(HOP_BODY_LIMIT))
            .layer(axum::Extension(Arc::clone(&quota)));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("an ephemeral port is available");
        let addr = listener.local_addr().expect("the bound address is known");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let head = |len: usize| {
            format!(
                "POST /v1/forward HTTP/1.1\r\nHost: {addr}\r\n\
                 Content-Type: application/octet-stream\r\nContent-Length: {len}\r\n\
                 Connection: close\r\n\r\n"
            )
        };

        // The first body is a hop that is **streaming**: headers plus one byte, and it stays open.
        let mut held = tokio::net::TcpStream::connect(addr)
            .await
            .expect("the route is listening");
        held.write_all(head(256).as_bytes())
            .await
            .expect("the headers are written");
        held.write_all(&[1u8])
            .await
            .expect("the first byte is written");

        // Wait for the node to be *reading* it: the counter of the quota is the observable of that, and
        // a test that trusts the scheduler to have got there first is a race it happens to win.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while quota.in_flight() == 0 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            quota.in_flight(),
            1,
            "the first body has to be in flight before the second one arrives"
        );

        // The second body is complete and small, and it is the one refused: what the quota bounds is
        // the number of bodies in flight, not the size of the one that just arrived.
        let mut refused = tokio::net::TcpStream::connect(addr)
            .await
            .expect("the route is listening");
        refused
            .write_all(head(4).as_bytes())
            .await
            .expect("the headers are written");
        refused
            .write_all(b"abcd")
            .await
            .expect("the body is written");
        // No half-close here on purpose: an immediate FIN after a **complete** body is a race the node
        // loses (measured in `S91` over this same fixture: the refusal is written *and* the connection
        // torn down, so the client reads a clean EOF with zero bytes — not even an `RST`). It is a
        // half-closed client, not the mesh: the router of a node sends with `reqwest`, which writes the
        // body and waits for the answer, and that case gets the refusal whole. `Connection: close` is
        // in the request, so the node still closes after answering and the read below still ends.
        let mut raw = String::new();
        tokio::time::timeout(Duration::from_secs(10), refused.read_to_string(&mut raw))
            .await
            .expect("the refusal cannot wait for the body that is in flight")
            .expect("the refusal is readable");

        assert!(raw.starts_with("HTTP/1.1 503"), "{raw}");
        assert!(
            raw.contains(REFUSAL_HEADER),
            "the router of the upstream node reads the code from this header: {raw}"
        );
        assert!(
            raw.contains(error::TOO_MANY_HOP_BODIES_CODE),
            "the header and the body have to name the same refusal: {raw}"
        );

        // The body that was in flight ends: it is served whole (the quota refuses, it does not
        // truncate) and the permit it was holding goes back, which is what makes the next request
        // admissible. No half-close, for the reason above.
        held.write_all(&[2u8; 255])
            .await
            .expect("the rest of the body is written");
        let mut served = String::new();
        tokio::time::timeout(Duration::from_secs(10), held.read_to_string(&mut served))
            .await
            .expect("a body that is streaming is not a drip")
            .expect("the answer is readable");

        assert!(served.starts_with("HTTP/1.1 200"), "{served}");
        assert!(
            served.ends_with("256"),
            "the hop received its 256 bytes whole: {served}"
        );
        assert_eq!(quota.in_flight(), 0, "the body went and so did its permit");
    }
}
