pub use tessella_protocol::DomainError;

/// The machine-readable code of the one failure `TASK-68`/`S75` is about: a hop received a
/// **decode step** of a session it never prefilled (`F-1` of the failover verification of
/// 2026-09-24).
///
/// It lives here — and not inside the worker — because **three** places have to agree on it:
///
/// * the engine refuses with it ([`missing_prefill_error`], `worker/src/inference_engine.rs` and
///   `worker/src/scheduler.rs`);
/// * the HTTP edge publishes it as the `X-Tessella-Refusal` header next to the `409`
///   (`api/src/handlers/inference.rs`), so a hop does not have to parse prose;
/// * the router of the **upstream** node reads it and aborts instead of retrying the same bytes
///   against the next candidate (`worker/src/http_pipeline_router.rs`).
///
/// Without the shared code the three would drift, and a drifted detection is the bug itself: the
/// router would retry a step whose prefix does not exist in the new hop and the client would get
/// `200` with plausible garbage.
pub const MISSING_PREFILL_CODE: &str = "missing_prefill";

/// The header the HTTP edge publishes next to a **named refusal** (`TASK-68`/S75, `TASK-76`/S83), so
/// the node that was routing a step detects the cause **without parsing prose** (a body can be
/// truncated, logged or re-wrapped; a header is either there or it is not). The body carries the same
/// code as its first word, so a caller that reads only one of the two still gets it right.
///
/// The codes that travel in it are [`MISSING_PREFILL_CODE`], [`PROMPT_TOO_LONG_CODE`],
/// [`PAYLOAD_TOO_LARGE_CODE`] and [`TOO_MANY_HOP_BODIES_CODE`]: what they have in common is the only
/// thing the router has to decide — **retrying it against another candidate cannot fix it** (the first
/// three because the bytes do not fit anywhere, the last one because retrying would push the same
/// 70-80 MiB body at every candidate in turn) — and that is why one header is enough
/// ([`refusal_code`]).
pub const REFUSAL_HEADER: &str = "x-tessella-refusal";

/// The refusal of `TASK-68`/`S75`: this node was asked for a step of a session whose **prefix never
/// passed through it** (`seq_start_pos > 0` with no session of its own), so it has no K/V for the
/// tokens the payload declares as already processed.
///
/// It is a [`DomainError::Validation`] — the `4xx` the mesh already uses for "this request cannot be
/// served here" (`prompt_too_long`, `TASK-47`/S53) — and its message names **the session, the prefix
/// and this node's layers**, which is what the operator has to see. Serving it would not fail: it
/// would return a correctly shaped tensor computed against an **empty** KV cache, i.e. the client
/// would read `200` with plausible and wrong tokens (measured chain in `F-1`, `S74`).
pub fn missing_prefill_error(
    session_id: &str,
    seq_start_pos: usize,
    layers: (usize, usize),
) -> DomainError {
    DomainError::Validation(format!(
        "{MISSING_PREFILL_CODE}: [TESERROR-02] session {session_id} was never prefilled on layers [{}, {}) of this \
         node (seq_start_pos = {seq_start_pos}): the K/V of the {} tokens the payload declares as \
         already processed does not exist here, so answering would return plausible garbage instead \
         of failing. The generation has to be restarted, or the prefix re-sent to this \
         hop, by the node that owns the session.",
        layers.0, layers.1, seq_start_pos
    ))
}

/// Whether `error` is the refusal of [`missing_prefill_error`].
///
/// The variant is deliberately **not** a new `DomainError` arm: the enum is a protocol DTO that
/// travels archived (`rkyv`) in other contracts, so adding an arm is deliberately avoided
/// (`TASK-68`). The code in the message is the contract, and it is a constant so no
/// caller can misspell it.
pub fn is_missing_prefill(error: &DomainError) -> bool {
    matches!(error, DomainError::Validation(message) if message.starts_with(MISSING_PREFILL_CODE))
}

/// The machine-readable code of the prompt that does not fit in **one hop** (`TASK-47`/S53,
/// `TASK-76`/S83).
///
/// It lives here, next to [`MISSING_PREFILL_CODE`], because the same three places have to agree on
/// it: the engine refuses with it ([`prompt_too_long_error`]), the HTTP edge publishes it in
/// [`REFUSAL_HEADER`] with a `400` (`api/src/handlers/inference.rs`) and the router of the upstream
/// node reads it and **aborts** instead of retrying bytes that will not fit anywhere
/// (`worker/src/http_pipeline_router.rs`).
pub const PROMPT_TOO_LONG_CODE: &str = "prompt_too_long";

/// The refusal of a prompt whose hidden state does not fit in one hop of the pipeline
/// (`TASK-47`/S53, extended by `TASK-76`/S83).
///
/// The message names **the numbers** — what arrived, how big the hidden state would have been, how
/// many tokens fit and, since `S83`, **which of the two limits binds** (the wire or this node's
/// declared VRAM budget) — because the whole point is that the operator can act on it instead of
/// reading a `413` from a node it does not control.
pub fn prompt_too_long_error(
    tokens: usize,
    hidden_size: usize,
    dtype_bytes: usize,
    ceiling: &crate::constants::inference::HopCeiling,
) -> DomainError {
    use crate::constants::inference::{HopCeilingSource, MAX_HIDDEN_STATE_BYTES};
    let payload_bytes = tokens
        .saturating_mul(hidden_size)
        .saturating_mul(dtype_bytes);
    let why = match ceiling.source {
        HopCeilingSource::Wire => format!(
            "the wire carries at most {MAX_HIDDEN_STATE_BYTES} bytes per body ('POST /v1/forward') \
             and this node declares no VRAM ceiling for a longer prefill (max_input_tokens = 0), so \
             the wire is what binds"
        ),
        HopCeilingSource::Vram => format!(
            "this node's VRAM declares a prefill ceiling of {} tokens (max_input_tokens) below the \
             {} the wire could carry, so the prefill is what binds",
            ceiling.declared_tokens,
            ceiling
                .wire_tokens
                .map(|wire| wire.to_string())
                .unwrap_or_else(|| "unknown number of".to_string())
        ),
        HopCeilingSource::Both => format!(
            "both limits agree (the wire's {MAX_HIDDEN_STATE_BYTES} bytes per body at \
             'POST /v1/forward' and this node's declared VRAM ceiling, max_input_tokens)"
        ),
    };
    DomainError::Validation(format!(
        "{PROMPT_TOO_LONG_CODE}: [TESERROR-03] the hidden state of {tokens} tokens ({hidden_size}-wide, \
         {dtype_bytes} bytes per element = {payload_bytes} bytes) does not fit in one hop of this \
         pipeline: {why}. At most {} tokens travel (source: {}), and one hop carries at most \
         {MAX_HIDDEN_STATE_BYTES} bytes per body of 'POST /v1/forward'. Reduce the prompt to {} tokens \
         or fewer, or serve the first layers on a node with a smaller model.",
        ceiling.tokens,
        ceiling.source.as_str(),
        ceiling.tokens
    ))
}

/// Whether `error` is the refusal of [`prompt_too_long_error`].
pub fn is_prompt_too_long(error: &DomainError) -> bool {
    matches!(error, DomainError::Validation(message) if message.starts_with(PROMPT_TOO_LONG_CODE))
}

/// The machine-readable code of the body that overflows the wire **before** it is read
/// (`TASK-76`/S83).
///
/// The body limit of `POST /v1/forward` used to be axum's default of 2 MiB while
/// [`crate::constants::inference::MAX_HIDDEN_STATE_BYTES`] (64 MiB) was what every node published, so
/// the real ceiling was **32 times smaller** than the documented one (measured in `S76`: 2048 tokens
/// were answered with a bare `413` *and*, after the router suspended, with a `500`). The named code
/// is what lets the router tell "this body can never fit" (abort) apart from "this node is slow"
/// (retry).
pub const PAYLOAD_TOO_LARGE_CODE: &str = "payload_too_large";

/// The refusal of a body over the wire limit of `POST /v1/forward`. `bytes` is what arrived (or what
/// the request declared) and `limit` is
/// [`crate::constants::inference::MAX_HIDDEN_STATE_BYTES`] — both are named so the operator does not
/// have to guess which of the two numbers was the problem.
pub fn payload_too_large_error(bytes: usize, limit: usize) -> DomainError {
    DomainError::Validation(format!(
        "{PAYLOAD_TOO_LARGE_CODE}: [TESERROR-04] the body is {bytes} bytes and one hop carries at most {limit} \
         bytes (MAX_HIDDEN_STATE_BYTES). The node refuses it before buffering it, so nothing was \
         read and no prefill was paid for. Send a shorter prompt, or split the work across more \
         nodes."
    ))
}

/// Whether `error` is the refusal of [`payload_too_large_error`].
pub fn is_payload_too_large(error: &DomainError) -> bool {
    matches!(error, DomainError::Validation(message) if message.starts_with(PAYLOAD_TOO_LARGE_CODE))
}

/// The machine-readable code of the refusal a node answers when it is already **reading its quota** of
/// hop bodies (`TASK-84`/S91).
///
/// It lives here, with the other codes of the family, because the same three places have to agree on
/// it: the extractor of a hop's body refuses with it (`api/src/hop_body.rs`, *before* reading a single
/// byte of the request), the HTTP edge publishes it in [`REFUSAL_HEADER`] with a `503`
/// (`api/src/handlers/inference.rs`) and the router of the **upstream** node reads it and stops
/// (`worker/src/http_pipeline_router.rs`).
///
/// It is the one code of the family whose cause is **not** the bytes of the request: the same bytes
/// fit in this node as soon as it is free (measured in `S91`: one unauthenticated hop body costs
/// 70-80 MiB of host RAM *before* any authentication, so a node with no quota can be emptied from
/// outside). What it shares with `payload_too_large` is the decision it feeds: retrying it means
/// pushing the same body — 70-80 MiB each time — at the next candidate, and then at the next one,
/// which is the amplification a quota exists to stop. The refusal is therefore **terminal** at the
/// router, and the retry belongs to the client, which is what its `503` already means.
pub const TOO_MANY_HOP_BODIES_CODE: &str = "too_many_hop_bodies";

/// The refusal of a hop body this node will not read **now** (`TASK-84`/S91).
///
/// Both numbers are named (`in_flight` bodies against `limit`) because the operator's next move
/// depends on whether the node is at its quota — and the quota is published by `GET /status`, so the
/// two can be compared from outside — or was misconfigured. `limit` is the quota **of the node that
/// answered**, which is not the same number as the wire limit of one body
/// ([`crate::constants::inference::MAX_HIDDEN_STATE_BYTES`]): the message says which of the two is
/// binding.
pub fn too_many_hop_bodies_error(in_flight: usize, limit: usize) -> DomainError {
    DomainError::Validation(format!(
        "{TOO_MANY_HOP_BODIES_CODE}: [TESERROR-05] this node is already reading {in_flight} hop bodies and admits \
         {limit} at once, so the body of this request was not read (a hop body costs 70-80 MiB of \
         host RAM while it is buffered). The node is full, not broken: nothing of this request \
         was buffered and no prefill was paid for. Retry it later, or route this hop to another \
         candidate."
    ))
}

/// Whether `error` is the refusal of [`too_many_hop_bodies_error`].
pub fn is_too_many_hop_bodies(error: &DomainError) -> bool {
    matches!(error, DomainError::Validation(message) if message.starts_with(TOO_MANY_HOP_BODIES_CODE))
}

/// The code of a **named refusal**, i.e. of a `Validation` whose message starts with one of the codes
/// of this module. `None` for every other failure.
///
/// It is the single detector of the family, and it exists because the decision it feeds is the same
/// for all of them: a refusal is a **terminal** answer, so **retrying it against another candidate
/// cannot fix it** — unlike a dead, busy or degraded hop, which is exactly what retrying is for. The
/// API edge publishes `(status, code, REFUSAL_HEADER)` with it and the router aborts with it
/// (`TASK-68`/S75 for `missing_prefill`, `TASK-76`/S83 for the two size refusals, `TASK-84`/S91 for
/// the quota of hop bodies — the only one whose reason is «not now» instead of «never here»).
pub fn refusal_code(error: &DomainError) -> Option<&'static str> {
    for code in [
        MISSING_PREFILL_CODE,
        PROMPT_TOO_LONG_CODE,
        PAYLOAD_TOO_LARGE_CODE,
        TOO_MANY_HOP_BODIES_CODE,
    ] {
        if matches!(error, DomainError::Validation(message) if message.starts_with(code)) {
            return Some(code);
        }
    }
    None
}

/// The message of a named refusal: the one the HTTP edge puts in the body, with **the code as its
/// first word** (`refusal_message(…)` + `refusal_code(…)` are the two halves the API edge publishes
/// together, `hop_body::named_refusal`).
///
/// It is not [`DomainError`]'s `Display`: that one wraps the message in the name of the variant
/// (`Validation error: …`), and a body that starts with the code is what lets a caller which reads
/// only the body — a log, a `curl`, a harness — detect the refusal without parsing prose. `None` for
/// every failure that is not a refusal, so the caller cannot name one by accident.
pub fn refusal_message(error: &DomainError) -> Option<&str> {
    match error {
        DomainError::Validation(message) if refusal_code(error).is_some() => Some(message),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::inference::{HopCeiling, HopCeilingSource, MAX_HIDDEN_STATE_BYTES};

    fn ceiling(
        tokens: usize,
        source: HopCeilingSource,
        wire: Option<usize>,
        declared: usize,
    ) -> HopCeiling {
        HopCeiling {
            tokens,
            source,
            wire_tokens: wire,
            declared_tokens: declared,
        }
    }

    /// `TASK-76`/S83: which of the two limits binds is part of the refusal, because the operator's
    /// next move is different in each case (lower the prompt, or shard the model differently).
    #[test]
    fn the_refusal_names_which_limit_binds() {
        let wire = prompt_too_long_error(
            12_000,
            3584,
            2,
            &ceiling(9_362, HopCeilingSource::Wire, Some(9_362), 0),
        );
        let text = wire.to_string();
        assert!(is_prompt_too_long(&wire), "{text}");
        assert!(text.contains("max_input_tokens = 0"), "{text}");
        assert!(text.contains("wire is what binds"), "{text}");

        let vram = prompt_too_long_error(
            3_000,
            3584,
            2,
            &ceiling(2_048, HopCeilingSource::Vram, Some(9_362), 2_048),
        );
        let text = vram.to_string();
        assert!(text.contains("prefill ceiling of 2048 tokens"), "{text}");
        assert!(text.contains("prefill is what binds"), "{text}");
        // The wire number is still named: the reader has to see *both* to know how much room there is.
        assert!(text.contains("9362"), "{text}");

        let both = prompt_too_long_error(
            3_000,
            3584,
            2,
            &ceiling(2_048, HopCeilingSource::Both, Some(2_048), 2_048),
        );
        assert!(both.to_string().contains("both limits agree"), "{both}");
    }

    /// The named refusals are detected by one function, and nothing else is a refusal: a `Validation`
    /// of another kind still gets the plain `400` of the API edge.
    #[test]
    fn one_detector_names_the_refusals_and_leaves_the_rest_alone() {
        assert_eq!(
            refusal_code(&missing_prefill_error("s-1", 32, (0, 8))),
            Some(MISSING_PREFILL_CODE)
        );
        assert_eq!(
            refusal_code(&payload_too_large_error(70_000_000, MAX_HIDDEN_STATE_BYTES)),
            Some(PAYLOAD_TOO_LARGE_CODE)
        );
        assert!(is_payload_too_large(&payload_too_large_error(1, 2)));
        assert_eq!(
            refusal_code(&too_many_hop_bodies_error(8, 8)),
            Some(TOO_MANY_HOP_BODIES_CODE)
        );
        assert!(is_too_many_hop_bodies(&too_many_hop_bodies_error(8, 8)));

        let other = DomainError::Validation("a temperature of 3 is not a distribution".into());
        assert_eq!(refusal_code(&other), None);
        assert!(!is_missing_prefill(&other));
        assert!(!is_prompt_too_long(&other));
        assert!(!is_payload_too_large(&other));
        assert!(!is_too_many_hop_bodies(&other));
        assert_eq!(
            refusal_code(&DomainError::Inference("CUDA out of memory".into())),
            None
        );
    }

    /// `TASK-84`/S91: the refusal of a node that is already reading its **quota** of hop bodies says
    /// the two things the operator and the router need — how many are in flight against the limit —
    /// and it says what the refusal cost: nothing, because the body is never read.
    ///
    /// It is the one code of the family whose cause is **not** the bytes of the request (the same
    /// bytes fit here as soon as the node is free), so the message may not repeat the «they do not fit
    /// in any node» prose of the other three: that prose is what the router reads to decide, and here
    /// it would be false.
    #[test]
    fn the_quota_refusal_names_the_in_flight_bodies_and_says_the_body_was_not_read() {
        let error = too_many_hop_bodies_error(8, 8);
        let message = refusal_message(&error).expect("the quota is a named refusal");

        assert!(message.starts_with(TOO_MANY_HOP_BODIES_CODE), "{message}");
        assert!(message.contains("8"), "{message}");
        assert!(
            message.contains("not read"),
            "the refusal has to say it cost nothing: {message}"
        );
        assert!(
            message.contains("Retry"),
            "a full node is not a broken one, and the message says what to do about it: {message}"
        );
        assert!(
            !message.contains("same bytes"),
            "the prose of the size refusals is false here — the same bytes do fit in this node \
             later — and copying it is how the router would be told the wrong thing: {message}"
        );
    }

    /// The refusal of an oversized body names both numbers (what arrived and the limit) and says the
    /// body was never read: that is the difference between this and the `413` axum used to answer.
    #[test]
    fn the_oversized_body_refusal_names_both_numbers() {
        let error = payload_too_large_error(70_000_000, MAX_HIDDEN_STATE_BYTES);
        let message = refusal_message(&error).expect("an oversized body is a named refusal");

        // The body of the HTTP answer: the message, with the code as its first word.
        assert!(message.starts_with(PAYLOAD_TOO_LARGE_CODE), "{message}");
        assert!(message.contains("70000000"), "{message}");
        assert!(message.contains("67108864"), "{message}");
        assert!(message.contains("before buffering"), "{message}");

        // `Display` wraps it in the name of the variant, which is why the edge asks for the message by
        // name and not for `to_string()`.
        let displayed = error.to_string();
        assert!(
            !displayed.starts_with(PAYLOAD_TOO_LARGE_CODE),
            "{displayed}"
        );
        assert!(displayed.contains(message), "{displayed}");

        // A failure that is not one of the three refusals has no message to publish.
        assert!(refusal_message(&DomainError::Inference("CUDA out of memory".into())).is_none());
        assert!(
            refusal_message(&DomainError::Validation("a temperature of 3".into())).is_none(),
            "a plain validation failure is not a refusal"
        );
    }
}
