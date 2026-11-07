//! HTTP half of the token-stream capability check (H-10 · `TASK-06` **mitigation 2**).
//!
//! The audit's H-10 had two halves: **where** the final shard is allowed to POST the generated tokens
//! (`tessella_core::callback::validate_callback_url`, mitigation 1) and **who** may write into
//! `/v1/token_stream/{request_id}` (mitigation 2, here). The second half used to live entirely in this
//! crate, which forced the data plane — the engine that **issues** the capabilities while sealing
//! payloads — to depend on the HTTP crate just to share the registry (`SOLID-89`).
//!
//! What moved to [`tessella_core::callback::capability`] is everything that is not transport: the
//! registry ([`CallbackCapabilities`]), the capability derivation ([`callback_capability`]), the
//! decision ([`CallbackAuthz`]) and the TTL / rate-limit constants. What stays here is the part that
//! is genuinely HTTP — reading `Authorization: Bearer <capability>` out of the headers — and it is
//! re-exported under the names the handler already used, so the move is invisible to it.

use std::time::Instant;

use axum::http::{HeaderMap, header};

pub use tessella_core::callback::capability::{
    CALLBACK_CAPABILITY_TTL, CALLBACK_RATE_LIMIT_PER_SECOND, CALLBACK_RATE_LIMIT_WINDOW,
    CallbackAuthz, CallbackCapabilities, MAX_PENDING_CAPABILITIES, callback_capability,
};

/// Scheme of the `Authorization` header (`Bearer`, case-insensitive per RFC 7235).
const BEARER_SCHEME: &str = "Bearer";

/// Reads `Authorization: Bearer <capability>` out of `headers`, or `None` when the header is
/// missing, uses another scheme or carries an empty value.
#[must_use]
pub fn bearer_capability(headers: &HeaderMap) -> Option<&str> {
    let raw = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, value) = raw.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case(BEARER_SCHEME) {
        return None;
    }
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

/// Decides whether a callback may write into the stream of `request_id`.
///
/// Entry point of the handler, kept free of `ApiState` so the decision is unit testable without
/// building an HTTP state. A missing registry (`None`) **fails closed**: the handler cannot tell
/// whether the caller was authorised, so it must not accept the write.
#[must_use]
pub fn authorize_callback(
    capabilities: Option<&CallbackCapabilities>,
    request_id: &str,
    headers: &HeaderMap,
    now: Instant,
) -> CallbackAuthz {
    let Some(capabilities) = capabilities else {
        return CallbackAuthz::Unauthorized;
    };
    let Some(capability) = bearer_capability(headers) else {
        return CallbackAuthz::Unauthorized;
    };
    capabilities.authorize(request_id, capability, now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, HeaderValue::from_str(value).unwrap());
        headers
    }

    fn signature(byte: u8) -> Vec<u8> {
        vec![byte; 64]
    }

    #[test]
    fn bearer_capability_parses_the_authorization_header() {
        assert_eq!(bearer_capability(&headers("Bearer abc")), Some("abc"));
        // The scheme is case-insensitive and padding spaces are tolerated.
        assert_eq!(bearer_capability(&headers("bearer  abc ")), Some("abc"));
        assert_eq!(bearer_capability(&headers("Basic abc")), None);
        assert_eq!(bearer_capability(&headers("Bearer ")), None);
        assert_eq!(bearer_capability(&headers("abc")), None);
        assert_eq!(bearer_capability(&HeaderMap::new()), None);
    }

    #[test]
    fn authorize_callback_fails_closed_on_a_missing_registry_header_or_capability() {
        let now = Instant::now();
        let capabilities = CallbackCapabilities::new();
        capabilities.register("req", &signature(13), now);
        let valid = format!("Bearer {}", callback_capability(&signature(13)));

        assert_eq!(
            authorize_callback(None, "req", &headers(&valid), now),
            CallbackAuthz::Unauthorized
        );
        assert_eq!(
            authorize_callback(Some(&capabilities), "req", &HeaderMap::new(), now),
            CallbackAuthz::Unauthorized
        );
        assert_eq!(
            authorize_callback(Some(&capabilities), "req", &headers("Basic x"), now),
            CallbackAuthz::Unauthorized
        );
        assert_eq!(
            authorize_callback(Some(&capabilities), "req", &headers(&valid), now),
            CallbackAuthz::Accepted
        );
        assert_eq!(
            authorize_callback(Some(&capabilities), "req", &headers(&valid), now),
            CallbackAuthz::Unauthorized
        );
    }
}
