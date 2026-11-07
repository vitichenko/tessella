//! Guard for the `entry_node_url` token-stream callback (H-10).
//!
//! The **final shard** of a distributed inference chain POSTs the generated tokens to
//! `{entry_node_url}/v1/token_stream/{request_id}`. That field travels inside the request
//! payload, so the emitter — or any intermediate hop — can point it at an arbitrary host:
//! SSRF plus exfiltration of the model output. The mitigation lives in one pure function
//! ([`validate_callback_url`]) that the worker applies **before** using *and* before
//! propagating the field, so the destination has to be a node of this mesh (or match an
//! operator allow-list) and private ranges need an explicit opt-in.
//!
//! The predicate [`is_private_ip`] used to live in `musivum/crates/core/src/orchestrator.rs`
//! (SSRF check of `api_addr` at registration); it moved here so both sides share one copy.

pub mod capability;

use std::fmt;
use std::net::IpAddr;
use url::{Host, Url};

/// `true` for addresses that a node-controlled URL must not reach by accident.
///
/// Range-for-range this is the predicate Musivum already used to reject `api_addr`
/// (`orchestrator.rs:34-43`), plus link-local IPv6 (`fe80::/10`), which the original missed
/// even though it is as internal as `169.254.0.0/16`.
pub fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback() || v4.is_link_local() || v4.is_private() || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unicast_link_local()
                || (v6.segments()[0] & 0xfe00 == 0xfc00)
        }
    }
}

/// Why an `entry_node_url` was refused (H-10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallbackRejection {
    /// Not an absolute URL (`""`, `not a url`, `http://`).
    Malformed,
    /// Scheme other than `http`/`https` (`ftp`, `file`, …).
    UnsupportedScheme(String),
    /// The URL carries userinfo (`http://user:pass@host`).
    Credentials,
    /// The URL carries a query or a fragment, so appending the callback path is ambiguous.
    QueryOrFragment,
    /// Private/loopback/link-local/ULA target without `allow_private_addrs`.
    PrivateAddress(String),
    /// Host neither part of the known topology nor covered by the allow-list.
    UnknownHost(String),
}

impl fmt::Display for CallbackRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed => write!(f, "not an absolute URL"),
            Self::UnsupportedScheme(scheme) => write!(
                f,
                "unsupported scheme '{scheme}' (only http/https are accepted)"
            ),
            Self::Credentials => write!(f, "the URL carries userinfo (user:password@host)"),
            Self::QueryOrFragment => write!(
                f,
                "the URL carries a query or a fragment, so appending the callback path is ambiguous"
            ),
            Self::PrivateAddress(host) => write!(
                f,
                "'{host}' is a private/loopback/link-local address (set allow_private_addrs = true \
                 in worker.toml to accept it)"
            ),
            Self::UnknownHost(host) => write!(
                f,
                "'{host}' is neither a node of this mesh nor covered by callback_allowed_hosts"
            ),
        }
    }
}

impl std::error::Error for CallbackRejection {}

/// Allow-list/deny-list rules applied to `entry_node_url` (H-10).
#[derive(Debug, Clone, Default)]
pub struct CallbackTargetPolicy {
    /// Accept private/loopback/link-local targets (`ALLOW_PRIVATE_ADDRS`). Default: `false`.
    pub allow_private_addrs: bool,
    /// Host suffixes accepted even when the host is not in the known topology (reverse proxies
    /// such as `*.proxy.runpod.net`). Empty by default.
    pub allowed_host_suffixes: Vec<String>,
}

impl CallbackTargetPolicy {
    pub fn new(allow_private_addrs: bool, allowed_host_suffixes: Vec<String>) -> Self {
        Self {
            allow_private_addrs,
            allowed_host_suffixes,
        }
    }

    /// `true` when `host` is one of the suffixes or a dot-separated subdomain of it.
    ///
    /// The dot boundary is what keeps `notproxy.runpod.net` out of an allow-list entry for
    /// `proxy.runpod.net`; a bare suffix match would also accept `proxy.runpod.net.evil.example`.
    pub fn allows_host_suffix(&self, host: &str) -> bool {
        allows_host_suffix(&self.allowed_host_suffixes, host)
    }
}

/// Checks whether `host` matches any of the given host suffixes on a dot boundary.
///
/// The dot boundary is what keeps `notproxy.runpod.net` out of an allow-list entry for
/// `proxy.runpod.net`; a bare suffix match would also accept `proxy.runpod.net.evil.example`.
pub fn allows_host_suffix<S: AsRef<str>>(allowed_suffixes: &[S], host: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }
    allowed_suffixes.iter().any(|suffix| {
        let suffix = suffix
            .as_ref()
            .trim()
            .trim_matches('.')
            .to_ascii_lowercase();
        !suffix.is_empty() && (host == suffix || host.ends_with(&format!(".{suffix}")))
    })
}

/// Extracts the (lower-cased, unbracketed) host of a URL, or `None` when it is not parseable.
///
/// Used to turn a peer's `http_addr` from the routing table into a comparable host.
pub fn host_of(raw: &str) -> Option<String> {
    let url = Url::parse(raw.trim()).ok()?;
    url.host_str()
        .map(|host| host.trim_matches(['[', ']']).to_ascii_lowercase())
}

/// Normalises a host entry, which may be a bare host (`10.0.0.5`, `[2001:db8::1]`,
/// `gw.example.com:8443`), bracketed IPv6 with port (`[::1]:9001`), or a full URL (`http://10.0.0.5:9001`).
///
/// The routing table hands over `host_of(peer.http_addr)` (bare host) while a human writing
/// `callback_allowed_hosts` or registration `api_addr` may paste either form; accepting both keeps
/// the comparison honest.
pub fn normalise_host_entry(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(host) = host_of(trimmed) {
        return Some(host);
    }
    // Handle bracketed IPv6 with or without port: "[::1]:9001" or "[2001:db8::1]"
    if let Some(rest) = trimmed.strip_prefix('[')
        && let Some((ipv6, after_bracket)) = rest.split_once(']')
    {
        let after_bracket = after_bracket.trim();
        if after_bracket.is_empty()
            || (after_bracket.starts_with(':')
                && after_bracket[1..].chars().all(|c| c.is_ascii_digit()))
        {
            let ipv6_clean = ipv6.trim().to_ascii_lowercase();
            if !ipv6_clean.is_empty() {
                return Some(ipv6_clean);
            }
        }
    }
    let mut host = trimmed
        .trim_matches(['[', ']'])
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if host.is_empty() {
        return None;
    }
    if let Some((head, port)) = host.rsplit_once(':')
        && !port.is_empty()
        && port.chars().all(|c| c.is_ascii_digit())
        && !head.contains(':')
    {
        host = head.to_string();
    }
    Some(host)
}

/// Parses a comma-separated allow-list of host suffixes (e.g. `proxy.runpod.net,example.com`).
pub fn parse_host_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|entry| entry.trim().trim_matches('.').to_string())
        .filter(|entry| !entry.is_empty())
        .collect()
}

/// Validates `entry_node_url` and returns the normalised base URL to append the callback path to.
///
/// Two independent rules, each with one job (H-10):
///
/// 1. **Membership** — the host must be in `known_hosts` (the peers of the routing table plus the
///    node's own advertised URL) or match an entry of `policy.allowed_host_suffixes`. This is what
///    stops the output of a model from being POSTed to an arbitrary internet host.
/// 2. **Privacy** — an IP literal in a private/loopback/link-local/ULA range additionally needs
///    `policy.allow_private_addrs`. Even a host that claims to be a peer cannot turn this into an
///    SSRF pivot to internal services.
///
/// No DNS resolution is performed (a name is compared as a string), so this guard is cheap, pure
/// and cannot be played with through DNS rebinding; the counterpart is that a name that resolves
/// inside a private range is only as trustworthy as the configuration that admitted it.
pub fn validate_callback_url(
    raw: &str,
    policy: &CallbackTargetPolicy,
    known_hosts: &[String],
) -> Result<String, CallbackRejection> {
    let trimmed = raw.trim();
    let url = Url::parse(trimmed).map_err(|_| CallbackRejection::Malformed)?;

    match url.scheme() {
        "http" | "https" => {}
        other => return Err(CallbackRejection::UnsupportedScheme(other.to_string())),
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(CallbackRejection::Credentials);
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(CallbackRejection::QueryOrFragment);
    }

    let Some(host) = url.host() else {
        return Err(CallbackRejection::Malformed);
    };
    let host_name = host_of(trimmed).unwrap_or_default();
    let member = known_hosts
        .iter()
        .any(|known| normalise_host_entry(known).is_some_and(|h| h == host_name))
        || policy.allows_host_suffix(&host_name);
    if !member {
        return Err(CallbackRejection::UnknownHost(host_name));
    }

    let literal_ip = match host {
        Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
        Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
        Host::Domain(_) => None,
    };
    if let Some(ip) = literal_ip
        && is_private_ip(ip)
        && !policy.allow_private_addrs
    {
        return Err(CallbackRejection::PrivateAddress(host_name));
    }

    Ok(url.as_str().trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(allow_private_addrs: bool, suffixes: &[&str]) -> CallbackTargetPolicy {
        CallbackTargetPolicy::new(
            allow_private_addrs,
            suffixes.iter().map(|s| (*s).to_string()).collect(),
        )
    }

    fn known(hosts: &[&str]) -> Vec<String> {
        hosts.iter().map(|h| (*h).to_string()).collect()
    }

    /// The predicate Musivum already used (`orchestrator.rs:34-43`) keeps rejecting exactly
    /// the same ranges: a move must not change who is refused.
    #[test]
    fn private_ip_predicate_covers_every_range_musivum_already_rejected() {
        for raw in [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.1",
            "192.168.1.7",
            "169.254.169.254",
            "0.0.0.0",
            "::1",
            "::",
            "fd00::1",
        ] {
            let ip: IpAddr = raw.parse().expect("test literal");
            assert!(is_private_ip(ip), "{raw} must stay refused");
        }
        for raw in ["8.8.8.8", "1.1.1.1", "198.51.100.7", "2606:4700::1111"] {
            let ip: IpAddr = raw.parse().expect("test literal");
            assert!(!is_private_ip(ip), "{raw} is a public address");
        }
    }

    /// Improvement over the moved copy: a link-local IPv6 target (`fe80::/10`) is just as
    /// unreachable-and-internal as `169.254.0.0/16`, and the original predicate missed it.
    #[test]
    fn private_ip_predicate_also_rejects_ipv6_link_local() {
        let ip: IpAddr = "fe80::1".parse().expect("test literal");
        assert!(is_private_ip(ip));
    }

    /// Criterion 1 of TASK-06: the cloud metadata endpoint must never be called back.
    #[test]
    fn cloud_metadata_address_is_refused() {
        // Not a peer of this mesh: refused for that reason. Membership is checked first on
        // purpose, so the message does not invite flipping `allow_private_addrs`, which would
        // not help a host that is not a node anyway.
        let err = validate_callback_url(
            "http://169.254.169.254/latest/meta-data/",
            &policy(false, &[]),
            &known(&[]),
        )
        .expect_err("169.254.169.254 must not be a callback target");
        assert_eq!(
            err,
            CallbackRejection::UnknownHost("169.254.169.254".to_string())
        );

        // Even admitted as a peer (poisoned topology), a link-local address needs the opt-in,
        // and the message names the knob that turns it on.
        let err = validate_callback_url(
            "http://169.254.169.254/latest/meta-data/",
            &policy(false, &[]),
            &known(&["169.254.169.254"]),
        )
        .expect_err("a link-local address must need allow_private_addrs");
        assert_eq!(
            err,
            CallbackRejection::PrivateAddress("169.254.169.254".to_string())
        );
        assert!(err.to_string().contains("allow_private_addrs"));
    }

    /// Criterion 1 of TASK-06: a private target needs `ALLOW_PRIVATE_ADDRS`.
    #[test]
    fn private_targets_need_the_explicit_flag() {
        for raw in [
            "http://10.0.0.5:9001",
            "http://127.0.0.1:10001",
            "http://[fd00::1]:9001",
        ] {
            assert!(
                validate_callback_url(raw, &policy(false, &[]), &known(&[])).is_err(),
                "{raw} must be refused without the flag"
            );
        }
        assert_eq!(
            validate_callback_url(
                "http://10.0.0.5:9001",
                &policy(true, &[]),
                &known(&["10.0.0.5"])
            )
            .expect("allowed private target"),
            "http://10.0.0.5:9001"
        );
    }

    /// Defence in depth: being a known peer does **not** waive the private-address opt-in, so
    /// a poisoned topology entry cannot turn the guard into an SSRF proxy for internal IPs.
    #[test]
    fn a_known_peer_still_needs_the_flag_when_it_is_private() {
        let err = validate_callback_url(
            "http://10.0.0.5:9001",
            &policy(false, &[]),
            &known(&["10.0.0.5"]),
        )
        .expect_err("known private peer must still need the flag");
        assert_eq!(
            err,
            CallbackRejection::PrivateAddress("10.0.0.5".to_string())
        );

        // An allow-list suffix does not waive it either.
        assert!(
            validate_callback_url(
                "http://10.0.0.5/",
                &policy(false, &["10.0.0.5"]),
                &known(&[])
            )
            .is_err()
        );
    }

    /// Criterion 2 of TASK-06: a host outside the mesh is refused, whatever its IP class.
    #[test]
    fn unknown_hosts_are_refused() {
        for raw in [
            "http://evil.example.com/v1/token_stream/x",
            "http://198.51.100.7:9001",
            "https://attacker.example",
        ] {
            let err = validate_callback_url(raw, &policy(false, &[]), &known(&["198.51.100.6"]))
                .expect_err("a host outside the topology must be refused");
            assert!(
                matches!(err, CallbackRejection::UnknownHost(_)),
                "{raw} produced {err:?}"
            );
        }
    }

    /// Criterion 2, positive half: a peer advertised by `/peers` is a legitimate destination.
    /// The known-host list accepts both the bare host the routing table derives and a full URL.
    #[test]
    fn known_topology_hosts_are_accepted() {
        assert_eq!(
            validate_callback_url(
                "http://198.51.100.7:9001/",
                &policy(false, &[]),
                &known(&["198.51.100.7", "203.0.113.9"])
            )
            .expect("known peer"),
            "http://198.51.100.7:9001"
        );
        assert!(
            validate_callback_url(
                "http://198.51.100.7:9001/",
                &policy(false, &[]),
                &known(&["http://198.51.100.7:9001", "GW.Example.com"])
            )
            .is_ok(),
            "a full URL (or a differently-cased host) must match too"
        );
    }

    /// The suffix allow-list is the escape hatch for proxies whose advertised host is not in
    /// the topology, and it matches on a dot boundary only (`notrunpod.net` must not sneak in).
    #[test]
    fn allowed_suffixes_match_only_on_a_dot_boundary() {
        let p = policy(false, &["proxy.runpod.net"]);
        assert!(
            validate_callback_url("https://abc-8080.proxy.runpod.net", &p, &known(&[])).is_ok()
        );
        assert!(validate_callback_url("https://proxy.runpod.net", &p, &known(&[])).is_ok());
        for raw in [
            "https://notproxy.runpod.net",
            "https://proxy.runpod.net.evil.example",
            "https://runpod.net",
        ] {
            assert!(
                validate_callback_url(raw, &p, &known(&[])).is_err(),
                "{raw} must not match the suffix"
            );
        }
    }

    /// The node's own advertised URL (a known host by construction) is accepted, and the value
    /// is returned normalised: no trailing slash, no surrounding whitespace, so the caller can
    /// append `/v1/token_stream/{id}` without building a broken URL.
    #[test]
    fn own_url_with_a_base_path_is_normalised() {
        assert_eq!(
            validate_callback_url(
                "  http://203.0.113.9:9001/edge/  ",
                &policy(false, &["example.com"]),
                &known(&["203.0.113.9"])
            )
            .expect("own url"),
            "http://203.0.113.9:9001/edge"
        );
        assert_eq!(
            validate_callback_url(
                "https://gw.example.com/edge/",
                &policy(false, &["example.com"]),
                &known(&[])
            )
            .expect("suffix-allowed url"),
            "https://gw.example.com/edge"
        );
    }

    /// Malformed values, other schemes, embedded credentials and a query/fragment are all
    /// refused: appending a path to them would be ambiguous or would leak credentials.
    #[test]
    fn malformed_or_unsupported_values_are_refused() {
        let p = policy(false, &[]);
        let cases = [
            ("", CallbackRejection::Malformed),
            ("not a url", CallbackRejection::Malformed),
            ("http://", CallbackRejection::Malformed),
            (
                "ftp://198.51.100.7",
                CallbackRejection::UnsupportedScheme("ftp".to_string()),
            ),
            (
                "file:///etc/passwd",
                CallbackRejection::UnsupportedScheme("file".to_string()),
            ),
            (
                "http://user:pass@198.51.100.7",
                CallbackRejection::Credentials,
            ),
            (
                "http://198.51.100.7/?a=b",
                CallbackRejection::QueryOrFragment,
            ),
            (
                "http://198.51.100.7/#frag",
                CallbackRejection::QueryOrFragment,
            ),
        ];
        for (raw, expected) in cases {
            let err = validate_callback_url(raw, &p, &known(&["198.51.100.7"]))
                .expect_err("the value must be refused");
            assert_eq!(err, expected, "unexpected rejection for {raw:?}");
        }
    }

    /// `host_of` is how the caller turns a peer's `http_addr` into a comparable host.
    #[test]
    fn host_of_extracts_the_host_lowercased() {
        assert_eq!(
            host_of("http://203.0.113.9:9001").as_deref(),
            Some("203.0.113.9")
        );
        assert_eq!(
            host_of("https://GW.Example.COM/edge").as_deref(),
            Some("gw.example.com")
        );
        assert_eq!(
            host_of("http://[2001:db8::1]:9001").as_deref(),
            Some("2001:db8::1")
        );
        assert_eq!(host_of("not a url"), None);
    }

    #[test]
    fn normalise_host_entry_handles_schemes_ports_and_ipv6() {
        assert_eq!(
            normalise_host_entry("http://1.2.3.4:9001").as_deref(),
            Some("1.2.3.4")
        );
        assert_eq!(
            normalise_host_entry("1.2.3.4:9001").as_deref(),
            Some("1.2.3.4")
        );
        assert_eq!(
            normalise_host_entry("https://abc-8080.proxy.runpod.net").as_deref(),
            Some("abc-8080.proxy.runpod.net")
        );
        assert_eq!(
            normalise_host_entry("http://[::1]:9001").as_deref(),
            Some("::1")
        );
        assert_eq!(normalise_host_entry("[::1]:9001").as_deref(), Some("::1"));
        assert_eq!(normalise_host_entry("::1").as_deref(), Some("::1"));
        assert_eq!(
            normalise_host_entry("[2001:db8::1]").as_deref(),
            Some("2001:db8::1")
        );
        assert_eq!(
            normalise_host_entry("gw.example.com:8443").as_deref(),
            Some("gw.example.com")
        );
        assert_eq!(normalise_host_entry(""), None);
        assert_eq!(normalise_host_entry("   "), None);
    }

    #[test]
    fn test_allows_host_suffix_and_parse_list() {
        let suffixes = parse_host_list("proxy.runpod.net, .example.com. ");
        assert_eq!(suffixes, vec!["proxy.runpod.net", "example.com"]);

        assert!(allows_host_suffix(&suffixes, "abc-8080.proxy.runpod.net"));
        assert!(allows_host_suffix(&suffixes, "proxy.runpod.net"));
        assert!(allows_host_suffix(&suffixes, "sub.example.com"));
        assert!(!allows_host_suffix(&suffixes, "notproxy.runpod.net"));
        assert!(!allows_host_suffix(&suffixes, "proxy.runpod.net.evil.com"));
        assert!(!allows_host_suffix(
            &Vec::<String>::new(),
            "abc-8080.proxy.runpod.net"
        ));
    }
}
