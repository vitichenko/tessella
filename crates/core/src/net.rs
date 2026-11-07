//! Address arithmetic shared by both workspaces (H-18 · `TASK-14`).
//!
//! The audit found four hand-written copies of the same string work: the worker and the
//! orchestrator both rebuilt `"/ip4/<ip>/tcp/<port>"` with `format!` — without validating the
//! host — and both walked a multiaddr string by hand to read a port back out. A divergence in any
//! of those copies is silent: the address simply stops matching the peer that announced it. The
//! arithmetic therefore lives here, once, with tests.
//!
//! Deliberately **no DNS**: `/ip4/<hostname>` is not a valid multiaddr, so a host that is not a
//! literal IP is rejected (`None`) instead of being formatted into an address that a peer can only
//! fail to parse. That is exactly the defect H-04 fixed on `/p2p_id`.

use std::net::IpAddr;
use std::str::FromStr;

/// Formats a literal IP as a dialable multiaddr with a TCP port: `/ip4/1.2.3.4/tcp/9000`, or
/// `/ip6/2001:db8::1/tcp/9000` for an IPv6 address (libp2p encodes IPv6 **without** brackets).
///
/// Returns `None` when `ip` is not a literal IP address — an empty string, a DNS name or a host
/// that still carries its port (`"1.2.3.4:9000"`) — because none of those can be encoded as an
/// `ip4`/`ip6` component. Surrounding whitespace is ignored.
pub fn multiaddr_tcp(ip: &str, port: u16) -> Option<String> {
    let ip = IpAddr::from_str(ip.trim()).ok()?;
    let version = if ip.is_ipv6() { "ip6" } else { "ip4" };
    Some(format!("/{version}/{ip}/tcp/{port}"))
}

/// Reads the TCP port of a multiaddr string (`/ip4/1.2.3.4/tcp/9000/p2p/12D3KooW…`).
///
/// Returns `None` when the string has no `/tcp/<port>` component or the component is not a valid
/// port. Used instead of walking the protocol list by hand, so the same string is parsed the same
/// way in the orchestrator and in the worker.
pub fn tcp_port_of(multiaddr: &str) -> Option<u16> {
    let mut parts = multiaddr.split('/');
    while let Some(part) = parts.next() {
        if part == "tcp" {
            return parts.next().and_then(|value| value.parse::<u16>().ok());
        }
    }
    None
}

/// Reads the IP component of a multiaddr string (`/ip4/1.2.3.4/tcp/9000` → `1.2.3.4`).
///
/// Returns the first `ip4`/`ip6` component, or `None` when the string has none or it is not a
/// literal IP.
pub fn multiaddr_ip(multiaddr: &str) -> Option<IpAddr> {
    let mut parts = multiaddr.split('/');
    while let Some(part) = parts.next() {
        if part == "ip4" || part == "ip6" {
            return parts.next().and_then(|value| IpAddr::from_str(value).ok());
        }
    }
    None
}

/// Splits an `http(s)` URL into its scheme and its **explicit** port (H-18 · `TASK-14`).
///
/// `None` as the port means "the URL does not spell one out" (the scheme default); callers rebuild
/// the URL with the port they read, or without one. Returns `None` when `url` cannot be parsed at
/// all, so a caller that replaces the host of a malformed URL leaves it untouched instead of
/// inventing one. Surrounding whitespace is ignored.
pub fn url_scheme_and_port(url: &str) -> Option<(String, Option<u16>)> {
    let parsed = url::Url::parse(url.trim()).ok()?;
    Some((parsed.scheme().to_string(), parsed.port()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn multiaddr_tcp_formats_ipv4_and_ipv6_literals() {
        assert_eq!(
            multiaddr_tcp("203.0.113.5", 54326).as_deref(),
            Some("/ip4/203.0.113.5/tcp/54326")
        );
        assert_eq!(
            multiaddr_tcp(" 10.0.0.2 ", 9001).as_deref(),
            Some("/ip4/10.0.0.2/tcp/9001")
        );
        // libp2p encodes IPv6 without brackets, which is what `/p2p_id` had to learn in H-04.
        assert_eq!(
            multiaddr_tcp("2001:db8::1", 9080).as_deref(),
            Some("/ip6/2001:db8::1/tcp/9080")
        );
    }

    #[test]
    fn multiaddr_tcp_rejects_anything_that_is_not_a_literal_ip() {
        // A DNS name produced `/ip4/musivum.dev/tcp/9080/...` (H-04) and an empty host produced
        // `/ip4//tcp/9080/...`; both were advertised verbatim by the old code.
        assert_eq!(multiaddr_tcp("musivum.dev", 9080), None);
        assert_eq!(multiaddr_tcp("", 9080), None);
        assert_eq!(multiaddr_tcp("   ", 9080), None);
        // A host that still carries its port is not an `IpAddr` either.
        assert_eq!(multiaddr_tcp("1.2.3.4:9000", 9000), None);
        assert_eq!(multiaddr_tcp("[2001:db8::1]:9080", 9080), None);
    }

    #[test]
    fn tcp_port_of_reads_the_port_out_of_a_real_multiaddr() {
        assert_eq!(tcp_port_of("/ip4/0.0.0.0/tcp/9080"), Some(9080));
        assert_eq!(
            tcp_port_of("/ip4/203.0.113.5/tcp/54326/p2p/12D3KooWTest"),
            Some(54326)
        );
        assert_eq!(
            tcp_port_of("/ip6/2001:db8::1/tcp/9001/p2p-circuit"),
            Some(9001)
        );
    }

    #[test]
    fn tcp_port_of_is_none_when_there_is_no_usable_port() {
        assert_eq!(tcp_port_of("/ip4/1.2.3.4/udp/9000"), None);
        assert_eq!(tcp_port_of("/ip4/1.2.3.4/tcp"), None);
        assert_eq!(tcp_port_of("/ip4/1.2.3.4/tcp/not-a-port"), None);
        assert_eq!(tcp_port_of(""), None);
    }

    #[test]
    fn multiaddr_ip_reads_the_ip_and_ignores_the_rest() {
        assert_eq!(
            multiaddr_ip("/ip4/203.0.113.5/tcp/54326/p2p/12D3KooWTest"),
            Some(IpAddr::from_str("203.0.113.5").unwrap())
        );
        assert_eq!(
            multiaddr_ip("/ip6/2001:db8::1/tcp/9001"),
            Some(IpAddr::from_str("2001:db8::1").unwrap())
        );
        assert_eq!(multiaddr_ip("/ip4/musivum.dev/tcp/9080"), None);
        assert_eq!(multiaddr_ip("/ip4//tcp/9080"), None);
        assert_eq!(multiaddr_ip("/dns4/example.com/tcp/443"), None);
    }

    #[test]
    fn url_scheme_and_port_separates_the_scheme_from_the_explicit_port() {
        assert_eq!(
            url_scheme_and_port("http://203.0.113.5:54325"),
            Some(("http".to_string(), Some(54325)))
        );
        assert_eq!(
            url_scheme_and_port("http://203.0.113.5:54325/api/v1"),
            Some(("http".to_string(), Some(54325)))
        );
        assert_eq!(
            url_scheme_and_port("https://pod-8080.proxy.runpod.net"),
            Some(("https".to_string(), None))
        );
        assert_eq!(
            url_scheme_and_port("http://10.0.0.2:8080/"),
            Some(("http".to_string(), Some(8080)))
        );
    }

    #[test]
    fn url_scheme_and_port_is_none_for_something_that_is_not_a_url() {
        assert_eq!(url_scheme_and_port(""), None);
        assert_eq!(url_scheme_and_port("not a url"), None);
    }

    #[test]
    fn a_formatted_multiaddr_round_trips_through_both_readers() {
        let addr = multiaddr_tcp("203.0.113.7", 60000).expect("literal ip");
        assert_eq!(tcp_port_of(&addr), Some(60000));
        assert_eq!(
            multiaddr_ip(&addr),
            Some(IpAddr::from_str("203.0.113.7").unwrap())
        );
    }
}
