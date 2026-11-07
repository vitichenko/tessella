//! H-10: decides whether the final shard may POST generated tokens back to `entry_node_url`.
//!
//! [`CallbackGuard`] is the worker-side half of `tessella_core::callback`: it feeds the pure
//! validator with the hosts this node actually knows (its own advertised URL plus every peer of
//! the routing table) and turns a refusal into a warning plus «drop the field». Dropping it is
//! safe by design: the token-stream callback is an optimisation of the SSE path, so the tokens
//! still reach the client through the synchronous call stack (see `forward_hidden`).

use crate::config::WorkerConfig;
use crate::routing_table::RoutingTable;
use std::sync::Arc;
use tessella_core::callback::{host_of, validate_callback_url, CallbackTargetPolicy};
use tokio::sync::RwLock;
use tracing::warn;

/// Decides whether an `entry_node_url` may be used as a token-stream callback target (H-10).
pub struct CallbackGuard {
    policy: CallbackTargetPolicy,
    /// This node's own advertised URL — a legitimate destination by construction.
    own_api_url: Option<String>,
    /// Peers announced by the orchestrator: the source of «hosts of this mesh».
    routing_table: Arc<RwLock<RoutingTable>>,
}

impl CallbackGuard {
    pub fn new(
        policy: CallbackTargetPolicy,
        own_api_url: Option<String>,
        routing_table: Arc<RwLock<RoutingTable>>,
    ) -> Self {
        Self {
            policy,
            own_api_url,
            routing_table,
        }
    }

    /// Builds the guard from `worker.toml`/env (`allow_private_addrs`, `callback_allowed_hosts`).
    pub fn from_config(
        config: &WorkerConfig,
        own_api_url: Option<String>,
        routing_table: Arc<RwLock<RoutingTable>>,
    ) -> Self {
        Self::new(
            CallbackTargetPolicy::new(
                config.allow_private_addrs,
                config.callback_allowed_hosts.clone(),
            ),
            own_api_url,
            routing_table,
        )
    }

    /// Hosts of this mesh as known right now: the peers of the routing table plus this node.
    ///
    /// The routing table is read at call time on purpose — a peer that joined after boot is
    /// honoured without a restart, and one that left stops being a valid destination.
    pub async fn known_hosts(&self) -> Vec<String> {
        let mut hosts: Vec<String> = self
            .own_api_url
            .as_deref()
            .and_then(host_of)
            .into_iter()
            .collect();
        let table = self.routing_table.read().await;
        for peer in table.peers() {
            if let Some(host) = host_of(&peer.http_addr) {
                if !hosts.contains(&host) {
                    hosts.push(host);
                }
            }
        }
        hosts
    }

    /// Returns the normalised base URL to call back, or `None` when the destination must be
    /// dropped (with a warning that names the reason and the knob that would admit it).
    pub async fn sanitize(&self, raw: &str) -> Option<String> {
        let known_hosts = self.known_hosts().await;
        match validate_callback_url(raw, &self.policy, &known_hosts) {
            Ok(target) => Some(target),
            Err(rejection) => {
                warn!(
                    "🚫 Refusing token-stream callback to '{}': {}. Dropping entry_node_url for \
                     this request; the tokens still travel on the synchronous path (H-10)",
                    raw, rejection
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessella_core::models::LayerPeer;

    fn peer(peer_id: &str, http_addr: &str) -> LayerPeer {
        LayerPeer {
            peer_id: peer_id.to_string(),
            http_addr: http_addr.to_string(),
            p2p_addr: "/ip4/203.0.113.9/tcp/9001".to_string(),
            layer_ranges: vec![(0, 10)],
            expert_range: None,
            queue_depth: 0,
            public_ip: None,
            public_api_port: None,
            canary: false,
        }
    }

    /// Builds a guard over a routing table filled synchronously (`insert_peer` is not async).
    fn guard(
        allow_private_addrs: bool,
        allowed_hosts: &[&str],
        own_url: Option<&str>,
        peers: &[(&str, &str)],
    ) -> CallbackGuard {
        let mut table = RoutingTable::new();
        for (peer_id, http_addr) in peers {
            table.insert_peer(peer(peer_id, http_addr));
        }
        let config = WorkerConfig {
            allow_private_addrs,
            callback_allowed_hosts: allowed_hosts.iter().map(|h| (*h).to_string()).collect(),
            ..Default::default()
        };
        CallbackGuard::from_config(
            &config,
            own_url.map(std::string::ToString::to_string),
            Arc::new(RwLock::new(table)),
        )
    }

    #[tokio::test]
    async fn known_hosts_lists_the_own_url_and_every_peer_without_duplicates() {
        let g = guard(
            false,
            &[],
            Some("http://203.0.113.9:9001"),
            &[
                ("peer-1", "http://198.51.100.7:9002"),
                ("peer-2", "http://203.0.113.9:9002"),
            ],
        );
        let hosts = g.known_hosts().await;
        assert!(hosts.contains(&"203.0.113.9".to_string()), "{hosts:?}");
        assert!(hosts.contains(&"198.51.100.7".to_string()), "{hosts:?}");
        assert_eq!(
            hosts.iter().filter(|h| *h == "203.0.113.9").count(),
            1,
            "the same host advertised twice must be listed once: {hosts:?}"
        );
    }

    #[tokio::test]
    async fn sanitize_keeps_the_url_of_a_known_peer() {
        let g = guard(false, &[], None, &[("peer-1", "http://198.51.100.7:9002")]);
        assert_eq!(
            g.sanitize("http://198.51.100.7:9002/").await.as_deref(),
            Some("http://198.51.100.7:9002")
        );
    }

    #[tokio::test]
    async fn sanitize_keeps_the_nodes_own_url() {
        let g = guard(false, &[], Some("http://203.0.113.9:9001/"), &[]);
        assert_eq!(
            g.sanitize("http://203.0.113.9:9001").await.as_deref(),
            Some("http://203.0.113.9:9001")
        );
    }

    /// The end of H-10: a foreign destination is dropped before it can be used *or* propagated.
    #[tokio::test]
    async fn sanitize_drops_a_destination_that_is_not_a_node_of_the_mesh() {
        let g = guard(false, &[], Some("http://203.0.113.9:9001"), &[]);
        for raw in [
            "http://attacker.example/v1/token_stream/x",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5:9001",
            "",
            "not a url",
            "ftp://203.0.113.9",
        ] {
            assert_eq!(g.sanitize(raw).await, None, "{raw:?} must be dropped");
        }
    }

    #[tokio::test]
    async fn sanitize_honours_allow_private_addrs_and_the_allow_list() {
        let private_peer = &[("peer-1", "http://10.0.0.5:9002")];

        // Known peer + explicit opt-in ⇒ accepted (RunPod-style private network).
        let permissive = guard(true, &[], None, private_peer);
        assert_eq!(
            permissive.sanitize("http://10.0.0.5:9002").await.as_deref(),
            Some("http://10.0.0.5:9002")
        );

        // Known peer but no opt-in ⇒ still refused (defence in depth).
        let strict = guard(false, &[], None, private_peer);
        assert_eq!(strict.sanitize("http://10.0.0.5:9002").await, None);

        // A reverse proxy whose host is not in the topology is admitted by suffix.
        let proxied = guard(false, &["proxy.runpod.net"], None, &[]);
        assert_eq!(
            proxied
                .sanitize("https://abc-8080.proxy.runpod.net/")
                .await
                .as_deref(),
            Some("https://abc-8080.proxy.runpod.net")
        );
    }
}
