//! Kademlia discovery end to end, **in memory** (`TASK-15`/S21 · H-06 · D-4=(b)).
//!
//! The card asks for «a two-swarm test **only if** it can be done with an in-memory transport; if it
//! needs loopback sockets, leave it `#[ignore]`d and say so». It can: `MemoryTransport` needs no
//! socket at all, so this file exercises the DHT the way the plan's §1 demands (no daemons, no fixed
//! ports) while still being a real exchange of Kademlia messages between two nodes.
//!
//! What it proves, and why the unit tests in `src/kademlia.rs` are not enough:
//!
//! * two nodes with [`kademlia_behaviour`] and **no** tracker at all find each other (the consumer
//!   adds the provider address out of band, exactly what `plan_bootstrap` feeds it in production);
//! * the provider record of a model survives the wire: `start_providing` on one node becomes a
//!   `GetProvidersOk::FoundProviders` on the other, through a real query/response pair;
//! * `Mode::Server` is what makes that possible: a node left in the library's default auto mode is a
//!   *client* until it confirms an external address, which a worker behind NAT never does, so it
//!   would answer nothing. libp2p-kad 0.45 exposes no `mode()` getter, so the mode is pinned here
//!   *behaviourally*: if the provider did not answer, this test ends in the timeout branch.

use libp2p::core::transport::{MemoryTransport, Transport};
use libp2p::core::upgrade::Version;
use libp2p::futures::StreamExt;
use libp2p::swarm::{Config as SwarmConfig, Swarm, SwarmEvent};
use libp2p::{Multiaddr, PeerId, identity, kad, noise, yamux};
use std::collections::HashSet;
use std::time::Duration;
use tessella_network::kademlia::{kademlia_behaviour, model_record_key};

type KadSwarm = Swarm<kad::Behaviour<kad::store::MemoryStore>>;

/// The model both nodes talk about.
const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";

/// Builds a Kademlia-only node on the in-memory transport, in server mode.
fn memory_node() -> (KadSwarm, PeerId) {
    let key = identity::Keypair::generate_ed25519();
    let peer_id = PeerId::from(key.public());
    let transport = MemoryTransport::default()
        .upgrade(Version::V1)
        .authenticate(noise::Config::new(&key).expect("noise config"))
        .multiplex(yamux::Config::default())
        .boxed();
    let swarm = Swarm::new(
        transport,
        kademlia_behaviour(peer_id, true),
        peer_id,
        // `SwarmConfig::default()` keeps `idle_connection_timeout` at **zero**, so a connection
        // with no pending request is dropped the moment it is established and every DHT query fails
        // locally. The real builders set an explicit timeout for the same reason.
        SwarmConfig::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60)),
    );
    (swarm, peer_id)
}

/// Pumps a swarm until it yields a Kademlia event (every other event is skipped).
async fn next_kad_event(swarm: &mut KadSwarm) -> kad::Event {
    loop {
        match swarm.select_next_some().await {
            SwarmEvent::Behaviour(event) => return event,
            _ => continue,
        }
    }
}

#[tokio::test]
async fn a_provider_is_found_through_the_dht_without_any_tracker() {
    // libp2p-kad explains at `debug!` why a query request failed; without a subscriber those lines
    // are invisible in a red run. `RUST_LOG` decides what is printed (`libp2p_kad=debug`).
    //
    // This is a **global** subscriber (not the scoped `set_default` of `TASK-61`/S67), so it installs
    // a dispatcher for the whole binary and every callsite of this test process computes its
    // `Interest` with it: there is no "another thread registered it first as `never`" to repair here.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let (mut provider, provider_id) = memory_node();
    let (mut consumer, consumer_id) = memory_node();

    // Both listen on an in-memory address (no socket, no fixed port).
    provider
        .listen_on("/memory/0".parse::<Multiaddr>().expect("listen addr"))
        .expect("listen");
    let provider_addr = loop {
        match provider.select_next_some().await {
            SwarmEvent::NewListenAddr { address, .. } => break address,
            _ => continue,
        }
    };
    consumer
        .listen_on("/memory/0".parse::<Multiaddr>().expect("listen addr"))
        .expect("listen");

    // The consumer learns the provider address out of band — exactly what the persistent cache and
    // the tracker response do in production (`plan_bootstrap` hands over this same (peer, addr) pair).
    consumer
        .behaviour_mut()
        .add_address(&provider_id, provider_addr.clone());

    // Connect explicitly first: a Kademlia query needs a *live* connection, and letting the two
    // nodes discover each other through the query itself would make the first attempt race with the
    // connection setup (the DHT retries, but the test should not depend on that luck).
    consumer
        .dial(provider_addr.clone())
        .expect("dial the seed provider");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        tokio::select! {
            _ = next_kad_event(&mut provider) => {}
            event = consumer.select_next_some() => {
                if let SwarmEvent::ConnectionEstablished { peer_id, .. } = event
                    && peer_id == provider_id
                {
                    break;
                }
            }
            () = tokio::time::sleep_until(deadline) => {
                panic!("the consumer never connected to the seed provider");
            }
        }
    }

    // Bootstrap from that single seed: this is the call the card wants to stop being «0 results».
    // The query runs asynchronously, so the provider lookup starts only once the routing table
    // really knows the seed — with `BucketInserts::OnConnected` (the library default) an address
    // added with `add_address` stays *pending* until the connection is established.
    let bootstrap_id = consumer
        .behaviour_mut()
        .bootstrap()
        .expect("one seed is enough to bootstrap");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        tokio::select! {
            _ = next_kad_event(&mut provider) => {}
            event = next_kad_event(&mut consumer) => {
                if let kad::Event::OutboundQueryProgressed {
                    id,
                    result: kad::QueryResult::Bootstrap(Ok(kad::BootstrapOk { num_remaining, .. })),
                    ..
                } = event
                    && id == bootstrap_id
                    && num_remaining == 0
                {
                    break;
                }
            }
            () = tokio::time::sleep_until(deadline) => {
                panic!("the consumer could not bootstrap from the one seed it was given");
            }
        }
    }

    let key = model_record_key(MODEL_ID);
    provider
        .behaviour_mut()
        .start_providing(key.clone())
        .expect("provider record");
    let query_id = consumer.behaviour_mut().get_providers(key.clone());

    // Drive both swarms until the consumer gets an answer: the provider has to *respond*, which is
    // the half that requires `Mode::Server`.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut found: Option<HashSet<PeerId>> = None;
    while found.is_none() {
        tokio::select! {
            event = next_kad_event(&mut provider) => {
                // The publication may still be in flight when the query arrives; both nodes keep
                // being polled, so the only thing this arm has to do is not swallow a failure.
                if let kad::Event::OutboundQueryProgressed {
                    result: kad::QueryResult::StartProviding(Err(e)),
                    ..
                } = event
                {
                    panic!("start_providing failed: {e}");
                }
            }
            event = next_kad_event(&mut consumer) => {
                match event {
                    kad::Event::OutboundQueryProgressed {
                        id,
                        result: kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders { key: answered, providers })),
                        ..
                    } if id == query_id && answered == key => {
                        found = Some(providers.into_iter().collect());
                    }
                    kad::Event::OutboundQueryProgressed {
                        result: kad::QueryResult::GetProviders(Err(e)), ..
                    } => panic!("the provider query failed: {e}"),
                    _ => {}
                }
            }
            () = tokio::time::sleep_until(deadline) => {
                panic!(
                    "the DHT never answered the provider query for {MODEL_ID}: the consumer did not \
                     reach the provider over the in-memory transport"
                );
            }
        }
    }

    let providers = found.expect("answer");
    assert!(
        providers.contains(&provider_id),
        "the provider must be in the answer, got {providers:?}"
    );
    assert!(
        !providers.contains(&consumer_id),
        "a node that does not serve the model must not be announced as its provider"
    );
}
