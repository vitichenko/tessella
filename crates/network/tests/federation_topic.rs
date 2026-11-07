//! The federation notice topic, end to end and **in memory** (`TASK-54`/S60 · D-14).
//!
//! The card asks for an offline test in `tessella-network` proving that «publishing on
//! `FEDERATION_TOPIC` delivers the payload to the subscriber», and names `kademlia_discovery.rs` as
//! the pattern. So this file builds two real swarms on `MemoryTransport` (no socket, no fixed port,
//! nothing to bind — §1 of the plan holds) and exchanges a real gossipsub message.
//!
//! What each assertion is for:
//!
//! 1. the constant names a topic gossipsub actually delivers on: the payload published by one node
//!    arrives at the other, byte for byte, with the sender's authenticated `source` — that source is
//!    exactly what the replica uses to admit a notice («only the pinned primary», the same rule S22
//!    established: gossip is a signal, never a door);
//! 2. the routing is the **subscription**, not luck: a message published on the membership topic,
//!    which this second node never subscribes to, is not delivered — so assertion 1 is not passing
//!    because every publish reaches every peer.
//!
//! Known limit, stated instead of hidden: this file proves the *wire*. That the notice is published
//! only by a primary and only triggers a pull in a replica is Rust-level logic, tested in
//! `musivum-api` (the pure decode/admission functions) and in `musivum --lib` (the loop that answers
//! a notice with a pull); the end-to-end latency is measured by the harness
//! (`scripts/e2e/swarm.sh --replica --scenario federation_replica`).

use libp2p::core::transport::{MemoryTransport, Transport};
use libp2p::core::upgrade::Version;
use libp2p::futures::StreamExt;
use libp2p::gossipsub;
use libp2p::swarm::{Config as SwarmConfig, Swarm, SwarmEvent};
use libp2p::{Multiaddr, PeerId, identity, noise, yamux};
use std::time::{Duration, Instant};
use tessella_network::{FEDERATION_TOPIC, GossipTopic, WORKERS_TOPIC, gossipsub_config};

/// What the primary publishes and the replica must receive.
const NOTICE: &[u8] =
    br#"{"version":1,"change":"Ledger","ledger_height":7,"published_at":1700000000}"#;

/// A minimal gossipsub node on the in-memory transport.
///
/// Deliberately **smaller** than `create_federation_swarm`: what has to be shared between the two
/// orchestrators is the topic string and the gossipsub policy (validation mode, message ids), and
/// both come from [`gossipsub_config`]. Kademlia, identify and relay play no part in delivering a
/// notice, and building them here would only add machinery that cannot fail differently.
fn memory_node() -> (Swarm<gossipsub::Behaviour>, PeerId) {
    let key = identity::Keypair::generate_ed25519();
    let peer_id = PeerId::from(key.public());
    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(key.clone()),
        gossipsub_config().expect("gossipsub config"),
    )
    .expect("gossipsub behaviour");

    let transport = MemoryTransport::default()
        .upgrade(Version::V1)
        .authenticate(noise::Config::new(&key).expect("noise config"))
        .multiplex(yamux::Config::default())
        .boxed();
    let swarm = Swarm::new(
        transport,
        gossipsub,
        peer_id,
        // `SwarmConfig::default()` keeps `idle_connection_timeout` at **zero**, so a connection with
        // no pending message is dropped the moment it is established and the mesh never forms. The
        // production builders set an explicit timeout for the same reason.
        SwarmConfig::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60)),
    );
    (swarm, peer_id)
}

/// A message as the subscriber sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Received {
    topic: gossipsub::TopicHash,
    source: Option<PeerId>,
    data: Vec<u8>,
}

/// Drives both swarms until `done` holds or `timeout` runs out, collecting every received message.
///
/// Cancelling a branch of the `select!` is safe: a `Swarm` buffers its events and only hands one out
/// when it is ready, so a dropped `select_next_some()` never eats an event.
async fn drive_until<F>(
    a: &mut Swarm<gossipsub::Behaviour>,
    b: &mut Swarm<gossipsub::Behaviour>,
    received: &mut Vec<Received>,
    timeout: Duration,
    mut done: F,
) where
    F: FnMut(&Swarm<gossipsub::Behaviour>, &Swarm<gossipsub::Behaviour>, &[Received]) -> bool,
{
    let deadline = Instant::now() + timeout;
    loop {
        if done(a, b, received) || Instant::now() >= deadline {
            return;
        }
        let slice = (deadline - Instant::now()).min(Duration::from_millis(100));
        tokio::select! {
            event = a.select_next_some() => record(event, received),
            event = b.select_next_some() => record(event, received),
            () = tokio::time::sleep(slice) => {}
        }
    }
}

fn record(event: SwarmEvent<gossipsub::Event>, received: &mut Vec<Received>) {
    if let SwarmEvent::Behaviour(gossipsub::Event::Message { message, .. }) = event {
        received.push(Received {
            topic: message.topic,
            source: message.source,
            data: message.data,
        });
    }
}

/// The criterion of `TASK-54`: one constant, both roles, and the payload really crosses the wire.
#[tokio::test]
async fn a_notice_published_on_the_federation_topic_reaches_the_subscriber() {
    let primary_topic = GossipTopic::new(FEDERATION_TOPIC);
    let (mut primary, primary_id) = memory_node();
    let (mut replica, replica_id) = memory_node();

    // Both roles subscribe to **the same** constant (a primary only publishes, but subscribing costs
    // nothing and keeps the two roles symmetric if a future step wants the replica to answer).
    primary
        .behaviour_mut()
        .subscribe(&primary_topic)
        .expect("primary subscribes");
    replica
        .behaviour_mut()
        .subscribe(&primary_topic)
        .expect("replica subscribes");

    primary
        .listen_on("/memory/0".parse::<Multiaddr>().expect("listen addr"))
        .expect("listen");
    let primary_addr = loop {
        match primary.select_next_some().await {
            SwarmEvent::NewListenAddr { address, .. } => break address,
            _ => continue,
        }
    };
    replica
        .dial(primary_addr)
        .expect("dial the primary over the in-memory transport");

    // The mesh has to be **formed** before publishing: `publish` picks its recipients from
    // `mesh[&topic]`, so publishing into an empty mesh sends nothing at all (the lesson `TASK-12`/S17
    // left in `gossipsub_mesh.rs`).
    let topic_hash = primary_topic.hash();
    let mut received: Vec<Received> = Vec::new();
    drive_until(
        &mut primary,
        &mut replica,
        &mut received,
        Duration::from_secs(20),
        |primary, _, _| primary.behaviour().mesh_peers(&topic_hash).next().is_some(),
    )
    .await;
    assert!(
        primary.behaviour().mesh_peers(&topic_hash).next().is_some(),
        "gossipsub never formed a mesh for {FEDERATION_TOPIC} (is the heartbeat running?)"
    );

    primary
        .behaviour_mut()
        .publish(primary_topic.clone(), NOTICE)
        .expect("publishing a notice must work");

    drive_until(
        &mut primary,
        &mut replica,
        &mut received,
        Duration::from_secs(20),
        |_, _, received| !received.is_empty(),
    )
    .await;

    assert_eq!(
        received.len(),
        1,
        "the subscriber must see exactly one notice: {received:?}"
    );
    let notice = &received[0];
    assert_eq!(
        notice.topic, topic_hash,
        "it arrived on the federation topic"
    );
    assert_eq!(notice.data, NOTICE, "the payload crossed the wire intact");
    assert_eq!(
        notice.source,
        Some(primary_id),
        "and gossipsub authenticates the sender, which is what a replica admits by"
    );
    assert_ne!(notice.source, Some(replica_id));

    // The negative control: the second node never subscribed to the membership topic, so gossipsub
    // refuses the publish outright (`InsufficientPeers`). That is proof the delivery above was the
    // **subscription** on `FEDERATION_TOPIC` and not «every publish reaches every peer».
    match primary
        .behaviour_mut()
        .publish(GossipTopic::new(WORKERS_TOPIC), NOTICE)
    {
        Err(gossipsub::PublishError::InsufficientPeers) => {}
        other => panic!(
            "a topic nobody subscribed to must not be publishable (it was {other:?}); if this \
             changed, the assertion above no longer proves the subscription is what routes"
        ),
    }
    assert_eq!(
        received.len(),
        1,
        "and no delivery happened for the other topic: {received:?}"
    );
}
