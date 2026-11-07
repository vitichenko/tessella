//! Gossipsub deduplication **at the swarm level** (the gap `TASK-11`/S5 left, closed by
//! `TASK-12`/S17).
//!
//! S5 replaced the content hash that only the worker used with the library default
//! (`source + sequence number`) and pinned it with two *offline* tests that call
//! `Config::message_id()` on synthetic `gossipsub::Message`s — building a `Swarm` in that module
//! would have meant binding a port in the middle of a step. This file closes the gap the
//! way the card asks: two **real** swarms, subscribed to the real topic, connected over an
//! **ephemeral loopback port** (`/ip4/127.0.0.1/tcp/0`, never a fixed one), exchanging a real
//! message.
//!
//! What it proves, and why each assertion is not redundant with S5:
//!
//! 1. a message published by one peer **arrives** at the other (liveness of the wire);
//! 2. the receiver can publish the **same payload** afterwards — under the removed content-hash
//!    policy this returned `PublishError::Duplicate`, because the id of the message it had just
//!    received was exactly the id of the message it wanted to send;
//! 3. the publisher of the first copy still **receives** the second one — the old policy
//!    suppressed it in the duplicate cache, which is precisely the "legitimate announcement is
//!    dropped" failure H-03 described;
//! 4. the same peer can publish the same payload twice — the sequence number makes the second
//!    one a different message instead of a duplicate.
//!
//! Known limit, stated instead of hidden: the "exact duplicate (`same source + same seqno`) is
//! dropped" half cannot be injected here. A duplicate arriving on the wire would have to be
//! crafted by hand, and `MessageAuthenticity::Signed` rejects a hand-made message (its signature
//! would not verify), while gossipsub's own handler event type is private to the library. That
//! half is therefore pinned where it can be: `Config::message_id()` in
//! `tessella/crates/network/src/lib.rs` (S5), which asserts that the same sender with the same
//! sequence number and different data produces the **same** id.

use libp2p::futures::StreamExt;
use libp2p::gossipsub::{self, PublishError};
use libp2p::{Multiaddr, PeerId, Swarm};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tessella_network::{
    GossipTopic, MoeBehaviour, MoeBehaviourEvent, SwarmEvent, create_worker_swarm, generate_keypair,
};

/// The membership topic both roles use (`WorkerJoined`/`WorkerLeft`). It is the topic whose
/// `MessageId` policy S5 fixed, so the swarm-level test must use it too.
const TOPIC_NAME: &str = tessella_network::WORKERS_TOPIC;

/// The same payload from both senders: the scenario that used to collapse into one message.
const PAYLOAD: &[u8] =
    br#"{"WorkerJoined":{"peer_id":"alice","model_id":"Qwen/Qwen2.5-0.5B-Instruct"}}"#;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Delivered {
    receiver: PeerId,
    source: PeerId,
    data: Vec<u8>,
}

/// Collects a `Message` event, ignoring every other event a swarm emits.
fn record(
    log: &Arc<Mutex<Vec<Delivered>>>,
    receiver: PeerId,
    event: SwarmEvent<MoeBehaviourEvent>,
) {
    if let SwarmEvent::Behaviour(MoeBehaviourEvent::Gossipsub(gossipsub::Event::Message {
        message,
        ..
    })) = event
    {
        log.lock()
            .expect("delivery log is not poisoned")
            .push(Delivered {
                receiver,
                source: message.source.unwrap_or(receiver),
                data: message.data,
            });
    }
}

/// `/ip4/127.0.0.1/tcp/0`: the OS picks the port, so two runs (or two suites in parallel) can
/// never collide on it.
fn ephemeral() -> Multiaddr {
    "/ip4/127.0.0.1/tcp/0"
        .parse()
        .expect("a hard-coded multiaddr must parse")
}

async fn wait_for_listen_addr(swarm: &mut Swarm<MoeBehaviour>) -> anyhow::Result<Multiaddr> {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok(SwarmEvent::NewListenAddr { address, .. }) =
            tokio::time::timeout(Duration::from_millis(500), swarm.select_next_some()).await
        {
            return Ok(address);
        }
    }
    anyhow::bail!("the swarm never reported a listen address")
}

struct Mesh {
    alice: Swarm<MoeBehaviour>,
    bob: Swarm<MoeBehaviour>,
    alice_id: PeerId,
    bob_id: PeerId,
    log: Arc<Mutex<Vec<Delivered>>>,
}

impl Mesh {
    /// Two worker swarms, both subscribed, connected to each other, with a **formed mesh**.
    ///
    /// Waiting for the mesh matters: gossipsub's `publish` picks its recipients from
    /// `mesh[&topic]` as soon as the topic is in the mesh, so publishing to an empty mesh sends
    /// nothing at all (it only falls back to the fanout when the topic is *not* in the mesh).
    async fn connect() -> anyhow::Result<Self> {
        let topic = GossipTopic::new(TOPIC_NAME);
        let mut alice = create_worker_swarm(generate_keypair(), false, true)?;
        let mut bob = create_worker_swarm(generate_keypair(), false, true)?;

        // Subscribe before dialing: the subscription list travels with the connection, so the
        // two peers learn each other's topics without an extra round-trip.
        alice.behaviour_mut().gossipsub.subscribe(&topic)?;
        bob.behaviour_mut().gossipsub.subscribe(&topic)?;

        alice.listen_on(ephemeral())?;
        bob.listen_on(ephemeral())?;
        let bob_addr = wait_for_listen_addr(&mut bob).await?;
        alice.dial(bob_addr)?;

        let mut mesh = Self {
            alice_id: *alice.local_peer_id(),
            bob_id: *bob.local_peer_id(),
            alice,
            bob,
            log: Arc::new(Mutex::new(Vec::new())),
        };

        let topic_hash = topic.hash();
        let formed = mesh
            .drive_until(Duration::from_secs(20), |alice, _bob| {
                alice
                    .behaviour()
                    .gossipsub
                    .mesh_peers(&topic_hash)
                    .next()
                    .is_some()
            })
            .await;
        assert!(
            formed,
            "gossipsub never formed a mesh between the two swarms (is the heartbeat running?)"
        );
        Ok(mesh)
    }

    /// Polls both swarms until `done` holds, the deadline passes or `wait` deliveries arrive.
    ///
    /// Cancelling a branch of the `select!` is safe: `Swarm` buffers its events and only hands
    /// one out when it is ready, so a dropped `select_next_some()` never eats an event.
    async fn drive_until<F>(&mut self, timeout: Duration, mut done: F) -> bool
    where
        F: FnMut(&Swarm<MoeBehaviour>, &Swarm<MoeBehaviour>) -> bool,
    {
        let deadline = Instant::now() + timeout;
        loop {
            if done(&self.alice, &self.bob) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let slice = (deadline - now).min(Duration::from_millis(100));
            let log = self.log.clone();
            let alice_id = self.alice_id;
            let bob_id = self.bob_id;
            tokio::select! {
                event = self.alice.select_next_some() => record(&log, alice_id, event),
                event = self.bob.select_next_some() => record(&log, bob_id, event),
                () = tokio::time::sleep(slice) => {}
            }
        }
    }

    /// Collects deliveries until `want` of them have been logged.
    async fn collect(&mut self, want: usize) -> bool {
        let log = self.log.clone();
        self.drive_until(Duration::from_secs(20), move |_a, _b| {
            log.lock().expect("delivery log is not poisoned").len() >= want
        })
        .await
    }

    fn deliveries(&self) -> Vec<Delivered> {
        self.log
            .lock()
            .expect("delivery log is not poisoned")
            .clone()
    }
}

/// H-03 plus the gap S5 left: what the *swarms* do with the same payload from two senders.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_same_payload_from_two_peers_is_delivered_twice_and_never_suppressed()
-> anyhow::Result<()> {
    let topic = GossipTopic::new(TOPIC_NAME);
    let mut mesh = Mesh::connect().await?;
    let alice_id = mesh.alice_id;
    let bob_id = mesh.bob_id;

    // 1. A fresh announcement on the wire.
    mesh.alice
        .behaviour_mut()
        .gossipsub
        .publish(topic.clone(), PAYLOAD)
        .expect("publishing a fresh announcement must work");
    assert!(
        mesh.collect(1).await,
        "bob never received alice's announcement"
    );

    // 2. The receiver sends the same payload. With the removed content-hash policy this was
    // `PublishError::Duplicate`: the id of the message bob had just received *was* the hash of
    // these bytes, so its own cache claimed it had already published them.
    match mesh
        .bob
        .behaviour_mut()
        .gossipsub
        .publish(topic.clone(), PAYLOAD)
    {
        Ok(_) => {}
        Err(PublishError::Duplicate) => panic!(
            "bob could not announce the same payload alice announced: the message id is not \
             origin-aware (H-03)"
        ),
        Err(e) => panic!("bob could not publish the same announcement: {e}"),
    }

    // 3. ...and the original publisher must receive it: with the old policy alice's duplicate
    // cache already contained the id, so the legitimate announcement of bob was dropped.
    assert!(
        mesh.collect(2).await,
        "alice never received bob's copy of the same payload (was it suppressed as a duplicate?)"
    );

    // 4. A fresh sequence number is a new message, not a duplicate.
    match mesh
        .alice
        .behaviour_mut()
        .gossipsub
        .publish(topic.clone(), PAYLOAD)
    {
        Ok(_) => {}
        Err(PublishError::Duplicate) => {
            panic!("re-publishing with a new sequence number was treated as a duplicate (H-03)")
        }
        Err(e) => panic!("alice could not publish the announcement again: {e}"),
    }
    assert!(
        mesh.collect(3).await,
        "bob never received the second announcement of alice"
    );

    let deliveries = mesh.deliveries();
    assert_eq!(
        deliveries.len(),
        3,
        "unexpected extra deliveries: {deliveries:?}"
    );
    for delivery in &deliveries {
        assert_eq!(
            delivery.data, PAYLOAD,
            "a swarm modified or re-sent something else"
        );
    }
    let alice_to_bob = deliveries
        .iter()
        .filter(|d| d.source == alice_id && d.receiver == bob_id)
        .count();
    let bob_to_alice = deliveries
        .iter()
        .filter(|d| d.source == bob_id && d.receiver == alice_id)
        .count();
    assert_eq!(alice_to_bob, 2, "two copies from alice, none suppressed");
    assert_eq!(bob_to_alice, 1, "one copy from bob, not suppressed either");

    Ok(())
}
