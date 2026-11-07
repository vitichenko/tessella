use anyhow::Result;
use libp2p::{
    Multiaddr as Libp2pMultiaddr, PeerId, Swarm, SwarmBuilder, dcutr,
    gossipsub::{self, MessageAuthenticity},
    identify,
    identity::Keypair as Libp2pKeypair,
    kad, mdns, noise, relay,
    swarm::NetworkBehaviour,
    tcp, yamux,
};

use std::time::Duration;

// Re-export wrappers
pub type NetworkKeypair = Libp2pKeypair;
pub type NetworkPeerId = PeerId;
pub type NetworkAddress = Libp2pMultiaddr;

pub fn generate_keypair() -> NetworkKeypair {
    Libp2pKeypair::generate_ed25519()
}

pub fn keypair_from_bytes(bytes: &[u8]) -> Result<NetworkKeypair> {
    Libp2pKeypair::from_protobuf_encoding(bytes).map_err(Into::into)
}

pub fn keypair_to_bytes(kp: &NetworkKeypair) -> Result<Vec<u8>> {
    kp.to_protobuf_encoding().map_err(Into::into)
}

pub fn peer_id_from_keypair(kp: &NetworkKeypair) -> String {
    PeerId::from(kp.public()).to_string()
}

/// The **raw** Ed25519 public key (32 bytes, RFC 8032) of `kp` (`TASK-19`).
///
/// This is the representation an operator pins in the worker (`orchestrator_public_key`) and the
/// one `/p2p_id` publishes as `orchestrator_pubkey`: 64 hex chars, ready to paste. The *wire*
/// format of signatures is the protobuf encoding (72 hex, see [`peer_id_from_public_key_bytes`]);
/// both describe the same key, so this helper is what keeps the published value and the pinned
/// value from drifting apart.
///
/// # Errors
///
/// Returns an error if the keypair is not Ed25519 (libp2p supports other curves).
pub fn ed25519_public_key_bytes(kp: &NetworkKeypair) -> Result<[u8; 32]> {
    let ed25519 = kp
        .public()
        .try_into_ed25519()
        .map_err(|e| anyhow::anyhow!("not an ed25519 keypair: {}", e))?;
    Ok(ed25519.to_bytes())
}

/// Lowercase hex (64 chars) of [`ed25519_public_key_bytes`].
///
/// # Errors
///
/// Same as [`ed25519_public_key_bytes`].
pub fn ed25519_public_key_hex(kp: &NetworkKeypair) -> Result<String> {
    Ok(hex::encode(ed25519_public_key_bytes(kp)?))
}

/// Derives the libp2p `PeerId` from a **protobuf-encoded** public key, i.e. from exactly
/// the bytes a peer publishes with `Keypair::public().encode_protobuf()` (36 bytes ⇒ 72
/// hex chars for ed25519 — *not* the 32 raw key bytes some callers assume).
///
/// Consumers use this to bind an advertised key to the identity that owns a resource
/// (H-13: `GET /api/v1/session_grant` issues the ticket for the `PeerId` derived from the
/// requester's `pubkey`, so quoting somebody else's `peer_id` buys nothing; the standalone
/// `/api/v1/karma_ticket` was retired in `TASK-58`/S64).
///
/// # Errors
///
/// Returns an error if the bytes are not a decodable libp2p public key (a raw key, a
/// secret key or arbitrary bytes are all rejected), so callers can answer `400`.
pub fn peer_id_from_public_key_bytes(pubkey_bytes: &[u8]) -> Result<String> {
    let pubkey = libp2p::identity::PublicKey::try_decode_protobuf(pubkey_bytes)?;
    Ok(PeerId::from_public_key(&pubkey).to_string())
}

pub fn verify_signature(pubkey_bytes: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool> {
    if pubkey_bytes.is_empty() || sig.is_empty() {
        return Ok(false);
    }
    let pubkey = libp2p::identity::PublicKey::try_decode_protobuf(pubkey_bytes)?;
    Ok(pubkey.verify(msg, sig))
}

/// Signs a peer-authenticated HTTP DTO with `kp` (D-3 · `TASK-03` Step B).
///
/// Fills in **both** authentication fields of `req`: `public_key` with
/// `kp.public().encode_protobuf()` (the protobuf encoding that
/// [`peer_id_from_public_key_bytes`] and [`verify_signature`] decode) and `signature` with the
/// Ed25519 signature of [`PeerSignedRequest::peer_canonical_bytes`], which excludes those same
/// two fields. Callers must sign right before sending, after every other field is final.
///
/// # Errors
///
/// Returns an error when the request cannot be canonicalized or the key cannot sign.
pub fn sign_peer_request<T: tessella_protocol::PeerSignedRequest>(
    kp: &NetworkKeypair,
    req: &mut T,
) -> Result<()> {
    let canonical = req.peer_canonical_bytes()?;
    let signature = kp.sign(&canonical)?;
    req.set_peer_auth(kp.public().encode_protobuf(), signature);
    Ok(())
}

/// Fills in **all** signature fields of an inference request with the canonical preimage v1
/// (H-09 · `TASK-05`): `signature_version = 1`, the caller's `nonce`/`iat`, the Ed25519 signature
/// over [`tessella_protocol::signing::SignContext::v1_preimage`] and the protobuf public key.
///
/// `iat` is a parameter instead of `SystemTime::now()` so the sealer is deterministic and can be
/// tested offline; callers pass the current Unix seconds. `nonce` must be unique per emitted
/// payload (the entry node generates one per decoding step), otherwise a legitimate multi-step
/// session would look like a replay.
///
/// # Errors
///
/// Returns an error when the preimage cannot be serialized or the key cannot sign.
pub fn seal_signed_request<T: tessella_protocol::signing::SignedRequest>(
    kp: &NetworkKeypair,
    req: &mut T,
    nonce: u64,
    iat: u64,
) -> Result<()> {
    // Write version/nonce/iat first: the preimage is computed from the request itself, so a seal
    // can never cover a stale triple.
    req.set_signature_fields(
        tessella_protocol::signing::SIGNATURE_VERSION_V1,
        nonce,
        iat,
        Vec::new(),
        Vec::new(),
    );
    let preimage = req.signed_bytes()?;
    let signature = kp.sign(&preimage)?;
    req.set_signature_fields(
        tessella_protocol::signing::SIGNATURE_VERSION_V1,
        nonce,
        iat,
        signature,
        kp.public().encode_protobuf(),
    );
    Ok(())
}

/// Why a signed inference request was rejected (H-09 · `TASK-05`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignedRequestError {
    /// No `client_pubkey`/`signature` at all.
    MissingSignature,
    /// `signature_version` is not `1` (canonical). `0` is the **retired** legacy preimage — what an
    /// unsealed request carries — and anything else is unknown; both are rejected before the
    /// signature is computed.
    UnsupportedVersion(u8),
    /// The preimage could not be built or the key could not be decoded.
    Malformed(String),
    /// The signature does not verify against the declared key and version.
    InvalidSignature,
    /// `iat` is too far in the future (± `SIGNATURE_CLOCK_SKEW_SECS`).
    NotYetValid { iat: u64, now: u64 },
    /// `iat` is too old (± `SIGNATURE_CLOCK_SKEW_SECS`).
    Expired { iat: u64, now: u64 },
    /// The `(session_id, nonce)` pair was already seen inside the replay TTL.
    Replay { session_id: String, nonce: u64 },
}

impl std::fmt::Display for SignedRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignedRequestError::MissingSignature => {
                write!(f, "missing signature or client public key")
            }
            SignedRequestError::UnsupportedVersion(v) => write!(
                f,
                "unsupported signature_version {v}: only 1 (canonical) is accepted; the legacy \
                 preimage was retired by TASK-26"
            ),
            SignedRequestError::Malformed(e) => write!(f, "malformed signed request: {e}"),
            SignedRequestError::InvalidSignature => write!(f, "invalid signature"),
            SignedRequestError::NotYetValid { iat, now } => write!(
                f,
                "signature iat {iat} is ahead of the local clock ({now}) beyond the allowed skew"
            ),
            SignedRequestError::Expired { iat, now } => write!(
                f,
                "signature iat {iat} is older than the allowed skew of the local clock ({now})"
            ),
            SignedRequestError::Replay { session_id, nonce } => {
                write!(
                    f,
                    "replay detected: (session '{session_id}', nonce {nonce})"
                )
            }
        }
    }
}

impl std::error::Error for SignedRequestError {}

/// Verifies an inference request against the contract of H-09 / D-2=(b):
///
/// 1. it carries both authentication fields,
/// 2. its `signature_version` is **`1`**: the retired legacy `0` and any unknown value are rejected
///    here, before any cryptographic work, rather than guessed,
/// 3. the signature verifies over the canonical preimage v1,
/// 4. its `iat` is inside `± SIGNATURE_CLOCK_SKEW_SECS` of `now`, and
/// 5. its `(session_id, nonce)` had not been seen inside the replay TTL.
///
/// The one-release migration window `D-2` announced (`0`/absent accepted with a `warn!`) was
/// **closed** by `TASK-26`/S32: that preimage covered a single field and carried no nonce, so it
/// could not detect replays at all. `now` is a parameter so the whole policy is deterministic in
/// tests; `context` names the endpoint in the logs (`POST /forward`, `forward_hidden`, …).
///
/// # Errors
///
/// [`SignedRequestError`], with one variant per rejection reason.
pub fn verify_signed_request<T: tessella_protocol::signing::SignedRequest>(
    req: &T,
    replay: &tessella_protocol::signing::ReplayCache,
    now: u64,
    context: &str,
) -> Result<(), SignedRequestError> {
    use tessella_protocol::signing::{SIGNATURE_CLOCK_SKEW_SECS, SIGNATURE_VERSION_V1};

    if req.client_pubkey().is_empty() || req.signature_bytes().is_empty() {
        return Err(SignedRequestError::MissingSignature);
    }
    let version = req.signature_version();
    if version != SIGNATURE_VERSION_V1 {
        // `0` is no longer «the old path»: it is a version this node has no preimage for, so the
        // decision is taken here, before hashing anything a peer could send (the retired branch used
        // to compute `signed_bytes()` first and then accept the request). `debug!` rather than
        // `warn!`: this runs on a public wire and every rejection is already reported by the caller.
        tracing::debug!(
            signature_version = version,
            endpoint = %context,
            "🔏 [TESERROR-09] Rejected non-v1 signed request: unsupported or legacy signature format"
        );
        return Err(SignedRequestError::UnsupportedVersion(version));
    }
    let preimage = req
        .signed_bytes()
        .map_err(|e| SignedRequestError::Malformed(e.to_string()))?;
    match verify_signature(req.client_pubkey(), &preimage, req.signature_bytes()) {
        Ok(true) => {}
        Ok(false) => return Err(SignedRequestError::InvalidSignature),
        Err(e) => return Err(SignedRequestError::Malformed(e.to_string())),
    }

    let iat = req.iat();
    if iat > now.saturating_add(SIGNATURE_CLOCK_SKEW_SECS) {
        return Err(SignedRequestError::NotYetValid { iat, now });
    }
    if iat.saturating_add(SIGNATURE_CLOCK_SKEW_SECS) < now {
        return Err(SignedRequestError::Expired { iat, now });
    }

    let session_id = req.sign_context().session_id.to_string();
    if !replay.check_and_record(&session_id, req.nonce(), now) {
        return Err(SignedRequestError::Replay {
            session_id,
            nonce: req.nonce(),
        });
    }
    Ok(())
}

pub mod error;
pub use error::NetworkError;

pub mod peer_network;
pub use peer_network::*;

pub use libp2p::gossipsub::{IdentTopic as GossipTopic, Message as GossipsubMessage};
pub use libp2p::swarm::SwarmEvent;
pub mod events {
    pub use libp2p::gossipsub::Event as GossipsubEvent;
    pub use libp2p::identify::Event as IdentifyEvent;
    pub use libp2p::kad::{Event as KademliaEvent, GetProvidersOk, QueryResult};
}
/// Real Kademlia discovery (H-06 · `TASK-15`): the DHT configuration, the record key of a model,
/// the bootstrap plan and the admission predicate the DHT shares with the connection handler.
pub mod kademlia;
pub mod multiaddr {
    pub use libp2p::multiaddr::Protocol;
}

/// The peer membership topic both roles speak (`TASK-16` · S22).
///
/// It carries `WorkerGossipMsg` in JSON: `WorkerJoined` is published by a worker when it starts and
/// `WorkerLeft` by the orchestrator when it evicts a peer. Single definition so the two roles cannot
/// disagree about the string (and so a grep for the topic finds every use).
pub const WORKERS_TOPIC: &str = "tessella/workers/v1";

/// The federation notice topic (`TASK-54` · S60 · D-14).
///
/// It carries a `FederationNotice` in JSON, published **only by the primary orchestrator** when the
/// state a replica copies (the ledger, the reputation or the weight manifests) changed, and read by
/// the replica, which answers it with an **immediate pull**.
///
/// One definition for the two roles, like [`WORKERS_TOPIC`], so the string cannot drift — and so a
/// `grep` finds every use. What travels on it is a **hint, never data**: gossip is unordered and
/// stores nothing, so the truth keeps being the signed `GET /ledger/snapshot` page with its cursor
/// (`TASK-34`/S40). A notice that is lost only means «the pull interval applies», which is why
/// `federation_sync_interval_secs` survives as the safety net.
pub const FEDERATION_TOPIC: &str = "tessella/federation/v1";

/// P2P Behaviour for Worker/Client nodes.
///
/// `TASK-25`/S31: the dead `/moe/inference/1.0.0` request-response transport was removed. Hidden
/// states travel over HTTP (`POST /v1/forward`).
/// `TASK-01`: the dead `/moe/weight-sync/1.0.0` request-response transport (`weight_sync`) and its
/// `CustomJsonCodec` were also purged, as weights are fetched via `hf_downloader`.
/// The P2P surface is pinned by `test_the_worker_p2p_surface_is_pinned`.
#[derive(NetworkBehaviour)]
pub struct MoeBehaviour {
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
    pub mdns: libp2p::swarm::behaviour::toggle::Toggle<mdns::tokio::Behaviour>,
    pub gossipsub: gossipsub::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub dcutr: dcutr::Behaviour,
}

/// Builds the gossipsub configuration used by **both** swarm roles (worker/client and
/// orchestrator).
///
/// There is deliberately a single builder: H-03 came from the worker applying a
/// content-only `message_id_fn` (a hash of `message.data`) while the orchestrator kept
/// the library defaults, i.e. two different deduplication semantics inside the same
/// swarm. A `MessageId` is a **local** deduplication key — it never travels on the wire
/// (`behaviour.rs`: `publish()`/`handle_received_message()` compute it per node), so the
/// divergence did not break interoperability between versions; what it did was let a
/// peer publish an identical payload first and make the legitimate announcement of
/// another sender be dropped as a duplicate (gossip suppression).
///
/// The policy is therefore the libp2p-gossipsub default: `source + sequence number`
/// (`Config::message_id`). Two senders never collide, a duplicate of the same
/// announcement from the same sender is still filtered, and any future change has to be
/// made here — in one place — for both roles.
///
/// `ValidationMode::Strict` is spelled out even though it is already gossipsub 0.46's
/// default, because the audit also suspected a divergence there. Both facts are pinned
/// by `test_gossipsub_config_uses_the_library_default_message_id`.
pub fn gossipsub_config() -> Result<gossipsub::Config> {
    gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .build()
        .map_err(|e| anyhow::anyhow!("Gossipsub config error: {}", e))
}

/// P2P Behaviour for Musivum nodes.
#[derive(NetworkBehaviour)]
pub struct MusivumBehaviour {
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
    pub gossipsub: gossipsub::Behaviour,
    pub relay: relay::Behaviour,
}

/// Creates a P2P Swarm for Worker/Client nodes.
///
/// Both roles build their gossipsub behaviour from [`gossipsub_config`], the single builder of the
/// stack (H-03 · S5/`TASK-11`), so the deduplication and validation policy cannot diverge between a
/// worker and the orchestrator. The Kademlia behaviour comes from
/// [`kademlia::kademlia_behaviour`] (H-06 · `TASK-15`); `kademlia_enabled` selects whether this node
/// joins the DHT as a server at all, which is the operator's escape hatch on a memory-tight Edge
/// node (the cost of the DHT has not been measured on real hardware yet).
pub fn create_worker_swarm(
    local_key: libp2p::identity::Keypair,
    mdns_enabled: bool,
    kademlia_enabled: bool,
) -> Result<Swarm<MoeBehaviour>> {
    let local_peer_id = PeerId::from(local_key.public());

    let swarm = SwarmBuilder::with_existing_identity(local_key.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_relay_client(noise::Config::new, yamux::Config::default)?
        .with_behaviour(|key, relay_client| {
            let kad = kademlia::kademlia_behaviour(local_peer_id, kademlia_enabled);

            let identify =
                identify::Behaviour::new(identify::Config::new("/moe/1.0.0".into(), key.public()));

            let mdns = if mdns_enabled {
                Some(mdns::tokio::Behaviour::new(
                    mdns::Config::default(),
                    local_peer_id,
                )?)
            } else {
                None
            };

            // Gossipsub setup: one shared policy for both roles (see `gossipsub_config`).
            let gossipsub = gossipsub::Behaviour::new(
                MessageAuthenticity::Signed(local_key),
                gossipsub_config()?,
            )
            .map_err(|e| anyhow::anyhow!("Gossipsub error: {}", e))?;

            let dcutr = dcutr::Behaviour::new(local_peer_id);

            Ok(MoeBehaviour {
                kademlia: kad,
                identify,
                mdns: mdns.into(),
                gossipsub,
                relay_client,
                dcutr,
            })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(600)))
        .build();

    Ok(swarm)
}

/// Creates a P2P Swarm for Musivum nodes (Relay Server).
///
/// Shares the gossipsub policy with the worker role through [`gossipsub_config`] (H-03 · S5) and
/// behaves as a relay/kademlia bootstrap for the workers.
///
/// The Kademlia behaviour is the same one the worker builds (H-06 · `TASK-15`), always in
/// `Mode::Server`: the orchestrator is the natural DHT seed of the mesh (it is the node every
/// worker knows how to reach) and it publishes a provider record for every model of its catalogue.
pub fn create_federation_swarm(
    local_key: libp2p::identity::Keypair,
) -> Result<Swarm<MusivumBehaviour>> {
    let local_peer_id = PeerId::from(local_key.public());

    let swarm = SwarmBuilder::with_existing_identity(local_key.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let kad = kademlia::kademlia_behaviour(local_peer_id, true);

            let identify =
                identify::Behaviour::new(identify::Config::new("/moe/1.0.0".into(), key.public()));

            // Gossipsub setup: the very same policy as the worker/client role.
            let gossipsub = gossipsub::Behaviour::new(
                MessageAuthenticity::Signed(local_key),
                gossipsub_config()?,
            )
            .map_err(|e| anyhow::anyhow!("Gossipsub error: {}", e))?;

            let relay = relay::Behaviour::new(local_peer_id, relay::Config::default());

            Ok(MusivumBehaviour {
                kademlia: kad,
                identify,
                gossipsub,
                relay,
            })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(600)))
        .build();

    Ok(swarm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessella_core::MoEForwardRequest;
    use tessella_core::models::DeregisterRequest;
    use tessella_protocol::PeerSignedRequest;
    use tessella_protocol::signing::SignedRequest;

    // `test_cryptographic_request_signing_and_verification` lived here and documented the reach
    // of H-09: it only tampered with `session_id`, because that was all the signature covered.
    // It is replaced by the v1 block at the end of this module (tokens, callback destination,
    // karma tier, nonce, replay, versions and the time window), which is a superset of it.

    /// Builds a synthetic gossipsub message. `gossipsub::Message` has public fields, so
    /// the deduplication policy can be exercised without a `Swarm` (building a swarm is
    /// only safe as long as nobody calls `listen_on`, and §1 forbids binding ports).
    fn gossip_message(source: PeerId, sequence_number: u64, data: &[u8]) -> gossipsub::Message {
        gossipsub::Message {
            source: Some(source),
            data: data.to_vec(),
            sequence_number: Some(sequence_number),
            topic: GossipTopic::new(WORKERS_TOPIC).hash(),
        }
    }

    fn gossip_peer_id() -> PeerId {
        PeerId::from(generate_keypair().public())
    }

    #[test]
    fn test_gossipsub_config_uses_the_library_default_message_id() {
        let config = gossipsub_config().expect("gossipsub config");
        let default = gossipsub::Config::default();

        // H-03 suspected a divergence here: the orchestrator was believed to be
        // permissive while the worker was Strict. Strict is gossipsub 0.46's default and
        // is now explicit for both roles, so it cannot silently drift.
        assert!(matches!(
            config.validation_mode(),
            gossipsub::ValidationMode::Strict
        ));
        assert!(matches!(
            default.validation_mode(),
            gossipsub::ValidationMode::Strict
        ));
        assert_eq!(config.heartbeat_interval(), Duration::from_secs(1));

        let alice = gossip_peer_id();
        let bob = gossip_peer_id();
        let messages = [
            gossip_message(alice, 1, b"{\"WorkerJoined\":{\"peer_id\":\"alice\"}}"),
            gossip_message(alice, 2, b"{\"WorkerJoined\":{\"peer_id\":\"alice\"}}"),
            gossip_message(bob, 1, b"{\"WorkerJoined\":{\"peer_id\":\"alice\"}}"),
            gossip_message(bob, 7, b"topology-frame"),
        ];

        // No custom `message_id_fn`: the ids are exactly the ones the library computes
        // (source + sequence number). Re-introducing a content hash fails here.
        for message in &messages {
            assert_eq!(config.message_id(message), default.message_id(message));
        }
    }

    #[test]
    fn test_gossipsub_message_ids_are_origin_aware() {
        let config = gossipsub_config().expect("gossipsub config");
        let alice = gossip_peer_id();
        let bob = gossip_peer_id();
        let payload = b"{\"WorkerJoined\":{\"peer_id\":\"same-payload\"}}";

        // The removed implementation (H-03) hashed only `message.data`, which is kept
        // here as a regression demonstration: whoever published first could make the
        // legitimate announcement of the other sender be dropped as a duplicate.
        let legacy = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(1))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .message_id_fn(|message: &gossipsub::Message| {
                use std::hash::Hasher;
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                hasher.write(&message.data);
                gossipsub::MessageId::from(hasher.finish().to_string())
            })
            .build()
            .expect("legacy gossipsub config");
        assert_eq!(
            legacy.message_id(&gossip_message(alice, 1, payload)),
            legacy.message_id(&gossip_message(bob, 1, payload)),
            "the removed content-hash implementation collided across senders"
        );

        // Current policy: the id carries the sender, so two senders never collide...
        assert_ne!(
            config.message_id(&gossip_message(alice, 1, payload)),
            config.message_id(&gossip_message(bob, 1, payload))
        );
        // ...the same sender re-sending the same sequence number is deduplicated...
        assert_eq!(
            config.message_id(&gossip_message(alice, 1, b"first")),
            config.message_id(&gossip_message(alice, 1, b"second"))
        );
        // ...and a new sequence number is always a new message (no over-deduplication).
        assert_ne!(
            config.message_id(&gossip_message(alice, 1, payload)),
            config.message_id(&gossip_message(alice, 2, payload))
        );
    }

    /// H-13/TASK-04: the orchestrator has to derive the owning `peer_id` from the public
    /// key a requester quotes (`KarmaTicket` binding), i.e. from exactly the bytes a
    /// worker publishes with `Keypair::public().encode_protobuf()`.
    #[test]
    fn test_peer_id_from_public_key_bytes_matches_the_signing_keypair() {
        let keypair = generate_keypair();
        let protobuf_pubkey = keypair.public().encode_protobuf();

        assert_eq!(
            peer_id_from_public_key_bytes(&protobuf_pubkey).unwrap(),
            peer_id_from_keypair(&keypair)
        );
        // The protobuf encoding is 4 header bytes + the 32-byte ed25519 key (36 bytes ⇒
        // 72 hex chars), *not* the 32 raw bytes a "64 hex chars" length check assumes.
        assert_eq!(protobuf_pubkey.len(), 36);

        // Another node's key must never resolve to this peer id (that would hand out the
        // other peer's tier: the whole point of the binding).
        let other = generate_keypair().public().encode_protobuf();
        assert_ne!(
            peer_id_from_public_key_bytes(&other).unwrap(),
            peer_id_from_keypair(&keypair)
        );
    }

    #[test]
    fn test_peer_id_from_public_key_bytes_rejects_non_libp2p_keys() {
        assert!(peer_id_from_public_key_bytes(&[]).is_err());
        // A raw 32-byte ed25519 key (the old "64 hex chars" contract) is not a protobuf
        // public key and must be rejected instead of silently accepted.
        assert!(peer_id_from_public_key_bytes(&[0u8; 32]).is_err());
        assert!(peer_id_from_public_key_bytes(&[0xff; 36]).is_err());
    }

    fn signed_deregister(keypair: &NetworkKeypair, peer_id: &str) -> DeregisterRequest {
        let mut req = DeregisterRequest {
            peer_id: peer_id.to_string(),
            reason: "shutdown".to_string(),
            public_key: Vec::new(),
            signature: Vec::new(),
        };
        sign_peer_request(keypair, &mut req).expect("signing a fresh request must succeed");
        req
    }

    /// D-3/TASK-03 Step B: the worker signs the destructive requests with the same key that
    /// owns its `peer_id`, so the orchestrator can bind the request to an identity instead of
    /// trusting the `peer_id` string in the body. Everything here is offline (no `Swarm`).
    #[test]
    fn test_sign_peer_request_is_verifiable_by_the_owning_peer_id() {
        let keypair = generate_keypair();
        let peer_id = peer_id_from_keypair(&keypair);
        let req = signed_deregister(&keypair, &peer_id);

        assert!(req.has_peer_auth());
        assert_eq!(req.public_key.len(), 36, "protobuf-encoded ed25519 key");

        // The public key must resolve to the peer id the request claims.
        assert_eq!(
            peer_id_from_public_key_bytes(&req.public_key).unwrap(),
            peer_id
        );
        // …and the signature must cover the request *without* the authentication fields.
        let mut stripped = req.clone();
        stripped.set_peer_auth(Vec::new(), Vec::new());
        assert!(
            verify_signature(
                &req.public_key,
                &stripped.peer_canonical_bytes().unwrap(),
                &req.signature
            )
            .unwrap()
        );
    }

    /// Re-signing the same key twice must be verifiable both times (ed25519 is deterministic
    /// here, but the point is that the canonical bytes do not depend on the signature itself).
    #[test]
    fn test_signed_request_can_be_verified_after_a_round_trip_through_json() {
        let keypair = generate_keypair();
        let peer_id = peer_id_from_keypair(&keypair);
        let req = signed_deregister(&keypair, &peer_id);

        let wire = serde_json::to_string(&req).unwrap();
        let parsed: DeregisterRequest = serde_json::from_str(&wire).unwrap();

        assert_eq!(parsed.public_key, req.public_key);
        assert_eq!(parsed.signature, req.signature);
        let mut stripped = parsed.clone();
        stripped.set_peer_auth(Vec::new(), Vec::new());
        assert!(
            verify_signature(
                &parsed.public_key,
                &stripped.peer_canonical_bytes().unwrap(),
                &parsed.signature
            )
            .unwrap(),
            "the signature must survive a JSON round trip"
        );
    }

    /// Any field modified after signing must invalidate the signature, otherwise an attacker
    /// could rewrite the target of a signed report.
    #[test]
    fn test_signature_does_not_cover_a_modified_request() {
        let keypair = generate_keypair();
        let peer_id = peer_id_from_keypair(&keypair);
        let mut req = signed_deregister(&keypair, &peer_id);

        req.peer_id = peer_id_from_keypair(&generate_keypair());
        let mut stripped = req.clone();
        stripped.set_peer_auth(Vec::new(), Vec::new());
        assert!(
            !verify_signature(
                &req.public_key,
                &stripped.peer_canonical_bytes().unwrap(),
                &req.signature
            )
            .unwrap(),
            "tampering with the body must break the signature"
        );
    }

    // ---------------------------------------------------------------------------------
    // H-09 / TASK-05 — canonical signed preimage v1 (integrity + anti-replay)
    // ---------------------------------------------------------------------------------

    use tessella_protocol::QueuePriority;
    use tessella_protocol::models::KarmaTicket;
    use tessella_protocol::signing::{
        ReplayCache, SIGNATURE_CLOCK_SKEW_SECS, SIGNATURE_VERSION_LEGACY, SIGNATURE_VERSION_V1,
    };

    const NOW: u64 = 1_700_000_000;

    /// A realistic `MoEForwardRequest` sealed as v1: exactly what an entry node emits and every
    /// hop verifies. Building it needs no socket and no GPU.
    fn sealed_v1_request(keypair: &NetworkKeypair) -> MoEForwardRequest {
        let mut req = MoEForwardRequest {
            input_tensor: None,
            client_peer_id: Some("client-1".to_string()),
            session_id: Some("session-12345".to_string()),
            tokens: vec![1, 2, 3],
            speculative_count: 0,
            entry_node_url: Some("http://203.0.113.9:9001".to_string()),
            signature: Vec::new(),
            client_pubkey: Vec::new(),
            karma_ticket: Some(KarmaTicket {
                client_pubkey: Vec::new(),
                tier: QueuePriority::Vip,
                expiry: NOW + 600,
                orchestrator_signature: Vec::new(),
            }),
            signature_version: SIGNATURE_VERSION_LEGACY,
            nonce: 0,
            iat: 0,
        };
        seal_signed_request(keypair, &mut req, 7, NOW).expect("sealing a fresh request");
        req
    }

    #[test]
    fn test_sealed_v1_request_verifies_even_after_a_json_round_trip() {
        let keypair = generate_keypair();
        let req = sealed_v1_request(&keypair);

        assert_eq!(req.signature_version, SIGNATURE_VERSION_V1);
        assert_eq!(req.nonce, 7);
        assert_eq!(req.iat, NOW);
        assert_eq!(req.client_pubkey.len(), 36, "protobuf-encoded ed25519 key");

        let wire = serde_json::to_string(&req).unwrap();
        let parsed: MoEForwardRequest = serde_json::from_str(&wire).unwrap();
        let cache = ReplayCache::new();
        assert_eq!(
            verify_signed_request(&parsed, &cache, NOW, "POST /forward"),
            Ok(())
        );
    }

    /// The end of H-09: mutating any signed field must invalidate the signature. Before this
    /// step only `session_id` was covered, so `tokens`, the callback destination and the queue
    /// tier could be rewritten by any hop.
    #[test]
    fn test_v1_signature_covers_tokens_entry_node_url_and_karma_ticket() {
        let keypair = generate_keypair();
        let req = sealed_v1_request(&keypair);

        let mut tampered: Vec<(&str, MoEForwardRequest)> = Vec::new();
        let mut tokens = req.clone();
        tokens.tokens = vec![9, 9, 9];
        tampered.push(("tokens", tokens));
        let mut callback = req.clone();
        callback.entry_node_url = Some("http://attacker.example/v1/token_stream/x".to_string());
        tampered.push(("entry_node_url", callback));
        let mut ticket = req.clone();
        ticket.karma_ticket.as_mut().unwrap().tier = QueuePriority::Leech;
        tampered.push(("karma_tier", ticket));
        let mut session = req.clone();
        session.session_id = Some("session-54321".to_string());
        tampered.push(("session_id", session));
        let mut nonce = req.clone();
        nonce.nonce = 8;
        tampered.push(("nonce", nonce));

        for (field, mutated) in tampered {
            let cache = ReplayCache::new();
            assert_eq!(
                verify_signed_request(&mutated, &cache, NOW, "POST /forward"),
                Err(SignedRequestError::InvalidSignature),
                "tampering with {field} must break the signature"
            );
            assert!(cache.is_empty(), "a rejected request is not recorded");
        }
    }

    /// Criterion 2 of `TASK-05`: the same signed request cannot be used twice.
    #[test]
    fn test_replayed_v1_request_is_rejected_by_the_nonce_cache() {
        let keypair = generate_keypair();
        let req = sealed_v1_request(&keypair);
        let cache = ReplayCache::new();

        assert_eq!(
            verify_signed_request(&req, &cache, NOW, "POST /v1/forward"),
            Ok(())
        );
        let replay = verify_signed_request(&req, &cache, NOW + 1, "POST /v1/forward");
        assert!(
            matches!(replay, Err(SignedRequestError::Replay { nonce, .. }) if nonce == 7),
            "{replay:?}"
        );
    }

    /// Criterion 4: an unknown version must not fall back to any other path.
    #[test]
    fn test_unknown_signature_version_is_rejected() {
        let keypair = generate_keypair();
        let mut req = sealed_v1_request(&keypair);
        req.signature_version = 9;

        let cache = ReplayCache::new();
        assert_eq!(
            verify_signed_request(&req, &cache, NOW, "POST /forward"),
            Err(SignedRequestError::UnsupportedVersion(9))
        );
    }

    #[test]
    fn test_v1_request_outside_the_clock_window_is_rejected() {
        let keypair = generate_keypair();

        let mut future = sealed_v1_request(&keypair);
        seal_signed_request(
            &keypair,
            &mut future,
            1,
            NOW + SIGNATURE_CLOCK_SKEW_SECS + 1,
        )
        .unwrap();
        let cache = ReplayCache::new();
        assert!(matches!(
            verify_signed_request(&future, &cache, NOW, "POST /forward"),
            Err(SignedRequestError::NotYetValid { .. })
        ));

        let mut stale = sealed_v1_request(&keypair);
        seal_signed_request(&keypair, &mut stale, 2, NOW - SIGNATURE_CLOCK_SKEW_SECS - 1).unwrap();
        let cache = ReplayCache::new();
        assert!(matches!(
            verify_signed_request(&stale, &cache, NOW, "POST /forward"),
            Err(SignedRequestError::Expired { .. })
        ));
    }

    #[test]
    fn test_unsigned_request_is_rejected() {
        let mut req = sealed_v1_request(&generate_keypair());
        req.signature.clear();
        req.client_pubkey.clear();

        let cache = ReplayCache::new();
        assert_eq!(
            verify_signed_request(&req, &cache, NOW, "POST /forward"),
            Err(SignedRequestError::MissingSignature)
        );
    }

    /// Criterion 2 of `TASK-26`/S32 (D-2: «require v1 in ≥2 releases»): the migration window is
    /// **closed**. A request that declares the retired version is rejected even when a legitimate
    /// key signed it over the preimage that version used to have, and the reason is the version
    /// itself (`UnsupportedVersion`), which is checked **before** any cryptographic work.
    #[test]
    fn test_a_correctly_signed_v0_request_is_rejected() {
        let keypair = generate_keypair();
        let mut req = sealed_v1_request(&keypair);
        // Byte-for-byte what the previous releases signed: `bincode(session_id)` (`u64` little-endian
        // length prefix + bytes, pinned by
        // `tessella_inference::hidden_state::tests::test_the_retired_v0_preimage_is_not_usable_any_more`).
        // `tessella-network` does not depend on `bincode`, so the bytes are built here.
        let retired_preimage = [13u64.to_le_bytes().as_slice(), b"session-12345"].concat();
        assert_eq!(retired_preimage.len(), 8 + 13);
        req.set_signature_fields(
            SIGNATURE_VERSION_LEGACY,
            0,
            0,
            keypair.sign(&retired_preimage).unwrap(),
            keypair.public().encode_protobuf(),
        );

        let cache = ReplayCache::new();
        assert_eq!(
            verify_signed_request(&req, &cache, NOW, "POST /v1/forward"),
            Err(SignedRequestError::UnsupportedVersion(
                SIGNATURE_VERSION_LEGACY
            )),
            "a valid legacy signature is not a way in any more"
        );
        assert!(cache.is_empty(), "a rejected request is never recorded");
    }
    /// TASK-19: la pubkey que `/p2p_id` publica es la clave **cruda** de 32 bytes (64 hex), que es
    /// exactamente el formato que el worker acepta en `orchestrator_public_key`. Publicar el
    /// protobuf (72 hex) habría obligado al operador a convertirlo y habría roto el pinning de
    /// cualquier worker ya desplegado.
    #[test]
    fn test_ed25519_public_key_hex_is_the_raw_key_the_worker_pins() {
        let keypair = generate_keypair();

        let bytes =
            ed25519_public_key_bytes(&keypair).expect("an ed25519 keypair has 32 raw key bytes");
        let hex_key = ed25519_public_key_hex(&keypair).expect("hex of the raw key");

        assert_eq!(bytes.len(), 32, "RFC 8032 public keys are 32 raw bytes");
        assert_eq!(
            hex_key.len(),
            64,
            "raw key => 64 hex chars, not the 72 of the protobuf encoding"
        );
        assert_eq!(hex_key, hex::encode(bytes));
        assert_eq!(
            hex_key,
            hex::encode(&keypair.public().encode_protobuf()[4..]),
            "the tail of the protobuf encoding is the same key"
        );

        // Las dos representaciones describen al mismo peer: el `peer_id` que publica `/p2p_id` y
        // la pubkey que publica no pueden divergir.
        assert_eq!(
            peer_id_from_public_key_bytes(&keypair.public().encode_protobuf()).unwrap(),
            peer_id_from_keypair(&keypair)
        );
    }

    /// Criterio 1 de TASK-19: la identidad sobrevive a un reinicio porque lo que se persiste es la
    /// keypair; al recargarla da la misma pubkey y el mismo `peer_id`.
    #[test]
    fn test_the_published_key_survives_a_keypair_round_trip() {
        let keypair = generate_keypair();
        let stored = keypair_to_bytes(&keypair).expect("protobuf encoding");
        let reloaded = keypair_from_bytes(&stored).expect("reload");

        assert_eq!(
            ed25519_public_key_hex(&keypair).unwrap(),
            ed25519_public_key_hex(&reloaded).unwrap()
        );
        assert_eq!(
            peer_id_from_keypair(&keypair),
            peer_id_from_keypair(&reloaded)
        );
    }

    /// `TASK-25`/S31 · The P2P surface of a worker is pinned **by the compiler**: the `match` below
    /// is exhaustive on purpose (no `_` arm), so adding a behaviour — e.g. bringing back the dead
    /// `/moe/inference/1.0.0` hidden-state transport that S31 retired because nobody produced it —
    /// `test_the_worker_p2p_surface_is_pinned` does an exhaustive `match` (no `_`) on `MoeBehaviourEvent`.
    /// Adding or removing any P2P behaviour fails the build here, ensuring any changes are explicit.
    ///
    /// Hidden states travel over **HTTP** (`POST /v1/forward`); weights are downloaded via HF downloader.
    #[test]
    fn test_the_worker_p2p_surface_is_pinned() {
        fn surface_of(event: &MoeBehaviourEvent) -> &'static str {
            match event {
                MoeBehaviourEvent::Kademlia(_) => "kademlia",
                MoeBehaviourEvent::Identify(_) => "identify",
                MoeBehaviourEvent::Mdns(_) => "mdns",
                MoeBehaviourEvent::Gossipsub(_) => "gossipsub",
                MoeBehaviourEvent::RelayClient(_) => "relay-client",
                MoeBehaviourEvent::Dcutr(_) => "dcutr",
            }
        }

        // `kad::Event::ModeChanged` is the one `MoeBehaviourEvent` a test can build without a live
        // swarm (building one binds nothing, but it needs a tokio runtime and a keypair); building
        // it is what turns the exhaustive `match` above into a compiled assertion.
        let event = MoeBehaviourEvent::Kademlia(kad::Event::ModeChanged {
            new_mode: kad::Mode::Client,
        });
        assert_eq!(surface_of(&event), "kademlia");
    }
}
