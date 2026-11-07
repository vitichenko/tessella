# tessella-network

> Peer-to-peer control plane, discovery, and gossip protocol layer built on libp2p 0.53.

## Responsibility

`tessella-network` manages authenticated peer connectivity and distributed network state:
- **Authenticated Noise Handshake:** Enforces mutual cryptographic identity verification using libp2p `Noise` with Ed25519 `PeerId`s (not TLS/mTLS).
- **Kademlia Distributed Hash Table (DHT):** Discovers model providers via a dedicated DHT protocol (`/tessella/kad/1.0.0`, `Mode::Server`, 4-hour record TTLs, provider re-publication every 15 minutes) under keys formatted as `tessella/model/<model_id>`.
- **GossipSub Mesh Broadcast:** Manages pub-sub topics for cluster membership announcements (`WorkerGossipMsg`) and federation notifications.
- **Local Discovery:** Employs mDNS for automatic zero-configuration peer discovery on local area networks.
- **Transport Separation Invariant (`D-10` & `TASK-01`):** libp2p handles discovery, topology, relay, and presence. Activation tensors travel exclusively over HTTP/1.1 (`POST /v1/forward`) in `tessella-api`, while model weights are downloaded via `hf_downloader`. The legacy P2P request-response transport (`/moe/weight-sync/1.0.0` and `CustomJsonCodec`) was purged in `TASK-01`. *(The legacy `CircuitBreaker` was retired in `TASK-50` and replaced by `PeerFailureRegistry` in `tessella-core`).*

## Modules

| File | Subsystem | Responsibility |
| :--- | :--- | :--- |
| `src/lib.rs` | Swarm & Network Behaviours | Assembles `MoeBehaviour` (GossipSub, Kademlia, mDNS, Identify, RelayClient, Dcutr). Configures TCP transport, Noise authentication, and Yamux multiplexing. |
| `src/kademlia.rs` | DHT Protocol & Discovery | Implements `/tessella/kad/1.0.0`, provider query state machines, bootstrap planning, and admission filtering. |

## Usage

```rust
use tessella_network::{
    create_worker_swarm, generate_keypair, NetworkKeypair,
    kademlia::{kademlia_config, BootstrapPlan},
};
```
