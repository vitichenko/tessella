use std::sync::Arc;
use tessella_core::error::DomainError;
use tessella_core::traits::PeerNetwork;
use tokio::sync::{mpsc, oneshot};

use crate::{NetworkKeypair, verify_signature};

/// Commands sent to the swarm event loop (`SOLID-90`).
#[derive(Debug)]
pub enum SwarmCommand {
    Publish {
        topic: String,
        data: Vec<u8>,
        respond_to: Option<oneshot::Sender<Result<(), DomainError>>>,
    },
    Subscribe {
        topic: String,
        respond_to: Option<oneshot::Sender<Result<(), DomainError>>>,
    },
    Dial {
        addr: String,
        respond_to: Option<oneshot::Sender<Result<(), DomainError>>>,
    },
}

/// Concrete libp2p implementation of the [`PeerNetwork`] port (`SOLID-90`).
///
/// Encapsulates cryptographic operations (signing, public key export, verification) and
/// dispatches network actions (publish, subscribe, dial) to the swarm event loop via a channel.
#[derive(Clone)]
pub struct DefaultPeerNetwork {
    peer_id: String,
    public_key_bytes: Vec<u8>,
    keypair: Arc<NetworkKeypair>,
    cmd_tx: mpsc::Sender<SwarmCommand>,
}

impl DefaultPeerNetwork {
    pub fn new(keypair: NetworkKeypair, cmd_tx: mpsc::Sender<SwarmCommand>) -> Self {
        let peer_id = libp2p::PeerId::from(keypair.public()).to_string();
        let public_key_bytes = keypair.public().encode_protobuf();
        Self {
            peer_id,
            public_key_bytes,
            keypair: Arc::new(keypair),
            cmd_tx,
        }
    }

    /// Access the underlying keypair if needed for low-level swarm creation.
    pub fn keypair(&self) -> &NetworkKeypair {
        &self.keypair
    }

    /// Access the underlying keypair Arc.
    pub fn keypair_arc(&self) -> Arc<NetworkKeypair> {
        self.keypair.clone()
    }
}

use crate::error::NetworkError;

#[async_trait::async_trait]
impl PeerNetwork for DefaultPeerNetwork {
    fn peer_id(&self) -> &str {
        &self.peer_id
    }

    fn public_key(&self) -> &[u8] {
        &self.public_key_bytes
    }

    fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, DomainError> {
        self.keypair
            .sign(msg)
            .map_err(|e| DomainError::from(NetworkError::Crypto(format!("signature error: {e}"))))
    }

    fn verify(&self, pubkey: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, DomainError> {
        verify_signature(pubkey, msg, sig).map_err(|e| {
            DomainError::from(NetworkError::Crypto(format!("verification error: {e}")))
        })
    }

    async fn publish(&self, topic: &str, data: Vec<u8>) -> Result<(), DomainError> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(SwarmCommand::Publish {
                topic: topic.to_string(),
                data,
                respond_to: Some(tx),
            })
            .await
            .map_err(|e| {
                DomainError::from(NetworkError::Swarm(format!(
                    "swarm command channel send error: {e}"
                )))
            })?;
        rx.await.map_err(|e| {
            DomainError::from(NetworkError::Swarm(format!(
                "swarm command response error: {e}"
            )))
        })?
    }

    async fn subscribe(&self, topic: &str) -> Result<(), DomainError> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(SwarmCommand::Subscribe {
                topic: topic.to_string(),
                respond_to: Some(tx),
            })
            .await
            .map_err(|e| {
                DomainError::from(NetworkError::Swarm(format!(
                    "swarm command channel send error: {e}"
                )))
            })?;
        rx.await.map_err(|e| {
            DomainError::from(NetworkError::Swarm(format!(
                "swarm command response error: {e}"
            )))
        })?
    }

    async fn dial(&self, addr: &str) -> Result<(), DomainError> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(SwarmCommand::Dial {
                addr: addr.to_string(),
                respond_to: Some(tx),
            })
            .await
            .map_err(|e| {
                DomainError::from(NetworkError::Dial(format!(
                    "swarm command channel send error: {e}"
                )))
            })?;
        rx.await.map_err(|e| {
            DomainError::from(NetworkError::Dial(format!(
                "swarm command response error: {e}"
            )))
        })?
    }
}

/// Helper to build a [`DefaultPeerNetwork`] together with its command receiver.
pub fn create_peer_network_handle(
    keypair: NetworkKeypair,
    buffer: usize,
) -> (Arc<DefaultPeerNetwork>, mpsc::Receiver<SwarmCommand>) {
    let (tx, rx) = mpsc::channel(buffer);
    let adapter = DefaultPeerNetwork::new(keypair, tx);
    (Arc::new(adapter), rx)
}
