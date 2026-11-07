use crate::error::DomainError;

/// Port for peer-to-peer network operations (`SOLID-90`).
///
/// Inverts the `worker -> network` dependency so that the data-plane daemon depends on an
/// abstraction rather than on concrete libp2p types or free functions.
#[async_trait::async_trait]
pub trait PeerNetwork: Send + Sync {
    /// Returns the local PeerId string.
    fn peer_id(&self) -> &str;

    /// Protobuf-encoded public key bytes of this node's identity.
    fn public_key(&self) -> &[u8];

    /// Signs an arbitrary message using the network private key.
    fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, DomainError>;

    /// Verifies a signature against an identity public key.
    fn verify(&self, pubkey: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, DomainError>;

    /// Publishes raw bytes to a gossipsub topic.
    async fn publish(&self, topic: &str, data: Vec<u8>) -> Result<(), DomainError>;

    /// Subscribes to a gossipsub topic.
    async fn subscribe(&self, topic: &str) -> Result<(), DomainError>;

    /// Dials a remote peer multiaddress.
    async fn dial(&self, addr: &str) -> Result<(), DomainError>;

    /// Fills in both authentication fields of a peer-authenticated request (D-3).
    fn sign_peer_request(
        &self,
        req: &mut dyn tessella_protocol::PeerSignedRequest,
    ) -> Result<(), DomainError> {
        let canonical = req.peer_canonical_bytes()?;
        let signature = self.sign(&canonical)?;
        req.set_peer_auth(self.public_key().to_vec(), signature);
        Ok(())
    }

    /// Fills in all signature fields of an inference request with preimage v1 (H-09).
    fn seal_signed_request(
        &self,
        req: &mut dyn tessella_protocol::signing::SignedRequest,
        nonce: u64,
        iat: u64,
    ) -> Result<(), DomainError> {
        req.set_signature_fields(
            tessella_protocol::signing::SIGNATURE_VERSION_V1,
            nonce,
            iat,
            Vec::new(),
            Vec::new(),
        );
        let preimage = req.signed_bytes()?;
        let signature = self.sign(&preimage)?;
        req.set_signature_fields(
            tessella_protocol::signing::SIGNATURE_VERSION_V1,
            nonce,
            iat,
            signature,
            self.public_key().to_vec(),
        );
        Ok(())
    }
}
