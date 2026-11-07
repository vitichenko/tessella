use crate::error::DomainError;
use crate::types::Role;
use ed25519_dalek::Verifier;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub enum FederationRole {
    Primary,
    Replica { primary_url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct ReputationScore {
    pub peer_id: String,
    pub score: f64,
    pub last_updated: u64,
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, PartialEq,
)]
pub enum IncidentType {
    InvalidSignature,
    ConflictingProof,
    ExcessiveFailures,
    WeightPoisoning,
    /// Work whose evidence does not corroborate (`TASK-27`/S33): a claim that diverges from another
    /// claim of the same work, or a chain that does not link.
    ///
    /// It applies the **configured** `[reputation] slash_penalty`, unlike
    /// [`IncidentType::ConflictingProof`], which is a full reset for proven fraud. With two
    /// claimants nobody can tell the liar from the victim, and a hop that never reported may be a
    /// bystander: a total reset there would hand an attacker a way to wipe an honest peer's
    /// reputation.
    UncorroboratedWork,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub enum ScoreEvent {
    IncidentRecorded(IncidentType),
    WorkProofAccepted,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct WeightManifest {
    pub model_id: String,
    pub filename: String,
    pub role: Role,
    pub blake3_hash: [u8; 32],
    pub size_bytes: u64,
    pub registered_at: u64,
    pub tracker_public_key: Vec<u8>,
    pub tracker_signature: Vec<u8>,
}

impl WeightManifest {
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, DomainError> {
        let mut cloned = self.clone();
        cloned.tracker_signature = vec![];
        serde_json::to_vec(&cloned).map_err(|e| DomainError::Serialization(e.to_string()))
    }

    pub fn verify(&self) -> Result<(), DomainError> {
        let pubkey = ed25519_dalek::VerifyingKey::from_bytes(
            &self.tracker_public_key[4..]
                .try_into()
                .map_err(|_| DomainError::Validation("Invalid musivum public key length".into()))?,
        )
        .map_err(|e| DomainError::Validation(e.to_string()))?;

        let sig = ed25519_dalek::Signature::from_slice(&self.tracker_signature)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        pubkey
            .verify(&self.to_canonical_bytes()?, &sig)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct TransferReceipt {
    pub sender_peer_id: String,
    pub receiver_peer_id: String,
    pub model_id: String,
    pub role: Role,
    pub blake3_hash: [u8; 32],
    pub size_bytes: u64,
    pub timestamp: u64,
}

impl TransferReceipt {
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, DomainError> {
        serde_json::to_vec(self).map_err(|e| DomainError::Serialization(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct SignedTransferReceipt {
    pub receipt: TransferReceipt,
    pub sender_signature: Vec<u8>,
    pub sender_public_key: Vec<u8>,
}

impl SignedTransferReceipt {
    pub fn verify(&self) -> Result<(), DomainError> {
        let pubkey = ed25519_dalek::VerifyingKey::from_bytes(
            &self.sender_public_key[4..]
                .try_into()
                .map_err(|_| DomainError::Validation("Invalid sender public key length".into()))?,
        )
        .map_err(|e| DomainError::Validation(e.to_string()))?;

        let sig = ed25519_dalek::Signature::from_slice(&self.sender_signature)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        pubkey
            .verify(&self.receipt.to_canonical_bytes()?, &sig)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        Ok(())
    }
}

/// Total layers of standard models, used by both worker and orchestrator.
#[must_use]
pub fn get_total_layers(model_id: &str) -> Option<usize> {
    let lower = model_id.to_lowercase();
    if lower.contains("tessella-moe-micro") {
        Some(4)
    } else if lower.contains("0.5b") {
        Some(24)
    } else if lower.contains("1.5b") {
        Some(28)
    } else if lower.contains("3b") {
        Some(36)
    } else if lower.contains("7b") {
        Some(28)
    } else if lower.contains("14b") {
        Some(48)
    } else if lower.contains("72b") {
        Some(80)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TASK-86`/S93: the layer **count** of the protocol and the layer **cost** of this table answer for
    /// the same models, because both match the same id fragments. The nine sizes of `Qwen2.5` are the
    /// ones published on the Hub (`Qwen/Qwen2.5-<size>-Instruct`; the `-Coder-` spelling of a size
    /// declares the very same `num_hidden_layers`), and `tessella-moe-micro` is the fixture of this
    /// repository, whose `config.json` and cached revision both say **4** — this table said 8, so a canary
    /// of that model validated a range no model had.
    #[test]
    fn the_layer_table_matches_the_configs_it_summarises() {
        for (model_id, layers) in [
            ("Qwen/Qwen2.5-0.5B-Instruct", 24),
            ("Qwen/Qwen2.5-1.5B-Instruct", 28),
            ("Qwen/Qwen2.5-3B-Instruct", 36),
            ("Qwen/Qwen2.5-7B-Instruct", 28),
            ("Qwen/Qwen2.5-14B-Instruct", 48),
            ("Qwen/Qwen2.5-72B-Instruct", 80),
            ("Qwen/Qwen2.5-Coder-1.5B-Instruct", 28),
            ("Qwen/Qwen2.5-Coder-14B-Instruct", 48),
            // `tessella/models/tessella-moe-micro/config.json` (`scripts/create_micro_moe.py`).
            ("victorespada/tessella-moe-micro", 4),
            // The id is matched case-insensitively, like the table of layer costs.
            ("QWEN/QWEN2.5-3B-INSTRUCT", 36),
        ] {
            assert_eq!(
                get_total_layers(model_id),
                Some(layers),
                "the layer count of '{model_id}' must be the one of its config.json"
            );
        }

        // An id the registry does not know states nothing: the caller decides (`unwrap_or(..)`), and a
        // canary of it aborts the boot naming the field it is missing (`worker/src/canary.rs`).
        assert_eq!(get_total_layers("test-model"), None);
        assert_eq!(get_total_layers("acme/mystery-model"), None);
    }
}
