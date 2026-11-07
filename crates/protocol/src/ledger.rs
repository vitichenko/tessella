use crate::error::DomainError;
use ed25519_dalek::Verifier;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    Archive,
    RkyvDeserialize,
    RkyvSerialize,
    PartialEq,
    Eq,
    Hash,
)]
#[archive(check_bytes)]
pub enum LedgerEntryType {
    Earned,
    Spent,
    Slashed,
}

impl LedgerEntryType {
    /// The exact text this variant is stored as in `ledger.entry_type` (`TASK-42`/S48).
    ///
    /// Before S48 the storage said the same thing in three ways: local entries were inserted with SQL
    /// literals (`'Earned'`, `'Spent'`, `'Slashed'`), federated ones with `format!("{:?}", …)`, and
    /// reads went through `serde_json::from_str("\"{raw}\"")`. The three happened to agree, and that is
    /// precisely why they were dangerous: a `#[serde(rename_all)]` (or a variant rename) would have
    /// stopped `'Earned'` from parsing, so every credit would have been read back as `Spent` — money
    /// moving in the wrong direction, in silence. One exhaustive `match` is the contract, and the
    /// compiler breaks the build if a variant is ever added without deciding its text.
    #[must_use]
    pub const fn to_db(&self) -> &'static str {
        match self {
            LedgerEntryType::Earned => "Earned",
            LedgerEntryType::Spent => "Spent",
            LedgerEntryType::Slashed => "Slashed",
        }
    }

    /// Parses what [`Self::to_db`] wrote.
    ///
    /// `None` means «this build does not know that text» and it is deliberately **not** a fallback:
    /// the DTO does not invent a value (the storage decides, logs it and keeps its historical
    /// `Spent`), because a wrong ledger type is not a formatting detail.
    #[must_use]
    pub fn from_db(raw: &str) -> Option<Self> {
        match raw {
            "Earned" => Some(LedgerEntryType::Earned),
            "Spent" => Some(LedgerEntryType::Spent),
            "Slashed" => Some(LedgerEntryType::Slashed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct LedgerEntry {
    pub peer_id: String,
    pub entry_type: LedgerEntryType,
    pub amount: f32,
    pub proof: String,
    pub timestamp: u64,
    pub source: String,
}

impl LedgerEntry {
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, DomainError> {
        serde_json::to_vec(self).map_err(|e| DomainError::Serialization(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct SignedLedgerEntry {
    pub entry: LedgerEntry,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct LedgerSnapshot {
    pub entries: Vec<SignedLedgerEntry>,
    pub musivum_peer_id: String,
    pub tracker_public_key: Vec<u8>,
    pub snapshot_height: u64,
    pub snapshot_hash: [u8; 32],
    pub tracker_signature: Vec<u8>,
    pub timestamp: u64,
}

impl LedgerSnapshot {
    pub fn compute_hash(entries: &[SignedLedgerEntry]) -> Result<[u8; 32], DomainError> {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for entry in entries {
            hasher.update(entry.entry.to_canonical_bytes()?);
            hasher.update(&entry.signature);
        }
        Ok(hasher.finalize().into())
    }

    pub fn verify(&self) -> Result<(), DomainError> {
        let expected_hash = Self::compute_hash(&self.entries)?;
        if self.snapshot_hash != expected_hash {
            return Err(DomainError::Validation("Snapshot hash mismatch".into()));
        }

        let pubkey = ed25519_dalek::VerifyingKey::from_bytes(
            &self.tracker_public_key[4..]
                .try_into()
                .map_err(|_| DomainError::Validation("Invalid musivum public key length".into()))?,
        )
        .map_err(|e| DomainError::Validation(e.to_string()))?;

        let sig = ed25519_dalek::Signature::from_slice(&self.tracker_signature)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        pubkey
            .verify(&self.snapshot_hash, &sig)
            .map_err(|e| DomainError::Validation(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TASK-42`/S48: el texto con el que un tipo de asiento se guarda en SQLite es **una** decisión
    /// y se recorre en los dos sentidos. Antes de S48 el almacenamiento escribía `Earned`/`Spent` con
    /// literales SQL y los asientos federados con `format!("{:?}", …)`, y leía con serde: coincidían
    /// por casualidad, así que un `rename_all` habría convertido cada crédito en `Spent` sin que nada
    /// fallara.
    #[test]
    fn the_ledger_entry_type_text_round_trips_in_one_place() {
        for entry_type in [
            LedgerEntryType::Earned,
            LedgerEntryType::Spent,
            LedgerEntryType::Slashed,
        ] {
            assert_eq!(
                LedgerEntryType::from_db(entry_type.to_db()),
                Some(entry_type.clone()),
                "{entry_type:?} debe sobrevivir al viaje por la base de datos"
            );
            // El texto es el del enum (no un `Debug` reformateado ni un `rename_all`): es lo que ya
            // hay escrito en las bases de los operadores.
            assert_eq!(format!("{entry_type:?}"), entry_type.to_db());
        }

        assert_eq!(
            LedgerEntryType::from_db("Withdrawal"),
            None,
            "un texto desconocido no se convierte en un valor: lo decide quien lee, con log"
        );
    }
}
