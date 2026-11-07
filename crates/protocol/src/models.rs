//! Re-export facade for protocol domain planes (SOLID-91).
//!
//! Types are partitioned by plane into dedicated modules:
//! - [`crate::worker`]: Worker contracts, node information, and topology models.
//! - [`crate::orchestrator`]: Orchestrator contracts, receipts, manifests, and roles.
//! - [`crate::audit`]: Audit execution, reports, work tickets, and proofs.
//! - [`crate::ledger`]: Ledger entries, types, and snapshots.

pub use crate::audit::*;
pub use crate::ledger::*;
pub use crate::orchestrator::*;
pub use crate::worker::*;
