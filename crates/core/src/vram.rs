//! VRAM arithmetic shared by the worker's registration paths (H-18 · `TASK-14`).
//!
//! `(vram_budget_gb * GB_TO_MB) as u32` was written out four times inside
//! `worker/src/network_manager.rs`. A divergence there is not cosmetic: `vram_mb` is what the
//! orchestrator uses to decide how many layers a node can hold (`orchestrator.rs`), so two copies
//! that disagree mean the node is assigned work its budget does not support.

/// Converts the configured VRAM budget (GiB) into the megabytes the protocol reports.
///
/// The float→integer cast in Rust saturates, so a negative or non-finite budget becomes `0` (an
/// obviously wrong value the orchestrator will not route real layers to) instead of panicking;
/// `WorkerConfig::validate` already rejects those budgets before this is reached.
pub fn vram_mb_from_gb(gb: f32) -> u32 {
    (gb * crate::constants::inference::GB_TO_MB as f32) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::inference::GB_TO_MB;

    #[test]
    fn vram_mb_from_gb_uses_the_shared_conversion_factor() {
        assert_eq!(vram_mb_from_gb(1.0), GB_TO_MB as u32);
        assert_eq!(vram_mb_from_gb(2.0), 2048);
        assert_eq!(vram_mb_from_gb(0.5), 512);
        // The worker's default budget (`worker.toml` and the CLI help both use 2.0).
        assert_eq!(vram_mb_from_gb(2.5), 2560);
    }

    #[test]
    fn vram_mb_from_gb_saturates_instead_of_panicking_or_wrapping() {
        assert_eq!(vram_mb_from_gb(-1.0), 0);
        assert_eq!(vram_mb_from_gb(f32::NAN), 0);
        assert_eq!(vram_mb_from_gb(f32::INFINITY), u32::MAX);
    }
}
