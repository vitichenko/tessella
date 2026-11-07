//! The shard of a node as one or more contiguous ranges of a single model (`TASK-96`/S102).
//!
//! A node serves a **set** of disjoint half-open ranges, in ascending order. The mesh used to
//! carry that shape in one place only — `Role::Worker.layer_ranges: Vec<(usize, usize)>`, already
//! published in `NodeInfo.role` — while `WorkerAssignment`, `WorkerReady` and `LayerPeer` carried a
//! single `(usize, usize)` and every reader of the role discarded all but the first entry
//! (`.first().cloned().unwrap_or((0, 0))`). One range per node was therefore an artefact of the
//! read path, not a property of the arithmetic: a worker with surplus VRAM can take a second
//! disjoint chunk of the **same** replica group, and the only thing the mesh needs from it is to be
//! able to ask «which of your ranges serves this step?».
//!
//! The invariant that makes a set of ranges usable is the hop contract: **one hop executes exactly
//! one contiguous range** (`InferenceBackend::apply_layers`). A node that owns two ranges owns two
//! backends, one per range, and the hop that reaches it names the step it is about to serve — never
//! «the node's range», because there is no such thing any more.
//!
//! Ranges are half-open (`(start, end)` covers `start..end`) and ascending. Overlapping or
//! touching ranges are **refused** ([`validate`]) instead of merged: a node that declares
//! `(0, 12)` and `(8, 20)` would load the same layers twice and its routing entry would answer two
//! different ranges for the same step, which is exactly the silent double-load `Invariant 17`
//! («explicit failure over silent degradation») forbids.

use crate::error::DomainError;

/// The layers a node serves: disjoint, ascending, half-open ranges of one model.
pub type LayerRanges = Vec<(usize, usize)>;

/// Does `layer` belong to any of `ranges`?
#[must_use]
pub fn covers(ranges: &[(usize, usize)], layer: usize) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| start <= layer && layer < end)
}

/// The range of `ranges` that contains `layer`, if any.
///
/// This is the lookup the router uses to answer «which range of this peer serves the step I am
/// about to send it?»: the answer travels to the peer as the step's range, so a node serving
/// `[0, 4)` and `[8, 12)` never receives a hop that would make it run both.
#[must_use]
pub fn range_covering(ranges: &[(usize, usize)], layer: usize) -> Option<(usize, usize)> {
    ranges
        .iter()
        .copied()
        .find(|&(start, end)| start <= layer && layer < end)
}

/// The range that starts at layer 0 — the one holding the token embedding table, so the only range
/// that can serve the front of a prompt. `None` when the node does not serve the front.
#[must_use]
pub fn head_range(ranges: &[(usize, usize)]) -> Option<(usize, usize)> {
    ranges.iter().copied().find(|&(start, _)| start == 0)
}

/// The range that reaches `total_layers` — the one holding the final norm and the LM head, so the
/// only range that can close a pipeline. `None` when the node is not the tail.
#[must_use]
pub fn tail_range(ranges: &[(usize, usize)], total_layers: usize) -> Option<(usize, usize)> {
    ranges.iter().copied().find(|&(_, end)| end >= total_layers)
}

/// Total layers of the model this shard covers (the sum of its ranges).
#[must_use]
pub fn coverage(ranges: &[(usize, usize)]) -> usize {
    ranges.iter().map(|&(start, end)| end - start).sum()
}

/// Refuses a shard that cannot be served or routed to.
///
/// Every refusal is loud and names the offending range (Invariant 17): a range that is empty or
/// inverted would load nothing and answer hops forever, one that leaves the model would index past
/// its layers, and two ranges that overlap or touch would be loaded twice and make
/// [`range_covering`] ambiguous.
pub fn validate(ranges: &[(usize, usize)], total_layers: usize) -> Result<(), DomainError> {
    if ranges.is_empty() {
        return Err(DomainError::Config(
            "Invalid assignment: it declares no layer range, so the node has nothing to serve"
                .to_string(),
        ));
    }

    for &(start, end) in ranges {
        if start >= end {
            return Err(DomainError::Config(format!(
                "Invalid assignment: layer_start ({start}) >= layer_end ({end})"
            )));
        }
        if end > total_layers {
            return Err(DomainError::Config(format!(
                "Assignment end ({end}) exceeds model total_layers ({total_layers})"
            )));
        }
    }

    for (index, window) in ranges.windows(2).enumerate() {
        let (previous_start, previous_end) = window[0];
        let (next_start, next_end) = window[1];
        if next_start < previous_end {
            return Err(DomainError::Config(format!(
                "Invalid assignment: ranges {window:?} overlap, so layers \
                 [{next_start}, {previous_end}) would be loaded twice (index {index})"
            )));
        }
        if next_start == previous_end {
            return Err(DomainError::Config(format!(
                "Invalid assignment: ranges {window:?} touch and would be one range \
                 [{previous_start}, {next_end}) instead of two (index {index})"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_is_covered_by_the_range_that_contains_it() {
        let shard = vec![(0, 4), (8, 12)];

        assert!(covers(&shard, 0));
        assert!(covers(&shard, 3));
        assert!(!covers(&shard, 4), "the range is half-open");
        assert!(!covers(&shard, 7), "the hole between the two ranges");
        assert!(covers(&shard, 8));
        assert!(covers(&shard, 11));
        assert!(!covers(&shard, 12));
    }

    /// The contrast that justifies the second range existing: one node, two answers.
    #[test]
    fn the_covering_range_distinguishes_two_disjoint_ranges() {
        let shard = vec![(0, 4), (8, 12)];

        assert_eq!(range_covering(&shard, 2), Some((0, 4)));
        assert_eq!(range_covering(&shard, 9), Some((8, 12)));
        assert_eq!(range_covering(&shard, 6), None, "nobody serves the hole");
    }

    #[test]
    fn head_and_tail_are_the_ranges_that_hold_the_edge_tensors() {
        let shard = vec![(0, 4), (8, 12)];

        assert_eq!(head_range(&shard), Some((0, 4)));
        assert_eq!(tail_range(&shard, 12), Some((8, 12)));
        assert_eq!(
            tail_range(&shard, 28),
            None,
            "the model is longer than this shard"
        );
        assert_eq!(coverage(&shard), 8);
    }

    /// A tail-only node has no front and a front-only node has no tail: both refusals are what
    /// keeps a prompt from being scheduled onto a node that cannot start or close it.
    #[test]
    fn a_node_without_the_front_or_the_tail_says_so() {
        let tail_only = vec![(21, 28)];
        assert_eq!(head_range(&tail_only), None);
        assert_eq!(tail_range(&tail_only, 28), Some((21, 28)));

        let front_only = vec![(0, 12)];
        assert_eq!(head_range(&front_only), Some((0, 12)));
        assert_eq!(tail_range(&front_only, 28), None);
    }

    #[test]
    fn a_well_formed_shard_is_accepted() {
        assert!(validate(&[(0, 12), (21, 28)], 28).is_ok());
        assert!(validate(&[(0, 28)], 28).is_ok());
    }

    #[test]
    fn overlapping_and_touching_ranges_are_refused_by_name() {
        let overlap = validate(&[(0, 12), (8, 20)], 28).unwrap_err().to_string();
        assert!(overlap.contains("overlap"), "{overlap}");
        assert!(
            overlap.contains("[8, 12)"),
            "the doubled layers are named: {overlap}"
        );

        let touching = validate(&[(0, 12), (12, 20)], 28).unwrap_err().to_string();
        assert!(touching.contains("touch"), "{touching}");
    }

    #[test]
    fn an_empty_inverted_or_out_of_range_shard_is_refused() {
        assert!(
            validate(&[], 28)
                .unwrap_err()
                .to_string()
                .contains("no layer range")
        );
        assert!(
            validate(&[(4, 4)], 28)
                .unwrap_err()
                .to_string()
                .contains("layer_start (4) >= layer_end (4)")
        );
        assert!(
            validate(&[(21, 29)], 28)
                .unwrap_err()
                .to_string()
                .contains("exceeds model total_layers (28)")
        );
    }
}
