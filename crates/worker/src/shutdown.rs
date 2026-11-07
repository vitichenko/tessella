//! Bounded, cancel-aware shutdown of the worker's long-lived tasks (H-08, `TASK-10` Part A).
//!
//! The helper itself lives in `tessella_core::shutdown` since S35 (`TASK-29`): Musivum needs
//! exactly the same policy for its own loops (federation sync, SSE pusher, autoscaler) and cannot
//! depend on this crate, so «cancel first, join with a bounded grace, warn and abandon» has **one**
//! definition shared by both workspaces instead of three copies. This module is the worker's stable
//! entry point to it: the call sites (`crate::shutdown::cancel_and_drain`,
//! `crate::shutdown::tick_or_cancel`) and the docs that mention them keep working unchanged.
//!
//! Nothing in the shared helper builds a `Swarm`, binds a socket or touches the GPU, so every
//! branch is unit tested without a daemon. Its tests live next to the implementation.

pub use tessella_core::shutdown::{
    cancel_and_drain, cancel_and_drain_within, sleep_or_cancel, tick_or_cancel, DrainOutcome,
    SHUTDOWN_GRACE,
};
