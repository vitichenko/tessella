/// Constants for the Tessella network and inference engine.
pub mod network {
    /// Default port offset for the HTTP API relative to the P2P port.
    pub const API_PORT_OFFSET: u16 = 1000;
}

pub mod inference {
    use std::time::Duration;

    /// Default VRAM to MB conversion factor.
    pub const GB_TO_MB: f64 = 1024.0;

    /// Bytes in a Megabyte.
    pub const MB_TO_BYTES: usize = 1024 * 1024;

    /// Maximum size for an inference hidden state (64 MB).
    pub const MAX_HIDDEN_STATE_BYTES: usize = 64 * 1024 * 1024;

    /// How many hop bodies one node reads **at the same time**, i.e. how much of its RAM the wire can
    /// occupy (`TASK-84`/S91): **8**.
    ///
    /// The number comes from the budget of the reference node, not from the traffic: one hop body costs
    /// 70-80 MiB of host RAM before authentication (`S91`: measured flat and linear across 4 passes, with
    /// VRAM untouched — the bytes die in the host), so `8 × ~80 MiB ≈ 640 MiB` is ~8 % of the ~8 GiB it
    /// has free. A hundred of them is the machine, and the honest concurrency of the mesh is **one or two**
    /// (the scheduler serializes the sessions of a node, and a real hop carries ~3,5 MiB, not 64).
    ///
    /// It is published by `GET /status` next to the ceiling and the wait, because the three numbers are the
    /// same deal: how much, until when, and how many at once. `SOLID-89` moved it here from
    /// `tessella-api::hop_body` (re-exported there) because it is a property of the wire contract, and the
    /// data plane's integration tests must be able to name it without linking the HTTP crate.
    pub const HOP_BODY_CONCURRENCY: usize = 8;

    /// How long a hop reader waits **between two frames** of the same body (`TASK-84`/S91): **5 s**.
    ///
    /// It is not a timeout of the body: a hop body of the real ceiling (64 MiB) takes as long as it takes,
    /// and the total is bounded by the `hop_timeout_secs` of the node that *sends* it — the only timer of a
    /// forward (`TASK-76`/S83). What this cuts is the socket that sends nothing: a hop is not a browser,
    /// its frames arrive back to back, so a silence of five seconds is a request held open to keep the
    /// quota and the RAM of the node waiting. The answer is a plain `408` (not a named refusal: the same
    /// body, sent as a stream, is served — so the router may try another candidate).
    pub const HOP_BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(5);

    /// How many tokens of hidden state fit in **one hop** (`TASK-47`/S53, VR-06).
    ///
    /// The first hop of a request transports the hidden state of the whole prompt
    /// (`n_tokens × hidden_size × dtype_bytes`, `Payload::from_tensor` of a `(1, n_tokens)` tensor),
    /// and the wire refuses more than [`MAX_HIDDEN_STATE_BYTES`] per body (`POST /v1/forward`,
    /// `handlers/inference.rs`). This is the arithmetic ceiling of that wire: the model's context
    /// window does not decide how long a prompt may be, the transport does.
    ///
    /// The `dtype_bytes` is the element size of the hidden state this node emits (2 for F16/BF16 —
    /// what CUDA produces — and 4 for F32, which is what a `metal` node casts to). The caller
    /// derives it from the weights it actually loaded (`Weight::dtype()`), not from a constant.
    ///
    /// Returns `None` when the size is unknown (`hidden_size == 0` or `dtype_bytes == 0`), so a
    /// caller can skip the check instead of rejecting traffic, and `None` on overflow.
    pub fn max_input_tokens_per_hop(hidden_size: usize, dtype_bytes: usize) -> Option<usize> {
        let per_token = hidden_size.checked_mul(dtype_bytes)?;
        if per_token == 0 {
            return None;
        }
        Some(MAX_HIDDEN_STATE_BYTES / per_token)
    }

    /// Which of the two ceilings that bound **one hop** decided the number (`TASK-76`/S83).
    ///
    /// Two independent limits cap how long a prompt may be before anyone pays for its prefill: the
    /// wire ([`MAX_HIDDEN_STATE_BYTES`] per body at `POST /v1/forward`) and what this node's VRAM can
    /// actually prefill (the `max_input_tokens` declaration of `worker.toml`). They used to be
    /// published as if they were one number, and the published one was the wire's while the node
    /// really died (`CUDA_ERROR_OUT_OF_MEMORY`, measured in `S76`): knowing **which** of the two
    /// binds is what tells the operator whether to shard the model differently or to lower the
    /// prompt.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum HopCeilingSource {
        /// The wire alone: `MAX_HIDDEN_STATE_BYTES` per body, the node declared no VRAM ceiling.
        Wire,
        /// This node's declaration alone: its VRAM caps the prefill below what the wire could carry.
        Vram,
        /// Both, with the same number: the `min` has two reasons.
        Both,
    }

    impl HopCeilingSource {
        /// The name published by `GET /status` (`max_input_tokens_source`) and used in the refusal
        /// message. Stable strings: they are read by operators and by the harness.
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Wire => "wire",
                Self::Vram => "vram",
                Self::Both => "wire_and_vram",
            }
        }
    }

    /// How long a prompt this node's pipeline can carry **in one hop** (`TASK-76`/S83): the **minimum
    /// of the two limits that are known before a prefill is paid for**.
    ///
    /// * the wire: `MAX_HIDDEN_STATE_BYTES / (hidden_size × dtype_bytes)` tokens
    ///   ([`max_input_tokens_per_hop`], `TASK-47`/S53);
    /// * the node's own declaration: `max_input_tokens` of `worker.toml`, the prompt its VRAM can
    ///   prefill without raising `CUDA_ERROR_OUT_OF_MEMORY` (`0` = undeclared, i.e. the operator
    ///   measured nothing and only the wire binds).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct HopCeiling {
        /// The binding number: `min(wire, declared)`, i.e. what a client may send.
        pub tokens: usize,
        /// Which limit produced [`HopCeiling::tokens`].
        pub source: HopCeilingSource,
        /// What the wire could carry by itself (`None` when the model width is unknown).
        pub wire_tokens: Option<usize>,
        /// What the operator declared this node's VRAM can prefill (`0` = undeclared).
        pub declared_tokens: usize,
    }

    /// The ceiling of one hop: `min(wire, declared)` (`TASK-76`/S83).
    ///
    /// Returns `None` when **neither** limit is known (an unreadable model config and an undeclared
    /// VRAM budget), so a caller skips the check instead of rejecting traffic with a `0`-token
    /// ceiling. A tie is reported as [`HopCeilingSource::Both`]: the same number with two reasons,
    /// which is not the same as "only the wire".
    pub fn hop_ceiling(wire_tokens: Option<usize>, declared_tokens: usize) -> Option<HopCeiling> {
        let (tokens, source) = match (wire_tokens, declared_tokens) {
            (None, 0) => return None,
            (None, declared) => (declared, HopCeilingSource::Vram),
            (Some(wire), 0) => (wire, HopCeilingSource::Wire),
            (Some(wire), declared) if declared == wire => (wire, HopCeilingSource::Both),
            (Some(wire), declared) if declared < wire => (declared, HopCeilingSource::Vram),
            (Some(wire), _) => (wire, HopCeilingSource::Wire),
        };
        Some(HopCeiling {
            tokens,
            source,
            wire_tokens,
            declared_tokens,
        })
    }

    /// Fraction of a wire limit (in percent) at which a payload deserves a warning instead of
    /// silence (`TASK-47`/S53). One definition, shared by the node that **emits** a hidden state
    /// (`worker::inference_engine`) and by the one that **receives** it (`api::handlers::inference`),
    /// so the two cannot disagree about what "close to the limit" means.
    pub const NEAR_LIMIT_PERCENT: usize = 80;

    /// Whether `value` of `limit` is inside the warning band (≥ [`NEAR_LIMIT_PERCENT`] % of it).
    /// A zero limit is never "near": it is an unknown size, not a full one.
    pub fn is_near_the_limit(value: usize, limit: usize) -> bool {
        limit > 0 && value.saturating_mul(100) >= limit.saturating_mul(NEAR_LIMIT_PERCENT)
    }
}

#[cfg(test)]
mod tests {
    use super::inference::{MAX_HIDDEN_STATE_BYTES, max_input_tokens_per_hop};

    /// The numbers VR-06 derived from the model configs of the reference machine:
    /// 64 MiB = 67 108 864 bytes, at 2 bytes per element. They are the *ceiling* of the wire, not
    /// the context window the `config.json` advertises (32 768 for both the 7B and the 14B).
    #[test]
    fn max_input_tokens_per_hop_is_the_measured_ceiling_of_each_model_width() {
        assert_eq!(max_input_tokens_per_hop(896, 2), Some(37_449)); // Qwen2.5-0.5B
        assert_eq!(max_input_tokens_per_hop(1536, 2), Some(21_845)); // Qwen2.5-1.5B
        assert_eq!(max_input_tokens_per_hop(3584, 2), Some(9_362)); // 7B
        assert_eq!(max_input_tokens_per_hop(5120, 2), Some(6_553)); // 14B
    }

    /// A node whose activations are F32 (`metal`) carries **half** the tokens for the same wire
    /// limit: the ceiling is a property of the pair (model width, dtype), not of the model alone.
    #[test]
    fn max_input_tokens_per_hop_halves_when_the_hidden_state_is_f32() {
        assert_eq!(max_input_tokens_per_hop(896, 4), Some(18_724));
        assert_eq!(max_input_tokens_per_hop(3584, 4), Some(4_681));
    }

    /// An unknown size is `None` (skip the check), never a `0`-token ceiling that would reject
    /// every request, and the arithmetic never panics.
    #[test]
    fn max_input_tokens_per_hop_reports_unknown_sizes_instead_of_rejecting_everything() {
        assert_eq!(max_input_tokens_per_hop(0, 2), None);
        assert_eq!(max_input_tokens_per_hop(896, 0), None);
        assert_eq!(max_input_tokens_per_hop(usize::MAX, usize::MAX), None);
        assert_eq!(max_input_tokens_per_hop(1, 1), Some(MAX_HIDDEN_STATE_BYTES));
    }

    /// The warning band is one definition for the emitting node (tokens against the ceiling) and for
    /// the receiving one (bytes against the wire limit): its lower bound is included and a zero limit
    /// is never "near".
    #[test]
    fn is_near_the_limit_has_a_single_inclusive_band() {
        use super::inference::is_near_the_limit;

        assert!(!is_near_the_limit(79, 100));
        assert!(is_near_the_limit(80, 100));
        assert!(is_near_the_limit(100, 100));
        assert!(is_near_the_limit(200, 100));
        assert!(
            !is_near_the_limit(1, 0),
            "an unknown limit is not a full one"
        );
        assert!(!is_near_the_limit(
            MAX_HIDDEN_STATE_BYTES / 4 * 3,
            MAX_HIDDEN_STATE_BYTES
        ));
        assert!(is_near_the_limit(
            MAX_HIDDEN_STATE_BYTES - 1,
            MAX_HIDDEN_STATE_BYTES
        ));
        // The band is integer arithmetic — `value * 100 >= limit * 80`, never a float — so its exact
        // lower edge for a 64 MiB limit is `ceil(80 %)`: the truncated 80 % is still outside.
        let truncated_80_percent = MAX_HIDDEN_STATE_BYTES * 80 / 100;
        assert!(!is_near_the_limit(
            truncated_80_percent,
            MAX_HIDDEN_STATE_BYTES
        ));
        assert!(is_near_the_limit(
            truncated_80_percent + 1,
            MAX_HIDDEN_STATE_BYTES
        ));
    }

    /// `TASK-76`/S83: one hop has **two** limits and what a client may send is the smaller one. The
    /// published number used to be the wire's alone (64 MiB), while a node with 1,2 GB of VRAM died
    /// with `CUDA_ERROR_OUT_OF_MEMORY` at 4096 tokens: the declaration of the node had to take part.
    #[test]
    fn the_ceiling_is_the_minimum_of_the_wire_and_the_declared_vram() {
        use super::inference::{HopCeilingSource, hop_ceiling};

        // Only the wire: the operator measured nothing (`max_input_tokens = 0`).
        let wire_only = hop_ceiling(Some(9_362), 0).expect("the wire alone is a known ceiling");
        assert_eq!(wire_only.tokens, 9_362);
        assert_eq!(wire_only.source, HopCeilingSource::Wire);

        // Only the declaration: the model width was not readable, and the node still knows its VRAM.
        let vram_only = hop_ceiling(None, 2_048).expect("a declaration is a known ceiling");
        assert_eq!(vram_only.tokens, 2_048);
        assert_eq!(vram_only.source, HopCeilingSource::Vram);

        // The 1,2 GB case of `S76`: the VRAM binds, well below the 9 362 the wire would carry.
        let vram_binds = hop_ceiling(Some(9_362), 2_048).expect("both limits are known");
        assert_eq!(vram_binds.tokens, 2_048);
        assert_eq!(vram_binds.source, HopCeilingSource::Vram);
        assert_eq!(vram_binds.declared_tokens, 2_048);
        assert_eq!(vram_binds.wire_tokens, Some(9_362));

        // A declaration **above** the wire is not a licence to send more bytes.
        let wire_binds = hop_ceiling(Some(1_000), 65_536).expect("both limits are known");
        assert_eq!(wire_binds.tokens, 1_000);
        assert_eq!(wire_binds.source, HopCeilingSource::Wire);

        // The same number with two reasons is not "only the wire".
        let tie = hop_ceiling(Some(2_048), 2_048).expect("both limits are known");
        assert_eq!(tie.tokens, 2_048);
        assert_eq!(tie.source, HopCeilingSource::Both);
        assert_eq!(tie.source.as_str(), "wire_and_vram");

        // Neither limit known: skip the check, never reject with a 0-token ceiling.
        assert_eq!(hop_ceiling(None, 0), None);
    }
}
