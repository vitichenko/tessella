use crate::error::DomainError;

/// The compute audit a sampled session carries (`TASK-27`/S33, re-scoped by `TASK-70`/S77).
///
/// It is the pair the orchestrator signed into the `WorkTicket`: **which** step to compare (the first
/// layer of the range a hop owns) and **how many** extra executions of it to buy. Both come from the
/// signed ticket, so a hop can neither choose the step it is compared on nor inflate the bill.
///
/// Two facts make the marked step special, and both are derived from `layer` alone:
///
/// * the hop whose range **starts** at `layer` executes it **in isolation** — it applies its layers and
///   returns its own output without routing onward (the node that asked continues the pipeline), which
///   is what makes its answer comparable with a replica's; and
/// * the node that **routed** that step is the auditor: it signs the `AuditReport` of that range, and
///   the orchestrator only accepts it from the entry node or from the peer whose own assignment ends
///   where the marked range starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditOrder {
    /// First layer of the marked range: the step that is compared.
    pub layer: usize,
    /// Extra executions requested on top of the primary (already clamped by [`AuditOrder::of`]).
    pub replicas: usize,
    /// Peer id of the operator's **canary** (`TASK-71` · S78): the auditor has to make sure this node
    /// is one of the executions it compares, because its answer is the known-wrong one the
    /// orchestrator looks for in the report. `None` when the session was sampled without a canary.
    ///
    /// It is read from the same signed ticket as the other two, so the order can neither invent a
    /// canary (a hop would be accusing a third party) nor drop the real one.
    pub canary: Option<String>,
}

impl AuditOrder {
    /// Reads the order out of a **signed** ticket: `audit_layer` is `None` when the session was not
    /// sampled (or when the mesh had no step with a replica to compare), and a broken
    /// `audit_replicas` (zero, or above the ceiling of the protocol) turns the audit **off** instead of
    /// buying an unbounded number of executions.
    #[must_use]
    pub fn of(
        audit_layer: Option<u32>,
        audit_replicas: u8,
        audit_canary: Option<&str>,
    ) -> Option<Self> {
        let layer = usize::try_from(audit_layer?).ok()?;
        let replicas = usize::from(audit_replicas).clamp(
            1,
            usize::from(tessella_protocol::models::MAX_AUDIT_REPLICAS),
        );
        Some(Self {
            layer,
            replicas,
            canary: audit_canary.map(str::to_string),
        })
    }

    /// Whether the step that **starts** at `start` is the marked one.
    #[must_use]
    pub fn marks(&self, start: usize) -> bool {
        self.layer == start
    }
}

#[async_trait::async_trait]
pub trait PipelineRouter: Send + Sync {
    /// Routes a serialized HiddenStatePayload to the worker responsible
    /// for `layer_start`. `total_layers` defines the pipeline end boundary.
    ///
    /// `audit` (`TASK-27`/S33 · `TASK-70`/S77) is the compute-audit order of the session, already read
    /// from its **signed** `WorkTicket`: `None` means «do not audit» and is the behaviour of every
    /// deployment (`audit_sample_rate = 0.0`), so an unaudited session pays nothing extra. When it is
    /// `Some`, this node compares the marked step with `audit.replicas` executions on peers that hold
    /// the **same** assignment range (a peer with another range computes another step, and its answer
    /// would be a false divergence) and reports what it saw, signed, to the orchestrator.
    ///
    /// It used to be a `bool` plus a second lookup of the count inside the router, with a fallback
    /// for a payload that carried no ticket — a combination that could execute a shard twice with
    /// nothing authorising it. One order from one place closes both.
    async fn route(
        &self,
        layer_start: usize,
        total_layers: usize,
        payload_bytes: Vec<u8>,
        model_id: &str,
        audit: Option<AuditOrder>,
    ) -> Result<Vec<u8>, DomainError>;

    /// Refreshes the routing table from tracker.
    async fn refresh(&self) -> Result<(), DomainError>;

    /// Audits the front step (layer 0) against replicas (`TASK-80`/S87, `D-34 = (C)`).
    async fn audit_front(
        &self,
        step: (usize, usize),
        request_bytes: &[u8],
        primary_peer: &str,
        primary_response: &[u8],
        audit: &AuditOrder,
    ) -> Result<(), DomainError> {
        let _ = (step, request_bytes, primary_peer, primary_response, audit);
        Ok(())
    }
}
