# tessella-core

> Domain business logic, assignment arithmetic, routing algorithms, unified error types, and core service traits for the Tessella workspace.

## Responsibility

`tessella-core` implements the fundamental domain rules of the worker node:
- **Assignment Arithmetic:** VRAM budgeting, parameter-byte scaling (CUDA vs. Metal vs. Q8_0), embedding/LM-head offsets, and whole-layer range assignment (`assignment.rs`).
- **Routing Engine:** Peer selection based on measured latency, exploration probability, and queue priority (`routing.rs`).
- **Domain Errors:** The centralized `DomainError` enum used across storage, inference, network, and HTTP pipeline layers (`error.rs`).
- **Service Traits:** Abstractions decoupling storage, routing, credits, discovery, and distributed inference (`traits/`).
- **Network & Target Policies:** Callback URL validation, private network filters, and Multiaddr utilities (`callback.rs`, `net.rs`).

It depends on `tessella-protocol` for shared wire types and DTOs, without depending on concrete execution backends (`candle`) or HTTP routers (`axum`).

## Modules

| File / Subsystem | Responsibility | Key Types & Functions |
| :--- | :--- | :--- |
| `src/assignment.rs` | Deterministic layer-range planning based on declared VRAM claims. | `calculate_layer_range_from_mb()`, `calculate_layer_cost_mb()` |
| `src/routing.rs` | Routing table holding known layer peers and latency metrics. | `RoutingTable`, `LayerPeer`, peer latency penalty scoring |
| `src/error.rs` | Canonical domain error types and standard error codes. | `DomainError`, `missing_prefill_error()`, `payload_too_large_error()` |
| `src/constants.rs` | Network limits, memory conversion constants, and hop ceilings. | `MAX_HIDDEN_STATE_BYTES`, `hop_ceiling()`, `HopCeilingSource` |
| `src/config.rs` | Core network and reputation configuration defaults. | `NetworkConfig`, `ReputationConfig` |
| `src/callback.rs` | Security sanitization and private IP filtering for callback URLs. | `CallbackTargetPolicy`, `is_private_ip()` |
| `src/chat.rs` | Structured chat message definition. | `ChatMessage` |
| `src/vram.rs` | Hardware VRAM detection and memory budget parsing. | `VramCalculator` |
| `src/net.rs` | Multiaddr protocol parsing helpers. | `multiaddr_ip()`, `multiaddr_tcp()` |
| `src/time.rs` | Monotonic and Unix epoch time helpers. | `unix_now_secs()` |
| `src/shutdown.rs` | Coordinated graceful shutdown triggers. | `ShutdownSignal` |
| `src/telemetry/` | Observability primitives (peer failure counters, GPU health probes). | `PeerFailureRegistry`, `GpuHealth` |
| `src/traits/` | Core interface contracts for dependency inversion. | `PipelineRouter`, `StorageService`, `DistributedInference`, `CreditService` |

## Usage

```rust
use tessella_core::{
    error::DomainError,
    routing::{RoutingTable, LayerPeer},
    assignment::calculate_layer_range_from_mb,
    constants::inference::hop_ceiling,
    models::ChatMessage,
};
```
