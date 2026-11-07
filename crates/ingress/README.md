# tessella-ingress

> The HTTP ingress **port** of the data plane: what the API surface is served *for*, without the HTTP crate (`SOLID-89`).

## Responsibility

`tessella-ingress` exists to break one dependency: until `SOLID-89`, `tessella-worker` declared `tessella-api` only to call `start_api_server` and to name `ApiState`, so the daemon linked Axum and the whole HTTP crate and could not be built — nor tested — without them. The dependency is inverted here:

- **The port:** `ApiHost` is the trait the daemon calls to serve its HTTP surface (`serve(addr, state)`). Like the function it replaces, an implementation never returns on success: it serves until the process stops, so only its `Err` (bind failure, server death) is acted on.
- **The state:** `ApiState` is the process data the surface publishes (engine, storage, health, capability registry, token-stream channels), together with the `Musivum`-role records (`MusivumState`, `RegisteredWorker`).
- **No HTTP framework:** nothing in the dependency tree pulls Axum. `tessella-api` implements the port (`axum_host::AxumApiHost`) and re-exports these types, so the handlers did not have to change; `tessella-node` is the composition root that builds the implementation and injects it.

The state is *process data*, not a wire contract — every field is a trait object the daemon already owns or a handle to state it shares with it — which is why it does not belong with the wire DTOs of `tessella-protocol`.

## Modules

| File | Content |
| :--- | :--- |
| `src/lib.rs` | `ApiHost` (the port), `ApiState` (the published state and its `forget_stream` capability revoker), `MusivumState` and `RegisteredWorker`. Depends on `tessella-core`, `tessella-protocol`, `tessella-network` and `tessella-inference`. |

## Usage

```rust
use tessella_ingress::{ApiHost, ApiState};
use std::sync::Arc;

// The daemon receives the surface; it never builds it (`tessella-node` does).
async fn run(api_host: Arc<dyn ApiHost>, addr: SocketAddr, state: ApiState) -> anyhow::Result<()> {
    api_host.serve(addr, state).await
}
```
