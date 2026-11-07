# tessella-api

> High-throughput Axum HTTP data plane exposing worker inference routes, OpenAI-compatible completion endpoints, and single-use streaming callbacks.

## Responsibility

`tessella-api` is the **Axum adapter** of the ingress port (`SOLID-89`): it implements `tessella_ingress::ApiHost` (`axum_host::AxumApiHost`) and re-exports the shared state, so no handler path changed when the port was extracted. It provides the HTTP/1.1 transport boundary for the worker node:
- **Inter-Hop Tensor Forwarding:** Accepts serialized `HiddenStatePayload` frames (`POST /v1/forward`), enforcing the 64 MiB payload body limit (`HOP_BODY_LIMIT`) and admitting at most `HOP_BODY_CONCURRENCY` (**8**) hop bodies **at once** (`TASK-84`/S91): one body costs 70-80 MiB of host RAM *before* it is authenticated, so the quota is what bounds the memory a node can be made to spend from outside.
- **OpenAI-Compatible Inference Surface:** Serves `POST /v1/chat/completions` (supporting SSE streaming and non-streaming responses), `POST /v1/generate`, and `POST /v1/fim/completions`.
- **Dynamic Session Resumption:** Coordinates KV cache continuity across multi-turn chats via canonical session IDs (`session_id`).
- **Capability-Protected Token Stream:** Receives generated tokens streamed back from the pipeline's tail node (`POST /v1/token_stream/:request_id`) via single-use Ed25519 capability signatures.
- **Node Health & Assignment Updates:** Exposes `GET /status` (publishing hop input ceilings, the hop-body quota and its live counter, GPU health, and peer failure metrics) and `POST /api/v1/update_assignment` for orchestrator re-partitioning.

## Endpoints

| Route | Method | Purpose | Protection / Body Limits |
| :--- | :--- | :--- | :--- |
| `/v1/forward` | `POST` | Processes incoming hidden state activations across assigned layers. | Signed frame verification, anti-replay nonces, `HOP_BODY_LIMIT` (64 MiB) and the hop-body quota (`HOP_BODY_CONCURRENCY`, `TASK-84`/S91). |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat endpoint (SSE streaming or unary JSON). | Validated against model context ceiling. |
| `/v1/generate` | `POST` | Raw prompt completion endpoint. | Bounded by max output tokens. |
| `/v1/fim/completions` | `POST` | Fill-In-the-Middle (FIM) code completion endpoint. | Bounded by max output tokens. |
| `/v1/token_stream/:request_id` | `POST` | Callback endpoint where tail shard pushes generated tokens. | Single-use ephemeral capability signature (`H-10`). |
| `/api/v1/update_assignment` | `POST` | Receives updated whole-layer assignment from Musivum. | Orchestrator signature verification. |
| `/status` | `GET` | Reports worker health, layer range, hop input ceilings, and the hop-body quota with its live counter. | Public diagnostic endpoint. |

## Modules

| File | Content |
| :--- | :--- |
| `src/lib.rs` | Axum `Router` assembly, route definitions and server bootstrap (`start_api_server`); re-exports `ApiState`/`MusivumState`/`RegisteredWorker` from `tessella-ingress` and `TokenStreamPayload` from `tessella-protocol` (`SOLID-89`). |
| `src/axum_host.rs` | `AxumApiHost`: this crate's router seen by the daemon through the `tessella_ingress::ApiHost` port. |
| `src/hop_body.rs` | Custom streaming body extractor enforcing the 64 MiB tensor buffer limit (`HOP_BODY_LIMIT`), the **named refusals** that go with it (`X-Tessella-Refusal`), the `HopBodyQuota` admission control (`HOP_BODY_CONCURRENCY`, `TASK-84`/S91) and the idle deadline that cuts a body which stops arriving (`HOP_BODY_IDLE_TIMEOUT`). The two knobs live in `tessella_core::constants::inference` (`SOLID-89`) and are re-exported here. |
| `src/callback_auth.rs` | Ephemeral, single-use capability verification for callback streams: the registry and its TTL/rate-limit rules live in `tessella_core::callback::capability` (`SOLID-89`); this module keeps the `HeaderMap` adapter. |
| `src/handlers/inference.rs` | Handlers for `/v1/forward`, `/v1/chat/completions`, `/v1/generate`, `/v1/fim/completions`, and `/v1/token_stream`. |
| `src/handlers/worker.rs` | Administrative handlers: `/status` health reporting and `/api/v1/update_assignment`. |

## Usage

```rust
use tessella_api::{start_api_server, ApiState, ChatCompletionRequest};
```

The daemon does not call `start_api_server` directly any more (`SOLID-89`): its composition root
(`tessella-node`) installs the router behind the port and hands it over as `Arc<dyn ApiHost>`.

```rust
use tessella_api::axum_host::AxumApiHost;
use tessella_ingress::ApiHost;

let host = AxumApiHost::new();
host.serve(addr, state).await?;   // serves until the process stops
```
