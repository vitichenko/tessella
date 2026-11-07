# tessella-worker

> Autonomous compute node daemon responsible for whole-layer inference execution, HTTP activation pipeline routing, and economy CLI operations.

## Responsibility

`tessella-worker` is the **data-plane library** of the Tessella network — the daemon binary itself is produced by the composition root [`tessella-node`](../node/README.md) (`SOLID-89`):
- **Registration & Liveness:** Registers with Musivum, validates hardware constraints (GPU-only, `D-6`), obtains its contiguous `layer_range`, and maintains heartbeat announcements.
- **Selective Weight Ingestion:** Downloads only the Hugging Face Hub safetensor shards necessary for its assigned layer range.
- **Pipeline Tensor Forwarding:** Runs the forward pass across assigned layers and forwards activations to the next hop over HTTP (`POST /v1/forward`) using `HttpPipelineRouter`.
- **Proof-of-Compute Settlement:** Verifies orchestrator-issued `WorkTicket` grants and durably buffers signed `WorkProof` records into local SQLite WAL before batch submission.
- **Economy CLI Operations:** Owns the `tessella-worker economy ...` subcommands (`balance`, `ledger`, `reputation`, `federation`), requiring `MUSIVUM_ADMIN_TOKEN`.
- **HTTP surface as a port:** the routes below are served by an injected `Arc<dyn tessella_ingress::ApiHost>` that `WorkerService::new` receives from its composition root, so this crate never links Axum (`SOLID-89`).

## Endpoints

| Route | Purpose |
| :--- | :--- |
| `POST /v1/forward` | Process hidden state activations for assigned layers |
| `POST /v1/chat/completions` | OpenAI-shaped chat completions (entry node tokenizes, runs initial prefill, and coordinates streaming) |
| `POST /v1/generate`, `POST /v1/fim/completions` | Raw generation and fill-in-the-middle completions |
| `POST /v1/token_stream/{request_id}` | Capability-protected callback where tail shard pushes generated tokens |
| `POST /api/v1/update_assignment` | Applies updated whole-layer assignment from Musivum |
| `GET /status` | Health, assignment, hop ceiling, and peer failure snapshot |

## Comprehensive Module Inventory

| File | Subsystem | Responsibility |
| :--- | :--- | :--- |
| `src/main.rs` | *Moved to `crates/node` (`SOLID-89`)* | The daemon binary (CLI parsing with `clap`, key management, `init` wizard) now lives in the composition root, which is the only package that links this library with `tessella-api`. |
| `src/service.rs` | Worker Service Lifecycle | `WorkerService`: Spawns and supervises background tasks, coordinates graceful shutdown via cancellation tokens, and takes the HTTP surface as `Arc<dyn ApiHost>` (`SOLID-89`). |
| `src/inference_engine.rs` | Forward Execution & Proofs | Connects Candle backends to pipeline requests, validates tickets, and writes signed `WorkProof`s to SQLite. |
| `src/http_pipeline_router.rs` | HTTP Tensor Routing | Dispatches activation tensors to subsequent pipeline nodes with timeout enforcement and error recovery. |
| `src/network_manager.rs` | P2P Swarm Coordination | Orchestrates libp2p Swarm events, multiaddress advertising, and identity management. |
| `src/registration.rs` | Orchestrator Handshake | Boot registration with Musivum, bounded by `registration_grace_secs` (`D-25`). |
| `src/heartbeat.rs` | Periodic Announcements | Announce state machine, backoff, and anti-flood protection. |
| `src/economy_cli.rs` | Operator Economy Commands | Implements `tessella-worker economy ...` subcommands interacting with Musivum's economy endpoints. |
| `src/hf_loader.rs` | Weight Downloader | Downloads safetensors shards from Hugging Face Hub scoped to assigned layers. |
| `src/config.rs` | Configuration Engine | Parses `worker.toml`, enforces GPU device constraints, and parses network timeouts. |
| `src/membership.rs` | Cluster Membership | GossipSub membership tracking, peer join/leave state transitions. |
| `src/model_cache.rs` | Weight Cache Management | Disk cache retention, revision pinning, and non-blocking prune operations. |
| `src/peer_discovery.rs` | Discovery Orchestration | Multi-source peer discovery (tracker > gossip > Kademlia DHT > cache). |
| `src/routing_table.rs` | Routing Cache | Worker-side view of active model peers, latency scoring, and exploration candidates. |
| `src/scheduler.rs` | Priority-QoS Scheduler | Prioritizes prefill operations over decodes according to ticket tiers. |
| `src/callback_guard.rs` | Callback Security Guard | Filters callback destinations, disallowing private addresses unless configured. |
| `src/canary.rs` | Shard Health Validation | Model probe testing without full weight loading. |
| `src/task_scheduler.rs` | Periodic Sweeper Tasks | Schedules background SQLite proof drainage and peer refreshes. |
| `src/topology_cache.rs` | Network Topology Cache | Persists active cluster topology to disk to withstand transient tracker downtime. |

## Usage

```bash
# Launch worker daemon (`-p tessella-node` since SOLID-89; the binary is still `tessella-worker`)
HF_TOKEN=hf_xxx cargo run -p tessella-node -- \
  --musivum-url http://localhost:8080 \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --vram-gb 2.0

# Build/test the data-plane library itself
cargo test -p tessella-worker

# Inspect economic balance (requires operator admin token)
export MUSIVUM_ADMIN_TOKEN="<orchestrator_admin_token>"
tessella-worker economy balance
```
