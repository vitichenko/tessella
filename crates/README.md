# 📦 Tessella Crates

> Modular architecture and crate hierarchy of the Tessella Data Plane workspace.

## Architectural Layers

```text
Level 4: Binaries & Clients       [tessella-node] ──► [tessella-cli]
                                   (bin: tessella-worker)        │
                                         │                       │
Level 3: HTTP Data Plane                 │  [tessella-ingress]   │
                                         │  (ApiHost port)       │
                                         └──► [tessella-api] ◄───┘
                                                     │
Level 2: Compute & Transport     [tessella-inference]  [tessella-network]  [tessella-storage]
                                         │                      │                  │
Level 1: Domain Logic                    └──────────────► [tessella-core] ◄────────┘
                                                                 │
Level 0: Contracts & Primitives  [candle-kernels] ──► [tessella-protocol] ◄── [tessella-telemetry]
```

> **The data plane does not depend on the HTTP crate (`SOLID-89`).** `worker → api` is gone: `worker`
> holds `Arc<dyn tessella_ingress::ApiHost>` and `tessella-node` is the only package that links both
> sides of the port. `cargo tree --invert tessella-api` shows `tessella-cli` and `tessella-node`.

---

## Crate Directory

### Level 0: Foundation Contracts & Telemetry
- **[`protocol`](protocol/)**: The canonical bridge contract between Tessella and Musivum. Contains network DTOs, wire enums, `ModelConfig`, `WorkTicket`, `WorkProof`, and cryptographic signing primitives.
- **[`telemetry`](telemetry/)**: Observability foundation. Initializes `tracing`, `tracing-subscriber`, and `tracing-appender` for console and rolling disk logs.
- **[`candle-kernels`](candle-kernels/)**: Custom CUDA kernels patched into Hugging Face `candle` via `[patch.crates-io]`. A workspace member since `TOOL-11`, so the gate's `fmt`/`clippy -D warnings`/`test` reach it; its build script shells out to `nvcc`, which must be on `PATH`.

### Level 1: Domain Logic
- **[`core`](core/)**: Pure business and domain logic. Implements VRAM budgeting, whole-layer assignment arithmetic, the `RoutingTable` and `LayerPeer` types, unified `DomainError` definitions, and service traits.

### Level 2: Compute, Storage & Network
- **[`inference`](inference/)**: The Candle-based ML engine. Hosts backends for the Qwen2 family (`Qwen2ForCausalLM` dense and `Qwen2MoeForCausalLM` MoE), safetensors shard loading, GQA attention, RoPE with absolute offsets, sampling, and activation serialization (`HiddenStatePayload`).
- **[`network`](network/)**: P2P discovery and gossip via `libp2p` 0.53. Manages Noise encryption, Kademlia DHT (`/tessella/kad/1.0.0`), GossipSub mesh topics, and mDNS. *(Tensor activations travel over HTTP, never libp2p).*
- **[`storage`](storage/)**: Local SQLite persistence. Manages the durable Write-Ahead Log of signed work proofs (`pending_work_proofs`), peer cache, and Hugging Face artifact bookkeeping.

### Level 3: HTTP Data Plane
- **[`ingress`](ingress/)**: The HTTP ingress **port**. Owns the `ApiHost` trait (`serve(bind, state)`), the shared `ApiState`, `MusivumState` and `RegisteredWorker` types, and the re-exported capability auth primitives. No HTTP framework: it exists so the data plane can serve HTTP without linking Axum (`SOLID-89`).
- **[`api`](api/)**: Axum adapter implementing the `ApiHost` port (`AxumApiHost`). Serves inter-hop tensor forwards (`/v1/forward`), OpenAI-compatible completion endpoints (`/v1/chat/completions`), FIM completions, health endpoints (`/status`), and single-use capability callbacks (`/v1/token_stream/{request_id}`).

### Level 4: Applications & Binaries
- **[`worker`](worker/)**: The headless compute node **library**. Manages lifecycle orchestration (`WorkerService`), registration with Musivum, HTTP pipeline routing with automatic failover, and operator economy CLI subcommands (`tessella-worker economy ...`). It receives the HTTP surface as an injected `Arc<dyn ApiHost>`.
- **[`node`](node/)**: The **composition root**. Builds `AxumApiHost` over `ApiState` and injects it into `WorkerService::new`, then runs the daemon. Its binary is named `tessella-worker` (paths, Docker `CMD` and scripts are unchanged) and it is the only package that links both the data plane and the HTTP crate (`SOLID-89`).
- **[`cli`](cli/)**: The user-facing terminal client. Provides interactive chat REPL and FIM streaming via the worker's HTTP API.

---

> **External Control Plane:** The orchestrator lives in a separate workspace: **Musivum** (`../musivum`).
> Musivum consumes `tessella-protocol`, `tessella-core`, `tessella-network`, `tessella-storage`, and `tessella-telemetry` as path dependencies.
