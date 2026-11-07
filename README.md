# 🌌 Tessella: The BitTorrent of AI Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Written in Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Backend: Candle](https://img.shields.io/badge/Backend-Candle-yellow.svg)](https://github.com/huggingface/candle)
[![Status](https://img.shields.io/badge/Status-v0.0.1%20%C2%B7%20open%20pre--release-orange.svg)](https://github.com/vitichenko/tessella)

> [!NOTE]
> **Tessella is in active open development (`v0.0.1` pre-release).** We welcome community feedback, testing, issues, and contributions. The wire protocol and schemas are evolving rapidly as we scale the mesh.

Tessella is the open-source **data plane** of a Peer-to-Peer AI inference network, written 100% in Rust:
a worker holds a contiguous block of layers of one model and applies it to the hidden state the previous
node sends, so the model does not have to fit in a single GPU.

The **control plane** — layer scheduling, Karma ledger, reputation, and node discovery — is
coordinated by **Musivum** (documentation, manual, and live metrics at [musivum.dev](https://musivum.dev)). A Tessella worker connects to a tracker (such as the public mesh at `https://tracker1.musivum.dev` or a local development tracker) to receive model assignments and peer routes.

Modern AI is heavily centralized in massive data centers. Tessella proposes an alternative path: **harnessing the idle computing power of consumer-grade GPUs worldwide.**

## ✨ Core Pillars & Features

* 🌍 **Energy efficiency by pipeline parallelism:** a node loads only the contiguous block of layers it was assigned, which is what keeps the VRAM claim honest. **Expert parallelism is not implemented** (`TASK-37`/S43 — a shard holding part of the experts still executes the whole layer, so gathering several of them cannot be summed correctly). The 2-D split (layers × experts) is owned by `TASK-64`/`S70`, which is **blocked** since `S65` (decision `D-24`).
* 🔐 **Encrypted transport & signed payloads:** every P2P link uses the libp2p **Noise** handshake with PeerId verification (`crates/network/README.md`: it is **not** TLS/mTLS), and the inference hops travel over **HTTP** (`POST /v1/forward`, decision `D-10`) carrying signed payloads plus a callback capability (`H-10`/`TASK-06`). An intermediate node only sees the hidden state it has to process, and the node that streams the last tokens is authenticated by that capability.
* 🧠 **GPU-only, quantization-aware planning:** `device = "cpu"` **aborts the worker** (decision `D-6`). The planner budgets the real bytes per parameter (2 on CUDA, 4 on Metal, `1.0625` with `Q8_0`), the KV cache of one session and the embedding/LM-head tensors of the pipeline's edge (`TASK-46`/S52). `Q4K` was retired — it had no loader (`D-17`).
* 🤝 **Karma economy over the layers you really serve:** every credit needs the `WorkTicket` the orchestrator issued (`D-7`/`D-13`), and the signed proof is buffered locally in SQLite so a crash does not lose it. The balance buys a queue tier (`> 1000` VIP, `> 0` standard, otherwise leech — `musivum/crates/economy/src/karma.rs`) *unless* the mesh declares `deployment_mode = "local"`, where nobody is privileged (`D-19`). Reading the economy (`/ledger`, `/credits`, `/reputation`) needs the orchestrator's `admin_token` (`D-26`/S73): `export MUSIVUM_ADMIN_TOKEN=…` for the `tessella-worker economy …` commands.

## 🏗️ Workspace Architecture

| Crate | Role |
| :--- | :--- |
| **`tessella-worker`** | The data-plane **library** of the headless compute node: registers with Musivum, downloads only its `layer_range` from the Hugging Face Hub, runs the forward pass, routes the hidden state to the next hop over HTTP, owns the economy CLI (`tessella-worker economy …`) and receives its HTTP surface as an injected `tessella_ingress::ApiHost` (`SOLID-89`). |
| **`tessella-node`** | The **composition root**: it links the data plane with the Axum crate, injects the surface and runs the daemon. Its binary is `tessella-worker` (same name, same path), so the operator's commands did not change (`SOLID-89`). |
| **`tessella-cli`** | The chat client: `chat` (interactive REPL) and `fim`. It talks to a worker's API — it holds no weights, does not tokenize locally and does not manage the economy. |
| **`tessella-ingress`** | The HTTP ingress **port**: `ApiHost` (the trait the daemon serves its surface through) and the `ApiState` it publishes. No HTTP framework, so the data plane can be built and tested without one (`SOLID-89`). |
| **`tessella-api`** | The worker's Axum surface, as the implementation of that port: `POST /v1/forward`, `POST /v1/chat/completions` (OpenAI-shaped), `/v1/generate`, `/v1/fim/completions`, `/v1/token_stream/{request_id}`, `GET /status`. |
| **`tessella-inference`** | The math on HF `candle`: **one** backend for the two architectures the registry accepts (`Qwen2ForCausalLM` dense and `Qwen2MoeForCausalLM` MoE), the shard loader, sampling and the KV cache. |
| **`tessella-network`** | libp2p: Noise transport, mDNS, GossipSub (membership and federation topics), Kademlia discovery, circuit breaker. |
| **`tessella-storage`** | SQLite: registration cache, the durable WAL of signed work proofs, `known_peers`, HF download bookkeeping. |
| **`tessella-protocol`** | The **only** bridge to Musivum: shared DTOs, enums and metric structs. No logic. |
| **`tessella-core`** | Domain logic: configuration, traits, shared constants, assignment arithmetic. |
| **`tessella-telemetry`** | Logging/tracing setup shared by both binaries. |

`crates/candle-kernels/` is a workspace member **and** the `[patch.crates-io]` path of `candle-core`, so
`fmt --check`, `clippy --all-targets -- -D warnings` and `test --workspace` cover it too (`TOOL-11`);
`crates/bench/` is an empty directory outside the workspace. Per-crate detail: [`crates/README.md`](crates/README.md).

## ⚡ Quickstart

### 0. Prerequisites

* **Rust 1.89+**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y` — the
  workspace declares `rust-version = "1.89"` in `[workspace.package]` (measured by `TOOL-13`), and
  `scripts/dev/launcher.sh --gate` builds it with that toolchain.
* **A GPU**: CUDA (with its toolkit) or Apple Metal. **Tessella is GPU-only** (decision `D-6`): a
  `device = "cpu"` aborts the worker with *"CPU device is not allowed in Tessella"*
  (`crates/worker/src/config.rs`). The default `cuda` feature builds `crates/candle-kernels`, whose build
  script shells out to a bare `nvcc`: with no `nvcc` on `PATH` the build panics inside `bindgen_cuda`
  (`TOOL-11`). `scripts/dev/launcher.sh` resolves it itself (`$CUDA_HOME`, `$CUDA_PATH`, `/usr/local/cuda`,
  `/opt/cuda`, `/usr/lib/cuda`) and prints which one it used.
* **A running Musivum orchestrator** (the proprietary control plane). The worker registers there, receives
  its `layer_range` and asks for a `session_grant` per request; without it, a node boots, waits up to
  `registration_grace_secs` (300 s) for it and serves nothing.
* **A Hugging Face token** (read) if the model is gated: <https://huggingface.co/settings/tokens>

### 📥 1. Clone and build

```bash
git clone https://github.com/vitichenko/tessella.git
cd tessella
cargo build --release                                   # or: cargo install --path crates/worker --locked
```

*(With a local build, the binaries are `./target/release/tessella-worker` and `./target/release/tessella-cli`.)*

### 🚀 2. Initialize identity & configuration

```bash
tessella-worker init        # wizard: orchestrator URL, model, VRAM claim, keypair, ports
tessella-worker keys show   # PeerId + public key — what the orchestrator sees
```

### 🧠 3. Contribute as a worker

```bash
export HF_TOKEN=hf_your_token
tessella-worker                                         # reads worker_config.toml
tessella-worker -c /path/to/worker.toml                 # explicit config
tessella-worker --tracker-url https://tracker1.musivum.dev \
                --model-id Qwen/Qwen2.5-32B-Instruct --vram-gb 24.0
```

The **claim** you declare (`vram_gb`) — not the free VRAM — is what decides how many layers you are
offered (`scripts/e2e/plan_split.py` predicts the split before anything starts). The worker logs `Ready`
and serves its own HTTP API on the `api_port` of its config.

#### 💡 Multi-GPU Swarm Example: Splitting Qwen2.5-32B across 2 or 3 GPUs

Frontier models like `Qwen/Qwen2.5-32B-Instruct` (64 layers, ~64 GB weights in FP16/BF16) exceed the VRAM of a single consumer GPU. Tessella pools multiple worker nodes together to serve the model collaboratively:

* **2-GPU Setup (e.g. 2× 24GB RTX 3090 / 4090):**
  ```bash
  # Worker 1 (Machine/GPU 1 - claims 24GB VRAM):
  tessella-worker --tracker-url https://tracker1.musivum.dev \
                  --model-id Qwen/Qwen2.5-32B-Instruct \
                  --vram-gb 24.0 --api-port 8080 --p2p-port 9080
  # Automatically assigned layers 0..31 (embedding + first half of layers)

  # Worker 2 (Machine/GPU 2 - claims 24GB VRAM):
  tessella-worker --tracker-url https://tracker1.musivum.dev \
                  --model-id Qwen/Qwen2.5-32B-Instruct \
                  --vram-gb 24.0 --api-port 8081 --p2p-port 9081
  # Automatically assigned layers 32..63 (second half of layers + lm_head)
  ```

* **3-GPU Setup (e.g. 3× 16GB RTX 4080 / 4070 Ti Super or mixed cards):**
  ```bash
  # Worker 1: claims --vram-gb 16.0 -> assigned layers 0..21 (entry node)
  # Worker 2: claims --vram-gb 16.0 -> assigned layers 22..42 (middle hop)
  # Worker 3: claims --vram-gb 16.0 -> assigned layers 43..63 (exit node)
  ```

*(Note: For rapid single-GPU testing without downloading 64GB of weights, smaller models like `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct` can be run with `--vram-gb 4.0` or `--vram-gb 8.0`).*

### 💬 4. Chat with the swarm

```bash
tessella-cli chat "Explain Mixture of Experts in simple terms" --interactive
tessella-cli --api-url http://127.0.0.1:8080 --model-id Qwen/Qwen2.5-32B-Instruct chat --prompt "Hi"
```

Any OpenAI-shaped client can talk to a worker directly: `POST /v1/chat/completions` on the worker's API
port.

### ⚖️ 5. Read the economy (needs the operator's token)

```bash
export MUSIVUM_ADMIN_TOKEN="<the admin_token of the orchestrator>"   # D-26/S73
tessella-worker economy balance
tessella-worker economy ledger --limit 20
tessella-worker economy reputation <peer_id>
tessella-worker economy federation status        # no token needed: /federation/status is public
```

### ⚙️ 6. Behind NAT or in a pod (RunPod, VPC)

```bash
export MUSIVUM_URL="http://<orchestrator>:8080"
export TESSELLA_PUBLIC_IP="<your public IP>"
export TESSELLA_PUBLIC_P2P_PORT=54326 TESSELLA_PUBLIC_API_PORT=54325
# behind a TLS-terminating proxy, use the whole URL instead:
#   export TESSELLA_PUBLIC_API_URL="https://<pod-id>-8080.proxy.runpod.net"
```

The orchestrator must accept private addresses in those deployments (`ALLOW_PRIVATE_ADDRS=true` /
`allow_private_addrs`), or the node cannot even register. Full recipes: [`docs/HOWTO.md`](docs/HOWTO.md)
and [`README_DEPLOY.md`](README_DEPLOY.md).

## 🚧 What is NOT implemented (read this before promising anything)

| Capability | Status |
| :--- | :--- |
| Expert parallelism (2-D split: layers × experts) | **Not implemented** (`D-15`): whole layers only; `TASK-64` is blocked until it can be exercised and measured. |
| Hidden state over P2P | **Retired** (`D-10`): the hop-to-hop transport is HTTP `POST /v1/forward`. P2P is discovery, membership and topology. |
| P2P weight sync and `GET /substitute` | **Retired** (`TASK-36`/S42): weights always come from the Hugging Face Hub. |
| Continuous batching | **Retired** (`TASK-39`/S45): the Priority-QoS scheduler is real; the "batch flushed" fiction is gone. |
| `Q4K` quantization | **Retired** (`D-17`): it had no loader — use `q8_0` or unquantized. |
| CPU nodes | **Refused by design** (`D-6`). |
| Serving while the orchestrator is down | **Refused by design** (`D-25`): no session grant, no work — the node says so and waits within `registration_grace_secs`. |

## 💖 Sponsorship: The Hardware Bottleneck

Tessella is an open-source project built with passion. The critical path — P2P routing, signed work proofs,
distributed inference — works end to end (every step of the plan is closed by a real E2E gate), but the
plan is **not finished** and this machine's 4 GB GPU is the measured limit of every session.

**However, we have hit a physical limit.** To scale the network and test larger, production-grade models, the current development environment is completely out of memory. 

The immediate goal is to **fund the purchase of an RTX 4070 Ti (16GB VRAM) or a Mac Mini M4 Pro/Max** to serve as a high-availability "Seed Node" and debug the network under heavy production loads.

If you believe in a future where AI is not monopolized by three mega-corporations, please consider sponsoring this project:
* [☕ Buy me a Ko-fi](https://ko-fi.com/vitichenko)
* [💖 Sponsor me on GitHub](https://github.com/sponsors/vitichenko)

## 📄 License

The **data plane** in this repository is licensed under **Apache 2.0** — see [`LICENSE`](LICENSE). The
**control plane** (Musivum) lives in a separate, **proprietary** workspace and is not covered by this
licence (see [`../musivum/docs/VISION.md`](../musivum/docs/VISION.md)).

## 📜 Changelog & where to look next

There is **no auto-generated "last 7 days" section in this file** any more: the one that used to live here
was generated by `scripts/update_changelog.py` and stopped in June 2026, so it lied by omission. The real
changelog is hand-written and lives one directory up:

| Document | What it answers |
| :--- | :--- |
| [`../changelog.md`](../changelog.md) | every version, with the **BREAKING** changes and what to do about them |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | core invariants, crate hierarchy, and end-to-end distributed inference flow |
| [`docs/HOWTO.md`](docs/HOWTO.md) | the operator's guide: cases (1 worker, 2 workers, machines on the Internet), the env-var table and the troubleshooting list |
| [`docs/ECONOMY.md`](docs/ECONOMY.md) | what is paid, how the Karma tiers work, and where the economy can be read (`D-26`) |
| [`README_DEPLOY.md`](README_DEPLOY.md) | Docker / RunPod / GCP deployment: NAT, proxies, port maps, build contexts |
| [`crates/README.md`](crates/README.md) | one line per crate and per module |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | conventions, model scope, PR flow |
| [`../README.md`](../README.md) · [`../instructions.md`](../instructions.md) | how the monorepo is executed and the mandatory E2E gate of every step |

