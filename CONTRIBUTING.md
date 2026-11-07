# Contributing to Tessella

Thank you for your interest in making open AI inference truly collaborative and decentralized!

> [!NOTE]
> **Tessella is an open-source project licensed under Apache 2.0.**
> All contributors are required to sign the Contributor License Agreement (CLA) via GitHub CLA Assistant when submitting their first Pull Request.

Tessella is a project with a **very strict** architectural and product focus. To keep the codebase maintainable, performant, and secure, we ask all contributors to respect these guidelines.

## 🎯 Project Philosophy

1. **Extreme Specialization:** Tessella is NOT a general-purpose inference backend (like vLLM or TGI). We specifically target **Code Models** and **Mixture of Experts (MoE)** architectures served as **whole layers** (expert parallelism is retired, `D-15`).
2. **Minimalism:** We prefer to reinvent as little as possible. We use `candle` for tensors, `libp2p` for networking, and `axum` for HTTP endpoints.
3. **Zero-Trust Security:** The network must survive the disconnection of hostile or faulty nodes. Cryptographic signatures and strict error handling are non-negotiable.
4. **GPU-only runtime:** a `device = "cpu"` configuration **aborts** the worker (`D-6`). There is no CPU runtime path, so a change that needs a GPU-less machine has to be a *test-only* path — never a runtime one.

## 🏗️ Development Environment

1. Fork the repository on GitHub and clone your fork locally.
2. Create a branch for your feature or bugfix: `git checkout -b feature/my-new-thing`.
3. Install Rust (`rustup`) and make sure a CUDA (or Metal) device is visible.
4. Run `cargo build` and verify the workspace with `cargo test --workspace` — **serially** (the suite binds a fixed port) — plus `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all -- --check`.
5. Submit a Pull Request targeting the `develop` branch. Upon opening your PR, CLA Assistant will prompt you to review and sign the Apache 2.0 CLA.

## 🚨 Critical Coding Conventions (MUST)

Any change violating these rules is rejected — PR or hand-written commit — to maintain network stability:

- **Language:** All code (variables, methods, comments, logs) and commits MUST be in **English**. 
- **Zero Panic Vectors (`unwrap` / `expect`):** If an operation depends on the network, DashMaps, disk IO, or input from another peer, you MUST propagate the error using `Result` and `?`. We use `tokio::task::spawn_blocking` for heavy tensor math to prevent starving the async runtime.
- **Safe Concurrency:** NEVER use `std::sync::RwLock` or `std::sync::Mutex` across `.await` points. Always use `tokio::sync::RwLock`. Handle poisoned Mutexes explicitly.
- **Dynamic Configuration:** Dimensions (hidden_size, heads, etc.) MUST be read from `ModelConfig` dynamically. Hardcoding model parameters is forbidden.
- **Formatting and Lints:**
  - Must pass: `cargo fmt --all --check`
  - Must pass: `cargo clippy --workspace --all-targets --all-features -- -D warnings`

## 🧠 Model Scope (Why not Llama-3 / Mixtral?)

Tessella intentionally restricts support to the architectures it can actually load: exactly **one** backend
is registered (`Qwen2ForCausalLM` dense and `Qwen2MoeForCausalLM` MoE — the Qwen2 family), plus a
special-cased local `tessella-moe-micro` fixture, which are also what the E2E gate and the harness serve.
There is **no hardcoded model whitelist** (the loader resolves a Hub repo dynamically), but the registry
rejects any other architecture by name, so "add DeepSeek/Llama/Mixtral support" is a new backend, not a
config change.
**Why?**
- We exponentially reduce the testing and compatibility surface.
- By optimizing for a small set of architectures, we can fine-tune the P2P protocol (`HiddenStatePayload` serialization) and absolute RoPE logic to the extreme.
- We focus exclusively on the "assisted programming" use case for the open-source community.

*Please do not open PRs attempting to add generic support for arbitrary HuggingFace models.*

## 🔄 Pull Request Flow

**Closed right now** (see the note at the top of this file). The flow below is how a change is made
*inside* the project today, and what an external contributor will follow the day the window opens.

1. **Development:** Make your changes in your `feature/*` or `fix/*` branch.
2. **Testing:** Add unit tests for any computation logic (`tessella-inference`) or P2P routing/economy updates.
3. **Commit:** Use *Conventional Commits* (`feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`).
4. **Target:** Point your Pull Request to the **`develop`** branch (NEVER `main`).
5. **Review:** A maintainer will review your code. Be prepared to apply requested changes.

## 🐛 Bug Reporting

**No channel is open yet**: GitHub Issues is closed while the project is unfinished, and the board that
tracks the work is private. Keep the report — and the four facts below — ready for the day it opens:

- Tessella version (`git rev-parse HEAD`)
- OS, GPU, and CUDA/Metal version
- Full logs running with `RUST_LOG=debug`
- Exact steps to reproduce the issue.