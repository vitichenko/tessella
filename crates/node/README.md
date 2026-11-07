# tessella-node

> The composition root of the data plane: it links the worker library and the HTTP crate, injects the Axum surface, and runs the daemon binary `tessella-worker`.

## Responsibility

`tessella-node` exists for one reason (`SOLID-89`): `tessella-worker` must not depend on `tessella-api`, and Cargo shares `[dependencies]` between the `lib` and `bin` targets of one package — so the daemon binary had to move out of the worker package.

- **Wiring:** builds the concrete implementation of the ingress port (`tessella_api::axum_host::AxumApiHost`) and passes it to `WorkerService::new(config, api_host)` as an `Arc<dyn tessella_ingress::ApiHost>`.
- **Operator surface:** owns the CLI (`clap`): `--config`, `--musivum-url`, `--model-id`, `--vram-gb`, `--port`, `--clean-cache`, `--model-cache-prune`, plus the `init`, `clean-config`, `keys` and `economy` subcommands.
- **Invariant:** it is the **only** package in the workspace that links both the data plane and the HTTP crate. `cargo tree --invert tessella-api` must list `tessella-cli` and `tessella-node`, never `tessella-worker`.

The binary keeps the name `tessella-worker`, so the artifact path (`target/{debug,release}/tessella-worker`), the Dockerfile `CMD` and every `scripts/e2e/*.sh` invocation are unchanged; what changes is the *package* used to build it (`-p tessella-node`).

## Modules

| File | Content |
| :--- | :--- |
| `src/main.rs` | CLI parsing, interactive `init` wizard, key management (`keys show`/`keys generate`), economy subcommands, shutdown-signal handling, and the composition root itself: `AxumApiHost` → `WorkerService::new` → `run`. |

## Usage

```bash
# Launch the daemon (binary name and artifact path unchanged)
HF_TOKEN=hf_xxx cargo run -p tessella-node -- \
  --musivum-url http://localhost:8080 \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --vram-gb 2.0

# Build only the daemon
cargo build -p tessella-node --bin tessella-worker
```
