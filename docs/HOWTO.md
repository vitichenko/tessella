# Tessella: Practical Guide (How-To)

This guide will help you start nodes on the Tessella network to run large code models collaboratively.

## Prerequisites

- **Rust:** Version 1.89 or higher (`rust-version = "1.89"` in `[workspace.package]`; `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`).
- **Hardware:** a **GPU** — CUDA Toolkit installed (or macOS Metal). Tessella is **GPU-only** (decision `D-6`): `device = "cpu"` aborts the worker with *"CPU device is not allowed in Tessella"*, so there is no CPU fallback.
- **HuggingFace:** An access token (Read) from [HuggingFace](https://huggingface.co/settings/tokens).
- **Orchestrator:** a running **Musivum** (the control plane, a separate proprietary workspace): every case below starts one, because a worker is given no work without it.

## Step 1: Installation

```bash
git clone https://github.com/vitichenko/musivum-tessella
cd musivum-tessella
# Build musivum orchestrator
cd musivum && cargo build --release && cd ..
# Build tessella worker/CLI
cd tessella && cargo build --release && cd ..
```
*(If compiling for CUDA, the flag will activate automatically if `nvcc` is detected).*

---

## Case 1: A Local Worker (Development)

Ideal for testing if the model loads on your machine. The worker requires a Musivum orchestrator to connect to.

1. **Start Musivum orchestrator in one terminal:**
   ```bash
   cd musivum
   cargo run --release -p musivum -- --api-port 8080
   ```
   *(It will listen by default at `http://localhost:8080`).*

2. **Start a worker in another terminal:**
   ```bash
   export HF_TOKEN="hf_your_token_here"
   cd tessella
   cargo run --release -p tessella-node -- \
     --musivum-url http://localhost:8080 \
     --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct
   ```
   *(The worker will download weights from HF and report `Healthy` in the log).*

3. **Make a query with the CLI:**
   ```bash
   cd tessella
   cargo run --release -p tessella-cli -- chat "Write a hello world in Rust"
   ```

---

## Case 2: Two Workers on the Same Machine

If you have a model larger than your GPU's VRAM but enough system RAM, you can start multiple workers.

1. **Start Musivum orchestrator** (same as Case 1).

2. **Worker 1 (Different P2P port):**
   ```bash
   export HF_TOKEN="hf_..."
   cargo run --release -p tessella-node -- \
     --musivum-url http://localhost:8080 \
     --port 40040 \
     --model-id Qwen/Qwen2.5-Coder-3B-Instruct
   ```

3. **Worker 2 (Another P2P port):**
   ```bash
   export HF_TOKEN="hf_..."
   cargo run --release -p tessella-node -- \
     --musivum-url http://localhost:8080 \
     --port 40041 \
     --model-id Qwen/Qwen2.5-Coder-3B-Instruct
   ```
Musivum divides the model by the VRAM each worker **declares** (`--vram-gb`, or the value in its config):
each node is offered a contiguous `layer_range`. If the two ranges do not cover every layer the model stays
incomplete (the entry node never serves a request) — `scripts/e2e/plan_split.py` predicts the split.

---

## Case 3: Workers on Different Machines (Internet) — Splitting Qwen2.5-32B across GPUs

For large models like `Qwen/Qwen2.5-32B-Instruct` (~64 GB FP16 weights, 64 layers), multiple consumer GPUs pool their VRAM. The **Musivum orchestrator** coordinates assignments (e.g., public mesh at `https://tracker1.musivum.dev` or a self-hosted tracker).

1. **Worker 1 (Machine A / GPU 1 with 24GB VRAM):**
   ```bash
   export HF_TOKEN="hf_..."
   cargo run --release -p tessella-node -- \
     --tracker-url https://tracker1.musivum.dev \
     --model-id Qwen/Qwen2.5-32B-Instruct \
     --vram-gb 24.0 --api-port 8080 --p2p-port 9080
   # Receives layers 0..31 (embedding + first half of layers)
   ```

2. **Worker 2 (Machine B / GPU 2 with 24GB VRAM):**
   ```bash
   export HF_TOKEN="hf_..."
   cargo run --release -p tessella-node -- \
     --tracker-url https://tracker1.musivum.dev \
     --model-id Qwen/Qwen2.5-32B-Instruct \
     --vram-gb 24.0 --api-port 8080 --p2p-port 9080
   # Receives layers 32..63 (second half of layers + lm_head)
   ```

*(Or across 3 nodes with ~16GB–20GB each: Worker 1 takes layers 0..21, Worker 2 takes 22..42, and Worker 3 takes 43..63).*

*(Note: For P2P control swarm signaling, workers negotiate discovery via libp2p. However, the data plane for inference tensor forwarding is strictly direct HTTP (`POST /v1/forward`, **`D-10`**; Musivum never proxies inference tensors). Workers behind NAT or reverse proxies must expose a reachable HTTP endpoint via `TESSELLA_PUBLIC_API_URL` or public IP/port mapping).*

---

## A node that does not declare its `model_config`

`POST /register` takes an **optional** `model_config` (`hidden_size`, `intermediate_size`,
`num_attention_heads`, …). The case that has none is the **canary** of the operator (`TASK-82`/S89): it boots
with the model **id** and the layer **count** only — no `config.json`, no tokenizer, no shard.

Since `TASK-86`/S93 that absence no longer means *an unknown model*. The planner looks the shape up by
`model_id` in the same table that already knew the layer count of the model
(`tessella_core::assignment::layer_memory_config_for`, the `FALLBACK_SHAPES` rows), so a node without
`model_config` is planned **exactly** like one that reads its `config.json`: a 1.5B node is planned with
hidden 1536 / intermediate 8960 and ≈ 121 MB per layer, not with the conservative default of an unknown
model (hidden 4096 / intermediate 11008, ≈ 386 MB per layer). Before S93 the table knew the two spellings of
the `-Coder-` sizes and not the plain ones, so the model that `swarm.sh`, the docs and `ModelConfig::default()`
all use (`Qwen/Qwen2.5-1.5B-Instruct`) was planned as a 4096-wide one.

The conservative default has not gone anywhere: it is what a `model_id` that **neither** table knows gets (a
fixture, a family outside the registry). The two tables are kept coherent by a test — a model whose layer
count the mesh knows is never planned with the unknown-model cost — because a *wrong* entry is worse than the
default: a node that believes it holds more layers than it can fails later, when the shard does not load
(`CoreError::Capacity`).

To see what a node would be planned with, without loading anything:

```bash
# The dry-run of the tracker: the range, the terms of the cost and the budget of one claim.
curl 'http://127.0.0.1:8080/api/v1/assignment?model_id=Qwen/Qwen2.5-1.5B-Instruct&vram_mb=3072&device=cuda'
# A claim that cannot host a single layer answers 409 with the reason, not a made-up `1`.
curl 'http://127.0.0.1:8080/api/v1/assignment?model_id=acme/mystery-model&vram_mb=256'
```

`GET /peers` is the other half of the answer: a node that registered without a `model_config` is published
with `model_config: null` (and a canary is marked `canary: true`), so a client can tell which node declared
what it runs.

> **The limit, measured in S93.** Planning a node with the shape of *its* model means its claim now covers
> the real cost. On the reference 4 GB GPU (`GTX 1650`) that shows up in the node that owns the edge: at
> `--vram 3.0` the 1.5B plan puts 16 layers plus the 445 MB edge tensor on layer 0, which OOMs — with the
> conservative default the *canary* (which loads no weights at all) used to win that tramo, so the cost was
> being paid by a node that had nothing to load. `--vram 1.8` is what fits on that card (4 nodes of the
> `S78` split, `[0,8) [8,16) [16,24) [24,28)`); a bigger card or a smaller edge charge is a hardware
> question, not a planning one.

---

## Important Environment Variables

| Variable | Usage |
|----------|-----|
| `HF_TOKEN` | (Required) HuggingFace token to download models. |
| `RUST_LOG` | Log level. E.g., `RUST_LOG=info`, `RUST_LOG=tessella_worker=debug`. |
| `TESSELLA_DB_PATH` | Path to the worker's SQLite database. |
| `TESSELLA_PUBLIC_IP` | Overrides the public/external IP address announced to the P2P network and orchestrator. Critical when running behind NAT/proxies (e.g. RunPod, Docker). |
| `TESSELLA_PUBLIC_API_URL` | Full public HTTP URL announced to the swarm (e.g. `https://pod-8080.proxy.runpod.net`). Takes precedence over `TESSELLA_PUBLIC_IP` + `TESSELLA_PUBLIC_API_PORT`. Required behind TLS-terminating proxies. |
| `TESSELLA_PUBLIC_API_PORT` | Overrides the public port for HTTP API forwarding. Used when running worker containers behind NAT/proxies mapping host ports. |
| `TESSELLA_PUBLIC_P2P_PORT` | Overrides the external P2P Libp2p listen port announced in multiaddresses. Necessary for NAT traversal / port-forwarding setups. |
| `TESSELLA_VRAM_MB` | Manually overrides the worker's reported VRAM (in MB). It takes absolute precedence over hardware autodetection to manually enforce node weights and layer assignments in the cluster. |
| `TESSELLA_ALLOW_PRIVATE_ADDRS` | Accepts token-stream callbacks to **private/loopback/link-local** addresses (`1`/`true`/`yes`/`on`). Off by default (H-10): the worker only calls back hosts it knows. Needed in VPC setups (e.g. the RunPod private network) where the entry node announces a private URL. |
| `TESSELLA_CALLBACK_ALLOWED_HOSTS` | Comma-separated host **suffixes** additionally accepted as token-stream callback targets, for reverse proxies whose host is not one of the peers of the mesh (e.g. `proxy.runpod.net`). Matching is on a dot boundary (`abc.proxy.runpod.net` matches; `notproxy.runpod.net` does not). |
| `TESSELLA_WAL_CHECKPOINT_INTERVAL_SECS` | How often the worker consolidates and truncates its SQLite write-ahead log (`PRAGMA wal_checkpoint(TRUNCATE)`, H-21). Default `300` s; `0` disables the task. A non-numeric value aborts the boot. Musivum has the same knob (`wal_checkpoint_interval_secs`) with the same default. |
| `TESSELLA_KADEMLIA_ENABLED` | Whether this worker joins the Kademlia DHT (`TASK-15`/H-06): it announces the model it serves (`tessella/model/<model_id>`) and asks the DHT when the tracker has nothing to offer. Boolean (`1`/`true`/`yes`/`on`), **on by default**; a non-boolean value aborts the boot. `false` makes the worker behave exactly as before S21 (HTTP discovery only), and it is the escape hatch for a node that needs every megabyte. **The cost is measured now** (`TASK-48`/S54, 2026-09-23; the numbers went to the retired `dht-cost-2026.md`, archived in the canonical tarball): in an idle 60 s window the RSS attributable to the DHT is **below the resolution** of the measurement (<0.4 % of the process, with the sign changing between configurations) and the CPU is indistinguishable from the noise in a 1-node mesh (~+1.7 pp of one core in a 3-node mesh, along the path the DHT *uses* to make the nodes meet). The default therefore stays `on`; the periodic work (provider republication every 15 min, records every 30) and a large mesh are outside that window and would need a new measurement. |
| `TESSELLA_QUANTIZATION` | How the weights of this node are stored (`S51`/`TASK-45`): `none` (default, unquantized) or `q8_0`. `Q8_0` shards keep 8.5 bits per parameter and the orchestrator **budgets them** as such, so the split is not overcommitted (before S51 the planner always assumed 2 bytes/param, whatever the schema said). `Q4K` is **retired** — it had no loader — and asking for it aborts the boot naming the value. |
| `TESSELLA_REGISTRATION_GRACE_SECS` | How long this node may wait for the tracker **at boot** before giving up (`TASK-62`/S68, decision D-25). Default `300` s; `0` = the pre-S68 boot (one `POST /register`, `exit 1` on failure). Only an **unreachable** tracker is retried (backoff 2→30 s): a tracker that answers and refuses (`403`/`409`) is fatal and is never retried. A non-numeric value, or one above `3600`, aborts the boot. Note that it does **not** make the node serve during an outage: without the orchestrator's authorisation a session is refused (D-7/D-13). |
| `TESSELLA_CANARY_MODE` | `off` (default) · `echo` · `scale`: this node answers every step with a **known-wrong** tensor of the right shape and dtype instead of computing its layers (`TASK-71`/S78, decision `D-28`). It is the orchestrator's **canary**: a replica hop the orchestrator lists in `canary_workers` so a sampled session's auditor has to compare it, which is what makes the auditor's report checkable. `echo` answers the input unchanged and `scale` multiplies it by 2; both are false by construction for a stack of layers. A canary **loads no weight** (it downloads no shard and occupies no VRAM for weights), creates no backend, and **cannot** serve an entry-node session (it says so). It needs **no model directory** either (`TASK-82`/S89): it boots with a `models_dir` that does not exist — it reads no `config.json`, loads no tokenizer and downloads nothing — so a canary can be deployed on a machine that never cached the model. Any other spelling aborts the boot naming the accepted values. |
| `TESSELLA_MODEL_CACHE_PRUNE` | Whether this node deletes, from the cache on disk, the shards its **new** assignment no longer needs (`TASK-73`/S80): `off` (**default**) · `dry-run` (says what it *would* delete and touches nothing) · `on`. The same three modes are reachable without touching the config file: `model_cache_prune = "dry-run"` in `worker.toml` or `tessella-worker --model-cache-prune dry-run` (a typo aborts the boot naming the accepted values, in the three paths). The unit of a download is the **shard**, never the layer — serving **one** layer of a 7B pulls a **4 877 660 776 B (4,54 GiB)** file — and until S80 nothing removed what a *previous* range needed, so a node whose range moved three times kept **every** shard of the model. The prune only ever walks the directory of **this node's own model** (never the hub root, never a sibling model), only **after** the new range is loaded and serving, and only the `.safetensors`/`.gguf` shards that are now dead weight: a HuggingFace shard is a symlink into `blobs/`, so the symlink goes first and the blob only when nothing else under that model still resolves to it (`config.json`, `tokenizer.json` and **another revision** of the same model survive). It also refuses to touch anything while **another live process** holds this model: every node that loads weights takes a shared `flock` on `<model dir>/.tessella-cache.lock`, and the prune converts **its own** lock to exclusive, so a neighbour's node makes it back off (`LockedByAnotherProcess`, nothing deleted) while this node can still prune its own cache. Measured: one reassignment gives back **4 877 660 776 B (4,54 GiB)**, and the bytes the prune reports are exactly the bytes the directory loses. Any other spelling aborts the boot naming the accepted values. |
| `TESSELLA_P2P_INFERENCE_TIMEOUT_SECS` | How long a node waits for **one hop** to answer before declaring it dead (`TASK-76`/`S83`): the `p2p_inference_timeout_secs` key of `worker.toml`, default `10` s. Until S83 that key had **no reader at all** and the router used a hard-coded **5 s**, so a hop that was working on a long prompt (measured in S76: 2 048 tokens, 3,5 MiB) was counted as failed, burned, and the client read `500 … Pipeline timeout: no peer found for layer N within 30 seconds` after 30 s. The declared number now governs the wait, the **same parser** validates the three paths (`worker.toml`, this variable, and the legacy schema), and `GET /status` publishes it as `hop_timeout_secs`, so what the operator declared is checkable from outside. `0` is refused **at boot**, naming the key and its default (`default: 10`): a hop that may take no time at all is not a configuration. Widening it does **not** turn a mute node into an infinite wait: a hop that never answers is still declared dead and replaced. |


> **The membership topic is a hint, not a gate (`tessella/workers/v1`, `TASK-16`/S22 · measured in
> S58).** A worker that leaves the mesh becomes *not eligible* for routing at once (its entry is kept,
> `TASK-22` semantics), and the **orchestrator** is the one that announces a departure, because it is
> the only node that sees `ConnectionClosed`. The receiving worker accepts that announcement from the
> peer itself **or** from the orchestrator (`check_left_source`); anything else about a third party is
> still dropped. Two honest limits, both measured with real daemons in
> `scripts/e2e/swarm.sh --scenario membership_two_nodes`: (a) before S58 the tracker's announcement was
> rejected as a source mismatch, so the suppression never ran — it does now; (b) the announcement
> normally **loses the race** with the HTTP plane (`GET /peers` drops the peer within one refresh,
> which is by design: `/peers` is the authority), so seeing `📨 … a leave for the peer … was ignored —
> it is not in the routing table any more` is the normal case, not an error.

> **The orchestrator's departure announcements are logged.** When it evicts a peer, Musivum publishes
> the `WorkerLeft` and now **says what happened** (`📨 Membership: announced the departure of …`, or a
> `warn!` if the topic has no mesh to deliver to). Before S58 the publish result was discarded, so a
> lost signal left no trace anywhere.

> **Pinning the orchestrator (`orchestrator_public_key`, `TASK-19`).** The worker only grants a
> `KarmaTicket` its tier if the ticket is signed by the orchestrator and the orchestrator's key is
> the one configured in `orchestrator_public_key` (otherwise every request is served as `Leech`).
> That key is the **raw** Ed25519 public key — **64 hex chars, no `0x`** — and the orchestrator
> publishes it in `GET /p2p_id` as `orchestrator_pubkey`, so it can be copied verbatim:
>
> ```bash
> curl -s http://ORCHESTRATOR:8080/p2p_id | python3 -c 'import json,sys;print(json.load(sys.stdin)["orchestrator_pubkey"])'
> # → paste it into oracle=… of worker.toml as: orchestrator_public_key = "<64 hex>"
> ```
>
> It is a **config-only** key (no `TESSELLA_*` variable). Since S25 the orchestrator keeps its
> identity in a file (`config/musivum.key`), so the pinned value survives its restarts; before that
> a pinned key was invalidated by every restart. A value that cannot be decoded (72-hex protobuf, a
> truncated key, non-hex characters) is **reported with `error!`** and pinning is left off — it is
> never accepted in silence. If the orchestrator answers a different `orchestrator_pubkey` than the
> pinned one, the worker warns once (`🚨 … NOT the one pinned …`) and keeps using the pinned key.

> **The token-stream callback is authenticated (H-10 · `TASK-06` mitigation 2, S12).** The final
> shard of a pipeline POSTs to `{entry_node_url}/v1/token_stream/{request_id}` with
> `Authorization: Bearer <capability>`, where the capability is the hex of the signature the entry
> node sealed *that* payload with. The entry node registers one capability per sealed payload and
> consumes it on use, so a callback that is missing the header, unknown, from another stream or
> replayed is answered `401`, and a stream that exceeds 200 accepted callbacks per second answers
> `429`. Nothing to configure: both halves are the same worker build (`tessella_api::callback_auth`).
> A callback refused with `401` only costs the SSE fast path — the tokens still reach the client over
> the synchronous stack.

---

## Troubleshooting (Common Issues)

1. **`error: failed to run custom build command for 'candle-core v0.x.x'`**
   - **Cause:** `CUDA_PATH` is set but the CUDA toolkit libraries (toolkit/cuDNN) are missing.
   - **Solution:** Install the full CUDA Toolkit (including cuDNN). CPU-only builds are **not** supported: Tessella is **GPU-only** (decision D-6) and the worker refuses to start with any `device` other than `cuda`/`metal` — including `device = "cpu"` (whatever its casing) and any unrecognised value (a typo like `"cuad"` used to run on the CPU by accident; now it aborts the boot).

2. **`unsupported model architecture`**
   - **Cause:** The model architecture is not handled by the inference backend.
   - **Solution:** Use an architecture the backend registry accepts: **`Qwen2ForCausalLM`** (dense) or **`Qwen2MoeForCausalLM`** (MoE) — i.e. the Qwen2 family, plus the local `tessella-moe-micro` fixture, which is special-cased. There is no hardcoded *model* whitelist (the loader resolves the repo dynamically), but there is exactly **one** registered backend, so anything else (a Llama, a DeepSeek-V2, a Mixtral) fails with this message that names the two accepted architectures.

3. **Worker stuck saying `Waiting for 1 more peers to complete layer coverage...`**
   - **Cause:** The selected model requires more VRAM than your single node provides. Musivum has split the model, but nodes are missing to cover the final layers.
   - **Solution:** Start another worker on the network with the same model to aggregate VRAM.

4. **`Error: HTTP status 503` when using tessella-cli**
   - **Cause:** No node has loaded the `lm_head` (final layer) or the pipeline is incomplete.
   - **Solution:** Check the Musivum logs to verify that model coverage is at 100%.

5. **Constant warning: `Retrying shard ... (attempt X/3)`**
   - **Cause:** Unstable connection to HuggingFace.
   - **Solution:** Check your proxy or VPN. Weights are always downloaded from the Hugging Face Hub: the P2P weight sync (which needed a `GET /substitute` lookup on the orchestrator) was removed in `TASK-36`/S42 because it could never work, so nothing copies shards from a sibling worker any more.

6. **Windows socket binding conflicts (`hyper-v port exclusion` or `address already in use` error)**
   - **Cause:** Windows Hyper-V dynamic port exclusions can block standard bindings to the wildcard `0.0.0.0` address on high-range ports.
   - **Solution:** On Windows, the P2P manager conditionally defaults to listening on `127.0.0.1` to bypass Hyper-V exclusion conflicts, whereas it binds to `0.0.0.0` on Linux environments. If external or local network peer connectivity is needed on Windows, you must explicitly set `p2p_host` / `api_host` in the configuration or pass overriding environment variables.

7. **Warning: `🚫 Refusing token-stream callback to 'http://…': …` (H-10)**
   - **Cause:** the final shard of a pipeline validates `entry_node_url` before POSTing the generated tokens back to the entry node. A destination is only accepted when its host is a node of this mesh (its own URL or a peer of the routing table, i.e. what `GET /peers` announced) or matches `callback_allowed_hosts`; private/loopback/link-local addresses additionally need `allow_private_addrs = true`.
   - **Solution:** nothing urgent — the request still completes and the tokens travel back over the synchronous path; only the direct SSE fast-path is skipped. If the callback is expected here, check how the entry node advertises itself (`TESSELLA_PUBLIC_API_URL` / `TESSELLA_PUBLIC_IP`) and set `callback_allowed_hosts` (or `allow_private_addrs` for a private network) accordingly. See the cards `TASK-06` (mitigations 1 and 2) on the board.

8. **`Invalid or missing cryptographic signature`, `409 Signed request rejected: replay detected…` or `UnsupportedVersion` on a hop (H-09)**
   - **Cause:** since `TASK-05` every inference payload carries a **versioned** signature, and since `TASK-26` (S32) `signature_version = 1` is the **only** version accepted: the canonical preimage over session/request ids, `sha256(tokens)`, `entry_node_url`, karma tier, `nonce` and `iat` is checked for signature, clock skew (±120 s) and replay. The legacy preimage (`0`/absent = `bincode(session_id)`, the one-release migration window of D-2) was **retired**: a request that declares it is rejected with `400 unsupported signature_version 0: only 1 (canonical) is accepted`, before any cryptographic work — which is also what an **unsealed** payload produces, since `0` is the value `#[serde(default)]`/`Payload::from_tensor` leave behind.
   - **Solution:** upgrade **every node of the cluster at the same time** — `Payload` travels as `bincode`, which is positional, so a v1 node cannot read a v0 hidden-state payload and vice versa (announced as a breaking change in `changelog.md`). A `400 … only 1 (canonical) is accepted` means the emitter is an **older build** of the worker (or a payload forwarded without its signature): there is no flag to turn v0 back on. A `409` from a single node that repeats a payload means its nonce was already seen within 5 minutes; a `401` on a payload that was forwarded by hand means a field of the preimage was modified on the way (`entry_node_url` and the signature fields must be forwarded verbatim). Verify the clock of the emitter: an `iat` outside ±120 s is rejected. See the fixed-vector test in `tessella/crates/protocol/src/signing.rs`.

9. **`409 missing_prefill: session … was never prefilled on layers [a, b) of this node` (TASK-68)**
   - **Cause:** the payload is a **decode step** of an ongoing session (`seq_start_pos > 0`) that this
     node never pre-filled: it has no K/V for the prefix the payload declares, so it cannot serve the
     step. It appears when a hop **dies in the middle of a generation** and the router tries the layer
     on a replacement node (whose cache is empty by definition), or when a session was evicted after
     `SESSION_TTL_SECS` (300 s) of inactivity. Before `TASK-68`/S75 this case did **not** fail: it
     answered `200` with plausible and wrong tokens (a decode step carries no mask, so it attended to
     its own token only) — the worst possible failure in a public network, and invisible to the audit
     because two replacement hops produce the same wrong tensor.
   - **Solution:** nothing to fix on the node that refuses — it is telling the truth. The **requester**
     has to restart the generation for that session (a new one gets a clean prefill), and the operator
     should look at why the hop went away (`GET /status` of the nodes, `peer_failures`, the tracker's
     `/workers`). The router of the node that was routing the step **aborts** the request on this
     refusal instead of retrying the same bytes against another candidate (decision (a) of `TASK-68`):
     the same bytes cannot be answered elsewhere. The error carries the machine-readable header
     `X-Tessella-Refusal: missing_prefill` next to the `409`. See
     the card `TASK-68` on the board and `TASK-69`/S76 (recovery).
   - **Since `S76` the retry works (decision D-30, «declare the limit»):** the *entry node* drops its
     own K/V of that conversation as soon as a hop refuses (`🗑️ Dropping session '<key>': a hop of this
     pipeline no longer has its prefix …`), because keeping a cache no hop can extend would make the
     retry resume the same doomed offset and fail exactly the same way. The mesh does **not** re-send
     the prefix to the hop that lost it — that is option (b) of `TASK-69`/`S76`, declined **with the
     cost measured** (one extra prefill of the prefix: ~4 s per 1 000 prefix tokens on the reference
     machine, 0,5B, two hops) — so the honest contract is: *a generation whose hop dies mid-stream is
     lost and the client restarts it*. What never happens is a `200` with plausible garbage (`F-1`) or
     a retry that fails the same way twice.

10. **`400 prompt_too_long: … does not fit in one hop …` (`TASK-47`, enforced for real by `TASK-76`/`S83`)**
   - **Cause:** the first hop of a request transports the hidden state of the **whole prompt**
     (`n_tokens × hidden_size × dtype_bytes`), and **two** independent limits cap it: the wire
     (`MAX_HIDDEN_STATE_BYTES = 64 MiB` per body on `POST /v1/forward`, which the hop now really
     accepts — until `S83` axum's default **2 MiB** killed the hop first) and what this node's VRAM
     can prefill (`max_input_tokens` of `worker.toml`; `0` = undeclared). The ceiling of a hop is
     therefore `min(wire, declared)`, and of a pipeline, a property of the **wire** plus the first
     hop's declaration:

     | model width | tokens per hop (F16, 2 B/elem) | tokens per hop (F32, 4 B/elem) |
     | :--- | ---: | ---: |
     | 896 (0.5B) | 37 449 | 18 724 |
     | 1 536 (1.5B) | 21 845 | 10 922 |
     | 3 584 (7B) | 9 362 | 4 681 |
     | 5 120 (14B) | 6 553 | 3 276 |

     `max_input_tokens = floor(67 108 864 / (hidden_size × dtype_bytes))`, with `dtype_bytes` taken
     from the device the node runs on (CUDA casts activations to F16, everything else to F32 —
     `tessella_inference::backends::activation_dtype_bytes`).
   - **Solution:** check `GET /status` — it publishes `max_input_tokens`, `hidden_size` and the model —
     and keep the prompt below it. Otherwise shorten the context, or serve the first layers on a node
     with a **smaller** model (a narrower hidden state carries more tokens per hop). A single-node
     deployment (the whole model on one worker) has no such ceiling: nothing travels.

10. **The worker waits at boot: `🔁 Boot registration: up to 300s of retries if the tracker is
    unreachable` (`TASK-62`/S68, decision D-25)**
   - **Cause:** the tracker could not be **reached** (connection refused/reset/timeout/DNS), so the
     node keeps asking with a backoff of 2→30 s instead of exiting on the first attempt. This is the
     fix for a node that used to die with `Error: Failed to initialize worker service … error sending
     request for url (…/register)` when it started during an outage, which also meant its persistent
     peer cache (`known_peers`) was never used.
   - **Solution:** nothing if the tracker comes back within `registration_grace_secs` (300 s by
     default): the same process registers and carries on (`✅ Registered with the tracker … after N
     attempt(s)`). If the tracker is deliberately down for longer, raise
     `TESSELLA_REGISTRATION_GRACE_SECS` (a boot that may hang for more than an hour is refused) or set
     it to `0` to go back to failing fast.
   - **What it is NOT:** the retry is only for **reachability**. A tracker that answers and refuses
     (`403 worker_not_authorized`, `409 model_at_capacity`, an unreadable body) fails the boot at once
     and names the answer — the authority spoke. And a node never *serves* while its control plane is
     unreachable: the work of a session is authorised per request (`session_grant`/`WorkTicket`,
     D-7/D-13), so a request during the outage is refused immediately with an error that names the
     missing authority (`… the mesh does not serve a session without the orchestrator's authorisation
     …`). The topology does survive (no hop is lost), the service does not; that is the declared limit
     of D-25 and `scripts/e2e/scenarios/tracker_down.py` is its regression test.
   - **Note:** the ceiling is enforced by the entry node **before** the prefill, and the node warns at
     ≥ 80 % of it (`⚠️ Hidden state near the wire limit`); the receiving node warns in the same band
     before answering a `413`. The `413` is still the answer for a hand-crafted oversized body, which
     is why a `400` naming the numbers is a better contract than a `413` from a node you do not own.

11. **The answer ignores my `temperature` / `top_p`, or two identical requests differ**

   - **Cause (until `TASK-40`/S46):** the generation loop of the entry node hardcoded the swarm
     values (`0.2/0.9/40`) and drew from a thread-local RNG, so the `temperature` and `top_p` the API
     mapped from your request never reached the sampler and `temperature = 0.0` — documented as
     *greedy* — was not greedy. The second sampler (inside the session) computed a token nobody read.
   - **Now:** the parameters of **your request** are honoured: `temperature`, `top_p`, `top_k` and
     `seed` (the last two are new, `#[serde(default)]`). Rules:

     | you send | what happens |
     | :--- | :--- |
     | nothing | the **mesh standard**: `temperature = 0.2`, `top_p = 0.9`, `top_k = 40` (what the mesh always applied) |
     | `temperature: 0.0` | **greedy** decoding: the argmax, deterministic, no randomness |
     | `seed: <n>` | the same request (prompt + parameters + seed) produces the same text |
     | `top_k: 0` | no candidate truncation |
     | a value that cannot be sampled (`temperature < 0`/`NaN`, `top_p` outside `(0, 1]`) | `400 invalid_request` naming the field |

   - `max_tokens` omitted is bounded by the node's `[worker] max_new_tokens` (256 by default; it used
     to be a key nobody read and a hardcoded 512).
   - **The worker's `[worker] seed` and `[worker] temperature` keys are gone** (`TASK-40`/S46): the
     request's parameters are the client's and the mesh standard is a constant shared with the
     parameter contract. A `worker.toml` that still carries them **starts** and logs a `⚠️ … was
     retired` warning. The parameter contract the orchestrator checks (`/register`, `/announce`,
     `/ready`) is built from that same constant, so changing a node's configuration can no longer
     expel an honest node from the swarm.
   - **Note:** the mesh does not need every hop to agree on sampling because only the **entry node**
     samples — the last shard returns logits, and the entry node turns them into the next token. The
     contract still pins the **model**.

12. **The answer names a different `model` than the one I asked for**

   - **Cause:** the worker serves **one** model (its shard of the model the orchestrator assigned) and
     the `model` field of an OpenAI-shaped request is *not* validated against it. Until
     `TASK-56`/S62 the handlers echoed whatever you sent, so a request for `"gpt-4"` was answered with
     `200` and `"model": "gpt-4"` *while the tokens came from the node's own model*: a truthful-looking
     lie.
   - **Now:** the field is **accepted and ignored** (clients send `"default"`, `"gpt-4"` or the real
     repo id by habit) and every answer — chat, streaming chunks and FIM — names the model this node
     **really serves**. Mismatches are logged at `debug!` level. Rejecting the request with
     `model_not_found` was the alternative, and the project's own gate sends `"model": "default"`, so
     it would have broken every client that uses a placeholder.
   - **Where the model of a node comes from:** its `default_model` (the `model_repo` of the assignment);
     `GET /status` publishes it.

13. **A hop of my pipeline does not answer: where can I see it?**

   - **Cause:** the node that *sends* a hop knows immediately (connection refused, 5 s timeout, HTTP
     error, body that never arrives) but the orchestrator cannot be told by a third party:
     `/report_failure` authenticates a single identity (D-3), and an accusation against another peer
     without proof is the griefing S8/S28 closed.
   - **Now (`TASK-50`/S56):** `GET /status` publishes `peer_failures` — per peer and per reason
     (`network_error`, `timeout`, `http_error`, `connection_closed`, `dial_failed`,
     `divergent_audit`), with the cumulative
     total, the **consecutive** count (back to zero as soon as a hop to that peer completes) and the
     instant of the last failure — and the hop is penalised in this node's routing table, so the next
     layer prefers somebody else:

     ```json
     "peer_failures": {
       "total_failures": 3, "tracked_peers": 1, "untracked_failures": 0,
       "peers": [{"peer_id": "12D3Koo…", "total_failures": 3, "consecutive_failures": 3,
                  "last_failure_unix_secs": 1758500000, "last_reason": "timeout",
                  "by_reason": {"timeout": 3}}]
     }
     ```

   - **What the mesh does about a peer that stops answering, instead of an accusation:** the tracker
     sees the P2P connection close and the missing `/announce` heartbeat, the router re-routes locally,
     and a peer that claims work it did not do is what the **sampled compute audit** (`audit_sample_rate`,
     `TASK-27`/S33) is for.

14. **My session was audited: what did the node do, and where did the report go?**

   - **What the audited node does (`TASK-53`/S59):** the entry node runs the marked shard (the one that
     ends the pipeline) on `audit_replicas` extra workers of the same layer range — `2` by default, so
     three executions — and compares the `sha256` of the bare tensor (`Payload::data`; the envelope is
     excluded because each execution seals its own nonce/`iat`/signature).
   - **What it reports:** one signed `AuditReport` per audited step, `POST /api/v1/audit_report`, with the
     `request_id`, the range, the input digest, `(peer_id, output_hash)` of every execution and the work
     ticket of the session. Nothing is sent when there are fewer than two executions to compare: there is
     no evidence to send, and the orchestrator falls back to the S33 rule.
   - **What it costs locally:** the dissidents of a 2-1 split are counted in `GET /status` as
     `divergent_audit` and penalised in this node's routing table; with a tie, **every** execution is
     (the node cannot tell which one is broken). That is the only local consequence: the orchestrator is
     the one that decides the credit, and it blacklists nobody.
   - **If the report does not arrive** (the orchestrator answered a non-2xx, or the POST failed): the log
     says it and the credit of that range falls back to the S33 rule — the first claim is paid. The correct
     answer to «why was I paid / not paid» is in the orchestrator's log (`AUDIT REFUSAL` /
     `Audit report of … recorded`).

15. **`Orchestrator does not serve /api/v1/session_grant (404)`, or the first request of a session fails
    without reaching the GPU (`TASK-57`/S63)**

   - **What it means:** since S63 a worker cannot start a session without the **work ticket** the
     orchestrator mints in `GET /api/v1/session_grant`. The ticket is the authorisation of the session
     (H-12): without it the work could not be paid, so the node refuses to serve it instead of doing GPU
     work for free. The log line names the URL it tried.
   - **The usual cause is a mixed deployment**: the worker is newer than the orchestrator (a tracker built
     before S13 does not serve that endpoint at all). The ticket travels inside the hidden-state payload
     (`bincode`) and the orchestrator requires it as a field of `WorkProof`, so **the two sides deploy
     together** — there is no migration window to schedule any more.
   - **Documentation that used to say otherwise** (`require_work_ticket = false` to credit work without a
     ticket) was retired in S63: that key no longer exists and a configuration file that still carries it
     does not boot. The same applies to `peer_signature_policy`: the authentication fields are mandatory
     in every signed DTO, so an unsigned `/announce`, `/ready`, `/deregister`, `/report_failure` or
     `/register` is a `422` instead of a `401` with a warning.
   - **If the orchestrator answers `403`/`409` instead of `404`:** it means the orchestrator is the current
     one and the refusal is a policy one — `403` is «this worker is not registered or serves another
     model», `409` is «this ticket was not issued here / has expired». Both are in the orchestrator's log
     with the `request_id`.

16. **`⚠️ Heartbeat: cannot announce itself any more …` — the node has stopped talking to the tracker
    (`TASK-65`/S71, family of D-25)**

   - **What it means:** the heartbeat of this node (`POST /announce`, every `heartbeat_interval_secs`)
     did not end in a `2xx`, and the line names which of the three cases it is: `the tracker could not
     be reached …` (connection refused/reset/timeout: the tracker is down or unreachable from here),
     `the tracker refused the heartbeat (status …)` (it answered and said no — a signature or admission
     problem, a `5xx`), or `the tracker no longer knows this node (status 404/410)` (it restarted or
     lost its registry).
   - **One line per transition, not per tick:** this used to be a `match` that only looked at
     `404`/`410` and swallowed everything else in `_ => {}`, so a node that had been unable to announce
     itself for an hour looked exactly like a healthy one. Now the line appears when the state changes
     and once more when it recovers (`✅ Heartbeat: announcing itself again …`), so a tracker that is
     down for hours does not fill the log. The re-registration that follows a `404`/`410` also says
     whether it worked, and if it did not, whether the tracker was **unreachable** (it will be retried)
     or **refused** the node (waiting will not change its mind).
   - **What it costs while it lasts:** the node keeps routing from its cached mesh view (`/peers` is
     cached, no hop is lost), but the tracker does not count it alive: expect it to be missing from
     `GET /workers` and, if the model is split, its layers to be uncovered until the announce succeeds
     again.
   - **Solution:** read the quoted reason. `worker_not_authorized` is the admission mode of the
     orchestrator (`deployment_mode`/`authorized_workers`, D-19); a `5xx` is the tracker's own problem;
     `404`/`410` needs nothing (the node re-registers itself); `could not be reached` means checking the
     network path and `orchestrator_url` (`MUSIVUM_URL`/`ORCHESTRATOR_URL`, or the `--musivum-url` flag —
     there is no `--tracker`).

17. **`403`/`503` from the economy commands (`tessella-worker economy balance | ledger | reputation`)** —
    the economy surface is the operator's (`TASK-67`/S73 · **D-26**)

   - **What it means:** since S73 `GET /ledger`, `GET /credits` and `GET /reputation` are admin
     endpoints, exactly like `GET /ledger/snapshot` since S40: they serve balances, work proofs and the
     reputation of every node, which is the operator's business. `403` (or `401`) means the request went
     out without a token, or with a different one; `503` means the orchestrator has **no usable**
     `admin_token` configured, so its admin endpoints are off — the same answer `/slash` has always
     given.
   - **Solution:** export the orchestrator's token (the value of `admin_token` in its `musivum.toml`)
     and repeat:

     ```bash
     export MUSIVUM_ADMIN_TOKEN="$(openssl rand -hex 32)"   # or the one already in musivum.toml
     tessella-worker economy balance
     ```

   - **What still works without it:** `tessella-worker economy federation status` (it reads the public
     `GET /federation/status`) and everything about the mesh (`/peers`, `/workers`). What the token does
     **not** open: `GET /weights/manifest[s]` never needed one — the digest is what a node verifies its
     own weights against.

18. **Multi-turn: how `session_id` works, and who issues the id (`TASK-75`/`S82`, `TASK-78`/`S85`)**

   - **What it is:** `session_id` is the **continuity key** of the K/V cache of a conversation. Send the
     same value in the next turn and every node of the pipeline (the *entry node* **and** every hop)
     finds its own K/V and applies only the new tokens (`♻️ Resuming existing session '<id>' for
     multi-turn inference at pos N` + `Slicing input tokens: seq_start_pos=N, total_tokens=M`). Omit it
     (or send one this node never issued) and each turn is a fresh prefill — the answer names the session
     it opened, and *that* is the value to send next time.
   - **Before `S82` this did not work**, and in a pipeline of **three or more** nodes it was worse: the
     *entry node* stored the session under the work id while looking it up under the client's id (so a
     second turn **always** paid the whole prefill), and each hop rewrote `session_id` with the work id —
     which is part of the signed preimage — so the next hop rejected a perfectly good payload with
     `🔏 Rejecting signed request: invalid signature` and the client read a `500 … Pipeline timeout`
     after 30 s. Any client that used the field made the pipeline fail.
   - **The contract, in full:** the client must re-send the **whole conversation** each turn (the API is
     the OpenAI chat shape): the node slices the new suffix by the position it already applied. The K/V
     lives per node for `SESSION_TTL_SECS` (300 s, swept every 60 s), so a turn that arrives later
     prefills from scratch — and with the entry node's session already dropped, a hop that lost the
     prefix answers `409 missing_prefill` (item 9 above), which is the declared limit of D-30.
   - **The id is issued by the node (`TASK-78`/`S85`):** what a client sends in `session_id` is a
     **hint**, not an id. The node mints the canonical id (`sess-` + 32 hex digits) and honours the hint
     **only** when it names a session that node still holds — the same map the TTL sweeper walks — so an
     invented value, one copied from another client, or one whose session the sweeper already dropped
     opens a **new** conversation instead of continuing somebody else's. Two clients that pick the same
     value no longer share a K/V cache.
   - **Where the id comes back:** **every** answer whose request can carry a hint publishes it —
     `session_id` in the JSON body of `/v1/chat/completions` (non-streaming) and the
     `x-tessella-session-id` header of the streaming chat answer and of `/v1/fim/completions`. The header
     exists because a stream's last event is already defined by the OpenAI shape, and because a header is
     readable **before** the first byte: a client keeps the id even when the stream is cut. A client that
     sent **no** hint gets one anyway, which is what lets a single-turn client start resuming.
     `/v1/generate` publishes neither: it has no `session_id` to hint with.
   - **How a client resumes:** send the id the answer named back in `session_id` on the next turn — the
     CLI REPL does this and so does `scripts/e2e/scenarios/prefill_cost.py` — and the id of a resumed
     conversation does not change, so it can be kept for the whole conversation. An id this node never
     issued does **not** fail the request: it is served and the answer names a **different** session.
   - **What it is not: a credential.** The id is unguessable (`sess-` + 128 bits from the OS CSPRNG) and it
     only names a K/V cache that the TTL sweeper (300 s) or a restart already drops, so a leaked id grants
     no compute and no tokens — the admission of a hop (H-09) keeps verifying the signature of every
     payload on its own. It is still a secret of the conversation in the sense a cookie is: don't publish
     it.

19. **A hop body was 2 MiB while `/status` published 37 449 tokens — closed in `TASK-76`/`S83`**

   - **What was measured (`S76`):** `POST /v1/forward` had **no** body limit of its own, so axum's
     default applied: **2 MiB** per request. On the 0.5B F16 that was ≈ **1 170 tokens** of hidden state,
     against the **37 449** that `GET /status` published (computed from `MAX_HIDDEN_STATE_BYTES = 64 MiB`).
     Measured with the harness: 1 065 tokens (1,8 MiB) were served; 2 048 tokens (3,5 MiB) got
     `413 Payload Too Large … Failed to buffer the request body: length limit exceeded` **from the hop**,
     after the entry node had already paid its prefill, and the router turned it into
     `500 … Pipeline timeout: no peer found for layer N within 30 seconds` (it retried and suspended 30 s).
     Some long prompts did not even get that far: the attention matrix ran out of memory
     (`CUDA_ERROR_OUT_OF_MEMORY` on a 4 GB card).
   - **What it is now:** the inference routes mount `DefaultBodyLimit::max(HOP_BODY_LIMIT)` with
     `HOP_BODY_LIMIT = MAX_HIDDEN_STATE_BYTES` (**64 MiB**), and the body is read through an extractor
     that refuses **at the crossing**: a declared `Content-Length` over the limit is answered `413`
     *without* buffering the body. The published ceiling is therefore the real one, and the **two** limits
     that bind a hop are published separately: `max_input_tokens` (the binding number, `min(wire, declared)`)
     and `max_input_tokens_source` (`wire` · `vram` · `wire_and_vram`). A refusal is **named**
     (`X-Tessella-Refusal: prompt_too_long` · `payload_too_large`), so the router **aborts** instead of
     burning the next candidate and the client reads the `400`/`413` instead of a `500` after 30 s.
     Measured in `S83` with the harness: a 2 048-token prompt (3,5 MiB) is **served**; before the fix it
     was the `500` above. What still binds on a small card is the **VRAM** (`source = vram`): a
     4 096-token prefill answers `500 Inference error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of
     memory")`. Since `S84` that `OOM` is a statement about the **request** and the node keeps serving
     (item 20).
   - **The wait is declared too:** `p2p_inference_timeout_secs` (`TESSELLA_P2P_INFERENCE_TIMEOUT_SECS`,
     default `10` s, `0` refused at boot) governs the router, and `GET /status` publishes it as
     `hop_timeout_secs`. A hop that takes longer is **slow**, not dead: in `S83` a hop that answers in
     6 s is served in 6 s with `hop_timeout_secs = 30`, and a hop that never answers is still declared
     dead and replaced with `hop_timeout_secs = 1`. Treat a `Pipeline timeout` on a long prompt as the
     VRAM ceiling of the first hop, not as the wire.

20. **A transient GPU `OOM` left the node `unhealthy` for good — closed in `TASK-77`/`S84`**

   - **What was measured (`S76`):** a 4 096-token prefill that did not fit in a 4 GB card answered
     `500 … DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` and the node went out of the mesh
     **until a restart**: `InferenceEngine::is_healthy` was an `AtomicBool` set to `false` by five
     failure paths (the hop forward, the *entry node* forward, the `spawn_blocking` and the two of the
     scheduler) and set back to `true` by **none**, while the first line of `forward_hidden` and of
     `Session::forward` returns early while it is `false`. Every later request of that node — including
     the 38-token turns of the same scenario — read
     `500 {"error":{"message":"Inference error: GPU state is unhealthy/degraded"}}`, and the tracker
     still listed the node as serving.
   - **What it is now:** the two facts that one flag conflated are told apart. `CUDA_ERROR_OUT_OF_MEMORY`
     is a judgement about the **request**: it is counted (`GET /status` → `gpu.out_of_memory`), the
     client gets the refusal of *its* request, and the node keeps serving. A dead **device** is a
     judgement about the **node**, and it is only believed after a bounded probe (a 4-byte tensor
     allocated **and read back**, `device_probe`) fails — then the node declares itself `down` **with its
     reason** and **revalidates** every `revalidate_secs` (the heartbeat interval, published), on the
     arrival of a request and after a `reload_shard`, instead of "until a restart".
   - **How to read it:** `GET /status` publishes the whole state under `gpu` — `healthy`, `verdict`
     (`serving` · `down`), `kind` (`out_of_memory` · `device_lost`), `context`, `reason`, `probe_error`,
     `since_unix_secs`, `down_secs`, `revalidate_secs`, `out_of_memory`, `device_failures`, `recoveries`
     and `last_recovery` (`device_probe` · `shard_reload`):

     ```bash
     # $API_PORT is `api_port` of worker.toml (the sample config writes 8080; the e2e harness
     # derives it as p2p_port + 1000, tessella_core::constants::network::API_PORT_OFFSET)
     curl -s http://127.0.0.1:$API_PORT/status | python3 -m json.tool | sed -n '/"gpu"/,/}/p'
     ```

     A node that declared itself down also announces `Degraded` to the tracker (`POST /announce`), so the
     operator view and the node agree, and it announces `Healthy` again on the first revalidation that
     passes. `out_of_memory > 0` with `verdict = "serving"` is a **healthy** node that refused a
     too-large request: there is nothing to restart.
   - **Measured in `S84`** (real daemons, `swarm.sh --scenario prefill_cost` with the 4 096-token probe):
     the probe still answers the `500` above; `/status` reads
     `gpu.out_of_memory = 1, verdict = "serving", healthy = true` with no `down_secs`; the **next**
     request is served by the same node (`200` in 0.91 s) and the cluster stays healthy with 2/2 workers
     alive — `22 passed / 0 failed`. The 4 096 probe itself is the one `skip` of that run: a limit is a
     measured fact, not a pass.
   - **Resolved in `TASK-85`/`S92` (see item 22 below):** a node whose requests consecutively run out
     of memory reaches the `oom_streak` threshold, marks `gpu.verdict = "saturated"` and announces
     `PeerStatus::Degraded` to Musivum, auto-excluding itself from routing until recovered.

21. **The body of a hop had no quota: it cost 70-80 MiB of host RAM *before* being authenticated — closed
    in `TASK-84`/`S91`**

   - **What was measured (`S91`, four harness passes):** one **unauthenticated** hop body costs **70-80
     MiB** of host RAM — flat and linear, with VRAM **untouched** (+0 MiB in all four passes: the bytes
     die in the host before they reach the card) and `/status` green all the while. 24 bodies at once =
     **+1.6 GiB** of RSS with the node still answering `OK`, and the machine runs out of memory around
     the hundredth. The extractor only looked at its **size**: anyone with sockets could occupy a node
     without signing anything, and the `400 invalid payload` arrives **after** the 64 MiB have been
     buffered.
   - **What it is now:** a node reads **8** hop bodies at a time (`max_concurrent_hop_bodies`) and the
     ninth is refused **before a single byte is read** with `503` + `X-Tessella-Refusal:
     too_many_hop_bodies`. That refusal is **terminal** for the router: the same bytes do fit in that
     node once it is free (and in another candidate right now), but retrying them means pushing the same
     70-80 MiB body at every candidate in turn — the amplification a quota exists to stop — so the retry
     belongs to the **client**, which is what a `503` already means. The permit is released when the
     **bytes** are (not when the read ends): the value that carries a hop's body holds it while the
     handler works.
   - **The silence inside a body is declared too:** a body that stops arriving is cut after
     `hop_body_idle_timeout_secs` (**5 s**) and that answer carries **no** refusal header, because the
     same bytes sent as a stream *are* served. It is not a timeout of the body — the total is bounded by
     the `hop_timeout_secs` of the node that sends it — but of the **silence**: a hop is not a browser,
     its frames arrive back to back.
   - **How to read it:** `GET /status` publishes the three numbers of the same deal next to the ceiling
     (`max_input_tokens`) and the wait (`hop_timeout_secs`): `max_concurrent_hop_bodies` (how many at
     once), `hop_bodies_in_flight` (how many **now** — `== max_concurrent_hop_bodies` is a node at its
     quota, which keeps answering `/status` and is therefore **full, not down**) and
     `hop_body_idle_timeout_secs`.
   - **Measured in `S91`** with the harness (`swarm.sh --workers 2 --vram 1.2 --scenario hop_body_quota`):
     the three numbers published, the live counter never above the ceiling, the `503` **parsed from the
     wire** with the quota full (also with a real 64 MiB body in flight), the `408` of a drip with the
     permit going back, and the RSS inside the budget of the reference node (`quota × ~80 MiB`). The
     scenario also prints — as a **note**, not as a check — the one client-side race measured over that
     edge: a sender that fires an `FIN` **glued** to the last byte of a **complete** and unread body can
     lose the whole answer (a clean EOF with zero bytes, not even an `RST`). No node of the mesh does
     that (its router sends with `reqwest`, which writes the body and waits), so it is a client's race,
     not the node's.
   - **Honest lane (`b′`):** a node declares `Expect: 100-continue` on the bodies it **sends** (the hop
     and the audit), so a receiver that would refuse them answers before the bytes travel: at the quota,
     the `503` leaves **without** a `100 Continue` and the sender never pushes its 70-80 MiB. It is an
     **optimisation and not a control** — the header is optional by construction (the router's own
     `reqwest` does not wait for the `100`, and a hostile sender simply omits it), which is exactly why
     what protects the RAM is the quota.

22. **A saturated node with no VRAM on every request announced Healthy — closed in `TASK-85`/`S92`**

   - **What was measured (`S84` / `S92`):** because `TASK-77`/`S84` classified CUDA `OOM` as an error of the
     *request* (not the *device*), a node whose hardware was alive but whose VRAM was completely exhausted on
     every request kept `verdict: "serving"` and announced `PeerStatus::Healthy` to the Musivum tracker. This
     created a classic distributed "gray failure" (black hole): the node stayed in Musivum's `GET /peers`,
     and routers kept sending traffic to it only to receive repeated `500 OOM` errors.
   - **What it is now (Option a: `Degraded`):** `gpu_health` tracks the consecutive OOM streak within a sliding
     time window. When `active_oom_streak >= oom_streak`, local `/status` publishes `gpu.verdict = "saturated"`,
     and periodic heartbeat `/announce` sends `PeerStatus::Degraded`. Musivum's `MatchmakerService::discover()`
     strictly filters by `PeerStatus::Healthy`, cleanly auto-excluding the saturated node from `GET /peers`
     without removing it from the orchestrator's telemetry (`GET /workers`).
   - **Configuration:** configured in `worker.toml` under `[gpu]` or `[worker]` (or env vars `TESSELLA_OOM_STREAK`
     and `TESSELLA_STREAK_WINDOW_SECS`):
     - `oom_streak`: consecutive OOMs required to declare saturation (default: `3`, `0` aborts at boot).
     - `streak_window_secs`: time window in seconds for the OOM streak (default: `30`, `0` aborts at boot).
   - **Two recovery paths without process restart:**
     1. **Active recovery (`request_served`):** when any request is successfully served (`note_request_served()`),
        the streak resets to 0 and `gpu.last_recovery` becomes `"request_served"`.
     2. **Passive recovery (`window_expired`):** when `streak_window_secs` elapses without new failures
        (`check_window_expiration()`), the streak resets to 0 and `gpu.last_recovery` becomes `"window_expired"`.
     In either case, the node returns to `verdict: "serving"`, the next heartbeat announces `PeerStatus::Healthy`,
     and the node is automatically restored to Musivum's `GET /peers`.
   - **Measured in `S92`** with the harness (`MAX_INPUT_TOKENS=8192 ./scripts/e2e/swarm.sh --workers 2 --vram 1.2 --scenario saturated_node`):
     3 consecutive OOMs trigger `verdict = "saturated"`, the node announces `Degraded` and is immediately
     omitted from `GET /peers` (surviving peers = 1), a small request succeeds and triggers active recovery
     (`last_recovery = "request_served"`), and the node is cleanly restored to `GET /peers` (peers = 2).

23. **Worker Observability: WAL Queue Depth & Batch Upload Metrics (`TASK-94`/`S100`)**

   - **Money path telemetry on the worker**: Tessella workers persist signed inference work proofs in a durable SQLite WAL table (`pending_work_proofs`). A periodic background task (`sweep_once`) sweeps and uploads these proofs in batches to Musivum's `/api/v1/work_done_batch`.
   - **Exposition endpoint**: Standard Prometheus exposition format is served on `GET /metrics` on the worker's HTTP port (e.g. `http://localhost:8080/metrics` or `http://localhost:10001/metrics`).
   - **Security policy**:
     - Loopback interfaces (`127.0.0.1` / `::1`) are permitted without authentication (supporting local scraping agents and operator inspection).
     - If `admin_token` is configured (`admin_token` in `worker.toml` or `TESSELLA_ADMIN_TOKEN`), any non-loopback scrape requires `Authorization: Bearer <admin_token>`. Non-loopback requests without a configured token return `403 Forbidden` to protect economic activity and session patterns from third-party probing.
   - **Exposed metrics**:
     - `tessella_wal_pending_proofs_depth` (gauge): Physical number of proofs currently pending in SQLite WAL (accurate across cold boots and process crashes).
     - `tessella_proofs_queued_total` (counter): Monotonic count of work proofs saved to WAL.
     - `tessella_proofs_settled_total` (counter): Monotonic count of work proofs successfully uploaded and settled with orchestrator.
     - `tessella_proof_batch_uploads_total{status="success"|"failure"}` (counter): Cumulative count of batch upload attempts by outcome.
     - `tessella_proof_batch_upload_duration_seconds` (histogram): Duration of HTTP batch upload requests to the orchestrator with 13 buckets (`0.005` to `30.0` seconds).
   - **Literal `curl` output example**:
     ```bash
     $ curl -s http://127.0.0.1:8080/metrics
     # HELP tessella_wal_pending_proofs_depth Number of pending work proofs in SQLite WAL.
     # TYPE tessella_wal_pending_proofs_depth gauge
     tessella_wal_pending_proofs_depth 0
     # HELP tessella_proofs_queued_total Total work proofs queued to WAL.
     # TYPE tessella_proofs_queued_total counter
     tessella_proofs_queued_total 12
     # HELP tessella_proofs_settled_total Total work proofs successfully settled with orchestrator.
     # TYPE tessella_proofs_settled_total counter
     tessella_proofs_settled_total 12
     # HELP tessella_proof_batch_uploads_total Total batch upload attempts to orchestrator by status.
     # TYPE tessella_proof_batch_uploads_total counter
     tessella_proof_batch_uploads_total{status="success"} 2
     tessella_proof_batch_uploads_total{status="failure"} 0
     # HELP tessella_proof_batch_upload_duration_seconds Latency of proof batch uploads to orchestrator in seconds.
     # TYPE tessella_proof_batch_upload_duration_seconds histogram
     tessella_proof_batch_upload_duration_seconds_bucket{le="0.005"} 0
     tessella_proof_batch_upload_duration_seconds_bucket{le="0.01"} 1
     tessella_proof_batch_upload_duration_seconds_bucket{le="0.025"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="0.05"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="0.1"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="0.25"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="0.5"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="1"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="2.5"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="5"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="10"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="15"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="30"} 2
     tessella_proof_batch_upload_duration_seconds_bucket{le="+Inf"} 2
     tessella_proof_batch_upload_duration_seconds_sum 0.024500
     tessella_proof_batch_upload_duration_seconds_count 2
     ```
