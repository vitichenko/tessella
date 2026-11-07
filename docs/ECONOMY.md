# The Tessella Network Economy ⚖️

Tessella operates on a **Compute-for-Compute (C4C)** philosophy. Our goal is to provide decentralized, global AI inference without relying on fiat subscriptions or cryptocurrencies. 

Instead of money, the network runs on **Karma** and **Synchronous Presence**.

## Core Principles

### 1. The Entry Ticket: Synchronous Barter
To ask the network to process a prompt, **you must contribute to the network at that exact moment**. 
The Orchestrator (Musivum) issues a **session grant** only to an identity that presents itself with a
public key (`GET /api/v1/session_grant`, mandatory since `TASK-57`/S63), and the **Karma** of that
identity is what decides the queue tier (below). A deployment declared as `private`
(`TASK-49`/S55) refuses an identity that is not in `authorized_workers` with
`403 worker_not_authorized` instead of serving it at low priority.
*You only get to use the AI if you are part of the AI.*

> `[corregido en S73, 2026-09-23]` este apartado citaba un `403 Forbidden: No compute stake detected`
> que **no existe en el código** (la palabra `stake` no aparece en ningún `*.rs`): el mecanismo real es
> el `session_grant` + el tier de Karma, y el `403` de admisión lo da el modo `private`.

### 2. The Currency: the layers a node serves

Unlike traditional APIs that bill per token, Tessella pays for **compute**: every credit is authorised by a
`WorkTicket` the orchestrator issued for that session (`D-7`/`D-13`) and settles the **layers of the real
assignment** the node executed — dense or MoE alike, because the assignment is whole layers (`D-15`).

* **Mixture of Experts (MoE):** the expert routing happens inside the layer. The expert counters a worker
  reports in its heartbeat do feed a **Karma score** (`karma_ledger`, self-reported and best-effort — `D-5`,
  H-22; `matchmaker.rs` → `record_provider_activations`), but that score **has no reader today**: the queue
  tier is resolved from the **credit ledger** (`peer_credits`: `credits_earned - credits_spent`), which only
  the authorised `WorkProof`/`WorkTicket` path writes (`D-7`). The durable economic evidence is that signed
  proof in the node's SQLite WAL.
* **Dense Networks:** no gating at all; the same per-layer settlement.
* **Today the credits only go up.** `CreditService::record_consumption` exists (and would write
  `credits_spent`) but **no production path calls it**, so a balance is the sum of what a node has served and
  a client that never serves stays at `0` — which the tier function reads as *Leech*.

Regardless of the model architecture, all compute exchanges are recorded under the same Karma QoS system.

### 3. The Reputation System: Karma and QoS (Quality of Service)
The Orchestrator maintains a `Karma Score` for your PeerID, persisted in `karma_ledger` and restored at boot
(`TASK-30`):
* **Earning Karma (+):** When your GPU executes the layers of an authorised session for someone else in the network.
* **Losing priority (−):** a corroborated failure report (`TASK-57`) or a divergent audit (`TASK-53`, layer 3) applies the `slash_penalty` of the reputation table; proven fraud is a full reset.
* **Front Audit (`TASK-80`/S87 · `D-34`):** the first step of an entry node (layer 0) is auditable when marked by the orchestrator, re-executing tokens through replicas and verifying divergence against an operator canary control.

**Karma is not a bank balance, it is a Priority Queue:**
You can never run out of "money" and be blocked from using the network. Instead, your Karma dictates your routing priority when the network is congested:

* 🌟 **VIP Queue (Karma > 1000):** You contribute more than you consume. Your prompts are routed to the fastest workers instantly. Lowest Time-To-First-Token (TTFT).
* ⚖️ **Standard Queue (Karma > 0):** You consume roughly what you provide. Normal routing.
* 🐌 **Leech Queue (Karma ≤ 0):** You consume far more than you calculate. The Orchestrator places your requests in the lowest priority queue. You will experience high latency and possible timeouts if the network is busy.

Two switches change the tiers themselves: `deployment_mode = "local"` (`D-19`) grants **no tier to anybody**
(a LAN of trust, so the orders and the tiers stop depending on a score) and `economy_enabled = false` makes
every node VIP. Neither turns the **recording** off — the ledger keeps writing what each node serves, so a
node that later joins a `free` mesh keeps its balance.

## Enterprise & Private Clusters
For businesses that require strict data privacy, the Orchestrator (Musivum) can be self-hosted via an Enterprise License. When running in a private VPC, the `economy_enabled` toggle is set to `false`, disabling Karma checks and allowing unrestricted internal routing.

## 4. Where the economy can be read (`TASK-67`/S73 · D-26)

The economy is the **operator's** data, not the network's: `GET /ledger`, `GET /credits` and
`GET /reputation` require the orchestrator's `admin_token` — the same one `GET /ledger/snapshot` has
required since `TASK-34`/S40 — and so do the `tessella-worker economy balance | ledger | reputation`
commands, which read it from `MUSIVUM_ADMIN_TOKEN`:

```bash
export MUSIVUM_ADMIN_TOKEN="<the admin_token of the orchestrator you are asking>"
```

What stays public and why (the full frontier is in `musivum/README.md`): `GET /federation/status` (the
machine view of the federation), `GET /peers` (D-23: what a client needs to find the entrypoint of one
model), `GET /workers` (the live view of the mesh) and the weight **digests**
(`GET /weights/manifest[s]`, which a node uses to verify what it downloaded).
