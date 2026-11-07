# Tessella & Musivum Deployment Guide (RunPod / Multi-Node)

This document provides instructions for deploying Musivum (Orchestrator) and Tessella Workers across network boundaries, specifically optimized for environments like RunPod or GCP where NAT traversal and external IP advertisement are required.

> **Before you deploy**
>
> * **Deploy both sides together.** Since S63 the peer signature is mandatory in the registration DTOs, so a
>   worker of a different version cannot register against the orchestrator (`D-3`).
> * **Set an `admin_token`** in `musivum.toml` (`openssl rand -hex 32`) if you run a **replica**, or if you
>   want the operator's economy endpoints at all: `GET /ledger`, `/credits` and `/reputation` answer `503`
>   without it (`D-26`/S73), and the replica's ledger pull (`/ledger/snapshot`) has needed it since S40.
>   `admin_token` is read **only** from the TOML; the clients use `MUSIVUM_ADMIN_TOKEN`.
> * **GPU is mandatory** (`D-6`): `device = "cpu"` aborts the worker. The image is built with CUDA support,
>   so give the container `--gpus all`.
> * The builds below assume the **monorepo root** as the Docker context for Musivum (its workspace consumes
>   `../tessella` by path). `tessella/infra/` holds **legacy** templates that do not build — see
>   [`infra/README.md`](infra/README.md).

## 1. Deploying Musivum (Orchestrator)

The Orchestrator should be deployed on a stable, publicly accessible IP.

### Docker Run Command

```bash
docker run -d \
  --name musivum \
  -p 8080:8080 \
  -p 9080:9080 \
  -e MUSIVUM_EXTERNAL_IP="<YOUR_PUBLIC_IP>" \
  -e MUSIVUM_EXTERNAL_P2P_PORT="<EXTERNAL_P2P_PORT>" \
  -e MUSIVUM_LISTEN_ADDR="/ip4/0.0.0.0/tcp/9080" \
  -e MUSIVUM_ALLOW_PRIVATE_ADDRS="true" \
  -v /var/lib/musivum:/app/data \
  musivum:latest
```

*   **`MUSIVUM_EXTERNAL_IP`**: The public IP of your RunPod instance.
*   **`MUSIVUM_EXTERNAL_P2P_PORT`**: The externally visible P2P port when it differs from the internal one (NAT / RunPod remap). Advertised via `/p2p_id` so workers dial the right port.
*   **`MUSIVUM_LISTEN_ADDR`**: The internal interface to bind the P2P swarm.
*   **`MUSIVUM_ALLOW_PRIVATE_ADDRS`**: Set to `true` for Docker/RunPod environments where workers use internal IPs.
*   **Ports**: 8080 (HTTP API), 9080 (P2P Data Plane).
*   **Optional overrides**: `MUSIVUM_API_PORT`, `MUSIVUM_P2P_PORT` to change the default ports.

## 2. Deploying Tessella Workers

Workers need GPU access and must connect back to the Orchestrator.

### Docker Run Command (NVIDIA Container Toolkit required)

```bash
docker run -d \
  --name tessella-worker \
  --gpus all \
  -p 8080:8080 \
  -p 9080:9080 \
  -e MUSIVUM_URL="http://<MUSIVUM_PUBLIC_IP>:8080" \
  -e TESSELLA_EXTERNAL_IP="<WORKER_PUBLIC_IP>" \
  -e TESSELLA_PUBLIC_API_PORT="<EXTERNAL_API_PORT>" \
  -e TESSELLA_PUBLIC_P2P_PORT="<EXTERNAL_P2P_PORT>" \
  -e TESSELLA_PUBLIC_API_URL="http://<WORKER_PUBLIC_IP>:<EXTERNAL_API_PORT>" \
  -e VRAM_LIMIT="24.0" \
  -e HF_TOKEN="<YOUR_HF_TOKEN>" \
  tessella-worker:latest
```

*   **`MUSIVUM_URL`**: The HTTP API address of the Musivum instance.
*   **`TESSELLA_EXTERNAL_IP`**: The public IP of the worker instance (required for P2P routing).
*   **`TESSELLA_PUBLIC_API_PORT`**: The externally visible API port (must match your RunPod port mapping).
*   **`TESSELLA_PUBLIC_P2P_PORT`**: The externally visible P2P port (must match your RunPod port mapping).
*   **`TESSELLA_PUBLIC_API_URL`** *(recommended)*: The full public URL announced to peers, with scheme/host/port. Takes precedence over `TESSELLA_EXTERNAL_IP` + `TESSELLA_PUBLIC_API_PORT`. **Required** when the worker is exposed behind a TLS-terminating proxy (e.g. `https://<pod-id>-8080.proxy.runpod.net`). *Note: the Musivum orchestrator must configure `allowed_hostname_suffixes` (e.g. `["proxy.runpod.net"]`, **`D-8`**) to admit registrations with proxy hostnames.*
*   **`VRAM_LIMIT`**: Amount of VRAM (in GB) to allocate for model shards.

## 3. Network Configuration

Ensure the following ports are open on your firewalls / security groups:

| Component | Port | Protocol | Description |
| :--- | :--- | :--- | :--- |
| Musivum | 8080 | TCP | Control Plane (HTTP API) |
| Musivum | 9080 | TCP | Data Plane (P2P Swarm) |
| Worker | 8080 | TCP | Worker API |
| Worker | 9080 | TCP | Worker P2P |

## 4. Building Images

If you are building from source, run these from the **repository root**:

```bash
# Build Musivum. Context MUST be the repository root so the sibling
# ./tessella workspace (consumed via path dependencies) is available.
docker build -f musivum/Dockerfile -t musivum:latest .

# Build the Worker (context is the tessella workspace).
docker build -f tessella/Dockerfile -t tessella-worker:latest ./tessella
```

## 5. RunPod-Specific Notes

RunPod publishes container ports through NAT, so **the externally visible port is usually not the container-internal one** (e.g. you expose `8080` inside the container, but RunPod assigns `54325` externally). Whatever the orchestrator and the other workers observe must be the **external** value.

### Mode A — Direct TCP endpoints (recommended for the data plane)

Expose the container ports as **TCP ports** in the RunPod pod template (`8080`, `9080`) and note the external ports RunPod assigns:

```bash
docker run -d \
  --name tessella-worker \
  --gpus all \
  -e MUSIVUM_URL="http://<MUSIVUM_PUBLIC_IP>:8080" \
  -e TESSELLA_EXTERNAL_IP="<WORKER_PUBLIC_IP>" \
  -e TESSELLA_PUBLIC_API_PORT="54325" \
  -e TESSELLA_PUBLIC_P2P_PORT="54326" \
  -e HF_TOKEN="<YOUR_HF_TOKEN>" \
  tessella-worker:latest
```

The worker will announce `http://<WORKER_PUBLIC_IP>:54325` and `/ip4/<WORKER_PUBLIC_IP>/tcp/54326`.

### Mode B — HTTP proxy (TLS / custom domain)

If RunPod gives you an HTTPS endpoint such as `https://<pod-id>-8080.proxy.runpod.net`, there is no `ip:port` that represents it. Use the full-URL override instead (ensure Musivum has `allowed_hostname_suffixes = ["proxy.runpod.net"]` or `MUSIVUM_ALLOWED_HOSTNAME_SUFFIXES="proxy.runpod.net"`, **`D-8`**):

```bash
docker run -d \
  --name tessella-worker \
  --gpus all \
  -e MUSIVUM_URL="https://<musivum-domain>" \
  -e TESSELLA_PUBLIC_API_URL="https://<pod-id>-8080.proxy.runpod.net" \
  -e HF_TOKEN="<YOUR_HF_TOKEN>" \
  tessella-worker:latest
```

`TESSELLA_PUBLIC_API_URL` takes precedence over the `ip`/`port` pair for every HTTP announcement (registration, announce and heartbeat).

### Checklist

1. **Expose the ports** `8080` (HTTP API) and `9080` (P2P) in the RunPod pod template.
2. **Set `TESSELLA_EXTERNAL_IP`** to the pod's public IP (direct mode) **or** `TESSELLA_PUBLIC_API_URL` to the proxy URL (proxy mode).
3. **Set `TESSELLA_PUBLIC_API_PORT` / `TESSELLA_PUBLIC_P2P_PORT`** to the RunPod-assigned external ports when they differ from the internal ones.
4. **Set `MUSIVUM_ALLOW_PRIVATE_ADDRS=true`** only if workers communicate over the RunPod private network. Do **not** enable it on public deployments (it disables SSRF protection).

---
**Note:** For libp2p NAT traversal to work correctly on RunPod, ensure you map the ports in the RunPod console to match the container ports used in the `docker run` command, and set the corresponding `TESSELLA_PUBLIC_P2P_PORT`.
