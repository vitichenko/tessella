# 🏗️ Tessella Infrastructure (**legacy**)

> ⚠️ **These templates do not build.** They were written for `tessella-tracker`, a binary that stopped
> existing when the control plane moved to the **`musivum`** workspace (the rename touched the whole
> codebase; see `../changelog.md`). What is current today:
>
> * **Orchestrator image**: `musivum/Dockerfile`, built from the **monorepo root** (the workspace consumes
>   `../tessella` by path, so the build context has to include it).
> * **Orchestrator deployment to GCP**: `<monorepo>/scripts/deploy/deploy_gcp.sh` — compiles `musivum` in
>   release, uploads it with `config/musivum.toml` (mode 600: it carries the `admin_token`) and relaunches
>   it under `nohup`; `--clean` wipes the remote database.
> * **NAT / RunPod / proxy recipes** (ports, `TESSELLA_PUBLIC_*`, `ALLOW_PRIVATE_ADDRS`):
>   [`../README_DEPLOY.md`](../README_DEPLOY.md) and [`../docs/HOWTO.md`](../docs/HOWTO.md).

## What is in here, and why it cannot be used as-is

| File | Status |
| :--- | :--- |
| `Dockerfile.tracker` | **Broken**: it builds `--package tessella-tracker` and copies `/app/tessella-tracker`, a package and a binary that no longer exist. It also exposes the old ports (`8000`/`9000`) and assumes the `tessella` workspace as the build context, while the current orchestrator needs the monorepo root. Use `musivum/Dockerfile`. |
| `deploy_tracker.sh` | **Superseded**: its body was rewritten to compile and deploy **`musivum`** (binary + TOML, `nohup`, `--clean`), so it works, but `scripts/deploy/deploy_gcp.sh` is the maintained version. |
| `tessella-tracker.service` | **Broken**: the unit points at `/home/victor/projects/tessella-tracker/tessella-tracker` and passes `--external-address`, a flag that does not exist (the P2P listen address is `MUSIVUM_LISTEN_ADDR` or the TOML; the binary is `musivum --config config/musivum.toml`). Rewrite it before using it. |

Nothing in this directory is referenced by the monorepo docs or by `scripts/`: it is kept only for
history. If it is not going to be fixed, delete it — a template that cannot build is debt.

