//! The **manual** two-worker swarm E2E (H-24 · `TASK-12` · S17).
//!
//! `#[ignore]`d on purpose, and the reason is environmental, not laziness:
//!
//! * it needs a **GPU**: Tessella refuses any other device (D-6 = GPU-only,
//!   `tessella/crates/worker/src/config.rs::validated_device`), so it cannot run in CI;
//! * it needs the two binaries **already built** (they live in two separate workspaces);
//! * it starts daemons and binds ports, which only the closing-gate harness may do: **for the
//!   closing gate use `scripts/e2e/run_tests.sh`** (one worker,
//!   deterministic, ports and logs handled); this test is the *two-worker* variant, which the
//!   gate deliberately does not cover because it needs two model shards on one GPU.
//!
//! Runbook (from `tessella/`):
//!
//! ```text
//! (cd ../musivum && cargo build -p musivum --bin musivum)
//! cargo build -p tessella-node --bin tessella-worker   # `SOLID-89`: the binary's package is `tessella-node`
//! cargo test -p tessella-worker --test e2e_binary_swarm -- --ignored --nocapture
//! ```
//!
//! What it asserts (before: "`/workers` returns at least two entries", which proves neither that
//! the workers are usable nor that they survive):
//!
//! 1. both workers reach a healthy state (`Ready`/`Healthy`, the same predicate
//!    `scripts/e2e/run_tests.sh` uses) and `GET /peers?model_id=…` lists **both** of them;
//! 2. they are **still** listed after `E2E_SWARM_SURVIVAL_SECS` (30 by default) and neither child
//!    process died — this is what actually exercises H-01/`TASK-01`: a *connected* peer used to be
//!    evicted by the 45 s liveness timeout.
//!
//! Fixed in this rewrite (each one was a real defect of the old version):
//!
//! * **the pipes are drained**: each child writes to a file, so a full pipe buffer can never block
//!   it (`Stdio::piped()` with nobody reading was a guaranteed hang for a chatty worker);
//! * **ports are free ones** picked at run time instead of the fixed `8080` / `10001` / `10002`;
//! * **`device = "cuda"`**, not `"cpu"`: a `cpu` config aborts the worker at startup (D-6) and the
//!   failure would look like a networking problem;
//! * **nothing is written inside the repositories**: config, DB, keypair and logs live in
//!   `std::env::temp_dir()/tessella-e2e-swarm-<pid>` (the old version left `target/e2e_w{1,2}`
//!   with a `node_*.key` and a `worker.db` inside `tessella/`, which S30 had to clean up by hand);
//! * the two children are killed (and awaited) on **every** exit path, and the failure path prints
//!   the logs;
//! * **the claim decides the topology** (`E2E_SWARM_VRAM_GB`, default `0.6`): the layer budget of a
//!   claim is `claim × 80 % ÷ max(per-layer MB, 40)` = 40 layers for this 24-layer model, and
//!   `scripts/e2e/plan_split.py` prints the same arithmetic. With `0.6` each worker asks for 12
//!   layers and the model is split `[0, 12)` + `[12, 24)`; with `2.0` the first worker takes the
//!   **whole** model and the second one joins as **replica group 1** — which is `TASK-51`/S57: it
//!   used to be refused with `{"error":"model_at_capacity"}` before any assignment was computed
//!   (`auto_rebalance` defaults to `false`, and that guard is gone), and the first real execution of
//!   this test (S38) found exactly that refusal in 3 seconds. Both topologies are asserted below, so
//!   the test is its own evidence for the two cases.

use serde_json::Value;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};

const MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
/// A worker's HTTP API listens on its P2P port plus this offset
/// (`tessella_core::constants::network::API_PORT_OFFSET`).
const API_PORT_OFFSET: u16 = 1000;

/// An ephemeral port the OS has just proven free (there is an unavoidable race between closing
/// this listener and the daemon binding it, which is why the old *fixed* ports were worse: they
/// collide deterministically with anything already listening).
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("the OS must have a free port")
        .local_addr()
        .expect("a bound listener has an address")
        .port()
}

/// A P2P port whose API twin (`p2p + 1000`) is free too, so a worker can use both.
fn free_worker_port() -> u16 {
    for _ in 0..100 {
        let p2p = free_port();
        if p2p
            .checked_add(API_PORT_OFFSET)
            .is_some_and(|api| TcpListener::bind(("127.0.0.1", api)).is_ok())
        {
            return p2p;
        }
    }
    panic!("no free (p2p, p2p+{API_PORT_OFFSET}) port pair available");
}

/// `musivum/target/debug/musivum` or `tessella/target/debug/tessella-worker`.
fn binary(root: &Path, workspace: &str, name: &str) -> PathBuf {
    let mut path = root.join(workspace).join("target").join("debug").join(name);
    if cfg!(windows) {
        path.set_extension("exe");
    }
    path
}

/// `tessella/` — this file lives in `tessella/crates/worker/tests/`.
fn tessella_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/worker sits two levels below the workspace root")
        .to_path_buf()
}

/// Sends both streams of a child to **one** file, appending: no pipe to drain, no truncation
/// between stdout and stderr, and the log survives to be printed if something fails.
fn log_sink(dir: &Path, name: &str) -> anyhow::Result<(Stdio, Stdio, PathBuf)> {
    let path = dir.join(name);
    let open = || {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
    };
    Ok((Stdio::from(open()?), Stdio::from(open()?), path))
}

/// The VRAM claim both workers declare, in GB (`E2E_SWARM_VRAM_GB`).
///
/// It is **not** a claim about this machine: it is the lever that decides the topology, because the
/// orchestrator derives the layer budget from it (`tessella_core::assignment::compute_layer_budget`:
/// `vram_mb × 80 % ÷ max(per-layer MB, 40)`). `0.6` (the default) gives each worker 12 of the 24
/// layers, so the model is split `[0, 12)` + `[12, 24)`; `2.0` gives the first worker the whole model
/// and the second one joins as **replica group 1** (`TASK-51`/S57). The arithmetic is the one
/// `scripts/e2e/plan_split.py` prints.
fn claim_gb() -> f64 {
    std::env::var("E2E_SWARM_VRAM_GB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.6)
}

fn write_worker_config(dir: &Path, listen_port: u16, orchestrator_port: u16) -> anyhow::Result<()> {
    let claim = claim_gb();
    let config = format!(
        r#"musivum_url = "http://127.0.0.1:{orchestrator_port}"
default_model = "{MODEL}"
listen_port = {listen_port}
vram_gb = {claim}
device = "cuda"
db_path = "worker.db"
max_new_tokens = 32
heartbeat_interval_secs = 10
"#
    );
    std::fs::write(dir.join("worker_config.toml"), config)?;
    Ok(())
}

/// The `(layer_range, replica_group_id)` a `/peers` entry publishes.
///
/// The role travels **internally tagged** (`{"type":"Worker","layer_ranges":[[0,12]],…}`), which is
/// the contract `tessella_worker::peer_discovery` already parses.
fn role_of(peer: &Value) -> ((u64, u64), u64) {
    let role = &peer["role"];
    let range = role["layer_ranges"]
        .as_array()
        .and_then(|ranges| ranges.first())
        .and_then(|range| range.as_array())
        .map(|range| (range[0].as_u64().unwrap(), range[1].as_u64().unwrap()))
        .unwrap_or_else(|| panic!("a Worker role publishes its layer range: {peer}"));
    let group = role["replica_group_id"].as_u64().unwrap_or(0);
    (range, group)
}

/// The last lines of a log file, for the failure path.
fn tail(path: &Path) -> String {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let start = lines.len().saturating_sub(40);
            lines[start..].join("\n")
        }
        Err(e) => format!("<cannot read {}: {e}>", path.display()),
    }
}

/// The statuses `scripts/e2e/run_tests.sh` accepts as "the worker is up" (`/workers` reports `Ready`
/// once the shard is loaded; the mesh view may still call it `Healthy`).
fn is_up(status: &Value) -> bool {
    matches!(status.as_str(), Some("Healthy") | Some("Ready"))
}

/// How many workers the orchestrator currently reports as up, or `None` while it is not answering.
async fn up_workers(client: &reqwest::Client, base: &str) -> Option<usize> {
    let workers: Value = client
        .get(format!("{base}/workers"))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    Some(
        workers
            .as_array()?
            .iter()
            .filter(|w| is_up(&w["status"]))
            .count(),
    )
}

/// `GET /peers?model_id=…`: the mesh view a **client** uses to find workers.
async fn peers_of_model(client: &reqwest::Client, base: &str) -> anyhow::Result<Vec<Value>> {
    Ok(client
        .get(format!("{base}/peers?model_id={MODEL}"))
        .timeout(Duration::from_secs(5))
        .send()
        .await?
        .json()
        .await?)
}

/// Kills and reaps every child, so neither a failing nor a passing test can leave a daemon
/// listening on a port.
async fn stop(children: &mut [Child]) {
    for child in children.iter_mut() {
        let _ = child.start_kill();
    }
    for child in children.iter_mut() {
        let _ = child.wait().await;
    }
}

/// Old name: `test_e2e_binary_swarm`. See the module docs for the runbook and the reasons it is
/// `#[ignore]`d.
#[tokio::test]
#[ignore = "needs a GPU (D-6), pre-built binaries and free ports: see the module docs"]
async fn two_workers_register_stay_healthy_and_are_not_evicted() -> anyhow::Result<()> {
    let root = tessella_dir()
        .parent()
        .expect("the monorepo root is above tessella/")
        .to_path_buf();
    let musivum_bin = binary(&root, "musivum", "musivum");
    let worker_bin = binary(&root, "tessella", "tessella-worker");
    assert!(
        musivum_bin.exists(),
        "{} is missing: run `cargo build -p musivum --bin musivum` in {}/musivum (or use \
         scripts/e2e/run_tests.sh, which builds and runs the supported gate)",
        musivum_bin.display(),
        root.display()
    );
    assert!(
        worker_bin.exists(),
        "{} is missing: run `cargo build -p tessella-node --bin tessella-worker` in {}/tessella",
        worker_bin.display(),
        root.display()
    );

    // Everything this test creates lives in a temporary directory: the repositories are never
    // written to (S30 had to clean `tessella/target/e2e_w{1,2}` by hand after the old version).
    let run_dir = std::env::temp_dir().join(format!("tessella-e2e-swarm-{}", std::process::id()));
    if run_dir.exists() {
        std::fs::remove_dir_all(&run_dir)?;
    }
    std::fs::create_dir_all(&run_dir)?;

    let api_port = free_port();
    let base = format!("http://127.0.0.1:{api_port}");
    let (out, err, musivum_log) = log_sink(&run_dir, "musivum.log")?;
    let mut children = vec![Command::new(&musivum_bin)
        .current_dir(&run_dir)
        .args([
            "--api-port",
            &api_port.to_string(),
            "--p2p-port",
            &free_port().to_string(),
        ])
        .env("ALLOW_PRIVATE_ADDRS", "true")
        .env("MUSIVUM_DB_PATH", run_dir.join("musivum.db"))
        .stdout(out)
        .stderr(err)
        .kill_on_drop(true)
        .spawn()?];

    let mut worker_logs = Vec::new();
    for index in 0..2 {
        let dir = run_dir.join(format!("worker{index}"));
        std::fs::create_dir_all(&dir)?;
        write_worker_config(&dir, free_worker_port(), api_port)?;
        let config = dir.join("worker_config.toml");
        let (out, err, log) = log_sink(&run_dir, &format!("worker{index}.log"))?;
        worker_logs.push(log);
        children.push(
            Command::new(&worker_bin)
                .current_dir(&dir)
                .arg("-c")
                .arg(&config)
                .stdout(out)
                .stderr(err)
                .kill_on_drop(true)
                .spawn()?,
        );
    }

    let report = |what: &str| {
        anyhow::anyhow!(
            "{what}\n--- musivum ---\n{}\n--- worker0 ---\n{}\n--- worker1 ---\n{}",
            tail(&musivum_log),
            tail(&worker_logs[0]),
            tail(&worker_logs[1])
        )
    };

    // 1. Both workers up. The deadline is generous because each one loads a shard of the model.
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut up = 0;
    while Instant::now() < deadline {
        if let Some(count) = up_workers(&client, &base).await {
            up = count;
            if count >= 2 {
                break;
            }
        }
        // Do not wait five minutes for a worker that has already died.
        for (index, child) in children.iter_mut().enumerate().skip(1) {
            if let Some(status) = child.try_wait()? {
                stop(&mut children).await;
                return Err(report(&format!(
                    "worker {} exited with {status} before registering",
                    index - 1
                )));
            }
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
    if up < 2 {
        stop(&mut children).await;
        return Err(report(&format!(
            "only {up} worker(s) became healthy within 300 s"
        )));
    }

    // 2. The mesh must list both of them: this is the state that decides which layers they get.
    let peers = peers_of_model(&client, &base).await?;
    if peers.len() < 2 {
        stop(&mut children).await;
        return Err(report(&format!(
            "GET /peers?model_id={MODEL} returned {} node(s) instead of 2: {peers:?}",
            peers.len()
        )));
    }
    for peer in &peers {
        if !is_up(&peer["status"]) {
            stop(&mut children).await;
            return Err(report(&format!(
                "peer {} is '{}', not usable",
                peer["peer_id"], peer["status"]
            )));
        }
    }

    // 2bis. The topology is what the claim decides — and `TASK-51`/S57 fixed the half that used to
    // fail: a node that adds no new layer joins as the first member of the next replica group instead
    // of being refused with `model_at_capacity` (the first execution of this test, S38, hit that
    // refusal in 3 seconds).
    let topology: Vec<((u64, u64), u64)> = peers.iter().map(role_of).collect();
    println!(
        "topology with E2E_SWARM_VRAM_GB={}: {topology:?}",
        claim_gb()
    );
    if claim_gb() >= 1.0 {
        // The whole 24-layer model fits in one claim, so the second node is a replica of the first.
        assert!(
            topology.iter().all(|(range, _)| *range == (0, 24)),
            "each node hosts the whole model: {topology:?}"
        );
        assert!(
            topology.iter().any(|(_, group)| *group == 1),
            "the second node must open the next replica group (TASK-51/S57): {topology:?}"
        );
    } else {
        assert!(
            topology.iter().all(|(_, group)| *group == 0),
            "nobody is a replica when the model does not fit in one claim: {topology:?}"
        );
        assert_ne!(
            topology[0].0, topology[1].0,
            "the two nodes must split the model: {topology:?}"
        );
    }

    // 3. H-01/TASK-01: a peer that stays connected is not evicted by the liveness timeout.
    let survival: u64 = std::env::var("E2E_SWARM_SURVIVAL_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30);
    tokio::time::sleep(Duration::from_secs(survival)).await;

    for (index, child) in children.iter_mut().enumerate().skip(1) {
        if let Some(status) = child.try_wait()? {
            stop(&mut children).await;
            return Err(report(&format!(
                "worker {} exited with {status} during the {survival}s survival window",
                index - 1
            )));
        }
    }
    let survivors = peers_of_model(&client, &base)
        .await?
        .into_iter()
        .filter(|peer| is_up(&peer["status"]))
        .count();
    if survivors < 2 {
        stop(&mut children).await;
        return Err(report(&format!(
            "only {survivors} healthy peer(s) left after {survival}s: a connected peer was \
             evicted (H-01)"
        )));
    }

    stop(&mut children).await;
    let _ = std::fs::remove_dir_all(&run_dir);
    println!("✅ two workers registered, stayed healthy for {survival}s and were not evicted");
    Ok(())
}
