# tessella-telemetry

> Structured logging, log rolling, and tracing initialization for the Tessella workspace.

## Responsibility

`tessella-telemetry` initializes the runtime observability layer across Tessella binaries (`tessella-worker` and `tessella-cli`):
- Configures `tracing-subscriber` with `EnvFilter` (honoring `RUST_LOG`).
- Configures ANSI-colored terminal output for developer interactivity.
- Provides rolling non-blocking file appenders via `tracing-appender` for persistent daemon audit logs.

It is a lightweight leaf crate with zero dependencies on compute, network, or storage engines.

## Modules

| File | Content |
| :--- | :--- |
| `src/lib.rs` | Public re-exports for telemetry setup (`init_logging`). |
| `src/logging.rs` | Subscriber registry initialization, log formatting, and file appender configuration. |

## Usage

```rust
use tessella_telemetry::init_logging;

fn main() -> anyhow::Result<()> {
    let _guard = init_logging("worker.log")?;
    tracing::info!("Telemetry initialized");
    Ok(())
}
```
