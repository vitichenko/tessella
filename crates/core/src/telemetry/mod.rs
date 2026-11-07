use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initializes the tracing subscriber for server-side binaries (tracker, worker).
/// - Default: compact single-line format, no file/line numbers
/// - TESSELLA_LOG_FORMAT=json  → structured JSON (for log aggregators)
/// - TESSELLA_LOG_FORMAT=pretty → verbose multi-line (for local debugging only)
/// - RUST_LOG overrides the default filter
pub fn init_tracing() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "musivum=info,tessella=info,tokenizers=error,hyper=warn,reqwest=warn,h2=warn",
        )
    });

    let format = std::env::var("TESSELLA_LOG_FORMAT").unwrap_or_default();

    match format.as_str() {
        "json" => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().json())
                .try_init()
                .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;
        }
        "pretty" => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().pretty())
                .try_init()
                .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;
        }
        _ => {
            // Default: compact, one line per event, no file/line noise
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .compact()
                        .with_target(true)
                        .with_file(false)
                        .with_line_number(false)
                        .with_thread_ids(false)
                        .with_thread_names(false),
                )
                .try_init()
                .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;
        }
    }
    Ok(())
}
pub mod gpu_health;
pub mod metrics;
pub mod worker_metrics;
pub use worker_metrics::WorkerMetrics;
