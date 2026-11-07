#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initializes the logging system.
/// headless=true → stdout (compact, one line per event, no file/line)
/// headless=false → rolling file under logs/
pub fn init_logging(
    service_name: &str,
    headless: bool,
    default_level: &str,
) -> Result<tracing_appender::non_blocking::WorkerGuard, anyhow::Error> {
    // Suppress noisy third-party crates regardless of the caller's level
    let base_filter = format!("{default_level},tokenizers=error,hyper=warn,reqwest=warn,h2=warn");
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&base_filter));

    if headless {
        let (non_blocking, guard) = tracing_appender::non_blocking(std::io::stdout());
        tracing_subscriber::registry()
            .with(filter)
            .with(
                fmt::layer()
                    .compact() // single line, no blank lines between events
                    .with_writer(non_blocking)
                    .with_target(true) // keep module path
                    .with_file(false) // drop "at crates/..."
                    .with_line_number(false)
                    .with_thread_ids(false)
                    .with_thread_names(false),
            )
            .try_init()
            .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;
        Ok(guard)
    } else {
        let logs_dir = "logs";
        let mut fallback_to_stderr = false;
        if let Err(err) = std::fs::create_dir_all(logs_dir) {
            eprintln!("WARNING: could not create logs directory: {err}. Falling back to stderr.");
            fallback_to_stderr = true;
        }

        if fallback_to_stderr {
            let (non_blocking, guard) = tracing_appender::non_blocking(std::io::stderr());
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .compact()
                        .with_writer(non_blocking)
                        .with_target(true)
                        .with_file(false)
                        .with_line_number(false)
                        .with_thread_ids(false)
                        .with_thread_names(false),
                )
                .try_init()
                .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;
            Ok(guard)
        } else {
            let file_appender = RollingFileAppender::new(
                Rotation::DAILY,
                logs_dir,
                format!("{}.log", service_name),
            );
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .compact()
                        .with_writer(non_blocking)
                        .with_ansi(false)
                        .with_target(true)
                        .with_file(false)
                        .with_line_number(false)
                        .with_thread_ids(false)
                        .with_thread_names(false),
                )
                .try_init()
                .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;
            Ok(guard)
        }
    }
}
