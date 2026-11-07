pub mod callback_guard;
pub mod canary;
pub mod config;
pub mod economy_cli;
pub mod engine;
pub mod heartbeat;
pub mod hf_loader;
pub mod http_pipeline_router;
pub mod inference_engine;
pub mod membership;
pub mod model_cache;
pub mod network_manager;
pub mod peer_discovery;
pub mod registration;
pub mod routing_table;
pub mod runtime;
pub mod scheduler;
pub mod service;
pub mod shutdown;
pub mod task_scheduler;
pub mod topology_cache;

pub use config::WorkerConfig;
pub use service::WorkerService;

pub fn load_tokenizer(
    path: &std::path::Path,
) -> anyhow::Result<std::sync::Arc<tessella_inference::TokenizerWrapper>> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
    Ok(std::sync::Arc::new(
        tessella_inference::TokenizerWrapper::from_file(path_str)?,
    ))
}

#[cfg(test)]
mod tests;
