use crate::{InferenceBackend, ShardedWeights, TokenizerWrapper};
use candle_core::Device;
use std::path::Path;
use std::sync::Arc;
use tessella_core::error::DomainError;
use tessella_core::models::ModelConfig;
use tessella_core::telemetry::metrics::ExpertActivationRegistry;
use tracing::info;

/// Describes how to build the backend of one model family (`TASK-43`/S49: the factory receives the
/// node's expert-activation registry, because a MoE backend has to record its routing there and the
/// `static` that used to hold it made two engines in one process share the counters).
#[allow(clippy::type_complexity)]
pub struct BackendDescriptor {
    /// Returns true if this descriptor handles the given model configuration
    pub matches: fn(&ModelConfig) -> bool,
    /// Constructs the backend
    pub factory: fn(
        &Path,
        &ShardedWeights,
        &Device,
        Arc<TokenizerWrapper>,
        Arc<ExpertActivationRegistry>,
    ) -> Result<Box<dyn InferenceBackend>, DomainError>,
    /// EOS token IDs for this model family
    pub eos_token_ids: fn() -> Vec<u32>,
}

pub struct BackendRegistry {
    descriptors: Vec<BackendDescriptor>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            descriptors: vec![],
        }
    }

    pub fn register(&mut self, descriptor: BackendDescriptor) {
        self.descriptors.push(descriptor);
    }

    pub fn create(
        &self,
        config: &ModelConfig,
        model_dir: &Path,
        weights: &ShardedWeights,
        device: &Device,
        tokenizer: Arc<TokenizerWrapper>,
        activations: Arc<ExpertActivationRegistry>,
    ) -> Result<Box<dyn InferenceBackend>, DomainError> {
        let descriptor = self.find(config).ok_or_else(|| {
            let archs = config.architectures.join(", ");
            DomainError::Inference(format!(
                "[TESERROR-06] Unsupported model architecture '{}' for model '{}'. Supported architectures: Qwen2ForCausalLM, Qwen2MoeForCausalLM",
                archs, config.model_id
            ))
        })?;

        let backend = (descriptor.factory)(model_dir, weights, device, tokenizer, activations)?;
        let arch = config
            .architectures
            .first()
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        info!(
            "🔧 Selected backend: {} for architecture: {} (model: {})",
            "Qwen2Backend", // Ideally we'd get this from backend.name() or descriptor
            arch,
            config.model_id
        );
        Ok(backend)
    }

    pub fn eos_token_ids(&self, config: &ModelConfig) -> Vec<u32> {
        self.find(config)
            .map(|d| (d.eos_token_ids)())
            .unwrap_or_default() // safe fallback
    }

    fn find(&self, config: &ModelConfig) -> Option<&BackendDescriptor> {
        // Special case: force Qwen2Backend for local micro-moe testing
        if config.model_id.contains("tessella-moe-micro") {
            return self.descriptors.first();
        }
        self.descriptors.iter().find(|d| (d.matches)(config))
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}
