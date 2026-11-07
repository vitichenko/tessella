use crate::InferenceMode;
use crate::shard_loader::ShardedWeights;
pub mod registry;
pub use registry::{BackendDescriptor, BackendRegistry};
use tessella_core::error::DomainError;

pub struct BackendConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub eos_token_ids: Vec<u32>,
    pub supports_system_prompt: bool,
}

/// The dtype a session casts its activations to on `device` — F16 on CUDA, F32 elsewhere
/// (`TASK-47`/S53). This is **the** definition: the backends use it to pick their `DType` and the
/// engine uses [`activation_dtype_bytes`] to compute how many tokens of hidden state fit in one hop,
/// so the ceiling cannot drift away from the dtype the payload is actually serialized with.
pub fn activation_dtype(device: &candle_core::Device) -> candle_core::DType {
    if device.is_cuda() {
        candle_core::DType::F16
    } else {
        candle_core::DType::F32
    }
}

/// Element size in bytes of [`activation_dtype`] on `device` (2 for F16, 4 for F32).
pub fn activation_dtype_bytes(device: &candle_core::Device) -> usize {
    activation_dtype(device).size_in_bytes()
}

pub struct BackendCapabilities {
    pub supports_kv_cache: bool,
    pub supports_chat_template: bool,
    pub supports_moe_routing: bool,
    pub supports_sliding_window: bool,
    pub supports_quantized_weights: bool,
    pub default_stop_token_ids: Vec<u32>,
    pub default_temperature: Option<f64>,
}

#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync {
    fn model_id(&self) -> &str;
    fn config(&self) -> &BackendConfig;
    fn capabilities(&self) -> &BackendCapabilities;

    /// Creates a new inference session for this backend.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if the session cannot be initialized (e.g. OOM or config error).
    fn begin_session(&self) -> Result<Box<dyn InferenceSession>, DomainError>;

    /// Applies the appropriate prompt template for this model/family.
    ///
    /// - Chat: applies ChatML or the model's native template.
    /// - Fim: applies the model's native FIM template.
    ///
    /// Each backend implements its own format.
    /// Default: pass-through without modification (useful for stub backends).
    async fn apply_prompt_template(&self, _mode: &InferenceMode, raw_prompt: &str) -> String {
        // Default: pass-through. Stubs (KimiMoe, etc.) inherit this.
        raw_prompt.to_string()
    }
}

pub trait InferenceSession: Send + Sync {
    /// Executes a forward pass through the model.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Inference` if the computation fails.
    fn forward(&mut self, tokens: &[u32]) -> Result<candle_core::Tensor, DomainError>;

    /// Executes a partial forward pass through specific layers.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Inference` if the computation fails.
    fn apply_layers(
        &mut self,
        x: candle_core::Tensor,
        range: (usize, usize),
        seq_start_pos: usize,
    ) -> Result<candle_core::Tensor, DomainError>;

    fn reset(&mut self);

    /// Synchronizes the internal position offset (for RoPE/KV-cache)
    fn set_pos_offset(&mut self, pos: usize);

    /// Returns the current position offset of this session
    fn get_pos_offset(&self) -> usize;
}

/// Special tokens of a code MoE model.
/// Each backend must fill all relevant fields
/// from the real tokenizer — without numerical hardcodes.
#[derive(Debug, Clone)]
pub struct ModelTokens {
    /// IDs that signal end of generation (required, minimum one).
    pub eos_ids: Vec<u32>,
    /// Turn start token (e.g., <|im_start|> in Qwen).
    pub bos_turn: Option<u32>,
    /// Turn end token (e.g., <|im_end|> in Qwen).
    pub eos_turn: Option<u32>,
    /// FIM prefix (e.g., <|fim_prefix|> in Qwen2.5-Coder).
    pub fim_prefix: Option<u32>,
    /// FIM suffix.
    pub fim_suffix: Option<u32>,
    /// FIM middle fragment start token.
    pub fim_middle: Option<u32>,
}

impl ModelTokens {
    /// Helper to resolve a token by name from the tokenizer.
    /// Returns DomainError::Config if not found — never numerical fallback.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Config` if the token name is not found in the tokenizer.
    pub fn resolve(tokenizer: &tokenizers::Tokenizer, name: &str) -> Result<u32, DomainError> {
        tokenizer.token_to_id(name).ok_or_else(|| {
            DomainError::Config(format!(
                "Token '{}' not found in tokenizer. \
                   Ensure the tokenizer.json matches the model family.",
                name
            ))
        })
    }

    /// Same as resolve but returns None if it doesn't exist (for optional tokens).
    #[must_use]
    pub fn resolve_optional(tokenizer: &tokenizers::Tokenizer, name: &str) -> Option<u32> {
        tokenizer.token_to_id(name)
    }
}

pub fn create_qwen_backend(
    model_dir: &std::path::Path,
    sharded_weights: &ShardedWeights,
    device: &candle_core::Device,
    tokenizer: std::sync::Arc<crate::TokenizerWrapper>,
    activations: std::sync::Arc<tessella_core::telemetry::metrics::ExpertActivationRegistry>,
) -> Result<Box<dyn InferenceBackend>, DomainError> {
    Ok(Box::new(qwen2::Backend::new_with_candle_device(
        "qwen",
        model_dir,
        sharded_weights,
        device,
        tokenizer,
        activations,
    )?))
}

pub mod qwen2;
pub mod tokens;

pub use tessella_protocol::types::GenerationMetrics;

#[cfg(test)]
mod tests {
    use super::{activation_dtype, activation_dtype_bytes};
    use candle_core::DType;

    /// `TASK-47`/S53 · The activations of a session are F16 on CUDA and F32 elsewhere, which is what
    /// decides the **transport** ceiling of a hop: 2 bytes per element carry twice as many tokens as
    /// 4 for the same `MAX_HIDDEN_STATE_BYTES`. The ceiling used to be computed with a hardcoded 2,
    /// so a `metal` node (F32) would have accepted prompts that do not fit in one payload.
    #[test]
    fn activation_dtype_is_f32_off_cuda_and_two_bytes_per_element_per_definition() {
        let cpu = candle_core::Device::Cpu;
        assert_eq!(activation_dtype(&cpu), DType::F32);
        assert_eq!(activation_dtype_bytes(&cpu), 4);
        assert_eq!(
            activation_dtype_bytes(&cpu),
            DType::F32.size_in_bytes(),
            "the byte count is derived from the dtype, never a second literal"
        );
    }
}
