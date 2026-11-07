#[cfg(test)]
use std::str::FromStr;
pub mod attention;
pub mod backends;
pub mod conversion;
pub mod error;
pub mod generation;
pub mod hidden_state;
pub mod layer_factory;
pub mod models;
pub mod moe;
pub mod sampling;
pub mod shard_loader;

pub use backends::{BackendConfig, InferenceBackend};
pub use conversion::*;
pub use error::InferenceError;
pub use generation::{Config as GenerationConfig, Loop as GenerationLoop};
pub use hidden_state::{Payload as HiddenStatePayload, PayloadTensorExt};
pub use layer_factory::LayerFactory;
pub use models::*;
pub use moe::*;
pub use shard_loader::{ShardedWeights, load_sharded_weights};

use candle_core::DType;
use candle_nn::VarBuilder;
use tessella_core::error::DomainError;
use tessella_core::models::ComputeDevice;
use tokenizers::Tokenizer;

/// Opaque context for model initialization.
pub struct InferenceContext {
    pub(crate) vb: VarBuilder<'static>,
}

impl InferenceContext {
    /// Creates an empty context with the specified device.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Inference` if the device mapping fails.
    pub fn empty(device: &ComputeDevice) -> Result<Self, DomainError> {
        let candle_device = Converter::map_device(device)?;
        let dtype = if candle_device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        Ok(Self {
            vb: VarBuilder::zeros(dtype, &candle_device),
        })
    }

    /// Creates a context from a weight provider.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Storage` if weights cannot be loaded or deserialized.
    pub fn from_provider(
        provider: &dyn tessella_core::traits::WeightProvider,
        name: &str,
        device: &ComputeDevice,
    ) -> Result<Self, DomainError> {
        let candle_device = Converter::map_device(device)?;
        let data = provider.get_weights(name)?;
        let tensors = candle_core::safetensors::load_buffer(&data, &candle_device)
            .map_err(|e| DomainError::Storage(e.to_string()))?;

        Ok(Self {
            vb: VarBuilder::from_tensors(tensors, DType::F16, &candle_device),
        })
    }

    /// Creates a context from a file.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Config` if the model architecture is incompatible.
    /// Returns `DomainError::Storage` if the file cannot be read or parsed.
    pub fn from_file(path: &str, device: &ComputeDevice) -> Result<Self, DomainError> {
        let candle_device = Converter::map_device(device)?;

        // VALIDATION: Check for config.json in the parent directory
        let path_buf = std::path::PathBuf::from(path);
        if let Some(weights_dir) = path_buf.parent()
            && let Some(model_root) = weights_dir.parent()
        {
            let config_path = model_root.join("config.json");
            if config_path.exists()
                && let Ok(config_str) = std::fs::read_to_string(&config_path)
            {
                let config: serde_json::Value =
                    serde_json::from_str(&config_str).unwrap_or_default();
                let arch = config["architecture"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_lowercase();
                if arch != "llama-moe" && !arch.contains("qwen2") && !arch.contains("llama") {
                    return Err(DomainError::Config(format!(
                        "Incompatible model architecture: {arch}. Supported: llama-moe, qwen2, llama.",
                    )));
                }
            }
        }

        let data = std::fs::read(path).map_err(|e| {
            DomainError::Storage(format!("Failed to read weights from {path}: {e}"))
        })?;
        let tensors = candle_core::safetensors::load_buffer(&data, &candle_device)
            .map_err(|e| DomainError::Storage(e.to_string()))?;

        Ok(Self {
            vb: VarBuilder::from_tensors(tensors, DType::F16, &candle_device),
        })
    }
}

/// Wrapper around the tokenizer to hide infrastructure details.
#[derive(Clone)]
pub struct TokenizerWrapper {
    pub inner: Tokenizer,
}

impl TokenizerWrapper {
    /// Loads a tokenizer from a file.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Config` if the file cannot be read, parsed, or if critical tokens are missing.
    pub fn from_file(path: &str) -> Result<Self, DomainError> {
        let mut inner = Tokenizer::from_file(path)
            .map_err(|e| DomainError::Config(format!("Tokenizer load error: {e}")))?;

        // Manual registration of added_tokens for Qwen2.5/SmolLM2
        let json_str = std::fs::read_to_string(path).map_err(|e| {
            DomainError::Config(format!("Failed to read tokenizer file {path}: {e}"))
        })?;

        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| DomainError::Config(format!("Failed to parse tokenizer json: {e}")))?;

        if let Some(added) = json["added_tokens"].as_array() {
            let special_tokens: Vec<tokenizers::AddedToken> = added
                .iter()
                .filter(|t| t["special"].as_bool().unwrap_or(false))
                .filter_map(|t| {
                    let content = t["content"].as_str()?.to_string();
                    let token = tokenizers::AddedToken::from(content, true);
                    Some(token)
                })
                .collect();

            if !special_tokens.is_empty() {
                inner.add_special_tokens(&special_tokens);
            }
        }

        // Qwen2.5 Canonical IDs Verification & Registration
        use crate::backends::tokens::qwen;
        let im_start = inner.token_to_id(qwen::IM_START);
        let im_end = inner.token_to_id(qwen::IM_END);
        let eos = inner.token_to_id(qwen::END_OF_TEXT);

        tracing::info!(
            "🔤 Qwen2.5 special tokens check: im_start={im_start:?}, im_end={im_end:?}, eos={eos:?}",
        );

        if im_start.is_none() || im_end.is_none() || eos.is_none() {
            tracing::warn!("⚠️ Some special tokens missing in tokenizer.json, adding manually");
            use tokenizers::AddedToken;
            inner.add_special_tokens(&[
                AddedToken::from(qwen::IM_START, true),
                AddedToken::from(qwen::IM_END, true),
                AddedToken::from(qwen::END_OF_TEXT, true),
            ]);
        }

        // Final validation for Qwen2.5
        if let Some(id) = inner.token_to_id(qwen::IM_END) {
            tracing::info!("✅ Tokenizer ready: {} detected with ID {id}", qwen::IM_END);
        } else {
            tracing::error!("❌ Tokenizer ERROR: {} still not found!", qwen::IM_END);
            return Err(DomainError::Config(format!(
                "Critical special token {} missing",
                qwen::IM_END
            )));
        }

        Ok(Self { inner })
    }

    /// Encodes text into token IDs.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Inference` if the encoding fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, DomainError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes token IDs back into text.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::Inference` if the decoding fails.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, DomainError> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| DomainError::Inference(e.to_string()))
    }

    /// Applies the Qwen2.5 ChatML chat template.
    /// system: if None, use the default Qwen system prompt.
    #[must_use]
    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)], // [("role", "content"), ...]
        add_generation_prompt: bool,
    ) -> String {
        apply_chat_template_raw(messages, add_generation_prompt)
    }

    /// Applies the Qwen2.5-Coder Fill-in-the-Middle (FIM) template.
    #[must_use]
    pub fn apply_fim_template(&self, prefix: &str, suffix: &str) -> String {
        apply_fim_template(prefix, suffix)
    }

    /// Returns the list of EOS token IDs for this tokenizer.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        // Common Qwen/Llama EOS tokens
        for token in ["<|endoftext|>", "<|im_end|>", "</s>"] {
            if let Some(id) = self.inner.token_to_id(token) {
                ids.push(id);
            }
        }
        ids
    }

    /// Creates a dummy tokenizer for unit testing.
    #[cfg(test)]
    pub fn dummy() -> Self {
        let json = r#"{
            "version": "1.0",
            "added_tokens": [{"id": 0, "content": "<|endoftext|>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}],
            "model": {"type": "BPE", "vocab": {"<|endoftext|>": 0, "a": 1, "b": 2}, "merges": []}
        }"#;
        let inner = Tokenizer::from_str(json).unwrap();
        Self { inner }
    }
}

impl tessella_core::traits::TokenizerService for TokenizerWrapper {
    fn encode(&self, text: &str) -> Result<Vec<u32>, DomainError> {
        self.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, DomainError> {
        self.decode(tokens)
    }

    fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> String {
        self.apply_chat_template(messages, add_generation_prompt)
    }

    fn apply_fim_template(&self, prefix: &str, suffix: &str) -> String {
        self.apply_fim_template(prefix, suffix)
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        self.eos_token_ids()
    }
}

pub use tessella_core::chat::{apply_chat_template_raw, apply_fim_template};
pub use tessella_core::models::InferenceMode;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template_basic() {
        let result = apply_chat_template_raw(&[("user", "Hello")], true);
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chat_template_custom_system() {
        let result = apply_chat_template_raw(&[("system", "Be concise."), ("user", "Hi")], true);
        assert_eq!(result.matches("<|im_start|>system").count(), 1);
    }

    #[test]
    fn test_fim_template_canonical() {
        let prefix = "def hello():\n    ";
        let suffix = "\n    return \"world\"";
        let result = apply_fim_template(prefix, suffix);

        // Canonical Qwen2.5-Coder format
        assert_eq!(
            result,
            "<|fim_prefix|>def hello():\n    <|fim_suffix|>\n    return \"world\"<|fim_middle|>"
        );

        // Verification of isolation: No ChatML tokens
        assert!(!result.contains("<|im_start|>"));
        assert!(!result.contains("<|im_end|>"));
    }
}
