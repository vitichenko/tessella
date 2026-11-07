use crate::error::DomainError;

/// Abstract tokenizer service trait for text encoding, decoding, and prompt templating.
///
/// Decouples higher-level presentation layers (HTTP, CLI) from concrete tokenizers
/// and machine-learning frameworks (DIP/ISP).
pub trait TokenizerService: Send + Sync {
    /// Encodes a text prompt into token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>, DomainError>;

    /// Decodes token IDs back into text.
    fn decode(&self, tokens: &[u32]) -> Result<String, DomainError>;

    /// Applies the chat template to conversation messages.
    fn apply_chat_template(&self, messages: &[(&str, &str)], add_generation_prompt: bool)
    -> String;

    /// Applies the Fill-in-the-Middle (FIM) template.
    fn apply_fim_template(&self, prefix: &str, suffix: &str) -> String;

    /// Returns the configured EOS token IDs.
    fn eos_token_ids(&self) -> Vec<u32>;
}
