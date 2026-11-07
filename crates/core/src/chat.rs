use serde::{Deserialize, Serialize};

pub use tessella_protocol::ChatMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FimRequest {
    pub prefix: String,
    pub suffix: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub session_id: Option<String>,
}

/// Applies the Qwen2.5-Coder Fill-in-the-Middle (FIM) template.
/// Canonical format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
#[must_use]
pub fn apply_fim_template(prefix: &str, suffix: &str) -> String {
    format!("<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>")
}

/// Applies the ChatML chat template.
#[must_use]
pub fn apply_chat_template_raw(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let system_default = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    let mut result = String::new();

    // Insert system if not provided in messages
    let has_system = messages.iter().any(|(r, _)| *r == "system");
    if !has_system {
        result.push_str(&format!("<|im_start|>system\n{system_default}<|im_end|>\n"));
    }

    for (role, content) in messages {
        result.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }

    if add_generation_prompt {
        result.push_str("<|im_start|>assistant\n");
    }

    result
}
