/// Architecture-specific special tokens.
///
/// These are defined here instead of core because they are model-specific.
/// Added tokens are grouped by family.
pub mod qwen {
    pub const IM_START: &str = "<|im_start|>";
    pub const IM_END: &str = "<|im_end|>";
    pub const END_OF_TEXT: &str = "<|endoftext|>";
}

pub mod llama {
    pub const BOS: &str = "<s>";
    pub const EOS: &str = "</s>";
}

pub mod deepseek {
    pub const BOS: &str = "<｜begin of sentence｜>";
    pub const EOS: &str = "<｜end of sentence｜>";
}
