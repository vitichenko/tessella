use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

use crate::attention::{KvCacheEntry, QwenAttention, RotaryEmbedding};
use crate::backends::{
    BackendCapabilities, BackendConfig, InferenceBackend, InferenceSession, ModelTokens,
};
use crate::moe::MoeBlock;
use crate::shard_loader::ShardedWeights;
use tessella_core::error::DomainError;
use tessella_core::telemetry::metrics::ExpertActivationRegistry;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_theta: f32,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
}

pub struct MLP {
    gate_proj: crate::models::LinearLayer,
    up_proj: crate::models::LinearLayer,
    down_proj: crate::models::LinearLayer,
}

impl MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj: crate::models::LinearLayer::Full(gate_proj),
            up_proj: crate::models::LinearLayer::Full(up_proj),
            down_proj: crate::models::LinearLayer::Full(down_proj),
        })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward(&activated)
    }
}

pub enum Qwen2Mlp {
    Dense(MLP),
    Moe(crate::moe::MoeBlock),
}

pub struct Layer {
    pub self_attn: QwenAttention,
    pub mlp: Qwen2Mlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl Layer {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        rope: Arc<RotaryEmbedding>,
        weights: Option<&std::collections::HashMap<String, crate::shard_loader::Weight>>,
        factory: &crate::LayerFactory,
        expert_range: Option<(usize, usize)>,
        activations: Arc<ExpertActivationRegistry>,
    ) -> candle_core::Result<Self> {
        let model_cfg = tessella_core::models::ModelConfig {
            model_id: "qwen2".to_string(),
            quantization: factory.quantization,
            model_type: Some("qwen2".to_string()),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            vocab_size: cfg.vocab_size,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            intermediate_size: cfg.intermediate_size,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            moe_intermediate_size: cfg.moe_intermediate_size,
            shared_expert_intermediate_size: cfg.shared_expert_intermediate_size,
            moe_layer_freq: None,
            norm_topk_prob: None,
            max_position_embeddings: cfg.max_position_embeddings,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta as f64,
            eos_token_id: None,
        };

        let self_attn = QwenAttention::new(&model_cfg, vb.pp("self_attn"), rope, weights, factory)
            .map_err(candle_core::Error::wrap)?;

        let mlp = if cfg.num_experts.unwrap_or(0) > 0 {
            let moe = MoeBlock::new(&model_cfg, vb.pp("mlp"), expert_range, activations)
                .map_err(candle_core::Error::wrap)?;
            Qwen2Mlp::Moe(moe)
        } else {
            let dense = MLP::new(cfg, vb.pp("mlp"))?;
            Qwen2Mlp::Dense(dense)
        };

        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

pub struct Qwen2Model {
    pub embed_tokens: Option<Embedding>,
    pub layers: Vec<Layer>,
    pub norm: Option<RmsNorm>,
    pub layer_range: (usize, usize),
}

impl Qwen2Model {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        layer_range: (usize, usize),
        sharded_weights: &ShardedWeights,
        activations: Arc<ExpertActivationRegistry>,
    ) -> candle_core::Result<Self> {
        let (start, end) = layer_range;

        let embed_tokens = if start == 0 {
            Some(candle_nn::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.pp("embed_tokens"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::new();
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f64,
            cfg.hidden_size / cfg.num_attention_heads,
            cfg.max_position_embeddings,
            vb.device(),
        )?);

        for i in layer_range.0..layer_range.1 {
            let layer_vb = vb.pp(format!("layers.{}", i));
            let layer_weights = sharded_weights.layers.get(&i);
            let factory = crate::LayerFactory::new(sharded_weights.quantization);
            let layer = Layer::new(
                cfg,
                layer_vb,
                rope.clone(),
                layer_weights,
                &factory,
                sharded_weights.expert_range,
                activations.clone(),
            )
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            layers.push(layer);
        }

        let norm = if end >= cfg.num_hidden_layers {
            Some(candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("norm"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            layer_range,
        })
    }
}

pub struct ModelForCausalLM {
    pub model: Qwen2Model,
    pub lm_head: Option<crate::models::LinearLayer>,
    pub config: Config,
    pub total_layers: usize,
}

impl ModelForCausalLM {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        layer_range: (usize, usize),
        sharded_weights: &ShardedWeights,
        activations: Arc<ExpertActivationRegistry>,
    ) -> candle_core::Result<Self> {
        let model = Qwen2Model::new(
            cfg,
            vb.pp("model"),
            layer_range,
            sharded_weights,
            activations,
        )?;

        let lm_head = if layer_range.1 >= cfg.num_hidden_layers {
            let factory = crate::LayerFactory::new(sharded_weights.quantization);
            if let Some(_w) = sharded_weights.global.get("lm_head.weight") {
                Some(
                    factory
                        .linear("lm_head", &sharded_weights.global)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                )
            } else if cfg.tie_word_embeddings {
                // Ties use embed_tokens.weight
                if let Some(_w) = sharded_weights.global.get("model.embed_tokens.weight") {
                    Some(
                        factory
                            .linear("model.embed_tokens", &sharded_weights.global)
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                    )
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            model,
            lm_head,
            config: cfg.clone(),
            total_layers: cfg.num_hidden_layers,
        })
    }
}

pub struct Backend {
    model_id: String,
    device: Device,
    config: BackendConfig,
    capabilities: BackendCapabilities,
    model: Arc<ModelForCausalLM>,
}

impl Backend {
    /// Builds the backend on an explicit candle device.
    ///
    /// `activations` is the registry of the engine that owns this backend (`TASK-43`/S49): the MoE
    /// blocks record their routing there, and the engine's heartbeat drains **the same** registry into
    /// `/announce`. It is a parameter and not a `static` so two engines in one process cannot mix
    /// their telemetry.
    pub fn new_with_candle_device(
        model_id: &str,
        model_dir: &std::path::Path,
        sharded_weights: &ShardedWeights,
        candle_device: &Device,
        tokenizer: Arc<crate::TokenizerWrapper>,
        activations: Arc<ExpertActivationRegistry>,
    ) -> std::result::Result<Self, DomainError> {
        let cfg = load_qwen2_config(model_dir)?;
        let layer_range = sharded_weights.layer_range;

        use crate::backends::tokens::qwen;
        let eos = ModelTokens::resolve(&tokenizer.inner, qwen::END_OF_TEXT)?;
        let im_end = ModelTokens::resolve(&tokenizer.inner, qwen::IM_END)?;
        let tokens = ModelTokens {
            eos_ids: vec![eos, im_end],
            bos_turn: Some(eos), // Use EOS as placeholder for u32 fields
            eos_turn: Some(im_end),
            fim_prefix: None,
            fim_suffix: None,
            fim_middle: None,
        };

        let backend_config = BackendConfig {
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_position_embeddings,
            eos_token_ids: tokens.eos_ids.clone(),
            supports_system_prompt: true,
        };

        let dtype = crate::backends::activation_dtype(candle_device);

        let capabilities = BackendCapabilities {
            supports_kv_cache: true,
            supports_chat_template: true,
            supports_moe_routing: false,
            supports_sliding_window: cfg.use_sliding_window,
            supports_quantized_weights: true,
            default_stop_token_ids: tokens.eos_ids.clone(),
            default_temperature: Some(0.7),
        };

        let mut tensor_weights = std::collections::HashMap::new();
        for (k, v) in sharded_weights.global.iter() {
            if let crate::shard_loader::Weight::Tensor(t) = v {
                tensor_weights.insert(k.clone(), t.clone());
            }
        }
        for layer_weights in sharded_weights.layers.values() {
            for (k, v) in layer_weights.iter() {
                if let crate::shard_loader::Weight::Tensor(t) = v {
                    tensor_weights.insert(k.clone(), t.clone());
                }
            }
        }

        let vb = VarBuilder::from_tensors(tensor_weights, dtype, candle_device);
        let model = ModelForCausalLM::new(&cfg, vb, layer_range, sharded_weights, activations)
            .map_err(|e| DomainError::Inference(format!("Failed to build Qwen2 weights: {}", e)))?;

        Ok(Self {
            model_id: model_id.to_string(),
            device: candle_device.clone(),
            config: backend_config,
            capabilities,
            model: Arc::new(model),
        })
    }
}

#[async_trait::async_trait]
impl InferenceBackend for Backend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn config(&self) -> &BackendConfig {
        &self.config
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn begin_session(&self) -> Result<Box<dyn InferenceSession>, DomainError> {
        Ok(Box::new(Session {
            model: self.model.clone(),
            device: self.device.clone(),
            pos_offset: 0,
            kv_cache: std::collections::HashMap::new(),
        }))
    }
}

pub struct Session {
    model: Arc<ModelForCausalLM>,
    device: Device,
    pos_offset: usize,
    kv_cache: std::collections::HashMap<usize, KvCacheEntry>,
}

impl InferenceSession for Session {
    fn forward(&mut self, tokens: &[u32]) -> Result<Tensor, DomainError> {
        let input = Tensor::new(tokens, &self.device)
            .map_err(|e| DomainError::Inference(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        self.apply_layers(input, self.model.model.layer_range, self.pos_offset)
    }

    fn apply_layers(
        &mut self,
        mut x: Tensor,
        _range: (usize, usize),
        seq_start_pos: usize,
    ) -> Result<Tensor, DomainError> {
        let seq_len = x
            .dim(1)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        let target_dtype = if self.device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };

        // Ensure input x matches the backend's preferred floating-point type if not U32
        if x.dtype() != DType::U32 && x.dtype() != target_dtype {
            x = x
                .to_dtype(target_dtype)
                .map_err(|e| DomainError::Inference(e.to_string()))?;
        }

        let mask = if seq_len > 1 {
            let seq_len_k = seq_start_pos + seq_len;
            Some(
                crate::attention::get_causal_mask_with_offset(
                    seq_len,
                    seq_len_k,
                    seq_start_pos,
                    &self.device,
                )
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .to_dtype(target_dtype)
                .map_err(|e| DomainError::Inference(e.to_string()))?,
            )
        } else {
            None
        };

        // 1. Initial Embeddings
        if x.dtype() == DType::U32 {
            if self.model.model.layer_range.0 == 0 {
                if let Some(ref embed) = self.model.model.embed_tokens {
                    x = embed.forward(&x).map_err(|e| {
                        DomainError::Inference(format!("Embedding forward failed: {}", e))
                    })?;
                    // Ensure we are in target_dtype after embedding
                    x = x
                        .to_dtype(target_dtype)
                        .map_err(|e| DomainError::Inference(e.to_string()))?;
                } else {
                    return Err(DomainError::Inference(
                        "Worker 0 missing embedding layer".into(),
                    ));
                }
            } else {
                return Err(DomainError::Inference(format!(
                    "Worker {} received U32 tokens (dtype {:?}) but expected {:?} hidden states from previous worker. Payload seq_start_pos={}",
                    self.model.model.layer_range.0,
                    x.dtype(),
                    target_dtype,
                    seq_start_pos
                )));
            }
        }

        // 2. Transformer Layers
        let (start, _end) = self.model.model.layer_range;
        for (local_idx, layer) in self.model.model.layers.iter().enumerate() {
            let global_idx = start + local_idx;
            let residual = x.clone();

            // RMSNorm often benefits from F32 accumulation, but input/weight must match.
            // If weights are target_dtype, input must be target_dtype.
            let x_norm = layer
                .input_layernorm
                .forward(&x)
                .map_err(|e| DomainError::Inference(e.to_string()))?;

            let (attn_out, k, v) = layer
                .self_attn
                .forward(
                    &x_norm,
                    mask.as_ref(),
                    self.kv_cache.get_mut(&global_idx),
                    seq_start_pos,
                )
                .map_err(|e| DomainError::Inference(e.to_string()))?;

            // If cache was None, we need to create it and store it
            self.kv_cache
                .entry(global_idx)
                .or_insert_with(|| KvCacheEntry {
                    k,
                    v,
                    last_accessed: std::time::Instant::now(),
                });

            x = (attn_out + residual).map_err(|e| DomainError::Inference(e.to_string()))?;

            let residual = x.clone();
            let x_norm = layer
                .post_attention_layernorm
                .forward(&x)
                .map_err(|e| DomainError::Inference(e.to_string()))?;
            let mlp_out = match &layer.mlp {
                Qwen2Mlp::Dense(mlp) => mlp
                    .forward(&x_norm)
                    .map_err(|e| DomainError::Inference(e.to_string())),
                Qwen2Mlp::Moe(moe) => moe.forward(&x_norm),
            }?;
            x = (mlp_out + residual).map_err(|e| DomainError::Inference(e.to_string()))?;
        }

        // 3. Final Norm and Head (if last worker)
        if self.model.model.layer_range.1 >= self.model.total_layers {
            if let Some(ref norm) = self.model.model.norm {
                x = norm
                    .forward(&x)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;
            }
            if let Some(ref head) = self.model.lm_head {
                x = head
                    .forward(&x)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;

                // For generation, we only need the last token's logits
                let (_b, seq_len, _v) = x
                    .dims3()
                    .map_err(|e| DomainError::Inference(e.to_string()))?;
                if seq_len > 1 {
                    x = x
                        .narrow(1, seq_len - 1, 1)
                        .map_err(|e| DomainError::Inference(e.to_string()))?;
                }
                x = x
                    .squeeze(0)
                    .map_err(|e| DomainError::Inference(e.to_string()))?
                    .squeeze(0)
                    .map_err(|e| DomainError::Inference(e.to_string()))?
                    .to_dtype(DType::F32)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;
            }
        }

        self.pos_offset += seq_len;
        Ok(x)
    }

    fn reset(&mut self) {
        self.pos_offset = 0;
        self.kv_cache.clear();
    }

    fn set_pos_offset(&mut self, pos: usize) {
        self.pos_offset = pos;
    }

    fn get_pos_offset(&self) -> usize {
        self.pos_offset
    }
}

fn load_qwen2_config(model_dir: &std::path::Path) -> Result<Config, DomainError> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| DomainError::Config(format!("Failed to read config.json: {}", e)))?;
    serde_json::from_str(&config_str)
        .map_err(|e| DomainError::Config(format!("Failed to parse config.json: {}", e)))
}
