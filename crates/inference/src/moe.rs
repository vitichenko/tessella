use crate::models::CandleMlpExpert;
use candle_core::{D, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::sync::Arc;
use tessella_core::error::DomainError;
use tessella_core::models::ModelConfig;
use tessella_core::telemetry::metrics::ExpertActivationRegistry;

use candle_transformers::models::deepseek2::TopKLastDimOp;

pub struct MoeGate {
    linear: Linear,
    _top_k: usize,
    norm_topk_prob: bool,
}

impl MoeGate {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self, DomainError> {
        let num_experts = config
            .num_experts
            .ok_or_else(|| DomainError::Config("num_experts missing for MoE gate".into()))?;
        let top_k = config.num_experts_per_tok.ok_or_else(|| {
            DomainError::Config("num_experts_per_tok missing for MoE gate".into())
        })?;
        let norm_topk_prob = config.norm_topk_prob.unwrap_or(false);

        // Qwen2-MoE uses a linear layer without bias for the gate
        let linear = candle_nn::linear_no_bias(config.hidden_size, num_experts, vb)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        Ok(Self {
            linear,
            _top_k: top_k,
            norm_topk_prob,
        })
    }

    /// Returns (indices, weights) both of shape [batch*seq, top_k]
    pub fn forward(&self, hidden: &Tensor) -> Result<(Tensor, Tensor), DomainError> {
        let logits = self
            .linear
            .forward(hidden)
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        let probs = candle_nn::ops::softmax(
            &logits
                .to_dtype(candle_core::DType::F32)
                .map_err(|e| DomainError::Inference(e.to_string()))?,
            candle_core::D::Minus1,
        )
        .map_err(|e| DomainError::Inference(e.to_string()))?
        .to_dtype(logits.dtype())
        .map_err(|e| DomainError::Inference(e.to_string()))?;

        let topk_output = probs
            .topk(self._top_k)
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let (weights, indices) = (topk_output.values, topk_output.indices);

        if self.norm_topk_prob && self._top_k > 1 {
            let sum = weights
                .sum_keepdim(D::Minus1)
                .map_err(|e| DomainError::Inference(e.to_string()))?;
            let weights = weights
                .broadcast_div(&sum)
                .map_err(|e| DomainError::Inference(e.to_string()))?;
            Ok((indices, weights))
        } else {
            Ok((indices, weights))
        }
    }
}

pub struct MoeBlock {
    gate: MoeGate,
    shared_expert: Option<CandleMlpExpert>,
    routed_experts: Vec<Option<CandleMlpExpert>>,
    num_experts: usize,
    /// Where the routing of this block is counted (`TASK-43`/S49).
    ///
    /// One registry per **engine** —not one per process—, so two nodes in the same process do not mix
    /// their telemetry. The same `Arc` is what the heartbeat drains into `/announce`, and the engine is
    /// who owns it, so the hot path and the announce cannot disagree.
    activations: Arc<ExpertActivationRegistry>,
}

impl MoeBlock {
    pub fn new(
        config: &ModelConfig,
        vb: VarBuilder,
        expert_range: Option<(usize, usize)>,
        activations: Arc<ExpertActivationRegistry>,
    ) -> Result<Self, DomainError> {
        let num_experts = config
            .num_experts
            .ok_or_else(|| DomainError::Config("num_experts missing for MoeBlock".into()))?;
        let _top_k = config.num_experts_per_tok.ok_or_else(|| {
            DomainError::Config("num_experts_per_tok missing for MoeBlock".into())
        })?;

        // `S43`/`TASK-37`: expert parallelism is not implemented. A partial range used to be accepted
        // here, and the experts outside it contributed zeros (`None => continue` below) while the
        // attention, the residual and the shared expert were computed all the same. That is a whole
        // layer returned for a fraction of the experts: gathering it with its siblings counts those
        // parts once per peer and produces wrong logits with no warning at all. Refusing is the only
        // honest answer, and the policy has one definition (`tessella-core::assignment`).
        if let Some(reason) =
            tessella_core::assignment::unsupported_expert_range_reason(expert_range, num_experts)
        {
            return Err(DomainError::Config(reason));
        }

        let gate = MoeGate::new(config, vb.pp("gate"))?;

        let shared_expert = if let Some(size) = config.shared_expert_intermediate_size {
            Some(CandleMlpExpert::from_vb(
                0,
                config.hidden_size,
                size,
                vb.pp("shared_expert"),
            )?)
        } else {
            None
        };

        let mut routed_experts = Vec::with_capacity(num_experts);
        let experts_vb = vb.pp("experts");
        for i in 0..num_experts {
            let in_range = if let Some((start, end)) = expert_range {
                i >= start && i < end
            } else {
                true
            };

            if in_range {
                routed_experts.push(Some(CandleMlpExpert::from_vb(
                    i as u32,
                    config.hidden_size,
                    config.intermediate_size,
                    experts_vb.pp(i),
                )?));
            } else {
                routed_experts.push(None);
            }
        }

        Ok(Self {
            gate,
            shared_expert,
            routed_experts,
            num_experts,
            activations,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor, DomainError> {
        let (b, s, h) = hidden
            .dims3()
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let hidden_flat = hidden
            .reshape((b * s, h))
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        let (indices, weights) = self.gate.forward(&hidden_flat)?;

        let mut final_output =
            Tensor::zeros_like(&hidden_flat).map_err(|e| DomainError::Inference(e.to_string()))?;

        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared
                .forward_tensor(&hidden_flat)?
                .to_dtype(hidden.dtype())
                .map_err(|e| DomainError::Inference(e.to_string()))?;
            final_output =
                (final_output + shared_out).map_err(|e| DomainError::Inference(e.to_string()))?;
        }

        let indices_vec = indices
            .to_vec2::<u32>()
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let weights_vec = weights
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| DomainError::Inference(e.to_string()))?
            .to_vec2::<f32>()
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        for expert_idx in 0..self.num_experts {
            // Skip experts not in our range
            let expert = match &self.routed_experts[expert_idx] {
                Some(e) => e,
                None => continue,
            };

            let mut token_indices = Vec::new();
            let mut token_weights = Vec::new();

            for (token_idx, token_experts) in indices_vec.iter().enumerate() {
                for (k, &exp_id) in token_experts.iter().enumerate() {
                    if exp_id as usize == expert_idx {
                        token_indices.push(token_idx as u32);
                        token_weights.push(weights_vec[token_idx][k]);
                        // S10-T2: Track expert activation in this engine's registry (S49/TASK-43).
                        self.activations.record(exp_id);
                    }
                }
            }

            if !token_indices.is_empty() {
                let device = hidden.device();
                let idx_tensor = Tensor::new(token_indices.as_slice(), device)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;
                let weight_tensor = Tensor::new(token_weights.as_slice(), device)
                    .map_err(|e| DomainError::Inference(e.to_string()))?
                    .reshape((token_indices.len(), 1))
                    .map_err(|e| DomainError::Inference(e.to_string()))?
                    .to_dtype(hidden.dtype())
                    .map_err(|e| DomainError::Inference(e.to_string()))?;

                let expert_inputs = hidden_flat
                    .index_select(&idx_tensor, 0)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;

                let expert_outputs = expert
                    .forward_tensor(&expert_inputs)?
                    .to_dtype(hidden.dtype())
                    .map_err(|e| DomainError::Inference(e.to_string()))?;

                let weighted_outputs = expert_outputs
                    .broadcast_mul(&weight_tensor)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;

                final_output = final_output
                    .index_add(&idx_tensor, &weighted_outputs, 0)
                    .map_err(|e| DomainError::Inference(e.to_string()))?;
            }
        }

        final_output
            .reshape((b, s, h))
            .map_err(|e| DomainError::Inference(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_moe_gate_forward() {
        let config = ModelConfig {
            model_id: "test".to_string(),
            model_type: Some("qwen2_moe".to_string()),
            architectures: vec!["Qwen2MoeForCausalLM".to_string()],
            hidden_size: 16,
            num_experts: Some(8),
            num_experts_per_tok: Some(2),
            ..ModelConfig::default()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let gate = MoeGate::new(&config, vb).unwrap();
        let hidden = Tensor::randn(0.0f32, 1.0f32, (2, 4, 16), &device).unwrap();

        let (indices, weights) = gate.forward(&hidden.reshape((8, 16)).unwrap()).unwrap();

        assert_eq!(indices.dims(), &[8, 2]);
        assert_eq!(weights.dims(), &[8, 2]);

        let sum = weights.sum_keepdim(1).unwrap().to_vec2::<f32>().unwrap();
        for s in sum {
            assert!(s[0] <= 1.0001);
        }
    }

    #[test]
    fn test_moe_block_forward() {
        let config = ModelConfig {
            model_id: "test".to_string(),
            model_type: Some("qwen2_moe".to_string()),
            architectures: vec!["Qwen2MoeForCausalLM".to_string()],
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: Some(2),
            num_experts_per_tok: Some(1),
            ..ModelConfig::default()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let block =
            MoeBlock::new(&config, vb, None, Arc::new(ExpertActivationRegistry::new())).unwrap();
        let hidden = Tensor::randn(0.0f32, 1.0f32, (1, 2, 16), &device).unwrap();

        let output = block.forward(&hidden).unwrap();
        assert_eq!(output.dims(), &[1, 2, 16]);
    }

    /// Criterio 2 de `TASK-43`: el camino caliente cuenta en **su** registro, no en uno de proceso.
    ///
    /// Es el candado del defecto: `record` se llamaba sobre un `static`, así que dos motores en el mismo
    /// proceso mezclaban sus contadores y `drain()` de uno vaciaba al otro. Aquí se comprueba que el
    /// `MoeBlock` escribe en el registro que se le inyecta y que **otro** registro no se entera.
    #[test]
    fn test_moe_block_records_activations_in_its_own_registry() {
        let config = ModelConfig {
            model_id: "test".to_string(),
            model_type: Some("qwen2_moe".to_string()),
            architectures: vec!["Qwen2MoeForCausalLM".to_string()],
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: Some(2),
            num_experts_per_tok: Some(1),
            ..ModelConfig::default()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let mine = Arc::new(ExpertActivationRegistry::new());
        let other = Arc::new(ExpertActivationRegistry::new());

        let block = MoeBlock::new(&config, vb, None, mine.clone()).unwrap();
        // 2 tokens × top-1 = 2 activaciones repartidas entre los dos expertos.
        let hidden = Tensor::randn(0.0f32, 1.0f32, (1, 2, 16), &device).unwrap();
        let _ = block.forward(&hidden).unwrap();

        assert!(
            (1..=2).contains(&mine.experts()),
            "dos tokens con top-1 caen en uno o dos de los dos expertos, no en otro sitio: {}",
            mine.experts()
        );
        let usage = mine.drain();
        let total: u64 = usage.values().sum();
        assert_eq!(total, 2, "una activación por token enrutado: {usage:?}");
        assert!(usage.keys().all(|expert| *expert < 2), "{usage:?}");
        assert_eq!(other.experts(), 0, "el otro registro no se enteró");
        assert!(other.drain().is_empty());
    }

    /// `S43`/`TASK-37`: a shard that holds only part of the experts cannot be served. It used to be
    /// accepted here — the experts outside the range contributed zeros while attention, residual and
    /// the shared expert were computed anyway — so the shard returned a whole layer for a fraction of
    /// the experts and the gather summed those parts once per peer.
    #[test]
    fn test_moe_block_refuses_a_partial_expert_range() {
        let config = ModelConfig {
            model_id: "test".to_string(),
            model_type: Some("qwen2_moe".to_string()),
            architectures: vec!["Qwen2MoeForCausalLM".to_string()],
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: Some(8),
            num_experts_per_tok: Some(2),
            ..ModelConfig::default()
        };
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let error = match MoeBlock::new(
            &config,
            vb,
            Some((0, 4)),
            Arc::new(ExpertActivationRegistry::new()),
        ) {
            Ok(_) => panic!("a partial expert range must be refused, not zero-filled"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains("expert parallelism is not implemented") && message.contains("[0, 4)"),
            "the error names the reason and the range: {message}"
        );

        // The full expert set is the same thing as no range at all: both are whole-layer shards.
        let varmap_full = VarMap::new();
        let vb_full = VarBuilder::from_varmap(&varmap_full, candle_core::DType::F32, &device);
        assert!(
            MoeBlock::new(
                &config,
                vb_full,
                Some((0, 8)),
                Arc::new(ExpertActivationRegistry::new())
            )
            .is_ok()
        );

        let varmap_none = VarMap::new();
        let vb_none = VarBuilder::from_varmap(&varmap_none, candle_core::DType::F32, &device);
        assert!(
            MoeBlock::new(
                &config,
                vb_none,
                None,
                Arc::new(ExpertActivationRegistry::new())
            )
            .is_ok()
        );
    }
}
