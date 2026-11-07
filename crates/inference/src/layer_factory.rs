use crate::models::{LinearLayer, QuantizedLinear};
use crate::shard_loader::Weight;
use std::collections::HashMap;
use tessella_core::error::DomainError;
use tessella_core::models::QuantizationConfig;

pub struct LayerFactory {
    pub quantization: QuantizationConfig,
}

impl LayerFactory {
    pub fn new(quantization: QuantizationConfig) -> Self {
        Self { quantization }
    }

    pub fn linear(
        &self,
        name: &str,
        weights: &HashMap<String, Weight>,
    ) -> Result<LinearLayer, DomainError> {
        let key = weights
            .keys()
            .find(|k| k.contains(name) && k.contains("weight"))
            .ok_or_else(|| DomainError::Inference(format!("Weight not found for {}", name)))?;
        let w = weights.get(key).unwrap();
        match w {
            Weight::Tensor(t) => {
                // Find bias if it exists
                let bias_key = weights
                    .keys()
                    .find(|k| k.contains(name) && k.contains("bias"));
                let bias = bias_key.and_then(|k| {
                    if let Weight::Tensor(bt) = weights.get(k)? {
                        Some(bt.clone())
                    } else {
                        None
                    }
                });
                Ok(LinearLayer::Full(candle_nn::Linear::new(t.clone(), bias)))
            }
            Weight::Quantized(q) => {
                if matches!(self.quantization, QuantizationConfig::Q8_0) {
                    Ok(LinearLayer::Quantized(QuantizedLinear::try_from_weight(w)?))
                } else {
                    // Fallback: the user requested no quantization but the shard is GGUF.
                    // Dequantize-on-load to F16 to honor the preference.
                    tracing::warn!(
                        "⚠️ Falling back to dequantize-on-load for {} (QuantizationConfig::None set with GGUF shard)",
                        name
                    );
                    let device = q.device();
                    let tensor = q.dequantize(&device).map_err(|e| {
                        DomainError::Inference(format!("Fallback dequantization failed: {}", e))
                    })?;
                    Ok(LinearLayer::Full(candle_nn::Linear::new(tensor, None)))
                }
            }
        }
    }
}
