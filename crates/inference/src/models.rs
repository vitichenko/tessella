use crate::conversion::Converter;
use candle_core::quantized::QMatMul;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use tessella_core::error::DomainError;
use tessella_core::models::{ComputeDevice, ExpertId};

pub struct QuantizedLinear {
    inner: QMatMul,
}

impl QuantizedLinear {
    pub fn new(qmatmul: QMatMul) -> Self {
        Self { inner: qmatmul }
    }

    /// Explicitly creates a QuantizedLinear from a weight, ensuring it is actually quantized.
    pub fn try_from_weight(weight: &crate::shard_loader::Weight) -> Result<Self, DomainError> {
        match weight {
            crate::shard_loader::Weight::Quantized(q) => {
                let qmatmul = QMatMul::from_arc(q.clone()).map_err(|e| {
                    DomainError::Inference(format!("Failed to create QMatMul: {}", e))
                })?;
                Ok(Self::new(qmatmul))
            }
            crate::shard_loader::Weight::Tensor(t) => Err(DomainError::Inference(format!(
                "CRITICAL: Attempted to create QuantizedLinear from Dense F16 tensor (shape: {:?}). \
                     This is a logic error in the backend selection.",
                t.shape()
            ))),
        }
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
    }
}

pub enum LinearLayer {
    Full(Linear),
    Quantized(QuantizedLinear),
}

impl Module for LinearLayer {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Full(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}

/// Llama-style SwiGLU MLP Expert implemented with Candle.
pub struct CandleMlpExpert {
    id: ExpertId,
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
    pub varmap: VarMap,
    device: Device,
    pub weights: HashMap<String, crate::shard_loader::Weight>, // Empty = dummy mode
}

impl CandleMlpExpert {
    /// Creates a new MLP expert.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if weights are invalid or device mapping fails.
    pub fn new(
        id: ExpertId,
        dim: usize,
        intermediate_size: usize,
        ctx: &crate::InferenceContext,
        device: ComputeDevice,
        weights: Option<HashMap<String, crate::shard_loader::Weight>>,
        factory: &crate::LayerFactory,
    ) -> Result<Self, DomainError> {
        let device = Converter::map_device(&device)?;

        let (gate_proj, up_proj, down_proj) = if let Some(ref w_map) = weights {
            let gate = factory.linear("mlp.gate_proj", w_map)?;
            let up = factory.linear("mlp.up_proj", w_map)?;
            let down = factory.linear("mlp.down_proj", w_map)?;
            (gate, up, down)
        } else {
            let gate_proj =
                candle_nn::linear_no_bias(dim, intermediate_size, ctx.vb.pp("gate_proj"))
                    .map_err(|e| DomainError::Inference(e.to_string()))?;
            let up_proj = candle_nn::linear_no_bias(dim, intermediate_size, ctx.vb.pp("up_proj"))
                .map_err(|e| DomainError::Inference(e.to_string()))?;
            let down_proj =
                candle_nn::linear_no_bias(intermediate_size, dim, ctx.vb.pp("down_proj"))
                    .map_err(|e| DomainError::Inference(e.to_string()))?;
            (
                LinearLayer::Full(gate_proj),
                LinearLayer::Full(up_proj),
                LinearLayer::Full(down_proj),
            )
        };

        Ok(Self {
            id,
            gate_proj,
            up_proj,
            down_proj,
            varmap: VarMap::new(),
            device,
            weights: weights.unwrap_or_default(),
        })
    }

    /// Saves the model weights to a file.
    ///
    /// # Errors
    ///
    /// Returns `candle_core::Error` if saving fails.
    pub fn save(&self, path: impl AsRef<Path>) -> candle_core::Result<()> {
        self.varmap.save(path)
    }

    /// Loads the model weights from a file.
    ///
    /// # Errors
    ///
    /// Returns `candle_core::Error` if loading fails.
    pub fn load(&mut self, path: impl AsRef<Path>, _device: &Device) -> candle_core::Result<()> {
        self.varmap.load(path)
    }

    /// Creates an expert from a VarBuilder.
    pub fn from_vb(
        id: ExpertId,
        dim: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self, DomainError> {
        let gate_proj = candle_nn::linear_no_bias(dim, intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let up_proj = candle_nn::linear_no_bias(dim, intermediate_size, vb.pp("up_proj"))
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let down_proj = candle_nn::linear_no_bias(intermediate_size, dim, vb.pp("down_proj"))
            .map_err(|e| DomainError::Inference(e.to_string()))?;

        Ok(Self {
            id,
            gate_proj: LinearLayer::Full(gate_proj),
            up_proj: LinearLayer::Full(up_proj),
            down_proj: LinearLayer::Full(down_proj),
            varmap: VarMap::new(),
            device: vb.device().clone(),
            weights: HashMap::new(),
        })
    }

    pub fn forward_tensor(&self, t: &Tensor) -> Result<Tensor, DomainError> {
        let gate = self
            .gate_proj
            .forward(t)
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let up = self
            .up_proj
            .forward(t)
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let activated =
            (candle_nn::ops::silu(&gate).map_err(|e| DomainError::Inference(e.to_string()))? * up)
                .map_err(|e| DomainError::Inference(e.to_string()))?;
        self.down_proj
            .forward(&activated)
            .map_err(|e| DomainError::Inference(e.to_string()))
    }

    pub fn id(&self) -> ExpertId {
        self.id
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceContext;

    #[test]
    fn expert_dummy_mode_works() {
        let device = ComputeDevice::Cpu;
        let ctx = InferenceContext::empty(&device).unwrap();
        let factory = crate::LayerFactory::new(tessella_core::models::QuantizationConfig::None);
        let expert = CandleMlpExpert::new(42, 1024, 2048, &ctx, device, None, &factory).unwrap();

        let input = Tensor::zeros(
            (1, 1024),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = expert.forward_tensor(&input).unwrap();
        assert_eq!(output.dims(), &[1, 1024]);
    }
}
