use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::info;

#[derive(Clone)]
pub enum Weight {
    Tensor(Tensor),
    Quantized(Arc<QTensor>),
}

impl Weight {
    pub fn device(&self) -> Device {
        match self {
            Self::Tensor(t) => t.device().clone(),
            Self::Quantized(q) => q.device().clone(),
        }
    }

    pub fn dtype(&self) -> candle_core::DType {
        match self {
            Self::Tensor(t) => t.dtype(),
            Self::Quantized(_) => candle_core::DType::F32,
        }
    }

    pub fn shape(&self) -> candle_core::Shape {
        match self {
            Self::Tensor(t) => t.shape().clone(),
            Self::Quantized(q) => q.shape().clone(),
        }
    }
}

#[derive(Clone)]
pub struct ShardedWeights {
    /// layer_index → tensor_name → Weight (Dense/Attention)
    pub layers: HashMap<usize, HashMap<String, Weight>>,
    /// Non-layer tensors (embeddings, final norm, lm_head)
    pub global: HashMap<String, Weight>,
    /// Range of layers assigned to this shard (start, end)
    pub layer_range: (usize, usize),
    /// Range of experts assigned to MoE layers (start, end)
    pub expert_range: Option<(usize, usize)>,
    /// The user's quantization preference
    pub quantization: tessella_core::models::QuantizationConfig,
}

/// Loads model tensors filtering by layer_range and expert_range.
///
/// - layer_range: (start, end) — only load layers [start, end)
/// - expert_range: (start, end) — only load experts [start, end) if applicable
/// - total_layers: Total number of layers in the model to identify the final worker
fn should_load_tensor(
    name: &str,
    layer_start: usize,
    layer_end: usize,
    expert_range: Option<(usize, usize)>,
    total_layers: usize,
) -> bool {
    // 1. Sharded Global Tensors
    if name.contains("embed_tokens") {
        // Only the first worker (entrypoint) needs the embedding table
        // Also load it if it's the last worker in case of tied weights
        return layer_start == 0 || layer_end >= total_layers;
    }

    if name.contains("model.norm.weight") || name.contains("lm_head") {
        // Only the last worker (final head) needs the final norm and LM head
        return layer_end >= total_layers;
    }

    // 2. Layer Tensors
    if let Some(layer_num) = extract_layer_index(name) {
        if layer_num < layer_start || layer_num >= layer_end {
            return false;
        }

        // MoE Expert Filtering
        if name.contains(".experts.")
            && let Some(expert_idx) = extract_expert_index(name)
            && let Some((start, end)) = expert_range
        {
            if expert_idx < start || expert_idx >= end {
                tracing::info!(
                    "⏭️ Skipping expert {} (outside assigned range [{}, {}))",
                    expert_idx,
                    start,
                    end
                );
                return false;
            }
            tracing::info!("📥 Loading expert {} for layer {}", expert_idx, layer_num);
        }

        // Always load everything else in the layer (attention, gate, shared expert, norms)
        return true;
    }

    false
}

/// Loads sharded weights from a list of safetensor files.
pub async fn load_sharded_weights(
    shard_files: &[std::path::PathBuf],
    layer_range: (usize, usize),
    expert_range: Option<(usize, usize)>,
    quantization: tessella_core::models::QuantizationConfig,
    device: &Device,
) -> anyhow::Result<ShardedWeights> {
    let (layer_start, layer_end) = layer_range;

    // We need to know the total layers to decide if we load the final head.
    let mut detected_total_layers = 0;

    let mut layers: HashMap<usize, HashMap<String, Weight>> = HashMap::new();
    let mut global: HashMap<String, Weight> = HashMap::new();

    info!(
        "📂 Loading shards for layers [{}, {}) from {} files",
        layer_start,
        layer_end,
        shard_files.len()
    );

    for shard_path in shard_files {
        let is_gguf = shard_path.extension().map(|e| e == "gguf").unwrap_or(false);
        if is_gguf {
            let mut file = std::fs::File::open(shard_path)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
            for (tensor_name, _) in content.tensor_infos.iter() {
                if let Some(layer_idx) = extract_layer_index(tensor_name)
                    && layer_idx + 1 > detected_total_layers
                {
                    detected_total_layers = layer_idx + 1;
                }
            }
        } else {
            let file = std::fs::File::open(shard_path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let st = SafeTensors::deserialize(&mmap)?;

            for (tensor_name, _) in st.tensors() {
                if let Some(layer_idx) = extract_layer_index(&tensor_name)
                    && layer_idx + 1 > detected_total_layers
                {
                    detected_total_layers = layer_idx + 1;
                }
            }
        }
    }

    if detected_total_layers == 0 {
        detected_total_layers = layer_end; // Fallback
    }

    for shard_path in shard_files {
        let mut loaded_count = 0;
        let mut total_bytes = 0;

        let is_gguf = shard_path.extension().map(|e| e == "gguf").unwrap_or(false);

        let mut process_weight = |tensor_name: &str,
                                  weight: Weight,
                                  layer_idx_opt: Option<usize>| {
            let tensor_bytes = match &weight {
                Weight::Tensor(t) => t.layout().shape().elem_count() * t.dtype().size_in_bytes(),
                Weight::Quantized(q) => q.storage_size_in_bytes(),
            };
            total_bytes += tensor_bytes;
            loaded_count += 1;

            if let Some(layer_idx) = layer_idx_opt {
                tracing::debug!(
                    "📥 Layer {}: Loaded {} ({:?}, {} MB)",
                    layer_idx,
                    tensor_name,
                    weight.dtype(),
                    tensor_bytes / tessella_core::constants::inference::MB_TO_BYTES
                );

                layers
                    .entry(layer_idx)
                    .or_default()
                    .insert(tensor_name.to_string(), weight);
            } else {
                global.insert(tensor_name.to_string(), weight);
            }
        };

        if is_gguf {
            // Native GGUF loading: keep tensors quantized
            let mut file = std::fs::File::open(shard_path)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

            let tensor_names: Vec<String> = content.tensor_infos.keys().cloned().collect();

            for tensor_name in tensor_names {
                if !should_load_tensor(
                    &tensor_name,
                    layer_start,
                    layer_end,
                    expert_range,
                    detected_total_layers,
                ) {
                    continue;
                }

                let qtensor = content.tensor(&mut file, &tensor_name, device)?;
                let weight = if tensor_name.contains("weight")
                    && !tensor_name.contains("norm")
                    && !tensor_name.contains("embed")
                {
                    // Only quantize actual weights of Linear layers
                    Weight::Quantized(Arc::new(qtensor))
                } else {
                    // Embeddings, norms, and biases (rare in GGUF) should be dequantized
                    let tensor = qtensor.dequantize(device)?;
                    let tensor = if device.is_cuda() {
                        tensor.to_dtype(candle_core::DType::F16)?
                    } else {
                        tensor.to_dtype(candle_core::DType::F32)?
                    };
                    Weight::Tensor(tensor)
                };

                let layer_idx = extract_layer_index(&tensor_name);
                process_weight(&tensor_name, weight, layer_idx);
            }
        } else {
            let file = std::fs::File::open(shard_path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let st = SafeTensors::deserialize(&mmap)?;

            for (tensor_name, tensor_view) in st.tensors() {
                if !should_load_tensor(
                    &tensor_name,
                    layer_start,
                    layer_end,
                    expert_range,
                    detected_total_layers,
                ) {
                    continue;
                }

                let tensor = tensor_view_to_candle(&tensor_view, device)?;
                let layer_idx = extract_layer_index(&tensor_name);
                process_weight(&tensor_name, Weight::Tensor(tensor), layer_idx);
            }
        }

        tracing::info!(
            "📊 Shard Load Summary: {} tensors, approx. {} MB total VRAM",
            loaded_count,
            total_bytes / tessella_core::constants::inference::MB_TO_BYTES
        );
    }

    let loaded_layers = layers.len();
    let loaded_tensors: usize = layers.values().map(|m| m.len()).sum::<usize>() + global.len();
    info!(
        "✅ Loaded {} layers, {} tensors total ({} global: embed_tokens={}, norm={}, lm_head={})",
        loaded_layers,
        loaded_tensors,
        global.len(),
        global.keys().any(|k| k.contains("embed_tokens")),
        global.keys().any(|k| k.contains("model.norm")),
        global.keys().any(|k| k.contains("lm_head")),
    );

    // Verify that the shard contains layers
    if layers.is_empty() && global.is_empty() {
        anyhow::bail!("No layers or global tensors found in the specified range");
    }

    // [DIAGNOSTIC] TENSOR NAME MAPPING
    let mut tensor_names: Vec<String> = Vec::new();
    if let Some(l0) = layers.get(&0) {
        tensor_names.extend(l0.keys().cloned());
    } else {
        tensor_names.extend(global.keys().cloned());
    }
    tensor_names.sort(); // for consistent output
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        info!("tensor[{}]: {}", i, name);
    }

    // [DIAGNOSTIC] DTYPE CHECK
    if let Some(l0) = layers.get(&0) {
        if let Some(w) = l0
            .get("model.layers.0.self_attn.q_proj.weight")
            .or_else(|| l0.values().next())
        {
            info!("dtype check: {:?}, shape: {:?}", w.dtype(), w.shape());
        }
    } else if let Some(w) = global.values().next() {
        info!(
            "dtype check (global): {:?}, shape: {:?}",
            w.dtype(),
            w.shape()
        );
    }

    Ok(ShardedWeights {
        layers,
        global,
        layer_range,
        expert_range,
        quantization,
    })
}

/// Extracts the expert index from a tensor name.
/// Supported patterns:
/// - model.layers.{N}.mlp.experts.{E}.*
/// - model.layers.{N}.block_sparse_moe.experts.{E}.*
/// - transformer.h.{N}.mlp.experts.{E}.*
fn extract_expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "experts"
            && let Some(idx_str) = parts.get(i + 1)
        {
            return idx_str.parse().ok();
        }
    }
    None
}

/// Extracts the layer index from a tensor name.
/// "model.layers.12.self_attn.q_proj.weight" → Some(12)
fn extract_layer_index(name: &str) -> Option<usize> {
    // Search for "layers.{N}." pattern in the name
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers"
            && let Some(idx_str) = parts.get(i + 1)
        {
            return idx_str.parse().ok();
        }
    }
    None
}

pub fn find_safetensor_files(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .map(|e| e == "safetensors" || e == "gguf")
            .unwrap_or(false)
        {
            files.push(path);
        }
    }
    files.sort(); // Deterministic order
    Ok(files)
}

/// Converts a safetensors TensorView to a Candle Tensor on the target device.
///
/// FIX: BF16 is read as real BF16 and then cast to F16 — NOT reinterpreted
/// BF16 bytes directly as F16 (bit layouts are different).
fn tensor_view_to_candle(
    view: &safetensors::tensor::TensorView<'_>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    use candle_core::DType;

    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();

    // F16 is the most universal 16-bit format for CUDA GPUs.
    // We convert to F16 on CPU to avoid VRAM spikes and kernel issues on older hardware.
    let target_dtype = if device.is_cuda() {
        DType::F16
    } else {
        DType::F32
    };

    let tensor = match view.dtype() {
        safetensors::Dtype::F32 => Tensor::from_raw_buffer(data, DType::F32, &shape, &Device::Cpu)?
            .to_dtype(target_dtype)?
            .to_device(device)?,
        safetensors::Dtype::F16 => Tensor::from_raw_buffer(data, DType::F16, &shape, &Device::Cpu)?
            .to_dtype(target_dtype)?
            .to_device(device)?,
        safetensors::Dtype::BF16 => {
            Tensor::from_raw_buffer(data, DType::BF16, &shape, &Device::Cpu)?
                .to_dtype(target_dtype)?
                .to_device(device)?
        }
        safetensors::Dtype::F64 => Tensor::from_raw_buffer(data, DType::F64, &shape, &Device::Cpu)?
            .to_dtype(target_dtype)?
            .to_device(device)?,
        other => anyhow::bail!(
            "Non-float dtype {:?} — not a weight tensor, skipping",
            other
        ),
    };

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(
            extract_layer_index("model.layers.12.self_attn.q_proj.weight"),
            Some(12)
        );
        assert_eq!(
            extract_layer_index("model.layers.0.mlp.up_proj.weight"),
            Some(0)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
    }

    #[test]
    fn test_find_safetensor_files_ignores_non_safetensors() {
        use std::fs;
        // Create temporary directory with files that are NOT safetensors
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.path().join("model.bin"), b"\x00\x01").unwrap();

        let files = find_safetensor_files(dir.path()).unwrap();
        assert!(files.is_empty(), "Should find no .safetensors files");
    }

    #[test]
    fn test_find_safetensor_files_finds_safetensors() {
        use std::fs;
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), b"\x00").unwrap();
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"\x00").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();

        let files = find_safetensor_files(dir.path()).unwrap();
        assert_eq!(files.len(), 2, "Should find exactly 2 .safetensors files");
    }

    #[test]
    fn test_moe_weight_loading() {
        use candle_core::{DType, Device};
        let device = Device::Cpu;

        let mut raw_weights = HashMap::new();
        // Layer 0: Dense
        raw_weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::zeros((16, 16), DType::F32, &device).unwrap(),
        );
        // Layer 1: MoE
        raw_weights.insert(
            "model.layers.1.mlp.gate.weight".to_string(),
            Tensor::zeros((4, 16), DType::F32, &device).unwrap(),
        );
        raw_weights.insert(
            "model.layers.1.mlp.experts.0.up_proj.weight".to_string(),
            Tensor::zeros((16, 16), DType::F32, &device).unwrap(),
        );
        raw_weights.insert(
            "model.layers.1.mlp.experts.1.up_proj.weight".to_string(),
            Tensor::zeros((16, 16), DType::F32, &device).unwrap(),
        );
        raw_weights.insert(
            "model.layers.1.mlp.shared_expert.up_proj.weight".to_string(),
            Tensor::zeros((16, 16), DType::F32, &device).unwrap(),
        );
        // Global
        raw_weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((100, 16), DType::F32, &device).unwrap(),
        );

        let start = 1;
        let end = 2;

        let mut layers = HashMap::new();
        let mut global = HashMap::new();
        for (name, tensor) in raw_weights {
            if should_load_tensor(&name, start, end, None, end) {
                if let Some(layer_idx) = extract_layer_index(&name) {
                    layers
                        .entry(layer_idx)
                        .or_insert(HashMap::new())
                        .insert(name, tensor);
                } else {
                    global.insert(name, tensor);
                }
            }
        }

        // Assertions
        assert!(global.contains_key("model.embed_tokens.weight"));
        assert!(!layers.contains_key(&0)); // Layer 0 is out of range (1..2)
        assert!(layers[&1].contains_key("model.layers.1.mlp.experts.0.up_proj.weight"));
        assert!(layers[&1].contains_key("model.layers.1.mlp.experts.1.up_proj.weight"));
        assert!(layers[&1].contains_key("model.layers.1.mlp.gate.weight"));
        assert!(layers[&1].contains_key("model.layers.1.mlp.shared_expert.up_proj.weight"));
    }
}
