# tessella-inference

> Candle-based tensor execution engine, safetensor shard loading, attention mechanisms, and activation serialization for the Qwen2 architecture family.

## Responsibility

`tessella-inference` executes the local neural network forward pass for the layer range assigned to a worker:
- **Architectural Scope:** Specialized exclusively for the **Qwen2 family** (`Qwen2ForCausalLM` dense and `Qwen2MoeForCausalLM` MoE), alongside the local `tessella-moe-micro` testing fixture.
- **Whole-Layer Execution:** Executes assigned layers sequentially using Hugging Face `candle` tensor kernels (CUDA / Metal).
- **Activation Serialization:** Packs and unpacks intermediate hidden state tensors via `HiddenStatePayload` with Blake3 checksum validation.
- **Attention & KV Caching:** Implements Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE) with absolute sequence offsets, and per-session KV cache buffers. *(The legacy global `KvCacheStore` was retired in `TASK-46`).*
- **Safetensor Shard Loading:** Inspects model index files and selectively memory-maps only the safetensor shards containing weights for the assigned layer range.
- **Sampling:** Applies temperature, top-p, top-k, and reproducible pseudo-random seed sampling to logits on the final pipeline shard.

## Modules

| File / Subsystem | Purpose | Key Types & Functions |
| :--- | :--- | :--- |
| `src/backends/` | Concrete model execution backends and dynamic backend registry. | `BackendRegistry`, `InferenceBackend`, `InferenceSession`, `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM` |
| `src/hidden_state.rs` | Serialization format for inter-hop activation tensors. | `HiddenStatePayload`, `Payload::from_tensor()`, `Payload::to_bytes()` |
| `src/shard_loader.rs` | Selective safetensors weight loading from local disk cache. | `load_sharded_weights()`, `find_safetensor_files()` |
| `src/attention.rs` | Attention layers, RoPE with absolute offset, and GQA. | `QwenAttention`, `RotaryEmbedding`, `get_causal_mask_with_offset()` |
| `src/sampling.rs` | Logit sampling and token selection. | `sample_next()`, `SamplingParams` |
| `src/generation.rs` | Autoregressive generation loop and EOS detection. | `resolve_eos_config()`, `EosConfig` |
| `src/moe.rs` | Mixture of Experts gating and block execution. | `MoeBlock`, `MoeGate` |
| `src/models.rs` | Quantized and unquantized linear layer wrappers. | `LinearLayer`, `QuantizedLinear` |
| `src/layer_factory.rs` | Quantization-aware layer instantiation. | `LayerFactory`, `QuantizationConfig` |
| `src/conversion.rs` | Tensor data type casting and formatting helpers. | DType conversion routines |

## Usage

```rust
use tessella_inference::{
    HiddenStatePayload,
    backends::{BackendRegistry, InferenceSession, InferenceBackend},
    sampling::SamplingParams,
};
```
