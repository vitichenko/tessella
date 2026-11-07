use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tessella_core::models::ModelConfig;

pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(rope_theta: f64, head_dim: usize, max_pos: usize, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let theta = Tensor::new(theta, device)?;
        let idx_theta = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        // Concatenate idx_theta with itself to match head_dim
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], 1)?;

        Ok(Self {
            sin: idx_theta.sin()?,
            cos: idx_theta.cos()?,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;

        let q_embed = self.apply_rope(q, &cos, &sin)?;
        let k_embed = self.apply_rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_b_sz, _n_heads, seq_len, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;

        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        let cos = cos.reshape((1, 1, seq_len, head_dim))?;
        let sin = sin.reshape((1, 1, seq_len, head_dim))?;

        let cos1 = cos.narrow(3, 0, half_dim)?;
        let cos2 = cos.narrow(3, half_dim, half_dim)?;
        let sin1 = sin.narrow(3, 0, half_dim)?;
        let sin2 = sin.narrow(3, half_dim, half_dim)?;

        // HF RoPE formula:
        // out_1 = x1 * cos1 - x2 * sin1
        // out_2 = x2 * cos2 + x1 * sin2
        let out1 = (x1.broadcast_mul(&cos1)? - x2.broadcast_mul(&sin1)?)?;
        let out2 = (x2.broadcast_mul(&cos2)? + x1.broadcast_mul(&sin2)?)?;

        Tensor::cat(&[out1, out2], 3)
    }
}

pub struct KvCacheEntry {
    pub k: Tensor,
    pub v: Tensor,
    pub last_accessed: Instant,
}

// `S52`/`TASK-46`: a `KvCacheStore` used to live here — a shared `DashMap<(request_id, layer), entry>`
// with an LRU bound and a TTL, created per worker and handed to `ApiState`. **Nothing ever inserted
// into it**: the only KV cache that exists is the one each `qwen2` session keeps for its own local
// layers (`backends/qwen2.rs`), with `inference_engine` bounding the sessions themselves. Its LRU and
// its TTL walked an always-empty map, `get_stats()` had no consumer and the two keys that tuned it
// (`kv_cache_max_entries`, `kv_cache_ttl_secs`) were read to build it and nothing else. It was retired
// instead of kept «for later»: a second truth for the same data is the defect this card closes, and
// the card that wants a shared, byte-bounded cache has to design it with the backend that fills it.

pub fn get_causal_mask_with_offset(
    seq_len_q: usize,
    seq_len_k: usize,
    seq_start_pos: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len_q)
        .flat_map(|i| {
            (0..seq_len_k).map(move |j| {
                if j > seq_start_pos + i {
                    f32::MIN
                } else {
                    0f32
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (seq_len_q, seq_len_k), device)
}

pub struct QwenAttention {
    pub q_proj: crate::models::LinearLayer,
    pub k_proj: crate::models::LinearLayer,
    pub v_proj: crate::models::LinearLayer,
    pub o_proj: crate::models::LinearLayer,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope: Arc<RotaryEmbedding>,
}

impl QwenAttention {
    /// Creates a new QwenAttention layer.
    ///
    /// # Errors
    /// Returns `candle_core::Error` if linear layer creation fails.
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        rope: Arc<RotaryEmbedding>,
        weights: Option<&HashMap<String, crate::shard_loader::Weight>>,
        factory: &crate::LayerFactory,
    ) -> std::result::Result<Self, tessella_core::error::DomainError> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let (q_proj, k_proj, v_proj, o_proj) = if let Some(w_map) = weights {
            let q = factory.linear("q_proj", w_map)?;
            let k = factory.linear("k_proj", w_map)?;
            let v = factory.linear("v_proj", w_map)?;
            let o = factory.linear("o_proj", w_map)?;
            (q, k, v, o)
        } else {
            let n_kv_heads = cfg.num_key_value_heads;
            let q_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))
                .map_err(|e| tessella_core::error::DomainError::Inference(e.to_string()))?;
            let k_proj = candle_nn::linear(cfg.hidden_size, n_kv_heads * head_dim, vb.pp("k_proj"))
                .map_err(|e| tessella_core::error::DomainError::Inference(e.to_string()))?;
            let v_proj = candle_nn::linear(cfg.hidden_size, n_kv_heads * head_dim, vb.pp("v_proj"))
                .map_err(|e| tessella_core::error::DomainError::Inference(e.to_string()))?;
            let o_proj =
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))
                    .map_err(|e| tessella_core::error::DomainError::Inference(e.to_string()))?;
            (
                crate::models::LinearLayer::Full(q_proj),
                crate::models::LinearLayer::Full(k_proj),
                crate::models::LinearLayer::Full(v_proj),
                crate::models::LinearLayer::Full(o_proj),
            )
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope,
        })
    }

    /// Forward pass for QwenAttention with KV-cache and RoPE support.
    /// Returns (output, k, v).
    ///
    /// # Errors
    /// Returns `candle_core::Error` if matrix multiplications or reshapes fail.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut KvCacheEntry>,
        position_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _hidden_size) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v_final = v
            .reshape((batch_size, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE
        let (q, mut k_final) = self.rope.apply(&q, &k, position_offset)?;

        // Apply KV-cache
        if let Some(cache_entry) = cache {
            k_final = Tensor::cat(&[&cache_entry.k, &k_final], 2)?;
            v_final = Tensor::cat(&[&cache_entry.v, &v_final], 2)?;
            cache_entry.k = k_final.clone();
            cache_entry.v = v_final.clone();
            cache_entry.last_accessed = Instant::now();
        }

        let (k_rep, v_rep) = if self.n_heads != self.n_kv_heads {
            let n_rep = self.n_heads / self.n_kv_heads;
            let k_rep = repeat_kv(&k_final, n_rep)?;
            let v_rep = repeat_kv(&v_final, n_rep)?;
            (k_rep, v_rep)
        } else {
            (k_final.clone(), v_final.clone())
        };

        let scale = 1.0 / f64::sqrt(self.head_dim as f64);
        let att = q.matmul(&k_rep.transpose(2, 3)?)?;
        let att = (att * scale)?;

        let att = match mask {
            Some(m) => {
                let m = m.to_dtype(q.dtype())?;
                att.broadcast_add(&m)?
            }
            None => att,
        };

        // Numerical stability: softmax in F32
        let att = att.to_dtype(DType::F32)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let att = att.to_dtype(q.dtype())?;

        let out = att.matmul(&v_rep)?;

        let out =
            out.transpose(1, 2)?
                .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;
        let output = self.o_proj.forward(&out)?;

        Ok((output, k_final, v_final))
    }
}

/// Repeats key/value heads for Grouped Query Attention.
///
/// # Errors
/// Returns `candle_core::Error` if tensor operations fail.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b_sz, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b_sz, n_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_heads * n_rep, seq_len, head_dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_qwen_attention_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            model_id: "test".to_string(),
            model_type: Some("qwen2".to_string()),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hidden_size: 5120,
            num_attention_heads: 40,
            num_key_value_heads: 8, // GQA factor 5
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let rope = Arc::new(RotaryEmbedding::new(
            10000.0,
            cfg.hidden_size / cfg.num_attention_heads,
            2048,
            &device,
        )?);
        let factory = crate::layer_factory::LayerFactory::new(
            tessella_core::models::QuantizationConfig::None,
        );
        let attn = QwenAttention::new(&cfg, vb, rope, None, &factory)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let x = Tensor::zeros((1, 4, 5120), DType::F32, &device)?;
        let (out, _k, _v) = attn.forward(&x, None, None, 0)?;

        assert_eq!(out.dims(), &[1, 4, 5120]);
        Ok(())
    }

    #[test]
    fn test_kv_cache_concatenation() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCacheEntry {
            k: Tensor::zeros((1, 8, 4, 128), DType::F32, &device)?,
            v: Tensor::zeros((1, 8, 4, 128), DType::F32, &device)?,
            last_accessed: Instant::now(),
        };

        let cfg = ModelConfig {
            model_id: "test".to_string(),
            model_type: Some("qwen2".to_string()),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hidden_size: 1024,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let rope = Arc::new(RotaryEmbedding::new(
            10000.0,
            cfg.hidden_size / cfg.num_attention_heads,
            2048,
            &device,
        )?);
        let factory = crate::layer_factory::LayerFactory::new(
            tessella_core::models::QuantizationConfig::None,
        );
        let attn = QwenAttention::new(&cfg, vb, rope, None, &factory)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Forward with 1 token
        let x = Tensor::zeros((1, 1, 1024), DType::F32, &device)?;
        let (_out, k_final, v_final) = attn.forward(&x, None, Some(&mut cache), 4)?;

        // Should now have 5 tokens in cache (4 initial + 1 new)
        assert_eq!(k_final.dims()[2], 5);
        assert_eq!(v_final.dims()[2], 5);
        assert_eq!(cache.k.dims()[2], 5);

        Ok(())
    }

    #[test]
    fn test_rope_offset() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(10000.0, 128, 1024, &device)?;

        let q = Tensor::zeros((1, 8, 1, 128), DType::F32, &device)?;
        let k = Tensor::zeros((1, 8, 1, 128), DType::F32, &device)?;

        // RoPE at pos 512 with offset 0
        let (_, k_pos512) = rope.apply(&q, &k, 512)?;

        // RoPE at pos 0 with offset 512
        let (_, k_offset512) = rope.apply(&q, &k, 512)?;

        // Should be identical
        let diff = (k_pos512 - k_offset512)?
            .abs()?
            .sum_all()?
            .to_vec0::<f32>()?;
        assert!(diff < 1e-6);

        Ok(())
    }
}
