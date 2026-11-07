use crate::error::DomainError;
use crate::models::GenerationMetrics;
use crate::sampling::{self, SamplingParams};
use crate::traits::{DistributedSession, TokenizerService};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::future::Future;
use tracing::{debug, error};

/// Configuration for autoregressive token generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.0, // Greedy by default
            top_p: 1.0,       // Disable nucleus sampling by default
            top_k: sampling::SWARM_TOP_K as usize,
            seed: None,
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationConfig {
    /// Returns the sampling parameters for this configuration.
    #[must_use]
    pub fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature as f32,
            top_p: self.top_p as f32,
            top_k: self.top_k,
        }
    }
}

/// Runs the standard autoregressive generation loop over a distributed session.
///
/// Decoupled from concrete tensors or frameworks: operates strictly over
/// `DistributedSession::forward` (`Vec<u32>` -> `ForwardOutput { logits }`)
/// and [`TokenizerService`].
pub async fn run_generation_loop<C, CFut>(
    session: &mut dyn DistributedSession,
    tokenizer: &dyn TokenizerService,
    config: &GenerationConfig,
    prompt: &str,
    token_callback: C,
    eos_ids: &[u32],
) -> Result<(String, GenerationMetrics), DomainError>
where
    C: Fn(String) -> CFut,
    CFut: Future<Output = bool>,
{
    let start_time = std::time::Instant::now();
    let mut ttft = 0.0;
    let mut tokens_generated = 0;

    let params = config.sampling_params();
    let mut rng: StdRng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let initial_ids = tokenizer.encode(prompt)?;
    let mut generated_text = String::new();
    let mut current_input_ids = initial_ids;

    debug!("Starting generation: max_tokens={}", config.max_new_tokens);
    for step in 0..config.max_new_tokens {
        let output = session.forward(current_input_ids).await?;

        if step == 0 {
            ttft = start_time.elapsed().as_secs_f64() * 1000.0;
        }

        let logits_vec = output
            .logits
            .ok_or_else(|| DomainError::Inference("No logits in ForwardOutput".into()))?;

        let has_nan = logits_vec.iter().any(|x| x.is_nan());
        let has_inf = logits_vec.iter().any(|x| x.is_infinite());

        let uniform = if params.is_greedy() {
            0.0
        } else {
            rng.gen_range(0.0..1.0f32)
        };

        let next_token = if has_nan || has_inf {
            error!("Logits contain NaN or Inf. Returning EOS sentinel.");
            if let Some(&eos_id) = eos_ids.first() {
                eos_id
            } else {
                return Err(DomainError::Inference(
                    "Logits contain NaN or Inf and no EOS token available".into(),
                ));
            }
        } else {
            sampling::sample_next(&logits_vec, &params, uniform).ok_or_else(|| {
                DomainError::Inference("the model produced no logits to sample from".into())
            })?
        };

        tokens_generated += 1;
        let decoded = tokenizer.decode(&[next_token])?;
        generated_text.push_str(&decoded);
        if !token_callback(decoded).await {
            break;
        }

        if eos_ids.contains(&next_token) {
            break;
        }

        let mut should_stop = false;
        for stop_seq in &config.stop_sequences {
            if generated_text.ends_with(stop_seq) {
                should_stop = true;
                break;
            }
        }
        if should_stop {
            break;
        }

        current_input_ids = vec![next_token];
    }

    let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let decode_time_ms = total_time_ms - ttft;
    let inter_token_latency = if tokens_generated > 1 {
        decode_time_ms / (tokens_generated - 1) as f64
    } else {
        0.0
    };
    let tps = if total_time_ms > 0.0 {
        (tokens_generated as f64) / (total_time_ms / 1000.0)
    } else {
        0.0
    };

    let metrics = GenerationMetrics {
        ttft_ms: ttft,
        inter_token_latency_ms: inter_token_latency,
        total_time_ms,
        tokens_per_second: tps,
    };

    Ok((generated_text, metrics))
}
