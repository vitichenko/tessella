use crate::TokenizerWrapper;
use candle_core::{Device, IndexOp, Tensor};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::future::Future;
use tessella_core::error::DomainError;
use tessella_core::models::ModelConfig;
use tracing::{debug, error};

pub struct EosConfig {
    pub ids: Vec<u32>,
}

impl EosConfig {
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.ids.contains(&token_id)
    }
}

/// Configuration for the generation loop.
///
/// The sampling fields are the **request's** (`TASK-40`/S46, D-16): they are mapped from the
/// client's body by the API handlers and honoured here. `0.0` temperature means greedy.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    /// Makes the generation reproducible when set. `None` = a fresh entropy-seeded RNG.
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.0, // Greedy by default
            top_p: 1.0,       // Disable nucleus sampling by default
            top_k: crate::sampling::SWARM_TOP_K as usize,
            seed: None,
            stop_sequences: Vec::new(),
        }
    }
}

impl Config {
    /// The [`crate::sampling::SamplingParams`] this loop samples with.
    #[must_use]
    pub fn sampling_params(&self) -> crate::sampling::SamplingParams {
        crate::sampling::SamplingParams {
            temperature: self.temperature as f32,
            top_p: self.top_p as f32,
            top_k: self.top_k,
        }
    }
}

/// Interactive distributed token generator.
#[derive(Clone)]
pub struct Loop {
    pub config: Config,
    pub model_config: ModelConfig,
    pub device: Device,
}

impl Loop {
    pub fn new(config: Config, model_config: ModelConfig, device: Device) -> Self {
        Self {
            config,
            model_config,
            device,
        }
    }

    /// Autoregressive generation loop.
    pub async fn run<P, Fut, C, CFut>(
        &self,
        prompt: &str,
        tokenizer: &TokenizerWrapper,
        pipeline_fn: P,
        token_callback: C,
        eos_ids: &[u32],
    ) -> Result<(String, crate::backends::GenerationMetrics), DomainError>
    where
        P: Fn(Tensor) -> Fut,
        Fut: Future<Output = Result<Tensor, DomainError>>,
        C: Fn(String) -> CFut,
        CFut: Future<Output = bool>,
    {
        let start_time = std::time::Instant::now();
        let mut ttft = 0.0;
        let mut tokens_generated = 0;

        // `TASK-40`/S46 (D-16): the loop samples with the parameters of its `Config` — which the API
        // maps from the client's body — and never with a literal. The RNG lives here so a `seed`
        // makes the whole generation reproducible (the old code drew from `thread_rng()` per step).
        let params = self.config.sampling_params();
        let mut rng: StdRng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // a. Tokenize prompt
        let tokenizer_cl = tokenizer.clone();
        let prompt_cl = prompt.to_string();
        let initial_ids = tokio::task::spawn_blocking(move || tokenizer_cl.encode(&prompt_cl))
            .await
            .map_err(|e| DomainError::Inference(format!("spawn_blocking error: {}", e)))??;
        let _prompt_tokens = initial_ids.len();
        let mut generated_text = String::new();
        let mut current_input_ids = initial_ids;

        // b. Loop up to max_new_tokens
        debug!(
            "Starting generation: max_tokens={}, device={:?}",
            self.config.max_new_tokens, self.device
        );
        for step in 0..self.config.max_new_tokens {
            // i. Prepare input tensor [1, seq_len]
            let input_tensor = Tensor::new(current_input_ids.as_slice(), &self.device)
                .map_err(|e| DomainError::Inference(format!("Failed to create input tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| {
                    DomainError::Inference(format!("Failed to unsqueeze input tensor: {e}"))
                })?;

            // ii. Call pipeline_fn
            let logits = pipeline_fn(input_tensor).await?;

            if step == 0 {
                ttft = start_time.elapsed().as_secs_f64() * 1000.0;
            }

            // iii. Extract last position logits
            let logits = match logits.dims().len() {
                3 => {
                    let seq_len = logits
                        .dim(1)
                        .map_err(|e| DomainError::Inference(e.to_string()))?;
                    logits
                        .i((0, seq_len - 1, ..))
                        .map_err(|e| DomainError::Inference(e.to_string()))?
                }
                2 => logits
                    .i(0)
                    .map_err(|e| DomainError::Inference(e.to_string()))?,
                1 => logits,
                _ => {
                    return Err(DomainError::Inference(format!(
                        "Unexpected logits shape: {:?}",
                        logits.dims()
                    )));
                }
            };

            // iv. Validate for NaN/Inf on device before sampling
            let has_nan = logits
                .ne(&logits)
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .max_all()
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .to_scalar::<u8>()
                .map_err(|e| DomainError::Inference(e.to_string()))?
                != 0;

            let has_inf = logits
                .to_dtype(candle_core::DType::F32)
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .abs()
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .max_all()
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .to_scalar::<f32>()
                .map_err(|e| DomainError::Inference(e.to_string()))?
                .is_infinite();

            // One draw per step — skipped when decoding is greedy, where it cannot matter.
            // Computing it here (and not inside `spawn_blocking`) is what keeps the RNG advancing
            // across steps instead of being reset by every clone of the loop.
            let uniform = if params.is_greedy() {
                0.0
            } else {
                rng.gen_range(0.0..1.0f32)
            };
            let device_cl = self.device.clone();
            let eos_ids_cl = eos_ids.to_vec();
            let logits_cl = logits.clone();
            let next_token = tokio::task::spawn_blocking(move || -> Result<u32, DomainError> {
                if has_nan || has_inf {
                    error!(
                        "Logits contain NaN or Inf (device={:?}). Returning EOS sentinel.",
                        device_cl
                    );
                    // Return sentinel EOS token instead of crashing. Use the first EOS ID if available.
                    if let Some(&eos_id) = eos_ids_cl.first() {
                        Ok(eos_id)
                    } else {
                        Err(DomainError::Inference(
                            "Logits contain NaN or Inf and no EOS token available".into(),
                        ))
                    }
                } else {
                    // v. Sample the next token with the **one** implementation of the stack.
                    //
                    // Until S46 (`TASK-40`, SP-07) this block hardcoded `temp = 0.2`, `top_k = 40`
                    // and `top_p = 0.9` and drew from `thread_rng()`, so the `temperature`/`top_p`
                    // the API had just mapped from the client's body were ignored (a request asking
                    // for `temperature = 0.0`, documented as greedy, still sampled at 0.2) and the
                    // `seed` of the node did not exist here at all. `params` and `uniform` are
                    // computed outside this closure (`spawn_blocking` cannot own the RNG of the
                    // loop without resetting its state on every step).
                    let logits_vec = logits_cl
                        .to_vec1::<f32>()
                        .map_err(|e| DomainError::Inference(e.to_string()))?;
                    crate::sampling::sample_next(&logits_vec, &params, uniform).ok_or_else(|| {
                        DomainError::Inference("the model produced no logits to sample from".into())
                    })
                }
            })
            .await
            .map_err(|e| DomainError::Inference(format!("spawn_blocking error: {}", e)))??;

            // v. Decode token
            tokens_generated += 1;
            let tokenizer_cl = tokenizer.clone();
            let decoded = tokio::task::spawn_blocking(move || tokenizer_cl.decode(&[next_token]))
                .await
                .map_err(|e| DomainError::Inference(format!("spawn_blocking error: {}", e)))??;
            generated_text.push_str(&decoded);
            if !token_callback(decoded).await {
                break;
            }

            // vi. Check stop conditions
            if eos_ids.contains(&next_token) {
                break;
            }

            let mut should_stop = false;
            for stop_seq in &self.config.stop_sequences {
                if generated_text.ends_with(stop_seq) {
                    should_stop = true;
                    break;
                }
            }
            if should_stop {
                break;
            }

            // vii. Update input for next step: only the next_token
            current_input_ids = vec![next_token];
        }

        let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let decode_time_ms = total_time_ms - ttft;
        let inter_token_latency = if tokens_generated > 1 {
            decode_time_ms / (tokens_generated - 1) as f64
        } else {
            0.0
        };
        let tps = (tokens_generated as f64) / (total_time_ms / 1000.0);

        let metrics = crate::backends::GenerationMetrics {
            ttft_ms: ttft,
            inter_token_latency_ms: inter_token_latency,
            total_time_ms,
            tokens_per_second: tps,
        };

        Ok((generated_text, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenizerWrapper;
    use candle_core::Device;

    #[tokio::test]
    async fn test_generation_greedy_is_deterministic() {
        let device = Device::Cpu;
        let config = Config {
            max_new_tokens: 3,
            temperature: 0.0,
            ..Default::default()
        };
        let gen_loop = Loop::new(config, ModelConfig::default(), device);
        let tokenizer = TokenizerWrapper::dummy();

        let pipeline = |input: Tensor| async move {
            let dev = input.device();
            let mut logits = vec![0.0f32; 3];
            logits[1] = 10.0; // Token 'a' (index 1) has highest prob
            Ok(Tensor::new(logits, dev).unwrap())
        };

        let (result1, _) = gen_loop
            .run("a", &tokenizer, pipeline, |_| async { true }, &[0])
            .await
            .unwrap();
        let (result2, _) = gen_loop
            .run("a", &tokenizer, pipeline, |_| async { true }, &[0])
            .await
            .unwrap();

        assert_eq!(result1, result2);
        assert!(result1.contains("a"));
    }

    #[tokio::test]
    async fn test_generation_stops_on_eos() {
        let device = Device::Cpu;
        let config = Config {
            max_new_tokens: 10,
            temperature: 0.0,
            ..Default::default()
        };
        let gen_loop = Loop::new(config, ModelConfig::default(), device);
        let tokenizer = TokenizerWrapper::dummy();

        let pipeline = |input: Tensor| async move {
            let dev = input.device();
            let mut logits = vec![0.0f32; 3];
            logits[0] = 10.0; // Token 0 is EOS
            Ok(Tensor::new(logits, dev).unwrap())
        };

        let (result, _) = gen_loop
            .run("test", &tokenizer, pipeline, |_| async { true }, &[0])
            .await
            .unwrap();
        // Since decode skips special tokens (skip_special_tokens=true),
        // the EOS token "<|endoftext|>" won't be in the result string.
        // We just verify it produced something (or not, if it was just EOS) and stopped.
        // In this mock, the first token is EOS, so result should be empty.
        assert_eq!(result, "");
    }

    /// `TASK-40`/S46 (D-16): `temperature = 0.0` is greedy **for real**.
    ///
    /// Before this step the loop hardcoded `0.2` and drew from `thread_rng()`, so a request asking
    /// for `temperature = 0.0` (what the §1bis gate, the harness and `tessella-cli` send, all of
    /// them documented as *deterministic*) got a **random** answer: S50's `sampling_repro` scenario
    /// measured it in vivo (two identical requests, two different texts). With the honoured
    /// parameters the answer is the argmax regardless of the seed.
    #[tokio::test]
    async fn test_generation_greedy_is_the_argmax_for_any_seed() {
        let tokenizer = TokenizerWrapper::dummy();
        let pipeline = |input: Tensor| async move {
            let dev = input.device();
            let mut logits = vec![0.0f32; 3];
            logits[1] = 3.0; // 'a' wins
            logits[2] = 2.999; // 'b' is a hair behind: a 0.2 sampler picks it ~40 % of the time
            Ok(Tensor::new(logits, dev).unwrap())
        };
        let greedy = |seed: Option<u64>| Config {
            max_new_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            top_k: crate::sampling::SWARM_TOP_K as usize,
            seed,
            stop_sequences: Vec::new(),
        };

        let (eleven, _) = Loop::new(greedy(Some(11)), ModelConfig::default(), Device::Cpu)
            .run("a", &tokenizer, pipeline, |_| async { true }, &[7])
            .await
            .unwrap();
        let (twenty_two, _) = Loop::new(greedy(Some(22)), ModelConfig::default(), Device::Cpu)
            .run("a", &tokenizer, pipeline, |_| async { true }, &[7])
            .await
            .unwrap();

        assert_eq!(eleven, twenty_two);
        assert_eq!(eleven, "aaaa", "greedy must always take the argmax ('a')");
    }

    /// The same request — same prompt, same parameters, same seed — gives the same text, which is
    /// what makes a test based on the answer possible at all.
    #[tokio::test]
    async fn test_generation_is_reproducible_with_a_seed() {
        let tokenizer = TokenizerWrapper::dummy();
        let pipeline = |input: Tensor| async move {
            let dev = input.device();
            let mut logits = vec![0.0f32; 3];
            logits[1] = 1.2;
            logits[2] = 1.1;
            Ok(Tensor::new(logits, dev).unwrap())
        };
        let sampled = |seed: Option<u64>| Config {
            max_new_tokens: 6,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed,
            stop_sequences: Vec::new(),
        };

        let (first, _) = Loop::new(sampled(Some(7)), ModelConfig::default(), Device::Cpu)
            .run("a", &tokenizer, pipeline, |_| async { true }, &[7])
            .await
            .unwrap();
        let (second, _) = Loop::new(sampled(Some(7)), ModelConfig::default(), Device::Cpu)
            .run("a", &tokenizer, pipeline, |_| async { true }, &[7])
            .await
            .unwrap();

        assert_eq!(first, second);
        assert!(first.contains('a') || first.contains('b'), "{first:?}");
    }

    /// The mapping from the loop's configuration to the sampler is the only translation between
    /// what the client asked for and what is applied.
    #[test]
    fn test_the_config_maps_its_sampling_parameters() {
        let params = Config {
            temperature: 0.7,
            top_p: 0.5,
            top_k: 8,
            ..Default::default()
        }
        .sampling_params();
        assert_eq!(
            (params.temperature, params.top_p, params.top_k),
            (0.7, 0.5, 8)
        );
        assert!(Config::default().sampling_params().is_greedy());
    }
}
