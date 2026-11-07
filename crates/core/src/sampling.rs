//! The **single** sampling implementation of the stack (`TASK-40`/S46, SP-07 · decision D-16).
//!
//! Moved to `tessella-core` as part of `SOLID-05` to decouple presentation layers from `tessella-inference`.

pub use tessella_protocol::{SWARM_TEMPERATURE, SWARM_TOP_K, SWARM_TOP_P};

/// The sampling parameters of **one** request.
///
/// The randomness is an explicit input of [`sample_next`] (`uniform`), not an RNG hidden inside:
/// the sampler is pure, so a test can pin a draw and the caller owns reproducibility (the `seed`
/// of a request, or of a node) without threading a generator through `spawn_blocking`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// `0.0` (or less) means **greedy**: the highest logit wins, deterministically.
    pub temperature: f32,
    /// Nucleus-sampling cutoff. `>= 1.0` disables it.
    pub top_p: f32,
    /// Candidate-set size. `0` disables the truncation.
    pub top_k: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::swarm_standard()
    }
}

impl SamplingParams {
    /// The mesh's standard: what a request gets when it does not ask for anything else. Reading
    /// these numbers out of a node's config is what used to expel it from the swarm.
    #[must_use]
    pub const fn swarm_standard() -> Self {
        Self {
            temperature: SWARM_TEMPERATURE,
            top_p: SWARM_TOP_P,
            top_k: SWARM_TOP_K as usize,
        }
    }

    /// Greedy decoding: always the argmax, no randomness involved.
    #[must_use]
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        }
    }

    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    #[must_use]
    pub fn is_swarm_standard(&self) -> bool {
        *self == Self::swarm_standard()
    }

    /// Builds the parameters of one OpenAI-shaped request (`D-16`).
    ///
    /// Omission (`None`) means **the swarm standard**, never a rejection: the §1bis gate, the
    /// harness and `tessella-cli` have always sent their own value, and a client that says nothing
    /// must keep getting what this mesh is calibrated for. A value that cannot be sampled at all
    /// (negative/NaN/infinite temperature, `top_p` outside `(0, 1]`) is **rejected with the
    /// reason** instead of being silently replaced: accepting a parameter and ignoring it is the
    /// defect this task is about.
    ///
    /// # Errors
    ///
    /// Returns the human-readable reason for a value that cannot be honoured.
    pub fn from_request(
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> Result<Self, String> {
        let standard = Self::swarm_standard();
        if let Some(raw) = temperature
            && (!raw.is_finite() || raw < 0.0)
        {
            return Err(format!(
                "temperature must be a finite value >= 0 (got {raw}); omit it to use the swarm \
                 standard {}",
                standard.temperature
            ));
        }
        if let Some(raw) = top_p
            && (!raw.is_finite() || raw <= 0.0 || raw > 1.0)
        {
            return Err(format!(
                "top_p must be in (0, 1] (got {raw}); omit it to use the swarm standard {}",
                standard.top_p
            ));
        }
        Ok(Self {
            temperature: temperature.unwrap_or(standard.temperature as f64) as f32,
            top_p: top_p.unwrap_or(standard.top_p as f64) as f32,
            // A `top_k` bigger than any vocabulary is not an error: it simply does not truncate.
            top_k: top_k.unwrap_or(standard.top_k),
        })
    }
}

/// Samples one token: the **only** place in the stack where a logit vector becomes a token.
///
/// `uniform` is the caller's draw in `[0, 1]` (ignored when decoding is greedy). Returns `None`
/// for an empty logit vector.
#[must_use]
pub fn sample_next(logits: &[f32], params: &SamplingParams, uniform: f32) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }

    // Greedy: deterministic argmax, ties broken by the lowest index. A NaN never wins (every
    // comparison against it is false), which is why the running best starts at -inf instead of at
    // `logits[0]`: with a NaN in the first position the old shape returned it as the "maximum".
    if params.is_greedy() {
        let mut best = 0usize;
        let mut best_logit = f32::NEG_INFINITY;
        for (index, &logit) in logits.iter().enumerate() {
            if logit > best_logit {
                best_logit = logit;
                best = index;
            }
        }
        return u32::try_from(best).ok();
    }

    // Temperature, then the candidate set (top-k, then the nucleus), in the same order the retired
    // copies used: probabilities are softmaxed **after** the top-k truncation.
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (index, logit / params.temperature))
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if params.top_k > 0 {
        candidates.truncate(params.top_k);
    }

    let max_logit = candidates.first().map(|(_, logit)| *logit).unwrap_or(0.0);
    let exps: Vec<(usize, f32)> = candidates
        .into_iter()
        .map(|(index, logit)| (index, (logit - max_logit).exp()))
        .collect();
    let sum: f32 = exps.iter().map(|(_, value)| *value).sum();
    if !sum.is_finite() || sum <= 0.0 {
        // A degenerate distribution still has an answer: its first candidate (the truncated
        // argmax), which keeps the sampler total instead of panicking on a softmax edge case.
        return exps.first().map(|(index, _)| *index as u32);
    }
    let mut probabilities: Vec<(usize, f32)> = exps
        .into_iter()
        .map(|(index, value)| (index, value / sum))
        .collect();

    if params.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut keep = probabilities.len();
        for (index, (_, probability)) in probabilities.iter().enumerate() {
            cumulative += probability;
            if cumulative >= params.top_p {
                keep = index + 1;
                break;
            }
        }
        probabilities.truncate(keep.max(1));
    }

    let total: f32 = probabilities
        .iter()
        .map(|(_, probability)| *probability)
        .sum();
    if total > 0.0 {
        for (_, probability) in probabilities.iter_mut() {
            *probability /= total;
        }
    }

    let draw = uniform.clamp(0.0, 1.0);
    let mut running = 0.0f32;
    let mut token = probabilities.last().map(|(index, _)| *index as u32);
    for (index, probability) in &probabilities {
        running += probability;
        if draw <= running {
            token = Some(*index as u32);
            break;
        }
    }
    token
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `temperature = 0.0` is greedy **by definition**, whatever the draw is.
    #[test]
    fn test_greedy_is_the_argmax_and_ignores_the_draw() {
        let logits = [0.1f32, 5.0, 0.2];
        assert_eq!(
            sample_next(&logits, &SamplingParams::greedy(), 0.0),
            Some(1)
        );
        assert_eq!(
            sample_next(&logits, &SamplingParams::greedy(), 0.999_999),
            Some(1)
        );
    }
}
