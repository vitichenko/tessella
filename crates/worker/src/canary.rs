//! The operator's **canary** replica (`TASK-71` · S78 · decision `D-28`).
//!
//! The compute audit of capa 3 compares the marked step of a sampled session across replicas
//! (`TASK-53`/S59 → `TASK-70`/S77) and pays the credit from what the **auditor** reports — and the
//! auditor is the entry node, the very node whose work nobody could check. A report saying «no
//! divergence» credited exactly what S13 credited, so a *lazy validator* had nothing to lose.
//!
//! The canary is the cheapest way to check the checker: a node of the operator that answers a step
//! with a **well-formed but wrong** tensor. Because the orchestrator knows the canary's answer is
//! false **without executing the model** (a stack of layers is never the identity), a report that
//! does not show that divergence is not evidence — the auditor loses the session's reward and takes
//! a reputational incident.
//!
//! What this module implements is the *answer*, not a whole deployment: [`CanarySession`] is an
//! `InferenceSession` that never touches a weight. `InferenceEngine::initialize_models` stops
//! before creating a backend and `NetworkManager::acquire_weights` returns an empty shard when the
//! node is a canary, so a canary loads no weights and uses no VRAM for them.

use candle_core::Tensor;
use std::path::PathBuf;
use tessella_core::error::DomainError;
use tessella_core::models::ModelConfig;
use tessella_inference::backends::InferenceSession;
use tracing::info;

use crate::config::WorkerConfig;

/// The factor `CanaryMode::Scale` multiplies the hidden state by.
///
/// It is deliberately far from `1.0`: the audit compares the `sha256` of the tensor bytes, so *any*
/// difference is detected, but a mesh whose client reads a nearly-correct answer would hide the
/// poison until the digest is compared. A factor of two is unambiguous in every dtype and cannot be
/// mistaken for numerical noise.
pub const CANARY_SCALE_FACTOR: f32 = 2.0;

/// The session a canary node hands to the engine for every step it is asked to serve.
///
/// It has no KV cache, no weights and no backend: it takes the hidden state it received and answers
/// with the same tensor transformed in a known-wrong way. The *shape* and the *dtype* are preserved
/// on purpose — a wrong shape would be a protocol error that no audit is needed to catch, and the
/// point of the canary is to be indistinguishable from a hop by everything except its answer.
pub struct CanarySession {
    mode: crate::config::CanaryMode,
    pos_offset: usize,
}

impl CanarySession {
    #[must_use]
    pub fn new(mode: crate::config::CanaryMode) -> Self {
        Self {
            mode,
            pos_offset: 0,
        }
    }

    /// Applies the canary's known-wrong transformation to a hidden state.
    ///
    /// `Echo` returns the tensor **unchanged** — a real stack of layers never does, which is exactly
    /// why the orchestrator can trust its own judgement about the answer. `Scale` multiplies every
    /// element by [`CANARY_SCALE_FACTOR`]. Both preserve shape and dtype, and neither needs a single
    /// weight.
    pub fn mis_transform(&self, x: Tensor) -> Result<Tensor, DomainError> {
        use crate::config::CanaryMode;
        match self.mode {
            CanaryMode::Off => Err(DomainError::Inference(
                "CanarySession created for a node whose canary mode is off: this is a bug in the \
                 worker's boot (TASK-71)"
                    .to_string(),
            )),
            CanaryMode::Echo => Ok(x),
            CanaryMode::Scale => x
                .affine(f64::from(CANARY_SCALE_FACTOR), 0.0)
                .map_err(|e| DomainError::Inference(format!("canary scale failed: {e}"))),
        }
    }
}

impl InferenceSession for CanarySession {
    fn forward(&mut self, _tokens: &[u32]) -> Result<Tensor, DomainError> {
        // A canary is a **replica hop**, never an entry node: the full-pipeline path needs the
        // embedding table and the sampling loop, and a node that answers every layer with a wrong
        // tensor cannot serve a client. Failing by name is the honest answer (`D-28`), and it is what
        // keeps the canary from silently poisoning sessions it can never serve.
        Err(DomainError::Validation(
            "this node is the operator's canary (`TESSELLA_CANARY_MODE`): it answers `/v1/forward` \
             with a known-wrong tensor on purpose and cannot serve an entry-node session, which \
             needs the embedding table and the model it deliberately does not load (TASK-71/S78)"
                .to_string(),
        ))
    }

    fn apply_layers(
        &mut self,
        x: Tensor,
        range: (usize, usize),
        seq_start_pos: usize,
    ) -> Result<Tensor, DomainError> {
        tracing::warn!(
            "🐤 CANARY STEP: this node answers layers {range:?} at position {seq_start_pos} with a \
             known-wrong tensor on purpose ({:?}). Nothing of this answer is computed (TASK-71/S78)",
            self.mode
        );
        self.mis_transform(x)
    }

    fn reset(&mut self) {
        self.pos_offset = 0;
    }

    fn set_pos_offset(&mut self, pos: usize) {
        self.pos_offset = pos;
    }

    fn get_pos_offset(&self) -> usize {
        self.pos_offset
    }
}

// ─── El arranque sin modelo (`TASK-82`/S89) ────────────────────────────────────────────────────

/// What the boot of a node knows about its model, before a single weight is read (`TASK-82`/S89).
///
/// The shape is deliberately the **same** for the two kinds of node: what changes is the content,
/// never the path the boot takes. That is what keeps the canary from being a second boot code path
/// that drifts.
#[derive(Debug)]
pub struct BootModel {
    /// Where the model of this node lives. For a canary it is the operator's `models_dir`
    /// **verbatim**: nothing is resolved inside it and nothing is created — the directory need not
    /// even exist.
    pub model_dir: PathBuf,
    /// The tokenizer to load; `None` for a canary, which has no tokenizer to load and whose
    /// entry-node path refuses by name (`TASK-71`/S78).
    pub tokenizer_path: Option<PathBuf>,
    /// The metadata the boot may state. For a canary it is a *placeholder*: see
    /// [`canary_model_config`].
    pub model_config: ModelConfig,
}

/// Boots the model side of a node: what it reads, what it does not, and what it may state.
///
/// `TASK-82`/S89: a canary boots with **no model directory** — it reads no `config.json`, downloads
/// nothing, loads no tokenizer and uses no VRAM for weights (the last two are `TASK-71`/S78). The
/// `config.json` and the tokenizer it used to drag along were why the harness had to provision a
/// model directory for the canary: a directory that made the operator pay for a model whose entire
/// point is not to have one, and the one thing a canary must never do is resolve it — a
/// `models_dir` that gets filled is indistinguishable, from the disk, from a node that *serves* the
/// model for real.
///
/// An ordinary node is untouched: it resolves its model (downloading it if the cache is cold) and
/// reads `config.json`, exactly as before.
///
/// # Errors
/// A canary whose model the layer table cannot look up aborts the boot naming `num_hidden_layers`
/// (see [`canary_model_config`]); an ordinary node keeps failing with whatever
/// [`NetworkManager::prepare_model_metadata`](crate::network_manager::NetworkManager::prepare_model_metadata)
/// fails with.
pub async fn boot_model(config: &WorkerConfig) -> anyhow::Result<BootModel> {
    if !config.canary_mode.is_on() {
        let (model_dir, tokenizer_path, model_config) =
            crate::network_manager::NetworkManager::prepare_model_metadata(
                &config.default_model,
                config,
            )
            .await?;
        return Ok(BootModel {
            model_dir,
            tokenizer_path: Some(tokenizer_path),
            model_config,
        });
    }

    let model_config = canary_model_config(config)?;
    info!(
        "🐤 Canary node ({:?}): booting with no model directory — no `config.json`, no tokenizer, \
         no weight, no VRAM for weights. It states {} layers of `{}` and no shape, and answers \
         every step with a known-wrong tensor (TASK-82/S89, TASK-71/S78)",
        config.canary_mode, model_config.num_hidden_layers, model_config.model_id
    );
    Ok(BootModel {
        // The declared directory, unread. It travels because the engine and the cache-prune keep
        // their own paths from it; nothing of the canary ever writes inside it.
        model_dir: PathBuf::from(&config.models_dir),
        tokenizer_path: None,
        model_config,
    })
}

/// The metadata a node **without a model** is allowed to state (`TASK-82`/S89).
///
/// Two fields and no more. `model_id` is the model the operator declared — what this node would be
/// serving. `num_hidden_layers` is the count it validates its range against, read from the same
/// table the tracker plans with ([`tessella_protocol::get_total_layers`]), so the two can never
/// disagree about a model they share.
///
/// Every other field stays at its default, and that is the point: a shape nobody read is stated as
/// unknown rather than invented. The consequence is deliberate — with no `hidden_size` there is no
/// hop ceiling ([`crate::inference_engine::hop_token_ceiling`] answers `None`) — because a canary
/// cannot say how much traffic fits in a layer it does not have, and a *wrong* number there would be
/// worse than none: the mesh would size hops against a budget that does not exist.
///
/// The layer count, unlike the shape, is **not** optional. A canary serves a layer range like any
/// node and validates it like any node, so a model the table cannot look up aborts the boot naming
/// `num_hidden_layers` — the field it is missing — instead of assuming a number for it.
///
/// # Errors
/// Returns an error naming `num_hidden_layers` and the model when the table cannot look the model up.
fn canary_model_config(config: &WorkerConfig) -> anyhow::Result<ModelConfig> {
    let model_id = config.default_model.as_str();
    let num_hidden_layers = tessella_protocol::get_total_layers(model_id).ok_or_else(|| {
        anyhow::anyhow!(
            "Missing num_hidden_layers in models/{model_id}/config.json, and the layer table of \
             the protocol does not know `{model_id}` either: a canary serves a layer range and \
             cannot validate it without the layer count of its model (TASK-82/S89)"
        )
    })?;
    Ok(ModelConfig {
        model_id: model_id.to_string(),
        num_hidden_layers,
        // `TASK-82`/S89, and the reason `no_shape()` exists: `ModelConfig::default()` is
        // `qwen2_5_0_5b()` — a **real** shape of a **real** model. Inheriting it here would make a
        // canary of any model publish 896 columns nothing read, and every consumer of a width (the hop
        // ceiling of `inference_engine`) would use it as if it were a measurement.
        ..ModelConfig::no_shape()
    })
}

/// What a node states about its model in its registration and at `/status` (`TASK-82`/S89).
///
/// A canary states **nothing**: it holds no shape, so there is no `ModelConfig` to send. The
/// orchestrator then plans with the same table it uses for every node that registers without one,
/// and `total_layers` (below, in `register_and_assign`) carries the single number a canary *can*
/// state.
///
/// An ordinary node states its own config, unmodified — this function is the one place that decides,
/// so the registration and `/status` cannot disagree about which node holds a model.
#[must_use]
pub fn registration_model_config(
    config: &WorkerConfig,
    model: Option<&ModelConfig>,
) -> Option<ModelConfig> {
    if config.canary_mode.is_on() {
        return None;
    }
    model.cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CanaryMode;
    use candle_core::Device;

    fn hidden(values: &[f32], dtype: candle_core::DType) -> Tensor {
        Tensor::from_vec(values.to_vec(), (1, values.len(), 1), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    /// The property the orchestrator relies on: the canary's answer is **always wrong**. `Echo`
    /// returns the input (a real layer never does) and `Scale` multiplies it by a factor nobody can
    /// mistake for the identity.
    #[test]
    fn the_canary_never_computes_a_layer() {
        let input = hidden(&[1.0, 2.0, 3.0], candle_core::DType::F32);

        let echoed = CanarySession::new(CanaryMode::Echo)
            .mis_transform(input.clone())
            .unwrap();
        assert_eq!(
            echoed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0],
            "echo answers the input unchanged: a stack of layers is never the identity"
        );

        let scaled = CanarySession::new(CanaryMode::Scale)
            .mis_transform(input.clone())
            .unwrap();
        let scaled = scaled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(scaled, vec![2.0, 4.0, 6.0]);
        assert_ne!(
            scaled,
            input.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the whole point of the canary is that its answer differs from an honest one"
        );
    }

    /// **Shape and dtype are preserved in both modes** (`F16` is what the hop path carries). The
    /// audit compares the `sha256` of the tensor bytes, so a wrong dtype would be a protocol error
    /// instead of the silent divergence the canary exists to produce.
    #[test]
    fn the_canary_answer_has_the_shape_and_dtype_of_a_real_one() {
        for dtype in [candle_core::DType::F16, candle_core::DType::F32] {
            let input = hidden(&[1.0, -2.0, 0.5, 4.0], dtype);
            for mode in [CanaryMode::Echo, CanaryMode::Scale] {
                let out = CanarySession::new(mode)
                    .mis_transform(input.clone())
                    .unwrap();
                assert_eq!(out.dims(), input.dims(), "{mode:?} changed the shape");
                assert_eq!(out.dtype(), input.dtype(), "{mode:?} changed the dtype");
            }
        }
    }

    /// A node whose canary mode is `off` has no `CanarySession`, and asking one for an answer is a
    /// boot bug — never a silent pass-through (which would serve a corrupt tensor to the client).
    #[test]
    fn a_canary_session_is_never_built_for_an_off_node() {
        let session = CanarySession::new(CanaryMode::Off);
        assert!(session
            .mis_transform(hidden(&[1.0], candle_core::DType::F32))
            .is_err());
    }

    /// The entry-node path of a canary fails **by name**: it cannot embed tokens without the model,
    /// and saying so is what keeps a canary from being deployed as a router.
    #[test]
    fn a_canary_cannot_serve_an_entry_node_session() {
        let mut session = CanarySession::new(CanaryMode::Echo);
        let err = session.forward(&[1, 2, 3]).unwrap_err();
        assert!(
            err.to_string().contains("canary"),
            "the refusal has to name the canary: {err}"
        );
    }

    // ─── El arranque sin modelo (`TASK-82`/S89) ────────────────────────────────────────────────

    /// The model of a real deployment, the one whose layer count the tracker's own table knows.
    const A_KNOWN_MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";

    /// A path **outside** the repositories and removed before use, so «nadie lo creó» is a fact of
    /// this run and not of a previous one that crashed half way.
    fn temp_models_dir(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("tessella-s89-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    fn config_of(models_dir: &std::path::Path, canary: CanaryMode) -> crate::config::WorkerConfig {
        crate::config::WorkerConfig {
            default_model: A_KNOWN_MODEL.to_string(),
            models_dir: models_dir.to_string_lossy().to_string(),
            canary_mode: canary,
            ..crate::config::WorkerConfig::default()
        }
    }

    /// `TASK-82`/S89, the acceptance criterion at the unit level: a canary boots with a `models_dir`
    /// that **does not exist**, and both halves are asserted — it resolves (nothing to read, nothing
    /// to download) *and* it leaves the directory alone: a boot that downloaded the model into it
    /// would pass the first half on its own.
    ///
    /// The metadata it can state is its `model_id` plus the layer count the tracker plans with; the
    /// rest of the shape stays unknown, which is why the hop ceiling comes back `None` — an unknown
    /// size, never a division by zero.
    #[tokio::test]
    async fn a_canary_boots_without_a_model_directory() {
        let models_dir = temp_models_dir("canary-no-model");
        let config = config_of(&models_dir, CanaryMode::Echo);

        let boot = boot_model(&config)
            .await
            .expect("a canary boots with no model directory at all");

        assert_eq!(
            boot.tokenizer_path, None,
            "a canary loads no tokenizer: the entry-node path refuses by name (TASK-71/S78)"
        );
        assert_eq!(boot.model_config.model_id, A_KNOWN_MODEL);
        assert_eq!(
            boot.model_config.total_layers(),
            24,
            "the layer count is the one the tracker plans with: a canary still validates its range"
        );
        assert_eq!(
            boot.model_config.hidden_size, 0,
            "the count is not a shape: a canary states the layers it serves, not a width it cannot read"
        );
        assert!(
            crate::inference_engine::hop_token_ceiling(&boot.model_config, &Device::Cpu, 0)
                .is_none(),
            "an unknown shape has no ceiling: it is not a division by zero"
        );
        assert!(
            !models_dir.exists(),
            "the boot of a canary must not create nor resolve anything inside `models_dir`"
        );
    }

    /// The risk the card names: with no `config.json` a canary cannot know which range it serves. It
    /// refuses to boot **naming the field** it is missing instead of assuming a number — the same rule
    /// the manifest of the tracker follows for a node it cannot look up.
    #[tokio::test]
    async fn a_canary_of_an_unknown_model_refuses_to_boot_naming_the_field() {
        let models_dir = temp_models_dir("canary-unknown-model");
        let config = crate::config::WorkerConfig {
            default_model: "acme/nonexistent-canary-9b".to_string(),
            ..config_of(&models_dir, CanaryMode::Scale)
        };

        let error = boot_model(&config)
            .await
            .expect_err("a model that cannot be looked up is not a boot this node may finish")
            .to_string();
        assert!(
            error.contains("num_hidden_layers"),
            "the refusal has to name the field it is missing: {error}"
        );
        assert!(
            error.contains("acme/nonexistent-canary-9b"),
            "and the model it was looking for: {error}"
        );
    }

    /// The other half of the card's criteria: the **ordinary** node does not change. A `models_dir`
    /// whose copy of the model is not a model (`config.json` without `num_hidden_layers`) still aborts
    /// the boot, and the error proves the path *read* the directory: a node with no canary mode keeps
    /// needing its model. The fixture writes `config.json` **and** `tokenizer.json`, so `resolve_index`
    /// takes its local branch and this test never touches the network.
    #[tokio::test]
    async fn an_ordinary_node_still_needs_a_real_model_directory() {
        let models_dir = temp_models_dir("node-with-poisoned-model");
        let model_dir = models_dir.join("Qwen2.5-0.5B-Instruct");
        std::fs::create_dir_all(&model_dir).expect("the fixture writes its own model directory");
        std::fs::write(model_dir.join("config.json"), r#"{"model_type":"qwen2"}"#)
            .expect("a `config.json` that is JSON but is not a model");
        std::fs::write(model_dir.join("tokenizer.json"), "{}").expect("a tokenizer placeholder");

        let error = boot_model(&config_of(&models_dir, CanaryMode::Off))
            .await
            .expect_err("a node with no canary mode boots its model or it does not boot")
            .to_string();
        assert!(
            error.contains("num_hidden_layers"),
            "the ordinary path resolves and reads the model directory: {error}"
        );

        let _ = std::fs::remove_dir_all(&models_dir);
    }

    /// What the node tells the tracker about its model (`TASK-82`/S89). A canary states **no**
    /// `ModelConfig` — it holds no shape — and the orchestrator then plans with the same table it uses
    /// for every node that registers without one. An ordinary node states its own, as it always did.
    #[test]
    fn a_canary_reports_no_model_config() {
        let model = tessella_core::models::ModelConfig {
            model_id: A_KNOWN_MODEL.to_string(),
            num_hidden_layers: 24,
            ..Default::default()
        };
        let here = std::path::Path::new("./models");

        assert_eq!(
            registration_model_config(&config_of(here, CanaryMode::Echo), Some(&model)),
            None,
            "a canary holds no model: it cannot state a shape"
        );
        assert_eq!(
            registration_model_config(&config_of(here, CanaryMode::Off), Some(&model)),
            Some(model),
            "the ordinary node reports its own config, exactly as before"
        );
    }
}
