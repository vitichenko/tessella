use crate::domain::ModelConfig;
use crate::models::{ComputeDevice, QuantizationConfig};
use crate::network::WorkerAssignment;

/// `S51`/`TASK-45`: bytes per element of the dtype the loader casts the weights to — and with which
/// the KV cache grows — on a device.
///
/// `shard_loader::tensor_view_to_candle` uses **F16 on CUDA and F32 everywhere else**, the same rule
/// `tessella_inference::backends::activation_dtype` implements for the activations. The planner
/// cannot link against `candle`, so it consumes the byte count from here and the backends keep the
/// `DType` side of the same rule. This used to be an unconditional `2` (`// 2 for F16`), which made a
/// `metal` node — a `device` D-6 allows — receive a budget for half the memory it really loads.
#[must_use]
pub const fn load_dtype_bytes(is_cuda: bool) -> usize {
    if is_cuda {
        2 // F16
    } else {
        4 // F32
    }
}

/// Whether this device runs the CUDA backend (the only one that loads F16).
#[must_use]
pub fn is_cuda_device(device: &ComputeDevice) -> bool {
    matches!(device, ComputeDevice::Cuda { .. })
}

/// `S51`/`TASK-45`: the **single** definition of the bytes per parameter a shard of this node
/// occupies in VRAM, from the two things that decide it: the device (the loader's dtype) and the
/// quantization the operator asked for.
///
/// `Q8_0` is the only quantization that exists — `Q4K` was retired because it had no loader, no
/// consumer and no measurement (`TASK-45`/S51). Its `gguf` blocks store 32 weights as 32 `i8` plus
/// one `f16` scale ⇒ 8.5 bits per parameter, and `LayerFactory` keeps them quantized
/// (`QuantizedLinear`), so the figure does not depend on the device. `None` means «unquantized»: the
/// dtype the loader casts to.
#[must_use]
pub fn bytes_per_param(device: &ComputeDevice, quantization: QuantizationConfig) -> f64 {
    match quantization {
        QuantizationConfig::Q8_0 => 1.0625,
        QuantizationConfig::None => load_dtype_bytes(is_cuda_device(device)) as f64,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerMemoryConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize, // dense MLP intermediate OR shared expert size
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub is_moe: bool,
    pub num_experts: usize,              // 0 if dense
    pub num_experts_per_tok: usize,      // top-k, 0 if dense
    pub expert_intermediate_size: usize, // per-expert ffn size, 0 if dense
    /// `S51`/`TASK-45`: bytes per parameter of the weights this node loads
    /// ([`bytes_per_param`]). It replaced a hardcoded `2`.
    pub bytes_per_param: f64,
    /// `S51`/`TASK-45`: bytes per element of the activations and of the KV cache on this device
    /// ([`load_dtype_bytes`]). It is not `bytes_per_param`: a `Q8_0` node keeps its weights
    /// quantized but its KV cache is still F16/F32.
    pub activation_bytes: usize,
    /// `S52`/`TASK-46`: vocabulary of the model, needed for the token embedding table / LM head that
    /// the **edge** of the pipeline loads. `0` means «not modelled»: the 20 % headroom is then the
    /// only margin and the budget says so instead of inventing a number.
    pub vocab_size: usize,
    /// `S52`/`TASK-46`: context the KV cache can grow to. `0` means «not modelled» (same honesty).
    /// The KV cache is per session and per local layer, and before S52 it was outside the budget
    /// completely.
    pub max_position_embeddings: usize,
}

/// Hard floor for the per-layer cost: one layer of a dense model always carries activations,
/// quantization scales and buffers on top of its weights.
///
/// It is **not** the KV cache: that is modelled per layer since `S52`/`TASK-46` (the old comment
/// here claimed this constant included it, which is why the cache was never budgeted).
const MIN_MB_PER_LAYER: u64 = 40;

/// How many concurrent sessions per node the layer budget assumes.
///
/// One: the minimum that makes the KV cache part of the decision. It is a named constant and not a
/// knob because no other number has a measurement behind it — `TASK-46`'s revisita is «raise it when
/// a deployment shows several concurrent sessions per node, with the measurement», and until then an
/// operator-facing switch would only move the split arbitrarily.
pub const KV_SESSIONS_PER_NODE: usize = 1;

/// Where the per-layer figure of a node comes from, term by term (`S51`+`S52`).
///
/// The refusal messages and the operator-facing endpoints print these numbers, so «why did this node
/// get three layers?» has an answer that is not a log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerBudget {
    /// Whole layers that fit in the claim (`0` when not a single one does).
    pub layers: usize,
    /// Weights of one whole layer: attention + FFN/MoE.
    pub weights_per_layer_mb: u64,
    /// KV cache of one whole layer for [`KV_SESSIONS_PER_NODE`] sessions at the model's context.
    pub kv_per_layer_mb: u64,
    /// `weights_per_layer_mb + kv_per_layer_mb`: what one **more** layer really costs.
    pub per_layer_mb: u64,
    /// Embedding table / LM head the edge of the pipeline keeps resident (`0` when not modelled).
    pub global_tensor_mb: u64,
    /// What is left of the claim after the 20 % headroom and the global tensors.
    pub usable_vram_mb: u64,
}

impl LayerBudget {
    /// VRAM this node would need for `layers` full layers, global tensors included.
    #[must_use]
    pub fn required_mb(&self, layers: usize) -> u64 {
        layers as u64 * self.per_layer_mb + self.global_tensor_mb
    }
}

/// `S52`/`TASK-46`: the breakdown behind [`compute_layer_budget`] — same arithmetic, every term
/// visible.
///
/// **The edge tensors are charged once per node** (`global_tensor_mb`), and that is an assumption
/// about *where* they are loaded, not a measurement (`TASK-96`/S102). A node serving several ranges
/// of the same model may be the front of the pipeline for one and interior for another — or hold
/// both edges at once — and this arithmetic still charges the embedding table and the LM head a
/// single time. It holds while those tensors are one resident copy per process, which is what the
/// backend does today (`shard_loader` loads them for the range that starts at 0 and for the range
/// that reaches `total_layers`, once each). A shape that needed a second copy — a node serving two
/// **fronts** of two different models, say — would need a per-frontier term here, and that is the
/// shape this budget does not price.
#[must_use]
pub fn layer_budget(vram_mb: u32, config: &LayerMemoryConfig, kv_sessions: usize) -> LayerBudget {
    const MB: f64 = 1_048_576.0;

    let vram_mb = f64::from(vram_mb);
    let hidden_size = config.hidden_size as f64;
    let intermediate_size = config.intermediate_size as f64;
    let num_kv_heads = config.num_kv_heads as f64;
    let head_dim = config.head_dim as f64;
    let bpp = config.bytes_per_param;

    // Attention block (Q, K, V, O projections)
    let attn_mem_mb = (hidden_size * hidden_size
        + hidden_size * num_kv_heads * head_dim * 2.0
        + hidden_size * hidden_size)
        * bpp
        / MB;

    // FFN block
    let ffn_mem_mb = if config.is_moe {
        let num_experts = config.num_experts as f64;
        let expert_intermediate_size = config.expert_intermediate_size as f64;

        // shared expert (always active)
        let shared_mem = hidden_size * intermediate_size * 3.0 * bpp / MB;
        // routed experts (all must be resident)
        let expert_mem = num_experts * expert_intermediate_size * hidden_size * 3.0 * bpp / MB;
        // gate weights
        let gate_mem = hidden_size * num_experts * bpp / MB;

        shared_mem + expert_mem + gate_mem
    } else {
        hidden_size * intermediate_size * 3.0 * bpp / MB
    };

    let weights_per_layer_mb = (attn_mem_mb + ffn_mem_mb).max(MIN_MB_PER_LAYER as f64);

    // `S52`/`TASK-46`: KV cache of one local layer, for `kv_sessions` sessions at the context the
    // model declares. Two tensors (K and V) of `num_kv_heads × head_dim` elements per token. When the
    // model does not declare a context this stays 0 and the 20 % headroom is the (undeclared) margin.
    let kv_per_layer_mb = if config.max_position_embeddings > 0 && kv_sessions > 0 {
        (2.0 * num_kv_heads
            * head_dim
            * config.activation_bytes as f64
            * config.max_position_embeddings as f64
            * kv_sessions as f64)
            / MB
    } else {
        0.0
    };

    let per_layer_mb = weights_per_layer_mb + kv_per_layer_mb;

    // `S52`/`TASK-46`: the token embedding table (entry node) or the final norm + LM head (last
    // node). `shard_loader::should_load_tensor` gives a hop one of the two — never both, in a
    // pipeline with two or more nodes — so the worst case of a node is one copy. It was outside the
    // budget entirely, and it is 2.18 GB on a 7B.
    let global_tensor_mb = if config.vocab_size > 0 {
        (config.vocab_size as f64 * hidden_size * bpp) / MB
    } else {
        0.0
    };

    // 20% headroom reservation: activations, allocator fragmentation and the CUDA context. It is
    // applied to the claim, and the global tensors are subtracted **after** it, because they are real
    // resident weights and not headroom.
    let usable_vram_mb = (vram_mb * 80.0 / 100.0 - global_tensor_mb).max(0.0);
    let layers = (usable_vram_mb / per_layer_mb).floor() as usize;

    LayerBudget {
        layers,
        weights_per_layer_mb: weights_per_layer_mb.round() as u64,
        kv_per_layer_mb: kv_per_layer_mb.round() as u64,
        per_layer_mb: per_layer_mb.round() as u64,
        global_tensor_mb: global_tensor_mb.round() as u64,
        usable_vram_mb: usable_vram_mb.round() as u64,
    }
}

/// Returns how many contiguous layers a node with `vram_mb` VRAM can hold for the given model config,
/// plus the per-layer cost the decision was based on. Reserves 20% headroom for activations and
/// allocator fragmentation; see [`layer_budget`] for every term of the figure.
pub fn compute_layer_budget(vram_mb: u32, config: &LayerMemoryConfig) -> (usize, u64) {
    let budget = layer_budget(vram_mb, config, KV_SESSIONS_PER_NODE);
    (budget.layers, budget.per_layer_mb)
}

/// Hard floor of a VRAM claim (`S52`/`TASK-46`): below it no model fits one layer, so the value is not
/// a policy but a floor of the arithmetic.
pub const MIN_VRAM_MB: u32 = 256;

/// The claim floor this deployment really enforces (`S52`/`TASK-46` · VR-03).
///
/// It was `if reg.vram_mb < 256` hardcoded in the orchestrator while `min_vram_gb` — 0.5 in the code,
/// 0.4 in the example and in the operator's TOML, «2 GB» in the docs — was read by **nobody**: an
/// operator raising it to keep small nodes out of the cluster got exactly nothing, and no error said
/// so. Now the operator's value is the floor, and [`MIN_VRAM_MB`] is the absolute minimum.
#[must_use]
pub fn effective_vram_floor_mb(min_vram_gb: f32) -> u32 {
    crate::vram::vram_mb_from_gb(min_vram_gb).max(MIN_VRAM_MB)
}

/// `S43`/`TASK-37`: can a shard with this `expert_range` serve a **whole** layer?
///
/// Expert parallelism is **not implemented**: a hop that holds a fraction of the experts still
/// executes the whole layer (attention, residual and shared expert included), so gathering its
/// answer together with its siblings' counts those parts once per peer and produces wrong logits
/// with no warning. Only the full expert set — or no range at all, which is the same thing — is a
/// configuration the mesh can serve, and it is the only one the assignment produces since S43.
///
/// This is the **single** definition of the policy: `tessella-inference` uses it to refuse such a
/// shard when it builds a `MoeBlock`, and `crate::routing` applies its conservative form (a
/// `LayerPeer` does not carry the expert count) when it picks a route.
pub fn serves_whole_layers(expert_range: Option<(usize, usize)>, num_experts: usize) -> bool {
    match expert_range {
        None => true,
        Some((start, end)) => num_experts > 0 && start == 0 && end == num_experts,
    }
}

/// Why a shard with this `expert_range` cannot be served (`None` when it can).
///
/// The message is what a worker prints when it refuses the shard, so it says what the range would
/// cost, not just that it is wrong.
pub fn unsupported_expert_range_reason(
    expert_range: Option<(usize, usize)>,
    num_experts: usize,
) -> Option<String> {
    let (start, end) = expert_range?;
    if serves_whole_layers(expert_range, num_experts) {
        return None;
    }
    Some(format!(
        "expert parallelism is not implemented, so a shard holding experts [{start}, {end}) of \
         {num_experts} cannot be served: it returns a whole layer (attention + residual + shared \
         expert) for a fraction of the experts, and gathering those answers counts those parts once \
         per peer (`TASK-37`/S43)"
    ))
}

/// `S65`: whether adopting `next` changes the **weights** a node holds, i.e. whether the shard has to
/// be re-loaded (and not only the assignment bookkeeping updated).
///
/// The worker used to answer this by comparing **only** the layer range and returning before
/// touching anything when it matched (`network_manager::reload_shard`), so a new assignment that
/// moved the expert range — the assignment an expert-parallel orchestrator would send — or the model
/// was **silently ignored** and the engine kept the old one. That is precisely the half the retired
/// expert autoscaler never had (`TASK-38`/S44: «a replication no worker could adopt»), and it is a
/// prerequisite of the 2-D split (`TASK-64`).
///
/// The three fields compared are exactly the ones [`layer_budget`] and `shard_loader` read to decide
/// what to load; `replica_group_id` selects no tensor, so a replica-group-only change is adopted
/// without re-loading anything. Since `TASK-96`/S102 the first of them is a **set** of ranges, which
/// is the whole point: a second range means a second shard to download and a second backend to
/// build, so an assignment that only adds one is a weight change like any other.
#[must_use]
pub fn requires_weight_reload(current: &WorkerAssignment, next: &WorkerAssignment) -> bool {
    current.layer_ranges != next.layer_ranges
        || current.expert_range != next.expert_range
        || current.model_repo != next.model_repo
}

/// Why a node claiming `vram_mb` cannot host a single layer of `config` (`None` when it can).
///
/// `S43`/`TASK-37`: such a node used to receive a one-layer shard anyway (`budget.max(1)`) or, in an
/// MoE model, an expert shard whose gather was arithmetically wrong. Both were lies; the
/// registration is refused with this reason instead, naming the numbers involved.
pub fn capacity_reason(vram_mb: u32, config: &LayerMemoryConfig) -> Option<String> {
    let budget = layer_budget(vram_mb, config, KV_SESSIONS_PER_NODE);
    if budget.layers > 0 {
        return None;
    }

    // `S52`/`TASK-46`: the message names every term of the figure, so an operator can see whether the
    // node is short of weights, of KV cache or of room for the embedding table.
    Some(format!(
        "[TESERROR-01] cannot host a single layer of this model: a claim of {vram_mb} MB leaves {} MB usable \
         (80% of the claim minus {} MB of embedding/LM-head tensors) and one layer costs {} MB \
         ({} MB of weights + {} MB of KV cache for {KV_SESSIONS_PER_NODE} session(s) at {} tokens of \
         context). Expert parallelism is not implemented either, so a partial expert \
         shard is not an option: raise the VRAM claim or use a node with more VRAM",
        budget.usable_vram_mb,
        budget.global_tensor_mb,
        budget.per_layer_mb,
        budget.weights_per_layer_mb,
        budget.kv_per_layer_mb,
        config.max_position_embeddings
    ))
}

/// The shape of a model, without the two things that decide the dtype (`S51`) or the terms of the
/// budget that are not per-parameter (`S52`). Both [`LayerMemoryConfig::from_model_config`] and the
/// fallback table below resolve it through [`LayerShape::resolve`], so the two paths cannot drift.
#[derive(Debug, Clone)]
struct LayerShape {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_moe: bool,
    num_experts: usize,
    num_experts_per_tok: usize,
    expert_intermediate_size: usize,
    /// `0` = the model is not in `config.json` form and its vocabulary is unknown: the embedding
    /// table is then **not** modelled (the 20 % headroom is the only margin) instead of inventing one.
    vocab_size: usize,
    /// Same rule as `vocab_size`: `0` means «context unknown» and the KV cache is not modelled.
    max_position_embeddings: usize,
}

impl LayerShape {
    fn resolve(
        self,
        device: &ComputeDevice,
        quantization: QuantizationConfig,
    ) -> LayerMemoryConfig {
        LayerMemoryConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            is_moe: self.is_moe,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            expert_intermediate_size: self.expert_intermediate_size,
            bytes_per_param: bytes_per_param(device, quantization),
            activation_bytes: load_dtype_bytes(is_cuda_device(device)),
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
        }
    }
}

impl LayerMemoryConfig {
    /// `S51`/`TASK-45`: the dtype is derived from the device and from the quantization the model
    /// config declares (`ModelConfig::quantization`, which the worker fills with its own setting), so
    /// the budget no longer assumes 2 bytes per parameter for everybody.
    pub fn from_model_config(config: &ModelConfig, device: &ComputeDevice) -> Self {
        LayerShape {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config
                .hidden_size
                .checked_div(config.num_attention_heads)
                .unwrap_or(32),
            is_moe: config.is_moe(),
            num_experts: config.num_experts.unwrap_or(0),
            num_experts_per_tok: config.num_experts_per_tok.unwrap_or(0),
            expert_intermediate_size: config
                .moe_intermediate_size
                .unwrap_or(config.shared_expert_intermediate_size.unwrap_or(0)),
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
        }
        .resolve(device, config.quantization)
    }
}

/// One row of [`FALLBACK_SHAPES`]: the fragment of a model id that selects it and the shape of one of
/// its layers.
///
/// `TASK-86`/S93: the table is **data** and has **one row per model size**, because the chain of
/// `if id.contains(..)` it replaces had one branch per id *spelling*: the `-Coder-` variant of a size
/// was listed and the plain one was not, so `Qwen/Qwen2.5-1.5B-Instruct` — the model of `swarm.sh`, of
/// the docs and of `ModelConfig::default()` — was planned as a 4096-wide model while
/// `Qwen/Qwen2.5-Coder-1.5B-Instruct` was planned correctly. Measured on 2026-09-26 (`CANARY=on
/// --workers 2 --vram 3.0`, 1.5B): the two nodes that read their `config.json` got 16 layers each; the
/// canary, with the same claim, got the 6 layers of that unknown-model arithmetic and took layer 0, so
/// the mesh had no entry node at all (`TASK-83`/S90 forbids a canary as one).
///
/// Every number is the one of the `config.json` of the model: the two spellings of a size are the same
/// shape field by field (verified against `Qwen/Qwen2.5-<size>-Instruct` and
/// `Qwen/Qwen2.5-Coder-<size>-Instruct` on the Hub, 2026-09-26) and `max_position_embeddings` is
/// `32_768` in all of them — the **base** models of the same sizes declare `131_072`, and the mesh
/// serves the `-Instruct` ones.
#[derive(Debug, Clone, Copy)]
struct FallbackShape {
    /// Lower-case substring of the model id, and deliberately the **same fragment** the layer count of
    /// `tessella_protocol::get_total_layers` matches. That is what keeps the two tables coherent: a
    /// model whose number of layers the mesh knows is never planned with the unknown-model default.
    id_fragment: &'static str,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    /// `(num_experts, num_experts_per_tok, expert_intermediate_size)`; `None` for a dense model.
    experts: Option<(usize, usize, usize)>,
}

impl FallbackShape {
    /// A dense `Qwen2.5`-style row. `head_dim` is **derived** (`hidden_size / num_heads`, the
    /// expression `LayerMemoryConfig::from_model_config` uses), so a row cannot disagree with the model
    /// it summarises about it: it is 64 in the 0.5B and 128 in every other size.
    const fn dense(
        id_fragment: &'static str,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
    ) -> Self {
        Self {
            id_fragment,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim: hidden_size / num_heads,
            vocab_size,
            max_position_embeddings,
            experts: None,
        }
    }

    /// An MoE row whose `config.json` this repository does not have: it states no vocabulary and no
    /// context, the convention of `ModelConfig::no_shape()`, so neither the embedding table nor the KV
    /// cache is modelled for it.
    const fn moe(
        id_fragment: &'static str,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        experts: (usize, usize, usize),
    ) -> Self {
        Self {
            id_fragment,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size: 0,
            max_position_embeddings: 0,
            experts: Some(experts),
        }
    }

    fn to_layer_shape(self) -> LayerShape {
        let (num_experts, num_experts_per_tok, expert_intermediate_size) =
            self.experts.unwrap_or((0, 0, 0));
        LayerShape {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            is_moe: self.experts.is_some(),
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
        }
    }
}

/// Every model this table knows, **first match wins**.
///
/// The MoE rows come before the sizes they contain (`qwen2-moe-a2.7b` contains `7b`). There is no row
/// for `test-model`: it is the id the tests of `musivum` use as an *unknown* model, no `config.json` of
/// this repository backs a 1920-wide one, and the conservative default is the figure its own refusal
/// test names — see `a_model_of_an_unknown_size_states_nothing`, which pins both directions.
const FALLBACK_SHAPES: &[FallbackShape] = &[
    // The fixture of this repository (`tessella/models/tessella-moe-micro/config.json`, written by
    // `scripts/create_micro_moe.py`): a repository with a real vocabulary and context, so both **are**
    // modelled — `plan_split.py` reads them from that file and prints 115.0 MB per layer (113 + KV 2).
    FallbackShape {
        id_fragment: "tessella-moe-micro",
        hidden_size: 1024,
        intermediate_size: 2048,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 128,
        vocab_size: 32_000,
        max_position_embeddings: 2_048,
        experts: Some((8, 2, 2048)),
    },
    // `Qwen1.5-MoE-A2.7B`, **before** the `7b` row (its id contains `7b`). `TASK-86`/S93 leaves the
    // shape as it was: the fragment is not the id the Hub publishes and `intermediate_size` is not the
    // shared-expert size of that model (`5_632`). A card of its own.
    FallbackShape::moe("qwen2-moe-a2.7b", 2048, 1408, 16, 16, 128, (60, 4, 1408)),
    // `Qwen2.5-<size>-Instruct` and `Qwen2.5-Coder-<size>-Instruct`: one row per size, with the same
    // fragments `get_total_layers` uses.
    // `TASK-82`/S89 measured the 0.5B case: `CANARY=on --workers 2 --vram 1.2` gave the nodes that
    // reported their own `config.json` 12 layers each and the canary, with the same claim, `[[0, 2]]`.
    // The shape is the one of the `-Coder-` variant and of `ModelConfig::qwen2_5_0_5b()` (both pinned
    // by the test below).
    FallbackShape::dense("0.5b", 896, 4_864, 14, 2, 151_936, 32_768),
    FallbackShape::dense("1.5b", 1536, 8_960, 12, 2, 151_936, 32_768),
    FallbackShape::dense("3b", 2_048, 11_008, 16, 2, 151_936, 32_768),
    FallbackShape::dense("7b", 3584, 18_944, 28, 4, 152_064, 32_768),
    // `S52`/`TASK-46` (VR-05): the real `intermediate_size` of Qwen2.5-14B is **13824** (verified
    // against `Qwen/Qwen2.5-14B-Instruct` and `…-Coder-14B-Instruct` on the Hub), and this table used to
    // carry a value **twice as big**, which inflated the per-layer cost of a 14B by 2× and made
    // `/workers` publish the inflated figure. The reference test below compares every row with the
    // `config.json` it summarises, so no row can drift again.
    FallbackShape::dense("14b", 5_120, 13_824, 40, 8, 152_064, 32_768),
    FallbackShape::dense("72b", 8_192, 29_568, 64, 8, 152_064, 32_768),
];

/// The shape of a model **the table does not know**, and not a row of [`FALLBACK_SHAPES`]: an empty
/// fragment would match every id.
///
/// `hidden 4096` / `intermediate 11008` is the arithmetic the mesh has always used for an unknown model
/// (386 MB per layer, the figure the refusal test of `musivum` names). It is the shape of a 7B: it is
/// pessimistic for the small models of a home cluster — which is what makes it safe — and optimistic
/// for a wide one (a 72B declares `hidden 8192`), so it is not a bound. A node that cannot read its
/// `config.json` is planned wrong in one direction or the other, which is why the normal path is the
/// one that carries a `Some(config)`.
const CONSERVATIVE_SHAPE: FallbackShape = FallbackShape::dense("", 4096, 11_008, 32, 32, 0, 0);

/// The memory config of `model_id`, from the model config the worker reported (the normal path) or
/// from [`FALLBACK_SHAPES`], the table of the models this repository can summarise.
///
/// `S51`/`TASK-45`: it needs the **device** now, because the per-parameter size depends on the dtype
/// the loader casts to (F16 on CUDA, F32 elsewhere). [`FALLBACK_SHAPES`] carries the vocabulary and the
/// context of every model whose `config.json` this repository has or cites, so the budget models both
/// terms for them; a row without a `config.json` behind it says `vocab_size: 0` and
/// `max_position_embeddings: 0`, which is how the budget states that it is not modelling those terms.
pub fn layer_memory_config_for(
    model_id: &str,
    config: Option<&ModelConfig>,
    device: &ComputeDevice,
) -> LayerMemoryConfig {
    if let Some(cfg) = config {
        return LayerMemoryConfig::from_model_config(cfg, device);
    }

    let id = model_id.to_lowercase();
    // First match wins: [`FALLBACK_SHAPES`] keeps the MoE rows before the sizes they contain.
    let shape = FALLBACK_SHAPES
        .iter()
        .find(|row| id.contains(row.id_fragment))
        .copied()
        .unwrap_or(CONSERVATIVE_SHAPE);

    shape
        .to_layer_shape()
        .resolve(device, QuantizationConfig::None)
}

// `S43`/`TASK-37`: `get_total_layers_resolved`, `assign_worker` and `compute_expert_budget` used to
// live here. None of them had a caller — the real sharding happens in
// `musivum/crates/core/src/orchestrator.rs` — and `assign_worker` was the only place in
// `tessella-core` that produced an expert-parallel assignment (`expert_range: Some(..)`), a
// configuration the mesh cannot serve correctly (see [`serves_whole_layers`]). They were removed
// instead of kept «for later»: the card that implements layer × expert sharding (`TASK-64`/`S70`, blocked in S65)
// re-adds what it needs, with tests.

#[cfg(test)]
mod tests {
    use super::*;

    const CUDA: ComputeDevice = ComputeDevice::Cuda { device_id: 0 };
    const METAL: ComputeDevice = ComputeDevice::Metal;

    /// A fixture with the `S51`/`S52` terms explicit. `vocab_size` and context are `0` (not modelled)
    /// unless a test sets them, so the arithmetic of the older tests stays comparable.
    fn dense(
        hidden_size: usize,
        intermediate_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> LayerMemoryConfig {
        LayerShape {
            hidden_size,
            intermediate_size,
            num_heads: 12,
            num_kv_heads,
            head_dim,
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            expert_intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
        }
        .resolve(&CUDA, QuantizationConfig::None)
    }

    fn moe(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        expert_intermediate_size: usize,
    ) -> LayerMemoryConfig {
        LayerShape {
            hidden_size,
            intermediate_size,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 128,
            is_moe: true,
            num_experts,
            num_experts_per_tok: 4,
            expert_intermediate_size,
            vocab_size: 0,
            max_position_embeddings: 0,
        }
        .resolve(&CUDA, QuantizationConfig::None)
    }

    /// A `ModelConfig` with the fields a budget cares about and the real values of a HF `config.json`
    /// (the reference the fallback table is compared against).
    fn model_config(
        model_id: &str,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
    ) -> ModelConfig {
        ModelConfig {
            model_id: model_id.to_string(),
            quantization: QuantizationConfig::None,
            model_type: Some("qwen2".to_string()),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hidden_size,
            num_hidden_layers: 28,
            num_experts: None,
            num_experts_per_tok: None,
            shared_expert_intermediate_size: None,
            moe_layer_freq: None,
            norm_topk_prob: None,
            vocab_size,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            moe_intermediate_size: None,
            max_position_embeddings,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            eos_token_id: Some(151645),
        }
    }

    #[test]
    fn test_dense_budget_6gb() {
        // Qwen2.5-Coder-1.5B on 6GB GPU
        let config = dense(1536, 8960, 2, 128);
        let budget = compute_layer_budget(6144, &config).0;
        assert!(
            budget >= 4,
            "6GB should fit at least 4 layers, got {}",
            budget
        );
    }

    #[test]
    fn test_moe_budget_smaller_than_dense_same_vram() {
        // MoE layers are heavier due to all expert weights being resident
        let dense = dense(2048, 11008, 16, 128);
        let moe = moe(2048, 1408, 60, 1408);
        let dense_budget = compute_layer_budget(8192, &dense).0;
        let moe_budget = compute_layer_budget(8192, &moe).0;
        assert!(
            moe_budget < dense_budget,
            "MoE budget ({}) should be smaller than dense ({}) for same VRAM",
            moe_budget,
            dense_budget
        );
    }
    /// `S43`/`TASK-37`: expert parallelism is not a configuration this mesh can serve, and the
    /// policy has one definition (`serves_whole_layers`) shared by the router, the loader and the
    /// orchestrator.
    #[test]
    fn only_the_full_expert_set_counts_as_a_whole_layer_shard() {
        assert!(
            serves_whole_layers(None, 8),
            "no range means the whole layer"
        );
        assert!(
            serves_whole_layers(Some((0, 8)), 8),
            "the full range is the same as no range at all"
        );
        assert!(!serves_whole_layers(Some((0, 4)), 8));
        assert!(!serves_whole_layers(Some((4, 8)), 8));
        assert!(
            !serves_whole_layers(Some((0, 0)), 0),
            "a model without experts cannot be sharded by experts"
        );

        assert_eq!(unsupported_expert_range_reason(None, 8), None);
        assert_eq!(unsupported_expert_range_reason(Some((0, 8)), 8), None);
        let reason =
            unsupported_expert_range_reason(Some((0, 4)), 8).expect("a partial range has a reason");
        assert!(
            reason.contains("[0, 4)") && reason.contains("8"),
            "the reason names the range and the expert count: {reason}"
        );
    }

    /// `S51`/`TASK-45` (VR-02): the planner used to assume 2 bytes per parameter for everybody while
    /// the loader casts to F32 on Metal and keeps `Q8_0` shards at 8.5 bits. The figure has one
    /// definition now and both terms are visible.
    #[test]
    fn bytes_per_param_follows_the_device_and_the_quantization() {
        assert_eq!(bytes_per_param(&CUDA, QuantizationConfig::None), 2.0);
        assert_eq!(bytes_per_param(&METAL, QuantizationConfig::None), 4.0);
        assert_eq!(bytes_per_param(&CUDA, QuantizationConfig::Q8_0), 1.0625);
        assert_eq!(
            bytes_per_param(&METAL, QuantizationConfig::Q8_0),
            1.0625,
            "a quantized shard keeps its weights quantized on any device (LayerFactory)"
        );
        assert_eq!(
            bytes_per_param(&ComputeDevice::Cpu, QuantizationConfig::None),
            4.0,
            "the CPU backend would load F32 too (it is refused elsewhere, D-6)"
        );
        assert_eq!(load_dtype_bytes(true), 2);
        assert_eq!(load_dtype_bytes(false), 4);
    }

    /// The bug VR-02 in numbers: a Metal node is budgeted for the F32 it really loads, so it gets
    /// **fewer** layers than a CUDA node with the same claim. Before this fix both got the same, which
    /// is a 2× overcommit on a `device` D-6 allows.
    #[test]
    fn a_metal_node_gets_fewer_layers_than_a_cuda_node_with_the_same_claim() {
        let shape = LayerShape {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            expert_intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
        };
        let cuda = shape.clone().resolve(&CUDA, QuantizationConfig::None);
        let metal = shape.resolve(&METAL, QuantizationConfig::None);

        let cuda_budget = layer_budget(4096, &cuda, KV_SESSIONS_PER_NODE);
        let metal_budget = layer_budget(4096, &metal, KV_SESSIONS_PER_NODE);

        assert!(
            (metal_budget.weights_per_layer_mb as i64
                - 2 * cuda_budget.weights_per_layer_mb as i64)
                .abs()
                <= 2,
            "F32 is twice F16 per parameter (cuda={} MB, metal={} MB)",
            cuda_budget.weights_per_layer_mb,
            metal_budget.weights_per_layer_mb
        );
        assert!(
            metal_budget.layers < cuda_budget.layers,
            "the Metal node must not get the F16 budget (cuda={}, metal={})",
            cuda_budget.layers,
            metal_budget.layers
        );
    }

    /// The other half of VR-02: `Q8_0` used to change no assignment at all, so the benefit the README
    /// announces never reached the mesh.
    #[test]
    fn a_q8_0_node_gets_more_layers_than_an_unquantized_one() {
        let shape = LayerShape {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            expert_intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
        };
        let unquantized = shape.clone().resolve(&CUDA, QuantizationConfig::None);
        let quantized = shape.resolve(&CUDA, QuantizationConfig::Q8_0);

        let plain = layer_budget(2048, &unquantized, KV_SESSIONS_PER_NODE);
        let eight_bit = layer_budget(2048, &quantized, KV_SESSIONS_PER_NODE);

        assert!(plain.layers > 0);
        assert!(
            eight_bit.layers > plain.layers,
            "8.5 bits per parameter must fit more layers (plain={}, q8_0={})",
            plain.layers,
            eight_bit.layers
        );
        assert_eq!(
            quantized.activation_bytes, unquantized.activation_bytes,
            "the KV cache is still F16; only the weights are quantized"
        );
    }

    /// `S52`/`TASK-46` (VR-04): the embedding table or the LM head of a 7B is 2,18 GB the budget did
    /// not count. A node that may be the edge of the pipeline must pay for it.
    #[test]
    fn a_node_pays_for_the_embedding_table_it_would_hold_at_the_edge() {
        let shape = LayerShape {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            expert_intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
        };
        let interior = shape.clone().resolve(&CUDA, QuantizationConfig::None);
        let edge = LayerShape {
            vocab_size: 152_064, // Qwen2.5-Coder-7B
            ..shape
        }
        .resolve(&CUDA, QuantizationConfig::None);

        let interior_budget = layer_budget(2048, &interior, KV_SESSIONS_PER_NODE);
        let edge_budget = layer_budget(2048, &edge, KV_SESSIONS_PER_NODE);

        assert_eq!(interior_budget.global_tensor_mb, 0);
        assert!(
            edge_budget.global_tensor_mb > 1000,
            "152064 × 3584 × 2 bytes is more than a GB, got {} MB",
            edge_budget.global_tensor_mb
        );
        assert!(
            edge_budget.layers < interior_budget.layers,
            "the node that would hold the embedding table gets fewer layers (interior={}, edge={})",
            interior_budget.layers,
            edge_budget.layers
        );
        assert!(
            (edge_budget.usable_vram_mb as i64
                - (interior_budget.usable_vram_mb - edge_budget.global_tensor_mb) as i64)
                .abs()
                <= 1,
            "the global tensors are subtracted from the usable claim (edge={} MB, interior={} MB, \
             globals={} MB)",
            edge_budget.usable_vram_mb,
            interior_budget.usable_vram_mb,
            edge_budget.global_tensor_mb
        );
        assert_eq!(
            edge_budget.required_mb(edge_budget.layers),
            (edge_budget.layers as u64) * edge_budget.per_layer_mb + edge_budget.global_tensor_mb
        );
    }

    /// `S52`/`TASK-46` (VR-04): the KV cache is per session and per local layer, and it used to be
    /// outside the budget entirely — the constant whose comment claimed to include it did not.
    #[test]
    fn the_kv_cache_of_one_layer_is_part_of_the_budget() {
        let shape = LayerShape {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            expert_intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
        };
        let without_context = shape.clone().resolve(&CUDA, QuantizationConfig::None);
        let with_context = LayerShape {
            max_position_embeddings: 32_768,
            ..shape
        }
        .resolve(&CUDA, QuantizationConfig::None);

        let no_ctx = layer_budget(2048, &without_context, KV_SESSIONS_PER_NODE);
        let ctx = layer_budget(2048, &with_context, KV_SESSIONS_PER_NODE);

        assert_eq!(no_ctx.kv_per_layer_mb, 0, "no declared context, no figure");
        assert_eq!(
            ctx.kv_per_layer_mb, 64,
            "2 × 4 kv_heads × 128 head_dim × 2 bytes × 32768 tokens = 64 MB per layer"
        );
        assert_eq!(
            ctx.per_layer_mb,
            ctx.weights_per_layer_mb + ctx.kv_per_layer_mb
        );
        assert!(
            ctx.layers <= no_ctx.layers,
            "counting the KV cache cannot fit more layers"
        );

        let four_sessions = layer_budget(2048, &with_context, 4);
        assert_eq!(
            four_sessions.kv_per_layer_mb,
            ctx.kv_per_layer_mb * 4,
            "the KV term scales with the sessions of the node"
        );
    }

    /// `S52`/`TASK-46` (VR-05) and `TASK-86`/S93: every row of [`FALLBACK_SHAPES`] is compared against a
    /// `config.json` that exists. The `intermediate_size` of a 14B was **2×** the real value (`13_824`),
    /// and before `TASK-86`/S93 only the `-Coder-` spelling of each size was in the table, so the plain
    /// spelling of 1.5B, 3B, 7B, 14B and 72B fell through to the conservative default. This test is what
    /// keeps a row from drifting again: one entry per size and spelling, plus the local MoE fixture.
    #[test]
    fn the_backup_table_matches_the_reference_configs_it_summarises() {
        let cases = [
            ("Qwen/Qwen2.5-0.5B-Instruct", 896, 4_864, 14, 2, 151_936),
            (
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                896,
                4_864,
                14,
                2,
                151_936,
            ),
            ("Qwen/Qwen2.5-1.5B-Instruct", 1536, 8_960, 12, 2, 151_936),
            (
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                1536,
                8_960,
                12,
                2,
                151_936,
            ),
            ("Qwen/Qwen2.5-3B-Instruct", 2_048, 11_008, 16, 2, 151_936),
            (
                "Qwen/Qwen2.5-Coder-3B-Instruct",
                2_048,
                11_008,
                16,
                2,
                151_936,
            ),
            ("Qwen/Qwen2.5-7B-Instruct", 3584, 18_944, 28, 4, 152_064),
            (
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                3584,
                18_944,
                28,
                4,
                152_064,
            ),
            ("Qwen/Qwen2.5-14B-Instruct", 5_120, 13_824, 40, 8, 152_064),
            (
                "Qwen/Qwen2.5-Coder-14B-Instruct",
                5_120,
                13_824,
                40,
                8,
                152_064,
            ),
            ("Qwen/Qwen2.5-72B-Instruct", 8_192, 29_568, 64, 8, 152_064),
        ];

        for (model_id, hidden, inter, heads, kv_heads, vocab) in cases {
            let reference = model_config(model_id, hidden, inter, heads, kv_heads, vocab, 32_768);
            let from_reference = LayerMemoryConfig::from_model_config(&reference, &CUDA);
            let from_table = layer_memory_config_for(model_id, None, &CUDA);

            assert_eq!(
                from_table, from_reference,
                "the fallback table of '{model_id}' must match the real config.json of that model"
            );
        }

        // The MoE fixture of this repository (`tessella/models/tessella-moe-micro/config.json`, written
        // by `scripts/create_micro_moe.py`): 8 experts, 2 per token, `moe_intermediate_size` 2048, a
        // vocabulary of 32000 and a context of 2048 — the last two are the terms `TASK-86`/S93 added to
        // the row, which the row used to state as zero.
        let mut micro = model_config(
            "victorespada/tessella-moe-micro",
            1024,
            2048,
            8,
            2,
            32_000,
            2_048,
        );
        micro.num_experts = Some(8);
        micro.num_experts_per_tok = Some(2);
        micro.moe_intermediate_size = Some(2048);
        assert_eq!(
            layer_memory_config_for("victorespada/tessella-moe-micro", None, &CUDA),
            LayerMemoryConfig::from_model_config(&micro, &CUDA),
            "the fallback table of the local MoE fixture must match its config.json"
        );

        // The `0.5B` case has a constructor in the protocol crate: compare against it as well, so the
        // table is not checked against a single hand-written reference.
        assert_eq!(
            layer_memory_config_for("Qwen/Qwen2.5-Coder-0.5B-Instruct", None, &CUDA),
            LayerMemoryConfig::from_model_config(&ModelConfig::qwen2_5_0_5b(), &CUDA),
            "qwen2_5_0_5b() is the 0.5B reference of the protocol crate"
        );
    }

    /// `TASK-86`/S93: the two tables answer for the **same** models, because they match the same id
    /// fragments — the layer count of `tessella_protocol::get_total_layers` and the layer cost of
    /// [`FALLBACK_SHAPES`]. The test pins it in both directions, which is the drift S93 found: a model
    /// with a known count and the cost of an unknown one (1.5B: 28 layers, 386 MB per layer).
    #[test]
    fn a_model_of_an_unknown_size_states_nothing() {
        let unknown = layer_memory_config_for("acme/mystery-model", None, &CUDA);
        let (layers_of_256, per_layer) = compute_layer_budget(256, &unknown);
        assert_eq!(
            layers_of_256, 0,
            "a 256 MB claim cannot host a layer of an unknown model"
        );
        assert_eq!(
            per_layer, 386,
            "the unknown-model arithmetic is hidden 4096 / intermediate 11008: 386 MB per layer"
        );

        for model_id in [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "victorespada/tessella-moe-micro",
        ] {
            assert!(
                tessella_protocol::get_total_layers(model_id).is_some(),
                "'{model_id}' has a known layer count"
            );
            assert_ne!(
                layer_memory_config_for(model_id, None, &CUDA),
                unknown,
                "'{model_id}' has a known layer count, so its cost cannot be the unknown one"
            );
        }

        // The other direction: an id no table knows is the *unknown* model of both. That is what the
        // fixture `test-model` is for in the tests of `musivum`, where a 256 MB claim must not host a
        // layer of it — so it has a count in neither table and the cost of an unknown model here.
        assert_eq!(
            tessella_protocol::get_total_layers("test-model"),
            None,
            "`test-model` is the unknown-model fixture, not an entry of the registry"
        );
        assert_eq!(
            layer_memory_config_for("test-model", None, &CUDA),
            unknown,
            "an unknown id is sized with the conservative default"
        );
    }

    /// `TASK-82`/S89: a node that registers **without** a `model_config` — a canary, which reads no
    /// `config.json` — is planned for the model it declares, not for the conservative default. The
    /// figure is the one `swarm.sh` documents (`vram_gb = 1.2` ⇒ 12 layers of the 0.5B) and it is the
    /// difference between a canary that shares the plan of the nodes around it and one that takes a
    /// sliver: measured on 2026-09-25 (`CANARY=on --workers 2 --vram 1.2`), the canary got `[[0, 2]]`
    /// — the only node covering layer 0, so the mesh served nobody — while the two nodes that
    /// reported their own config got 12 layers each.
    #[test]
    fn a_node_that_declares_no_model_config_is_sized_for_its_model() {
        let declared = ModelConfig::qwen2_5_0_5b();
        let from_config = LayerMemoryConfig::from_model_config(&declared, &CUDA);
        let from_table = layer_memory_config_for(&declared.model_id, None, &CUDA);
        assert_eq!(
            from_table, from_config,
            "a canary is planned like the model it declares, not like an unknown one"
        );

        // 1.2 GB is `(1.2 * 1024.0) as u32`, the claim the registration of every harness node sends.
        let (layers, _) = compute_layer_budget(1_228, &from_table);
        assert_eq!(
            layers, 12,
            "1.2 GB of claim buys 12 layers of `{}`",
            declared.model_id
        );

        let conservative = layer_memory_config_for("acme/mystery-model", None, &CUDA);
        let (conservative_layers, _) = compute_layer_budget(1_228, &conservative);
        assert_eq!(
            conservative_layers, 2,
            "a model the table does not know keeps the conservative default (hidden 4096): that is \
             the range the canary was given before this entry"
        );
        assert!(
            conservative_layers < layers,
            "the entry has to make a difference, otherwise it is dead weight"
        );
    }

    /// `S52`/`TASK-46` (VR-03): the operator's `min_vram_gb` is the floor now, and the hard floor of
    /// the arithmetic is still there for a deployment that configures nothing (or nonsense).
    #[test]
    fn the_vram_floor_is_the_operator_value_and_never_below_the_hard_floor() {
        assert_eq!(effective_vram_floor_mb(0.5), 512, "the code default");
        assert_eq!(effective_vram_floor_mb(0.4), 409, "the example before S52");
        assert_eq!(effective_vram_floor_mb(2.0), 2048);
        assert_ne!(
            effective_vram_floor_mb(2.0),
            effective_vram_floor_mb(0.5),
            "two configured values must give two different floors"
        );
        assert_eq!(effective_vram_floor_mb(0.0), MIN_VRAM_MB);
        assert_eq!(effective_vram_floor_mb(-1.0), MIN_VRAM_MB);
        assert_eq!(
            effective_vram_floor_mb(f32::NAN),
            MIN_VRAM_MB,
            "a nonsense value falls back to the hard floor instead of admitting everything"
        );
    }

    /// The node that cannot host one layer used to be handed a one-layer shard anyway (or an expert
    /// shard with a wrong gather). The registration is refused now, and the reason carries the
    /// numbers so the operator can act on it.
    #[test]
    fn a_node_that_cannot_host_a_layer_gets_an_explicit_reason_in_mb() {
        let moe = moe(2048, 1408, 60, 1408);

        assert_eq!(
            compute_layer_budget(256, &moe).0,
            0,
            "the fixture must be too heavy for a 256 MB claim"
        );
        let breakdown = layer_budget(256, &moe, KV_SESSIONS_PER_NODE);
        let reason = capacity_reason(256, &moe).expect("it cannot host a layer");
        assert!(
            reason.contains("256")
                && reason.contains(&breakdown.usable_vram_mb.to_string())
                && reason.contains(&breakdown.per_layer_mb.to_string())
                && reason.contains(&breakdown.weights_per_layer_mb.to_string()),
            "the reason names the claim, the usable VRAM and both terms of the per-layer cost: \
             {reason}"
        );
        assert!(
            reason
                .to_lowercase()
                .contains("expert parallelism is not implemented"),
            "and it says why a partial shard is not the way out: {reason}"
        );

        assert_eq!(
            capacity_reason(8192, &moe),
            None,
            "8 GB does fit layers of this model"
        );
    }

    fn assignment(
        layer_range: (usize, usize),
        expert_range: Option<(usize, usize)>,
    ) -> WorkerAssignment {
        WorkerAssignment {
            layer_ranges: vec![layer_range],
            model_repo: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            replica_group_id: 0,
            expert_range,
        }
    }

    /// `S65`: qué significa «adoptar una asignación» en el worker.
    ///
    /// `reload_shard` comparaba **solo** el rango de capas y devolvía `Ok(())` cuando no cambiaba, así
    /// que una asignación que solo moviera los expertos —justo la que mandaría un orquestador con
    /// reparto 2-D— (o el modelo) se **ignoraba en silencio**, con el motor quedándose con la
    /// asignación vieja. Es exactamente la mitad que le faltó al autoscaler de expertos retirado en
    /// S44 («una mutación que ningún worker podía adoptar»). La decisión es una función pura para
    /// poder fijarla sin arrancar un motor, un modelo ni un daemon.
    #[test]
    fn adopting_an_assignment_reloads_the_weights_only_when_they_change() {
        let base = assignment((0, 12), None);

        assert!(
            !requires_weight_reload(&base, &assignment((0, 12), None)),
            "the same assignment does not reload anything"
        );
        assert!(
            requires_weight_reload(&base, &assignment((0, 18), None)),
            "another layer range selects other weights"
        );
        assert!(
            requires_weight_reload(&base, &assignment((0, 12), Some((0, 4)))),
            "an expert range that appears selects other weights"
        );
        assert!(
            requires_weight_reload(
                &assignment((0, 12), Some((0, 4))),
                &assignment((0, 12), Some((4, 8)))
            ),
            "another expert range selects other weights"
        );
        assert!(
            requires_weight_reload(
                &assignment((0, 12), Some((0, 4))),
                &assignment((0, 12), None)
            ),
            "and an expert range that disappears selects the whole layer again"
        );
        assert!(
            requires_weight_reload(
                &base,
                &WorkerAssignment {
                    model_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
                    ..assignment((0, 12), None)
                }
            ),
            "another model is other weights"
        );
        assert!(
            !requires_weight_reload(
                &base,
                &WorkerAssignment {
                    replica_group_id: 3,
                    ..assignment((0, 12), None)
                }
            ),
            "the replica group chooses no tensor: it is adopted without reloading the shard"
        );
    }
}
