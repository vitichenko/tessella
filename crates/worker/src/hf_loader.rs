use anyhow::Context;
use hf_hub::api::tokio::ApiBuilder;
use std::path::PathBuf;
use tracing::info;

pub struct HfArtifacts {
    pub shard_paths: Vec<PathBuf>,
}

/// Resolves only the index and config from HF Hub or local cache.
/// Index is optional for single-shard models.
pub async fn resolve_index(
    model_id: &str,
    token: Option<&str>,
    base_dir: &std::path::Path,
) -> anyhow::Result<(PathBuf, PathBuf, Option<serde_json::Value>)> {
    // 1. Check if model exists locally in base_dir
    let local_path = base_dir.join(model_id.split('/').next_back().unwrap_or(model_id));
    if local_path.exists() && local_path.join("config.json").exists() {
        info!("📂 Using local model from {:?}", local_path);
        let config_path = local_path.join("config.json");
        let tokenizer_path = local_path.join("tokenizer.json");
        let index_path = local_path.join("model.safetensors.index.json");

        let index = if index_path.exists() {
            let content = tokio::fs::read_to_string(&index_path).await?;
            Some(serde_json::from_str(&content)?)
        } else {
            None
        };

        return Ok((config_path, tokenizer_path, index));
    }

    let api = ApiBuilder::new()
        .with_token(token.map(String::from))
        .build()
        .context("Failed to build HF API")?;

    let repo = api.model(model_id.to_string());

    let config_path = repo.get("config.json").await?;

    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .context("Failed to download tokenizer.json")?;

    let _ = repo.get("tokenizer_config.json").await;
    let _ = repo.get("special_tokens_map.json").await;

    // index.json is optional
    let index = match repo.get("model.safetensors.index.json").await {
        Ok(path) => {
            let content = tokio::fs::read_to_string(&path).await?;
            Some(serde_json::from_str(&content)?)
        }
        Err(_) => None,
    };

    Ok((config_path, tokenizer_path, index))
}

/// Helper to determine if a tensor is relevant to the given worker's role.
///
/// `TASK-96`/S102: the shard is a **set** of ranges, so a tensor is relevant when any of its layer
/// indices is covered by any of them. The two globals keep their old rule, which the caller derives
/// from the same set: the node that owns the range starting at 0 needs the embedding table and the
/// node that owns the range reaching the last layer needs the final norm and the LM head.
fn is_tensor_relevant(
    name: &str,
    layer_ranges: &[(usize, usize)],
    is_entrypoint: bool,
    is_final: bool,
) -> bool {
    if name.starts_with("model.layers.") {
        if let Some(parts) = name.strip_prefix("model.layers.") {
            if let Some(idx_str) = parts.split('.').next() {
                if let Ok(layer_idx) = idx_str.parse::<usize>() {
                    return tessella_protocol::layer_ranges::covers(layer_ranges, layer_idx);
                }
            }
        }
    } else {
        if is_entrypoint && name.contains("embed_tokens") {
            return true;
        }
        if is_final && (name.contains("norm") || name.contains("lm_head")) {
            return true;
        }
    }
    false
}

/// Resolves model artifacts for a specific layer range.
///
/// # Errors
/// Returns error if model repo is unreachable or essential files are missing.
pub async fn resolve_shards_for_layers(
    model_id: &str,
    layer_ranges: &[(usize, usize)],
    is_entrypoint: bool,
    is_final: bool,
    token: Option<&str>,
    quantization: &tessella_core::models::QuantizationConfig,
    base_dir: &std::path::Path,
) -> anyhow::Result<HfArtifacts> {
    // 1. Check if model exists locally
    let local_path = base_dir.join(model_id.split('/').next_back().unwrap_or(model_id));
    if local_path.exists() && local_path.join("config.json").exists() {
        let shard_path = local_path.join("model.safetensors");
        if shard_path.exists() {
            return Ok(HfArtifacts {
                shard_paths: vec![shard_path],
            });
        }
    }

    info!(
        "📥 Resolving shards for layers {:?} of model {}...",
        layer_ranges, model_id
    );

    let api = ApiBuilder::new()
        .with_token(token.map(String::from))
        .build()
        .context("Failed to build HF API")?;

    let repo = std::sync::Arc::new(api.model(model_id.to_string()));

    // 1. Download config.json
    let _config_path = repo
        .get("config.json")
        .await
        .context("Failed to download config.json")?;

    let _tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .context("Failed to download tokenizer.json")?;

    let _ = repo.get("tokenizer_config.json").await;
    let _ = repo.get("special_tokens_map.json").await;

    let mut shard_paths = Vec::new();

    if *quantization == tessella_core::models::QuantizationConfig::Q8_0 {
        let info = repo.info().await.context("Failed to get repo info")?;
        let mut gguf_files = Vec::new();
        for sibling in info.siblings {
            if sibling.rfilename.ends_with(".gguf") {
                gguf_files.push(sibling.rfilename);
            }
        }

        if !gguf_files.is_empty() {
            let target_file = gguf_files
                .iter()
                .find(|f| f.to_lowercase().contains("q8_0"))
                .or_else(|| gguf_files.first())
                .unwrap();
            info!("  🔍 Found GGUF file: {}", target_file);
            let path = repo.get(target_file).await?;
            shard_paths.push(path);
            return Ok(HfArtifacts { shard_paths });
        } else {
            tracing::warn!(
                "⚠️ No GGUF files found in repo {}, falling back to safetensors.",
                model_id
            );
        }
    }

    // 2. Check for single shard or multi-shard
    if let Ok(path) = repo.get("model.safetensors").await {
        shard_paths.push(path);
        info!("  ✓ model.safetensors (single shard)");
    } else {
        // Multi-shard: download index and identify relevant shards
        let index_path = repo
            .get("model.safetensors.index.json")
            .await
            .context("model.safetensors or index not found")?;

        let index_content = tokio::fs::read_to_string(&index_path).await?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;

        let mut relevant_shards = std::collections::HashSet::new();

        if let Some(weight_map) = index["weight_map"].as_object() {
            for (tensor_name, shard_name) in weight_map {
                let shard_name = shard_name.as_str().context("Invalid shard name in index")?;

                if is_tensor_relevant(tensor_name, layer_ranges, is_entrypoint, is_final) {
                    relevant_shards.insert(shard_name.to_string());
                }
            }
        }

        info!("  🔍 Identified {} relevant shards", relevant_shards.len());

        // 3. Download identified shards in parallel with concurrency limit
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(3));
        let mut join_handles = Vec::new();

        for shard in relevant_shards {
            let repo_task = repo.clone();
            let sem = semaphore.clone();
            let shard_name = shard.clone();

            join_handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await?;

                let mut last_error = None;
                for attempt in 0..3 {
                    if attempt > 0 {
                        let delay = std::time::Duration::from_secs(1 << attempt);
                        tokio::time::sleep(delay).await;
                        tracing::warn!("Retrying shard {} (attempt {}/3)", shard_name, attempt + 1);
                    }

                    info!("  📥 Downloading shard: {}", shard_name);
                    match repo_task.get(&shard_name).await {
                        Ok(path) => return Ok(path),
                        Err(e) => last_error = Some(e),
                    }
                }
                Err(anyhow::anyhow!(
                    "Failed to download shard {} after 3 attempts: {:?}",
                    shard_name,
                    last_error
                ))
            }));
        }

        for handle in join_handles {
            let path = handle.await.context("Join handle failed")??;
            shard_paths.push(path);
        }
    }

    Ok(HfArtifacts { shard_paths })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_filtering_logic() {
        let index_json = serde_json::json!({
            "weight_map": {
                "model.embed_tokens.weight": "model-01.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-01.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-02.safetensors",
                "model.layers.2.self_attn.q_proj.weight": "model-03.safetensors",
                "model.norm.weight": "model-04.safetensors",
                "lm_head.weight": "model-04.safetensors"
            }
        });

        let mut relevant_shards = std::collections::HashSet::new();
        let layer_ranges = [(1usize, 2usize)]; // Only layer 1
        let is_entrypoint = false;
        let is_final = false;

        let index_opt = Some(index_json);

        if let Some(index) = index_opt {
            if let Some(weight_map) = index["weight_map"].as_object() {
                for (tensor_name, shard_name) in weight_map {
                    let shard_name = shard_name.as_str().unwrap();

                    if is_tensor_relevant(tensor_name, &layer_ranges, is_entrypoint, is_final) {
                        relevant_shards.insert(shard_name.to_string());
                    }
                }
            }
        }

        assert_eq!(relevant_shards.len(), 1);
        assert!(relevant_shards.contains("model-02.safetensors"));
    }

    #[test]
    fn test_entrypoint_global_tensors() {
        let index_json = serde_json::json!({
            "weight_map": {
                "model.embed_tokens.weight": "model-01.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-01.safetensors",
                "model.norm.weight": "model-04.safetensors"
            }
        });

        let mut relevant_shards = std::collections::HashSet::new();
        let layer_ranges = [(10usize, 20usize)];
        let is_entrypoint = true;
        let is_final = false;

        let index_opt = Some(index_json);

        if let Some(index) = index_opt {
            if let Some(weight_map) = index["weight_map"].as_object() {
                for (tensor_name, shard_name) in weight_map {
                    let shard_name = shard_name.as_str().unwrap();

                    if is_tensor_relevant(tensor_name, &layer_ranges, is_entrypoint, is_final) {
                        relevant_shards.insert(shard_name.to_string());
                    }
                }
            }
        }

        assert!(relevant_shards.contains("model-01.safetensors"));
        assert!(!relevant_shards.contains("model-04.safetensors"));
    }
}
