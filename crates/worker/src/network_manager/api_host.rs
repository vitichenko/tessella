//! API host bootstrap and state management collaborator for NetworkManager (`SOLID-90`).

use std::net::SocketAddr;
use std::sync::Arc;
use tessella_core::error::DomainError;
use tessella_core::traits::DistributedInference;
use tessella_ingress::{ApiHost, ApiState};

use super::NetworkManager;

/// Builds the [`ApiState`] published by this worker node.
pub fn setup_api_state(
    network: &NetworkManager,
    service: Arc<dyn DistributedInference>,
) -> ApiState {
    let ceiling = crate::inference_engine::hop_token_ceiling(
        &network.engine.model_config,
        &network.engine.device,
        network.engine.max_input_tokens,
    );
    ApiState {
        dist_inference: Some(service),
        storage: Some(
            network.storage.clone() as Arc<dyn tessella_core::traits::StorageService + Send + Sync>
        ),
        credits: Some(
            network.storage.clone() as Arc<dyn tessella_core::traits::CreditService + Send + Sync>
        ),
        discovery: None,
        slashing: None,
        peer_slashing: None,
        musivum_state: None,
        max_new_tokens: network.runtime.config.max_new_tokens,
        max_input_tokens: ceiling.map(|ceiling| ceiling.tokens),
        max_input_tokens_source: ceiling.map(|ceiling| ceiling.source.as_str()),
        hop_timeout_secs: network.runtime.hop_wait.as_secs(),
        musivum_keypair: None,
        admin_token: network.runtime.config.admin_token.clone(),
        default_model: network.runtime.config.default_model.clone(),
        model_config: crate::canary::registration_model_config(
            &network.runtime.config,
            Some(&network.engine.model_config),
        ),
        peer_failures: Some(network.runtime.peer_failures.clone()),
        gpu_health: Some(network.engine.gpu_health.clone()),
        tokenizer: network
            .load_tokenizer()
            .ok()
            .map(|t| t as Arc<dyn tessella_core::traits::TokenizerService>),
        token_stream_senders: Some(network.token_stream_senders.clone()),
        orchestrator_url: Some(network.runtime.config.orchestrator_url.clone()),
        own_api_url: network.engine.own_api_url.clone(),
        replay_cache: Arc::new(tessella_protocol::signing::ReplayCache::new()),
        callback_capabilities: Some(network.engine.callback_capabilities.clone()),
        metrics: Some(network.runtime.metrics.clone()),
        current_assignment_version: Arc::new(std::sync::atomic::AtomicU64::new(
            network.storage.get_assignment_version_sync().unwrap_or(0),
        )),
    }
}

/// Serves the API over the injected port.
pub async fn serve_api(
    api_host: Arc<dyn ApiHost>,
    addr: SocketAddr,
    state: ApiState,
) -> Result<(), DomainError> {
    api_host
        .serve(addr, state)
        .await
        .map_err(|e| DomainError::Network(format!("API host server error: {e}")))
}
