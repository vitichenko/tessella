use super::resolve_preferred_http_addr;
use super::{orchestrator_key_conflict, MusivumRelay, NetworkManager};
use libp2p;
use libp2p::identity::Keypair;
use serde_json::json;

/// A real peer id plus the multiaddr the orchestrator would advertise for it.
/// The `/p2p/<id>` component is checksummed by libp2p, so a placeholder
/// string would not parse and the test would not exercise the real path.
fn advertised() -> (String, String) {
    let peer_id = tessella_network::peer_id_from_keypair(&Keypair::generate_ed25519());
    let addr = format!("/ip4/203.0.113.5/tcp/54326/p2p/{}", peer_id);
    (peer_id, addr)
}

fn payload(peer_id: &str, p2p_addr: Option<&str>) -> serde_json::Value {
    json!({ "peer_id": peer_id, "p2p_port": 54326, "p2p_addr": p2p_addr })
}

/// H-18: `resolve_preferred_http_addr` used to parse the multiaddr and the URL by hand. These
/// tests pin the behaviour it must keep now that both strings go through
/// `tessella_core::net`.
#[test]
fn resolve_preferred_http_addr_prefers_the_tailscale_address() {
    assert_eq!(
        resolve_preferred_http_addr(
            "http://10.0.0.5:10001",
            "/ip4/100.64.0.1/tcp/9001/p2p/12D3KooWTest"
        ),
        "http://100.64.0.1:10001"
    );
    // The scheme and the port of the announced URL are preserved.
    assert_eq!(
        resolve_preferred_http_addr("https://worker.internal:10443", "/ip4/100.64.0.7/tcp/9001"),
        "https://100.64.0.7:10443"
    );
}

#[test]
fn resolve_preferred_http_addr_keeps_a_foreign_address_untouched() {
    // Not a 100.x (Tailscale/CGNAT) address: the announced URL is the best we have.
    assert_eq!(
        resolve_preferred_http_addr("http://10.0.0.5:10001", "/ip4/203.0.113.5/tcp/9001"),
        "http://10.0.0.5:10001"
    );
    // An IPv6 multiaddr is not a Tailscale (100.x) address either.
    assert_eq!(
        resolve_preferred_http_addr("http://10.0.0.5:10001", "/ip6/2001:db8::1/tcp/9001"),
        "http://10.0.0.5:10001"
    );
    // A peer that announces no P2P address at all.
    assert_eq!(
        resolve_preferred_http_addr("http://10.0.0.5:10001", ""),
        "http://10.0.0.5:10001"
    );
}

#[test]
fn resolve_preferred_http_addr_does_not_invent_a_host_for_an_unparseable_url() {
    // Before H-18 a malformed URL still produced `http://100.64.0.1` (the port was simply
    // dropped). Neither string is now parsed by hand, so the announced value is returned as it
    // is instead of being rewritten into something the caller never sent.
    assert_eq!(
        resolve_preferred_http_addr("not a url", "/ip4/100.64.0.1/tcp/9001"),
        "not a url"
    );
}

#[test]
fn test_from_payload_reads_peer_id_and_dialable_addr() {
    let (peer_id, addr) = advertised();
    let relay = MusivumRelay::from_payload(&payload(&peer_id, Some(&addr))).unwrap();

    assert_eq!(relay.peer_id.as_deref(), Some(peer_id.as_str()));
    assert_eq!(
        relay.p2p_addr.as_ref().map(ToString::to_string).as_deref(),
        Some(addr.as_str())
    );
    assert!(!relay.needs_addr());
}

#[test]
fn test_from_payload_requires_a_peer_id() {
    // The peer id is what authorizes the orchestrator's inbound connections,
    // so a payload without it is not usable at all.
    assert!(MusivumRelay::from_payload(&json!({ "p2p_addr": null })).is_err());
}

/// H-25 (`TASK-12`): what the worker announces as its real IP is the first **non-loopback**
/// IPv4 listener, not the first listener and not the configured address.
#[test]
fn real_ip_prefers_the_first_non_loopback_listener() {
    let listeners: Vec<libp2p::Multiaddr> = vec![
        "/ip4/127.0.0.1/tcp/9001".parse().unwrap(),
        "/ip4/192.168.1.24/tcp/9001".parse().unwrap(),
        "/ip4/10.0.0.9/tcp/9001".parse().unwrap(),
    ];
    assert_eq!(
        NetworkManager::real_ip_from_listeners(listeners.iter(), "203.0.113.5"),
        "192.168.1.24"
    );
}

/// The fallback is the address the operator configured: it is used whenever the swarm has no
/// usable listener (not started yet, loopback-only, or IPv6-only).
#[test]
fn real_ip_falls_back_to_the_configured_address() {
    let none: Vec<libp2p::Multiaddr> = Vec::new();
    assert_eq!(
        NetworkManager::real_ip_from_listeners(none.iter(), "203.0.113.5"),
        "203.0.113.5"
    );

    let loopback: Vec<libp2p::Multiaddr> = vec!["/ip4/127.0.0.1/tcp/9001".parse().unwrap()];
    assert_eq!(
        NetworkManager::real_ip_from_listeners(loopback.iter(), "203.0.113.5"),
        "203.0.113.5"
    );

    // An IPv6 listener is not an IPv4 address, and the value is announced in a
    // `/ip4/...` multiaddr, so it cannot be used here.
    let only_v6: Vec<libp2p::Multiaddr> = vec!["/ip6/2001:db8::1/tcp/9001".parse().unwrap()];
    assert_eq!(
        NetworkManager::real_ip_from_listeners(only_v6.iter(), "203.0.113.5"),
        "203.0.113.5"
    );
}

/// A wildcard listener is not loopback, so it is reported verbatim: it is what the kernel
/// bound, and overriding it is the job of the operator (`public_ip`), not of this helper.
#[test]
fn real_ip_reports_a_wildcard_listener_verbatim() {
    let wildcard: Vec<libp2p::Multiaddr> = vec!["/ip4/0.0.0.0/tcp/9001".parse().unwrap()];
    assert_eq!(
        NetworkManager::real_ip_from_listeners(wildcard.iter(), "203.0.113.5"),
        "0.0.0.0"
    );
}

#[test]
fn test_from_payload_accepts_null_p2p_addr() {
    // The orchestrator answers `null` when it cannot advertise a literal IP
    // (H-04): the worker keeps using HTTP and keeps retrying.
    let relay = MusivumRelay::from_payload(&payload("12D3KooWNotParsed", None)).unwrap();

    assert!(relay.needs_addr());
    assert_eq!(relay.peer_id.as_deref(), Some("12D3KooWNotParsed"));
}

#[test]
fn test_from_payload_drops_an_unparseable_p2p_addr() {
    // Regression for H-04: `/ip4//tcp/9080/...` used to be stored verbatim
    // and only reported by a warning, so the relay never came up.
    let relay =
        MusivumRelay::from_payload(&payload("12D3KooWNotParsed", Some("/ip4//tcp/9080"))).unwrap();

    assert!(
        relay.needs_addr(),
        "a malformed address must stay unknown so that re-discovery keeps running"
    );
}

#[test]
fn test_update_dials_once_per_advertised_address() {
    let (peer_id, addr) = advertised();
    let mut relay = MusivumRelay::default();
    assert!(relay.needs_addr());

    let first = relay
        .update(MusivumRelay::from_payload(&payload(&peer_id, Some(&addr))).unwrap())
        .expect("the first discovery must return the address to dial");
    assert_eq!(first.to_string(), addr);
    assert!(!relay.needs_addr());

    // The next heartbeat sees the same address: nothing new to dial.
    let again = MusivumRelay::from_payload(&payload(&peer_id, Some(&addr))).unwrap();
    assert!(relay.update(again).is_none());

    // A NAT remap advertises a different address → dial again.
    let remapped = format!("/ip4/198.51.100.7/tcp/60000/p2p/{}", peer_id);
    let fresh = MusivumRelay::from_payload(&payload(&peer_id, Some(&remapped))).unwrap();
    let dialed = relay
        .update(fresh)
        .expect("a changed address must be dialed");
    assert_eq!(dialed.to_string(), remapped);
}

#[test]
fn test_update_keeps_a_known_address_when_the_payload_has_none() {
    let (peer_id, addr) = advertised();
    let mut relay = MusivumRelay::from_payload(&payload(&peer_id, Some(&addr))).unwrap();

    assert!(relay
        .update(MusivumRelay::from_payload(&payload(&peer_id, None)).unwrap())
        .is_none());
    assert_eq!(
        relay.p2p_addr.as_ref().map(ToString::to_string).as_deref(),
        Some(addr.as_str()),
        "a temporarily missing advertisement must not drop a working relay"
    );
}

#[test]
fn test_update_learns_the_peer_id_without_a_dialable_addr() {
    // A failed boot fetch used to leave both fields empty forever, so the
    // orchestrator's inbound connections were rejected as "unregistered".
    let mut relay = MusivumRelay::default();
    let fresh = MusivumRelay::from_payload(&payload("12D3KooWLearned", None)).unwrap();

    assert!(relay.update(fresh).is_none());
    assert!(relay.needs_addr());
    assert!(relay.is_peer("12D3KooWLearned"));
}

#[test]
fn test_forget_addr_reenables_discovery_and_keeps_the_peer_id() {
    let (peer_id, addr) = advertised();
    let mut relay = MusivumRelay::from_payload(&payload(&peer_id, Some(&addr))).unwrap();

    assert!(relay.is_peer(&peer_id));
    assert!(!relay.is_peer("12D3KooWSomebodyElse"));

    assert!(relay.forget_addr());
    assert!(relay.needs_addr());
    assert_eq!(
        relay.peer_id.as_deref(),
        Some(peer_id.as_str()),
        "the peer id is stable across NAT remaps and must be kept"
    );
    assert!(!relay.forget_addr(), "there is nothing left to forget");
}

fn payload_with_key(
    peer_id: &str,
    p2p_addr: Option<&str>,
    orchestrator_pubkey: Option<&str>,
) -> serde_json::Value {
    json!({
        "peer_id": peer_id,
        "p2p_port": 54326,
        "p2p_addr": p2p_addr,
        "orchestrator_pubkey": orchestrator_pubkey,
    })
}

/// TASK-19: `/p2p_id` publica la pubkey del orquestador y el worker la conserva para poder
/// avisar si algún día deja de coincidir con la que el operador fijó. El campo es aditivo: un
/// orquestador anterior a S25 simplemente no lo manda.
#[test]
fn test_from_payload_reads_the_published_orchestrator_pubkey() {
    let (peer_id, addr) = advertised();
    let raw = "ab".repeat(32);

    let relay =
        MusivumRelay::from_payload(&payload_with_key(&peer_id, Some(&addr), Some(&raw))).unwrap();
    assert_eq!(relay.pubkey.as_deref(), Some(raw.as_str()));

    let legacy = MusivumRelay::from_payload(&payload(&peer_id, Some(&addr))).unwrap();
    assert_eq!(legacy.pubkey, None, "pre-S25 payloads carry no pubkey");

    let blank =
        MusivumRelay::from_payload(&payload_with_key(&peer_id, Some(&addr), Some("   "))).unwrap();
    assert_eq!(blank.pubkey, None, "a blank value is not a key");
}

/// Un payload sin pubkey no borra la que ya se conocía: el orquestador puede dejar de
/// publicarla (replica antigua) sin que el worker pierda la única prueba de identidad que tiene.
#[test]
fn test_update_keeps_the_last_pubkey_when_a_payload_does_not_carry_one() {
    let (peer_id, addr) = advertised();
    let raw = "cd".repeat(32);
    let mut relay =
        MusivumRelay::from_payload(&payload_with_key(&peer_id, Some(&addr), Some(&raw))).unwrap();

    relay.update(MusivumRelay::from_payload(&payload(&peer_id, Some(&addr))).unwrap());
    assert_eq!(relay.pubkey.as_deref(), Some(raw.as_str()));
}

/// Criterio 3 de TASK-19: con la clave fijada por el operador, una identidad distinta es una
/// señal de suplantación (o de que el orquestador cambió de clave). Un orquestador que todavía
/// no publica la clave no puede acusarse de nada.
#[test]
fn orchestrator_key_conflict_only_fires_on_a_real_mismatch() {
    let pinned = "ab".repeat(32);
    let other = "cd".repeat(32);

    assert!(!orchestrator_key_conflict(None, Some(&pinned)));
    assert!(!orchestrator_key_conflict(Some(&pinned), None));
    assert!(!orchestrator_key_conflict(Some(&pinned), Some(&pinned)));
    assert!(
        !orchestrator_key_conflict(Some(&pinned.to_uppercase()), Some(&pinned)),
        "hex is case-insensitive, so the casing of a pasted key is not a conflict"
    );
    assert!(orchestrator_key_conflict(Some(&pinned), Some(&other)));
}

/// `SOLID-89`: la superficie HTTP del daemon es un **puerto** (`tessella_ingress::ApiHost`) que
/// llega inyectado desde la raíz de composición (`tessella-node`), así que la costura se puede
/// observar desde este crate: un host que no enlaza ningún framework HTTP ocupa el lugar de la
/// superficie, y el daemon le entrega el estado que construyó con sus propias tripas — los
/// **mismos `Arc`** que comparte con su engine, que es lo que hace que el callback del stream de
/// dos saltos aterrice (un mapa nuevo aquí lo rompería en silencio).
///
/// La malla no se puede arrancar en esta suite (Tessella es GPU-only, `D-6`, y un
/// `InferenceEngine` real no llega a tener backend aquí), y es justo por eso que el puerto
/// importa: lo único que el daemon hace con la superficie es el `select!` de `run_api_and_p2p`
/// sobre `serve` (y `❌ API Server halted` cuando retorna), que es lo que este test ejercita
/// llamando a `manager.api_host` como lo hace esa función.
mod api_host_port {
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use candle_core::Device;
    use dashmap::DashMap;
    use tessella_core::callback::capability::CallbackCapabilities;
    use tessella_core::traits::DistributedInference;
    use tessella_inference::ShardedWeights;
    use tessella_ingress::{ApiHost, ApiState};
    use tessella_network::peer_network::create_peer_network_handle;
    use tessella_network::{generate_keypair, peer_id_from_keypair};
    use tessella_protocol::{ModelConfig, WorkerAssignment};
    use tokio::sync::RwLock;

    use super::{MusivumRelay, NetworkManager};
    use crate::config::WorkerConfig;
    use crate::inference_engine::InferenceEngine;
    use crate::routing_table::RoutingTable;
    use crate::runtime::WorkerRuntime;
    use crate::topology_cache::TopologyCache;

    /// Las capas de la fixture: sólo importa que el engine tenga una asignación coherente.
    const LAYERS: usize = 4;

    /// Un host que no enlaza ningún framework HTTP: registra lo que le entregan y retorna — es
    /// exactamente la rama «❌ API Server halted» del daemon.
    #[derive(Default)]
    struct StubApiHost {
        served: Mutex<Vec<(SocketAddr, ApiState)>>,
    }

    #[async_trait::async_trait]
    impl ApiHost for StubApiHost {
        async fn serve(&self, addr: SocketAddr, state: ApiState) -> anyhow::Result<()> {
            self.served.lock().unwrap().push((addr, state));
            Ok(())
        }
    }

    /// El engine de la fixture: el mismo que usa `inference_engine.rs` en sus pruebas — CPU, sin
    /// backend y sin pesos, porque en una máquina sin CUDA no hay otra cosa (`D-6`).
    fn engine_of(
        model_repo: &str,
        keypair: &tessella_network::NetworkKeypair,
    ) -> Arc<InferenceEngine> {
        Arc::new(InferenceEngine::new(
            vec![(
                (0, LAYERS),
                Arc::new(ShardedWeights {
                    layers: HashMap::new(),
                    global: HashMap::new(),
                    layer_range: (0, LAYERS),
                    expert_range: None,
                    quantization: tessella_core::models::QuantizationConfig::default(),
                }),
            )],
            ModelConfig {
                hidden_size: 4,
                num_hidden_layers: LAYERS,
                ..Default::default()
            },
            WorkerAssignment {
                layer_ranges: vec![(0, LAYERS)],
                model_repo: model_repo.to_string(),
                replica_group_id: 0,
                expert_range: None,
            },
            Device::Cpu,
            Arc::new(tessella_inference::backends::BackendRegistry::new()),
            None,
            None,
            Some(keypair.clone()),
            None,
            crate::callback_guard::CallbackGuard::new(
                tessella_core::callback::CallbackTargetPolicy::new(false, vec![]),
                None,
                Arc::new(RwLock::new(RoutingTable::new())),
            ),
            Arc::new(CallbackCapabilities::new()),
            Arc::new(tessella_core::telemetry::metrics::ExpertActivationRegistry::new()),
            crate::config::CanaryMode::Off,
            1024,
            60,
            tessella_core::telemetry::gpu_health::DEFAULT_OOM_STREAK,
            tessella_core::telemetry::gpu_health::DEFAULT_STREAK_WINDOW_SECS,
        ))
    }

    /// La base de datos de la fixture: nombrada por proceso para que dos ejecuciones en paralelo
    /// no se pisen.
    fn db_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("solid89-port-{}.db", std::process::id()))
    }

    #[tokio::test]
    async fn the_daemon_hands_the_injected_port_the_state_it_built() {
        let config = WorkerConfig::default();
        let keypair = generate_keypair();
        let engine = engine_of(&config.default_model, &keypair);

        // El mapa del stream de tokens viaja por el proceso (`WorkerService` → `NetworkManager`
        // → `InferenceEngine` → `ApiState`): la fixture lo conserva para poder probar que el
        // estado lleva **el mismo**, no una copia.
        let senders = Arc::new(DashMap::new());
        let runtime = Arc::new(WorkerRuntime {
            config: config.clone(),
            peer_id: peer_id_from_keypair(&keypair),
            orchestrator_url: config.orchestrator_url.clone(),
            bind_ip: "127.0.0.1".to_string(),
            p2p_addr: "/ip4/127.0.0.1/tcp/0".to_string(),
            http_client: reqwest::Client::new(),
            peer_failures: Arc::new(tessella_core::telemetry::metrics::PeerFailureRegistry::new()),
            hop_wait: Duration::from_secs(30),
            keypair: Arc::new(keypair.clone()),
            metrics: Arc::new(tessella_core::telemetry::WorkerMetrics::new()),
        });
        let storage = Arc::new(
            tessella_storage::SqliteStorage::new(db_path(), None)
                .expect("una base de datos SQLite temporal"),
        );
        let host = Arc::new(StubApiHost::default());
        let (peer_network, cmd_rx) = create_peer_network_handle(keypair.clone(), 10);

        let manager = Arc::new(NetworkManager::new(
            config.clone(),
            runtime.clone(),
            Arc::new(RwLock::new(RoutingTable::new())),
            Arc::new(RwLock::new(TopologyCache::new(Duration::from_secs(300)))),
            engine.clone(),
            storage.clone(),
            senders.clone(),
            Arc::new(tokio::sync::Notify::new()),
            Arc::new(RwLock::new(MusivumRelay::default())),
            std::path::PathBuf::from("/tmp/solid89-model"),
            None,
            host.clone(),
            peer_network,
            Some(cmd_rx),
        ));

        // El puerto es el inyectado, no uno que el daemon se fabrique.
        let injected: Arc<dyn ApiHost> = host.clone();
        assert!(Arc::ptr_eq(&manager.api_host.clone(), &injected));

        // El estado lo construye el daemon con sus tripas...
        let served_engine: Arc<dyn DistributedInference> = engine.clone();
        let state = manager.setup_api_state(served_engine.clone());

        assert!(
            state
                .dist_inference
                .as_ref()
                .is_some_and(|s| Arc::ptr_eq(s, &served_engine)),
            "la superficie sirve el engine del daemon, no otro"
        );
        assert_eq!(state.hop_timeout_secs, runtime.hop_wait.as_secs());
        assert_eq!(state.max_new_tokens, config.max_new_tokens);

        // ...y comparte con el engine la caché de capacidades que él llena al sellar cada
        // payload (H-10) y el mapa del stream de tokens (S9/`TASK-12`): una copia aquí rompería
        // el SSE de dos saltos sin que nada fallara.
        assert!(
            Arc::ptr_eq(
                state.callback_capabilities.as_ref().unwrap(),
                &engine.callback_capabilities
            ),
            "la caché de capacidades es la del engine"
        );
        assert!(
            Arc::ptr_eq(state.token_stream_senders.as_ref().unwrap(), &senders),
            "el mapa del stream de tokens es el del daemon"
        );
        assert!(Arc::ptr_eq(
            state.gpu_health.as_ref().unwrap(),
            &engine.gpu_health
        ));
        assert_eq!(
            state
                .current_assignment_version
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );

        // Si el almacenamiento ya contenía una versión guardada (p. ej. tras reinicio), setup_api_state la inicializa (MSV-19)
        storage.set_assignment_version(99).await.unwrap();
        let state2 = manager.setup_api_state(served_engine.clone());
        assert_eq!(
            state2
                .current_assignment_version
                .load(std::sync::atomic::Ordering::SeqCst),
            99
        );

        // Servirlo es lo único que el daemon le pide al puerto: `run_api_and_p2p` hace
        // `self.api_host.clone().serve(addr, api_state)` y registra un error si retorna. Aquí
        // retorna, así que un host sin Axum basta para la superficie completa.
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        manager
            .api_host
            .clone()
            .serve(addr, state.clone())
            .await
            .expect("un host que retorna no es un error");

        let served = host.served.lock().unwrap();
        assert_eq!(served.len(), 1, "el puerto se sirve una sola vez");
        assert_eq!(served[0].0, addr);
        assert!(
            Arc::ptr_eq(served[0].1.token_stream_senders.as_ref().unwrap(), &senders),
            "lo que recibe el puerto es el estado del daemon"
        );
    }
}
