use super::*;
use tessella_core::constants::inference::HopCeilingSource;
use tessella_network::NetworkKeypair;

const NOW: u64 = 1_700_000_000;

/// The work ticket of the fixture. Since `TASK-57`/S63 every proof carries one: a proof
/// without the session's authorisation is not a claim the orchestrator could pay or judge.
fn ticket() -> tessella_core::models::WorkTicket {
    tessella_core::models::WorkTicket {
        request_id: "work-1".to_string(),
        model_id: "model-a".to_string(),
        entry_peer_id: "12D3KooWEntry".to_string(),
        issued_at: NOW,
        expires_at: NOW + 600,
        audit_layer: None,
        audit_replicas: 2,
        audit_canary: None,
        audit_reward: 0.0,
        orchestrator_signature: vec![7; 64],
    }
}

// H-16 (retired with the transport): the P2P response decode used to `unwrap()` bytes produced
// by `forward_hidden`, so a malformed payload panicked the task serving `/moe/inference`. That
// transport was removed in `TASK-25`/S31 — nothing ever produced an `InferenceRequest` — so
// the decode helper, `handle_p2p_inference` and their two tests went with it. The only
// hidden-state wire that remains is the HTTP one (`POST /v1/forward`), whose edge checks are
// covered by `token_stream_auth_tests` in `tessella-api` and by
// `tests/http_pipeline_two_hops.rs` here.

/// `TASK-07` layer 2: a hop's proof carries the hashes of **both** ends of its hop, so the
/// orchestrator can chain them (`output_hash(N) == input_hash(N+1)`), and it carries the work
/// ticket of the session, which is what makes the credit authoritative (H-12).
#[test]
fn hop_work_proof_chains_both_ends_and_keeps_the_ticket() {
    let first = build_hop_work_proof(
        "work-1",
        "peer-a".to_string(),
        (4, 12),
        Some(b"input-of-the-first-shard"),
        b"output-of-the-first-shard",
        ticket(),
        NOW,
    );
    let second = build_hop_work_proof(
        "work-1",
        "peer-b".to_string(),
        (12, 24),
        // Exactly the bytes the first hop produced: this is the property the orchestrator uses.
        Some(b"output-of-the-first-shard"),
        b"output-of-the-second-shard",
        ticket(),
        NOW,
    );

    assert_eq!(
        first.output_hash, second.input_hash,
        "the chain output_hash(N) == input_hash(N+1) must hold for honest hops"
    );
    assert_ne!(first.input_hash, first.output_hash);
    assert_eq!(first.credits_delta, 8.0);
    assert_eq!(second.credits_delta, 12.0);
    assert_eq!(first.work_ticket, ticket());
    assert_eq!(first.layer_range, (4, 12));
    assert_eq!(second.layer_range, (12, 24));
}

/// `TASK-27`: the evidence is the digest of the **tensor**, so two honest executions of the same
/// step (the primary shard and the replica the audit sends the same input to) produce the same
/// `output_hash`. With the old definition — the whole sealed payload — the nonce/`iat`/signature
/// of each execution made every pair of claims differ, which is what left capa 3 with nothing to
/// compare.
#[test]
fn two_honest_executions_of_the_same_input_produce_the_same_output_hash() {
    let input = b"the-tensor-bytes-of-one-prompt".to_vec();

    let primary = build_hop_work_proof(
        "work-1",
        "peer-a".to_string(),
        (12, 24),
        Some(&input),
        b"the-tensor-bytes-of-the-answer",
        ticket(),
        NOW,
    );
    let replica = build_hop_work_proof(
        "work-1",
        "peer-b".to_string(),
        (12, 24),
        Some(&input),
        b"the-tensor-bytes-of-the-answer",
        ticket(),
        NOW + 3,
    );

    assert_eq!(primary.output_hash, replica.output_hash);
    assert_eq!(primary.input_hash, replica.input_hash);
    assert_ne!(primary.worker_id, replica.worker_id);

    // And a liar is detectable: same input, different output.
    let liar = build_hop_work_proof(
        "work-1",
        "peer-b".to_string(),
        (12, 24),
        Some(&input),
        b"some-other-tensor",
        ticket(),
        NOW,
    );
    assert_eq!(liar.input_hash, primary.input_hash);
    assert_ne!(liar.output_hash, primary.output_hash);
}

/// `TASK-28`/S34: the entry node serves its own layers from the tokens, so its proof has no
/// incoming hidden state (all zeros) and its output digest is what the next hop hashes as its own
/// input: the first link of the chain is real even when the node that starts it is local.
#[test]
fn the_entry_node_proof_has_no_input_and_links_to_the_first_hop() {
    let entry = build_hop_work_proof(
        "work-1",
        "peer-entry".to_string(),
        (0, 4),
        None,
        b"hidden-state-emitted-by-the-entry-node",
        ticket(),
        NOW,
    );
    let first_hop = build_hop_work_proof(
        "work-1",
        "peer-a".to_string(),
        (4, 12),
        Some(b"hidden-state-emitted-by-the-entry-node"),
        b"output-of-the-first-shard",
        ticket(),
        NOW,
    );

    assert_eq!(entry.input_hash, [0u8; 32]);
    assert_eq!(entry.credits_delta, 4.0);
    assert_eq!(entry.layer_range, (0, 4));
    assert_eq!(
        entry.output_hash, first_hop.input_hash,
        "the chain must start at the entry node's own work"
    );
}

/// `TASK-96`/S102 · the point of the whole card: a node holding **two disjoint ranges** runs each
/// step on the backend built for the range that covers it, and refuses the hop that lands in the hole
/// between them instead of answering it with the other range's weights.
#[tokio::test]
async fn a_dual_range_node_serves_each_step_with_the_backend_of_its_range() {
    let (engine, first_applied, _) = engine_with_spy_backend_on((0, 4)).await;

    // The same node, now with a second chunk of the same chain: the assignment is the set, and the
    // second range gets its own backend with its own counter, so the test can see which one ran.
    let second_applied = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let second_began = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    {
        let mut assignment = engine.assignment.write().await;
        assignment.layer_ranges = vec![(0, 4), (8, 12)];
    }
    engine.backends.write().await.push((
        (8, 12),
        spy_backend(second_applied.clone(), second_began, None),
    ));

    let keypair = engine
        .keypair
        .clone()
        .expect("the fixture engine carries a keypair");

    // Step 3: the first range, on its own backend. `layer_idx` is the last layer the hop before us
    // executed, and it is not part of the signed preimage (the seal covers the tensor and the
    // session's identity), which is what lets this fixture place the step where it wants.
    let mut first_hop = payload_with_ids(&keypair, "work-first", None, 0, 3, 8101);
    first_hop.layer_idx = 2;
    engine
        .forward_hidden(&first_hop.to_bytes().expect("payload bytes"))
        .await
        .expect("the step of the first range is served");
    assert_eq!(
        first_applied.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the first range ran on its own backend"
    );
    assert_eq!(
        second_applied.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "and the second one did not run for a step it does not own"
    );

    // Step 8: the second range, on the backend that was built for it.
    let mut second_hop = payload_with_ids(&keypair, "work-second", None, 0, 3, 8102);
    second_hop.layer_idx = 7;
    engine
        .forward_hidden(&second_hop.to_bytes().expect("payload bytes"))
        .await
        .expect("the step of the second range is served");
    assert_eq!(
        second_applied.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the second range ran on the backend built for it"
    );
    assert_eq!(
        first_applied.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "and the first backend was not asked to run layers 8..11"
    );

    // The hole between the two ranges: nobody serves layer 4, so the hop is refused by name instead
    // of being answered with the weights of another range (invariant 17).
    let mut hole = payload_with_ids(&keypair, "work-hole", None, 0, 3, 8103);
    hole.layer_idx = 3; // the step is layer 4, which is not in [0, 4) ∪ [8, 12)
    let refusal = engine
        .forward_hidden(&hole.to_bytes().expect("payload bytes"))
        .await
        .expect_err("a hop for a range this node does not serve is refused")
        .to_string();
    assert!(
        refusal.contains("[(0, 4), (8, 12)]"),
        "the refusal names what the node does serve: {refusal}"
    );
    assert_eq!(
        second_applied.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the refused hop never reached a backend"
    );
}

/// `TASK-57`/S63: the canonical of a proof is **one** shape — no field is omitted, so two nodes
/// holding the same evidence produce the same bytes and the orchestrator finds the authorisation
/// inside the body it verified. The `skip_serializing_if` that used to drop the ticket existed so
/// that a pre-`TASK-07` fleet kept verifying; a hop without a ticket now writes no evidence at
/// all (`sign_and_store_work_proof` warns and returns).
#[test]
fn the_canonical_of_a_proof_always_carries_its_authorisation() {
    let proof = build_hop_work_proof(
        "work-1",
        "peer-a".to_string(),
        (0, 1),
        Some(b"in"),
        b"out",
        ticket(),
        NOW,
    );
    let bytes = proof.to_canonical_bytes().unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();

    assert!(text.contains("\"work_ticket\""), "{text}");
    assert!(text.contains("\"layer_range\""), "{text}");
    assert!(text.contains("\"input_hash\""), "{text}");
    assert_eq!(proof.credits_delta, 1.0);
    assert_ne!(
        proof.input_hash, [0u8; 32],
        "a real hop always hashes its input"
    );
}

fn client_pubkey(keypair: &NetworkKeypair) -> Vec<u8> {
    keypair.public().encode_protobuf()
}

/// The orchestrator half of the pair: a libp2p ed25519 key whose public part the
/// worker holds as `orchestrator_public_key` (see `worker/src/service.rs`, which
/// decodes the 32 key bytes of the protobuf encoding).
fn orchestrator_verifier(keypair: &NetworkKeypair) -> ed25519_dalek::VerifyingKey {
    let protobuf = keypair.public().encode_protobuf();
    let key_bytes: [u8; 32] = protobuf[4..].try_into().expect("ed25519 key bytes");
    ed25519_dalek::VerifyingKey::from_bytes(&key_bytes).expect("ed25519 verifying key")
}

/// Signs `ticket` the way `handle_karma_ticket` does: over `to_canonical_bytes()`.
fn signed_ticket(
    orchestrator: &NetworkKeypair,
    bound_to: &[u8],
    tier: QueuePriority,
) -> KarmaTicket {
    let mut ticket = KarmaTicket {
        client_pubkey: bound_to.to_vec(),
        tier,
        expiry: NOW + 600,
        orchestrator_signature: Vec::new(),
    };
    let canonical = ticket.to_canonical_bytes().expect("canonical ticket");
    ticket.orchestrator_signature = orchestrator.sign(&canonical).expect("orchestrator sig");
    ticket
}

#[test]
fn bound_ticket_grants_its_tier() {
    let orchestrator = tessella_network::generate_keypair();
    let verifier = orchestrator_verifier(&orchestrator);
    let requester = tessella_network::generate_keypair();
    let ticket = signed_ticket(
        &orchestrator,
        &client_pubkey(&requester),
        QueuePriority::Vip,
    );

    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&requester),
            "req-1",
            NOW,
            Some(&verifier)
        ),
        QueuePriority::Vip
    );
}

/// The bug of H-13: a ticket issued for a rich peer used to upgrade **any** bearer.
#[test]
fn ticket_bound_to_another_peer_is_downgraded_to_leech() {
    let orchestrator = tessella_network::generate_keypair();
    let verifier = orchestrator_verifier(&orchestrator);
    let rich = tessella_network::generate_keypair();
    let attacker = tessella_network::generate_keypair();
    let ticket = signed_ticket(&orchestrator, &client_pubkey(&rich), QueuePriority::Vip);

    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&attacker),
            "req-2",
            NOW,
            Some(&verifier)
        ),
        QueuePriority::Leech
    );
}

/// Tickets issued before this change carry no `client_pubkey`: they cannot be bound,
/// so they must not grant priority either (they expire within the hour).
#[test]
fn unbound_ticket_is_downgraded_to_leech() {
    let orchestrator = tessella_network::generate_keypair();
    let verifier = orchestrator_verifier(&orchestrator);
    let requester = tessella_network::generate_keypair();
    let ticket = signed_ticket(&orchestrator, &[], QueuePriority::Vip);

    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&requester),
            "req-3",
            NOW,
            Some(&verifier)
        ),
        QueuePriority::Leech
    );
}

#[test]
fn expired_ticket_is_downgraded_to_leech() {
    let orchestrator = tessella_network::generate_keypair();
    let verifier = orchestrator_verifier(&orchestrator);
    let requester = tessella_network::generate_keypair();
    let ticket = signed_ticket(
        &orchestrator,
        &client_pubkey(&requester),
        QueuePriority::Standard,
    );

    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&requester),
            "req-4",
            NOW + 601,
            Some(&verifier)
        ),
        QueuePriority::Leech
    );

    // ...and it is still honoured one second before it expires: the signature covers
    // the binding, not the clock.
    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&requester),
            "req-4",
            NOW + 599,
            Some(&verifier)
        ),
        QueuePriority::Standard
    );
}

#[test]
fn ticket_signed_by_another_key_is_downgraded_to_leech() {
    let orchestrator = tessella_network::generate_keypair();
    let verifier = orchestrator_verifier(&orchestrator);
    // A rogue "orchestrator": it signs a perfectly bound ticket of its own.
    let rogue = tessella_network::generate_keypair();
    let requester = tessella_network::generate_keypair();
    let ticket = signed_ticket(&rogue, &client_pubkey(&requester), QueuePriority::Vip);

    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&requester),
            "req-5",
            NOW,
            Some(&verifier)
        ),
        QueuePriority::Leech
    );
}

#[test]
fn without_an_orchestrator_key_no_ticket_grants_priority() {
    let orchestrator = tessella_network::generate_keypair();
    let requester = tessella_network::generate_keypair();
    let ticket = signed_ticket(
        &orchestrator,
        &client_pubkey(&requester),
        QueuePriority::Vip,
    );

    assert_eq!(
        resolve_ticket_priority(&ticket, &client_pubkey(&requester), "req-6", NOW, None),
        QueuePriority::Leech
    );
}

// ---------------------------------------------------------------------------------
// H-09 / TASK-05 — the payload that travels hop by hop, sealed with the local key
// ---------------------------------------------------------------------------------

/// The exact DTO a hop receives: sealed with the keypair the worker persists, verified with
/// the engine's own replay cache. No GPU, no backend and no socket are involved.
#[test]
fn sealed_hidden_state_payload_is_verified_and_a_replay_is_rejected() {
    let keypair = tessella_network::generate_keypair();
    let mut payload =
        HiddenStatePayload::from_bytes(&sample_payload_bytes()).expect("sample payload");
    tessella_network::seal_signed_request(&keypair, &mut payload, 4242, NOW)
        .expect("sealing the payload");

    assert_eq!(payload.signature_version, 1);
    assert_eq!(payload.client_pubkey.len(), 36);

    let cache = ReplayCache::new();
    assert_eq!(
        tessella_network::verify_signed_request(&payload, &cache, NOW, "forward_hidden"),
        Ok(())
    );
    let replay =
        tessella_network::verify_signed_request(&payload, &cache, NOW + 1, "forward_hidden");
    assert!(
        matches!(replay, Err(tessella_network::SignedRequestError::Replay { nonce, .. }) if nonce == 4242),
        "{replay:?}"
    );
}

/// Criterion 1 with the DTO that actually crosses a hop: rewriting the callback destination
/// after the entry node signed it must be rejected by the next hop (before this step the
/// signature only covered the session id).
#[test]
fn tampering_with_a_sealed_payload_breaks_the_signature() {
    let keypair = tessella_network::generate_keypair();
    let mut payload =
        HiddenStatePayload::from_bytes(&sample_payload_bytes()).expect("sample payload");
    tessella_network::seal_signed_request(&keypair, &mut payload, 1, NOW).expect("seal");

    let mut tampered = payload.clone();
    tampered.entry_node_url = Some("http://attacker.example/v1/token_stream/x".to_string());

    let cache = ReplayCache::new();
    assert_eq!(
        tessella_network::verify_signed_request(&tampered, &cache, NOW, "forward_hidden"),
        Err(tessella_network::SignedRequestError::InvalidSignature)
    );
}

/// H-10 (mitigation 2): the callback POST carries `Authorization: Bearer <hex signature of the
/// payload>` and the entry node accepts exactly that value, because both sides go through the
/// same primitive (`tessella_core::callback::capability`). This is the contract between the final shard
/// and the handler; the two-node exercise of the whole callback is `TASK-12`/S17.
#[test]
fn the_token_stream_callback_carries_the_capability_the_entry_node_issued() {
    let signature = vec![7u8; 64];
    let token_payload = tessella_protocol::TokenStreamPayload {
        request_id: "req-callback".to_string(),
        token: String::new(),
        finish_reason: None,
    };
    let url = "http://203.0.113.9:9001/v1/token_stream/req-callback";
    let request = token_stream_request(&reqwest::Client::new(), url, &signature, &token_payload)
        .build()
        .expect("the callback request must be buildable");

    assert_eq!(request.method(), reqwest::Method::POST);
    assert_eq!(request.url().as_str(), url);

    // The entry node of this pipeline: it sealed the payload, so it issued the capability.
    let entry_node = tessella_core::callback::capability::CallbackCapabilities::new();
    let now = std::time::Instant::now();
    entry_node.register("req-callback", &signature, now);

    let presented = request
        .headers()
        .get(reqwest::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim_start_matches("Bearer "))
        .expect("the callback must present a capability");
    assert_eq!(
        presented,
        tessella_core::callback::capability::callback_capability(&signature)
    );
    assert_eq!(
        entry_node.authorize("req-callback", presented, now),
        tessella_core::callback::capability::CallbackAuthz::Accepted
    );
}

/// A `Payload` as it looks inside a hop between shards: unsealed (the engine seals it right
/// before emitting it, see `Session::forward`).
fn sample_payload() -> HiddenStatePayload {
    HiddenStatePayload {
        request_id: "req-seal".to_string(),
        session_id: Some("session-seal".to_string()),
        seq_start_pos: 0,
        layer_idx: 3,
        shape: vec![1, 1, 2],
        dtype: tessella_inference::hidden_state::DTypeTag::F32,
        data: vec![0u8; 8],
        checksum: 0,
        entry_node_url: Some("http://203.0.113.9:9001".to_string()),
        signature: Vec::new(),
        client_pubkey: Vec::new(),
        karma_ticket: None,
        work_ticket: None,
        signature_version: 0,
        nonce: 0,
        iat: 0,
        tokens: Vec::new(),
    }
}

fn sample_payload_bytes() -> Vec<u8> {
    sample_payload().to_bytes().expect("payload bytes")
}

/// `TASK-63`/S69: the Karma ticket of the session grant travels **inside the payload** the
/// session seals — the live path — and the worker keeps no other copy of it. The retired
/// `Arc<Mutex<Option<KarmaTicket>>>` was written on every grant and read by nobody, and its
/// clone ended up in an `ApiState` field no handler consulted, so a regression would not have
/// been visible as a failure: this test fixes the half that *is* the contract.
#[test]
fn the_session_ticket_travels_in_the_payload_it_seals() {
    let mut payload = sample_payload();
    let karma = tessella_core::models::KarmaTicket {
        client_pubkey: vec![1, 2, 3],
        tier: tessella_protocol::QueuePriority::Vip,
        expiry: NOW + 600,
        orchestrator_signature: vec![9; 64],
    };
    // The id the engine issued for this conversation (`TASK-78`/S85): what travels is **this**, not
    // the work id of the turn, or every hop would name a different session.
    const ISSUED: &str = "sess-6f1c0a2b3d4e5f60718293a4b5c6d7e8";
    apply_session_metadata(
        &mut payload,
        ISSUED,
        &Some("http://203.0.113.9:9001".to_string()),
        &Some(karma),
        &Some(ticket()),
    );

    // The ticket travels **in the payload** (`KarmaTicket` has no `PartialEq`: compare the
    // fields that matter instead of the whole ticket).
    match &payload.karma_ticket {
        Some(travelling) => {
            assert_eq!(
                travelling.tier,
                tessella_protocol::QueuePriority::Vip,
                "the grant's ticket is in the payload, with its tier"
            );
            assert_eq!(travelling.expiry, NOW + 600);
            assert_eq!(travelling.client_pubkey, vec![1, 2, 3]);
        }
        None => panic!("the grant's ticket must travel in the payload it seals"),
    }
    assert_eq!(
        payload.work_ticket.as_ref().map(|t| t.request_id.as_str()),
        Some(ticket().request_id.as_str()),
        "and so is the authorisation that makes the credit payable"
    );
    assert_eq!(
        payload.request_id, "req-seal",
        "the helper does not touch the work id the payload already carried"
    );
    assert_eq!(
        payload.session_id.as_deref(),
        Some(ISSUED),
        "el id que emite el nodo es el que viaja en cada hop (no el id del trabajo del turno)"
    );
    assert_ne!(
        payload.session_id.as_deref(),
        Some(ticket().request_id.as_str()),
        "y no el del trabajo: dos turnos de la misma conversación son dos trabajos y una sesión"
    );
}

/// TASK-19: la clave que `/p2p_id` publica es exactamente la que fija `orchestrator_public_key`
/// en el worker, así que los `KarmaTicket` que el orquestador firmó siguen concediendo su tier
/// (también después de un reinicio, que es lo que antes los invalidaba); con otra clave no se
/// concede nada (fail-safe).
#[test]
fn the_published_orchestrator_pubkey_pins_the_tickets_it_signs() {
    let orchestrator = tessella_network::generate_keypair();
    let published = tessella_network::ed25519_public_key_hex(&orchestrator).unwrap();
    let pinned = crate::config::parse_orchestrator_public_key(&published)
        .expect("the published raw key must parse");

    assert_eq!(
        pinned,
        orchestrator_verifier(&orchestrator),
        "the published hex describes the same verifying key this module already derives"
    );

    let requester = tessella_network::generate_keypair();
    let ticket = signed_ticket(
        &orchestrator,
        &client_pubkey(&requester),
        QueuePriority::Vip,
    );

    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&requester),
            "req-pinned",
            NOW,
            Some(&pinned)
        ),
        QueuePriority::Vip
    );

    let impostor = orchestrator_verifier(&tessella_network::generate_keypair());
    assert_eq!(
        resolve_ticket_priority(
            &ticket,
            &client_pubkey(&requester),
            "req-pinned",
            NOW,
            Some(&impostor)
        ),
        QueuePriority::Leech
    );
}

// -------------------------------------------------------------------------------------
// TASK-47 / S53 · the transport ceiling of one hop (option B: declare it, never chunk)
// -------------------------------------------------------------------------------------

/// A model config with just the field the ceiling reads.
fn model_of_width(hidden_size: usize) -> ModelConfig {
    ModelConfig {
        hidden_size,
        ..Default::default()
    }
}

/// The ceiling is arithmetic: `MAX_HIDDEN_STATE_BYTES / (hidden_size × bytes per element)`.
/// CUDA activations are F16 (2 bytes) and a CPU/Metal node is F32 (4), so the same wire carries
/// **half** the tokens there — the reason the ceiling cannot be a constant per model.
#[test]
fn the_hop_ceiling_follows_the_model_width_and_the_device_dtype() {
    assert_eq!(
        hop_token_ceiling(&model_of_width(3584), &Device::Cpu, 0),
        Some(HopCeiling {
            tokens: 4_681,
            source: HopCeilingSource::Wire,
            wire_tokens: Some(4_681),
            declared_tokens: 0,
        })
    );
    assert_eq!(
        hop_token_ceiling(&model_of_width(896), &Device::Cpu, 0).map(|ceiling| ceiling.tokens),
        Some(18_724)
    );
    assert_eq!(
        hop_token_ceiling(&model_of_width(3584), &Device::Cpu, 0).map(|ceiling| ceiling.tokens),
        max_input_tokens_per_hop(3584, activation_dtype_bytes(&Device::Cpu))
    );
    // An unknown width skips the check: `None`, never a 0-token ceiling.
    assert_eq!(hop_token_ceiling(&model_of_width(0), &Device::Cpu, 0), None);
}

/// `TASK-76`/S83: the published ceiling is the `min` of the wire and of **what the operator
/// declares this node's VRAM can prefill**, and it travels with the reason. The `S76` measurement
/// is the case that matters: a node whose declaration is far below the wire (1,2 GB of VRAM with a
/// 1.5B model) used to publish the wire's number and die with `CUDA_ERROR_OUT_OF_MEMORY` at a
/// prompt **32 times** below it.
#[test]
fn the_declared_vram_ceiling_lowers_the_wire_and_names_itself() {
    // The 1.5B case of S76: the wire carries 21 845 tokens (F16, 1536-wide) and the node declares
    // it can prefill 2 048. What a client may send is the declaration, and the reason is `vram`.
    let declared = hop_token_ceiling(&model_of_width(1536), &Device::Cpu, 2_048)
        .expect("both limits are known");
    assert_eq!(declared.tokens, 2_048);
    assert_eq!(declared.source, HopCeilingSource::Vram);
    assert_eq!(declared.wire_tokens, Some(21_845 / 2)); // CPU activations are F32, not F16
    assert_eq!(declared.declared_tokens, 2_048);

    // A declaration above the wire does not raise anything: the wire still binds.
    let above = hop_token_ceiling(&model_of_width(3584), &Device::Cpu, 10_000_000)
        .expect("both limits are known");
    assert_eq!(above.tokens, 4_681);
    assert_eq!(above.source, HopCeilingSource::Wire);

    // The two agree: the same number with two reasons, which is not "only the wire".
    let tie = hop_token_ceiling(&model_of_width(3584), &Device::Cpu, 4_681)
        .expect("both limits are known");
    assert_eq!(tie.tokens, 4_681);
    assert_eq!(tie.source, HopCeilingSource::Both);

    // An unreadable model config no longer hides the declaration: the node knows its VRAM.
    let vram_only = hop_token_ceiling(&model_of_width(0), &Device::Cpu, 1_024)
        .expect("a declaration is a known ceiling");
    assert_eq!(vram_only.tokens, 1_024);
    assert_eq!(vram_only.source, HopCeilingSource::Vram);
    assert_eq!(vram_only.wire_tokens, None);
}

/// The rejection guard is `>`, never `>=`: exactly the ceiling still travels — warned, because it
/// is inside the last 20 % of the budget — and one token over is rejected.
#[test]
fn the_transport_verdict_only_rejects_what_does_not_fit() {
    // 64 MiB / (3584 × 2) = 9 362 tokens (the 7B case of VR-06), with nothing declared.
    let wire = HopCeiling {
        tokens: 9_362,
        source: HopCeilingSource::Wire,
        wire_tokens: Some(9_362),
        declared_tokens: 0,
    };
    assert_eq!(
        hop_transport_verdict(9_362, 3584, 2, 0),
        HopTransport::NearLimit {
            tokens: 9_362,
            ceiling: wire,
        },
        "exactly the ceiling must still travel (the guard is `>`)"
    );
    assert_eq!(
        hop_transport_verdict(9_363, 3584, 2, 0),
        HopTransport::TooLarge {
            tokens: 9_363,
            ceiling: wire,
        }
    );
    assert_eq!(hop_transport_verdict(7_489, 3584, 2, 0), HopTransport::Fits);
    assert_eq!(
        hop_transport_verdict(7_490, 3584, 2, 0),
        HopTransport::NearLimit {
            tokens: 7_490,
            ceiling: wire,
        }
    );
    // An unknown size never rejects traffic (the exact payload byte check is the backstop).
    assert_eq!(
        hop_transport_verdict(usize::MAX, 0, 2, 0),
        HopTransport::Fits
    );

    // `TASK-76`/S83: the verdict is taken against the `min`, so a declaration below the wire
    // rejects a prompt the wire would have carried — before the prefill is paid for.
    let declared = HopCeiling {
        tokens: 2_048,
        source: HopCeilingSource::Vram,
        wire_tokens: Some(9_362),
        declared_tokens: 2_048,
    };
    assert_eq!(
        hop_transport_verdict(2_049, 3584, 2, 2_048),
        HopTransport::TooLarge {
            tokens: 2_049,
            ceiling: declared,
        }
    );
    assert_eq!(
        hop_transport_verdict(2_048, 3584, 2, 2_048),
        HopTransport::NearLimit {
            tokens: 2_048,
            ceiling: declared,
        },
        "the declaration is the whole budget, so its last token is already `NearLimit`"
    );
}

/// The rejection is a `Validation` whose message carries **the numbers**: what arrived, how big
/// the payload would have been, the wire limit and how many tokens fit. Before this change the
/// client saw a bare `413` from an intermediate node — after the prefill had been paid for.
#[test]
fn the_rejection_names_the_prompt_the_payload_and_the_ceiling() {
    let wire_ceiling =
        hop_ceiling(max_input_tokens_per_hop(3584, 2), 0).expect("the wire is a known ceiling");
    let error = tessella_core::error::prompt_too_long_error(12_000, 3584, 2, &wire_ceiling);
    assert!(matches!(error, DomainError::Validation(_)), "{error:?}");
    assert!(tessella_core::error::is_prompt_too_long(&error));
    let message = error.to_string();
    for expected in [
        "prompt_too_long",
        "12000",    // tokens that arrived
        "86016000", // 12 000 × 3584 × 2 bytes: the hidden state it would have needed
        "67108864", // MAX_HIDDEN_STATE_BYTES
        "9362",     // tokens that do fit
        "source: wire",
        "POST /v1/forward",
    ] {
        assert!(
            message.contains(expected),
            "{expected} missing from: {message}"
        );
    }
}

/// `TASK-76`/S83: when the limit that binds is the node's **own declaration**, the refusal says so
/// and names the declaration — that is the difference between "send a shorter prompt" and "shard
/// the model differently", which is the whole reason the number travels with its source.
#[test]
fn the_rejection_of_a_declared_ceiling_names_the_declaration() {
    let declared =
        hop_ceiling(max_input_tokens_per_hop(1536, 2), 2_048).expect("both limits are known");
    let error = tessella_core::error::prompt_too_long_error(4_096, 1536, 2, &declared);
    assert!(tessella_core::error::is_prompt_too_long(&error));
    let message = error.to_string();
    for expected in [
        "prompt_too_long",
        "4096", // tokens that arrived
        "2048", // the declared ceiling
        "max_input_tokens",
        "source: vram",
    ] {
        assert!(
            message.contains(expected),
            "{expected} missing from: {message}"
        );
    }
    // The declaration binds, which is the point: the wire would have carried this prompt.
    assert_eq!(declared.tokens, 2_048);
    assert!(declared.wire_tokens.unwrap() > declared.tokens);
}
// -----------------------------------------------------------------------------------------
// TASK-68 / S75 · F-1: no step whose prefix never passed through this node
// -----------------------------------------------------------------------------------------

/// A backend that answers **anything** with a plausible tensor (the input itself) and counts how
/// many times it was asked to run its layers.
///
/// It is the honest stand-in for the real backend in a suite that cannot use a GPU (`D-6`): the
/// corruption of `F-1` happens *inside* the backend — an **empty** K/V cache plus a decode step
/// with no mask (`qwen2.rs:426-441`, `attention.rs:206-212`) — so the only thing this fixture can
/// measure is what keeps the backend out of that situation: **whether the engine asks it at all**.
/// That is exactly the claim of the guard.
struct SpyBackend {
    applied: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// How many **backend** sessions it opened. It is what tells a resumed session from a fresh
    /// one (`TASK-75`/`S82`): a node that finds the K/V cache of the previous turn asks the
    /// backend for a new session exactly **once**.
    began: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    config: tessella_inference::backends::BackendConfig,
    capabilities: tessella_inference::backends::BackendCapabilities,
    /// Un fallo inyectable, apagable desde el test (`TASK-77`/S84): es la única forma de reproducir
    /// sin GPU el `prefill` que no cupo **y** su vuelta —el mismo engine, la petición siguiente—,
    /// porque sobre `Device::Cpu` nada falla solo.
    fault: Option<SpyFault>,
}

/// Un fallo del backend que el test puede encender y apagar: `active` a `true` = todo paso de capas
/// falla con `message`. Un `OOM` real es exactamente esto —transitorio—, y el bug de `S76` fue
/// tratarlo como permanente.
#[derive(Clone)]
struct SpyFault {
    active: std::sync::Arc<std::sync::atomic::AtomicBool>,
    message: String,
}

impl SpyFault {
    fn current(&self) -> Option<DomainError> {
        self.active
            .load(std::sync::atomic::Ordering::Relaxed)
            .then(|| DomainError::Inference(self.message.clone()))
    }
}

/// La sonda de device de un test: contesta mientras `alive` esté a `true` (`TASK-77`/S84) y cuenta
/// cuántas veces se le preguntó —que es cómo se fija **cuándo se paga**: un `OOM` no la consulta—.
/// Sobre un `Device::Cpu` la sonda real contesta siempre, así que la caída de device —y su vuelta— se
/// simulan aquí, que es el único sitio determinista para hacerlo sin GPU.
fn probe_that_answers(
    alive: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> (DeviceProbe, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = calls.clone();
    let probe = Box::new(move |_device: &Device| {
        counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if alive.load(std::sync::atomic::Ordering::Relaxed) {
            Ok(())
        } else {
            Err("CUDA_ERROR_ILLEGAL_ADDRESS: the device is gone".to_string())
        }
    });
    (probe, calls)
}

struct SpySession {
    applied: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pos_offset: usize,
    fault: Option<SpyFault>,
}

impl tessella_inference::backends::InferenceBackend for SpyBackend {
    fn model_id(&self) -> &str {
        "spy"
    }

    fn config(&self) -> &tessella_inference::backends::BackendConfig {
        &self.config
    }

    fn capabilities(&self) -> &tessella_inference::backends::BackendCapabilities {
        &self.capabilities
    }

    fn begin_session(
        &self,
    ) -> Result<Box<dyn tessella_inference::backends::InferenceSession>, DomainError> {
        self.began
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Box::new(SpySession {
            applied: self.applied.clone(),
            pos_offset: 0,
            fault: self.fault.clone(),
        }))
    }
}

impl tessella_inference::backends::InferenceSession for SpySession {
    fn forward(&mut self, tokens: &[u32]) -> Result<candle_core::Tensor, DomainError> {
        // A backend's `forward` answers with a **hidden state**, not with the token ids: the dtype
        // and the width have to be the ones a payload accepts, or the entry-node path of a session
        // cannot be exercised without a GPU. `TASK-75`/`S82` is the first test that needs it; the
        // hop tests build their payloads directly and never call this.
        let width = 4;
        let mut values = Vec::with_capacity(tokens.len() * width);
        for token in tokens {
            for _ in 0..width {
                values.push(*token as f32);
            }
        }
        candle_core::Tensor::from_vec(values, (1, tokens.len(), width), &Device::Cpu)
            .map_err(|e| DomainError::Inference(e.to_string()))
    }

    fn apply_layers(
        &mut self,
        x: candle_core::Tensor,
        _range: (usize, usize),
        seq_start_pos: usize,
    ) -> Result<candle_core::Tensor, DomainError> {
        self.applied
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // El fallo inyectado va **antes** que todo lo demás: el nodo tiene que verlo como un error
        // del paso de capas, que es lo que un `OOM` de verdad es (`TASK-77`/S84).
        if let Some(fault) = self.fault.as_ref().and_then(SpyFault::current) {
            return Err(fault);
        }
        self.pos_offset = seq_start_pos + x.dim(1).unwrap_or(1);
        // A real shard receives **either** a float hidden state (the hop path) **or** the token ids
        // of a prefill (`Session::forward` embeds them itself), and it answers with a float hidden
        // state in both cases. The hop tests only need the payload to round-trip, so a float input
        // is echoed unchanged; the entry-node tests need a dtype a payload accepts (`F32`), which is
        // what the token-id branch produces (`TASK-75`/`S82` is their first user).
        if x.dtype() == candle_core::DType::F32 {
            return Ok(x);
        }
        let width = x.dim(2).unwrap_or(4);
        let ids = x
            .flatten_all()
            .and_then(|flat| flat.to_vec1::<u32>())
            .map_err(|e| DomainError::Inference(e.to_string()))?;
        let mut values = Vec::with_capacity(ids.len() * width);
        for id in &ids {
            for _ in 0..width {
                values.push(*id as f32);
            }
        }
        candle_core::Tensor::from_vec(values, (1, ids.len(), width), x.device())
            .map_err(|e| DomainError::Inference(e.to_string()))
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

const SPY_LAYERS: (usize, usize) = (8, 24);
const SPY_TOTAL_LAYERS: usize = 24;

/// The K/V slot a **hop** stores its session in: the continuity key *and* the range that served the
/// step (`TASK-96`/S102). A node with two ranges owns two caches for the same conversation, so the
/// slot names both. The entry-node path (`Session`) keeps the plain key: it runs on the head backend
/// for the whole conversation.
fn hop_slot(key: &str) -> String {
    crate::inference_engine::cache_key(key, SPY_LAYERS)
}
/// Los segundos entre latidos que la fixture le declara al engine (`TASK-77`/S84): el mismo número
/// que `worker.toml`. Ningún test espera a que pasen — llaman a `revalidate_gpu` o entran por una
/// petición—, pero el valor viaja a `GET /status` como `gpu.revalidate_secs`.
const HEARTBEAT_SECS: u64 = 30;

/// An engine whose backend is [`SpyBackend`], on `Device::Cpu` and with no router: everything the
/// hop path of `forward_hidden` touches, and nothing that needs a GPU, a file or a network.
async fn engine_with_spy_backend() -> (Arc<InferenceEngine>, Arc<std::sync::atomic::AtomicUsize>) {
    let (engine, applied, _began) = engine_with_spy_backend_on(SPY_LAYERS).await;
    (engine, applied)
}

/// The same engine, on the layer range of your choice, plus the counter of backend sessions.
///
/// The range matters for anything that goes through `Session::forward`: a node that owns the
/// **last** layers never routes anything (`layer_range.1 < total_layers` is false), so a test that
/// wants the router to be called — or wants a refusal coming back from the pipeline — has to give
/// this engine a middle shard.
async fn engine_with_spy_backend_on(
    layer_range: (usize, usize),
) -> (
    Arc<InferenceEngine>,
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    engine_on(layer_range, crate::config::CanaryMode::Off, 0).await
}

/// The same engine, with this node **declaring** what its VRAM can prefill (`TASK-76`/S83).
///
/// It is the only way a test can drive the `vram` half of the ceiling: the fixture's `hidden_size`
/// is `4` and its activations are `f32`, so the `wire` half lands around four million tokens and no
/// test would reach it. `declared_tokens = 0` means undeclared, exactly as in `worker.toml`.
async fn engine_with_declared_ceiling(
    layer_range: (usize, usize),
    declared_tokens: usize,
) -> (
    Arc<InferenceEngine>,
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    engine_on(layer_range, crate::config::CanaryMode::Off, declared_tokens).await
}

/// El banco de pruebas del engine: rango de capas, canary, techo declarado y —de `TASK-77`/S84— un
/// fallo inyectable del backend y una sonda de device. Los tres primeros son los que usan las
/// fixtures de arriba; los dos últimos sólo los usan los tests del nodo que se cae y vuelve.
///
/// La asimetría del `canary` es el punto de `TASK-71`/S78: un canary contesta cada paso con un tensor
/// sabido-distinto sin cargar ningún shard, así que la fixture no puede darle un backend que pudiera
/// contestar bien (y los tests lo afirman: `engine.backend` se queda vacío).
async fn engine_on(
    layer_range: (usize, usize),
    canary: crate::config::CanaryMode,
    declared_tokens: usize,
) -> (
    Arc<InferenceEngine>,
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    engine_with_gpu_fault(layer_range, canary, declared_tokens, None, None).await
}

/// The backend every fixture here runs on: it counts the sessions it opens and the steps it applies,
/// and it can fail on demand. It is a function and not a literal inside the fixture because
/// `TASK-96`/S102 needs **one per range**.
fn spy_backend(
    applied: Arc<std::sync::atomic::AtomicUsize>,
    began: Arc<std::sync::atomic::AtomicUsize>,
    fault: Option<SpyFault>,
) -> Arc<dyn tessella_inference::backends::InferenceBackend> {
    std::sync::Arc::new(SpyBackend {
        applied,
        began,
        fault,
        config: tessella_inference::backends::BackendConfig {
            vocab_size: 32,
            max_seq_len: 512,
            eos_token_ids: vec![0],
            supports_system_prompt: true,
        },
        capabilities: tessella_inference::backends::BackendCapabilities {
            supports_kv_cache: true,
            supports_chat_template: true,
            supports_moe_routing: false,
            supports_sliding_window: false,
            supports_quantized_weights: false,
            default_stop_token_ids: vec![0],
            default_temperature: None,
        },
    })
}

/// Un engine de `TASK-77`/S84: un backend que falla con `message` mientras `fault` esté encendido y
/// una sonda de device que contesta mientras `alive` esté a `true`. Los dos son del test: apagarlos
/// es cómo se reproduce la vuelta **sin reiniciar nada**, que es la mitad que faltaba.
async fn engine_with_gpu_fault(
    layer_range: (usize, usize),
    canary: crate::config::CanaryMode,
    declared_tokens: usize,
    fault: Option<SpyFault>,
    probe: Option<DeviceProbe>,
) -> (
    Arc<InferenceEngine>,
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    let applied = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let began = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // A real keypair: `Session::forward` seals every payload it emits (H-09) and refuses to work
    // without one, so a test that goes through the entry-node path needs it. Reading it back from
    // the engine is how a test signs the payloads of the *same* identity.
    let keypair = tessella_network::generate_keypair();
    let spy = spy_backend(applied.clone(), began.clone(), fault);

    let mut engine = InferenceEngine::new(
        // One shard, one backend: the single-range node the whole suite is built on
        // (`TASK-96`/S102). The multi-range shape is a set of these and is exercised by its own
        // tests.
        vec![(
            layer_range,
            std::sync::Arc::new(tessella_inference::shard_loader::ShardedWeights {
                layers: std::collections::HashMap::new(),
                global: std::collections::HashMap::new(),
                layer_range,
                expert_range: None,
                quantization: tessella_core::models::QuantizationConfig::default(),
            }),
        )],
        ModelConfig {
            hidden_size: 4,
            num_hidden_layers: SPY_TOTAL_LAYERS,
            ..Default::default()
        },
        tessella_core::models::WorkerAssignment {
            layer_ranges: vec![layer_range],
            model_repo: "spy-model".to_string(),
            replica_group_id: 0,
            expert_range: None,
        },
        Device::Cpu,
        std::sync::Arc::new(tessella_inference::backends::BackendRegistry::new()),
        None,
        None,
        Some(keypair),
        None,
        crate::callback_guard::CallbackGuard::new(
            tessella_core::callback::CallbackTargetPolicy::new(false, vec![]),
            None,
            std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::routing_table::RoutingTable::new(),
            )),
        ),
        std::sync::Arc::new(tessella_core::callback::capability::CallbackCapabilities::new()),
        std::sync::Arc::new(tessella_core::telemetry::metrics::ExpertActivationRegistry::new()),
        canary,
        declared_tokens,
        // El compás del latido, que es con el que un nodo caído se revalida (`TASK-77`/S84). Es una
        // constante de la fixture: los tests de la vuelta la llaman a mano (`revalidate_gpu`) o
        // entran por una petición (`refuse_if_gpu_down`) — nunca esperan 30 s reales.
        HEARTBEAT_SECS,
        tessella_core::telemetry::gpu_health::DEFAULT_OOM_STREAK,
        tessella_core::telemetry::gpu_health::DEFAULT_STREAK_WINDOW_SECS,
    );
    // La sonda real `device_probe` contesta siempre sobre `Device::Cpu`, así que una fixture que
    // necesite la caída del device la sustituye aquí. `TASK-77`/S84: es el único punto del engine
    // por el que se decide si un fallo de device declara caído al nodo.
    if let Some(probe) = probe {
        engine.set_device_probe(probe);
    }
    // A canary has **no** backend on purpose (`initialize_models` stops before creating one in
    // production, so the fixture mirrors it): handing it a shard would let it answer correctly and
    // the test would pass for the wrong reason.
    if !canary.is_on() {
        *engine.backends.write().await = vec![(layer_range, spy)];
    }
    (std::sync::Arc::new(engine), applied, began)
}

/// The payload an **upstream hop** receives: `tokens` positions of hidden state, a declared
/// `seq_start_pos` and the canonical v1 seal (so `forward_hidden` authenticates it). The nonce is a
/// parameter because the engine's replay cache counts `(session_id, nonce)` exactly once.
fn hop_payload(
    keypair: &NetworkKeypair,
    rid: &str,
    seq_start_pos: usize,
    tokens: usize,
    nonce: u64,
) -> HiddenStatePayload {
    // The default fixture: the continuity id **is** the work id, which is what a client that sends
    // no `session_id` produces (`apply_session_metadata`) and what every hop emitted before
    // `TASK-75`/`S82`. The difference between the two ids is what `F-7` is about, and it has its own
    // fixture below.
    payload_with_ids(keypair, rid, Some(rid), seq_start_pos, tokens, nonce)
}

/// The payload of a **conversation**: the work id the orchestrator minted for *this* request
/// (`TASK-07`, new every turn) plus the client's continuity id (the same one every turn).
fn payload_with_ids(
    keypair: &NetworkKeypair,
    request_id: &str,
    session_id: Option<&str>,
    seq_start_pos: usize,
    tokens: usize,
    nonce: u64,
) -> HiddenStatePayload {
    let tensor =
        candle_core::Tensor::from_vec(vec![0.5_f32; tokens * 4], (1, tokens, 4), &Device::Cpu)
            .expect("a tiny CPU tensor");
    let mut payload =
        HiddenStatePayload::from_tensor(&tensor, request_id.to_string(), seq_start_pos as u32, 7)
            .expect("the payload is built from a real tensor");
    payload.session_id = session_id.map(str::to_string);
    tessella_network::seal_signed_request(
        keypair,
        &mut payload,
        nonce,
        tessella_core::time::unix_now_secs(),
    )
    .expect("the fixture signs with the canonical preimage v1");
    payload
}

/// `TASK-82`/S89, the half that touches the engine: a canary gets through `initialize_models`
/// with **no tokenizer and no model directory** (the `models_dir` of the operator may be empty)
/// and it keeps serving the step a ticket marks. `TASK-71`/S78 proved it serves with no backend;
/// what `S89` adds is that the boot does not ask for the model files either.
///
/// The two assertions after the call are one criterion: the boot returns `Ok` **and** leaves no
/// resolved directory behind — a boot that had downloaded the model would fail the second one.
#[tokio::test]
async fn a_canary_initializes_without_a_tokenizer_or_a_model_directory() {
    let (engine, _applied, _began) =
        engine_on(SPY_LAYERS, crate::config::CanaryMode::Echo, 0).await;
    let models_dir =
        std::env::temp_dir().join(format!("tessella-s89-engine-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&models_dir);

    engine
        .initialize_models(&models_dir, None)
        .await
        .expect("a canary boots with no tokenizer and no model directory: it reads neither");

    assert!(
        engine.backends.read().await.is_empty(),
        "no backend is created: a canary is deployable next to a real mesh"
    );
    assert!(
        !models_dir.exists(),
        "`initialize_models` must not resolve nor create the model directory of a canary"
    );
}

/// The other half: the ordinary node still **asks for its tokenizer by name**. Dropping the
/// requirement for the canary cannot open the normal path — a node with no canary mode that is
/// handed no tokenizer has to stop *before* it creates a backend, and say which piece is missing.
#[tokio::test]
async fn an_ordinary_node_without_a_tokenizer_fails_naming_it() {
    let (engine, _applied, _began) = engine_on(SPY_LAYERS, crate::config::CanaryMode::Off, 0).await;

    let error = engine
        .initialize_models(std::path::Path::new("./models"), None)
        .await
        .expect_err("a node that loads weights cannot boot without a tokenizer")
        .to_string();
    assert!(
        error.contains("tokeniz"),
        "the refusal has to name the missing piece: {error}"
    );
}

/// The acceptance test of the canary half of `TASK-71`/S78, at the point that matters: a node in
/// canary mode serves a step **without a backend and without a weight**, and the tensor it
/// answers with has the shape and dtype of a real one while its bytes are not what a shard would
/// have produced.
///
/// The payload carries a ticket whose marked step is **this** node's range, which is how the
/// audit asks for the isolated answer it compares (`TASK-70`/S77): the canary is exactly the node
/// the orchestrator can prove wrong without executing the model, so this is the shape the
/// orchestrator sees.
#[tokio::test]
async fn a_canary_answers_a_step_without_a_backend_or_a_weight() {
    let (engine, applied, began) = engine_on(SPY_LAYERS, crate::config::CanaryMode::Echo, 0).await;
    assert!(
        engine.backends.read().await.is_empty(),
        "a canary never creates a backend: that is what makes it deployable next to a real mesh"
    );

    let keypair = tessella_network::generate_keypair();
    let mut payload = hop_payload(&keypair, "req-canary", 0, 2, 4242);
    payload.work_ticket = Some(tessella_core::models::WorkTicket {
        request_id: "work-canary".to_string(),
        model_id: "spy-model".to_string(),
        entry_peer_id: "entry-peer".to_string(),
        issued_at: tessella_core::time::unix_now_secs(),
        expires_at: u64::MAX,
        audit_layer: Some(SPY_LAYERS.0 as u32),
        audit_replicas: 2,
        audit_canary: Some("the-canary-peer".to_string()),
        audit_reward: 1.0,
        orchestrator_signature: vec![5; 64],
    });

    let served = engine
        .forward_hidden(&payload.to_bytes().expect("payload bytes"))
        .await
        .expect("the marked step of a canary is answered in isolation, with no backend");

    let answer = HiddenStatePayload::from_bytes(&served).expect("the answer is a payload");
    assert_eq!(
        answer.shape, payload.shape,
        "the canary's answer has to have the shape of a real one: what trips is the digest, \
             never a shape error"
    );
    assert_eq!(
        format!("{:?}", answer.dtype),
        format!("{:?}", payload.dtype),
        "the dtype travels unchanged too"
    );
    assert_eq!(
        answer.data, payload.data,
        "`echo` answers the input unchanged — the known-wrong tensor the orchestrator expects"
    );
    assert_eq!(
        applied.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "not one layer is applied"
    );
    assert_eq!(
        began.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "not one backend session is created"
    );
}

/// **The bug of `F-1`, in one assertion**: a decode step (`1` position) that declares
/// `seq_start_pos = 12` on a node that never prefilled that session used to be **served** — the
/// engine created a session with an empty K/V cache and answered with a plausible tensor — with no
/// error, no `warn!` and no log line. This test is red against that behaviour (`expect_err` gets
/// `Ok`), and it also pins the *cost* of the fix: the refusal happens **before** the layer loop, so
/// nothing is computed for a step that cannot be answered.
#[tokio::test]
async fn a_decode_step_whose_session_this_node_never_prefilled_is_refused_before_any_compute() {
    let (engine, applied) = engine_with_spy_backend().await;
    let keypair = tessella_network::generate_keypair();
    let payload = hop_payload(&keypair, "req-f1", 12, 1, 4242);

    let error = engine
        .forward_hidden(&payload.to_bytes().expect("payload bytes"))
        .await
        .expect_err("a step whose prefix never passed through this node must not be served");

    assert!(
        tessella_core::error::is_missing_prefill(&error),
        "the refusal must carry the code the API and the router of the upstream node read: {error}"
    );
    let message = error.to_string();
    for expected in [
        "missing_prefill",
        "req-f1",             // the session that is missing here
        "seq_start_pos = 12", // the prefix the payload declared
        "[8, 24)",            // the layers of *this* node
    ] {
        assert!(
            message.contains(expected),
            "{expected} missing from: {message}"
        );
    }
    assert_eq!(
        applied.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the refusal must happen before `apply_layers`: no compute is paid for a step that \
             cannot be answered"
    );
}

/// The same defect has a second door: the TTL sweeper (`SESSION_TTL_SECS`,
/// `session_timestamps`) drops a session idle for more than 300 s, and the next step of that
/// session arrives with `seq_start_pos > 0`. A client that comes back after the TTL must read the
/// same **named** failure, not a plausible answer built on an empty cache.
#[tokio::test]
async fn a_session_evicted_by_the_ttl_produces_the_same_named_refusal_for_free() {
    let (engine, applied) = engine_with_spy_backend().await;
    let keypair = tessella_network::generate_keypair();

    // 1. A prefill (the whole prompt, `seq_start_pos = 0`) is served: this node owns the session.
    let prefill = hop_payload(&keypair, "req-ttl", 0, 3, 7);
    engine
        .forward_hidden(&prefill.to_bytes().expect("payload bytes"))
        .await
        .expect("the first step of a session is a normal hop");
    assert_eq!(applied.load(std::sync::atomic::Ordering::Relaxed), 1);

    // 2. The sweeper evicts it (what `run_session_sweeper` does with `idle > SESSION_TTL_SECS`).
    engine.sessions.remove(&hop_slot("req-ttl"));
    assert!(!engine.sessions.contains_key(&hop_slot("req-ttl")));

    // 3. The next step of the same session is a decode step that declares the prefix: refused.
    let decode = hop_payload(&keypair, "req-ttl", 12, 1, 8);
    let error = engine
        .forward_hidden(&decode.to_bytes().expect("payload bytes"))
        .await
        .expect_err("an expired session cannot answer a step whose prefix it no longer has");
    assert!(tessella_core::error::is_missing_prefill(&error), "{error}");
    assert_eq!(
        applied.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the refused step must not reach the backend either"
    );
}

/// The other half of a guard: it must **not** touch legitimate traffic. A decode step of a live
/// session declares `seq_start_pos > 0` by definition — that is what a decode step *is* — so the
/// refusal is conditioned on the session being **new**, not on the offset being non-zero.
#[tokio::test]
async fn a_legit_decode_step_of_a_live_session_is_served_as_before() {
    let (engine, applied) = engine_with_spy_backend().await;
    let keypair = tessella_network::generate_keypair();

    let prefill = hop_payload(&keypair, "req-multi-turn", 0, 3, 11);
    engine
        .forward_hidden(&prefill.to_bytes().expect("payload bytes"))
        .await
        .expect("the prefill is served");

    let decode = hop_payload(&keypair, "req-multi-turn", 3, 1, 12);
    let answer = engine
        .forward_hidden(&decode.to_bytes().expect("payload bytes"))
        .await
        .expect("the session is alive here: this step has to be served");
    let answer = HiddenStatePayload::from_bytes(&answer).expect("a real payload comes back");
    assert_eq!(
        answer.seq_start_pos, 3,
        "the prefix the payload declared survives the hop"
    );
    assert_eq!(
        applied.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "both steps reached the backend: the guard adds no false positive"
    );
}

// ─── `TASK-75`/`S82` · F-7: la clave de continuidad de una sesión ─────────────────────────────

/// A router that **echoes** the payload it was handed (so the hop path returns the very bytes this
/// node emitted) or **refuses** the step with the guard of `F-1` (the entry-node path has to react
/// to that). `Session::forward` needs one to reach the pipeline at all; the hop path needs one
/// whenever this node is not the tail.
struct StubRouter {
    refuse: bool,
}

impl StubRouter {
    fn echoing() -> Self {
        Self { refuse: false }
    }

    /// Refuses every step with the refusal of `TASK-68`/`S75`, as a hop that never prefilled the
    /// session does.
    fn refusing() -> Self {
        Self { refuse: true }
    }
}

#[async_trait::async_trait]
impl tessella_core::traits::PipelineRouter for StubRouter {
    async fn route(
        &self,
        _layer_start: usize,
        _total_layers: usize,
        payload_bytes: Vec<u8>,
        _model_id: &str,
        _audit: Option<tessella_core::traits::AuditOrder>,
    ) -> Result<Vec<u8>, DomainError> {
        if self.refuse {
            return Err(tessella_core::error::missing_prefill_error(
                "conv-7",
                12,
                (0, 8),
            ));
        }
        Ok(payload_bytes)
    }

    async fn refresh(&self) -> Result<(), DomainError> {
        Ok(())
    }
}

/// **The defect of `F-7` (the hop half)**: the K/V cache of a session belongs to the
/// *conversation*, not to the work the orchestrator minted for one turn. With `request_id` as the
/// key, a hop stores the state under an id that no later turn will ever name again.
#[tokio::test]
async fn a_hop_stores_its_session_under_the_continuity_key_of_the_payload() {
    let (engine, _applied) = engine_with_spy_backend().await;
    let keypair = tessella_network::generate_keypair();

    let prefill = payload_with_ids(&keypair, "work-1", Some("conv-7"), 0, 3, 7001);
    engine
        .forward_hidden(&prefill.to_bytes().expect("payload bytes"))
        .await
        .expect("the first step of a conversation is a normal hop");

    assert!(
        engine.sessions.contains_key(&hop_slot("conv-7")),
        "the session of a hop lives under the client's continuity id, and under the range that \
         served the step (`TASK-96`/S102): the K/V of one node's second chunk is not the K/V of its \
         first"
    );
    assert!(
        !engine.sessions.contains_key(&hop_slot("work-1")),
        "the work id is the id of the *economy*, not the key of a K/V cache"
    );
}

/// The same defect, from the side the client sees: turn 2 arrives with a **new** work id (the
/// orchestrator mints one per request, `TASK-07`) and the same conversation. The prefix *did* pass
/// through this node, so refusing it would be a false positive of the guard — and serving it from a
/// fresh session would be the silent corruption of `F-1`.
#[tokio::test]
async fn a_second_turn_with_a_new_work_id_resumes_the_hop_session() {
    let (engine, applied) = engine_with_spy_backend().await;
    let keypair = tessella_network::generate_keypair();

    let prefill = payload_with_ids(&keypair, "work-1", Some("conv-7"), 0, 3, 7002);
    engine
        .forward_hidden(&prefill.to_bytes().expect("payload bytes"))
        .await
        .expect("turn 1");

    let next_turn = payload_with_ids(&keypair, "work-2", Some("conv-7"), 3, 1, 7003);
    engine
        .forward_hidden(&next_turn.to_bytes().expect("payload bytes"))
        .await
        .expect("turn 2 has to be served: the prefix of this conversation passed through here");
    assert_eq!(
        applied.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "both turns reached the backend"
    );
}
/// **The defect of `F-7` (the chain half)**: the `session_id` is part of the signed preimage, so a
/// hop that rewrites it makes the *next* hop reject a perfectly good payload. With three nodes the
/// client of a conversation read `invalid signature` and a 30 s pipeline timeout (measured with the
/// harness in `S76`). What has to reach the next hop is the id the entry node sealed.
#[tokio::test]
async fn the_continuity_key_survives_the_hop_and_its_signature_still_verifies() {
    // A **middle** shard: `payload_with_ids` brings a payload whose last processed layer is 7, so the
    // step it asks for is layer 8 — the first of this node's range (`TASK-96`/S102: the range is what
    // the hop names, so a node that does not cover the step refuses it instead of running its own
    // layers again).
    let (engine, _applied, _began) = engine_with_spy_backend_on((8, 16)).await;
    let keypair = engine
        .keypair
        .clone()
        .expect("the fixture engine carries a keypair");
    let router: Arc<dyn tessella_core::traits::PipelineRouter> = Arc::new(StubRouter::echoing());
    engine.set_router(router).await;

    let prefill = payload_with_ids(&keypair, "work-1", Some("conv-7"), 0, 3, 7004);
    let out_bytes = engine
        .forward_hidden(&prefill.to_bytes().expect("payload bytes"))
        .await
        .expect("the middle shard of a conversation serves its step");
    let out = HiddenStatePayload::from_bytes(&out_bytes).expect("a real payload comes back");

    assert_eq!(
        out.session_id.as_deref(),
        Some("conv-7"),
        "the id the entry node sealed has to survive every hop (it is part of the preimage)"
    );
    tessella_network::verify_signed_request(
        &out,
        &tessella_protocol::signing::ReplayCache::new(),
        tessella_core::time::unix_now_secs(),
        "the next hop",
    )
    .expect("the payload of a hop still verifies at the next hop");
}

/// **The defect of `F-7` (the entry-node half)**: the session is **stored** under one id and
/// **looked up** under another, so the KV of the previous turn is unreachable by construction: the
/// entry node opens a new backend session every turn and the client pays the whole prefill again.
///
/// `TASK-78`/S85 (`D-36`) keeps the invariant —store and look up with the same value— and moves
/// **which** value it is about: the key is now the id this node issues, so the test pins the three
/// ids against each other. The client's hint is not a key (it is what two clients can share), the
/// work id is not a key (one conversation is many turns), and the id the answer publishes is the one
/// the next turn resumes with.
#[tokio::test]
async fn the_entry_node_saves_its_session_under_the_id_it_will_look_it_up_with() {
    let (engine, _applied, began) = engine_with_spy_backend_on((0, 8)).await;
    let router: Arc<dyn tessella_core::traits::PipelineRouter> = Arc::new(StubRouter::echoing());

    // A session **with** the orchestrator's work ticket: the work id of the economy (`work-9`) and
    // the id of the conversation (`conv-7` as the client's hint) are two different things, which is
    // the point.
    let mut work = ticket();
    work.request_id = "work-9".to_string();

    let (session, request_id) = engine
        .clone()
        .begin_session_with_router(
            router.clone(),
            Some("conv-7".to_string()),
            None,
            None,
            Some(work),
        )
        .await
        .expect("the entry node opens a session");
    assert_eq!(request_id, "work-9");
    // The key is the id **this node issued** (`TASK-78`/S85): it is the one the answer publishes and
    // the one a client has to send back, so it cannot be either of the two ids the caller offered.
    let issued = session.session_id().to_string();
    assert_ne!(
        issued, "conv-7",
        "the client's id is a hint, not the key of the K/V"
    );
    assert_ne!(
        issued, "work-9",
        "nor the work id of the turn: one conversation is many works"
    );
    assert!(
        !engine.sessions.contains_key("conv-7"),
        "before the drop there is nothing cached"
    );

    // No `forward` call is needed: the key is chosen when the session is **opened**, which is
    // exactly where the bug lived (the lookup used the client id and the `Drop` used the work id).
    // The forward path of an open session is covered by the two tests below.
    drop(session);

    assert!(
        engine.sessions.contains_key(&issued),
        "the K/V of the turn has to be stored under the id the next turn will name"
    );
    assert!(
        !engine.sessions.contains_key("work-9"),
        "storing it under the work id is the bug: no later turn ever names it again"
    );

    // …and the next turn of the conversation reuses it instead of opening a new backend session.
    let (second, _rid) = engine
        .clone()
        .begin_session_with_router(router, Some(issued.clone()), None, None, Some(ticket()))
        .await
        .expect("the entry node reopens the conversation");
    assert_eq!(
        began.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "one backend session for two turns: the KV was resumed, not rebuilt"
    );
    assert_eq!(
        second.session_id(),
        issued,
        "and the id does not change when the conversation is resumed: that is why the client can \
             keep it"
    );
}

/// The decision of `S76` (**D-30** = «declare the limit»): when the pipeline refuses a step because
/// a hop lost the prefix, the entry node **drops its own session** of that conversation. Without it
/// the client's retry would resume the same doomed offset and fail again; with it the retry
/// prefills the whole conversation and the mesh answers. The failure stays one line of `warn!` per
/// lost session: one `warn!` line per lost session, which is exactly the requirement.
#[tokio::test]
async fn a_session_whose_pipeline_lost_the_prefix_is_dropped_at_once() {
    let (engine, _applied, began) = engine_with_spy_backend_on((0, 8)).await;
    let router: Arc<dyn tessella_core::traits::PipelineRouter> = Arc::new(StubRouter::refusing());

    let (mut session, _rid) = engine
        .clone()
        .begin_session_with_router(router, Some("conv-7".to_string()), None, None, None)
        .await
        .expect("the entry node opens a session");
    // The id the answer would have published (`TASK-78`/S85): what has to disappear is the id a
    // client holds, not the hint it sent —the hint is a key of nothing.
    let issued = session.session_id().to_string();
    // `ForwardOutput` has no `Debug`: the match is the assertion, not a formatting convenience.
    let error = match session.forward(vec![1, 2, 3]).await {
        Ok(_) => panic!("a hop without the prefix cannot serve the step"),
        Err(error) => error,
    };
    assert!(tessella_core::error::is_missing_prefill(&error), "{error}");
    drop(session);

    assert!(
        !engine.sessions.contains_key(&issued),
        "the entry node cannot keep a K/V the mesh no longer has: the retry has to prefill"
    );
    assert!(
        engine.session_timestamps.is_empty(),
        "the session is gone from the map the TTL sweeper walks"
    );

    // The retry of that conversation starts from scratch — which is the promise of D-30. It starts
    // from scratch **even if the client names the id it was given**, which is exactly what a client
    // that comes back after a lost session does (`TASK-78`/S85): the node no longer honours that
    // hint, so it answers with a new conversation instead of an offset nobody holds.
    let (retry, _rid) = engine
        .clone()
        .begin_session_with_router(
            Arc::new(StubRouter::echoing()),
            Some(issued.clone()),
            None,
            None,
            None,
        )
        .await
        .expect("the retry opens a fresh session");
    assert_eq!(
        began.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "the retry prefills: a dropped session is a new backend session"
    );
    assert_ne!(
        retry.session_id(),
        issued,
        "and it names a new session: the id of the lost one is not resumed by naming it"
    );
}

// ─── `TASK-78`/`S85` · quién elige la clave de la sesión (`D-36`) ───────────────────────────

/// **El defecto de `TASK-78`/`S85`**: el `session_id` del cliente **era** la clave de la K/V, así
/// que dos peticionarios que eligieran el mismo valor —«prueba», la fecha de hoy, el id que otro
/// dejó en un log— compartían una conversación: el segundo turno se servía sobre el contexto del
/// primero y el modelo contestaba con la vida de nadie. El id pasa a ser una **pista**: lo emite el
/// nodo, sólo él reanuda, y dos peticionarios que eligen el mismo valor reciben dos ids distintos
/// (opción (B) de `D-36`).
#[tokio::test]
async fn two_requesters_that_choose_the_same_session_id_do_not_share_the_kv() {
    let (engine, _applied, began) = engine_with_spy_backend_on((0, 8)).await;
    let router = || -> Arc<dyn PipelineRouter> { Arc::new(StubRouter::echoing()) };
    const SHARED: &str = "la-sesion-de-todos";

    let (first, _work) = engine
        .clone()
        .begin_session_with_router(router(), Some(SHARED.to_string()), None, None, None)
        .await
        .expect("el primer peticionario abre su sesión");
    let (second, _work) = engine
        .clone()
        .begin_session_with_router(router(), Some(SHARED.to_string()), None, None, None)
        .await
        .expect("el segundo peticionario abre su sesión");

    let issued = first.session_id().to_string();
    let other = second.session_id().to_string();
    assert_ne!(
        issued, other,
        "dos peticionarios con el mismo id de cliente no pueden compartir la clave de la K/V"
    );
    assert_ne!(
        issued, SHARED,
        "y el id que reanuda lo emite el nodo: el del cliente es sólo una pista"
    );
    assert_eq!(
        began.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "cada peticionario prefilla su propia conversación (antes, el segundo reanudaba la del \
             primero, porque la pista era la clave)"
    );

    drop(first);
    drop(second);
    assert_eq!(
        engine.sessions.len(),
        2,
        "el nodo guarda las dos sesiones, no una compartida"
    );
    assert!(
        engine.sessions.contains_key(&issued) && engine.sessions.contains_key(&other),
        "cada id emitido es la clave de su propia K/V"
    );

    // El id **emitido** es el que reanuda: quien lo devuelve no vuelve a pagar el prefill.
    let (resumed, _work) = engine
        .clone()
        .begin_session_with_router(router(), Some(issued.clone()), None, None, None)
        .await
        .expect("el id emitido abre la sesión de su conversación");
    assert_eq!(
        resumed.session_id(),
        issued,
        "el id emitido es estable: es el que el cliente tiene que devolver"
    );
    assert_eq!(
        began.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "reanudar no es prefilar: la pista conocida es la única que se honra"
    );
}

/// La otra mitad de `D-36`: una pista que este nodo **no** ha emitido no se honra aunque venga con
/// pinta de id propio del cliente. Sin esto, dos clientes que inventen el mismo id volverían a
/// compartir la K/V (el defecto de arriba) y un id adivinable sería la única credencial de la
/// conversación.
#[tokio::test]
async fn a_hint_this_node_never_issued_is_replaced_by_one_it_issues() {
    let (engine, _applied, began) = engine_with_spy_backend_on((0, 8)).await;
    let router = || -> Arc<dyn PipelineRouter> { Arc::new(StubRouter::echoing()) };

    let (first, _work) = engine
        .clone()
        .begin_session_with_router(
            router(),
            Some("conv-inventado".to_string()),
            None,
            None,
            None,
        )
        .await
        .expect("la primera petición abre una sesión");
    let issued = first.session_id().to_string();
    assert_ne!(issued, "conv-inventado");
    drop(first);

    // El cliente vuelve con **su** id de siempre: la sesión de arriba no se reanuda (no era una
    // pista emitida por el nodo) y la K/V se prefilla otra vez.
    let (second, _work) = engine
        .clone()
        .begin_session_with_router(
            router(),
            Some("conv-inventado".to_string()),
            None,
            None,
            None,
        )
        .await
        .expect("la segunda petición abre su propia sesión");
    assert_ne!(
        second.session_id(),
        issued,
        "una pista no emitida por el nodo no puede apuntar a una sesión que sí lo fue"
    );
    assert_eq!(
        began.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "y por eso la segunda pista se prellena desde cero"
    );
    assert!(
        !engine.sessions.contains_key("conv-inventado"),
        "el valor que el cliente eligió no es nunca una clave de K/V"
    );

    // …y el id emitido **sí** reanuda, que es la mitad que el cliente necesita para el multi-turno.
    drop(second);
    let (_resumed, _work) = engine
        .clone()
        .begin_session_with_router(router(), Some(issued.clone()), None, None, None)
        .await
        .expect("el id emitido se reanuda");
    assert_eq!(
        began.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "el id emitido por el nodo es lo que hace reanudable una conversación"
    );
}

// ─── `TASK-76`/`S83` · la declaración de VRAM, por el camino de entrada ─────────────────────

/// The `S76` case, end to end and on the **entry path**: a node that declares 3 tokens for its
/// VRAM refuses a 4-token prompt **before** creating the backend session. The measurement that
/// motivated `S83` was the opposite — the node published the wire's number (which for this fixture
/// is millions of tokens), prefilled 2 048 and raised `CUDA_ERROR_OUT_OF_MEMORY` at 4 096.
///
/// The other half of the guard — that exactly the ceiling still travels, because the comparison is
/// `>` and not `>=` — is pinned by `the_transport_verdict_only_rejects_what_does_not_fit`, which is
/// where the boundary lives; here what matters is the **order**: the refusal comes before the
/// prefill, so a node never pays for a prompt it cannot carry.
#[tokio::test]
async fn the_entry_path_refuses_a_prompt_over_the_declared_ceiling_before_prefilling() {
    let (engine, applied, _began) = engine_with_declared_ceiling((0, 8), 3).await;
    let router: Arc<dyn PipelineRouter> = Arc::new(StubRouter::echoing());
    let (mut session, _rid) = engine
        .clone()
        .begin_session_with_router(router, Some("conv-83".to_string()), None, None, None)
        .await
        .expect("the entry node opens a session");

    // 4 tokens against a declaration of 3: refused, named, and without paying for the prefill.
    let error = match session.forward(vec![1, 2, 3, 4]).await {
        Ok(_) => panic!("a prompt over the declared ceiling must not be served"),
        Err(error) => error,
    };
    assert!(tessella_core::error::is_prompt_too_long(&error), "{error}");
    let message = error.to_string();
    for expected in ["prompt_too_long", "source: vram", "max_input_tokens"] {
        assert!(
            message.contains(expected),
            "{expected} missing from: {message}"
        );
    }
    assert_eq!(
        applied.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the refusal happens before the backend session: no prefill is paid for a prompt that \
             does not fit, which is the whole difference with the S76 measurement"
    );
}

// ─── `TASK-77`/`S84` · un fallo transitorio de GPU no puede sacar al nodo de la malla ───────────

/// El defecto que midió `S76`: un `prefill` que no cabe en la VRAM **no** deja al nodo fuera de la
/// malla para siempre.
///
/// El `OOM` es un juicio sobre la **petición** —el nodo sigue teniendo razón—, así que la prueba es
/// la vuelta: la petición siguiente, que sí cabe, la contesta el **mismo** engine, sin reiniciar nada
/// y sin que nadie intervenga. Y de paso: la sonda **no se paga**. Se cuenta cada consulta (`probe_calls`)
/// y aquí tiene que haber cero, porque en esta fixture el device está *muerto*: consultarlo sería
/// correr el riesgo de que un `OOM` declare caído a un nodo vivo —el que midió `S84`—. El veredicto
/// puro (`failure_verdict`) ya lo impide; el contador fija que además nadie malgasta la reserva.
#[tokio::test]
async fn an_out_of_memory_of_one_request_leaves_the_node_serving_the_next_one() {
    let fault_active = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let device_alive = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (probe, probe_calls) = probe_that_answers(device_alive);
    let (engine, _applied, _began) = engine_with_gpu_fault(
        SPY_LAYERS,
        crate::config::CanaryMode::Off,
        0,
        Some(SpyFault {
            active: fault_active.clone(),
            message: "CUDA_ERROR_OUT_OF_MEMORY: out of memory".to_string(),
        }),
        Some(probe),
    )
    .await;
    let keypair = tessella_network::generate_keypair();

    let too_big = hop_payload(&keypair, "req-oom", 0, 8_192, 21);
    let refused = engine
        .forward_hidden(&too_big.to_bytes().expect("payload bytes"))
        .await
        .expect_err("the step asked for more VRAM than the node has: it has to fail");
    assert!(
        tessella_core::telemetry::gpu_health::classify_gpu_failure(&refused.to_string()).is_some(),
        "the backend error reaches the caller as it was: {refused}"
    );

    // El nodo sigue sirviendo, y lo dice con nombres (es lo que `GET /status` publica).
    let report = engine.gpu_report();
    assert!(report.healthy, "un OOM no es un juicio sobre el nodo");
    assert_eq!(report.verdict, "serving");
    assert_eq!(report.kind, None);
    assert_eq!(report.out_of_memory, 1, "se cuenta lo que pasó");
    assert_eq!(report.device_failures, 0);
    assert_eq!(report.recoveries, 0, "no hay vuelta que contar");
    assert_eq!(
        probe_calls.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "un OOM no le pregunta al device: su veredicto no depende de la respuesta"
    );

    // La petición siguiente —la que sí cabe— la contesta este mismo engine.
    fault_active.store(false, std::sync::atomic::Ordering::Relaxed);
    let fits = hop_payload(&keypair, "req-oom-next", 0, 2, 22);
    engine
        .forward_hidden(&fits.to_bytes().expect("payload bytes"))
        .await
        .expect("the node is serving: the next request has to be served");
    assert_eq!(
        engine.gpu_report().out_of_memory,
        1,
        "y sin inventarse una vuelta que no ocurrió"
    );
}

/// Y el caso que separa la sonda del mensaje: un error que **habla** del device —el patrón que antes
/// bastaba para sacar al nodo de la malla— mientras el device contesta que sigue vivo. El nodo no se
/// cae y lo apunta como `device_failures`, que es la cuenta de los fallos que la sonda no confirmó.
///
/// Es el `OOM` de un allocator fragmentado y el pánico de un kernel ajeno: la petición murió, el
/// device no. Declarar caído al nodo por esto es lo que dejaba a un nodo sano fuera de la malla.
#[tokio::test]
async fn a_device_error_the_probe_does_not_confirm_keeps_the_node_in_the_mesh() {
    let fault_active = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    // El device contesta: la caída que el mensaje insinúa no está.
    let (probe, probe_calls) = probe_that_answers(std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(true),
    ));
    let (engine, _applied, _began) = engine_with_gpu_fault(
        SPY_LAYERS,
        crate::config::CanaryMode::Off,
        0,
        Some(SpyFault {
            active: fault_active.clone(),
            message: "CUDA error: an illegal memory access was encountered".to_string(),
        }),
        Some(probe),
    )
    .await;
    let keypair = tessella_network::generate_keypair();

    engine
        .forward_hidden(
            &hop_payload(&keypair, "req-not-lost", 0, 2, 41)
                .to_bytes()
                .expect("payload bytes"),
        )
        .await
        .expect_err("el paso no se pudo aplicar");

    let report = engine.gpu_report();
    assert!(
        report.healthy,
        "el device contestó: el nodo sigue en la malla"
    );
    assert_eq!(report.verdict, "serving");
    assert_eq!(report.kind, None);
    assert_eq!(report.device_failures, 1);
    assert_eq!(report.out_of_memory, 0);
    assert_eq!(
        probe_calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "aquí el veredicto **sí** depende de la respuesta: el mensaje habló del device, así que se le \
             pregunta una vez"
    );

    // Y una petición que llegue después no se encuentra un nodo cerrado.
    engine
        .refuse_if_gpu_down()
        .expect("un nodo sano no se niega a servir");
}

/// El otro juicio, entero: el device **deja** de contestar. Eso sí es un juicio sobre el nodo, así que
/// se declara caído —con su motivo, el que recibe el cliente que llega— y se anuncia `Degraded`.
///
/// Pero la vuelta no necesita a nadie: cada revalidación (`heartbeat`, y antes de negar una petición)
/// le vuelve a preguntar al device, y en cuanto contesta el nodo está sirviendo otra vez. Es la mitad
/// que faltaba en `S76`: allí no había **ninguna** línea que volviera a poner el flag en `true`.
#[tokio::test]
async fn a_node_that_lost_its_device_declares_itself_down_and_comes_back_on_its_own() {
    let fault_active = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let device_alive = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (probe, probe_calls) = probe_that_answers(device_alive.clone());
    let (engine, _applied, _began) = engine_with_gpu_fault(
        SPY_LAYERS,
        crate::config::CanaryMode::Off,
        0,
        Some(SpyFault {
            active: fault_active.clone(),
            message: "CUDA_ERROR_ILLEGAL_ADDRESS: illegal memory access on device".to_string(),
        }),
        Some(probe),
    )
    .await;
    let keypair = tessella_network::generate_keypair();

    engine
        .forward_hidden(
            &hop_payload(&keypair, "req-lost", 0, 2, 31)
                .to_bytes()
                .expect("payload bytes"),
        )
        .await
        .expect_err("el device se fue: el paso no se puede servir");

    let down = engine.gpu_report();
    assert!(!down.healthy);
    assert_eq!(down.verdict, "down");
    assert_eq!(down.kind, Some("device_lost"));
    assert_eq!(down.context.as_deref(), Some("forward_hidden"));
    assert!(
        down.reason
            .as_deref()
            .unwrap_or_default()
            .contains("illegal memory"),
        "{down:?}"
    );
    assert!(
        down.probe_error
            .as_deref()
            .unwrap_or_default()
            .contains("ILLEGAL_ADDRESS"),
        "se publica qué contestó la **última** sonda: {down:?}"
    );
    assert_eq!(
        down.device_failures, 0,
        "este fallo **sí** lo confirmó la sonda: lo publica `kind`/`reason`, no el contador de los \
             fallos de device que el device desmintió"
    );
    assert_eq!(down.recoveries, 0);
    assert_eq!(
        probe_calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "la caída se decidió con una sola pregunta al device: la que la confirmó"
    );
    assert_eq!(
        down.revalidate_secs, HEARTBEAT_SECS,
        "la cota de la vuelta es el compás del latido, y se publica"
    );
    assert!(down.down_secs.is_some(), "y desde cuándo está caído");
    assert!(down.since_unix_secs.is_some());

    // Mientras el device no contesta, la revalidación del latido falla y una petición se niega
    // **con el motivo** —negarla en silencio es lo que convierte una caída en un misterio—.
    assert!(
        engine.revalidate_gpu().is_err(),
        "el latido vuelve a preguntar en cada tick y aquí el device sigue muerto"
    );
    let refusal = engine
        .refuse_if_gpu_down()
        .expect_err("un nodo con el device muerto no puede servir");
    assert!(refusal.to_string().contains("illegal memory"), "{refusal}");
    assert!(
        !engine.gpu_report().healthy,
        "preguntar no resucita un device: sólo lo detecta en cuanto contesta"
    );
    assert_eq!(
        probe_calls.load(std::sync::atomic::Ordering::Relaxed),
        3,
        "las tres preguntas del camino cuesta abajo: la del fallo, la del latido y la de negar la \
             petición —negar sin preguntar dejaría al nodo fuera de la malla aunque el device estuviera \
             de vuelta—"
    );

    // El device vuelve. Nadie reinicia nada: el nodo se revalida y sirve.
    device_alive.store(true, std::sync::atomic::Ordering::Relaxed);
    fault_active.store(false, std::sync::atomic::Ordering::Relaxed);
    engine.revalidate_gpu().expect("the device answers again");

    let back = engine.gpu_report();
    assert!(back.healthy);
    assert_eq!(back.verdict, "serving");
    assert_eq!(back.recoveries, 1);
    assert_eq!(back.last_recovery, Some("device_probe"));
    assert_eq!(back.kind, None, "ya no hay motivo que publicar");
    assert!(back.down_secs.is_none());
    assert_eq!(
        probe_calls.load(std::sync::atomic::Ordering::Relaxed),
        4,
        "y la cuarta —la del latido que encontró al device de vuelta— es la que devolvió el nodo a la \
             malla: la vuelta es una pregunta más, no un reinicio"
    );

    engine
        .forward_hidden(
            &hop_payload(&keypair, "req-back", 0, 2, 32)
                .to_bytes()
                .expect("payload bytes"),
        )
        .await
        .expect("the node is back in the mesh and serves the next hop");
}
