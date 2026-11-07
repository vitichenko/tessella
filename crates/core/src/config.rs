use serde::{Deserialize, Serialize};

use crate::error::DomainError;

pub use tessella_protocol::QuantizationConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::module_name_repetitions)]
pub struct NetworkConfig {
    pub listen_port: u16,             // default: 8000
    pub p2p_port: u16,                // default: 9000
    pub heartbeat_secs: u64,          // default: 180
    pub musivum_addr: Option<String>, // URL de musivum
    pub musivum_p2p: Option<String>,  // multiaddr de musivum P2P
    pub relay_addr: Option<String>,   // relay externo si lo hay
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_port: 8000,
            p2p_port: 9000,
            heartbeat_secs: 180,
            musivum_addr: None,
            musivum_p2p: None,
            relay_addr: None,
        }
    }
}

/// Tunables of the reputation / liveness subsystem of the control plane.
///
/// **Single source of truth of `[reputation]` (H-20)**: the swarm liveness loop used to
/// hardcode `interval(15s)` + `> 45s` while these TOML keys were dropped in silence
/// (`heartbeat_secs` did not exist in this struct, and unknown keys were accepted).
///
/// `default` makes every key optional (like `MusivumConfig` does at its own level), while
/// `deny_unknown_fields` keeps typos and removed keys a startup error instead of a silent
/// no-op.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::module_name_repetitions)]
#[serde(default, deny_unknown_fields)]
pub struct ReputationConfig {
    /// A node with `failed_requests >= 10` and a reputation score below this is marked
    /// `Offline` by the periodic health check. Default: 0.3
    ///
    /// `0.0` is a documented escape: it disables this eviction path entirely (nothing is ever
    /// `< 0.0`).
    pub slash_threshold: f64,
    /// Multiplier applied to the **persisted** reputation score (`reputation_scores.score`) every
    /// time a slashable incident is recorded against a peer (`ScoreEvent::IncidentRecorded`,
    /// `TASK-17`). `0.0` = any incident resets the score, `1.0` = no penalty. Default: 0.5
    ///
    /// It used to be *"RESERVED: read from `[reputation]` but not consumed yet"* while the real
    /// factors were the literals `SCORE_PENALTY_INVALID_SIGNATURE = 0.5` and
    /// `SCORE_PENALTY_EXCESSIVE_FAILURES = 0.6` of `tessella-storage` — the same defect H-20 fixed
    /// for the liveness keys. Proven fraud (`ConflictingProof`, `WeightPoisoning`) is a full reset
    /// and ignores this knob on purpose.
    pub slash_penalty: f64,
    /// A node whose success ratio is **below** this is marked `Degraded` by the periodic health
    /// check (it is the step before eviction). Default: 0.7
    ///
    /// It used to be the literal `0.7` inside `NodeInfo::is_degraded()`, so it was invisible to the
    /// operator (`TASK-17`). `validate()` keeps `slash_threshold <= degraded_threshold`, so a node
    /// is always flagged `Degraded` before it is eligible for eviction by score.
    pub degraded_threshold: f64,
    /// How often the control plane sweeps the swarm for peers that are no longer
    /// connected. Must be `> 0` and `<= evict_after_secs`. Default: 15
    pub heartbeat_secs: u64,
    /// TTL of a peer that is **absent from `swarm.connected_peers()`**; a peer that is
    /// still connected is never evicted by time. Default: 600
    pub evict_after_secs: u64,
}

impl ReputationConfig {
    /// Rejects combinations that would misbehave at runtime instead of at startup:
    ///
    /// - A zero interval: `tokio::time::interval(Duration::ZERO)` panics, so
    ///   `heartbeat_secs = 0` must fail here, not inside the event loop.
    /// - `heartbeat_secs > evict_after_secs`: the sweep would run less often than the TTL
    ///   it is supposed to enforce, so the TTL would stop being an upper bound.
    /// - Reputation thresholds outside `[0, 1]` (the score is a success ratio) or inverted
    ///   (`slash_threshold > degraded_threshold`), which would make `Degraded` unreachable
    ///   (`TASK-17`).
    /// - A `slash_penalty` outside `[0, 1]`, which would *raise* the score of a peer that
    ///   misbehaved, or poison the column with a `NaN` (`TASK-17`).
    pub fn validate(&self) -> Result<(), DomainError> {
        if self.heartbeat_secs == 0 || self.evict_after_secs == 0 {
            return Err(DomainError::Validation(format!(
                "reputation.heartbeat_secs ({}) and reputation.evict_after_secs ({}) must both be > 0",
                self.heartbeat_secs, self.evict_after_secs
            )));
        }
        if self.heartbeat_secs > self.evict_after_secs {
            return Err(DomainError::Validation(format!(
                "reputation.heartbeat_secs ({}) must be <= reputation.evict_after_secs ({}), \
                 otherwise the liveness sweep is slower than the TTL it enforces",
                self.heartbeat_secs, self.evict_after_secs
            )));
        }
        if !self.slash_threshold.is_finite()
            || !self.degraded_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.slash_threshold)
            || !(0.0..=1.0).contains(&self.degraded_threshold)
        {
            return Err(DomainError::Validation(format!(
                "reputation.slash_threshold ({}) and reputation.degraded_threshold ({}) must both be \
                 finite values in [0, 1] (they are compared against a success ratio)",
                self.slash_threshold, self.degraded_threshold
            )));
        }
        if self.slash_threshold > self.degraded_threshold {
            return Err(DomainError::Validation(format!(
                "reputation.slash_threshold ({}) must be <= reputation.degraded_threshold ({}), \
                 otherwise a node is evicted before it is ever marked Degraded",
                self.slash_threshold, self.degraded_threshold
            )));
        }
        if !self.slash_penalty.is_finite() || !(0.0..=1.0).contains(&self.slash_penalty) {
            return Err(DomainError::Validation(format!(
                "reputation.slash_penalty ({}) must be a finite value in [0, 1]: it multiplies the \
                 persisted reputation score of a peer that misbehaved (0.0 = reset on any incident, \
                 1.0 = no penalty)",
                self.slash_penalty
            )));
        }
        Ok(())
    }
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            slash_threshold: 0.3,
            slash_penalty: 0.5,
            degraded_threshold: 0.7,
            heartbeat_secs: 15,
            evict_after_secs: 600,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H-20: `heartbeat_secs` vivía en el TOML del operador pero **no** en el struct, así
    /// que `#[serde(default)]` la descartaba en silencio y mandaban los literales 15 s /
    /// 45 s del event loop. Ahora forma parte del contrato de `[reputation]`.
    #[test]
    fn test_heartbeat_secs_is_part_of_the_contract() {
        let json = r#"{"slash_threshold":0.3,"slash_penalty":0.5,"heartbeat_secs":180,"evict_after_secs":600}"#;
        let cfg: ReputationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.heartbeat_secs, 180);
        assert_eq!(cfg.evict_after_secs, 600);
    }

    /// Un typo dentro de `[reputation]` ya no pasa desapercibido.
    #[test]
    fn test_unknown_reputation_key_is_rejected() {
        let json = r#"{"heartbeat_sec":180}"#;
        let err = serde_json::from_str::<ReputationConfig>(json)
            .unwrap_err()
            .to_string();
        assert!(err.contains("heartbeat_sec"), "unexpected error: {err}");
    }

    /// Los defaults reproducen el barrido que el event loop tenía hardcodeado (15 s) y el
    /// TTL que ya documentaba el TOML del operador (600 s).
    #[test]
    fn test_defaults() {
        let cfg = ReputationConfig::default();
        assert_eq!(cfg.heartbeat_secs, 15);
        assert_eq!(cfg.evict_after_secs, 600);
        assert!(cfg.validate().is_ok());
    }

    /// `tokio::time::interval(Duration::ZERO)` panica, así que un 0 en el TOML debe
    /// rechazarse en el arranque y no dentro del event loop.
    #[test]
    fn test_validate_rejects_zero_intervals() {
        let zero_sweep = ReputationConfig {
            heartbeat_secs: 0,
            ..Default::default()
        };
        assert!(zero_sweep.validate().is_err());

        let zero_ttl = ReputationConfig {
            evict_after_secs: 0,
            ..Default::default()
        };
        assert!(zero_ttl.validate().is_err());
    }

    /// El TTL solo significa algo si el barrido es al menos tan frecuente como el TTL.
    #[test]
    fn test_validate_rejects_sweep_slower_than_ttl() {
        let cfg = ReputationConfig {
            heartbeat_secs: 600,
            evict_after_secs: 45,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("heartbeat_secs"), "unexpected error: {err}");
    }

    /// TASK-17: el umbral de degradación era el literal `0.7` de `NodeInfo::is_degraded()`, invisible
    /// para el operador. Ahora es una clave de `[reputation]` con el mismo valor por defecto, así que
    /// quien no la escriba no cambia de comportamiento.
    #[test]
    fn test_degraded_threshold_is_part_of_the_contract() {
        assert_eq!(ReputationConfig::default().degraded_threshold, 0.7);

        // `#[serde(default)]`: la clave es opcional y una configuración parcial sigue valiendo.
        let json = r#"{"slash_threshold":0.3,"degraded_threshold":0.9}"#;
        let cfg: ReputationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.degraded_threshold, 0.9);
        assert_eq!(cfg.slash_penalty, 0.5, "el resto conserva sus defaults");
        cfg.validate().expect("0.3 <= 0.9 es coherente");

        // …y un typo sigue siendo un error de arranque.
        let err = serde_json::from_str::<ReputationConfig>(r#"{"degraded_thresold":0.9}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("degraded_thresold"), "unexpected error: {err}");
    }

    /// Un umbral de degradación **por debajo** del de eviction marcaría `Degraded` a un nodo que el
    /// barrido acaba de evacuar (o al que le queda menos margen que al evictado): incoherente, y
    /// ahora se rechaza en el arranque en vez de producir un estado imposible.
    #[test]
    fn test_validate_rejects_degraded_threshold_below_slash_threshold() {
        let inverted = ReputationConfig {
            slash_threshold: 0.8,
            degraded_threshold: 0.2,
            ..Default::default()
        };
        let err = inverted.validate().unwrap_err().to_string();
        assert!(
            err.contains("degraded_threshold"),
            "unexpected error: {err}"
        );

        // Desviación deliberada y documentada: `slash_threshold = 0.0` sigue siendo
        // el escape soportado para **desactivar** la eviction por score (`should_evict` nunca dispara
        // con 0.0, fijado por el test de `musivum::health`), así que el rango del umbral es [0, 1].
        let escape = ReputationConfig {
            slash_threshold: 0.0,
            degraded_threshold: 0.0,
            ..Default::default()
        };
        escape
            .validate()
            .expect("0.0/0.0 desactiva las dos ramas por score");

        let out_of_range = ReputationConfig {
            degraded_threshold: 1.5,
            ..Default::default()
        };
        assert!(
            out_of_range.validate().is_err(),
            "un umbral de degradación > 1.0 nunca se cumpliría"
        );
    }

    /// `slash_penalty` multiplica la reputación **persistida** en cada incidente registrado
    /// (TASK-17): fuera de `[0, 1]` subiría la nota de quien falla, y un `NaN`/`inf` dejaría el
    /// ledger de reputación sin sentido. `0.0` = reset por cualquier incidente, `1.0` = sin castigo.
    #[test]
    fn test_validate_rejects_penalties_out_of_range() {
        for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
            let cfg = ReputationConfig {
                slash_penalty: bad,
                ..Default::default()
            };
            assert!(
                cfg.validate().is_err(),
                "slash_penalty = {bad} must not boot"
            );
        }
        for ok in [0.0, 0.5, 1.0] {
            let cfg = ReputationConfig {
                slash_penalty: ok,
                ..Default::default()
            };
            cfg.validate()
                .unwrap_or_else(|e| panic!("slash_penalty = {ok} es válido: {e}"));
        }
    }
}
