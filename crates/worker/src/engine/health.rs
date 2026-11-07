use candle_core::Device;
use std::sync::Arc;
use tessella_core::error::DomainError;
use tessella_core::telemetry::gpu_health::{
    self, GpuFailureKind, GpuHealth, GpuHealthReport, GpuRecovery, GpuVerdict,
};
use tracing::{error, info, warn};

/// La sonda del device, tal como se inyecta (`TASK-77`/S84). El alias existe porque el tipo se nombra
/// cuatro veces —el campo, el setter, la fixture de test y el parámetro de la fixture— y clippy lee esa
/// repetición como complejidad escondida.
pub type DeviceProbe = Box<dyn Fn(&Device) -> Result<(), String> + Send + Sync>;

/// La sonda acotada al device (`TASK-77`/S84): 4 bytes que se reservan, se escriben y se leen de
/// vuelta.
///
/// It is the `device_answers` argument of `gpu_health::failure_verdict`, and it is deliberately the
/// smallest thing that can fail: the allocation asks the device (not the host) for memory and the copy
/// back needs the driver and the context alive, so a destroyed context, a driver that shut down or an
/// unusable device all make it fail — which is what a **lost** device is, and none of what an
/// over-sized request is.
///
/// It replaced `is_cuda_gpu_error`, the `bool` that answered «is this string about the GPU?» and could
/// not tell the two apart: with it, every `OOM` of every request took the node out of the mesh for
/// good (`TASK-77`). The classification now lives in `tessella_core::telemetry::gpu_health` and the
/// question «is this node alive?» is answered by asking the device, not by reading the message.
pub fn device_probe(device: &Device) -> Result<(), String> {
    candle_core::Tensor::zeros(&[1], candle_core::DType::F32, device)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map(|_| ())
        .map_err(|err| err.to_string())
}

/// Encapsulates GPU device health monitoring, failure classification, device probing,
/// and recovery re-validation (SOLID-08).
pub struct GpuHealthMonitor {
    pub gpu_health: Arc<GpuHealth>,
    pub device: Device,
    probe: DeviceProbe,
}

impl GpuHealthMonitor {
    pub fn new(
        device: Device,
        heartbeat_interval_secs: u64,
        oom_streak: u32,
        streak_window_secs: u64,
    ) -> Self {
        Self {
            gpu_health: Arc::new(GpuHealth::new(
                heartbeat_interval_secs,
                oom_streak,
                streak_window_secs,
            )),
            device,
            probe: Box::new(device_probe),
        }
    }

    /// Reemplaza la sonda del device ([`device_probe`]) por una de mentira. **Sólo para tests**: es el
    /// mismo contrato que `SpyBackend` cumple para el backend, y es lo que hace determinista el camino
    /// de «el device no contesta» sin una GPU que se pueda perder a voluntad.
    pub fn set_device_probe(&mut self, probe: DeviceProbe) {
        self.probe = probe;
    }

    /// Pregunta al device si sigue vivo ([`device_probe`]).
    pub fn probe_device(&self) -> Result<(), String> {
        (self.probe)(&self.device)
    }

    /// Lo que `GET /status` publica bajo `gpu` (`TASK-77`/S84).
    #[must_use]
    pub fn gpu_report(&self) -> GpuHealthReport {
        self.gpu_health.report()
    }

    /// Returns a shared reference to the inner `GpuHealth` telemetry struct.
    pub fn gpu_health(&self) -> Arc<GpuHealth> {
        self.gpu_health.clone()
    }

    pub fn is_healthy(&self) -> bool {
        self.gpu_health.is_healthy()
    }

    pub fn note_request_served(&self) {
        self.gpu_health.note_request_served();
    }

    /// Anota un fallo del backend y decide si este nodo sigue en la malla: la clasificación
    /// ([`gpu_health::classify_gpu_failure`]) y la decisión (`gpu_health::failure_verdict`) viven en el
    /// core, esta función es el único sitio del worker que las **aplica**.
    pub fn note_gpu_failure(&self, context: &str, failure: &str) -> Option<GpuVerdict> {
        let kind = gpu_health::classify_gpu_failure(failure)?;
        Some(self.note_gpu_failure_kind(context, kind, failure))
    }

    /// [`Self::note_gpu_failure`] con la clase ya decidida (lo usa `Scheduler::run_single`, que la
    /// necesita también para decidir **antes** de lanzar el paso).
    pub fn note_gpu_failure_kind(
        &self,
        context: &str,
        kind: GpuFailureKind,
        failure: &str,
    ) -> GpuVerdict {
        let (device_answers, probe_error) = if gpu_health::probe_needed(kind) {
            match self.probe_device() {
                Ok(()) => (true, String::new()),
                Err(probe_error) => (false, probe_error),
            }
        } else {
            (true, String::new())
        };
        let verdict = gpu_health::failure_verdict(kind, device_answers);

        match verdict {
            GpuVerdict::KeepServing => {
                if kind == GpuFailureKind::OutOfMemory {
                    self.gpu_health.note_request_out_of_memory();
                } else {
                    self.gpu_health.note_transient_device_failure();
                }
                warn!(
                    "⚠️ GPU: {} in {context} ({}). The device answered its probe, so this node keeps \
                     serving: it is a statement about this request, not about this node.",
                    kind.as_str(),
                    gpu_health::bounded(failure)
                );
            }
            GpuVerdict::DeclareDown => {
                let announced = self
                    .gpu_health
                    .mark_down(kind, context, failure, &probe_error);
                if announced {
                    error!(
                        "🛑 GPU: this node is DOWN ({} in {context}: {} — probe: {}). It leaves the \
                         mesh and revalidates its device on every heartbeat; it recovers on its own as \
                         soon as the probe answers again (TASK-77).",
                        kind.as_str(),
                        gpu_health::bounded(failure),
                        gpu_health::bounded(&probe_error)
                    );
                }
            }
        }
        verdict
    }

    /// Revalida el device de un nodo que se declaró caído (`TASK-77`/S84): una sonda acotada, y el nodo
    /// vuelve a la malla si el device contesta.
    pub fn revalidate_gpu(&self) -> Result<(), String> {
        if self.gpu_health.is_healthy() {
            return Ok(());
        }
        match self.probe_device() {
            Ok(()) => {
                let down_secs = self.gpu_health.report().down_secs.unwrap_or(0);
                if self.gpu_health.mark_recovered(GpuRecovery::DeviceProbe) {
                    info!(
                        "✅ GPU: the device answers again after {}s down — this node is serving, and the \
                         next heartbeat announces it healthy (TASK-77).",
                        down_secs
                    );
                }
                Ok(())
            }
            Err(probe_error) => {
                self.gpu_health.note_probe_failure(&probe_error);
                Err(probe_error)
            }
        }
    }

    /// El guardián de un nodo caído, antes de gastar un solo byte de cálculo: **revalida** y, sólo si el
    /// device sigue sin contestar, se niega **con el motivo**.
    pub fn refuse_if_gpu_down(&self) -> Result<(), DomainError> {
        if self.gpu_health.is_healthy() {
            return Ok(());
        }
        if self.revalidate_gpu().is_ok() {
            return Ok(());
        }
        let reason = self
            .gpu_health
            .down_reason()
            .unwrap_or_else(|| "GPU state is unhealthy/degraded".to_string());
        warn!("🚫 GPU: refusing a request — {reason}");
        Err(DomainError::Inference(format!(
            "this node's GPU is down, so it cannot serve this step: {reason}"
        )))
    }
}
