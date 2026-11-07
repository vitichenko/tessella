//! Qué le pasa a **este nodo** cuando su GPU falla (`TASK-77`/S84).
//!
//! `S76` midió el defecto: un `prefill` de 4096 tokens que no cabía en la VRAM dejó al nodo con
//! `is_healthy = false` **para siempre**. El `OOM` era transitorio —la petición siguiente, más corta,
//! cabía— y el nodo, en cambio, se quedaba fuera de la malla hasta que alguien lo reiniciaba: el
//! tracker recibía `Degraded` en cada `/announce` y dejaba de mandarle trabajo. En todo el árbol no
//! había **ninguna** línea que volviera a poner el flag en `true`, sólo un `reload_shard` con éxito.
//!
//! El flag no podía contar la verdad que hacía falta, porque la pregunta era una sola:
//! `is_cuda_gpu_error` devolvía un `bool` para `"oom"`, `"cuda"`, `"device"`, `"kernel"`…, así que el
//! llamante no podía separar dos hechos distintos:
//!
//! 1. **«esta petición no cupo»** — un juicio sobre la **petición**, no sobre el nodo. Lo que
//!    corresponde es contarlo (`GET /status` → `gpu.out_of_memory`) y seguir sirviendo: el nodo
//!    sigue teniendo razón, y el cliente que pidió demasiado se lleva la negativa de su petición.
//! 2. **«el device dejó de responder»** — un juicio sobre el **nodo**. Antes de creérselo, una sonda
//!    acotada al propio device (4 bytes, la misma que decidió el veredicto): si la sonda pasa, el
//!    fallo era de la petición y el nodo sigue en la malla; si no pasa, el nodo **se declara caído
//!    con un motivo** y lo revalida cada [`GpuHealth::revalidate_secs`] —no «hasta el reinicio»—.
//!
//! Este módulo es la decisión, en funciones puras y un puñado de contadores, para que se pueda probar
//! sin GPU, sin tracker y sin socket, como `heartbeat.rs` o
//! `musivum_api::failure_reports`. La sonda y las líneas de log viven en el llamante
//! (`InferenceEngine::note_gpu_failure`, que es quien tiene el `Device` y el contexto del fallo).
//!
//! **Lo que este módulo no hace**, declarado para que nadie lo lea de más: no detecta una fuga de
//! VRAM. Si un nodo `OOM`ea en **cada** petición y su device sigue respondiendo a la sonda, aquí no
//! se declara caído; lo que queda es el contador publicado y, en la malla, el registro de fallos por
//! peer del router (`S56`), que es quien deja de elegirlo. Escalar «N `OOM` seguidos ⇒ caído» exigiría
//! medir antes cuántos son normales, y eso no está medido.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use serde::Serialize;

/// Longest detail kept of a failure or of a probe (characters, never bytes: a cut in the middle of a
/// character is what turns a log line into a mojibake). The reason travels to a log line and to
/// `GET /status`, so it is bounded like every other field the operator reads.
pub const MAX_DETAIL_CHARS: usize = 400;

/// Por qué falló la GPU de este nodo (`TASK-77`/S84).
///
/// The distinction is the whole task: one of the two is a statement about a **request** and the other
/// about the **device**, and the caller can now act on them differently. Before, both arrived as the
/// same `true` from `is_cuda_gpu_error` and both took the node out of the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFailureKind {
    /// One allocation did not fit: what the failed step asked of the VRAM is bigger than what is
    /// free. `CUDA_ERROR_OUT_OF_MEMORY` and the `OOM` of a kernel belong here.
    OutOfMemory,
    /// The device itself stopped answering: an illegal access, a lost context, a driver or PTX
    /// failure. This is the only kind that can declare the node down, and only if the probe confirms.
    DeviceLost,
}

impl GpuFailureKind {
    /// The name published by `GET /status`. Stable: it is part of the operator-facing surface.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OutOfMemory => "out_of_memory",
            Self::DeviceLost => "device_lost",
        }
    }
}

/// Clasifica el error de un cálculo que falló (`TASK-77`/S84).
///
/// The **order** is the task: `CUDA_ERROR_OUT_OF_MEMORY` matches both the memory pattern and the
/// `cuda` one, and the first has to win, because it is what tells «this request did not fit» from
/// «this device died». `None` means the error says nothing about the GPU, which is what the old
/// `is_cuda_gpu_error` could not express either.
#[must_use]
pub fn classify_gpu_failure(err: &str) -> Option<GpuFailureKind> {
    let lower = err.to_lowercase();
    if lower.contains("out of memory") || lower.contains("oom") {
        return Some(GpuFailureKind::OutOfMemory);
    }
    // Los patrones del retirado `is_cuda_gpu_error`, verbatim: esta lista es la que venía decidiendo
    // con un `bool`, así que conservarla es lo que hace auditable el cambio.
    if lower.contains("cuda")
        || lower.contains("ptx")
        || lower.contains("illegal memory")
        || lower.contains("gpu")
        || lower.contains("device")
        || lower.contains("kernel")
    {
        return Some(GpuFailureKind::DeviceLost);
    }
    None
}

/// Qué dice un fallo de **esta** petición sobre **este nodo** (`TASK-77`/S84).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVerdict {
    /// The node stays in the mesh and keeps serving. It is the answer for an `OOM` (a statement about
    /// the request) and for a device failure the probe did not confirm.
    KeepServing,
    /// The device did not answer the bounded probe: the node declares itself down, **with the reason**,
    /// and revalidates itself on every heartbeat.
    DeclareDown,
}

/// Si hace falta preguntarle al device para decidir sobre `kind` (`TASK-77`/S84).
///
/// Es la mitad de [`failure_verdict`] que el llamante necesita **antes** de pagar la sonda: un `OOM` es
/// un juicio sobre la petición y su veredicto es el mismo conteste lo que conteste el device, así que
/// pedirlo sería gastar una reserva (y, en un allocator fragmentado, correr el riesgo de que una
/// petición grande declare caído a un nodo vivo). Está aquí, y no en el worker, para que las dos tablas
/// no puedan separarse: el test de abajo las ata.
#[must_use]
pub const fn probe_needed(kind: GpuFailureKind) -> bool {
    matches!(kind, GpuFailureKind::DeviceLost)
}

/// La decisión de un cálculo que falló, como función pura (`TASK-77`/S84).
///
/// `device_answers` is the answer of the bounded probe (4 bytes allocated and read back) run **after**
/// the failure, by the caller that owns the `Device`. It is a `bool` and not the probe's error string
/// because the decision does not depend on *why* the device did not answer, only on whether it did.
///
/// An `OOM` short-circuits the whole thing: it is a judgement about the request, so the probe is not
/// even consulted. The alternative —asking the device after every `OOM`— would let a request that was
/// merely too big declare the node down whenever the allocator happened to be fragmented, which is
/// the same class of lie this task removes, only in the other direction.
#[must_use]
pub const fn failure_verdict(kind: GpuFailureKind, device_answers: bool) -> GpuVerdict {
    if device_answers {
        return GpuVerdict::KeepServing;
    }
    match kind {
        GpuFailureKind::OutOfMemory => GpuVerdict::KeepServing,
        GpuFailureKind::DeviceLost => GpuVerdict::DeclareDown,
    }
}

/// Cómo volvió un nodo que se había declarado caído (`TASK-77`/S84).
///
/// Se publica (`GET /status` → `gpu.last_recovery`) porque las formas significan cosas distintas
/// para quien lee: un `shard_reload` es una intervención del operador o del tracker, un
/// `device_probe` es un nodo cuyo hardware volvió solo, un `window_expired` es la expiración pasiva
/// de la ventana de saturación OOM, y un `request_served` es una petición exitosa que rompió la racha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRecovery {
    /// `NetworkManager::reload_shard` succeeded: other weights are loaded and the device answers.
    ShardReload,
    /// The bounded probe of the device answered again (on a heartbeat tick, or on a request that
    /// arrived while the node was down).
    DeviceProbe,
    /// The OOM streak window expired without new failures (`TASK-85`/S92).
    WindowExpired,
    /// A request was successfully served, clearing the OOM streak (`TASK-85`/S92).
    RequestServed,
}

/// Number of [`GpuRecovery`] variants: it is the length of the published name table, and it is named
/// because `GpuRecovery::ALL.len()` is not usable in a const position.
pub const RECOVERY_COUNT: usize = 4;

impl GpuRecovery {
    pub const ALL: [Self; RECOVERY_COUNT] = [
        Self::ShardReload,
        Self::DeviceProbe,
        Self::WindowExpired,
        Self::RequestServed,
    ];

    /// The name published by `GET /status`. Stable: it is part of the operator-facing surface.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ShardReload => "shard_reload",
            Self::DeviceProbe => "device_probe",
            Self::WindowExpired => "window_expired",
            Self::RequestServed => "request_served",
        }
    }

    /// Index in the recovery counter. The enum has no explicit discriminants, so this is the
    /// declaration order and it cannot disagree with [`Self::ALL`].
    fn index(self) -> usize {
        self as usize
    }
}

/// Default consecutive OOMs required to declare saturation (`TASK-85`/S92).
pub const DEFAULT_OOM_STREAK: u32 = 3;

/// Default time window in seconds for the OOM streak (`TASK-85`/S92).
pub const DEFAULT_STREAK_WINDOW_SECS: u64 = 30;

/// El estado de la GPU de **un** nodo, como `GET /status` lo publica bajo `gpu` (`TASK-77`/S84 · `TASK-85`/S92).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuHealthReport {
    pub healthy: bool,
    /// `serving` | `down` | `saturated`: el estado en palabras. Se publica porque el operador
    /// lee un estado y no un booleano, y porque los veredictos de esta tarea tienen nombre.
    pub verdict: &'static str,
    /// `GpuFailureKind::as_str` del fallo que declaró caído al nodo (`None` mientras sirve).
    pub kind: Option<&'static str>,
    /// Dónde pasó (`forward_hidden`, `Scheduler::run_single`, …): es el sitio con el que el operador
    /// llega al resto del log.
    pub context: Option<String>,
    /// Qué contestó el device, acotado por [`MAX_DETAIL_CHARS`].
    pub reason: Option<String>,
    /// Qué contestó la **última** sonda de revalidación, acotado. Es el estado de ahora y no el primer
    /// fallo: un nodo caído desde hace una hora publica por qué **sigue** caído.
    pub probe_error: Option<String>,
    pub since_unix_secs: Option<u64>,
    /// Cuánto hace que el nodo se declaró caído (`None` mientras sirve).
    pub down_secs: Option<u64>,
    /// Cada cuántos segundos este nodo le pregunta a su propio device si volvió a la vida. Es el
    /// `heartbeat_interval_secs` de `worker.toml`: **la cota** de la recuperación, publicada para que
    /// el operador no tenga que adivinarla. `0` = el nodo no se revalida (una fixture, o un despliegue
    /// sin heartbeat), y es un estado que esta tarea se niega a esconder.
    pub revalidate_secs: u64,
    /// Umbral configurado de OOMs consecutivos para saturación (`TASK-85`/S92).
    pub oom_streak: u32,
    /// OOMs consecutivos activos acumulados en la ventana actual (`TASK-85`/S92).
    pub active_oom_streak: u32,
    /// Duración de la ventana de saturación en segundos (`TASK-85`/S92).
    pub streak_window_secs: u64,
    /// `OOM`s de peticiones sueltas. Un número alto aquí es un cliente mandando prompts que este nodo
    /// no puede prefillar, y es el número que **antes** sacaba al nodo de la malla.
    pub out_of_memory: u64,
    /// Fallos de device que la sonda **no** confirmó: la petición falló, el device respondió.
    pub device_failures: u64,
    /// Cuántas veces volvió este nodo de un `down` o de una saturación.
    pub recoveries: u64,
    /// `GpuRecovery::as_str` de la última vuelta (`None` = nunca se declaró caído ni saturado).
    pub last_recovery: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct GpuSaturationState {
    current_streak: u32,
    first_failure_unix_secs: Option<u64>,
}

/// La GPU de un nodo: si está sirviendo, y si no, por qué y desde cuándo (`TASK-77`/S84 · `TASK-85`/S92).
///
/// Cheap to read from every request path (atomics and, only when a node is down or saturated, one short lock) and
/// cheap to clone-free share: it lives inside `Arc<InferenceEngine>` / `Arc<Scheduler>`, which is where
/// the two places that used to write `is_healthy` by hand can now ask.
#[derive(Debug)]
pub struct GpuHealth {
    healthy: AtomicBool,
    out_of_memory: AtomicU64,
    device_failures: AtomicU64,
    recoveries: AtomicU64,
    /// `GpuRecovery::index() + 1` de la última vuelta; `0` = nunca hubo una.
    last_recovery: AtomicU8,
    /// El heartbeat del nodo que lo tiene: la cota de la revalidación, fijada en el arranque.
    revalidate_secs: u64,
    /// Umbral de OOMs consecutivos para declarar saturación.
    oom_streak: u32,
    /// Ventana de tiempo (en segundos) en la que se acumula la racha.
    streak_window_secs: u64,
    /// Dónde y por qué está caído el nodo. `None` mientras sirve. Es un `Mutex` y no más atómicos
    /// porque los tres textos viajan juntos y quien lee tiene que ver un veredicto entero, no medio.
    down: Mutex<Option<GpuDown>>,
    /// Racha de OOMs y tiempo del primer fallo de la racha (`TASK-85`/S92).
    saturation: Mutex<GpuSaturationState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuDown {
    kind: GpuFailureKind,
    context: String,
    reason: String,
    probe_error: String,
    since_unix_secs: u64,
}

impl GpuHealth {
    /// `revalidate_secs` = [`GpuHealthReport::revalidate_secs`], `oom_streak`, `streak_window_secs`.
    #[must_use]
    pub const fn new(revalidate_secs: u64, oom_streak: u32, streak_window_secs: u64) -> Self {
        Self {
            healthy: AtomicBool::new(true),
            out_of_memory: AtomicU64::new(0),
            device_failures: AtomicU64::new(0),
            recoveries: AtomicU64::new(0),
            last_recovery: AtomicU8::new(0),
            revalidate_secs,
            oom_streak,
            streak_window_secs,
            down: Mutex::new(None),
            saturation: Mutex::new(GpuSaturationState {
                current_streak: 0,
                first_failure_unix_secs: None,
            }),
        }
    }

    /// Cada cuántos segundos un nodo caído le pregunta a su device si volvió.
    #[must_use]
    pub const fn revalidate_secs(&self) -> u64 {
        self.revalidate_secs
    }

    /// Umbral configurado de OOMs consecutivos para saturación (`TASK-85`/S92).
    #[must_use]
    pub const fn oom_streak(&self) -> u32 {
        self.oom_streak
    }

    /// Ventana de tiempo (en segundos) en la que se acumula la racha (`TASK-85`/S92).
    #[must_use]
    pub const fn streak_window_secs(&self) -> u64 {
        self.streak_window_secs
    }

    /// The flag the rest of the node reads. The tracker gets the same fact through `/announce`.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    /// Cuenta un `OOM` de una petición con la marca de tiempo actual (`TASK-85`/S92).
    pub fn note_request_out_of_memory(&self) {
        self.note_request_out_of_memory_at(crate::time::unix_now_secs());
    }

    /// Cuenta un `OOM` de una petición en el segundo `now_secs` (`TASK-85`/S92).
    ///
    /// Si el fallo ocurre fuera de la ventana activa desde el primer fallo, se reinicia la racha.
    /// Si la racha alcanza o supera [`Self::oom_streak`], el nodo queda saturado.
    pub fn note_request_out_of_memory_at(&self, now_secs: u64) {
        self.out_of_memory.fetch_add(1, Ordering::Relaxed);
        let mut sat = self.lock_saturation();
        if sat
            .first_failure_unix_secs
            .is_some_and(|first| now_secs.saturating_sub(first) > self.streak_window_secs)
        {
            sat.current_streak = 0;
            sat.first_failure_unix_secs = None;
        }
        if sat.current_streak == 0 {
            sat.first_failure_unix_secs = Some(now_secs);
        }
        sat.current_streak = sat.current_streak.saturating_add(1);
    }

    /// Anota que una petición fue servida con éxito (`TASK-85`/S92).
    ///
    /// Rompe la racha de OOMs a 0. Si el nodo estaba saturado (`current_streak >= oom_streak`),
    /// registra la recuperación activa como [`GpuRecovery::RequestServed`].
    pub fn note_request_served(&self) {
        let mut sat = self.lock_saturation();
        if sat.current_streak >= self.oom_streak {
            self.record_recovery(GpuRecovery::RequestServed);
        }
        sat.current_streak = 0;
        sat.first_failure_unix_secs = None;
    }

    /// Comprueba si la ventana de saturación expiró con el reloj actual (`TASK-85`/S92).
    pub fn check_window_expiration(&self) -> bool {
        self.check_window_expiration_at(crate::time::unix_now_secs())
    }

    /// Comprueba si la ventana de saturación expiró en el instante `now_secs` (`TASK-85`/S92).
    ///
    /// Si el nodo estaba saturado y han transcurrido al menos `streak_window_secs` desde el primer
    /// fallo de la racha sin nuevos fallos, limpia la racha y anota [`GpuRecovery::WindowExpired`].
    pub fn check_window_expiration_at(&self, now_secs: u64) -> bool {
        let mut sat = self.lock_saturation();
        if sat.current_streak >= self.oom_streak
            && sat
                .first_failure_unix_secs
                .is_some_and(|first| now_secs.saturating_sub(first) >= self.streak_window_secs)
        {
            sat.current_streak = 0;
            sat.first_failure_unix_secs = None;
            self.record_recovery(GpuRecovery::WindowExpired);
            return true;
        }
        false
    }

    /// Devuelve `true` si el nodo ha acumulado al menos [`Self::oom_streak`] fallos en su ventana activa (`TASK-85`/S92).
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        self.is_saturated_at(crate::time::unix_now_secs())
    }

    /// Devuelve `true` si el nodo está saturado en el instante `now_secs` (`TASK-85`/S92).
    #[must_use]
    pub fn is_saturated_at(&self, now_secs: u64) -> bool {
        self.check_window_expiration_at(now_secs);
        self.lock_saturation().current_streak >= self.oom_streak
    }

    /// Cuenta un fallo de device que la sonda **sí** desmintió (el device respondió): el nodo se queda
    /// en la malla y lo que queda es el contador.
    pub fn note_transient_device_failure(&self) {
        self.device_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Declara el nodo caído con el motivo (`TASK-77`/S84).
    ///
    /// Devuelve `true` si esta llamada es la que **cambió** el estado, que es cuando el llamante
    /// escribe la línea de log: una por transición y no una por petición que falla.
    ///
    /// `since` no se mueve si el nodo ya estaba caído: el nodo lleva caído desde la primera vez, y
    /// reescribirlo cada reintento publicaría una antigüedad más corta que la real.
    pub fn mark_down(
        &self,
        kind: GpuFailureKind,
        context: &str,
        reason: &str,
        probe_error: &str,
    ) -> bool {
        let now = crate::time::unix_now_secs();
        let reason = bounded(reason);
        let probe_error = bounded(probe_error);
        let mut down = self.lock_down();
        let was_healthy = self.healthy.swap(false, Ordering::Relaxed);
        let since_unix_secs = down.as_ref().map_or(now, |held| held.since_unix_secs);
        *down = Some(GpuDown {
            kind,
            context: context.to_string(),
            reason,
            probe_error,
            since_unix_secs,
        });
        was_healthy
    }

    /// Anota qué contestó la última sonda de revalidación que **no** trajo al nodo de vuelta.
    ///
    /// El primer motivo por el que un nodo cayó es historia; esto es lo que contesta **ahora**, y es lo
    /// que le dice al operador si algo cambió (un `cudaMalloc` que sigue fallando no es la misma
    /// situación que un device que contesta y un driver que rechaza el kernel).
    pub fn note_probe_failure(&self, probe_error: &str) {
        let probe_error = bounded(probe_error);
        let mut down = self.lock_down();
        if let Some(held) = down.as_mut() {
            held.probe_error = probe_error;
        }
    }

    /// Registra una recuperación en el contador y en el slot de última vuelta.
    fn record_recovery(&self, how: GpuRecovery) {
        self.recoveries.fetch_add(1, Ordering::Relaxed);
        let slot = u8::try_from(how.index() + 1).unwrap_or(0);
        self.last_recovery.store(slot, Ordering::Relaxed);
    }

    /// Vuelve a la malla (`TASK-77`/S84). Devuelve `true` cuando esta llamada es la que **cambió** el
    /// estado, que es cuando el llamante escribe la línea de log (una por transición, también al
    /// volver).
    ///
    /// Un nodo que ya estaba sirviendo no inventa una vuelta: el contador es lo que el operador lee
    /// para saber si alguna vez hubo un `down`, así que sólo se mueve en una transición de verdad.
    pub fn mark_recovered(&self, how: GpuRecovery) -> bool {
        if self.healthy.swap(true, Ordering::Relaxed) {
            return false;
        }
        self.record_recovery(how);
        *self.lock_down() = None;
        true
    }

    /// Una línea con el motivo, para el error que recibe el cliente que llega a un nodo caído: negarse
    /// sin decir por qué es lo que convierte una caída en un misterio.
    #[must_use]
    pub fn down_reason(&self) -> Option<String> {
        self.lock_down().as_ref().map(|held| {
            format!(
                "{} in {}: {} (probe: {})",
                held.kind.as_str(),
                held.context,
                held.reason,
                held.probe_error,
            )
        })
    }

    /// El estado completo, como `GET /status` lo publica (`TASK-77`/S84 · `TASK-85`/S92).
    #[must_use]
    pub fn report(&self) -> GpuHealthReport {
        let healthy = self.is_healthy();
        let saturated = self.is_saturated();
        let down = self.lock_down();
        let now = crate::time::unix_now_secs();
        let slot = usize::from(self.last_recovery.load(Ordering::Relaxed));
        let active_streak = self.lock_saturation().current_streak;
        let verdict = if !healthy {
            "down"
        } else if saturated {
            "saturated"
        } else {
            "serving"
        };
        GpuHealthReport {
            healthy,
            verdict,
            kind: down.as_ref().map(|held| held.kind.as_str()),
            context: down.as_ref().map(|held| held.context.clone()),
            reason: down.as_ref().map(|held| held.reason.clone()),
            probe_error: down.as_ref().map(|held| held.probe_error.clone()),
            since_unix_secs: down.as_ref().map(|held| held.since_unix_secs),
            down_secs: down
                .as_ref()
                .map(|held| now.saturating_sub(held.since_unix_secs)),
            revalidate_secs: self.revalidate_secs,
            oom_streak: self.oom_streak,
            active_oom_streak: active_streak,
            streak_window_secs: self.streak_window_secs,
            out_of_memory: self.out_of_memory.load(Ordering::Relaxed),
            device_failures: self.device_failures.load(Ordering::Relaxed),
            recoveries: self.recoveries.load(Ordering::Relaxed),
            last_recovery: slot
                .checked_sub(1)
                .and_then(|index| GpuRecovery::ALL.get(index))
                .map(|how| how.as_str()),
        }
    }

    /// El registro de saturación, tolerando un mutex envenenado.
    fn lock_saturation(&self) -> std::sync::MutexGuard<'_, GpuSaturationState> {
        self.saturation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// El registro de caída, tolerando un mutex envenenado: dentro sólo hay datos planos, y perder el
    /// motivo de que un nodo esté caído por un pánico ajeno es peor que leerlo igual.
    fn lock_down(&self) -> std::sync::MutexGuard<'_, Option<GpuDown>> {
        self.down
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Recorta un detalle a [`MAX_DETAIL_CHARS`] caracteres, sin partir uno por la mitad y en una línea:
/// el motivo viaja a una línea de log y a un cuerpo JSON, y un `\n` ahí es una entrada falsificada.
///
/// Es `pub` porque el worker también escribe el motivo del backend en sus `warn!`/`error!`
/// (`TASK-77`/S84): un mensaje de error de la GPU con un salto de línea dentro es exactamente la misma
/// falsificación, y recortarlo en un solo sitio es lo que garantiza que el número publicado y el
/// número del log son el mismo.
pub fn bounded(detail: &str) -> String {
    let mut out = String::with_capacity(detail.len().min(MAX_DETAIL_CHARS + 1));
    for (index, ch) in detail.chars().enumerate() {
        if index == MAX_DETAIL_CHARS {
            out.push('…');
            return out;
        }
        out.push(if ch.is_control() { ' ' } else { ch });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Criterio 1 de `TASK-77`: un `OOM` es un juicio sobre la **petición**, no sobre el nodo. El caso
    /// que midió `S76` —`prefill` de 4096 tokens que no cabe— deja al nodo sirviendo.
    #[test]
    fn an_out_of_memory_does_not_take_the_node_down() {
        let kind = classify_gpu_failure("CUDA_ERROR_OUT_OF_MEMORY: out of memory")
            .expect("un OOM de CUDA sí es un fallo de GPU");
        assert_eq!(kind, GpuFailureKind::OutOfMemory, "y no un device perdido");

        // El veredicto ni siquiera consulta la sonda: una petición que no cupo no puede declarar caído
        // a un nodo aunque el allocator esté fragmentado y la sonda de 4 bytes falle.
        assert_eq!(failure_verdict(kind, false), GpuVerdict::KeepServing);
        assert_eq!(failure_verdict(kind, true), GpuVerdict::KeepServing);
    }

    /// Un `OOM` se cuenta y el nodo sigue `healthy`: el contador es lo que ve el operador, y es lo
    /// único que ve, porque no hay motivo de caída que publicar.
    #[test]
    fn an_out_of_memory_is_counted_without_moving_the_flag() {
        let health = GpuHealth::new(30, 3, 30);
        health.note_request_out_of_memory();
        health.note_request_out_of_memory();

        assert!(health.is_healthy(), "el nodo sigue en la malla");
        assert!(
            !health.is_saturated(),
            "2 OOMs está por debajo del umbral 3"
        );
        let report = health.report();
        assert_eq!(report.out_of_memory, 2);
        assert_eq!(report.active_oom_streak, 2);
        assert_eq!(report.oom_streak, 3);
        assert_eq!(report.streak_window_secs, 30);
        assert_eq!(report.verdict, "serving");
        assert_eq!(report.kind, None, "no hay motivo de caída que publicar");
        assert_eq!(report.since_unix_secs, None);
        assert_eq!(report.down_secs, None);
    }

    /// `TASK-85`/S92: el umbral se pinza por ambos lados. Por debajo de `oom_streak`, el veredicto
    /// sigue siendo `serving`; al alcanzarlo, el veredicto pasa a `saturated` sin tocar `healthy` ni `down`.
    #[test]
    fn crossing_oom_streak_threshold_changes_verdict_to_saturated() {
        let health = GpuHealth::new(30, 3, 30);

        // 1 y 2 fallos: por debajo del umbral, nada cambia
        health.note_request_out_of_memory();
        assert_eq!(health.report().verdict, "serving");
        assert!(!health.is_saturated());

        health.note_request_out_of_memory();
        assert_eq!(health.report().verdict, "serving");
        assert!(!health.is_saturated());

        // 3 fallos: cruza el umbral -> saturated
        health.note_request_out_of_memory();
        assert!(health.is_saturated(), "3 OOMs alcanza el umbral");
        assert!(
            health.is_healthy(),
            "el hardware sigue respondiendo a la sonda"
        );
        let report = health.report();
        assert_eq!(report.verdict, "saturated");
        assert_eq!(report.active_oom_streak, 3);
        assert_eq!(report.oom_streak, 3);
        assert_eq!(report.streak_window_secs, 30);
        assert_eq!(report.kind, None, "no es una caída de hardware");
    }

    /// `TASK-85`/S92 camino de vuelta 1: una petición servida con éxito rompe la racha y vuelve el
    /// nodo a `serving`, registrando `last_recovery = "request_served"`.
    #[test]
    fn saturated_node_recovers_when_request_is_served() {
        let health = GpuHealth::new(30, 3, 30);
        health.note_request_out_of_memory();
        health.note_request_out_of_memory();
        health.note_request_out_of_memory();
        assert_eq!(health.report().verdict, "saturated");

        health.note_request_served();
        assert!(!health.is_saturated());
        let report = health.report();
        assert_eq!(report.verdict, "serving");
        assert_eq!(report.active_oom_streak, 0);
        assert_eq!(report.recoveries, 1);
        assert_eq!(report.last_recovery, Some("request_served"));
    }

    /// `TASK-85`/S92 camino de vuelta 2: la expiración de la ventana temporal (`streak_window_secs`)
    /// limpia la racha sin reinicio y anota `last_recovery = "window_expired"`.
    #[test]
    fn saturated_node_recovers_when_window_expires() {
        let health = GpuHealth::new(30, 3, 30);
        let t0 = 1000;
        health.note_request_out_of_memory_at(t0);
        health.note_request_out_of_memory_at(t0 + 5);
        health.note_request_out_of_memory_at(t0 + 10);
        assert!(health.is_saturated_at(t0 + 15));

        // A los 29 segundos desde el primer fallo, sigue saturado
        assert!(health.is_saturated_at(t0 + 29));

        // A los 30 segundos (ventana expirada), vuelve solo a serving
        assert!(!health.is_saturated_at(t0 + 30));
        let report = health.report();
        assert_eq!(report.verdict, "serving");
        assert_eq!(report.active_oom_streak, 0);
        assert_eq!(report.recoveries, 1);
        assert_eq!(report.last_recovery, Some("window_expired"));
    }

    /// `TASK-85`/S92: OOMs aislados separados por más de la ventana reinician la racha y no acumulan.
    #[test]
    fn sub_threshold_ooms_reset_if_spaced_beyond_window() {
        let health = GpuHealth::new(30, 3, 30);
        let t0 = 1000;
        health.note_request_out_of_memory_at(t0);
        health.note_request_out_of_memory_at(t0 + 10);
        assert_eq!(health.report().active_oom_streak, 2);

        // El siguiente fallo llega a t0 + 35s (> 30s de ventana): la racha se reinicia a 1
        health.note_request_out_of_memory_at(t0 + 35);
        assert_eq!(health.report().active_oom_streak, 1);
        assert_eq!(health.report().verdict, "serving");
    }

    /// La sonda se paga **sólo** cuando el veredicto depende de ella: las dos tablas
    /// ([`probe_needed`] y [`failure_verdict`]) no pueden separarse, porque el llamante decide con la
    /// primera si consulta y con la segunda qué hace con la respuesta.
    #[test]
    fn the_device_probe_is_asked_only_when_the_verdict_depends_on_it() {
        for kind in [GpuFailureKind::OutOfMemory, GpuFailureKind::DeviceLost] {
            if probe_needed(kind) {
                continue;
            }
            assert_eq!(
                failure_verdict(kind, true),
                failure_verdict(kind, false),
                "{kind:?} se decidió sin preguntar, así que la respuesta del device no puede cambiarla"
            );
        }
        assert!(
            probe_needed(GpuFailureKind::DeviceLost),
            "el único caso en el que la respuesta del device decide es el device perdido"
        );
    }

    /// Criterio 2: un fallo de device que la sonda **desmiente** tampoco saca al nodo de la malla.
    #[test]
    fn a_lost_device_that_answers_the_probe_keeps_the_node_serving() {
        let kind = classify_gpu_failure("CUDA error: an illegal memory access was encountered")
            .expect("un acceso ilegal es un fallo de device");
        assert_eq!(kind, GpuFailureKind::DeviceLost);
        assert_eq!(failure_verdict(kind, true), GpuVerdict::KeepServing);

        let health = GpuHealth::new(30, 3, 30);
        health.note_transient_device_failure();

        assert!(health.is_healthy(), "la sonda respondió: el nodo sirve");
        let report = health.report();
        assert_eq!(report.device_failures, 1);
        assert_eq!(report.kind, None);
        assert_eq!(report.recoveries, 0, "no hubo vuelta porque no hubo caída");
    }

    /// Criterio 3: un device que **no** responde declara caído al nodo, con motivo y sitio, y todo eso
    /// se publica. Antes de esta tarea la información se perdía en un `false`.
    #[test]
    fn a_lost_device_that_does_not_answer_takes_the_node_down_with_its_reason() {
        let health = GpuHealth::new(15, 3, 30);
        let changed = health.mark_down(
            GpuFailureKind::DeviceLost,
            "forward_hidden",
            "CUDA error: an illegal memory access was encountered",
            "cudaMalloc failed: driver shutting down",
        );

        assert!(changed, "es la primera vez: la transición se anota");
        assert!(!health.is_healthy());
        let report = health.report();
        assert_eq!(report.verdict, "down");
        assert_eq!(report.kind, Some("device_lost"));
        assert_eq!(report.context.as_deref(), Some("forward_hidden"));
        assert_eq!(
            report.reason.as_deref(),
            Some("CUDA error: an illegal memory access was encountered")
        );
        assert_eq!(
            report.probe_error.as_deref(),
            Some("cudaMalloc failed: driver shutting down"),
            "por qué la sonda no pudo desmentirlo"
        );
        assert!(report.since_unix_secs.is_some());
        assert_eq!(report.down_secs, Some(0), "recién declarado");
        assert_eq!(
            report.revalidate_secs, 15,
            "la cota de la recuperación se publica"
        );

        let reason = health.down_reason().expect("un nodo caído tiene motivo");
        assert!(reason.contains("device_lost"), "{reason}");
        assert!(reason.contains("forward_hidden"), "{reason}");
    }

    /// Criterio 4: la mitad «o se recupera» de la tarea. Un nodo caído vuelve **solo** en cuanto su
    /// device responde, y `last_recovery` distingue las dos formas de volver.
    #[test]
    fn a_down_node_comes_back_when_its_device_answers_again() {
        let health = GpuHealth::new(30, 3, 30);
        health.mark_down(
            GpuFailureKind::DeviceLost,
            "forward_hidden",
            "CUDA error: illegal memory access",
            "cudaMalloc failed",
        );
        health.note_probe_failure("cudaMalloc failed: out of memory");

        let report = health.report();
        assert_eq!(
            report.probe_error.as_deref(),
            Some("cudaMalloc failed: out of memory"),
            "lo último que contestó el device es lo que se publica"
        );
        assert!(
            report.since_unix_secs.is_some(),
            "y el motivo inicial sigue fechando la caída"
        );

        assert!(health.mark_recovered(GpuRecovery::DeviceProbe));
        assert!(
            health.is_healthy(),
            "el nodo vuelve a la malla sin reinicio"
        );
        let report = health.report();
        assert_eq!(report.verdict, "serving");
        assert_eq!(report.kind, None, "ya no hay motivo de caída");
        assert_eq!(report.down_secs, None);
        assert_eq!(report.recoveries, 1);
        assert_eq!(report.last_recovery, Some("device_probe"));
        assert!(health.down_reason().is_none());
    }

    /// Criterio 4 (contra-cara): un nodo que nunca se cayó no puede «recuperarse». El contador es lo
    /// que el operador lee para saber si hubo un `down`, así que no se infla.
    #[test]
    fn a_node_that_never_went_down_does_not_invent_a_recovery() {
        let health = GpuHealth::new(30, 3, 30);

        assert!(!health.mark_recovered(GpuRecovery::ShardReload));
        let report = health.report();
        assert_eq!(report.recoveries, 0);
        assert_eq!(report.last_recovery, None);
    }

    /// Un segundo fallo mientras el nodo está caído **no** rejuvenece la antigüedad: reescribirla
    /// publicaría un `down_secs` más corto que el real. El reloj tiene granularidad de segundo, así que
    /// el caso necesita un segundo de verdad; el motivo sí se actualiza.
    #[test]
    fn a_second_failure_while_down_keeps_the_original_since() {
        let health = GpuHealth::new(30, 3, 30);
        health.mark_down(
            GpuFailureKind::DeviceLost,
            "forward_hidden",
            "primero",
            "sonda rota",
        );
        let since = health.report().since_unix_secs;
        std::thread::sleep(std::time::Duration::from_millis(1_100));
        health.mark_down(
            GpuFailureKind::DeviceLost,
            "forward_hidden",
            "segundo",
            "sonda rota",
        );

        let report = health.report();
        assert_eq!(
            report.since_unix_secs, since,
            "la caída empezó cuando empezó"
        );
        assert_eq!(
            report.reason.as_deref(),
            Some("segundo"),
            "el motivo sí es el de ahora"
        );
        assert!(
            report.down_secs.unwrap_or(0) >= 1,
            "y la antigüedad es la real: {:?}",
            report.down_secs
        );
    }

    /// El detalle viaja a una línea de log y a un cuerpo JSON: acotado y en **una** línea, porque un
    /// salto de línea dentro del motivo es una entrada de log falsificada.
    #[test]
    fn the_reason_is_bounded_to_one_line() {
        let health = GpuHealth::new(30, 3, 30);
        let long = format!("primera\nsegunda {}", "x".repeat(MAX_DETAIL_CHARS * 2));
        health.mark_down(
            GpuFailureKind::DeviceLost,
            "forward_hidden",
            &long,
            "probe\nline",
        );

        let report = health.report();
        let reason = report.reason.expect("hay motivo");
        assert_eq!(
            reason.chars().count(),
            MAX_DETAIL_CHARS + 1,
            "acotado: {reason}"
        );
        assert_eq!(reason.chars().next_back(), Some('…'));
        assert!(!reason.contains('\n'), "sin saltos de línea: {reason}");
        let probe = report.probe_error.expect("hay respuesta de la sonda");
        assert_eq!(probe, "probe line");
    }

    /// Un error que no habla de la GPU no es un fallo de GPU, así que el nodo no cambia de estado por
    /// él. El retirado `is_cuda_gpu_error` tampoco lo hacía, pero ahora se puede declarar.
    #[test]
    fn an_error_that_says_nothing_about_the_gpu_is_not_a_gpu_failure() {
        assert_eq!(classify_gpu_failure("shape mismatch at tensor 3"), None);
        assert_eq!(classify_gpu_failure(""), None);
    }

    /// Los nombres que se publican son los de la tabla: si alguien añade una forma de volver sin
    /// nombre, `ALL` y `as_str` se desincronizan y este test lo dice.
    #[test]
    fn every_recovery_has_a_published_name() {
        assert_eq!(GpuRecovery::ALL.len(), RECOVERY_COUNT);
        let mut names: Vec<&'static str> =
            GpuRecovery::ALL.iter().map(|how| how.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), RECOVERY_COUNT, "{names:?}");
    }

    /// El veredicto viaja como JSON (`GET /status` → `gpu`) sin envoltorios: el operador lee
    /// `gpu.verdict` y no `gpu.Some(...)`.
    #[test]
    fn the_report_serializes_the_verdict() {
        let health = GpuHealth::new(30, 3, 30);
        let serving = serde_json::to_value(health.report()).expect("el reporte es serializable");
        assert_eq!(serving["verdict"], "serving");
        assert_eq!(serving["kind"], serde_json::Value::Null);
        assert_eq!(serving["revalidate_secs"], 30);
        assert_eq!(serving["oom_streak"], 3);
        assert_eq!(serving["streak_window_secs"], 30);
        assert_eq!(serving["active_oom_streak"], 0);

        health.mark_down(
            GpuFailureKind::DeviceLost,
            "Scheduler::run_single",
            "boom",
            "sonda sin respuesta",
        );
        let down = serde_json::to_value(health.report()).expect("el reporte es serializable");
        assert_eq!(down["verdict"], "down");
        assert_eq!(down["healthy"], false);
        assert_eq!(down["kind"], "device_lost");
        assert_eq!(down["context"], "Scheduler::run_single");
    }
}
