use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Histogram buckets for proof batch upload duration in seconds.
///
/// Refined with Research AI recommendations:
/// - 5 ms (0.005) for fast local LAN / integration test runs
/// - 10 ms - 10 s standard range
/// - 15 s and 30 s for WAN / edge tail latency before HTTP timeouts
pub const BATCH_UPLOAD_LATENCY_BUCKETS: &[f64] = &[
    0.005, // 5 ms
    0.01,  // 10 ms
    0.025, // 25 ms
    0.05,  // 50 ms
    0.1,   // 100 ms
    0.25,  // 250 ms
    0.5,   // 500 ms
    1.0,   // 1 s
    2.5,   // 2.5 s
    5.0,   // 5 s
    10.0,  // 10 s
    15.0,  // 15 s
    30.0,  // 30 s
];

pub const BUCKET_COUNT: usize = 13;

/// Data plane worker metrics for Tessella (`TASK-94`).
///
/// Thread-safe and lock-free using atomic primitives. Adheres to Prometheus exposition
/// format specifications with zero cardinality inflation.
#[derive(Debug)]
pub struct WorkerMetrics {
    /// Total work proofs saved to local SQLite WAL.
    pub proofs_queued: AtomicU64,
    /// Total work proofs successfully uploaded and settled with orchestrator.
    pub proofs_settled: AtomicU64,
    /// Total batch upload attempts that succeeded (HTTP 2xx).
    pub batch_uploads_success: AtomicU64,
    /// Total batch upload attempts that failed (HTTP error or network failure).
    pub batch_uploads_failure: AtomicU64,

    // Histogram for batch upload latency
    pub batch_upload_latency_buckets: [AtomicU64; BUCKET_COUNT],
    pub batch_upload_latency_inf: AtomicU64,
    pub batch_upload_latency_sum_micros: AtomicU64,
    pub batch_upload_latency_count: AtomicU64,
}

impl Default for WorkerMetrics {
    fn default() -> Self {
        Self {
            proofs_queued: AtomicU64::new(0),
            proofs_settled: AtomicU64::new(0),
            batch_uploads_success: AtomicU64::new(0),
            batch_uploads_failure: AtomicU64::new(0),
            batch_upload_latency_buckets: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            batch_upload_latency_inf: AtomicU64::new(0),
            batch_upload_latency_sum_micros: AtomicU64::new(0),
            batch_upload_latency_count: AtomicU64::new(0),
        }
    }
}

impl WorkerMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn inc_proofs_queued(&self, count: u64) {
        self.proofs_queued.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_proofs_settled(&self, count: u64) {
        self.proofs_settled.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_batch_uploads_success(&self) {
        self.batch_uploads_success.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_batch_uploads_failure(&self) {
        self.batch_uploads_failure.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_batch_upload_duration(&self, duration_secs: f64) {
        if duration_secs < 0.0 || duration_secs.is_nan() {
            return;
        }

        let micros = (duration_secs * 1_000_000.0).round() as u64;
        self.batch_upload_latency_sum_micros
            .fetch_add(micros, Ordering::Relaxed);
        self.batch_upload_latency_count
            .fetch_add(1, Ordering::Relaxed);

        // In Prometheus histograms, every bucket whose upper bound is >= duration gets incremented
        for (i, &bound) in BATCH_UPLOAD_LATENCY_BUCKETS.iter().enumerate() {
            if duration_secs <= bound {
                self.batch_upload_latency_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.batch_upload_latency_inf
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Renders current metrics into Prometheus text format (v0.0.4).
    ///
    /// `wal_depth` is the current physical count of pending work proofs in SQLite,
    /// passed in by the caller so crash-recovery / cold-boot state is accurately reflected.
    pub fn render_prometheus(&self, wal_depth: u64) -> String {
        let mut out = String::with_capacity(1024);

        // 1. WAL pending proofs depth (gauge)
        out.push_str("# HELP tessella_wal_pending_proofs_depth Number of pending work proofs in SQLite WAL.\n");
        out.push_str("# TYPE tessella_wal_pending_proofs_depth gauge\n");
        let _ = writeln!(out, "tessella_wal_pending_proofs_depth {wal_depth}");

        // 2. Proofs queued total (counter)
        out.push_str("# HELP tessella_proofs_queued_total Total work proofs queued to WAL.\n");
        out.push_str("# TYPE tessella_proofs_queued_total counter\n");
        let _ = writeln!(
            out,
            "tessella_proofs_queued_total {}",
            self.proofs_queued.load(Ordering::Relaxed)
        );

        // 3. Proofs settled total (counter)
        out.push_str("# HELP tessella_proofs_settled_total Total work proofs successfully settled with orchestrator.\n");
        out.push_str("# TYPE tessella_proofs_settled_total counter\n");
        let _ = writeln!(
            out,
            "tessella_proofs_settled_total {}",
            self.proofs_settled.load(Ordering::Relaxed)
        );

        // 4. Batch uploads processed by status (counter)
        out.push_str("# HELP tessella_proof_batch_uploads_total Total batch upload attempts to orchestrator by status.\n");
        out.push_str("# TYPE tessella_proof_batch_uploads_total counter\n");
        let _ = writeln!(
            out,
            "tessella_proof_batch_uploads_total{{status=\"success\"}} {}",
            self.batch_uploads_success.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "tessella_proof_batch_uploads_total{{status=\"failure\"}} {}",
            self.batch_uploads_failure.load(Ordering::Relaxed)
        );

        // 5. Batch upload latency histogram
        out.push_str("# HELP tessella_proof_batch_upload_duration_seconds Latency of proof batch uploads to orchestrator in seconds.\n");
        out.push_str("# TYPE tessella_proof_batch_upload_duration_seconds histogram\n");
        for (i, &bound) in BATCH_UPLOAD_LATENCY_BUCKETS.iter().enumerate() {
            let count = self.batch_upload_latency_buckets[i].load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "tessella_proof_batch_upload_duration_seconds_bucket{{le=\"{bound}\"}} {count}"
            );
        }
        let inf_count = self.batch_upload_latency_inf.load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "tessella_proof_batch_upload_duration_seconds_bucket{{le=\"+Inf\"}} {inf_count}"
        );
        let sum_secs =
            self.batch_upload_latency_sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let _ = writeln!(
            out,
            "tessella_proof_batch_upload_duration_seconds_sum {sum_secs:.6}"
        );
        let total_count = self.batch_upload_latency_count.load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "tessella_proof_batch_upload_duration_seconds_count {total_count}"
        );

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_metrics_render_valid_zeros() {
        let metrics = WorkerMetrics::new();
        let rendered = metrics.render_prometheus(0);

        assert!(rendered.contains("tessella_wal_pending_proofs_depth 0"));
        assert!(rendered.contains("tessella_proofs_queued_total 0"));
        assert!(rendered.contains("tessella_proofs_settled_total 0"));
        assert!(rendered.contains("tessella_proof_batch_uploads_total{status=\"success\"} 0"));
        assert!(rendered.contains("tessella_proof_batch_uploads_total{status=\"failure\"} 0"));
        assert!(
            rendered
                .contains("tessella_proof_batch_upload_duration_seconds_bucket{le=\"0.005\"} 0")
        );
        assert!(
            rendered.contains("tessella_proof_batch_upload_duration_seconds_bucket{le=\"30\"} 0")
        );
        assert!(
            rendered.contains("tessella_proof_batch_upload_duration_seconds_bucket{le=\"+Inf\"} 0")
        );
        assert!(rendered.contains("tessella_proof_batch_upload_duration_seconds_count 0"));
        assert!(rendered.contains("tessella_proof_batch_upload_duration_seconds_sum 0.000000"));
    }

    #[test]
    fn metrics_increment_and_histogram_placement() {
        let metrics = WorkerMetrics::new();

        metrics.inc_proofs_queued(10);
        metrics.inc_proofs_settled(7);
        metrics.inc_batch_uploads_success();
        metrics.inc_batch_uploads_failure();

        // 8ms (0.008s): > 5ms (bucket 0), <= 10ms (bucket 1..12 + Inf)
        metrics.record_batch_upload_duration(0.008);

        let rendered = metrics.render_prometheus(3);
        assert!(rendered.contains("tessella_wal_pending_proofs_depth 3"));
        assert!(rendered.contains("tessella_proofs_queued_total 10"));
        assert!(rendered.contains("tessella_proofs_settled_total 7"));
        assert!(rendered.contains("tessella_proof_batch_uploads_total{status=\"success\"} 1"));
        assert!(rendered.contains("tessella_proof_batch_uploads_total{status=\"failure\"} 1"));

        assert!(
            rendered
                .contains("tessella_proof_batch_upload_duration_seconds_bucket{le=\"0.005\"} 0")
        );
        assert!(
            rendered.contains("tessella_proof_batch_upload_duration_seconds_bucket{le=\"0.01\"} 1")
        );
        assert!(
            rendered
                .contains("tessella_proof_batch_upload_duration_seconds_bucket{le=\"0.025\"} 1")
        );
        assert!(
            rendered.contains("tessella_proof_batch_upload_duration_seconds_bucket{le=\"+Inf\"} 1")
        );
        assert!(rendered.contains("tessella_proof_batch_upload_duration_seconds_count 1"));
        assert!(rendered.contains("tessella_proof_batch_upload_duration_seconds_sum 0.008000"));
    }
}
