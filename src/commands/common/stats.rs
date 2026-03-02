use std::time::{Duration, Instant};

use super::parser::ParserStats;

// Per-dispatch timing buckets used to separate host prep/encode overhead from
// time spent waiting for GPU completion.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BatchDispatchTimings {
    pub(crate) host_prep: Duration,
    pub(crate) command_encode: Duration,
    pub(crate) gpu_wait: Duration,
    pub(crate) result_readback: Duration,
    pub(crate) total: Duration,
}

// Aggregated run timings and batch stats printed in the final report.
#[derive(Debug, Default)]
pub(crate) struct RunTimings {
    pub(crate) wordlist_batch_build: Duration,
    pub(crate) wordlist_batch_plan: Duration,
    pub(crate) wordlist_batch_pack: Duration,
    pub(crate) host_prep: Duration,
    pub(crate) command_encode: Duration,
    pub(crate) gpu_wait: Duration,
    pub(crate) result_readback: Duration,
    pub(crate) dispatch_total: Duration,
    pub(crate) consumer_idle_wait: Duration,
    pub(crate) batch_count: u64,
    pub(crate) total_batch_candidates: u64,
    pub(crate) total_batch_word_bytes: u64,
    pub(crate) pipeline_depth: usize,
    pub(crate) packer_threads: usize,
    pub(crate) parser_threads: usize,
    pub(crate) parser_chunk_bytes: usize,
    pub(crate) parser_chunks: u64,
    pub(crate) parser_skipped_oversize: u64,
}

impl RunTimings {
    // Copy the latest parser-side counters/config into the reporting snapshot.
    //
    // The producer includes this with each message so the consumer can print a
    // complete final report even if the run ends early (match found) or at EOF.
    pub(crate) fn apply_parser_stats(&mut self, parser_stats: ParserStats) {
        self.parser_threads = parser_stats.parser_threads;
        self.parser_chunk_bytes = parser_stats.parser_chunk_bytes;
        self.parser_chunks = parser_stats.parser_chunks;
        self.parser_skipped_oversize = parser_stats.parser_skipped_oversize;
    }
}

// Snapshot of cumulative counters/timings at a periodic progress report.
// Deltas between snapshots produce interval ("windowed") rates and wait times.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RateReportSnapshot {
    pub(crate) reported_at: Instant,
    pub(crate) candidates_tested: u64,
    pub(crate) gpu_wait: Duration,
    pub(crate) wordlist_batch_build: Duration,
    pub(crate) consumer_idle_wait: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RateReportDelta {
    pub(crate) wall_time: Duration,
    pub(crate) candidates_tested: u64,
    pub(crate) gpu_wait: Duration,
    pub(crate) wordlist_batch_build: Duration,
    pub(crate) consumer_idle_wait: Duration,
}

impl RateReportSnapshot {
    pub(crate) fn capture(
        reported_at: Instant,
        candidates_tested: u64,
        timings: &RunTimings,
    ) -> Self {
        // Capture cumulative totals at one point in time; periodic logging uses
        // `delta_since` to turn two cumulative snapshots into interval metrics.
        Self {
            reported_at,
            candidates_tested,
            gpu_wait: timings.gpu_wait,
            wordlist_batch_build: timings.wordlist_batch_build,
            consumer_idle_wait: timings.consumer_idle_wait,
        }
    }

    pub(crate) fn delta_since(self, previous: Self) -> RateReportDelta {
        // Subtractions are saturating/checked so clock regressions or future
        // refactors cannot panic the reporter path.
        RateReportDelta {
            wall_time: self
                .reported_at
                .checked_duration_since(previous.reported_at)
                .unwrap_or_default(),
            candidates_tested: self
                .candidates_tested
                .saturating_sub(previous.candidates_tested),
            gpu_wait: saturating_duration_sub(self.gpu_wait, previous.gpu_wait),
            wordlist_batch_build: saturating_duration_sub(
                self.wordlist_batch_build,
                previous.wordlist_batch_build,
            ),
            consumer_idle_wait: saturating_duration_sub(
                self.consumer_idle_wait,
                previous.consumer_idle_wait,
            ),
        }
    }
}

// Helper used by both periodic and final reporting. Returns 0 when no time has
// elapsed yet to avoid division-by-zero during startup.
pub(crate) fn saturating_duration_sub(current: Duration, previous: Duration) -> Duration {
    // Defensive helper for "delta from cumulative totals" math used by the
    // progress reporter. Underflow should not happen in normal execution, but
    // returning zero keeps tests and future refactors safe.
    current.checked_sub(previous).unwrap_or_default()
}

pub(crate) fn rate_per_second(candidates_tested: u64, elapsed: Duration) -> f64 {
    // Used for both cumulative (`STATS`) and interval (`RATE`) throughput.
    if elapsed.is_zero() {
        0.0
    } else {
        candidates_tested as f64 / elapsed.as_secs_f64()
    }
}

pub(crate) fn format_duration_millis(duration: Duration) -> String {
    // Keep progress logs compact and stable: milliseconds are easy to compare
    // across runs when diagnosing host-side stalls.
    format!("{:.1}", duration.as_secs_f64() * 1_000.0)
}

// Render large counts/rates in a stable human-readable format so benchmark logs
// remain easy to scan while still preserving approximate magnitude.
pub(crate) fn format_human_count(value: f64) -> String {
    const UNITS: [(&str, f64); 4] = [
        ("trillion", 1_000_000_000_000.0),
        ("billion", 1_000_000_000.0),
        ("million", 1_000_000.0),
        ("thousand", 1_000.0),
    ];

    for (name, scale) in UNITS {
        if value >= scale {
            let scaled = value / scale;
            let precision = if scaled >= 100.0 {
                0
            } else if scaled >= 10.0 {
                1
            } else {
                2
            };
            return format!("{scaled:.precision$} {name}");
        }
    }

    format!("{value:.0}")
}

// Print the final timing breakdown and batch statistics. This intentionally
// separates end-to-end throughput from GPU-only throughput to highlight host
// bottlenecks (parsing, prep, synchronization).
pub(crate) fn print_final_stats(
    candidates_tested: u64,
    elapsed: Duration,
    rate_end_to_end: f64,
    rate_gpu_only: f64,
    timings: &RunTimings,
) {
    let avg_candidates_per_batch = if timings.batch_count == 0 {
        0.0
    } else {
        timings.total_batch_candidates as f64 / timings.batch_count as f64
    };
    let avg_word_bytes_per_batch = if timings.batch_count == 0 {
        0.0
    } else {
        timings.total_batch_word_bytes as f64 / timings.batch_count as f64
    };

    eprintln!("STATS");
    eprintln!("  tested: {}", format_human_count(candidates_tested as f64));
    eprintln!("  elapsed: {:.2}s", elapsed.as_secs_f64());
    eprintln!(
        "  rate_end_to_end: {}/s",
        format_human_count(rate_end_to_end)
    );
    eprintln!("  rate_gpu_only: {}/s", format_human_count(rate_gpu_only));
    eprintln!("  batches: {}", timings.batch_count);
    eprintln!(
        "  avg_candidates_per_batch: {}",
        format_human_count(avg_candidates_per_batch)
    );
    eprintln!(
        "  avg_word_bytes_per_batch: {} bytes",
        format_human_count(avg_word_bytes_per_batch)
    );
    eprintln!("  pipeline_depth: {}", timings.pipeline_depth);
    eprintln!("  packer_threads: {}", timings.packer_threads);
    eprintln!("  parser_threads: {}", timings.parser_threads);
    eprintln!(
        "  parser_chunk_bytes: {} bytes",
        format_human_count(timings.parser_chunk_bytes as f64)
    );
    eprintln!("  parser_chunks: {}", timings.parser_chunks);
    eprintln!(
        "  parser_skipped_oversize: {}",
        timings.parser_skipped_oversize
    );
    eprintln!(
        "  timing.wordlist_batch_build: {:.3}s",
        timings.wordlist_batch_build.as_secs_f64()
    );
    eprintln!(
        "  timing.wordlist_batch_plan: {:.3}s",
        timings.wordlist_batch_plan.as_secs_f64()
    );
    eprintln!(
        "  timing.wordlist_batch_pack: {:.3}s",
        timings.wordlist_batch_pack.as_secs_f64()
    );
    eprintln!(
        "  timing.consumer_idle_wait: {:.3}s",
        timings.consumer_idle_wait.as_secs_f64()
    );
    eprintln!(
        "  timing.host_prep: {:.3}s",
        timings.host_prep.as_secs_f64()
    );
    eprintln!(
        "  timing.command_encode: {:.3}s",
        timings.command_encode.as_secs_f64()
    );
    eprintln!("  timing.gpu_wait: {:.3}s", timings.gpu_wait.as_secs_f64());
    eprintln!(
        "  timing.result_readback: {:.3}s",
        timings.result_readback.as_secs_f64()
    );
    eprintln!(
        "  timing.dispatch_total: {:.3}s",
        timings.dispatch_total.as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_report_snapshot_delta_computes_window_metrics() {
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_secs(2);
        let previous = RateReportSnapshot {
            reported_at: t0,
            candidates_tested: 100,
            gpu_wait: Duration::from_millis(400),
            wordlist_batch_build: Duration::from_millis(50),
            consumer_idle_wait: Duration::from_millis(10),
        };
        let current = RateReportSnapshot {
            reported_at: t1,
            candidates_tested: 300,
            gpu_wait: Duration::from_millis(900),
            wordlist_batch_build: Duration::from_millis(90),
            consumer_idle_wait: Duration::from_millis(25),
        };

        let delta = current.delta_since(previous);
        assert_eq!(delta.wall_time, Duration::from_secs(2));
        assert_eq!(delta.candidates_tested, 200);
        assert_eq!(delta.gpu_wait, Duration::from_millis(500));
        assert_eq!(delta.wordlist_batch_build, Duration::from_millis(40));
        assert_eq!(delta.consumer_idle_wait, Duration::from_millis(15));

        let end_to_end_rate = rate_per_second(delta.candidates_tested, delta.wall_time);
        let gpu_only_rate = rate_per_second(delta.candidates_tested, delta.gpu_wait);
        assert!((end_to_end_rate - 100.0).abs() < 0.000_001);
        assert!((gpu_only_rate - 400.0).abs() < 0.000_001);
    }

    #[test]
    fn rate_per_second_returns_zero_for_zero_elapsed() {
        assert_eq!(rate_per_second(123, Duration::ZERO), 0.0);
    }

    #[test]
    fn saturating_duration_sub_returns_zero_on_underflow() {
        assert_eq!(
            saturating_duration_sub(Duration::from_millis(10), Duration::from_millis(25)),
            Duration::ZERO
        );
    }
}
