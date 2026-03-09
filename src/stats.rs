//! # Timing, rate calculation, and final statistics reporting
//!
//! GPU programs spend time in many places: reading the wordlist, packing
//! batches, encoding Metal commands, waiting for the GPU, and reading back
//! results. This module provides structs to track each phase separately so
//! you can pinpoint bottlenecks.
//!
//! ## Windowed vs. cumulative reporting
//!
//! - **Cumulative** — total counts/durations since the run started. Useful for
//!   the final summary (e.g., "tested 500 million keys in 1.12 seconds").
//! - **Windowed (interval)** — the delta between two cumulative snapshots taken
//!   a few seconds apart. This gives you a recent throughput rate that reflects
//!   current conditions (e.g., if the wordlist switches from short to long keys
//!   mid-run, the window rate will drop immediately, while the cumulative rate
//!   responds slowly).
//!
//! [`RateReportSnapshot`] captures cumulative totals at a point in time.
//! [`RateReportSnapshot::delta_since()`] computes the window by subtracting
//! a previous snapshot.

use std::time::{Duration, Instant};

use super::parser::ParserStats;

/// Per-dispatch timing buckets used to separate host prep/encode overhead from
/// time spent waiting for GPU completion.
///
/// Each GPU dispatch (one batch of candidate passwords) goes through four phases:
///
/// 1. **`host_prep`** — CPU-side work: copying passwords into Metal buffers,
///    setting up shader arguments.
/// 2. **`command_encode`** — Recording Metal commands into a command buffer.
///    This is CPU work but involves Metal API calls.
/// 3. **`gpu_wait`** — Blocking until the GPU finishes the compute kernel.
///    If this dominates, the GPU is the bottleneck (good — it means the CPU
///    is keeping up).
/// 4. **`result_readback`** — Reading the match result from the GPU buffer back
///    to CPU memory.
///
/// ## Why `#[derive(Default)]`?
///
/// `Default` gives us `BatchDispatchTimings::default()` which zero-initializes
/// all `Duration` fields. This is handy for accumulators — you start with
/// `Default::default()` and add to each field as dispatches complete.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BatchDispatchTimings {
    pub(crate) host_prep: Duration,
    pub(crate) command_encode: Duration,
    pub(crate) gpu_wait: Duration,
    pub(crate) result_readback: Duration,
    pub(crate) total: Duration,
}

/// Aggregated run timings and batch stats printed in the final report.
///
/// This is the "accumulator" struct for the entire cracking run. The consumer
/// thread adds each batch's timings to the running totals here. At the end of
/// the run (or when a match is found), `print_final_stats()` reads these
/// cumulative values to produce the summary.
///
/// ## Why not `Clone + Copy`?
///
/// Unlike `BatchDispatchTimings`, this struct is larger and is only ever owned
/// by the consumer thread. There is no need to copy it, so we skip `Copy` to
/// avoid accidentally duplicating a large struct on the stack.
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
    pub(crate) parser_threads: usize,
    pub(crate) parser_chunk_bytes: usize,
    pub(crate) parser_chunks: u64,
    pub(crate) parser_skipped_oversize: u64,
}

impl RunTimings {
    /// Copy the latest parser-side counters/config into the reporting snapshot.
    ///
    /// The producer includes this with each message so the consumer can print a
    /// complete final report even if the run ends early (match found) or at EOF.
    ///
    /// ## Why `&mut self`?
    ///
    /// This method mutates the struct in place. In Rust, `&mut self` is an
    /// exclusive (mutable) borrow — only one `&mut` reference can exist at a
    /// time. This guarantee (enforced at compile time by the borrow checker)
    /// means we never have data races on `RunTimings`, even though multiple
    /// threads exist in the pipeline.
    pub(crate) fn apply_parser_stats(&mut self, parser_stats: ParserStats) {
        self.parser_threads = parser_stats.parser_threads;
        self.parser_chunk_bytes = parser_stats.parser_chunk_bytes;
        self.parser_chunks = parser_stats.parser_chunks;
        self.parser_skipped_oversize = parser_stats.parser_skipped_oversize;
    }
}

/// Snapshot of cumulative counters/timings at a periodic progress report.
///
/// This is the "windowed reporting" pattern: instead of maintaining a separate
/// rolling-window buffer, we take a snapshot of cumulative totals every N
/// seconds. The delta between two consecutive snapshots gives the interval
/// metrics. This is simpler and uses O(1) memory regardless of run duration.
///
/// `Clone + Copy` because snapshots are small and get stored/compared by value.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RateReportSnapshot {
    pub(crate) reported_at: Instant,
    pub(crate) candidates_tested: u64,
    pub(crate) gpu_wait: Duration,
    pub(crate) wordlist_batch_build: Duration,
    pub(crate) consumer_idle_wait: Duration,
}

/// The difference between two [`RateReportSnapshot`]s — represents one
/// reporting interval (e.g., the last 2 seconds of the run).
///
/// `Default` zero-initializes all fields, which is the identity element for
/// addition — useful when there is no previous snapshot yet.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RateReportDelta {
    pub(crate) wall_time: Duration,
    pub(crate) candidates_tested: u64,
    pub(crate) gpu_wait: Duration,
    pub(crate) wordlist_batch_build: Duration,
    pub(crate) consumer_idle_wait: Duration,
}

impl RateReportSnapshot {
    /// Take a cumulative snapshot at the given instant.
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

    /// Compute the interval delta by subtracting a previous snapshot.
    ///
    /// ## Defensive arithmetic
    ///
    /// All subtractions here use `checked_*` or `saturating_*` variants. Why?
    /// - `Instant::checked_duration_since()` returns `None` if `previous` is
    ///   *after* `self` (can happen with clock adjustments on some platforms).
    /// - `u64::saturating_sub()` returns 0 instead of panicking on underflow.
    /// - `saturating_duration_sub()` (our helper) does the same for `Duration`.
    ///
    /// The reporter path should never crash — even with weird timing edge cases.
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

/// Subtract two `Duration`s, clamping to zero on underflow.
///
/// `Duration` is unsigned, so `checked_sub` returns `None` if `previous > current`.
/// We convert that to `Duration::ZERO` via `unwrap_or_default()`.
///
/// This pattern — "return a safe default instead of panicking" — is common in
/// reporting/logging code where correctness of the metric is less important than
/// not crashing the program.
pub(crate) fn saturating_duration_sub(current: Duration, previous: Duration) -> Duration {
    // Defensive helper for "delta from cumulative totals" math used by the
    // progress reporter. Underflow should not happen in normal execution, but
    // returning zero keeps tests and future refactors safe.
    current.checked_sub(previous).unwrap_or_default()
}

/// Compute throughput as candidates per second.
///
/// Returns `0.0` when `elapsed` is zero to avoid division by zero. This guard
/// fires during startup (before the first batch completes) and in tests with
/// synthetic zero-duration intervals.
///
/// ## The `as f64` cast
///
/// `candidates_tested as f64` is a widening conversion — `u64` has 64 bits of
/// integer precision, but `f64` only has 53 bits of mantissa. For values above
/// 2^53 (~9 quadrillion) you would lose precision, but that is far beyond any
/// realistic candidate count for this tool.
pub(crate) fn rate_per_second(candidates_tested: u64, elapsed: Duration) -> f64 {
    // Used for both cumulative (`STATS`) and interval (`RATE`) throughput.
    if elapsed.is_zero() {
        0.0
    } else {
        candidates_tested as f64 / elapsed.as_secs_f64()
    }
}

/// Format a `Duration` as milliseconds with one decimal place (e.g., `"12.3"`).
pub(crate) fn format_duration_millis(duration: Duration) -> String {
    // Keep progress logs compact and stable: milliseconds are easy to compare
    // across runs when diagnosing host-side stalls.
    format!("{:.1}", duration.as_secs_f64() * 1_000.0)
}

/// Format a duration as a compact human-readable ETA string.
///
/// Examples: "0s", "42s", "3m 12s", "1h 5m", "2d 3h".
pub(crate) fn format_eta(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    if total_secs < 60 {
        return format!("{total_secs}s");
    }
    let minutes = total_secs / 60;
    let secs = total_secs % 60;
    if minutes < 60 {
        return format!("{minutes}m {secs}s");
    }
    let hours = minutes / 60;
    let mins = minutes % 60;
    if hours < 24 {
        return format!("{hours}h {mins}m");
    }
    let days = hours / 24;
    let hrs = hours % 24;
    format!("{days}d {hrs}h")
}

/// Render large counts/rates in a stable human-readable format so benchmark logs
/// remain easy to scan while still preserving approximate magnitude.
///
/// ## Rust pattern: const array of tuples
///
/// `UNITS` is a `const` array of `(&str, f64)` tuples. `const` means the array
/// is computed at compile time and embedded in the binary — no heap allocation
/// at runtime. The `&str` references point to string literals in the binary's
/// read-only data segment, so they have `'static` lifetime.
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

/// Print the final timing breakdown and batch statistics.
///
/// This intentionally separates end-to-end throughput from GPU-only throughput
/// to highlight host bottlenecks (parsing, prep, synchronization). If
/// `rate_gpu_only` is much higher than `rate_end_to_end`, the CPU side is the
/// bottleneck — you might benefit from more parser threads or a deeper pipeline.
///
/// All output goes to `stderr` (via `eprintln!`) so that `stdout` can be piped
/// for the key result without stats noise.
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

    // This test verifies the "windowed reporting" math: given two cumulative
    // snapshots 2 seconds apart, the delta should reflect only the interval.
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

        // Compute both rate views from the same delta:
        // - end_to_end: 200 candidates / 2s wall = 100/s
        // - gpu_only:   200 candidates / 0.5s GPU = 400/s
        // The difference tells you how much time is spent outside the GPU.
        let end_to_end_rate = rate_per_second(delta.candidates_tested, delta.wall_time);
        let gpu_only_rate = rate_per_second(delta.candidates_tested, delta.gpu_wait);
        // Floating-point comparison: use an epsilon (small tolerance) because
        // f64 arithmetic is not exact. Never use `==` with floats.
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

    #[test]
    fn format_eta_covers_all_ranges() {
        assert_eq!(format_eta(Duration::from_secs(0)), "0s");
        assert_eq!(format_eta(Duration::from_secs(42)), "42s");
        assert_eq!(format_eta(Duration::from_secs(192)), "3m 12s");
        assert_eq!(format_eta(Duration::from_secs(3900)), "1h 5m");
        assert_eq!(format_eta(Duration::from_secs(97200)), "1d 3h");
    }
}
