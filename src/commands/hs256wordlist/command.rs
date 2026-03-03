//! The main dispatch loop for HS256 wordlist cracking.
//!
//! # Double-buffered producer/consumer pipeline
//!
//! This module implements the core throughput optimization: **double-buffered
//! dispatch**. The idea is to keep both the CPU and GPU busy simultaneously
//! by overlapping their work.
//!
//! ## The problem with naive serial dispatch
//!
//! A naive approach would be:
//! ```text
//! loop {
//!     batch = parse_next_batch();     // CPU busy, GPU idle
//!     result = gpu.dispatch(batch);   // GPU busy, CPU idle
//!     check(result);
//! }
//! ```
//! Here, the CPU and GPU take turns -- each waits for the other. If parsing
//! takes 5ms and GPU takes 10ms, total per-batch time is 15ms.
//!
//! ## The double-buffered solution
//!
//! Instead, we overlap CPU and GPU work:
//! ```text
//! loop {
//!     batch_N+1 = recv_next_batch();          // CPU works while GPU runs batch N
//!     result_N = gpu.wait_for_batch_N();       // GPU may already be done!
//!     gpu.commit(batch_N+1);                   // GPU starts batch N+1 immediately
//! }
//! ```
//! Now the CPU parses batch N+1 while the GPU crunches batch N. If parsing
//! (5ms) is faster than GPU (10ms), the batch is ready before the GPU
//! finishes -- zero idle time on the CPU. Per-batch time drops to max(5ms,
//! 10ms) = 10ms instead of 15ms.
//!
//! ## The `in_flight` pattern
//!
//! We track the currently-executing GPU batch in `in_flight: Option<InFlightBatch>`.
//! - `None` = no GPU work in progress (first iteration or just after drain).
//! - `Some(flight)` = GPU is processing this batch; we need to wait for it
//!   before reusing shared buffers.
//!
//! The loop order is deliberately: recv -> wait -> encode+commit. This
//! maximizes the overlap window: `recv` (which blocks on the producer channel)
//! runs concurrently with the GPU executing the in-flight batch.
//!
//! ## The `producer.recycle()` pattern (buffer reuse)
//!
//! Allocating Metal shared-memory buffers is expensive. Instead of allocating
//! new buffers for every batch, we recycle them: after the GPU finishes with
//! a batch's buffers, we send the `WordBatch` back to the producer via
//! `producer.recycle()`. The producer reuses these pre-allocated buffers for
//! the next batch. This is a bounded pool pattern:
//!
//! ```text
//! Producer  --(filled batch)--> Consumer (GPU dispatch)
//!    ^                              |
//!    |                              v
//!    +------(empty batch)----------+
//!           (recycle)
//! ```
//!
//! `pipeline_depth` controls how many batches can be in flight between the
//! producer and consumer. More depth = more memory usage but better overlap
//! for bursty workloads.

use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};

use super::args::{DEFAULT_PIPELINE_DEPTH, Hs256WordlistArgs};
use crate::common::gpu::{GpuBruteForcer, GpuCommandHandle, HmacVariant, metal::MetalBruteForcer};
use super::jwt::parse_hs256_jwt;
use crate::common::batch::APPROX_WORD_BATCH_BUFFER_BYTES;
use crate::common::producer::{ProducerMessage, WordlistProducer};
use crate::common::stats::{
    RateReportSnapshot, RunTimings, format_duration_millis, format_human_count, print_final_stats,
    rate_per_second,
};

/// Holds the command buffer + batch for a currently executing GPU dispatch.
///
/// This struct bundles everything needed to wait for and interpret a GPU result:
/// - `cmd_buf`: the Metal command buffer to wait on.
/// - `batch`: the `WordBatch` containing the candidate words (needed to look
///   up the matching word by index after the GPU returns a match).
/// - `candidate_count`: cached count for statistics.
///
/// Defined at module scope so `handle_match` can reference it.
struct InFlightBatch {
    cmd_buf: GpuCommandHandle,
    batch: crate::common::batch::WordBatch,
    candidate_count: u64,
}

/// Extract the matched secret from a completed GPU batch, print final stats,
/// and return `Ok(true)`.
///
/// Called from both the main loop (mid-stream match) and the post-EOF drain
/// (match in the last batch). Having this as a separate function avoids
/// duplicating the match-handling logic in two places.
///
/// The GPU returns a batch-local index (e.g., "candidate #42 in this batch
/// matched"). We use `word_string_lossy()` to look up the actual password
/// string at that index in the batch's packed word data.
fn handle_match(
    flight: &InFlightBatch,
    local_match_index: u32,
    candidates_tested: u64,
    started_at: Instant,
    timings: &RunTimings,
) -> anyhow::Result<bool> {
    let elapsed = started_at.elapsed();
    let rate_end_to_end = rate_per_second(candidates_tested, elapsed);
    let rate_gpu_only = rate_per_second(candidates_tested, timings.gpu_wait);
    print_final_stats(candidates_tested, elapsed, rate_end_to_end, rate_gpu_only, timings);
    let _global_index = flight
        .batch
        .candidate_index_base
        .checked_add(local_match_index as u64)
        .ok_or_else(|| anyhow!("candidate index overflow while reconstructing result"))?;
    let local_index = usize::try_from(local_match_index)
        .context("GPU returned invalid result index")?;
    let secret = flight
        .batch
        .word_string_lossy(local_index)
        .ok_or_else(|| anyhow!("GPU returned invalid local candidate index"))?;
    println!("HS256 key: {secret}");
    Ok(true)
}

/// End-to-end HS256 cracking flow:
/// 1) parse the JWT,
/// 2) initialize Metal + reusable buffers,
/// 3) overlap wordlist parsing with GPU dispatch,
/// 4) report the first matching secret (or NOT FOUND).
///
/// Returns `Ok(true)` if the key was found, `Ok(false)` if the entire wordlist
/// was exhausted without a match, or `Err(...)` on any fatal error.
pub fn run(args: Hs256WordlistArgs) -> anyhow::Result<bool> {
    // ---- Phase 1: Parse and validate the JWT ----
    // This extracts the signing input (header.payload as bytes) and the
    // 32-byte target HMAC-SHA256 signature we are trying to match.
    let (signing_input, target_signature) = parse_hs256_jwt(&args.jwt)?;

    // ---- Phase 2: Initialize Metal GPU runtime ----
    // Compiles shaders, creates pipeline states, allocates shared buffers.
    let mut gpu = MetalBruteForcer::new(HmacVariant::Hs256, &signing_input)?;

    // ---- Phase 3: Resolve configuration parameters ----
    let parser_config = args.parser_config();
    let pipeline_depth = args.pipeline_depth.unwrap_or(DEFAULT_PIPELINE_DEPTH);
    // Clamp packer_threads to [1, pipeline_depth]. Having more packers than
    // pipeline slots would just cause contention with no throughput benefit.
    let packer_threads = args
        .packer_threads
        .unwrap_or(2)
        .max(1)
        .min(pipeline_depth.max(1));
    if let Some(tpg) = args.threads_per_group {
        gpu.set_threadgroup_width(tpg)?;
    }
    let approx_prefetch_bytes = APPROX_WORD_BATCH_BUFFER_BYTES.saturating_mul(pipeline_depth);
    // Print diagnostic info to stderr (not stdout, so it does not interfere
    // with machine-readable output like "HS256 key: ...").
    eprintln!(
        "GPU device={} tew={} max_tpg={} selected_tpg={}",
        gpu.device_name(),
        gpu.thread_execution_width(),
        gpu.max_total_threads_per_threadgroup(),
        gpu.current_threadgroup_width()
    );
    eprintln!(
        "PIPELINE depth={} approx_batch_bytes={} bytes approx_prefetch_bytes={} bytes",
        pipeline_depth,
        format_human_count(APPROX_WORD_BATCH_BUFFER_BYTES as f64),
        format_human_count(approx_prefetch_bytes as f64),
    );
    eprintln!(
        "HOST parser_threads={} packer_threads={}",
        parser_config.parser_threads, packer_threads
    );

    // ---- Phase 4: Start the producer (wordlist parser) ----
    // The producer runs on a separate thread, reading the wordlist via mmap
    // and packing candidates into GPU-ready WordBatch buffers. We start it
    // early so it can begin filling the pipeline while we set up the GPU.
    // The `gpu.device.clone()` passes the Metal device handle so the producer
    // can allocate shared-memory buffers directly (avoiding copies later).
    let mut producer = WordlistProducer::spawn(
        args.wordlist.clone(),
        gpu.device().clone(),
        parser_config,
        pipeline_depth,
        packer_threads,
    );
    // ---- Phase 5: The main dispatch loop ----
    //
    // The closure `(|| { ... })()` is an immediately-invoked closure, a Rust
    // pattern that lets us use `?` (early return on error) inside the loop
    // while still running cleanup code (producer.stop/join) afterward. Without
    // the closure, a `?` would return from the entire `run()` function and
    // skip cleanup. This is similar to try/finally in other languages.
    let run_result = (|| -> anyhow::Result<bool> {
        let started_at = Instant::now();
        let mut candidates_tested: u64 = 0;
        // If autotune was not requested, mark it as already done so we skip it.
        let mut autotune_done = !args.autotune;
        let mut timings = RunTimings {
            pipeline_depth,
            packer_threads,
            parser_threads: parser_config.parser_threads,
            parser_chunk_bytes: parser_config.chunk_bytes,
            ..RunTimings::default()
        };
        let mut last_rate_report =
            RateReportSnapshot::capture(started_at, candidates_tested, &timings);

        // ============================================================
        // DOUBLE-BUFFERED DISPATCH STATE
        // ============================================================
        // `in_flight` holds the currently-executing GPU batch. This is the
        // heart of the double-buffering pattern:
        //
        // - `None` on the first iteration (no GPU work submitted yet).
        // - `Some(flight)` on subsequent iterations (GPU is crunching this
        //   batch while we recv the next one from the producer).
        //
        // The Option<T> type is Rust's way of expressing "this value may or
        // may not be present" -- like a nullable pointer, but the compiler
        // forces you to handle both cases. `.take()` extracts the value and
        // leaves `None` in its place (useful for one-shot consumption).
        let mut in_flight: Option<InFlightBatch> = None;

        loop {
            // ============================================================
            // Step A: Receive the next batch from the producer.
            // ============================================================
            // WHY RECV FIRST, BEFORE CHECKING IN_FLIGHT?
            //
            // This is the key insight of the double-buffered pattern. By
            // calling `producer.recv()` BEFORE `gpu.wait_and_readback()`,
            // we overlap the channel wait (which may block if the producer
            // is still packing the next batch) with the GPU's execution of
            // the in-flight batch.
            //
            // Timeline (best case, producer keeps up):
            //   recv:  [====]            <- CPU blocks on channel briefly
            //   GPU:   [==============]  <- crunching in_flight batch
            //   wait:               [=]  <- GPU already done when we check
            //
            // If we did wait first, then recv, both would be serial:
            //   GPU:   [==============]
            //   wait:  [==============]  <- CPU blocked the whole time
            //   recv:                  [====] <- CPU blocked again
            //
            // The overlapped version saves the entire recv time.
            let recv_started = Instant::now();
            let message = producer.recv()?;
            timings.consumer_idle_wait += recv_started.elapsed();

            // ============================================================
            // Step B: Complete the previous in-flight GPU dispatch.
            // ============================================================
            // We MUST wait for the previous dispatch before encoding a new
            // one because `params_buf` and `result_buf` are shared -- writing
            // new params while the GPU is still reading old ones would be a
            // data race.
            //
            // `.take()` moves the value out of `in_flight`, leaving `None`.
            // This ensures we only wait once per dispatch.
            if let Some(flight) = in_flight.take() {
                let (maybe_match, gpu_wait, result_readback) = gpu.wait_and_readback(&flight.cmd_buf);
                timings.gpu_wait += gpu_wait;
                timings.result_readback += result_readback;
                timings.dispatch_total += gpu_wait + result_readback;
                candidates_tested =
                    candidates_tested.saturating_add(flight.candidate_count);

                // If the GPU found a match, report it and exit immediately.
                if let Some(local_match_index) = maybe_match {
                    return handle_match(&flight, local_match_index, candidates_tested, started_at, &timings);
                }

                // Periodic rate reporting after completing a GPU batch.
                let now = Instant::now();
                let elapsed_since_last_report = now
                    .checked_duration_since(last_rate_report.reported_at)
                    .unwrap_or_default();
                if elapsed_since_last_report >= Duration::from_secs(1) {
                    let current_rate_report =
                        RateReportSnapshot::capture(now, candidates_tested, &timings);
                    let delta = current_rate_report.delta_since(last_rate_report);
                    let rate_end_to_end =
                        rate_per_second(delta.candidates_tested, delta.wall_time);
                    let rate_gpu_only =
                        rate_per_second(delta.candidates_tested, delta.gpu_wait);
                    let elapsed_total = now.duration_since(started_at);
                    eprintln!(
                        "RATE end_to_end={}/s gpu_only={}/s idle_wait={}ms build={}ms (delta={} in {:.2}s, total={} in {:.2}s, tpg={})",
                        format_human_count(rate_end_to_end),
                        format_human_count(rate_gpu_only),
                        format_duration_millis(delta.consumer_idle_wait),
                        format_duration_millis(delta.wordlist_batch_build),
                        format_human_count(delta.candidates_tested as f64),
                        delta.wall_time.as_secs_f64(),
                        format_human_count(candidates_tested as f64),
                        elapsed_total.as_secs_f64(),
                        gpu.current_threadgroup_width()
                    );
                    last_rate_report = current_rate_report;
                }

                // ============================================================
                // Step B2: Recycle the completed batch's buffers.
                // ============================================================
                // The GPU is done with this batch's Metal buffers, so we send
                // the WordBatch back to the producer for reuse. This avoids
                // allocating new shared-memory buffers for every batch.
                //
                // This is like returning a library book so someone else can
                // check it out -- the physical book (buffer) is reused, only
                // the contents (candidate words) change.
                producer.recycle(flight.batch);
            }

            // ============================================================
            // Step C: Process the received message.
            // ============================================================
            // The producer sends one of three message types:
            // - Batch: a new batch of candidates ready for GPU dispatch.
            // - Eof: the entire wordlist has been processed.
            // - Error: the producer encountered a fatal error.
            let (batch, build_time) = match message {
                ProducerMessage::Batch {
                    batch,
                    build_time,
                    plan_time,
                    pack_time,
                    parser_stats,
                } => {
                    timings.apply_parser_stats(parser_stats);
                    timings.wordlist_batch_plan += plan_time;
                    timings.wordlist_batch_pack += pack_time;
                    (batch, build_time)
                }
                ProducerMessage::Eof { parser_stats } => {
                    timings.apply_parser_stats(parser_stats);
                    break; // Exit the loop; drain the last in-flight batch below.
                }
                ProducerMessage::Error(err) => bail!("{err}"),
            };

            timings.wordlist_batch_build += build_time;
            // Run autotune on the very first batch (if --autotune was passed).
            // This burns a small sample (~16K candidates) to find the optimal
            // threadgroup width before entering steady-state dispatch.
            if !autotune_done {
                gpu.autotune_threadgroup_width(&target_signature, &batch)?;
                autotune_done = true;
            }

            let batch_candidate_count = batch.candidate_count() as u64;
            timings.batch_count = timings.batch_count.saturating_add(1);
            timings.total_batch_candidates = timings
                .total_batch_candidates
                .saturating_add(batch_candidate_count);
            timings.total_batch_word_bytes = timings
                .total_batch_word_bytes
                .saturating_add(batch.word_bytes_len() as u64);

            // ============================================================
            // Step D: Encode + commit the new batch (non-blocking).
            // ============================================================
            // After commit(), the GPU starts executing immediately. The CPU
            // loops back to Step A (recv) while the GPU crunches this batch.
            // This is the "double buffer" in action: we just submitted batch
            // N+1, and on the next iteration we will recv batch N+2 while
            // the GPU processes batch N+1.
            let (cmd_buf, host_prep, command_encode) =
                gpu.encode_and_commit(&target_signature, &batch)?;
            timings.host_prep += host_prep;
            timings.command_encode += command_encode;
            timings.dispatch_total += host_prep + command_encode;

            // Store the batch as in-flight. On the next loop iteration,
            // we will wait for this command buffer to complete (Step B).
            in_flight = Some(InFlightBatch {
                cmd_buf,
                batch,
                candidate_count: batch_candidate_count,
            });
        }

        // ============================================================
        // Post-loop: Drain the last in-flight batch after EOF.
        // ============================================================
        // When the producer sends EOF, we break out of the loop. But there
        // may still be one batch executing on the GPU. We must wait for it
        // and check its result before declaring "NOT FOUND".
        if let Some(flight) = in_flight.take() {
            let (maybe_match, gpu_wait, result_readback) = gpu.wait_and_readback(&flight.cmd_buf);
            timings.gpu_wait += gpu_wait;
            timings.result_readback += result_readback;
            timings.dispatch_total += gpu_wait + result_readback;
            candidates_tested =
                candidates_tested.saturating_add(flight.candidate_count);

            if let Some(local_match_index) = maybe_match {
                return handle_match(&flight, local_match_index, candidates_tested, started_at, &timings);
            }

            producer.recycle(flight.batch);
        }

        // If we get here, every candidate in the wordlist was tested and none
        // matched the target JWT signature.
        let elapsed = started_at.elapsed();
        let rate_end_to_end = rate_per_second(candidates_tested, elapsed);
        let rate_gpu_only = rate_per_second(candidates_tested, timings.gpu_wait);
        print_final_stats(
            candidates_tested,
            elapsed,
            rate_end_to_end,
            rate_gpu_only,
            &timings,
        );
        println!("NOT FOUND");
        Ok(false)
    })();

    // ---- Phase 6: Cleanup / shutdown ----
    //
    // Shutdown order matters, especially on early exit (error or match found):
    //
    // 1. `producer.stop()` -- Signal the producer thread to stop parsing.
    // 2. `producer.close_receiver()` -- Drop our end of the channel so the
    //    producer's bounded `send()` cannot block forever waiting for us to
    //    consume. Without this, the producer could deadlock if its channel
    //    buffer is full and we have stopped consuming.
    // 3. `producer.join()` -- Wait for the producer thread to exit cleanly.
    //
    // This pattern ensures no threads are left dangling, which is important
    // for clean resource release (mmap handles, Metal buffers, etc.).
    producer.stop();
    producer.close_receiver();
    let join_result = producer.join();

    // Preserve the primary command error if one occurred. Producer join errors
    // only matter when the main run path succeeded. This uses Rust's pattern
    // matching to express the error priority logic concisely.
    match (run_result, join_result) {
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Ok(found), Ok(())) => Ok(found),
    }
}
