//! End-to-end orchestration for the HS384 wordlist cracking command.
//!
//! # Learning note: the pipeline pattern
//!
//! This module follows the same producer-consumer pipeline pattern used by
//! every cracking subcommand (HS256, HS384, HS512):
//!
//!   1. **Parse the JWT** -- validate the algorithm and extract the signing
//!      input and target signature.
//!   2. **Initialize the GPU** -- compile the Metal kernel, allocate buffers.
//!   3. **Spawn a wordlist producer** -- background threads mmap the file,
//!      split it into chunks, parse lines, and pack GPU batches.
//!   4. **Double-buffered dispatch loop** -- while the GPU executes batch N,
//!      the host receives batch N+1 from the producer.  This overlap hides
//!      most of the host-side latency behind GPU execution time.
//!   5. **Report results** -- print the cracked secret or "NOT FOUND", plus
//!      detailed timing statistics.
//!
//! The only HS384-specific parts are the JWT parser (48-byte signature) and
//! the GPU wrapper (6-word comparison, `hs384_*` kernel names).  Everything
//! else is generic infrastructure shared via `crate::common`.

use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};

use super::args::{DEFAULT_PIPELINE_DEPTH, Hs384WordlistArgs};
use crate::common::gpu::{GpuBruteForcer, GpuCommandHandle, HmacVariant, metal::MetalBruteForcer};
use super::jwt::parse_hs384_jwt;
use crate::common::batch::APPROX_WORD_BATCH_BUFFER_BYTES;
use crate::common::producer::{ProducerMessage, WordlistProducer};
use crate::common::stats::{
    RateReportSnapshot, RunTimings, format_duration_millis, format_human_count, print_final_stats,
    rate_per_second,
};

/// Holds the command buffer + batch for a currently executing GPU dispatch.
/// Defined at module scope so `handle_match` can reference it.
///
/// This is the same pattern used in the HS256 and HS512 command modules.
struct InFlightBatch {
    cmd_buf: GpuCommandHandle,
    batch: crate::common::batch::WordBatch,
    candidate_count: u64,
}

// Extract the matched secret from a completed GPU batch, print final stats,
// and return `Ok(true)`. Called from both the main loop and the post-EOF drain.
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
    println!("HS384 key: {secret}");
    Ok(true)
}

/// End-to-end HS384 cracking flow:
/// 1) parse the JWT,
/// 2) initialize Metal + reusable buffers,
/// 3) overlap wordlist parsing with GPU dispatch,
/// 4) report the first matching secret (or NOT FOUND).
pub fn run(args: Hs384WordlistArgs) -> anyhow::Result<bool> {
    let (signing_input, target_signature) = parse_hs384_jwt(&args.jwt)?;
    let mut gpu = MetalBruteForcer::new(HmacVariant::Hs384, &signing_input)?;
    let parser_config = args.parser_config();
    let pipeline_depth = args.pipeline_depth.unwrap_or(DEFAULT_PIPELINE_DEPTH);
    let packer_threads = args
        .packer_threads
        .unwrap_or(2)
        .max(1)
        .min(pipeline_depth.max(1));
    if let Some(tpg) = args.threads_per_group {
        gpu.set_threadgroup_width(tpg)?;
    }
    let approx_prefetch_bytes = APPROX_WORD_BATCH_BUFFER_BYTES.saturating_mul(pipeline_depth);
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

    // Start the parser thread before entering the main loop so batch
    // construction can overlap with GPU setup and dispatch wait time.
    let mut producer = WordlistProducer::spawn(
        args.wordlist.clone(),
        gpu.device().clone(),
        parser_config,
        pipeline_depth,
        packer_threads,
    );
    let run_result = (|| -> anyhow::Result<bool> {
        let started_at = Instant::now();
        let mut candidates_tested: u64 = 0;
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

        // Double-buffered dispatch: receive batch N+1 while GPU executes batch N.
        // `in_flight` holds the command buffer + batch for the currently executing
        // GPU dispatch; the consumer loop overlaps recv with GPU execution.
        let mut in_flight: Option<InFlightBatch> = None;

        loop {
            // Receive the next batch while the GPU processes the in-flight one.
            // This overlap is the key throughput improvement: idle_wait now runs
            // concurrently with gpu_wait instead of serially.
            let recv_started = Instant::now();
            let message = producer.recv()?;
            timings.consumer_idle_wait += recv_started.elapsed();

            // Complete the previous in-flight GPU dispatch before encoding a new
            // one (params_buf/result_buf are shared across dispatches).
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

                producer.recycle(flight.batch);
            }

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
                    break;
                }
                ProducerMessage::Error(err) => bail!("{err}"),
            };

            timings.wordlist_batch_build += build_time;
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

            // Encode and commit immediately (non-blocking). The GPU starts
            // executing while we loop back to recv the next batch.
            let (cmd_buf, host_prep, command_encode) =
                gpu.encode_and_commit(&target_signature, &batch)?;
            timings.host_prep += host_prep;
            timings.command_encode += command_encode;
            timings.dispatch_total += host_prep + command_encode;

            in_flight = Some(InFlightBatch {
                cmd_buf,
                batch,
                candidate_count: batch_candidate_count,
            });
        }

        // Drain the last in-flight batch after EOF from the producer.
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

    // Shutdown order matters on early exit: signal stop, drop the receiver so a
    // bounded send cannot block, then join the producer thread.
    producer.stop();
    producer.close_receiver();
    let join_result = producer.join();

    // Preserve the primary command error if one occurred. Producer join errors
    // only matter when the main run path succeeded.
    match (run_result, join_result) {
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Ok(found), Ok(())) => Ok(found),
    }
}

#[cfg(test)]
mod tests {
    use super::super::args::Hs384WordlistArgs;
    use crate::common::test_support::{make_test_jwt, write_temp_wordlist};
    use std::path::PathBuf;

    fn make_args(jwt: &str, wordlist: PathBuf) -> Hs384WordlistArgs {
        Hs384WordlistArgs {
            jwt: jwt.to_string(),
            wordlist,
            threads_per_group: None,
            parser_threads: Some(1),
            pipeline_depth: Some(2),
            packer_threads: Some(1),
            autotune: false,
        }
    }

    #[test]
    fn hs384_cracks_known_secret() {
        let jwt = make_test_jwt("HS384", r#"{"sub":"test"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\nwrong2\npassword\nwrong3\n");
        let result = super::run(make_args(&jwt, path.clone())).unwrap();
        assert!(result, "expected HS384 crack to find 'password'");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hs384_reports_not_found() {
        let jwt = make_test_jwt("HS384", r#"{"sub":"test"}"#, b"43i358hsfksdbfffsdf");
        let path = write_temp_wordlist(b"wrong1\nwrong2\nwrong3\n");
        let result = super::run(make_args(&jwt, path.clone())).unwrap();
        assert!(!result, "expected HS384 crack to report not found");
        let _ = std::fs::remove_file(path);
    }
}
