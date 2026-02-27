use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};

use super::args::{DEFAULT_PIPELINE_DEPTH, Hs256WordlistArgs, ParserConfig};
use super::batch::APPROX_WORD_BATCH_BUFFER_BYTES;
use super::gpu::GpuHs256BruteForcer;
use super::jwt::parse_hs256_jwt;
use super::producer::{ProducerMessage, WordlistProducer};
use super::stats::{
    RateReportSnapshot, RunTimings, format_duration_millis, format_human_count, print_final_stats,
    rate_per_second,
};

/// End-to-end HS256 cracking flow:
/// 1) parse the JWT,
/// 2) initialize Metal + reusable buffers,
/// 3) overlap wordlist parsing with GPU dispatch,
/// 4) report the first matching secret (or NOT FOUND).
pub fn run(args: Hs256WordlistArgs) -> anyhow::Result<bool> {
    let (signing_input, target_signature) = parse_hs256_jwt(&args.jwt)?;
    let mut gpu = GpuHs256BruteForcer::new(&signing_input)?;
    let parser_config = ParserConfig::from_args(&args);
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
        gpu.device.clone(),
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

        loop {
            // Time how long the consumer waits on the producer. This measures
            // exposed parser latency (work not hidden behind GPU execution).
            let recv_started = Instant::now();
            let message = producer.recv()?;
            timings.consumer_idle_wait += recv_started.elapsed();

            let (batch, build_time) = match message {
                ProducerMessage::Batch {
                    batch,
                    build_time,
                    plan_time,
                    pack_time,
                    parser_stats,
                } => {
                    // Producer snapshots parser counters after each batch build.
                    // We refresh the consumer-side copy here so both periodic
                    // and final reports reflect the latest parser state.
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
                // Autotune once, using the first real batch shape as a proxy
                // for the rest of the run.
                gpu.autotune_threadgroup_width(target_signature, &batch)?;
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

            // Dispatch the batch and accumulate fine-grained timing buckets.
            let (maybe_match, batch_timings) = gpu.try_batch(target_signature, &batch)?;
            timings.host_prep += batch_timings.host_prep;
            timings.command_encode += batch_timings.command_encode;
            timings.gpu_wait += batch_timings.gpu_wait;
            timings.result_readback += batch_timings.result_readback;
            timings.dispatch_total += batch_timings.total;

            if let Some(local_match_index) = maybe_match {
                // The GPU returns a batch-local index. Reconstruct the absolute
                // wordlist index on the host so large (>u32) wordlists remain
                // supported without changing the GPU result slot width.
                candidates_tested = candidates_tested.saturating_add(batch_candidate_count);
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

                // Reconstruct the absolute wordlist index using the host-side
                // `u64` batch base. `_global_index` is currently only validated
                // (for overflow) and not printed, but keeping the computation
                // here documents the invariant and catches impossible states.
                let _global_index = batch
                    .candidate_index_base
                    .checked_add(local_match_index as u64)
                    .ok_or_else(|| {
                        anyhow!("candidate index overflow while reconstructing result")
                    })?;
                let local_index =
                    // The GPU result must fit `usize` because every dispatch is
                    // bounded by `MAX_CANDIDATES_PER_BATCH` (far below `usize`
                    // on supported targets). Convert explicitly for safety.
                    usize::try_from(local_match_index).context("GPU returned invalid result index")?;
                let secret = batch
                    .word_string_lossy(local_index)
                    .ok_or_else(|| anyhow!("GPU returned invalid local candidate index"))?;
                println!("HS256 key: {secret}");
                return Ok(true);
            }

            candidates_tested = candidates_tested.saturating_add(batch_candidate_count);
            let now = Instant::now();
            let elapsed_since_last_report = now
                .checked_duration_since(last_rate_report.reported_at)
                .unwrap_or_default();
            if elapsed_since_last_report >= Duration::from_secs(1) {
                let current_rate_report =
                    RateReportSnapshot::capture(now, candidates_tested, &timings);
                let delta = current_rate_report.delta_since(last_rate_report);
                // `RATE` is windowed (interval) by design: use deltas from the
                // last report, while final `STATS` remains cumulative.
                let rate_end_to_end = rate_per_second(delta.candidates_tested, delta.wall_time);
                let rate_gpu_only = rate_per_second(delta.candidates_tested, delta.gpu_wait);
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

            // After dispatch/readback completes with no match, the consumer no
            // longer needs the batch contents. Return ownership to the producer
            // so it can refill the same allocation on the next iteration.
            //
            // We intentionally do not recycle on the match path above because we
            // still need `batch` to reconstruct and print the winning secret.
            producer.recycle(batch);
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
