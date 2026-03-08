use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};

use super::args::{WordlistArgs, DEFAULT_PIPELINE_DEPTH};
use super::batch::APPROX_WORD_BATCH_BUFFER_BYTES;
use super::gpu::{CrackVariant, GpuBruteForcer, GpuCommandHandle};
#[cfg(target_os = "macos")]
use super::gpu::metal::{MetalAesKwBruteForcer, MetalBruteForcer};
#[cfg(target_os = "linux")]
use super::gpu::cuda::{CudaAesKwBruteForcer, CudaBruteForcer};
use super::jwt::{parse_jwe_aes_kw, parse_jwt};
use super::producer::{ProducerMessage, WordlistProducer};
use super::stats::{
    RateReportSnapshot, RunTimings, format_duration_millis, format_human_count, print_final_stats,
    rate_per_second,
};

struct InFlightBatch {
    cmd_buf: GpuCommandHandle,
    batch: super::batch::WordBatch,
    candidate_count: u64,
}

/// Print match result after a successful crack.
///
/// Shared by both the GPU (HMAC) and CPU (JWE) paths. Takes the
/// `WordBatch` directly rather than an `InFlightBatch` so it works
/// regardless of whether the batch came from GPU dispatch or CPU
/// processing.
fn handle_match(
    variant: CrackVariant,
    batch: &super::batch::WordBatch,
    local_match_index: u32,
    candidates_tested: u64,
    started_at: Instant,
    timings: &RunTimings,
) -> anyhow::Result<bool> {
    let elapsed = started_at.elapsed();
    let rate_end_to_end = rate_per_second(candidates_tested, elapsed);
    let rate_compute_only = rate_per_second(candidates_tested, timings.gpu_wait);
    print_final_stats(candidates_tested, elapsed, rate_end_to_end, rate_compute_only, timings);
    let local_index = usize::try_from(local_match_index)
        .context("returned invalid result index")?;
    let secret = batch
        .word_string_lossy(local_index)
        .ok_or_else(|| anyhow!("returned invalid local candidate index"))?;
    println!("{} key: {secret}", variant.label());
    Ok(true)
}

/// Run the full wordlist cracking pipeline for any supported token type.
///
/// Dispatches to the appropriate GPU compute backend based on `variant`:
///
/// - **`Hmac(hv)`**: Metal HMAC-SHA shader, double-buffered GPU dispatch.
/// - **`JweAesKw(akv)`**: Metal software AES shader for RFC 3394 Key Unwrap
///   (A128KW / A192KW / A256KW, compile-time specialised per key size).
///
/// Both paths share the same producer infrastructure (mmap, parser,
/// planner, packer, `WordBatch` recycling) and timing/reporting code
/// via the generic `run_gpu_crack<B: GpuBruteForcer>` dispatch loop.
///
/// Returns `Ok(true)` if the key was found, `Ok(false)` if not found.
pub(crate) fn run_wordlist_crack(
    variant: CrackVariant,
    args: WordlistArgs,
) -> anyhow::Result<bool> {
    match variant {
        CrackVariant::Hmac(hv) => {
            let (signing_input, target_signature) = parse_jwt(hv, &args.jwt)?;
            #[cfg(target_os = "macos")]
            let gpu = MetalBruteForcer::new(hv, &signing_input)?;
            #[cfg(target_os = "linux")]
            let gpu = CudaBruteForcer::new(hv, &signing_input)?;
            run_gpu_crack(variant, gpu, &target_signature, args)
        }
        CrackVariant::JweAesKw(akv) => {
            let encrypted_key = parse_jwe_aes_kw(akv, &args.jwt)?;
            let n = encrypted_key.len() / 8 - 1;
            #[cfg(target_os = "macos")]
            let gpu = MetalAesKwBruteForcer::new(akv, &encrypted_key, n)?;
            #[cfg(target_os = "linux")]
            let gpu = CudaAesKwBruteForcer::new(akv, &encrypted_key, n)?;
            run_gpu_crack(variant, gpu, &encrypted_key, args)
        }
    }
}

// ---------------------------------------------------------------------------
// Generic GPU crack path
// ---------------------------------------------------------------------------

/// GPU-accelerated cracking pipeline, generic over the brute-forcer backend.
///
/// This is the shared dispatch loop used by both HMAC-SHA (via
/// `MetalBruteForcer`) and JWE A128KW (via `MetalAesKwBruteForcer`).
/// The caller is responsible for parsing the token and constructing the
/// appropriate GPU backend; this function handles the double-buffered
/// GPU dispatch loop, timing, and result reporting.
///
/// ## Parameters
///
/// - `target_data`: the bytes passed to `gpu.encode_and_commit()`. For
///   HMAC, this is the JWT signature. For A128KW, the implementation
///   ignores it (encrypted_key is pre-loaded in the GPU buffer), but
///   we pass it through to satisfy the trait contract.
fn run_gpu_crack<B: GpuBruteForcer>(
    variant: CrackVariant,
    mut gpu: B,
    target_data: &[u8],
    args: WordlistArgs,
) -> anyhow::Result<bool> {
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

        let mut in_flight: Option<InFlightBatch> = None;

        loop {
            let recv_started = Instant::now();
            let message = producer.recv()?;
            timings.consumer_idle_wait += recv_started.elapsed();

            if let Some(flight) = in_flight.take() {
                let (maybe_match, gpu_wait, result_readback) = gpu.wait_and_readback(&flight.cmd_buf);
                timings.gpu_wait += gpu_wait;
                timings.result_readback += result_readback;
                timings.dispatch_total += gpu_wait + result_readback;
                candidates_tested =
                    candidates_tested.saturating_add(flight.candidate_count);

                if let Some(local_match_index) = maybe_match {
                    return handle_match(variant, &flight.batch, local_match_index, candidates_tested, started_at, &timings);
                }

                report_rate_if_due(
                    &mut last_rate_report,
                    candidates_tested,
                    started_at,
                    &timings,
                    Some(gpu.current_threadgroup_width()),
                );

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
                gpu.autotune_threadgroup_width(target_data, &batch)?;
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

            let (cmd_buf, host_prep, command_encode) =
                gpu.encode_and_commit(target_data, &batch)?;
            timings.host_prep += host_prep;
            timings.command_encode += command_encode;
            timings.dispatch_total += host_prep + command_encode;

            in_flight = Some(InFlightBatch {
                cmd_buf,
                batch,
                candidate_count: batch_candidate_count,
            });
        }

        // Drain the last in-flight batch after EOF.
        if let Some(flight) = in_flight.take() {
            let (maybe_match, gpu_wait, result_readback) = gpu.wait_and_readback(&flight.cmd_buf);
            timings.gpu_wait += gpu_wait;
            timings.result_readback += result_readback;
            timings.dispatch_total += gpu_wait + result_readback;
            candidates_tested =
                candidates_tested.saturating_add(flight.candidate_count);

            if let Some(local_match_index) = maybe_match {
                return handle_match(variant, &flight.batch, local_match_index, candidates_tested, started_at, &timings);
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

    producer.stop();
    producer.close_receiver();
    let join_result = producer.join();

    match (run_result, join_result) {
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Ok(found), Ok(())) => Ok(found),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Print a windowed throughput report if at least 1 second has elapsed
/// since the last report.
///
/// Shared by both the GPU and CPU consumer loops. The `tpg` parameter
/// is `Some(threadgroup_width)` for GPU dispatch and `None` for CPU mode
/// (where threadgroup width is not applicable).
fn report_rate_if_due(
    last_report: &mut RateReportSnapshot,
    candidates_tested: u64,
    started_at: Instant,
    timings: &RunTimings,
    tpg: Option<usize>,
) {
    let now = Instant::now();
    let elapsed_since_last = now
        .checked_duration_since(last_report.reported_at)
        .unwrap_or_default();
    if elapsed_since_last < Duration::from_secs(1) {
        return;
    }
    let current = RateReportSnapshot::capture(now, candidates_tested, timings);
    let delta = current.delta_since(*last_report);
    let rate_end_to_end = rate_per_second(delta.candidates_tested, delta.wall_time);
    let rate_compute = rate_per_second(delta.candidates_tested, delta.gpu_wait);
    let elapsed_total = now.duration_since(started_at);

    match tpg {
        Some(tpg_val) => {
            eprintln!(
                "RATE end_to_end={}/s gpu_only={}/s idle_wait={}ms build={}ms (delta={} in {:.2}s, total={} in {:.2}s, tpg={})",
                format_human_count(rate_end_to_end),
                format_human_count(rate_compute),
                format_duration_millis(delta.consumer_idle_wait),
                format_duration_millis(delta.wordlist_batch_build),
                format_human_count(delta.candidates_tested as f64),
                delta.wall_time.as_secs_f64(),
                format_human_count(candidates_tested as f64),
                elapsed_total.as_secs_f64(),
                tpg_val
            );
        }
        None => {
            eprintln!(
                "RATE end_to_end={}/s compute_only={}/s idle_wait={}ms build={}ms (delta={} in {:.2}s, total={} in {:.2}s)",
                format_human_count(rate_end_to_end),
                format_human_count(rate_compute),
                format_duration_millis(delta.consumer_idle_wait),
                format_duration_millis(delta.wordlist_batch_build),
                format_human_count(delta.candidates_tested as f64),
                delta.wall_time.as_secs_f64(),
                format_human_count(candidates_tested as f64),
                elapsed_total.as_secs_f64(),
            );
        }
    }

    *last_report = current;
}

#[cfg(test)]
mod tests {
    use super::super::gpu::{AesKwVariant, CrackVariant, HmacVariant};
    use super::super::test_support::{make_test_jwt, make_test_jwe, write_temp_wordlist};
    use super::*;
    use std::path::PathBuf;

    fn make_args(token: &str, wordlist: PathBuf) -> WordlistArgs {
        WordlistArgs {
            jwt: token.to_string(),
            wordlist,
            threads_per_group: None,
            parser_threads: Some(1),
            pipeline_depth: Some(2),
            packer_threads: Some(1),
            autotune: false,
        }
    }

    #[test]
    fn hs256_cracks_known_secret() {
        let jwt = make_test_jwt("HS256", r#"{"sub":"test"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\nwrong2\npassword\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs256), make_args(&jwt, path.clone())).unwrap();
        assert!(result, "expected HS256 crack to find 'password'");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hs256_reports_not_found() {
        let jwt = make_test_jwt("HS256", r#"{"sub":"test"}"#, b"43i358hsfksdbfffsdf");
        let path = write_temp_wordlist(b"wrong1\nwrong2\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs256), make_args(&jwt, path.clone())).unwrap();
        assert!(!result, "expected HS256 crack to report not found");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hs384_cracks_known_secret() {
        let jwt = make_test_jwt("HS384", r#"{"sub":"test"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\nwrong2\npassword\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs384), make_args(&jwt, path.clone())).unwrap();
        assert!(result, "expected HS384 crack to find 'password'");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hs384_reports_not_found() {
        let jwt = make_test_jwt("HS384", r#"{"sub":"test"}"#, b"43i358hsfksdbfffsdf");
        let path = write_temp_wordlist(b"wrong1\nwrong2\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs384), make_args(&jwt, path.clone())).unwrap();
        assert!(!result, "expected HS384 crack to report not found");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hs512_cracks_known_secret() {
        let jwt = make_test_jwt("HS512", r#"{"sub":"test"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\nwrong2\npassword\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs512), make_args(&jwt, path.clone())).unwrap();
        assert!(result, "expected HS512 crack to find 'password'");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hs512_reports_not_found() {
        let jwt = make_test_jwt("HS512", r#"{"sub":"test"}"#, b"43i358hsfksdbfffsdf");
        let path = write_temp_wordlist(b"wrong1\nwrong2\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs512), make_args(&jwt, path.clone())).unwrap();
        assert!(!result, "expected HS512 crack to report not found");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn autocrack_routes_hs256() {
        let jwt = make_test_jwt("HS256", r#"{"sub":"auto"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let variant = super::super::jwt::detect_token_variant(&jwt).unwrap();
        let result = run_wordlist_crack(variant, make_args(&jwt, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via HS256");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn autocrack_routes_hs384() {
        let jwt = make_test_jwt("HS384", r#"{"sub":"auto"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let variant = super::super::jwt::detect_token_variant(&jwt).unwrap();
        let result = run_wordlist_crack(variant, make_args(&jwt, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via HS384");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn autocrack_routes_hs512() {
        let jwt = make_test_jwt("HS512", r#"{"sub":"auto"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let variant = super::super::jwt::detect_token_variant(&jwt).unwrap();
        let result = run_wordlist_crack(variant, make_args(&jwt, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via HS512");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn autocrack_rejects_unsupported_alg() {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"test"}"#);
        let fake_sig = URL_SAFE_NO_PAD.encode(b"fakesignaturebytes");
        let jwt = format!("{header}.{payload}.{fake_sig}");

        let result = super::super::jwt::detect_token_variant(&jwt);
        let err = result.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("unsupported JWT algorithm: RS256"),
            "expected unsupported algorithm error, got: {msg}"
        );
    }

    /// JWE A128KW finds the known secret "password" in a small wordlist.
    #[test]
    fn jwe_a128kw_cracks_known_secret() {
        let jwe = make_test_jwe(AesKwVariant::A128kw, b"password");
        let path = write_temp_wordlist(b"wrong1\nwrong2\npassword\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A128kw), make_args(&jwe, path.clone())).unwrap();
        assert!(result, "expected A128KW crack to find 'password'");
        let _ = std::fs::remove_file(path);
    }

    /// JWE A128KW reports not-found when the secret is absent from the wordlist.
    #[test]
    fn jwe_a128kw_reports_not_found() {
        let jwe = make_test_jwe(AesKwVariant::A128kw, b"fsdgdfsgdsfsgdf");
        let path = write_temp_wordlist(b"wrong1\nwrong2\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A128kw), make_args(&jwe, path.clone())).unwrap();
        assert!(!result, "expected A128KW crack to report not found");
        let _ = std::fs::remove_file(path);
    }

    /// Autocrack routes a 5-part A128KW JWE token to the correct path.
    #[test]
    fn autocrack_detects_jwe_a128kw() {
        let jwe = make_test_jwe(AesKwVariant::A128kw, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let variant = super::super::jwt::detect_token_variant(&jwe).unwrap();
        assert_eq!(variant, CrackVariant::JweAesKw(AesKwVariant::A128kw));
        let result = run_wordlist_crack(variant, make_args(&jwe, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via A128KW");
        let _ = std::fs::remove_file(path);
    }

    /// JWE A192KW finds the known secret "password" in a small wordlist.
    #[test]
    fn jwe_a192kw_cracks_known_secret() {
        let jwe = make_test_jwe(AesKwVariant::A192kw, b"password");
        let path = write_temp_wordlist(b"wrong1\nwrong2\npassword\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A192kw), make_args(&jwe, path.clone())).unwrap();
        assert!(result, "expected A192KW crack to find 'password'");
        let _ = std::fs::remove_file(path);
    }

    /// JWE A192KW reports not-found when the secret is absent from the wordlist.
    #[test]
    fn jwe_a192kw_reports_not_found() {
        let jwe = make_test_jwe(AesKwVariant::A192kw, b"fsdgdfsgdsfsgdf");
        let path = write_temp_wordlist(b"wrong1\nwrong2\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A192kw), make_args(&jwe, path.clone())).unwrap();
        assert!(!result, "expected A192KW crack to report not found");
        let _ = std::fs::remove_file(path);
    }

    /// Autocrack routes a 5-part A192KW JWE token to the correct path.
    #[test]
    fn autocrack_detects_jwe_a192kw() {
        let jwe = make_test_jwe(AesKwVariant::A192kw, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let variant = super::super::jwt::detect_token_variant(&jwe).unwrap();
        assert_eq!(variant, CrackVariant::JweAesKw(AesKwVariant::A192kw));
        let result = run_wordlist_crack(variant, make_args(&jwe, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via A192KW");
        let _ = std::fs::remove_file(path);
    }

    /// JWE A256KW finds the known secret "password" in a small wordlist.
    #[test]
    fn jwe_a256kw_cracks_known_secret() {
        let jwe = make_test_jwe(AesKwVariant::A256kw, b"password");
        let path = write_temp_wordlist(b"wrong1\nwrong2\npassword\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A256kw), make_args(&jwe, path.clone())).unwrap();
        assert!(result, "expected A256KW crack to find 'password'");
        let _ = std::fs::remove_file(path);
    }

    /// JWE A256KW reports not-found when the secret is absent from the wordlist.
    #[test]
    fn jwe_a256kw_reports_not_found() {
        let jwe = make_test_jwe(AesKwVariant::A256kw, b"fsdgdfsgdsfsgdf");
        let path = write_temp_wordlist(b"wrong1\nwrong2\nwrong3\n");
        let result = run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A256kw), make_args(&jwe, path.clone())).unwrap();
        assert!(!result, "expected A256KW crack to report not found");
        let _ = std::fs::remove_file(path);
    }

    /// Autocrack routes a 5-part A256KW JWE token to the correct path.
    #[test]
    fn autocrack_detects_jwe_a256kw() {
        let jwe = make_test_jwe(AesKwVariant::A256kw, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let variant = super::super::jwt::detect_token_variant(&jwe).unwrap();
        assert_eq!(variant, CrackVariant::JweAesKw(AesKwVariant::A256kw));
        let result = run_wordlist_crack(variant, make_args(&jwe, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via A256KW");
        let _ = std::fs::remove_file(path);
    }
}
