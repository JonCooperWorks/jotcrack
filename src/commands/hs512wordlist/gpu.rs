//! GPU (Metal) dispatch layer for HS512 wordlist cracking.
//!
//! # Learning note: HS512 uses the full 8-word SHA-512 comparison
//!
//! This file is structurally identical to `hs384wordlist/gpu.rs`, with two
//! key differences:
//!
//!   1. **All 8 u64 words are populated.**  The host converts the 64-byte
//!      JWT signature into `[u64; 8]` and the GPU compares all 8 words
//!      against the computed HMAC-SHA512 output.  (HS384 only fills 6.)
//!
//!   2. **Different kernel function names.**  We select `hs512_wordlist` and
//!      `hs512_wordlist_short_keys` from the shared Metal source, instead of
//!      the `hs384_*` variants.
//!
//! Everything else -- the `Hs512BruteForceParams` struct layout, the
//! `include_str!()` embedding pattern, the short-key threshold of 128 bytes,
//! and the autotune/dispatch/readback pipeline -- is identical.
//!
//! # Learning note: the shared `Hs512BruteForceParams` struct
//!
//! Both the HS384 and HS512 Rust modules define their own local copy of
//! `Hs512BruteForceParams`.  This is intentional: each module is
//! self-contained, and the struct is trivially small.  The name says "Hs512"
//! because it was sized for 8 u64 words (the HS512 maximum); HS384 simply
//! leaves the last 2 words zeroed.

use anyhow::{Context, anyhow, bail};
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::{Duration, Instant};

use crate::commands::common::batch::{DispatchBatchView, WordBatch};
use crate::commands::common::stats::{BatchDispatchTimings, format_human_count};

// Embed the Metal source into the binary so release builds do not depend on a
// runtime-relative source file path.
//
// Learning note: the same `.metal` file is also `include_str!`'d by the HS384
// gpu.rs.  Each binary only needs one copy because the linker deduplicates
// identical string constants, but conceptually both modules reference the
// same shared kernel source.
const METAL_SOURCE_EMBEDDED: &str = include_str!("../common/hs512_wordlist.metal");
// We keep both kernels loaded and choose per batch based on candidate lengths.
const METAL_FUNCTION_NAME_MIXED: &str = "hs512_wordlist";
const METAL_FUNCTION_NAME_SHORT_KEYS: &str = "hs512_wordlist_short_keys";
// The GPU writes the first matching *batch-local* candidate index here (the
// thread's `gid`, i.e. `0..candidate_count-1`), not an absolute wordlist index.
//
// Keeping the GPU result slot as a single `u32` preserves the same tiny shared
// buffer/atomic path while letting the host scale absolute wordlist indexing to
// `u64` for very large files (multi-billion non-empty lines).
//
// `u32::MAX` remains the sentinel for "no match in this batch".
const RESULT_NOT_FOUND_SENTINEL: u32 = u32::MAX;

// Host -> Metal parameter block. `#[repr(C)]` is required because Rust and the
// Metal kernel must agree on the exact field order and byte layout.
//
// Learning note: this is the same struct used by HS384.  For HS512, all 8
// target_signature words are filled with real data (the full 64-byte digest).
// The GPU hs512_* kernels compare all 8 words; the hs384_* kernels compare
// only the first 6.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs512BruteForceParams {
    // Optimization: stored as [u64; 8] big-endian words (converted from [u8; 64]
    // at upload time in `encode_and_commit_view`) so the GPU kernel can compare
    // the final HMAC state words directly without per-thread byte-to-word loads.
    //
    // For HS512: all 8 words are populated (8 * 8 = 64 bytes = full SHA-512 digest).
    target_signature: [u64; 8],
    // `message_length` is precomputed on the host so the kernel does not need
    // to infer it from buffers or rely on implicit buffer metadata.
    message_length: u32,
    candidate_count: u32,
}

// Owns Metal objects and the small persistent buffers reused across dispatches.
//
// Large per-batch payload buffers now live in `WordBatch` and are recycled
// producer<->consumer, which removes the old extra host copy before dispatch.
pub(super) struct GpuHs512BruteForcer {
    pub(super) device: metal::Device,
    queue: metal::CommandQueue,
    pipeline_mixed: metal::ComputePipelineState,
    pipeline_short_keys: metal::ComputePipelineState,
    params_buf: metal::Buffer,
    msg_buf: metal::Buffer,
    result_buf: metal::Buffer,
    message_length: u32,
    threadgroup_width: usize,
}

// ---- Host -> Metal shared-buffer copy helpers ------------------------------
// After the no-copy refactor these are only for small state writes (params,
// result sentinel, one-time JWT message upload), not batch payload bytes.
fn copy_bytes_to_buffer(buffer: &metal::Buffer, data: &[u8]) {
    debug_assert!(buffer.length() as usize >= data.len());
    if data.is_empty() {
        return;
    }
    // SAFETY: `buffer` is shared CPU-visible memory and `data.len()` bytes fit in the allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast::<u8>(), data.len());
    }
}

// Convenience wrapper for POD values (params struct, sentinel result value).
fn copy_value_to_buffer<T>(buffer: &metal::Buffer, value: &T) {
    copy_bytes_to_buffer(buffer, bytes_of(value));
}

impl GpuHs512BruteForcer {
    // Compile the Metal kernels, create the command queue, and allocate the
    // small persistent buffers shared by every dispatch (params/JWT/result).
    pub(super) fn new(signing_input: &[u8]) -> anyhow::Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow!("no Metal device available"))?;
        let compile_options = CompileOptions::new();
        // Compile the embedded Metal source at runtime. The source is no longer
        // read from disk, which makes the binary self-contained.
        let library = device
            .new_library_with_source(METAL_SOURCE_EMBEDDED, &compile_options)
            .map_err(|e| anyhow!("failed to compile Metal kernel: {e}"))?;

        let mixed_function = library
            .get_function(METAL_FUNCTION_NAME_MIXED, None)
            .map_err(|e| {
                anyhow!("failed to get Metal function {METAL_FUNCTION_NAME_MIXED}: {e}")
            })?;
        let short_function = library
            .get_function(METAL_FUNCTION_NAME_SHORT_KEYS, None)
            .map_err(|e| {
                anyhow!("failed to get Metal function {METAL_FUNCTION_NAME_SHORT_KEYS}: {e}")
            })?;

        let pipeline_mixed = device
            .new_compute_pipeline_state_with_function(&mixed_function)
            .map_err(|e| anyhow!("failed to create mixed compute pipeline: {e}"))?;
        let pipeline_short_keys = device
            .new_compute_pipeline_state_with_function(&short_function)
            .map_err(|e| anyhow!("failed to create short-key compute pipeline: {e}"))?;

        let queue = device.new_command_queue();
        // Both pipelines may have different hardware limits, so use the lower
        // value to keep a single `threadgroup_width` valid for either kernel.
        let max_threads = (pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(pipeline_short_keys.max_total_threads_per_threadgroup() as usize);
        let threadgroup_width = 256usize.min(max_threads.max(1));

        let message_length =
            u32::try_from(signing_input.len()).context("JWT signing input too long")?;
        let options = MTLResourceOptions::StorageModeShared;

        let params_buf =
            device.new_buffer(std::mem::size_of::<Hs512BruteForceParams>() as u64, options);
        let msg_buf = device.new_buffer(signing_input.len().max(1) as u64, options);
        // The JWT signing input is constant across the entire run, so upload it
        // once and reuse the same shared buffer for every dispatch.
        copy_bytes_to_buffer(&msg_buf, signing_input);

        let result_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, options);
        copy_value_to_buffer(&result_buf, &RESULT_NOT_FOUND_SENTINEL);

        Ok(Self {
            device,
            queue,
            pipeline_mixed,
            pipeline_short_keys,
            params_buf,
            msg_buf,
            result_buf,
            message_length,
            threadgroup_width,
        })
    }

    // Select the specialized short-key kernel when every candidate in the
    // dispatch view is <= 128 bytes; otherwise fall back to the mixed kernel.
    // `DispatchBatchView` lets autotune reuse this logic on a sampled prefix.
    //
    // Learning note: the 128-byte threshold matches the HMAC-SHA512 block size.
    // Keys <= 128 bytes skip the hash-the-key-first branch in the kernel,
    // saving an entire SHA-512 computation per candidate.
    fn active_pipeline_for_view(
        &self,
        batch: DispatchBatchView<'_>,
    ) -> &metal::ComputePipelineState {
        if batch.max_word_len <= 128 {
            &self.pipeline_short_keys
        } else {
            &self.pipeline_mixed
        }
    }

    // Validate and apply a user-provided threadgroup width override.
    pub(super) fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()> {
        let max_threads = self.max_total_threads_per_threadgroup();
        if requested == 0 {
            bail!("--threads-per-group must be > 0");
        }
        if requested > max_threads {
            bail!(
                "--threads-per-group {} exceeds pipeline max {}",
                requested,
                max_threads
            );
        }
        self.threadgroup_width = requested;
        Ok(())
    }

    // Accessor used by logging and periodic rate reports.
    pub(super) fn current_threadgroup_width(&self) -> usize {
        self.threadgroup_width
    }

    // Conservative execution width used for autotune candidate generation.
    pub(super) fn thread_execution_width(&self) -> usize {
        (self.pipeline_mixed.thread_execution_width() as usize)
            .min(self.pipeline_short_keys.thread_execution_width() as usize)
    }

    // Conservative threadgroup cap valid for both kernels.
    pub(super) fn max_total_threads_per_threadgroup(&self) -> usize {
        (self.pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(self.pipeline_short_keys.max_total_threads_per_threadgroup() as usize)
    }

    // Human-readable device name for startup logging.
    pub(super) fn device_name(&self) -> &str {
        self.device.name()
    }

    // Encode, commit, and return immediately without waiting for GPU completion.
    // The returned `CommandBuffer` must be passed to `wait_and_readback` before
    // the next dispatch (params_buf/result_buf are shared across dispatches).
    pub(super) fn encode_and_commit(
        &self,
        target_signature: [u8; 64],
        batch: &WordBatch,
    ) -> anyhow::Result<(metal::CommandBuffer, Duration, Duration)> {
        self.encode_and_commit_view(
            target_signature,
            batch.as_dispatch_view(),
            self.threadgroup_width,
        )
    }

    fn encode_and_commit_view(
        &self,
        target_signature: [u8; 64],
        batch: DispatchBatchView<'_>,
        threadgroup_width: usize,
    ) -> anyhow::Result<(metal::CommandBuffer, Duration, Duration)> {
        let candidate_count =
            u32::try_from(batch.candidate_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            bail!("cannot encode empty batch for async dispatch");
        }

        // Optimization: convert [u8; 64] -> [u64; 8] big-endian on the host so
        // the kernel compares word-against-word without per-thread load_be_u64.
        //
        // Learning note: for HS512 we fill all 8 words (the full 64-byte digest).
        // Compare with HS384, which fills only 6 words and zeroes the rest.
        // The loop range (0..8 vs 0..6) is the only substantive difference
        // between the HS384 and HS512 versions of this function.
        let mut target_words = [0u64; 8];
        for i in 0..8 {
            let off = i * 8;
            target_words[i] = u64::from_be_bytes([
                target_signature[off],
                target_signature[off + 1],
                target_signature[off + 2],
                target_signature[off + 3],
                target_signature[off + 4],
                target_signature[off + 5],
                target_signature[off + 6],
                target_signature[off + 7],
            ]);
        }
        let params = Hs512BruteForceParams {
            target_signature: target_words,
            message_length: self.message_length,
            candidate_count,
        };

        let prep_started = Instant::now();
        copy_value_to_buffer(&self.params_buf, &params);
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        let host_prep = prep_started.elapsed();

        let encode_started = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = self.active_pipeline_for_view(batch);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.params_buf), 0);
        encoder.set_buffer(1, Some(&self.msg_buf), 0);
        encoder.set_buffer(2, Some(batch.word_bytes_buf), 0);
        encoder.set_buffer(3, Some(batch.word_offsets_buf), 0);
        encoder.set_buffer(4, Some(batch.word_lengths_buf), 0);
        encoder.set_buffer(5, Some(&self.result_buf), 0);

        let threads_per_group = MTLSize::new(threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(candidate_count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
        let command_encode = encode_started.elapsed();

        command_buffer.commit();

        // Retain the command buffer so it survives past the autorelease pool.
        let owned = command_buffer.to_owned();
        Ok((owned, host_prep, command_encode))
    }

    // Block until a previously committed command buffer completes, then read
    // the match result from shared memory.
    pub(super) fn wait_and_readback(
        &self,
        cmd_buf: &metal::CommandBufferRef,
    ) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        cmd_buf.wait_until_completed();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        // SAFETY: `result_buf` is a 4-byte shared buffer; GPU has completed.
        let result = unsafe { *result_ptr };
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    // Synchronous dispatch: encode, commit, wait, and readback in one call.
    // Used by autotune and test paths where pipelining is not needed.
    fn dispatch_batch_view(
        &self,
        target_signature: [u8; 64],
        batch: DispatchBatchView<'_>,
        threadgroup_width: usize,
    ) -> anyhow::Result<(Option<u32>, BatchDispatchTimings)> {
        let started = Instant::now();
        let mut timings = BatchDispatchTimings::default();
        let candidate_count =
            u32::try_from(batch.candidate_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            timings.total = started.elapsed();
            return Ok((None, timings));
        }

        let (cmd_buf, host_prep, command_encode) =
            self.encode_and_commit_view(target_signature, batch, threadgroup_width)?;
        timings.host_prep = host_prep;
        timings.command_encode = command_encode;

        let (maybe_match, gpu_wait, result_readback) = self.wait_and_readback(&cmd_buf);
        timings.gpu_wait = gpu_wait;
        timings.result_readback = result_readback;
        timings.total = started.elapsed();
        Ok((maybe_match, timings))
    }

    // Benchmark a small sample from the first batch across several candidate
    // threadgroup widths, then keep the fastest width for the rest of the run.
    //
    // We optimize for `gpu_wait` (not end-to-end time) because autotune is only
    // choosing a GPU execution parameter, not host parsing behavior.
    pub(super) fn autotune_threadgroup_width(
        &mut self,
        target_signature: [u8; 64],
        batch: &WordBatch,
    ) -> anyhow::Result<()> {
        let sample_count = batch.candidate_count().min(16_384);
        if sample_count == 0 {
            return Ok(());
        }
        let sample_view = batch
            .prefix_dispatch_view(sample_count)
            .ok_or_else(|| anyhow!("failed to build autotune sample view"))?;

        let tew = self.thread_execution_width();
        let max_threads = self.max_total_threads_per_threadgroup();
        let mut candidates = vec![
            tew,
            tew.saturating_mul(2),
            tew.saturating_mul(4),
            tew.saturating_mul(8),
            64,
            128,
            256,
            512,
            1024,
        ];
        candidates.retain(|&v| v > 0 && v <= max_threads);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(());
        }

        eprintln!(
            "AUTOTUNE: benchmarking {} candidates across {} threadgroup widths",
            sample_view.candidate_count,
            candidates.len()
        );

        // Compare widths using the same sampled prefix view for an apples-to-
        // apples GPU-only throughput measurement (same data, only width changes).
        let mut best: Option<(usize, f64)> = None;
        for width in candidates {
            let (_result, timings) =
                self.dispatch_batch_view(target_signature, sample_view, width)?;
            if timings.gpu_wait.is_zero() {
                continue;
            }
            let rate = sample_view.candidate_count as f64 / timings.gpu_wait.as_secs_f64();
            match best {
                Some((_, best_rate)) if rate <= best_rate => {}
                _ => best = Some((width, rate)),
            }
        }

        if let Some((best_width, best_rate)) = best {
            self.threadgroup_width = best_width;
            eprintln!(
                "AUTOTUNE: selected --threads-per-group {} ({}/s sample)",
                best_width,
                format_human_count(best_rate)
            );
        }
        Ok(())
    }
}

// Reinterpret a plain-old-data value as bytes for host-to-GPU uploads.
// Callers are responsible for using types whose in-memory layout is the same as
// the corresponding Metal-side struct/bytes they expect to upload.
fn bytes_of<T>(value: &T) -> &[u8] {
    // SAFETY: `value` is a valid pointer for `size_of::<T>()` bytes for the duration of the borrow.
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hs512_params_round_trip_size_matches() {
        let params = Hs512BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs512BruteForceParams>());
    }
}
