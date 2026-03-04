//! # Metal GPU backend for HMAC-SHA brute-force dispatch
//!
//! This module implements the macOS Metal compute backend. It compiles
//! embedded `.metal` shader sources at runtime, manages GPU buffer
//! allocations for the host↔GPU data contract, and drives the
//! encode → commit → wait → readback lifecycle for each batch of
//! password candidates.
//!
//! ## Two-pipeline strategy
//!
//! Each HMAC variant exposes two kernel entry points:
//!
//! - **`mixed`** — handles keys of any length. The shader performs both
//!   the inner and outer HMAC passes with full-length key support,
//!   including the extra SHA block required when key bytes exceed the
//!   hash block size (64 bytes for SHA-256, 128 bytes for SHA-384/512).
//!
//! - **`short_keys`** — optimised fast path for keys that fit within a
//!   single SHA block. Skips the key-hashing pre-step and saves one
//!   SHA compression round per candidate. When an entire batch consists
//!   of short keys (the common case for dictionary attacks), this
//!   pipeline is selected automatically.
//!
//! ## Memory model: `StorageModeShared`
//!
//! All buffers are allocated with `StorageModeShared`, which means the
//! CPU and GPU share the same physical memory (unified memory on Apple
//! Silicon). There is no explicit copy between host and device — writes
//! from the CPU are visible to the GPU after command buffer commit, and
//! GPU writes are visible to the CPU after `wait_until_completed()`.
//!
//! ## Safety overview
//!
//! This module contains three categories of `unsafe` code:
//!
//! 1. **`copy_bytes_to_buffer` / `copy_value_to_buffer`** — raw pointer
//!    writes into Metal shared buffers. Safe because buffer sizes are
//!    validated at allocation time and `debug_assert!` guards against
//!    overflow.
//!
//! 2. **`bytes_of`** — reinterprets a `#[repr(C)]` struct as a byte
//!    slice for bulk-copying into a Metal buffer. Safe because the
//!    structs are `Copy` + `repr(C)` with no padding or pointer fields.
//!
//! 3. **`wait_and_readback_impl`** — reads a `u32` from the result
//!    buffer after GPU completion. Safe because the buffer is exactly
//!    4 bytes, `StorageModeShared`, and the GPU has finished writing
//!    (guaranteed by `wait_until_completed()`).

use anyhow::{Context, anyhow, bail};
use metal::{CompileOptions, MTLSize};
use std::time::{Duration, Instant};

use crate::batch::{DispatchBatchView, WordBatch};
use crate::stats::{BatchDispatchTimings, format_human_count};

use super::{GpuBruteForcer, GpuCommandHandle, GpuDevice, HmacVariant, default_device};

// ---------------------------------------------------------------------------
// Embedded shader sources
//
// `include_str!` embeds the `.metal` source as a `&'static str` in the
// binary at compile time. The Metal runtime compiles it to GPU machine
// code on first use via `new_library_with_source`. This adds ~200ms to
// startup but avoids shipping pre-compiled metallib bundles.
//
// HS384 reuses the HS512 shader source — the same file contains entry
// points for both (hs384_wordlist, hs512_wordlist) since SHA-384 is a
// truncated SHA-512 with different initial hash values.
// ---------------------------------------------------------------------------

const METAL_SOURCE_HS256: &str = include_str!("hs256_wordlist.metal");

const METAL_SOURCE_HS512: &str = include_str!("hs512_wordlist.metal");

// ---------------------------------------------------------------------------
// Sentinel value for "no match in this batch"
//
// The GPU shader writes the batch-local index of a matching candidate
// into the result buffer. If no candidate matches, the buffer retains
// this sentinel value. `u32::MAX` is safe because the maximum batch
// size is bounded well below 2^32 candidates.
// ---------------------------------------------------------------------------

const RESULT_NOT_FOUND_SENTINEL: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Metal buffer binding indices
//
// These constants map to the `[[buffer(N)]]` attribute indices declared
// in the .metal shader signatures. Both the host (Rust) and device
// (Metal) sides must agree on which slot carries which data. Changing
// a value here without updating the corresponding shader will silently
// bind the wrong buffer — the GPU will read garbage.
//
// Slot layout (identical for all HMAC variants):
//   0: params struct (target signature + message length + candidate count)
//   1: JWT signing input bytes (header.payload, without the signature)
//   2: raw candidate password bytes (packed contiguously)
//   3: per-candidate byte offsets into the word_bytes buffer
//   4: per-candidate byte lengths
//   5: result (single u32: matching index or RESULT_NOT_FOUND_SENTINEL)
// ---------------------------------------------------------------------------

const BUF_PARAMS: u64 = 0;
const BUF_MESSAGE: u64 = 1;
const BUF_WORD_BYTES: u64 = 2;
const BUF_WORD_OFFSETS: u64 = 3;
const BUF_WORD_LENGTHS: u64 = 4;
const BUF_RESULT: u64 = 5;

// ---------------------------------------------------------------------------
// Host → GPU parameter blocks
//
// These structs are copied byte-for-byte into Metal shared buffers and
// read directly by the GPU shader. `#[repr(C)]` is required to guarantee
// a deterministic field layout that matches the corresponding Metal
// struct definition. Without it, the Rust compiler is free to reorder
// or pad fields, which would cause the GPU to misinterpret the data.
//
// Both structs use fixed-size arrays for `target_signature` rather than
// slices or pointers because GPU shaders cannot follow host pointers —
// all data must be inline in the buffer.
//
// `Hs512BruteForceParams` is shared between HS384 and HS512. For HS384,
// only the first 6 of the 8 `u64` words are meaningful (48 bytes / 8 =
// 6 words); the remaining 2 words are zeroed and ignored by the shader.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs256BruteForceParams {
    target_signature: [u32; 8],
    message_length: u32,
    candidate_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs512BruteForceParams {
    target_signature: [u64; 8],
    message_length: u32,
    candidate_count: u32,
}

// ---------------------------------------------------------------------------
// Shared buffer-write helpers
// ---------------------------------------------------------------------------

/// Copy raw bytes into a Metal shared buffer.
///
/// The caller must ensure `data.len() <= buffer.length()`. This is
/// enforced by a `debug_assert!` in debug builds but unchecked in
/// release for performance (these are called on every GPU dispatch).
fn copy_bytes_to_buffer(buffer: &metal::Buffer, data: &[u8]) {
    debug_assert!(buffer.length() as usize >= data.len());
    if data.is_empty() {
        return;
    }
    // SAFETY: `buffer.contents()` returns a valid pointer to the
    // buffer's shared memory region, which is at least `buffer.length()`
    // bytes. The `debug_assert!` above guarantees `data.len()` does not
    // exceed this. The buffer is `StorageModeShared`, so CPU writes are
    // permitted. No concurrent GPU access occurs here because the
    // command buffer has not been committed yet.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast::<u8>(), data.len());
    }
}

/// Copy a `#[repr(C)]` value into a Metal shared buffer by
/// reinterpreting it as raw bytes.
fn copy_value_to_buffer<T>(buffer: &metal::Buffer, value: &T) {
    copy_bytes_to_buffer(buffer, bytes_of(value));
}

/// Reinterpret a `#[repr(C)]` struct as a byte slice.
///
/// This is the moral equivalent of `std::mem::transmute` to `&[u8]`
/// but returns a slice of exactly `size_of::<T>()` bytes.
fn bytes_of<T>(value: &T) -> &[u8] {
    // SAFETY: `T` is always one of our `#[repr(C)]` param structs,
    // which are `Copy`, contain no pointers or padding-sensitive
    // fields, and have a fully deterministic layout. The resulting
    // slice borrows `value` for `'_`, preventing use-after-free.
    // The byte count is exactly `size_of::<T>()`, matching the
    // struct's in-memory representation.
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

// ---------------------------------------------------------------------------
// MetalBruteForcer
// ---------------------------------------------------------------------------

/// Metal compute backend for HMAC-SHA brute-force cracking.
///
/// Owns the GPU device, command queue, compiled pipelines, and the
/// persistent buffers (params, message, result) that are reused across
/// dispatches. Per-batch candidate data (word bytes, offsets, lengths)
/// is supplied externally via `WordBatch` buffers managed by the
/// producer pipeline.
///
/// A single `MetalBruteForcer` is created per cracking run and handles
/// all dispatches for one HMAC variant. It is not `Send` or `Sync` —
/// Metal command encoding must happen on the thread that created the
/// command queue.
pub(crate) struct MetalBruteForcer {
    variant: HmacVariant,
    device: metal::Device,
    queue: metal::CommandQueue,
    /// Pipeline for candidates of any key length (general path).
    pipeline_mixed: metal::ComputePipelineState,
    /// Pipeline optimised for keys shorter than the hash block size.
    pipeline_short_keys: metal::ComputePipelineState,
    /// Persistent buffer holding the `Hs{256,512}BruteForceParams` struct.
    params_buf: metal::Buffer,
    /// Persistent buffer holding the JWT signing input (header.payload).
    msg_buf: metal::Buffer,
    /// Persistent 4-byte buffer for the GPU to write the match result.
    result_buf: metal::Buffer,
    /// Cached length of the JWT signing input (avoids repeated conversion).
    message_length: u32,
    /// Number of threads per threadgroup for compute dispatch. Starts
    /// at min(256, hardware max) and may be adjusted by autotune.
    threadgroup_width: usize,
}

impl MetalBruteForcer {
    /// Compile the Metal shader, create pipelines, and allocate
    /// persistent buffers for the given HMAC variant and JWT signing input.
    pub(crate) fn new(variant: HmacVariant, signing_input: &[u8]) -> anyhow::Result<Self> {
        let device = default_device()?;
        let compile_options = CompileOptions::new();

        // Select the shader source and kernel entry point names.
        // HS384 and HS512 share the same source file but use different
        // entry points (the shader preprocessor emits both).
        let (source, fn_mixed, fn_short) = match variant {
            HmacVariant::Hs256 => (
                METAL_SOURCE_HS256,
                "hs256_wordlist",
                "hs256_wordlist_short_keys",
            ),
            HmacVariant::Hs384 => (
                METAL_SOURCE_HS512,
                "hs384_wordlist",
                "hs384_wordlist_short_keys",
            ),
            HmacVariant::Hs512 => (
                METAL_SOURCE_HS512,
                "hs512_wordlist",
                "hs512_wordlist_short_keys",
            ),
        };

        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| anyhow!("failed to compile Metal kernel: {e}"))?;

        let mixed_function = library
            .get_function(fn_mixed, None)
            .map_err(|e| anyhow!("failed to get Metal function {fn_mixed}: {e}"))?;
        let short_function = library
            .get_function(fn_short, None)
            .map_err(|e| anyhow!("failed to get Metal function {fn_short}: {e}"))?;

        let pipeline_mixed = device
            .new_compute_pipeline_state_with_function(&mixed_function)
            .map_err(|e| anyhow!("failed to create mixed compute pipeline: {e}"))?;
        let pipeline_short_keys = device
            .new_compute_pipeline_state_with_function(&short_function)
            .map_err(|e| anyhow!("failed to create short-key compute pipeline: {e}"))?;

        let queue = device.new_command_queue();

        // Use the more conservative (smaller) maximum of the two
        // pipelines to ensure the threadgroup width is valid for both.
        let max_threads = (pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(pipeline_short_keys.max_total_threads_per_threadgroup() as usize);
        let threadgroup_width = 256usize.min(max_threads.max(1));

        let message_length =
            u32::try_from(signing_input.len()).context("JWT signing input too long")?;

        let options = metal::MTLResourceOptions::StorageModeShared;

        // Allocate the params buffer sized for the variant's struct.
        // HS384 uses the HS512 struct (same layout, fewer meaningful words).
        let params_size = match variant {
            HmacVariant::Hs256 => std::mem::size_of::<Hs256BruteForceParams>(),
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                std::mem::size_of::<Hs512BruteForceParams>()
            }
        };
        let params_buf = device.new_buffer(params_size as u64, options);

        // Message buffer holds the JWT signing input for the entire run.
        // `.max(1)` avoids a zero-length Metal buffer (which is invalid).
        let msg_buf = device.new_buffer(signing_input.len().max(1) as u64, options);
        copy_bytes_to_buffer(&msg_buf, signing_input);

        // Result buffer: single u32. Initialised to the sentinel so a
        // readback before any dispatch returns "not found" rather than
        // uninitialised memory.
        let result_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, options);
        copy_value_to_buffer(&result_buf, &RESULT_NOT_FOUND_SENTINEL);

        Ok(Self {
            variant,
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

    // -----------------------------------------------------------------------
    // Pipeline selection
    // -----------------------------------------------------------------------

    /// The byte length threshold below which all candidates in a batch
    /// qualify for the short-key fast path. Matches the SHA block size
    /// for the variant (64 for SHA-256, 128 for SHA-384/512).
    fn short_key_threshold(&self) -> u16 {
        match self.variant {
            HmacVariant::Hs256 => 64,
            HmacVariant::Hs384 | HmacVariant::Hs512 => 128,
        }
    }

    /// Choose the compute pipeline based on the batch's longest
    /// candidate. If every candidate fits within the short-key
    /// threshold, the optimised pipeline is used.
    fn active_pipeline_for_view(
        &self,
        batch: DispatchBatchView<'_>,
    ) -> &metal::ComputePipelineState {
        if batch.max_word_len <= self.short_key_threshold() {
            &self.pipeline_short_keys
        } else {
            &self.pipeline_mixed
        }
    }

    // -----------------------------------------------------------------------
    // Params preparation
    // -----------------------------------------------------------------------

    /// Convert the target HMAC signature from raw bytes into the
    /// variant-specific params struct and copy it into the GPU buffer.
    ///
    /// HS256 signatures are 32 bytes → 8 × u32 (big-endian).
    /// HS384 signatures are 48 bytes → 6 × u64 (padded to 8 × u64).
    /// HS512 signatures are 64 bytes → 8 × u64 (big-endian).
    fn write_params(&self, target_signature: &[u8], candidate_count: u32) -> anyhow::Result<()> {
        let expected_len = self.variant.signature_len();
        if target_signature.len() != expected_len {
            bail!(
                "{} signature must be {} bytes, got {}",
                self.variant.label(),
                expected_len,
                target_signature.len()
            );
        }

        match self.variant {
            HmacVariant::Hs256 => {
                let mut target_words = [0u32; 8];
                for i in 0..8 {
                    let off = i * 4;
                    target_words[i] =
                        u32::from_be_bytes(target_signature[off..off + 4].try_into().unwrap());
                }
                let params = Hs256BruteForceParams {
                    target_signature: target_words,
                    message_length: self.message_length,
                    candidate_count,
                };
                copy_value_to_buffer(&self.params_buf, &params);
            }
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                let word_count = expected_len / 8;
                let mut target_words = [0u64; 8];
                for i in 0..word_count {
                    let off = i * 8;
                    target_words[i] =
                        u64::from_be_bytes(target_signature[off..off + 8].try_into().unwrap());
                }
                let params = Hs512BruteForceParams {
                    target_signature: target_words,
                    message_length: self.message_length,
                    candidate_count,
                };
                copy_value_to_buffer(&self.params_buf, &params);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Core dispatch helpers
    // -----------------------------------------------------------------------

    /// Encode a compute command for the given batch and commit it to
    /// the GPU. Returns the command buffer handle (for later
    /// `wait_until_completed`) plus host-side timing breakdowns.
    ///
    /// This is the async dispatch path: after `commit()`, the GPU
    /// executes independently while the CPU can prepare the next batch.
    fn encode_and_commit_view(
        &self,
        target_signature: &[u8],
        batch: DispatchBatchView<'_>,
        threadgroup_width: usize,
    ) -> anyhow::Result<(metal::CommandBuffer, Duration, Duration)> {
        let candidate_count =
            u32::try_from(batch.candidate_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            bail!("cannot encode empty batch for async dispatch");
        }

        // Phase 1: Host prep — write params and reset the result sentinel.
        let prep_started = Instant::now();
        self.write_params(target_signature, candidate_count)?;
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        let host_prep = prep_started.elapsed();

        // Phase 2: Command encoding — record the compute dispatch into
        // a command buffer. No GPU work happens yet; this is CPU-only
        // Metal API bookkeeping.
        let encode_started = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = self.active_pipeline_for_view(batch);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(BUF_PARAMS, Some(&self.params_buf), 0);
        encoder.set_buffer(BUF_MESSAGE, Some(&self.msg_buf), 0);
        encoder.set_buffer(BUF_WORD_BYTES, Some(batch.word_bytes_buf), 0);
        encoder.set_buffer(BUF_WORD_OFFSETS, Some(batch.word_offsets_buf), 0);
        encoder.set_buffer(BUF_WORD_LENGTHS, Some(batch.word_lengths_buf), 0);
        encoder.set_buffer(BUF_RESULT, Some(&self.result_buf), 0);

        // `dispatch_threads` uses Metal's non-uniform threadgroup API:
        // the runtime automatically handles the case where the grid
        // size is not evenly divisible by the threadgroup width.
        let threads_per_group = MTLSize::new(threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(candidate_count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
        let command_encode = encode_started.elapsed();

        // `commit()` submits the command buffer to the GPU. After this
        // point the GPU may begin execution at any time.
        command_buffer.commit();

        let owned = command_buffer.to_owned();
        Ok((owned, host_prep, command_encode))
    }

    /// Block until GPU execution completes, then read the result buffer.
    ///
    /// Returns `Some(batch_local_index)` if the GPU found a matching
    /// candidate, `None` if no match was found in this batch.
    fn wait_and_readback_impl(
        &self,
        cmd_buf: &metal::CommandBufferRef,
    ) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        cmd_buf.wait_until_completed();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        // SAFETY: `result_buf` is a 4-byte `StorageModeShared` buffer
        // allocated in `new()`. `wait_until_completed()` guarantees the
        // GPU has finished all writes. The pointer is valid, aligned
        // (Metal buffers are page-aligned), and the buffer outlives
        // this read (owned by `self`). No concurrent writes can occur
        // because we hold `&self` and no new command buffer referencing
        // `result_buf` has been committed.
        let result = unsafe { *result_ptr };
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    /// Synchronous dispatch: encode, commit, wait, and read back in one
    /// call. Used only by `autotune_threadgroup_width` where we need
    /// the result immediately to compare throughput across widths.
    fn dispatch_batch_view(
        &self,
        target_signature: &[u8],
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

        let (maybe_match, gpu_wait, result_readback) = self.wait_and_readback_impl(&cmd_buf);
        timings.gpu_wait = gpu_wait;
        timings.result_readback = result_readback;
        timings.total = started.elapsed();
        Ok((maybe_match, timings))
    }
}

// ---------------------------------------------------------------------------
// GpuBruteForcer trait implementation
//
// This bridges the platform-agnostic trait (defined in backend.rs) to
// the Metal-specific implementation above. The trait uses `WordBatch`
// (which owns the candidate buffers) while the internal methods use
// `DispatchBatchView` (a borrowed, zero-copy view into those buffers).
// ---------------------------------------------------------------------------

impl GpuBruteForcer for MetalBruteForcer {
    fn device(&self) -> &GpuDevice {
        &self.device
    }

    fn device_name(&self) -> &str {
        self.device.name()
    }

    fn thread_execution_width(&self) -> usize {
        (self.pipeline_mixed.thread_execution_width() as usize)
            .min(self.pipeline_short_keys.thread_execution_width() as usize)
    }

    fn max_total_threads_per_threadgroup(&self) -> usize {
        (self.pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(self.pipeline_short_keys.max_total_threads_per_threadgroup() as usize)
    }

    fn current_threadgroup_width(&self) -> usize {
        self.threadgroup_width
    }

    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()> {
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

    fn encode_and_commit(
        &self,
        target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)> {
        self.encode_and_commit_view(
            target_signature,
            batch.as_dispatch_view(),
            self.threadgroup_width,
        )
    }

    fn wait_and_readback(
        &self,
        handle: &GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration) {
        self.wait_and_readback_impl(handle)
    }

    /// Benchmark a range of threadgroup widths on a small sample and
    /// keep the fastest.
    ///
    /// The candidate widths are derived from the hardware's thread
    /// execution width (SIMD lane count) and common power-of-two sizes.
    /// Each width is timed on the same sample batch; the one with the
    /// highest candidates/second rate wins.
    ///
    /// This runs synchronously (blocking on each dispatch) because it
    /// only processes a small sample (≤16,384 candidates) and the
    /// timing accuracy requires no concurrent GPU work.
    fn autotune_threadgroup_width(
        &mut self,
        target_signature: &[u8],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hs256_params_round_trip_size_matches() {
        let params = Hs256BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs256BruteForceParams>());
    }

    #[test]
    fn hs512_params_round_trip_size_matches() {
        let params = Hs512BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs512BruteForceParams>());
    }
}
