//! # Metal GPU backend for brute-force dispatch
//!
//! This module implements the macOS Metal compute backend. It compiles
//! embedded `.metal` shader sources at runtime, manages GPU buffer
//! allocations for the host↔GPU data contract, and drives the
//! encode → commit → wait → readback lifecycle for each batch of
//! password candidates.
//!
//! ## Two compute backends
//!
//! - **`MetalBruteForcer`** — HMAC-SHA JWT cracking (HS256/384/512).
//!   Uses GPU-optimised software SHA implementations.
//!
//! - **`MetalAesKwBruteForcer`** — JWE AES Key Wrap cracking (A128KW,
//!   A192KW, A256KW) via software AES in a Metal compute shader.
//!   The shader is compiled three times with different `AES_KEY_BYTES`
//!   defines, producing specialised kernels for each key size.
//!   Trades per-thread cost (software AES vs CPU hardware AESD)
//!   for massive GPU parallelism.
//!
//! Both implement the `GpuBruteForcer` trait so the runner can use
//! a single generic dispatch loop.
//!
//! ## Two-pipeline strategy
//!
//! Each backend exposes two kernel entry points:
//!
//! - **`mixed`** — handles keys of any length. For HMAC, this includes
//!   keys exceeding the hash block size. For A128KW, this includes
//!   candidates > 16 bytes (SHA-256 hashed to derive the AES key).
//!
//! - **`short_keys`** — optimised fast path for short keys. For HMAC,
//!   keys within the SHA block size. For AES-KW, candidates ≤ the AES
//!   key size (16/24/32 bytes) are zero-padded directly (no SHA-256
//!   needed). This is the common case for dictionary attacks.
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
//!    validated at allocation time and `assert!` guards against
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

use super::{AesKwVariant, GpuBruteForcer, GpuCommandHandle, GpuDevice, HmacVariant, default_device};

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

const METAL_SOURCE_AESKW: &str = include_str!("aeskw_wordlist.metal");

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

/// Host → GPU parameters for the AES Key Wrap shader (A128KW/A192KW/A256KW).
///
/// Matches the Metal struct `AesKwBruteForceParams` in
/// `aeskw_wordlist.metal`. The AES key size is baked in at shader
/// compile time via `#define AES_KEY_BYTES`, so it does not appear
/// here. The encrypted key itself is passed via a separate buffer
/// (slot 1) rather than inlined here, because its length varies
/// (24–72 bytes for standard JWE `enc` algorithms).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AesKwBruteForceParams {
    /// Byte length of the encrypted_key in buffer(1).
    encrypted_key_len: u32,
    /// Number of 64-bit blocks in the CEK (n ≥ 2).
    n_blocks: u32,
    /// Number of candidate secrets in this batch.
    candidate_count: u32,
}

// ---------------------------------------------------------------------------
// Shared buffer-write helpers
// ---------------------------------------------------------------------------

/// Copy raw bytes into a Metal shared buffer.
///
/// The caller must ensure `data.len() <= buffer.length()`. This is
/// enforced by an `assert!` that is active in all builds (including
/// release). An earlier version used `debug_assert!`, which is compiled
/// out in release builds — meaning the only bounds check guarding an
/// `unsafe { copy_nonoverlapping }` into GPU-mapped memory was silently
/// removed in the binary users actually run. Since this function is
/// called once per GPU dispatch (not per candidate), the cost of a
/// release-mode bounds check is negligible: one integer comparison
/// amortised over millions of candidates.
fn copy_bytes_to_buffer(buffer: &metal::Buffer, data: &[u8]) {
    assert!(
        buffer.length() as usize >= data.len(),
        "copy_bytes_to_buffer: data length {} exceeds buffer length {}",
        data.len(),
        buffer.length()
    );
    if data.is_empty() {
        return;
    }
    // SAFETY: `buffer.contents()` returns a valid pointer to the
    // buffer's shared memory region, which is at least `buffer.length()`
    // bytes. The `assert!` above guarantees `data.len()` does not
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

// ---------------------------------------------------------------------------
// MetalAesKwBruteForcer
// ---------------------------------------------------------------------------

/// Metal compute backend for AES Key Wrap JWE cracking (A128KW/A192KW/A256KW).
///
/// Implements `GpuBruteForcer` using a software AES Metal compute shader.
/// The shader source (`aeskw_wordlist.metal`) is compiled with a
/// `#define AES_KEY_BYTES N` preprocessor directive that specialises it
/// for the target AES key size (16/24/32 bytes).  This produces fully
/// optimised kernels — the Metal compiler statically resolves loop bounds,
/// array sizes, and branch elimination at compile time.
///
/// The shader uses S-box lookup tables (512 bytes in `constant` address
/// space) and GF(2^8) arithmetic for InvMixColumns, trading per-thread
/// cost for massive GPU parallelism.
///
/// ## Buffer layout
///
/// Uses the same 6-slot binding convention as the HMAC backend:
///
/// | Slot | HMAC purpose          | AES-KW purpose                     |
/// |------|-----------------------|------------------------------------|
/// | 0    | params struct         | `AesKwBruteForceParams`            |
/// | 1    | JWT signing input     | encrypted_key bytes (pre-filled)   |
/// | 2    | candidate word bytes  | candidate word bytes               |
/// | 3    | candidate offsets     | candidate offsets                  |
/// | 4    | candidate lengths     | candidate lengths                  |
/// | 5    | result (atomic u32)   | result (atomic u32)                |
///
/// The encrypted_key is written once at construction and never changes
/// (unlike the HMAC backend where the signing input varies per token).
pub(crate) struct MetalAesKwBruteForcer {
    device: metal::Device,
    queue: metal::CommandQueue,
    /// Pipeline for candidates of any length (includes SHA-256 for long keys).
    pipeline_mixed: metal::ComputePipelineState,
    /// Pipeline optimised for candidates ≤ AES key size (zero-pad, no SHA-256).
    pipeline_short_keys: metal::ComputePipelineState,
    /// Persistent buffer holding `AesKwBruteForceParams`.
    params_buf: metal::Buffer,
    /// Persistent buffer holding the JWE encrypted_key bytes (slot 1).
    enc_key_buf: metal::Buffer,
    /// Persistent 4-byte buffer for the GPU to write the match result.
    result_buf: metal::Buffer,
    /// Which AES Key Wrap variant (determines short_key_threshold).
    variant: AesKwVariant,
    /// Byte length of the encrypted_key (stored for params writes).
    encrypted_key_len: u32,
    /// Number of 64-bit CEK blocks: `encrypted_key.len() / 8 - 1`.
    n_blocks: u32,
    /// Number of threads per threadgroup for compute dispatch.
    threadgroup_width: usize,
}

impl MetalAesKwBruteForcer {
    /// Compile the AES Key Wrap Metal shader for the given variant, create
    /// pipelines, and allocate persistent buffers.
    ///
    /// The shader source is prepended with `#define AES_KEY_BYTES N` to
    /// specialise it for the target AES key size.  Each variant gets its
    /// own compiled Metal library — there is no runtime branching in the
    /// shader.
    ///
    /// `encrypted_key` is the raw ciphertext from the JWE token's
    /// encrypted_key field (base64url-decoded). `n` is the number of
    /// 64-bit CEK blocks (`encrypted_key.len() / 8 - 1`).
    pub(crate) fn new(
        variant: AesKwVariant,
        encrypted_key: &[u8],
        n: usize,
    ) -> anyhow::Result<Self> {
        let device = default_device()?;
        let compile_options = CompileOptions::new();

        // Prepend the AES_KEY_BYTES define to specialise the shader.
        // This is simpler and more portable than using Metal's
        // CompileOptions preprocessor API.
        let specialised_source = format!(
            "#define AES_KEY_BYTES {}\n{}",
            variant.key_bytes(),
            METAL_SOURCE_AESKW
        );

        let label = variant.label();
        let library = device
            .new_library_with_source(&specialised_source, &compile_options)
            .map_err(|e| anyhow!("failed to compile {label} Metal kernel: {e}"))?;

        let mixed_function = library
            .get_function("aeskw_wordlist", None)
            .map_err(|e| anyhow!("failed to get Metal function aeskw_wordlist: {e}"))?;
        let short_function = library
            .get_function("aeskw_wordlist_short_keys", None)
            .map_err(|e| {
                anyhow!("failed to get Metal function aeskw_wordlist_short_keys: {e}")
            })?;

        let pipeline_mixed = device
            .new_compute_pipeline_state_with_function(&mixed_function)
            .map_err(|e| anyhow!("failed to create {label} mixed compute pipeline: {e}"))?;
        let pipeline_short_keys = device
            .new_compute_pipeline_state_with_function(&short_function)
            .map_err(|e| anyhow!("failed to create {label} short-key compute pipeline: {e}"))?;

        let queue = device.new_command_queue();

        let max_threads = (pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(pipeline_short_keys.max_total_threads_per_threadgroup() as usize);
        let threadgroup_width = 256usize.min(max_threads.max(1));

        let encrypted_key_len =
            u32::try_from(encrypted_key.len()).context("encrypted_key too long")?;
        let n_blocks = u32::try_from(n).context("n_blocks overflow")?;

        let options = metal::MTLResourceOptions::StorageModeShared;

        let params_buf = device.new_buffer(
            std::mem::size_of::<AesKwBruteForceParams>() as u64,
            options,
        );

        // Pre-fill the encrypted_key buffer — it stays constant for the
        // entire cracking run.
        let enc_key_buf = device.new_buffer(encrypted_key.len().max(1) as u64, options);
        copy_bytes_to_buffer(&enc_key_buf, encrypted_key);

        let result_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, options);
        copy_value_to_buffer(&result_buf, &RESULT_NOT_FOUND_SENTINEL);

        Ok(Self {
            device,
            queue,
            pipeline_mixed,
            pipeline_short_keys,
            params_buf,
            enc_key_buf,
            result_buf,
            variant,
            encrypted_key_len,
            n_blocks,
            threadgroup_width,
        })
    }

    // -----------------------------------------------------------------------
    // Pipeline selection
    // -----------------------------------------------------------------------

    /// The AES key size threshold for this variant. Candidates ≤ this size
    /// are zero-padded directly; longer candidates go through SHA-256 →
    /// truncate to key_bytes. Returns 16 for A128KW, 24 for A192KW,
    /// 32 for A256KW.
    fn short_key_threshold(&self) -> u16 {
        self.variant.key_bytes() as u16
    }

    /// Choose the compute pipeline based on the batch's longest candidate.
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
    // Core dispatch helpers
    // -----------------------------------------------------------------------

    /// Write the A128KW params struct into the GPU buffer.
    fn write_params(&self, candidate_count: u32) {
        let params = AesKwBruteForceParams {
            encrypted_key_len: self.encrypted_key_len,
            n_blocks: self.n_blocks,
            candidate_count,
        };
        copy_value_to_buffer(&self.params_buf, &params);
    }

    /// Encode and commit a compute dispatch for the given batch.
    fn encode_and_commit_view(
        &self,
        batch: DispatchBatchView<'_>,
        threadgroup_width: usize,
    ) -> anyhow::Result<(metal::CommandBuffer, Duration, Duration)> {
        let candidate_count =
            u32::try_from(batch.candidate_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            bail!("cannot encode empty batch for async dispatch");
        }

        let prep_started = Instant::now();
        self.write_params(candidate_count);
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        let host_prep = prep_started.elapsed();

        let encode_started = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = self.active_pipeline_for_view(batch);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(BUF_PARAMS, Some(&self.params_buf), 0);
        encoder.set_buffer(BUF_MESSAGE, Some(&self.enc_key_buf), 0);
        encoder.set_buffer(BUF_WORD_BYTES, Some(batch.word_bytes_buf), 0);
        encoder.set_buffer(BUF_WORD_OFFSETS, Some(batch.word_offsets_buf), 0);
        encoder.set_buffer(BUF_WORD_LENGTHS, Some(batch.word_lengths_buf), 0);
        encoder.set_buffer(BUF_RESULT, Some(&self.result_buf), 0);

        let threads_per_group = MTLSize::new(threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(candidate_count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
        let command_encode = encode_started.elapsed();

        command_buffer.commit();

        let owned = command_buffer.to_owned();
        Ok((owned, host_prep, command_encode))
    }

    /// Block until GPU execution completes, then read the result buffer.
    fn wait_and_readback_impl(
        &self,
        cmd_buf: &metal::CommandBufferRef,
    ) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        cmd_buf.wait_until_completed();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        // SAFETY: same justification as MetalBruteForcer::wait_and_readback_impl.
        let result = unsafe { *result_ptr };
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    /// Synchronous dispatch for autotune benchmarking.
    fn dispatch_batch_view(
        &self,
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
            self.encode_and_commit_view(batch, threadgroup_width)?;
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
// GpuBruteForcer trait implementation for MetalAesKwBruteForcer
//
// The `target_signature` parameter in `encode_and_commit` and
// `autotune_threadgroup_width` is **ignored** by this implementation.
// For HMAC backends, `target_signature` carries the JWT signature bytes.
// For AES-KW, the encrypted_key is pre-loaded into `enc_key_buf` at
// construction time and never changes. The parameter is accepted (but
// unused) to satisfy the trait contract, enabling the runner to use a
// single generic `run_gpu_crack<B: GpuBruteForcer>` dispatch loop.
// ---------------------------------------------------------------------------

impl GpuBruteForcer for MetalAesKwBruteForcer {
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
        _target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)> {
        // `_target_signature` is ignored — encrypted_key is in `enc_key_buf`.
        self.encode_and_commit_view(batch.as_dispatch_view(), self.threadgroup_width)
    }

    fn wait_and_readback(
        &self,
        handle: &GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration) {
        self.wait_and_readback_impl(handle)
    }

    fn autotune_threadgroup_width(
        &mut self,
        _target_signature: &[u8],
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
            let (_result, timings) = self.dispatch_batch_view(sample_view, width)?;
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

// ---------------------------------------------------------------------------
// MetalMarkovHmacBruteForcer
// ---------------------------------------------------------------------------

/// Metal compute backend for Markov chain + HMAC-SHA cracking.
///
/// Each GPU thread decodes a global keyspace index into per-position rank
/// selections, walks an order-1 Markov chain using a pre-trained lookup table,
/// and immediately computes HMAC-SHA on the generated candidate.
///
/// No `WordBatch` is used — the Markov table is uploaded once, and subsequent
/// dispatches only update the params struct (length, offset, count).
///
/// ## Buffer layout (Markov kernels)
///
/// | Slot | Purpose                                  |
/// |------|------------------------------------------|
/// | 0    | `Hs256MarkovParams` or `Hs512MarkovParams` |
/// | 1    | JWT signing input bytes                  |
/// | 2    | Markov lookup table (constant)           |
/// | 3    | result (atomic u32)                      |
pub(crate) struct MetalMarkovHmacBruteForcer {
    variant: HmacVariant,
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    params_buf: metal::Buffer,
    msg_buf: metal::Buffer,
    markov_table_buf: metal::Buffer,
    result_buf: metal::Buffer,
    message_length: u32,
    threadgroup_width: usize,
}

/// Markov buffer binding indices (different from the wordlist layout).
const MARKOV_BUF_PARAMS: u64 = 0;
const MARKOV_BUF_MESSAGE: u64 = 1;
const MARKOV_BUF_TABLE: u64 = 2;
const MARKOV_BUF_RESULT: u64 = 3;

/// Host → GPU parameters for Markov + HMAC-SHA256.
/// Must match the Metal struct `Hs256MarkovParams` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs256MarkovParams {
    target_signature: [u32; 8],
    message_length: u32,
    candidate_count: u32,
    pw_length: u32,
    threshold: u32,
    offset: u64,
}

/// Host → GPU parameters for Markov + HMAC-SHA384/512.
/// Must match the Metal struct `Hs512MarkovParams` field-for-field.
/// HS384 uses only the first 6 of 8 `u64` words (remaining zeroed).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs512MarkovParams {
    target_signature: [u64; 8],
    message_length: u32,
    candidate_count: u32,
    pw_length: u32,
    threshold: u32,
    offset: u64,
}

/// Host → GPU parameters for Markov + AES Key Wrap.
/// Must match the Metal struct `AesKwMarkovParams` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AesKwMarkovParams {
    encrypted_key_len: u32,
    n_blocks: u32,
    candidate_count: u32,
    pw_length: u32,
    threshold: u32,
    _pad: u32,
    offset: u64,
}

impl MetalMarkovHmacBruteForcer {
    /// Compile the Metal shader, create the Markov pipeline, and allocate
    /// persistent buffers.
    pub(crate) fn new(
        variant: HmacVariant,
        signing_input: &[u8],
        markov_table: &[u8],
    ) -> anyhow::Result<Self> {
        let device = default_device()?;
        let compile_options = CompileOptions::new();

        let (source, fn_name) = match variant {
            HmacVariant::Hs256 => (METAL_SOURCE_HS256, "hs256_markov"),
            HmacVariant::Hs384 => (METAL_SOURCE_HS512, "hs384_markov"),
            HmacVariant::Hs512 => (METAL_SOURCE_HS512, "hs512_markov"),
        };

        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| anyhow!("failed to compile Metal Markov kernel: {e}"))?;

        let function = library
            .get_function(fn_name, None)
            .map_err(|e| anyhow!("failed to get Metal function {fn_name}: {e}"))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow!("failed to create Markov compute pipeline: {e}"))?;

        let queue = device.new_command_queue();
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let threadgroup_width = 256usize.min(max_threads.max(1));

        let message_length =
            u32::try_from(signing_input.len()).context("JWT signing input too long")?;

        let options = metal::MTLResourceOptions::StorageModeShared;

        let params_size = match variant {
            HmacVariant::Hs256 => std::mem::size_of::<Hs256MarkovParams>(),
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                std::mem::size_of::<Hs512MarkovParams>()
            }
        };
        let params_buf = device.new_buffer(params_size as u64, options);

        let msg_buf = device.new_buffer(signing_input.len().max(1) as u64, options);
        copy_bytes_to_buffer(&msg_buf, signing_input);

        let markov_table_buf = device.new_buffer(markov_table.len().max(1) as u64, options);
        copy_bytes_to_buffer(&markov_table_buf, markov_table);

        let result_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, options);
        copy_value_to_buffer(&result_buf, &RESULT_NOT_FOUND_SENTINEL);

        Ok(Self {
            variant,
            device,
            queue,
            pipeline,
            params_buf,
            msg_buf,
            markov_table_buf,
            result_buf,
            message_length,
            threadgroup_width,
        })
    }

    /// Write Markov params into the GPU buffer.
    fn write_markov_params(
        &self,
        target_signature: &[u8],
        candidate_count: u32,
        pw_length: u32,
        threshold: u32,
        offset: u64,
    ) -> anyhow::Result<()> {
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
                let params = Hs256MarkovParams {
                    target_signature: target_words,
                    message_length: self.message_length,
                    candidate_count,
                    pw_length,
                    threshold,
                    offset,
                };
                copy_value_to_buffer(&self.params_buf, &params);
            }
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                let word_count = match self.variant {
                    HmacVariant::Hs384 => 6,
                    _ => 8,
                };
                let mut target_words = [0u64; 8];
                for i in 0..word_count {
                    let off = i * 8;
                    target_words[i] =
                        u64::from_be_bytes(target_signature[off..off + 8].try_into().unwrap());
                }
                let params = Hs512MarkovParams {
                    target_signature: target_words,
                    message_length: self.message_length,
                    candidate_count,
                    pw_length,
                    threshold,
                    offset,
                };
                copy_value_to_buffer(&self.params_buf, &params);
            }
        }
        Ok(())
    }
}

impl super::GpuMarkovBruteForcer for MetalMarkovHmacBruteForcer {
    fn device_name(&self) -> &str {
        self.device.name()
    }

    fn thread_execution_width(&self) -> usize {
        self.pipeline.thread_execution_width() as usize
    }

    fn max_total_threads_per_threadgroup(&self) -> usize {
        self.pipeline.max_total_threads_per_threadgroup() as usize
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

    fn encode_and_commit_markov(
        &self,
        target_data: &[u8],
        length: u32,
        threshold: u32,
        offset: u64,
        count: u32,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)> {
        if count == 0 {
            bail!("cannot encode empty Markov batch");
        }

        let prep_started = Instant::now();
        self.write_markov_params(target_data, count, length, threshold, offset)?;
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        let host_prep = prep_started.elapsed();

        let encode_started = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(MARKOV_BUF_PARAMS, Some(&self.params_buf), 0);
        encoder.set_buffer(MARKOV_BUF_MESSAGE, Some(&self.msg_buf), 0);
        encoder.set_buffer(MARKOV_BUF_TABLE, Some(&self.markov_table_buf), 0);
        encoder.set_buffer(MARKOV_BUF_RESULT, Some(&self.result_buf), 0);

        let threads_per_group = MTLSize::new(self.threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
        let command_encode = encode_started.elapsed();

        command_buffer.commit();
        let owned = command_buffer.to_owned();
        Ok((owned, host_prep, command_encode))
    }

    fn wait_and_readback(
        &self,
        handle: &GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        handle.wait_until_completed();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        let result = unsafe { *result_ptr };
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    fn autotune_markov(
        &mut self,
        target_data: &[u8],
        length: u32,
        threshold: u32,
    ) -> anyhow::Result<()> {
        const SAMPLE_COUNT: u32 = 16_384;

        let tew = self.thread_execution_width();
        let max_threads = self.max_total_threads_per_threadgroup();
        let mut candidates = vec![
            tew,
            tew.saturating_mul(2),
            tew.saturating_mul(4),
            tew.saturating_mul(8),
            64, 128, 256, 512, 1024,
        ];
        candidates.retain(|&v| v > 0 && v <= max_threads);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(());
        }

        eprintln!(
            "AUTOTUNE: benchmarking {} Markov candidates across {} threadgroup widths",
            SAMPLE_COUNT,
            candidates.len()
        );

        let mut best: Option<(usize, f64)> = None;
        for width in &candidates {
            self.threadgroup_width = *width;
            let (handle, _, _) =
                self.encode_and_commit_markov(target_data, length, threshold, 0, SAMPLE_COUNT)?;
            let (_, gpu_wait, _) = self.wait_and_readback(&handle);
            if gpu_wait.is_zero() {
                continue;
            }
            let rate = SAMPLE_COUNT as f64 / gpu_wait.as_secs_f64();
            match best {
                Some((_, best_rate)) if rate <= best_rate => {}
                _ => best = Some((*width, rate)),
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

// ---------------------------------------------------------------------------
// MetalMarkovAesKwBruteForcer
// ---------------------------------------------------------------------------

/// Metal compute backend for Markov chain + AES Key Wrap cracking.
///
/// Same fused-kernel approach as `MetalMarkovHmacBruteForcer` but for JWE
/// tokens using A128KW/A192KW/A256KW. The shader is compiled with
/// `#define AES_KEY_BYTES N` to specialise for each key size.
///
/// ## Buffer layout (Markov AES-KW kernels)
///
/// | Slot | Purpose                                  |
/// |------|------------------------------------------|
/// | 0    | `AesKwMarkovParams`                      |
/// | 1    | encrypted_key bytes (pre-filled)          |
/// | 2    | Markov lookup table (constant)           |
/// | 3    | result (atomic u32)                      |
pub(crate) struct MetalMarkovAesKwBruteForcer {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    params_buf: metal::Buffer,
    enc_key_buf: metal::Buffer,
    markov_table_buf: metal::Buffer,
    result_buf: metal::Buffer,
    encrypted_key_len: u32,
    n_blocks: u32,
    threadgroup_width: usize,
}

impl MetalMarkovAesKwBruteForcer {
    pub(crate) fn new(
        variant: AesKwVariant,
        encrypted_key: &[u8],
        n: usize,
        markov_table: &[u8],
    ) -> anyhow::Result<Self> {
        let device = default_device()?;
        let compile_options = CompileOptions::new();

        let specialised_source = format!(
            "#define AES_KEY_BYTES {}\n{}",
            variant.key_bytes(),
            METAL_SOURCE_AESKW
        );

        let label = variant.label();
        let library = device
            .new_library_with_source(&specialised_source, &compile_options)
            .map_err(|e| anyhow!("failed to compile {label} Markov Metal kernel: {e}"))?;

        let function = library
            .get_function("aeskw_markov", None)
            .map_err(|e| anyhow!("failed to get Metal function aeskw_markov: {e}"))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow!("failed to create {label} Markov compute pipeline: {e}"))?;

        let queue = device.new_command_queue();
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let threadgroup_width = 256usize.min(max_threads.max(1));

        let encrypted_key_len =
            u32::try_from(encrypted_key.len()).context("encrypted_key too long")?;
        let n_blocks = u32::try_from(n).context("n_blocks overflow")?;

        let options = metal::MTLResourceOptions::StorageModeShared;

        let params_buf = device.new_buffer(
            std::mem::size_of::<AesKwMarkovParams>() as u64,
            options,
        );

        let enc_key_buf = device.new_buffer(encrypted_key.len().max(1) as u64, options);
        copy_bytes_to_buffer(&enc_key_buf, encrypted_key);

        let markov_table_buf = device.new_buffer(markov_table.len().max(1) as u64, options);
        copy_bytes_to_buffer(&markov_table_buf, markov_table);

        let result_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, options);
        copy_value_to_buffer(&result_buf, &RESULT_NOT_FOUND_SENTINEL);

        Ok(Self {
            device,
            queue,
            pipeline,
            params_buf,
            enc_key_buf,
            markov_table_buf,
            result_buf,
            encrypted_key_len,
            n_blocks,
            threadgroup_width,
        })
    }
}

impl super::GpuMarkovBruteForcer for MetalMarkovAesKwBruteForcer {
    fn device_name(&self) -> &str {
        self.device.name()
    }

    fn thread_execution_width(&self) -> usize {
        self.pipeline.thread_execution_width() as usize
    }

    fn max_total_threads_per_threadgroup(&self) -> usize {
        self.pipeline.max_total_threads_per_threadgroup() as usize
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

    fn encode_and_commit_markov(
        &self,
        _target_data: &[u8],
        length: u32,
        threshold: u32,
        offset: u64,
        count: u32,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)> {
        if count == 0 {
            bail!("cannot encode empty Markov batch");
        }

        let prep_started = Instant::now();
        let params = AesKwMarkovParams {
            encrypted_key_len: self.encrypted_key_len,
            n_blocks: self.n_blocks,
            candidate_count: count,
            pw_length: length,
            threshold,
            _pad: 0,
            offset,
        };
        copy_value_to_buffer(&self.params_buf, &params);
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        let host_prep = prep_started.elapsed();

        let encode_started = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(MARKOV_BUF_PARAMS, Some(&self.params_buf), 0);
        encoder.set_buffer(MARKOV_BUF_MESSAGE, Some(&self.enc_key_buf), 0);
        encoder.set_buffer(MARKOV_BUF_TABLE, Some(&self.markov_table_buf), 0);
        encoder.set_buffer(MARKOV_BUF_RESULT, Some(&self.result_buf), 0);

        let threads_per_group = MTLSize::new(self.threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
        let command_encode = encode_started.elapsed();

        command_buffer.commit();
        let owned = command_buffer.to_owned();
        Ok((owned, host_prep, command_encode))
    }

    fn wait_and_readback(
        &self,
        handle: &GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        handle.wait_until_completed();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        let result = unsafe { *result_ptr };
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    fn autotune_markov(
        &mut self,
        _target_data: &[u8],
        length: u32,
        threshold: u32,
    ) -> anyhow::Result<()> {
        const SAMPLE_COUNT: u32 = 16_384;

        let tew = self.thread_execution_width();
        let max_threads = self.max_total_threads_per_threadgroup();
        let mut candidates = vec![
            tew,
            tew.saturating_mul(2),
            tew.saturating_mul(4),
            tew.saturating_mul(8),
            64, 128, 256, 512, 1024,
        ];
        candidates.retain(|&v| v > 0 && v <= max_threads);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(());
        }

        eprintln!(
            "AUTOTUNE: benchmarking {} Markov candidates across {} threadgroup widths",
            SAMPLE_COUNT,
            candidates.len()
        );

        let mut best: Option<(usize, f64)> = None;
        // target_data is not used for AES-KW (encrypted key is pre-loaded),
        // but we need a dummy slice for the trait signature.
        let dummy_target: &[u8] = &[];
        for width in &candidates {
            self.threadgroup_width = *width;
            let (handle, _, _) =
                self.encode_and_commit_markov(dummy_target, length, threshold, 0, SAMPLE_COUNT)?;
            let (_, gpu_wait, _) = self.wait_and_readback(&handle);
            if gpu_wait.is_zero() {
                continue;
            }
            let rate = SAMPLE_COUNT as f64 / gpu_wait.as_secs_f64();
            match best {
                Some((_, best_rate)) if rate <= best_rate => {}
                _ => best = Some((*width, rate)),
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

    #[test]
    fn aeskw_params_round_trip_size_matches() {
        let params = AesKwBruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<AesKwBruteForceParams>());
    }
}
