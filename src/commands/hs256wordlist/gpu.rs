//! Metal GPU setup, compute pipeline management, and dispatch mechanics.
//!
//! # What is Metal?
//!
//! Metal is Apple's low-level GPU programming API -- think of it as Apple's
//! equivalent of NVIDIA's CUDA or the cross-platform Vulkan compute API. It
//! gives you direct access to the GPU's massively parallel compute units.
//!
//! A modern Apple GPU (e.g., M1/M2/M3) has thousands of execution units that
//! can run the same function ("kernel") on different data simultaneously. This
//! is perfect for password cracking: each GPU thread computes HMAC-SHA256 for
//! a different candidate secret key, letting us test millions of keys per second.
//!
//! # Key Metal concepts used in this file
//!
//! ## Device
//! Represents the GPU hardware. `Device::system_default()` gets the default GPU.
//!
//! ## Library
//! A compiled collection of GPU functions (kernels). We compile from Metal
//! Shading Language source code at runtime using `new_library_with_source()`.
//!
//! ## Compute Pipeline State
//! A prepared, compiled GPU function ready for dispatch. Creating a pipeline
//! state compiles the kernel for the specific GPU hardware and validates it.
//! We create this once and reuse it for every dispatch -- pipeline creation is
//! expensive, dispatch is cheap.
//!
//! ## Command Queue
//! A serial queue of work to submit to the GPU. We create one queue and use
//! it for the lifetime of the program.
//!
//! ## Command Buffer
//! A container for one unit of GPU work. The lifecycle is:
//!   1. Create from the queue (`queue.new_command_buffer()`)
//!   2. Encode commands into it (set buffers, dispatch threads)
//!   3. Commit it to the GPU (`command_buffer.commit()`)
//!   4. Optionally wait for completion (`command_buffer.wait_until_completed()`)
//!
//! ## Shared Memory Buffers (MTLResourceOptions::StorageModeShared)
//! On Apple Silicon, CPU and GPU share unified memory. `StorageModeShared`
//! means both CPU and GPU can read/write the same physical memory -- no
//! explicit copy needed! This is a huge advantage over discrete GPUs (like
//! NVIDIA cards) where you must explicitly transfer data over PCIe.
//!
//! ## Threadgroups and Dispatch
//! GPU work is organized in a hierarchy:
//! - **Thread**: One execution unit running the kernel for one candidate.
//! - **Threadgroup**: A batch of threads (e.g., 256) that execute together
//!   and can share fast local memory. The GPU schedules at threadgroup
//!   granularity.
//! - **Grid**: The total number of threads across all threadgroups.
//!
//! `dispatch_threads(grid_size, threadgroup_size)` tells Metal: "run this
//! kernel with `grid_size` total threads, grouped in chunks of
//! `threadgroup_size`." Metal handles dividing the work across GPU cores.
//!
//! # Why two pipelines (short-key vs mixed)?
//!
//! HMAC-SHA256 processes the key in 64-byte blocks. When a key is <= 64 bytes
//! (very common in wordlists), the inner hash needs only one SHA-256 block,
//! which is simpler and faster. The `short_keys` kernel is optimized for this
//! common case, avoiding branches and extra block processing. The `mixed`
//! kernel handles keys of any length but is slightly slower due to the extra
//! generality. We pick the right kernel per batch at dispatch time.

use anyhow::{Context, anyhow, bail};
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::{Duration, Instant};

use crate::commands::common::batch::{DispatchBatchView, WordBatch};
use crate::commands::common::stats::{BatchDispatchTimings, format_human_count};

// Embed the Metal source into the binary so release builds do not depend on a
// runtime-relative source file path.
// `include_str!` is a Rust compile-time macro that reads a file's contents into
// a `&str` constant baked into the binary. This means the .metal shader source
// travels with the executable -- no need to ship a separate file.
const METAL_SOURCE_EMBEDDED: &str = include_str!("hs256_wordlist.metal");
// We keep both kernels loaded and choose per batch based on candidate lengths.
// These names must match the function names in the .metal shader file exactly.
const METAL_FUNCTION_NAME_MIXED: &str = "hs256_wordlist";
const METAL_FUNCTION_NAME_SHORT_KEYS: &str = "hs256_wordlist_short_keys";
// The GPU writes the first matching *batch-local* candidate index here (the
// thread's `gid`, i.e. `0..candidate_count-1`), not an absolute wordlist index.
//
// Keeping the GPU result slot as a single `u32` preserves the same tiny shared
// buffer/atomic path while letting the host scale absolute wordlist indexing to
// `u64` for very large files (multi-billion non-empty lines).
//
// `u32::MAX` remains the sentinel for "no match in this batch".
//
// Design note: Using a sentinel value instead of a separate "found" boolean
// saves a buffer slot and an atomic operation on the GPU. Since we never have
// 4 billion candidates in a single batch, `u32::MAX` is safe as a sentinel.
const RESULT_NOT_FOUND_SENTINEL: u32 = u32::MAX;

/// Host -> Metal parameter block.
///
/// # `#[repr(C)]` -- Critical for GPU interop
///
/// By default, Rust is free to reorder struct fields, insert padding, or use
/// any memory layout it wants for performance. This is great for normal Rust
/// code, but disastrous for GPU interop: the Metal kernel reads this struct's
/// bytes directly from shared memory, so both sides MUST agree on the exact
/// byte positions of every field.
///
/// `#[repr(C)]` forces Rust to use C's layout rules: fields appear in
/// declaration order with standard alignment padding. Since the Metal Shading
/// Language also follows C struct layout rules, both sides agree on the memory
/// layout. Without `#[repr(C)]`, the GPU would read garbage from the wrong
/// byte offsets.
///
/// This is the same reason C FFI (Foreign Function Interface) bindings in Rust
/// always use `#[repr(C)]` -- any time you share a struct's raw bytes with code
/// outside Rust's control, you need a guaranteed layout.
///
/// # Why `[u32; 8]` instead of `[u8; 32]` for the signature?
///
/// SHA-256 internally works on 32-bit words in big-endian order. By converting
/// the 32-byte signature to eight u32 words on the CPU (once), we avoid
/// having every GPU thread do the same byte-to-word conversion. With millions
/// of threads, this small optimization adds up significantly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs256BruteForceParams {
    // Optimization: stored as [u32; 8] big-endian words (converted from [u8; 32]
    // at upload time in `encode_and_commit_view`) so the GPU kernel can compare
    // the final HMAC state words directly without per-thread byte-to-word loads.
    target_signature: [u32; 8],
    // `message_length` is precomputed on the host so the kernel does not need
    // to infer it from buffers or rely on implicit buffer metadata.
    message_length: u32,
    candidate_count: u32,
}

/// Owns Metal objects and the small persistent buffers reused across dispatches.
///
/// This struct is the "GPU context" for the entire cracking run. It is created
/// once during initialization and used for every batch dispatch.
///
/// # Buffer ownership strategy
///
/// There are two kinds of buffers in this design:
///
/// 1. **Small persistent buffers** (owned here): `params_buf`, `msg_buf`,
///    `result_buf`. These are tiny (< 100 bytes) and shared across all
///    dispatches. We write new params before each dispatch and read the
///    result after.
///
/// 2. **Large per-batch payload buffers** (owned by `WordBatch`): The actual
///    candidate word data (megabytes). These live in `WordBatch` and are
///    recycled between the producer and consumer threads to avoid repeated
///    allocation. The GPU reads them directly -- no copy needed thanks to
///    Apple Silicon's unified memory.
///
/// # Why keep two pipelines?
///
/// `pipeline_mixed` handles keys of any length. `pipeline_short_keys` is an
/// optimized fast path for keys <= 64 bytes (one SHA-256 block). Most real
/// passwords/secrets are short, so the fast path handles the vast majority of
/// candidates. We pick which pipeline to use per-batch based on the longest
/// candidate in that batch.
pub(super) struct GpuHs256BruteForcer {
    /// The GPU device handle. `pub(super)` because `command.rs` clones it to
    /// pass to the producer thread for allocating shared-memory buffers.
    pub(super) device: metal::Device,
    /// Serial command queue -- all GPU work is submitted through this.
    queue: metal::CommandQueue,
    /// Compute pipeline for batches containing any-length keys.
    pipeline_mixed: metal::ComputePipelineState,
    /// Optimized compute pipeline for batches where all keys are <= 64 bytes.
    pipeline_short_keys: metal::ComputePipelineState,
    /// Small shared buffer for the `Hs256BruteForceParams` struct (< 48 bytes).
    params_buf: metal::Buffer,
    /// Shared buffer holding the JWT signing input (constant across all dispatches).
    msg_buf: metal::Buffer,
    /// Single u32 shared buffer where the GPU writes the match result index.
    result_buf: metal::Buffer,
    /// Length of the JWT signing input, cached as u32 for the params struct.
    message_length: u32,
    /// Current threadgroup width (threads per threadgroup). May be updated by
    /// autotune or user override.
    threadgroup_width: usize,
}

// ---- Host -> Metal shared-buffer copy helpers ------------------------------
// After the no-copy refactor these are only for small state writes (params,
// result sentinel, one-time JWT message upload), not batch payload bytes.

/// Copy raw bytes into a Metal shared-memory buffer.
///
/// # Why `unsafe`?
///
/// Rust's safety guarantees normally prevent you from writing to arbitrary
/// memory. But Metal buffers are allocated by the GPU driver and accessed via
/// raw pointers (`buffer.contents()` returns a `*mut c_void`). Rust cannot
/// verify at compile time that:
/// - The pointer is valid and non-null.
/// - The buffer is large enough for the data.
/// - No other thread/GPU is simultaneously reading this memory.
///
/// We use `unsafe` to tell the compiler: "I, the programmer, have manually
/// verified these conditions are met." The `debug_assert!` catches size
/// violations in debug builds, and our dispatch protocol ensures the GPU
/// is not reading while we write (we always wait for the previous dispatch
/// to complete before writing new params).
///
/// In Rust, `unsafe` is not "dangerous" -- it means "the compiler can't check
/// this, so the programmer takes responsibility." It should be used sparingly
/// and always with a `// SAFETY:` comment explaining why the invariants hold.
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

/// Convenience wrapper for POD (Plain Old Data) values like the params struct.
///
/// This uses `bytes_of()` to reinterpret the struct as raw bytes, then copies
/// those bytes into the Metal buffer. This is safe because our structs are
/// `#[repr(C)]` with no pointers or padding-dependent fields.
fn copy_value_to_buffer<T>(buffer: &metal::Buffer, value: &T) {
    copy_bytes_to_buffer(buffer, bytes_of(value));
}

impl GpuHs256BruteForcer {
    /// Initialize the Metal GPU runtime for HS256 cracking.
    ///
    /// This performs all the one-time setup that would be too expensive to do
    /// per-batch: compiling shaders, creating pipeline states, and allocating
    /// persistent shared-memory buffers.
    ///
    /// # Metal initialization sequence
    ///
    /// 1. **Get the device** -- `Device::system_default()` returns the system's
    ///    default GPU. On a MacBook, this is the integrated Apple Silicon GPU.
    ///
    /// 2. **Compile the shader** -- `new_library_with_source()` compiles the
    ///    Metal Shading Language (.metal) source code into GPU machine code at
    ///    runtime. This is like `nvcc` for CUDA, but happens at program startup
    ///    instead of build time. (Metal also supports precompiled `.metallib`
    ///    files, but runtime compilation from embedded source is simpler for
    ///    distribution.)
    ///
    /// 3. **Create pipeline states** -- A pipeline state is a fully compiled,
    ///    validated GPU function ready for dispatch. Creating one is expensive
    ///    (involves GPU-specific optimization), so we do it once and reuse it.
    ///    Think of it like compiling a program vs running it.
    ///
    /// 4. **Create the command queue** -- A queue that serializes GPU work
    ///    submissions. All command buffers are created from this queue.
    ///
    /// 5. **Allocate shared-memory buffers** -- Small buffers for params, the
    ///    JWT message, and the result slot. `StorageModeShared` means CPU and
    ///    GPU access the same physical memory (no copy needed on Apple Silicon).
    pub(super) fn new(signing_input: &[u8]) -> anyhow::Result<Self> {
        // Step 1: Get the GPU device handle.
        let device =
            Device::system_default().ok_or_else(|| anyhow!("no Metal device available"))?;
        let compile_options = CompileOptions::new();
        // Step 2: Compile the embedded Metal source at runtime. The source is
        // no longer read from disk, which makes the binary self-contained.
        let library = device
            .new_library_with_source(METAL_SOURCE_EMBEDDED, &compile_options)
            .map_err(|e| anyhow!("failed to compile Metal kernel: {e}"))?;

        // Step 3a: Look up the kernel functions by name from the compiled library.
        // These names must exactly match the function names in the .metal file.
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

        // Step 3b: Create compute pipeline states from the functions. This is
        // where Metal compiles the kernel for the specific GPU hardware and
        // validates resource usage (registers, memory, etc.).
        let pipeline_mixed = device
            .new_compute_pipeline_state_with_function(&mixed_function)
            .map_err(|e| anyhow!("failed to create mixed compute pipeline: {e}"))?;
        let pipeline_short_keys = device
            .new_compute_pipeline_state_with_function(&short_function)
            .map_err(|e| anyhow!("failed to create short-key compute pipeline: {e}"))?;

        // Step 4: Create the command queue for submitting GPU work.
        let queue = device.new_command_queue();
        // Both pipelines may have different hardware limits, so use the lower
        // value to keep a single `threadgroup_width` valid for either kernel.
        // The GPU hardware has a maximum threads-per-threadgroup (often 1024 on
        // Apple Silicon). We default to 256 or the hardware max, whichever is
        // smaller. The user can override this with --threads-per-group, and
        // --autotune will benchmark different widths to find the fastest.
        let max_threads = (pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(pipeline_short_keys.max_total_threads_per_threadgroup() as usize);
        let threadgroup_width = 256usize.min(max_threads.max(1));

        let message_length =
            u32::try_from(signing_input.len()).context("JWT signing input too long")?;

        // Step 5: Allocate shared-memory buffers.
        // `StorageModeShared` = CPU and GPU share the same physical memory.
        // This is the key advantage of Apple Silicon's unified memory architecture:
        // no PCIe transfers, no explicit memcpy between host and device.
        let options = MTLResourceOptions::StorageModeShared;

        let params_buf =
            device.new_buffer(std::mem::size_of::<Hs256BruteForceParams>() as u64, options);
        let msg_buf = device.new_buffer(signing_input.len().max(1) as u64, options);
        // The JWT signing input is constant across the entire run, so upload it
        // once and reuse the same shared buffer for every dispatch.
        copy_bytes_to_buffer(&msg_buf, signing_input);

        let result_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, options);
        // Initialize the result buffer with the "not found" sentinel so we can
        // distinguish "no match" from "match at index 0".
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

    /// Select the specialized short-key kernel when every candidate in the
    /// dispatch view is <= 64 bytes; otherwise fall back to the mixed kernel.
    ///
    /// This per-batch decision is cheap (one integer comparison) but yields
    /// significant GPU speedups because the short-key kernel skips the extra
    /// SHA-256 block processing loop that the mixed kernel needs for long keys.
    /// `DispatchBatchView` lets autotune reuse this logic on a sampled prefix.
    fn active_pipeline_for_view(
        &self,
        batch: DispatchBatchView<'_>,
    ) -> &metal::ComputePipelineState {
        // 64 bytes = one SHA-256 block. HMAC processes the key in blocks,
        // and keys <= 64 bytes fit in a single block (the fast path).
        if batch.max_word_len <= 64 {
            &self.pipeline_short_keys
        } else {
            &self.pipeline_mixed
        }
    }

    /// Validate and apply a user-provided threadgroup width override.
    ///
    /// Threadgroup width must be > 0 and <= the GPU hardware maximum.
    /// Exceeding the hardware limit would cause the GPU to reject the dispatch.
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

    /// Accessor used by logging and periodic rate reports.
    pub(super) fn current_threadgroup_width(&self) -> usize {
        self.threadgroup_width
    }

    /// The "thread execution width" (TEW) is the number of threads that
    /// execute in true lockstep on the GPU (like CUDA's "warp size", which
    /// is 32 on NVIDIA GPUs). On Apple Silicon this is typically 32.
    /// Threadgroup widths that are multiples of the TEW avoid wasted lanes.
    pub(super) fn thread_execution_width(&self) -> usize {
        (self.pipeline_mixed.thread_execution_width() as usize)
            .min(self.pipeline_short_keys.thread_execution_width() as usize)
    }

    /// Conservative threadgroup cap valid for both kernels.
    /// We take the minimum of both pipelines so one threadgroup_width works
    /// regardless of which kernel is selected for a given batch.
    pub(super) fn max_total_threads_per_threadgroup(&self) -> usize {
        (self.pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(self.pipeline_short_keys.max_total_threads_per_threadgroup() as usize)
    }

    /// Human-readable device name for startup logging (e.g., "Apple M2 Pro").
    pub(super) fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Encode, commit, and return immediately without waiting for GPU completion.
    ///
    /// # The encode -> commit -> wait lifecycle
    ///
    /// Metal GPU dispatch follows a three-phase pattern:
    ///
    /// 1. **Encode**: Record commands into a command buffer (set buffers,
    ///    specify the kernel, configure the dispatch grid). This is CPU work
    ///    that builds a "to-do list" for the GPU. No GPU work happens yet.
    ///
    /// 2. **Commit**: Submit the command buffer to the GPU command queue.
    ///    The GPU begins executing asynchronously. The CPU is free to do
    ///    other work (like preparing the next batch).
    ///
    /// 3. **Wait**: Block the CPU until the GPU finishes, then read results.
    ///    See `wait_and_readback()` below.
    ///
    /// By splitting encode+commit from wait, we enable the double-buffered
    /// overlap pattern in `command.rs`: while the GPU processes batch N,
    /// the CPU can parse and prepare batch N+1.
    ///
    /// # Important constraint
    ///
    /// The returned `CommandBuffer` MUST be passed to `wait_and_readback`
    /// before the next dispatch, because `params_buf` and `result_buf` are
    /// shared across dispatches. Starting a new dispatch while the GPU is
    /// still reading these buffers would cause a data race.
    pub(super) fn encode_and_commit(
        &self,
        target_signature: [u8; 32],
        batch: &WordBatch,
    ) -> anyhow::Result<(metal::CommandBuffer, Duration, Duration)> {
        self.encode_and_commit_view(
            target_signature,
            batch.as_dispatch_view(),
            self.threadgroup_width,
        )
    }

    /// Internal implementation of encode+commit that works with a `DispatchBatchView`.
    ///
    /// Using `DispatchBatchView` (a lightweight reference to batch data) instead
    /// of `&WordBatch` lets autotune dispatch a *prefix* of a batch without
    /// copying data -- the view just references a subset of the same buffers.
    fn encode_and_commit_view(
        &self,
        target_signature: [u8; 32],
        batch: DispatchBatchView<'_>,
        threadgroup_width: usize,
    ) -> anyhow::Result<(metal::CommandBuffer, Duration, Duration)> {
        let candidate_count =
            u32::try_from(batch.candidate_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            bail!("cannot encode empty batch for async dispatch");
        }

        // Optimization: convert [u8; 32] -> [u32; 8] big-endian on the host so
        // the kernel compares word-against-word without per-thread load_be_u32.
        // SHA-256 produces its output as 8 big-endian 32-bit words internally,
        // so by pre-converting once on the CPU, we save millions of GPU threads
        // from each doing the same byte-shuffling work.
        let mut target_words = [0u32; 8];
        for i in 0..8 {
            let off = i * 4;
            target_words[i] = u32::from_be_bytes([
                target_signature[off],
                target_signature[off + 1],
                target_signature[off + 2],
                target_signature[off + 3],
            ]);
        }
        let params = Hs256BruteForceParams {
            target_signature: target_words,
            message_length: self.message_length,
            candidate_count,
        };

        // --- Phase 1: Host preparation (CPU work) ---
        // Write the params struct and reset the result sentinel in shared memory.
        // This is fast (just memcpy of ~48 bytes + 4 bytes).
        let prep_started = Instant::now();
        copy_value_to_buffer(&self.params_buf, &params);
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        let host_prep = prep_started.elapsed();

        // --- Phase 2: Encode commands into a command buffer ---
        // This records the GPU commands but does NOT execute them yet.
        let encode_started = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        // A compute command encoder records compute (non-graphics) GPU commands.
        let encoder = command_buffer.new_compute_command_encoder();
        // Select which compiled kernel to use (short-key fast path or mixed).
        let pipeline = self.active_pipeline_for_view(batch);
        encoder.set_compute_pipeline_state(pipeline);
        // Bind buffers to kernel argument indices. These indices (0-5) must
        // match the `[[buffer(N)]]` attributes in the .metal shader code.
        // Buffer 0: params (target signature, message length, candidate count)
        encoder.set_buffer(0, Some(&self.params_buf), 0);
        // Buffer 1: JWT signing input (the message to HMAC-sign)
        encoder.set_buffer(1, Some(&self.msg_buf), 0);
        // Buffer 2: packed candidate word bytes (the big payload)
        encoder.set_buffer(2, Some(batch.word_bytes_buf), 0);
        // Buffer 3: per-candidate byte offset into word_bytes
        encoder.set_buffer(3, Some(batch.word_offsets_buf), 0);
        // Buffer 4: per-candidate length in bytes
        encoder.set_buffer(4, Some(batch.word_lengths_buf), 0);
        // Buffer 5: result slot (GPU writes matching index here via atomic)
        encoder.set_buffer(5, Some(&self.result_buf), 0);

        // Configure the dispatch grid. We use 1D dispatch (one thread per
        // candidate). Metal automatically divides the grid into threadgroups.
        // `threads_per_grid` = total number of threads = number of candidates.
        // `threads_per_group` = how many threads per threadgroup (e.g., 256).
        let threads_per_group = MTLSize::new(threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(candidate_count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        // Finalize the encoder. No more commands can be added after this.
        encoder.end_encoding();
        let command_encode = encode_started.elapsed();

        // --- Phase 3: Commit (submit to GPU) ---
        // After commit(), the GPU begins executing asynchronously. The CPU
        // returns immediately and can do other work while the GPU crunches.
        command_buffer.commit();

        // Retain the command buffer so it survives past Objective-C's
        // autorelease pool. Metal is built on Objective-C, which uses reference
        // counting. `.to_owned()` increments the retain count so Rust owns it.
        let owned = command_buffer.to_owned();
        Ok((owned, host_prep, command_encode))
    }

    /// Block until a previously committed command buffer completes, then read
    /// the match result from shared memory.
    ///
    /// This is the "wait" phase of the encode -> commit -> wait lifecycle.
    /// After `commit()`, the GPU executes asynchronously. This method blocks
    /// the calling CPU thread until the GPU finishes, then reads the result.
    ///
    /// # Why `unsafe` for the readback?
    ///
    /// Reading from the Metal buffer requires dereferencing a raw pointer
    /// (`result_buf.contents()` returns `*mut c_void`). This is safe here
    /// because:
    /// 1. We called `wait_until_completed()`, guaranteeing the GPU is done.
    /// 2. The buffer is 4 bytes (one u32) and was allocated by us.
    /// 3. No other thread writes to this buffer between wait and read.
    pub(super) fn wait_and_readback(
        &self,
        cmd_buf: &metal::CommandBufferRef,
    ) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        // This blocks until the GPU finishes all commands in cmd_buf.
        // The time spent here is "gpu_wait" -- ideally the CPU was doing
        // useful work (parsing the next batch) while waiting.
        cmd_buf.wait_until_completed();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        // SAFETY: `result_buf` is a 4-byte shared buffer; GPU has completed.
        let result = unsafe { *result_ptr };
        let result_readback = readback_started.elapsed();

        // Convert the sentinel value to Rust's Option type.
        // `None` = no match found in this batch.
        // `Some(index)` = the GPU found a match at this batch-local index.
        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    /// Synchronous dispatch: encode, commit, wait, and readback in one call.
    /// Used by autotune and test paths where pipelining is not needed.
    ///
    /// This is the "simple" dispatch path. The main cracking loop uses the
    /// split encode_and_commit + wait_and_readback for double-buffered overlap.
    fn dispatch_batch_view(
        &self,
        target_signature: [u8; 32],
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

    /// Benchmark a small sample from the first batch across several candidate
    /// threadgroup widths, then keep the fastest width for the rest of the run.
    ///
    /// # Why autotune?
    ///
    /// Different GPU architectures perform best with different threadgroup
    /// widths. For example, M1 might prefer 256 while M3 might prefer 512.
    /// Instead of hardcoding a value, autotune empirically tests several
    /// candidates on a small sample (~16K words) and picks the fastest.
    ///
    /// # Why optimize for `gpu_wait` only?
    ///
    /// We measure pure GPU execution time (not end-to-end time including host
    /// overhead) because threadgroup width only affects GPU scheduling. Host
    /// parsing speed is independent of this parameter. Measuring end-to-end
    /// would add noise from variable host work.
    ///
    /// # Candidate widths
    ///
    /// We test multiples of the thread execution width (TEW, typically 32)
    /// plus common power-of-2 values. Widths that are multiples of TEW
    /// avoid "wasted lanes" where some SIMD lanes sit idle. The candidate
    /// list is deduplicated and filtered to the hardware maximum.
    pub(super) fn autotune_threadgroup_width(
        &mut self,
        target_signature: [u8; 32],
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

/// Reinterpret a plain-old-data (POD) value as raw bytes for GPU uploads.
///
/// This is a common pattern in GPU programming: you have a typed struct on the
/// CPU side, but the GPU just sees a bag of bytes. This function lets you view
/// any `#[repr(C)]` struct as a `&[u8]` slice without copying.
///
/// # Why not use a crate like `bytemuck`?
///
/// We could use `bytemuck::bytes_of()` which does the same thing with safer
/// trait bounds. This hand-rolled version avoids the dependency and is fine
/// for the small number of types we use it with. In a larger codebase,
/// `bytemuck` would be preferred for its compile-time safety checks.
///
/// # Safety
///
/// Callers are responsible for using types whose in-memory layout is the same as
/// the corresponding Metal-side struct/bytes they expect to upload. Specifically:
/// - The type must be `#[repr(C)]` (guaranteed layout).
/// - The type must not contain pointers (they would be meaningless to the GPU).
/// - The type must not have padding bytes that could leak uninitialized memory.
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
    fn hs256_params_round_trip_size_matches() {
        let params = Hs256BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs256BruteForceParams>());
    }
}
