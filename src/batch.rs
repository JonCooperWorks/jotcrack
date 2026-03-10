//! # Batch data structure for GPU-accelerated password candidate dispatch
//!
//! This module defines the core data structure (`WordBatch`) that bridges the
//! CPU-side wordlist parser and the Metal GPU compute kernel. It is the single
//! most performance-critical data structure in the pipeline.
//!
//! ## Why not `Vec<String>`?
//!
//! A naive approach would store candidates as `Vec<String>`. That has two fatal
//! problems for GPU work:
//!
//! 1. **Memory layout**: Each `String` is a separate heap allocation scattered
//!    across memory. GPUs need data in flat, contiguous buffers — they cannot
//!    chase pointers or dereference `String` internals.
//!
//! 2. **Allocation overhead**: With millions of candidates per batch, creating
//!    and dropping millions of `String` objects causes enormous allocator churn
//!    that dominates CPU time.
//!
//! Instead, we use a *packed representation*: one contiguous byte array for all
//! candidate bytes, plus parallel offset/length tables. This is sometimes called
//! a "struct of arrays" (SoA) layout, and it maps directly to how GPU kernels
//! consume data.
//!
//! ## Memory layout diagram
//!
//! After pushing candidates "alpha" and "be":
//!
//! ```text
//! word_bytes:   [a][l][p][h][a][b][e]     ← concatenated, NO separators or null terminators
//! word_offsets: [0, 5]                     ← byte offset where each candidate starts
//! word_lengths: [5, 2]                     ← byte length of each candidate
//!
//! To reconstruct candidate i:
//!   start = word_offsets[i]          (e.g., offsets[1] = 5)
//!   len   = word_lengths[i]          (e.g., lengths[1] = 2)
//!   bytes = word_bytes[start..start+len]   → "be"
//! ```
//!
//! The GPU kernel does exactly this: thread `i` reads `offsets[i]` and
//! `lengths[i]`, then indexes into the `word_bytes` buffer to get its candidate.
//!
//! ## Metal shared buffers
//!
//! All three arrays live in Metal "shared" buffers (`MTLResourceOptions::StorageModeShared`).
//! On Apple Silicon, shared buffers use *unified memory* — the same physical RAM
//! is directly accessible by both the CPU and GPU with no copying. The CPU writes
//! candidates into the buffer, then the GPU reads from it in-place. This is a
//! key advantage of Apple's architecture over discrete GPUs, which would require
//! an explicit DMA transfer.

use anyhow::{Context, bail};

use super::gpu::{GpuBuffer, GpuDevice, alloc_shared_buffer, buffer_host_ptr};

// Larger batches improve amortization of dispatch overhead and host work.
//
// ## Sizing rationale
//
// `MAX_CANDIDATES_PER_BATCH` (roughly 6.2M) balances two competing concerns:
//   - **Large enough** to amortize the fixed overhead of each GPU dispatch
//     (kernel launch, command buffer encoding, synchronization). If batches are
//     too small, the GPU spends more time idle between dispatches than computing.
//   - **Small enough** that the host can fill a batch before the GPU finishes
//     the previous one, keeping the pipeline saturated.
//
// `MAX_WORD_BYTES_PER_BATCH` (32 MiB) caps the raw byte payload. This prevents
// pathologically long candidates (e.g., a 1 MB line in a wordlist) from blowing
// up a single batch's memory.
//
// Together, these two limits define the "shape" of a batch and are checked by
// `batch_shape_can_fit()` whenever candidates are added.
pub(crate) const MAX_CANDIDATES_PER_BATCH: usize = 6_182_240;
pub(crate) const MAX_WORD_BYTES_PER_BATCH: usize = 32 * 1024 * 1024;
// Actual allocation size for the word_bytes buffer. Larger than
// MAX_WORD_BYTES_PER_BATCH because the bulk-copy packing optimization
// copies contiguous mmap ranges including inter-candidate line endings
// (LF or CRLF). The GPU ignores these gap bytes — it reads only
// (offset, length) per candidate — but the buffer must be large enough
// to hold them. Worst case: 2 bytes (CRLF) per candidate gap.
pub(crate) const WORD_BYTES_ALLOC_SIZE: usize =
    MAX_WORD_BYTES_PER_BATCH + MAX_CANDIDATES_PER_BATCH * 2;
// Approximate shared-buffer bytes held by one `WordBatch` allocation. This is
// the dominant memory cost when increasing producer/consumer pipeline depth.
//
// Breakdown:
//   - word_bytes buffer:   ~44 MiB                       (WORD_BYTES_ALLOC_SIZE)
//   - word_offsets buffer: 6.2M × 4 bytes ≈ 23.6 MiB    (one u32 per candidate)
//   - word_lengths buffer: 6.2M × 2 bytes ≈ 11.8 MiB    (one u16 per candidate)
//   - Total: ~80 MiB per WordBatch
//
// When the pipeline depth is N, we have up to N WordBatch objects alive at once,
// so total GPU buffer memory is roughly N × 67 MiB. This is why pipeline_depth
// is a tunable parameter — deeper pipelines trade memory for better overlap.
pub(crate) const APPROX_WORD_BATCH_BUFFER_BYTES: usize = WORD_BYTES_ALLOC_SIZE
    + (MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u32>())
    + (MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u16>());

/// Packed representation of a wordlist batch sent to the GPU.
///
/// Instead of storing `Vec<String>` per candidate (which is allocation-heavy for
/// huge wordlists), we store one contiguous byte blob plus offset/length tables.
/// This minimizes host memory churn and matches the GPU kernel input format.
///
/// ## Ownership and lifecycle
///
/// A `WordBatch` is created once (allocating Metal buffers), then recycled many
/// times via `reset_for_reuse()`. The lifecycle is:
///
/// 1. **Allocate** — `WordBatch::new()` creates fixed-capacity Metal shared buffers.
/// 2. **Fill** — The producer thread pushes candidates via `push_candidate()` or
///    `push_segment_bulk()`, writing directly into the Metal buffer memory.
/// 3. **Dispatch** — The consumer thread binds the Metal buffers to a compute
///    kernel and dispatches GPU work via `as_dispatch_view()`.
/// 4. **Recycle** — After the GPU finishes, the batch is returned to the producer
///    via the recycle channel. `reset_for_reuse()` clears the logical lengths
///    (but does NOT zero the buffer memory — just resets the cursors).
///
/// This recycling pattern avoids repeated Metal buffer allocation, which involves
/// Objective-C method calls and kernel-level memory mapping.
#[derive(Debug)]
pub(crate) struct WordBatch {
    // Host-only absolute wordlist index of candidate #0 in this batch.
    //
    // This is intentionally `u64` so parsing/progress/result reconstruction can
    // exceed 4,294,967,295 non-empty candidates. It is not sent to the GPU
    // anymore; the kernel returns a batch-local match index (`u32`) instead.
    pub(crate) candidate_index_base: u64,

    // --- GPU shared buffers ---
    //
    // These are the actual GPU-visible allocations. On Apple Silicon, "shared"
    // means the same physical memory page is mapped into both CPU and GPU address
    // spaces (unified memory architecture). The CPU writes candidate data here,
    // and the GPU reads it directly — zero copies needed.
    //
    // Buffers that are directly bound to the kernel at dispatch time.
    // The producer fills them; the consumer reuses the same allocations.
    word_bytes_buf: GpuBuffer,
    word_offsets_buf: GpuBuffer,
    word_lengths_buf: GpuBuffer,

    // --- Cached raw pointers (performance optimization) ---
    //
    // `metal::Buffer::contents()` returns the CPU-visible base pointer of the
    // buffer. Under the hood, this is an Objective-C method call:
    //   `[buffer contents]` → objc_msgSend → pointer
    //
    // In a hot loop pushing millions of candidates, the overhead of repeated
    // Objective-C message dispatch is measurable. Since the pointer is stable
    // for the lifetime of the buffer (it never moves — Metal guarantees this
    // for shared storage mode), we cache it at construction time and use raw
    // pointer arithmetic in `push_candidate` / `push_segment_bulk`.
    //
    // ## Why typed `*mut T` instead of `usize`
    //
    // An earlier version stored these as `usize` and cast back to `*mut T` at
    // every use site. That compiles to the same machine code, but loses
    // *pointer provenance*: the compiler (and sanitizers like Miri) track which
    // allocation a pointer was derived from to prove that `.add()` / dereference
    // operations land within a valid allocation. Casting through `usize` erases
    // that provenance, which can cause Miri failures and — in theory — allow
    // future compiler optimisations to miscompile the code.
    //
    // Storing typed `*mut T` pointers preserves provenance end-to-end: the
    // pointer originates from `buffer_host_ptr()`, is cast to the target
    // element type (`u32` for offsets, `u16` for lengths) once at construction
    // time via `.cast::<T>()`, and is used via `.add(index)` without ever
    // round-tripping through an integer. This also removes the repetitive
    // `as *mut T` casts that previously cluttered every `unsafe` block.
    //
    // ## Why `*mut` and not `*const`
    //
    // The pointers are used for both reads (`from_raw_parts` in slice accessors)
    // and writes (in `push_candidate` / `push_segment_bulk`). Using `*mut`
    // keeps one pointer per buffer; the read-only accessors cast to `*const`
    // at the call site, which is a no-op at the machine level.
    //
    // Cached CPU-visible base pointers for the shared buffers.
    // This avoids repeated `contents()` Objective-C calls in the hot parser loop.
    word_bytes_ptr: *mut u8,
    word_offsets_ptr: *mut u32,
    word_lengths_ptr: *mut u16,

    // --- Logical lengths (the "fill cursor") ---
    //
    // The Metal buffers are allocated at their maximum capacity, but only a
    // prefix is actually initialized with valid data. These fields track how
    // much of each buffer is "live." This is analogous to `Vec::len()` vs
    // `Vec::capacity()` — the buffer capacity is fixed, and these are the
    // logical lengths.
    //
    // Logical initialized prefix lengths within the fixed-size Metal buffers.
    candidate_count: usize,
    word_bytes_len: usize,
    max_word_len: u16,
}

// ## Why `unsafe impl Send`?
//
// Raw pointers (`*mut T`) do not auto-implement `Send` or `Sync`. This is a
// deliberate Rust language decision: the compiler cannot verify that the
// pointed-to memory is safe to access from another thread, so it refuses to
// auto-derive thread-safety traits for types that contain raw pointers.
//
// Before a security fix, these fields were `usize` (which *is* `Send`), so
// `WordBatch` auto-derived `Send` implicitly. Changing to typed `*mut T`
// pointers — which is strictly more correct from a provenance standpoint —
// requires us to explicitly opt back in with `unsafe impl Send`.
//
// SAFETY: The `Send` impl is sound because:
//
//   1. **Exclusive ownership protocol** — `WordBatch` instances travel through
//      a single-owner channel: the producer thread fills a batch, sends it to
//      the consumer thread for GPU dispatch, and the consumer sends it back
//      for recycling. At no point do two threads hold `&mut` simultaneously.
//
//   2. **Pointer validity** — The cached pointers point into Metal shared
//      buffers (`StorageModeShared`) that are owned by `self` via the
//      `word_bytes_buf` / `word_offsets_buf` / `word_lengths_buf` fields.
//      These allocations live as long as the `WordBatch` itself, so the
//      pointers remain valid regardless of which thread accesses them.
//
//   3. **`Sync` for the JWE CPU path** — `WordBatch` is shared (`&WordBatch`)
//      across CPU worker threads during JWE A128KW cracking. The cached raw
//      pointers reference stable Metal shared-mode buffer memory that is never
//      relocated. During CPU batch processing, all access is read-only via
//      `word()` — the producer only writes after the batch is recycled.
unsafe impl Send for WordBatch {}

// SAFETY: The cached raw pointers (`word_bytes_ptr`, `word_offsets_ptr`,
// `word_lengths_ptr`) reference Metal `StorageModeShared` buffers, which
// on Apple Silicon are ordinary unified memory with stable addresses.
// Multiple threads may safely read from these buffers concurrently as long
// as no thread writes to them. This invariant is maintained by the pipeline
// design: during CPU batch processing (JWE path), worker threads call
// `word()` through `&self` (no mutation). The producer writes to a batch
// only after recycling, which happens *after* all readers finish.
unsafe impl Sync for WordBatch {}

/// Check whether a single candidate of `line_len` bytes fits in the current batch state.
///
/// This is the **single-candidate** capacity check, used by the line-by-line
/// pack path and the planner's per-line fallback. It enforces two independent limits:
///
/// 1. **Candidate count** — never exceed `MAX_CANDIDATES_PER_BATCH` (GPU thread grid sizing).
/// 2. **Byte budget** — total packed bytes must not exceed `MAX_WORD_BYTES_PER_BATCH`.
///
/// ## The "first candidate" exemption
///
/// Notice the `candidate_count != 0` guard on the byte check. This is intentional:
/// if the very first candidate in a batch is enormous (say, a 33 MiB line), we
/// still allow it to form a batch by itself. Without this exemption, a single
/// oversized line would be silently dropped or cause an infinite loop trying to
/// fit into every batch.
///
/// ## Why this is a free function, not a method
///
/// Both `WordBatch::can_fit()` (which has a `&self`) and the planner (which
/// tracks shape without a `WordBatch`) need the same logic. Extracting it as a
/// free function with plain integer arguments avoids duplication and makes the
/// boundary calculation testable in isolation.
// Shared batch-capacity rule used by both the direct pack path and the new
// planner stage so batch boundaries remain identical across implementations.
pub(crate) fn batch_shape_can_fit(
    candidate_count: usize,
    word_bytes_len: usize,
    line_len: usize,
) -> bool {
    // Preserve the original batching rule:
    // - always enforce candidate-count cap
    // - only enforce byte-cap after the first candidate is present
    let would_exceed_count = candidate_count >= MAX_CANDIDATES_PER_BATCH;
    let would_exceed_bytes =
        candidate_count != 0 && (word_bytes_len + line_len > MAX_WORD_BYTES_PER_BATCH);
    !(would_exceed_count || would_exceed_bytes)
}

/// Block-level batch capacity check — an optimization for the planner.
///
/// Instead of checking one candidate at a time (O(n) calls to `batch_shape_can_fit`),
/// this checks whether an **entire block** of `block_line_count` candidates
/// totalling `block_total_bytes` fits in the remaining batch capacity.
///
/// ## Why this matters for performance
///
/// The planner processes a memory-mapped wordlist that has been pre-parsed into
/// "chunks" of ~4096 lines each. If we had to call `batch_shape_can_fit()` for
/// every single line, the planner would be O(total_lines). With block-level
/// checks, most of the wordlist is consumed in O(total_lines / 4096) steps —
/// a massive speedup for multi-gigabyte wordlists.
///
/// ## Conservative correctness
///
/// The check is conservative: if the whole block fits, then every
/// intermediate state during line-by-line consumption also fits, so the
/// planner can skip 4096 lines in one step without altering batch
/// boundaries. This is safe because both limits (count and bytes) are
/// monotonically increasing — adding a subset of a block that fits
/// will always fit.
///
/// ## Precondition
///
/// Caller must ensure `candidate_count > 0` before calling — the first-
/// candidate byte-cap exemption (allowing a single large candidate to
/// form its own batch) is handled by falling through to the per-line
/// path in the planner.
// Optimization: block-level batch capacity check
//
// Companion to batch_shape_can_fit() for the block-summary planner.
// Instead of checking one candidate at a time, this checks whether an
// entire block of `block_line_count` candidates totalling
// `block_total_bytes` fits in the remaining batch capacity.
//
// The check is conservative: if the whole block fits, then every
// intermediate state during line-by-line consumption also fits, so the
// planner can skip 4096 lines in one step without altering batch
// boundaries.
//
// Caller must ensure `candidate_count > 0` before calling — the first-
// candidate byte-cap exemption (allowing a single large candidate to
// form its own batch) is handled by falling through to the per-line
// path in the planner.
#[cfg_attr(all(target_os = "linux", not(test)), allow(dead_code))]
pub(crate) fn batch_shape_can_fit_block(
    candidate_count: usize,
    word_bytes_len: usize,
    block_line_count: usize,
    block_total_bytes: usize,
) -> bool {
    let would_exceed_count = candidate_count + block_line_count > MAX_CANDIDATES_PER_BATCH;
    let would_exceed_bytes = word_bytes_len + block_total_bytes > MAX_WORD_BYTES_PER_BATCH;
    !(would_exceed_count || would_exceed_bytes)
}

impl WordBatch {
    /// Allocate fixed-capacity Metal shared buffers sized to the batch caps.
    ///
    /// ## Metal buffer allocation
    ///
    /// `device.new_buffer(size, options)` asks the Metal driver to allocate a
    /// buffer of `size` bytes with the given storage mode. With
    /// `StorageModeShared`, the buffer lives in unified memory accessible to
    /// both CPU and GPU.
    ///
    /// Key insight: the buffers are allocated at **maximum** capacity once, then
    /// reused across many batches. This is critical because Metal buffer
    /// allocation involves:
    ///   - An Objective-C method call across the Rust/ObjC bridge
    ///   - Kernel-level virtual memory mapping
    ///   - Potential page table updates
    ///
    /// All of these are far too expensive to do per-batch in a hot loop.
    ///
    /// ## Pointer caching
    ///
    /// After allocation, we immediately cache the CPU-visible base pointer for
    /// each buffer. `contents()` returns the same pointer every time for shared
    /// buffers, but calling it is an Objective-C message send. Caching it here
    /// means `push_candidate()` can use pure pointer arithmetic with zero
    /// FFI overhead.
    // Allocate fixed-capacity shared buffers sized to the batch caps so parser
    // writes can go straight into memory later bound to the kernel.
    pub(crate) fn new(device: &GpuDevice, candidate_index_base: u64) -> Self {
        let word_bytes_buf = alloc_shared_buffer(device, WORD_BYTES_ALLOC_SIZE);
        let word_offsets_buf =
            alloc_shared_buffer(device, MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u32>());
        let word_lengths_buf =
            alloc_shared_buffer(device, MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u16>());
        // Cache the CPU-visible base pointer for each buffer. The pointer is
        // stable for the lifetime of the buffer allocation (Metal guarantees
        // shared-mode buffers do not relocate), so we cache once and use raw
        // pointer arithmetic in `push_candidate` / `push_segment_bulk`.
        //
        // `buffer_host_ptr()` returns `*mut u8`. The offsets buffer stores
        // `u32` entries and the lengths buffer stores `u16` entries, so we
        // `.cast::<T>()` to get a correctly-typed pointer. This cast is purely
        // a type-system operation — it compiles to zero instructions — but it
        // lets `.add(index)` compute the correct stride (`index * size_of::<T>()`
        // bytes) without manual arithmetic at every use site.
        Self {
            candidate_index_base,
            word_bytes_ptr: buffer_host_ptr(&word_bytes_buf),
            word_offsets_ptr: buffer_host_ptr(&word_offsets_buf).cast::<u32>(),
            word_lengths_ptr: buffer_host_ptr(&word_lengths_buf).cast::<u16>(),
            word_bytes_buf,
            word_offsets_buf,
            word_lengths_buf,
            candidate_count: 0,
            word_bytes_len: 0,
            max_word_len: 0,
        }
    }

    /// Reset the batch for reuse without reallocating Metal buffers.
    ///
    /// This is the key to the "object pool" pattern: instead of dropping this
    /// `WordBatch` (which would free the Metal buffers) and allocating a new one,
    /// we just reset the logical cursors to zero. The underlying buffer memory
    /// still contains stale data from the previous batch, but that's fine —
    /// we will overwrite it before the GPU reads it.
    ///
    /// Think of it like rewinding a tape: the tape (buffer) stays, but we start
    /// recording from the beginning again.
    // Reuse batch allocations across producer iterations to reduce allocator
    // churn in the parsing hot path while preserving the same batch semantics.
    pub(crate) fn reset_for_reuse(&mut self, candidate_index_base: u64) {
        // Rebinding the base is what preserves globally correct indexing across
        // batches even when the backing Metal buffers are recycled.
        self.candidate_index_base = candidate_index_base;
        self.candidate_count = 0;
        self.word_bytes_len = 0;
        self.max_word_len = 0;
    }

    #[cfg(test)]
    // A batch is considered empty when it has no candidate metadata entries.
    pub(crate) fn is_empty(&self) -> bool {
        self.candidate_count == 0
    }

    // Number of candidates packaged for this dispatch.
    pub(crate) fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    // Number of packed candidate bytes for this dispatch.
    pub(crate) fn word_bytes_len(&self) -> usize {
        self.word_bytes_len
    }

    pub(crate) fn max_word_len(&self) -> u16 {
        self.max_word_len
    }

    fn offsets_buffer(&self) -> &GpuBuffer {
        &self.word_offsets_buf
    }

    fn lengths_buffer(&self) -> &GpuBuffer {
        &self.word_lengths_buf
    }

    fn word_bytes_buffer(&self) -> &GpuBuffer {
        &self.word_bytes_buf
    }

    /// View the initialized portion of the offsets buffer as a Rust slice.
    ///
    /// ## Why `unsafe` is needed here
    ///
    /// `std::slice::from_raw_parts` is unsafe because the compiler cannot verify:
    ///   1. The pointer is valid and properly aligned
    ///   2. The memory region `[ptr, ptr + len)` is initialized
    ///   3. No mutable alias exists for the same memory
    ///
    /// We maintain these invariants manually:
    ///   - The pointer comes from a Metal buffer that outlives `&self`
    ///   - Only the first `candidate_count` entries are claimed, and those
    ///     were initialized by `push_candidate` / `push_segment_bulk`
    ///   - `&self` (shared reference) ensures no `&mut self` methods run concurrently
    pub(crate) fn offsets_slice(&self) -> &[u32] {
        // SAFETY: `word_offsets_buf` stores `u32` entries; the first
        // `candidate_count` entries are initialized by `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(self.word_offsets_ptr as *const u32, self.candidate_count)
        }
    }

    pub(crate) fn lengths_slice(&self) -> &[u16] {
        // SAFETY: `word_lengths_buf` stores `u16` entries; the first
        // `candidate_count` entries are initialized by `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(self.word_lengths_ptr as *const u16, self.candidate_count)
        }
    }

    pub(crate) fn word_bytes_slice(&self) -> &[u8] {
        // SAFETY: the first `word_bytes_len` bytes are initialized by
        // `push_candidate`.
        unsafe { std::slice::from_raw_parts(self.word_bytes_ptr as *const u8, self.word_bytes_len) }
    }

    #[allow(dead_code)] // Used by MmapWordlistBatchReader (test-only sequential path)
    pub(crate) fn can_fit(&self, line_len: usize) -> bool {
        batch_shape_can_fit(self.candidate_count, self.word_bytes_len, line_len)
    }

    /// Append a single candidate to the batch.
    ///
    /// ## The unsafe block — a detailed walkthrough
    ///
    /// This is one of the most important `unsafe` blocks in the codebase. Let's
    /// break down exactly what it does and why it's sound:
    ///
    /// ```text
    /// *(self.word_offsets_ptr as *mut u32).add(index) = offset;
    /// ```
    /// This writes the byte offset of this candidate into the offsets table.
    /// - `word_offsets_ptr` is the base of the Metal buffer (cached at construction)
    /// - We cast to `*mut u32` because the buffer stores 32-bit offsets
    /// - `.add(index)` advances by `index * size_of::<u32>()` bytes (pointer arithmetic)
    /// - We dereference and assign — this is a raw memory write with no bounds check
    ///
    /// Why is this safe? Because:
    /// - `index < MAX_CANDIDATES_PER_BATCH` (checked by `can_fit` + debug_assert)
    /// - The buffer was allocated with capacity for exactly MAX_CANDIDATES_PER_BATCH u32s
    /// - `&mut self` guarantees exclusive access (no data races)
    ///
    /// The `copy_nonoverlapping` call is essentially a `memcpy` — it copies the
    /// candidate bytes directly into the word_bytes buffer at the current cursor.
    /// This is faster than a safe slice copy because the compiler can skip
    /// bounds checks (we already validated them above).
    #[allow(dead_code)] // Used by MmapWordlistBatchReader (test-only sequential path)
    pub(crate) fn push_candidate(&mut self, line: &[u8]) -> anyhow::Result<()> {
        if line.len() > u16::MAX as usize {
            bail!("wordlist entry exceeds {} bytes", u16::MAX);
        }
        if !self.can_fit(line.len()) {
            bail!("batch capacity exceeded while packing candidate");
        }

        // Offsets are byte offsets into the packed `word_bytes` blob, not global
        // wordlist indices. The kernel uses (offset, length) to reconstruct each
        // candidate inside the contiguous payload.
        let offset = u32::try_from(self.word_bytes_len).context("packed word bytes exceed u32")?;
        let index = self.candidate_count;
        let line_len = line.len();
        debug_assert!(index < MAX_CANDIDATES_PER_BATCH);
        debug_assert!(self.word_bytes_len + line_len <= MAX_WORD_BYTES_PER_BATCH);

        // SAFETY: capacities are fixed to the batch caps, `index` and
        // `word_bytes_len..word_bytes_len+line_len` are in-bounds, and writes are
        // serialized by exclusive `&mut self`.
        //
        // Layout invariant written here (must match Metal kernel expectations):
        // `offsets[i]` = start in `word_bytes`, `lengths[i]` = candidate length,
        // and `word_bytes` contains candidates concatenated with no separators.
        unsafe {
            *self.word_offsets_ptr.add(index) = offset;
            *self.word_lengths_ptr.add(index) = line_len as u16;
            std::ptr::copy_nonoverlapping(
                line.as_ptr(),
                self.word_bytes_ptr.add(self.word_bytes_len),
                line_len,
            );
        }

        self.candidate_count += 1;
        self.word_bytes_len += line_len;
        self.max_word_len = self.max_word_len.max(line_len as u16);
        Ok(())
    }

    /// Set word_bytes_len and max_word_len from pre-computed values.
    #[allow(dead_code)]
    pub(crate) fn set_plan_metadata(&mut self, word_bytes_len: usize, max_word_len: u16) {
        self.word_bytes_len = word_bytes_len;
        self.max_word_len = max_word_len;
    }

    /// Raw mutable pointer to the offsets buffer for direct bulk writes.
    ///
    /// Used by the Linux producer's `flush_staged_to_batch` to write
    /// candidate offsets directly into pinned GPU memory via memcpy,
    /// bypassing the per-candidate `push_candidate` path.
    #[allow(dead_code)]
    pub(crate) fn word_offsets_ptr_mut(&mut self) -> *mut u32 {
        self.word_offsets_ptr
    }

    /// Raw mutable pointer to the lengths buffer for direct bulk writes.
    #[allow(dead_code)]
    pub(crate) fn word_lengths_ptr_mut(&mut self) -> *mut u16 {
        self.word_lengths_ptr
    }

    /// Raw mutable pointer to the word_bytes buffer for direct bulk writes.
    ///
    /// Used in per-batch copy mode by the Linux producer to copy candidate
    /// bytes from the mmap into the batch's pinned buffer.
    #[allow(dead_code)]
    pub(crate) fn word_bytes_ptr_mut(&mut self) -> *mut u8 {
        self.word_bytes_ptr
    }

    /// Update logical cursors after a bulk write of pre-staged metadata.
    ///
    /// Called by `flush_staged_to_batch` after writing offsets, lengths,
    /// and (in per-batch mode) word bytes directly into the batch's pinned
    /// GPU memory. The arguments reflect the new state of the batch:
    ///   - `count`: number of candidates just flushed
    ///   - `word_bytes_len`: total word bytes in the batch after flush
    ///   - `max_word_len`: longest candidate in the batch (for kernel selection)
    #[allow(dead_code)]
    pub(crate) fn set_staged_counts(
        &mut self,
        count: usize,
        word_bytes_len: usize,
        max_word_len: u16,
    ) {
        self.candidate_count += count;
        self.word_bytes_len = word_bytes_len;
        self.max_word_len = max_word_len;
    }

    /// Reconstruct a candidate as a byte slice from the packed storage.
    ///
    /// This is the inverse of `push_candidate`: given a batch-local index,
    /// it looks up the offset and length, then returns a slice into
    /// `word_bytes`. This is used after the GPU reports a match index —
    /// the GPU returns a batch-local `u32`, and we use it here to recover
    /// the actual password candidate text.
    ///
    /// Returns `None` if the index is out of bounds (which would indicate
    /// a bug in the GPU kernel or result decoding).
    ///
    /// **Note**: On Linux (zero-copy path), offsets are absolute mmap
    /// positions and word_bytes is not populated. Use `word_from_source()`
    /// with the mmap instead.
    // Reconstruct a candidate slice from the packed storage.
    // Offsets and lengths are guaranteed to have matching indices by the batch
    // builders, so any `None` here indicates a bug or invalid GPU result.
    pub(crate) fn word(&self, local_index: usize) -> Option<&[u8]> {
        let start = *self.offsets_slice().get(local_index)? as usize;
        let len = *self.lengths_slice().get(local_index)? as usize;
        self.word_bytes_slice().get(start..start + len)
    }

    /// Reconstruct a candidate from an external source buffer using stored offsets.
    ///
    /// On Linux (zero-copy path), `push_segment_bulk` writes absolute mmap
    /// offsets and does not copy candidate bytes into `word_bytes`. This
    /// method reads from the original source (the mmap) using those absolute
    /// offsets.
    #[allow(dead_code)] // Used on Linux and in tests
    pub(crate) fn word_from_source<'a>(&self, local_index: usize, source: &'a [u8]) -> Option<&'a [u8]> {
        let start = *self.offsets_slice().get(local_index)? as usize;
        let len = *self.lengths_slice().get(local_index)? as usize;
        source.get(start..start + len)
    }

    // Human-readable reconstruction for final reporting. We keep this lossy to
    // avoid crashing on non-UTF-8 wordlist entries while still printing a key.
    #[allow(dead_code)] // Used on macOS
    pub(crate) fn word_string_lossy(&self, local_index: usize) -> Option<String> {
        Some(String::from_utf8_lossy(self.word(local_index)?).into_owned())
    }

    /// Reconstruct a candidate string from an external source (mmap).
    /// Used on Linux (zero-copy path) for match reporting.
    #[allow(dead_code)] // Used on Linux
    pub(crate) fn word_string_lossy_from_source(&self, local_index: usize, source: &[u8]) -> Option<String> {
        Some(String::from_utf8_lossy(self.word_from_source(local_index, source)?).into_owned())
    }

    /// Bulk-append a contiguous run of candidates from parsed chunk metadata.
    ///
    /// This is the **high-performance** path used by the production pipeline
    /// (as opposed to `push_candidate` which is the simpler per-line path).
    ///
    /// ## Why a bulk path exists
    ///
    /// `push_candidate` does per-candidate work: `can_fit()` check, `u32::try_from`,
    /// error wrapping. At 6M+ candidates per batch, that overhead adds up. The
    /// bulk path validates once up front (at plan time) and then runs a tight
    /// `unsafe` loop that does nothing but pointer writes — no branches, no
    /// error handling, no allocations.
    ///
    /// ## How it reads from the mmap
    ///
    /// The wordlist file is memory-mapped (`mmap`), and the parser has already
    /// identified line boundaries within each "chunk" of the mmap. The chunk
    /// metadata provides:
    ///   - `chunk_start`: byte offset of this chunk within the mmap
    ///   - `chunk_offsets_rel[i]`: byte offset of line `i` relative to chunk_start
    ///   - `chunk_lengths[i]`: byte length of line `i`
    ///
    /// So the absolute position of line `i` in the mmap is:
    ///   `mmap[chunk_start + chunk_offsets_rel[i] .. + chunk_lengths[i]]`
    ///
    /// ## The unsafe loop
    ///
    /// The inner loop uses `get_unchecked` to skip bounds checks on the chunk
    /// metadata arrays. This is sound because the planner has already verified
    /// that `line_start..line_end` is within the chunk's line count. In a loop
    /// running millions of iterations, bounds checks on every array access would
    /// be a significant cost.
    ///
    /// # Safety contract
    /// Caller must guarantee:
    /// - `chunk_offsets_rel[i] + chunk_start` and `chunk_lengths[i]` are valid
    ///   mmap byte ranges for all `i` in `line_start..line_end`
    /// - The total bytes fit in the batch (checked here via debug_assert)
    #[cfg_attr(all(target_os = "linux", not(test)), allow(dead_code))]
    pub(crate) fn push_segment_bulk(
        &mut self,
        mmap: &[u8],
        chunk_start: usize,
        chunk_offsets_rel: &[u32],
        chunk_lengths: &[u16],
        line_start: usize,
        line_end: usize,
    ) {
        let count = line_end - line_start;
        if count == 0 {
            return;
        }

        // Defense-in-depth: validate preconditions before entering unsafe code.
        //
        // The planner already guarantees these invariants (it sizes segments to
        // respect batch caps and only produces valid line ranges). These asserts
        // are *redundant* with the planner logic — they exist to catch bugs in
        // the planner itself, not in normal operation. If the planner ever
        // miscalculates a segment boundary, these fire immediately with a clear
        // message instead of silently corrupting GPU buffer memory.
        //
        // Cost: O(1) per bulk call (three integer comparisons), not per
        // candidate. At ~450 calls per 112 GB wordlist, the total overhead is
        // immeasurable — well under a microsecond.
        assert!(
            self.candidate_count + count <= MAX_CANDIDATES_PER_BATCH,
            "push_segment_bulk: candidate count {} + {} would exceed batch cap {}",
            self.candidate_count, count, MAX_CANDIDATES_PER_BATCH
        );
        assert!(
            line_end <= chunk_offsets_rel.len(),
            "push_segment_bulk: line_end {} exceeds chunk_offsets_rel length {}",
            line_end, chunk_offsets_rel.len()
        );
        assert!(
            line_end <= chunk_lengths.len(),
            "push_segment_bulk: line_end {} exceeds chunk_lengths length {}",
            line_end, chunk_lengths.len()
        );

        let base_idx = self.candidate_count;
        let wb_cursor = self.word_bytes_len;

        // SAFETY: all capacity checks were performed at plan time via
        // `batch_shape_can_fit`. The plan guarantees candidate_count and
        // word_bytes_len will not exceed the batch caps.
        unsafe {
            let offsets_base = self.word_offsets_ptr.add(base_idx);
            let lengths_base = self.word_lengths_ptr.add(base_idx);

            // BULK METADATA: copy lengths in one memcpy (contiguous u16 slice).
            std::ptr::copy_nonoverlapping(
                chunk_lengths.as_ptr().add(line_start),
                lengths_base,
                count,
            );

            // Copy candidate bytes into the batch's word_bytes buffer (single
            // bulk memcpy including inter-candidate gaps), then write batch-
            // relative offsets.
            let wb_base = self.word_bytes_ptr;
            let seg_first_offset =
                *chunk_offsets_rel.get_unchecked(line_start) as usize;
            let last_idx = line_end - 1;
            let seg_end_offset = *chunk_offsets_rel.get_unchecked(last_idx) as usize
                + *chunk_lengths.get_unchecked(last_idx) as usize;
            let seg_byte_range = seg_end_offset - seg_first_offset;

            std::ptr::copy_nonoverlapping(
                mmap.as_ptr().add(chunk_start + seg_first_offset),
                wb_base.add(wb_cursor),
                seg_byte_range,
            );

            // Offsets: copy from chunk-relative, then adjust to batch-relative.
            std::ptr::copy_nonoverlapping(
                chunk_offsets_rel.as_ptr().add(line_start),
                offsets_base,
                count,
            );
            let adjustment = (wb_cursor as u32).wrapping_sub(seg_first_offset as u32);
            let offsets_slice = std::slice::from_raw_parts_mut(offsets_base, count);
            for o in offsets_slice.iter_mut() {
                *o = o.wrapping_add(adjustment);
            }
        }

        self.candidate_count += count;

        {
            // Account for the bulk-copied bytes (including inter-candidate gaps).
            let seg_first_offset = chunk_offsets_rel[line_start] as usize;
            let last_idx = line_end - 1;
            let seg_end_offset = chunk_offsets_rel[last_idx] as usize
                + chunk_lengths[last_idx] as usize;
            self.word_bytes_len = wb_cursor + (seg_end_offset - seg_first_offset);

            let seg_max = chunk_lengths[line_start..line_end]
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            self.max_word_len = self.max_word_len.max(seg_max);

            debug_assert!(
                self.word_bytes_len <= WORD_BYTES_ALLOC_SIZE,
                "push_segment_bulk: word_bytes_len {} exceeds alloc size {}",
                self.word_bytes_len, WORD_BYTES_ALLOC_SIZE
            );
        }
    }

    /// Create a zero-copy view of this batch for GPU dispatch.
    ///
    /// ## Zero-copy dispatch pattern
    ///
    /// `DispatchBatchView` borrows the Metal buffers without copying any data.
    /// The lifetime `'_` ties the view to `&self`, so Rust's borrow checker
    /// guarantees the buffers cannot be mutated (via `reset_for_reuse` or
    /// `push_candidate`) while a view exists.
    ///
    /// This is a classic Rust pattern: use borrowing to get a "window" into
    /// data without copying, while the type system prevents use-after-free
    /// and data races at compile time.
    pub(crate) fn as_dispatch_view(&self) -> DispatchBatchView<'_> {
        // Views let us dispatch or autotune against the same buffers without
        // cloning `WordBatch` or copying any payload bytes.
        DispatchBatchView {
            candidate_count: self.candidate_count,
            max_word_len: self.max_word_len(),
            word_bytes_len: self.word_bytes_len,
            word_bytes_buf: self.word_bytes_buffer(),
            word_offsets_buf: self.offsets_buffer(),
            word_lengths_buf: self.lengths_buffer(),
        }
    }

    /// Create a view of only the first `sample_count` candidates.
    ///
    /// Used by the autotune system to benchmark GPU kernel performance on a
    /// small prefix of a real batch. We recompute `max_word_len` over just the
    /// sampled prefix so the autotuner selects the same kernel variant it would
    /// use for that subset size.
    ///
    /// Note: the Metal buffers still contain ALL candidates — we just tell the
    /// GPU to only process the first `sample_count`. The offset/length tables
    /// are indexed 0..sample_count, so extra data past the end is never read.
    #[allow(dead_code)] // Used on macOS for autotune
    pub(crate) fn prefix_dispatch_view(
        &self,
        sample_count: usize,
    ) -> Option<DispatchBatchView<'_>> {
        if sample_count == 0 || sample_count > self.candidate_count {
            return None;
        }
        // Recompute the maximum length for the sampled prefix so autotune picks
        // the same kernel variant (short-key vs mixed) it would use for that
        // subset during a real dispatch.
        let lengths = self.lengths_slice();
        let max_word_len = lengths[..sample_count].iter().copied().max().unwrap_or(0);
        Some(DispatchBatchView {
            candidate_count: sample_count,
            max_word_len,
            word_bytes_len: self.word_bytes_len,
            word_bytes_buf: &self.word_bytes_buf,
            word_offsets_buf: &self.word_offsets_buf,
            word_lengths_buf: &self.word_lengths_buf,
        })
    }
}

/// A borrowed, read-only view of a `WordBatch` for GPU dispatch.
///
/// ## Why a separate view type?
///
/// The GPU dispatch code needs the Metal buffer references and some metadata,
/// but it should NOT be able to mutate the batch (push candidates, reset, etc.).
/// By extracting a `DispatchBatchView` that only holds `&metal::Buffer`
/// references, we use Rust's type system to enforce read-only access at
/// compile time.
///
/// ## Zero-copy and lifetimes
///
/// The `'a` lifetime parameter ties this view to the `WordBatch` it borrows
/// from. The Rust compiler will reject any code that tries to use a
/// `DispatchBatchView` after the underlying `WordBatch` has been mutated or
/// dropped. This is a compile-time guarantee — no runtime cost.
///
/// ## `Clone + Copy`
///
/// Since this struct contains only a `usize`, a `u16`, and three `&`-references
/// (all of which are `Copy`), the whole view is `Copy`. This means it can be
/// passed around freely without any allocation or reference counting.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Fields used on macOS (Metal) and CUDA paths
pub(crate) struct DispatchBatchView<'a> {
    pub(crate) candidate_count: usize,
    pub(crate) max_word_len: u16,
    /// Number of valid bytes in the word_bytes buffer.
    ///
    /// Used by the CUDA per-batch copy path to know how many bytes to
    /// transfer from pinned host memory to the device word_bytes buffer.
    /// In zero-copy mode this value is informational only (the kernel
    /// reads from the pre-uploaded mmap, not from per-batch word_bytes).
    pub(crate) word_bytes_len: usize,
    pub(crate) word_bytes_buf: &'a GpuBuffer,
    pub(crate) word_offsets_buf: &'a GpuBuffer,
    pub(crate) word_lengths_buf: &'a GpuBuffer,
}

#[cfg(test)]
mod tests {
    use super::super::test_support::test_device;
    use super::*;

    #[test]
    fn word_batch_push_candidate_tracks_layout() {
        let device = test_device();
        let mut batch = WordBatch::new(&device, 42);
        batch.push_candidate(b"alpha").unwrap();
        batch.push_candidate(b"be").unwrap();

        assert_eq!(batch.candidate_index_base, 42);
        assert_eq!(batch.candidate_count(), 2);
        assert_eq!(batch.word_bytes_len(), 7);
        assert_eq!(batch.max_word_len(), 5);
        assert_eq!(batch.offsets_slice(), &[0, 5]);
        assert_eq!(batch.lengths_slice(), &[5, 2]);
        assert_eq!(batch.word_bytes_slice(), b"alphabe");
        assert_eq!(batch.word(0).unwrap(), b"alpha");
        assert_eq!(batch.word(1).unwrap(), b"be");
    }

    #[test]
    fn batch_shape_can_fit_enforces_capacity_rules() {
        assert!(batch_shape_can_fit(0, 0, 1));
        assert!(!batch_shape_can_fit(MAX_CANDIDATES_PER_BATCH, 0, 1));
        assert!(batch_shape_can_fit(0, MAX_WORD_BYTES_PER_BATCH, 1));
        assert!(!batch_shape_can_fit(1, MAX_WORD_BYTES_PER_BATCH, 1));
    }
}
