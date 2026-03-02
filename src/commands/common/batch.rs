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
use metal::{Device, MTLResourceOptions};

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
// Approximate shared-buffer bytes held by one `WordBatch` allocation. This is
// the dominant memory cost when increasing producer/consumer pipeline depth.
//
// Breakdown:
//   - word_bytes buffer:   32 MiB                        (MAX_WORD_BYTES_PER_BATCH)
//   - word_offsets buffer: 6.2M × 4 bytes ≈ 23.6 MiB    (one u32 per candidate)
//   - word_lengths buffer: 6.2M × 2 bytes ≈ 11.8 MiB    (one u16 per candidate)
//   - Total: ~67 MiB per WordBatch
//
// When the pipeline depth is N, we have up to N WordBatch objects alive at once,
// so total GPU buffer memory is roughly N × 67 MiB. This is why pipeline_depth
// is a tunable parameter — deeper pipelines trade memory for better overlap.
pub(crate) const APPROX_WORD_BATCH_BUFFER_BYTES: usize = MAX_WORD_BYTES_PER_BATCH
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

    // --- Metal shared buffers ---
    //
    // These are the actual GPU-visible allocations. On Apple Silicon, "shared"
    // means the same physical memory page is mapped into both CPU and GPU address
    // spaces (unified memory architecture). The CPU writes candidate data here,
    // and the GPU reads it directly — zero copies needed.
    //
    // Buffers that are directly bound to the kernel at dispatch time.
    // The producer fills them; the consumer reuses the same allocations.
    word_bytes_buf: metal::Buffer,
    word_offsets_buf: metal::Buffer,
    word_lengths_buf: metal::Buffer,

    // --- Cached raw pointers (performance optimization) ---
    //
    // `metal::Buffer::contents()` returns the CPU-visible base pointer of the
    // buffer. Under the hood, this is an Objective-C method call:
    //   `[buffer contents]` → objc_msgSend → pointer
    //
    // In a hot loop pushing millions of candidates, the overhead of repeated
    // Objective-C message dispatch is measurable. Since the pointer is stable
    // for the lifetime of the buffer (it never moves — Metal guarantees this
    // for shared storage mode), we cache it as a plain `usize` at construction
    // time and use raw pointer arithmetic in `push_candidate`.
    //
    // Why `usize` instead of `*mut u8`? It's a stylistic choice — `usize` avoids
    // accidentally dereferencing outside an `unsafe` block and makes the "this is
    // just a number" nature explicit. We cast back to `*mut T` only inside the
    // unsafe blocks where we actually write.
    //
    // Cached CPU-visible base pointers for the shared buffers.
    // This avoids repeated `contents()` Objective-C calls in the hot parser loop.
    word_bytes_ptr: usize,
    word_offsets_ptr: usize,
    word_lengths_ptr: usize,

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
    pub(crate) fn new(device: &Device, candidate_index_base: u64) -> Self {
        let options = MTLResourceOptions::StorageModeShared;
        let word_bytes_buf = device.new_buffer(MAX_WORD_BYTES_PER_BATCH as u64, options);
        let word_offsets_buf = device.new_buffer(
            (MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u32>()) as u64,
            options,
        );
        let word_lengths_buf = device.new_buffer(
            (MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u16>()) as u64,
            options,
        );
        // `contents()` is stable for the lifetime of each buffer allocation, so
        // we cache the pointers once and use raw pointer math in `push_candidate`.
        Self {
            candidate_index_base,
            word_bytes_ptr: word_bytes_buf.contents() as usize,
            word_offsets_ptr: word_offsets_buf.contents() as usize,
            word_lengths_ptr: word_lengths_buf.contents() as usize,
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

    fn offsets_buffer(&self) -> &metal::Buffer {
        &self.word_offsets_buf
    }

    fn lengths_buffer(&self) -> &metal::Buffer {
        &self.word_lengths_buf
    }

    fn word_bytes_buffer(&self) -> &metal::Buffer {
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
            *(self.word_offsets_ptr as *mut u32).add(index) = offset;
            *(self.word_lengths_ptr as *mut u16).add(index) = line_len as u16;
            std::ptr::copy_nonoverlapping(
                line.as_ptr(),
                (self.word_bytes_ptr as *mut u8).add(self.word_bytes_len),
                line_len,
            );
        }

        self.candidate_count += 1;
        self.word_bytes_len += line_len;
        self.max_word_len = self.max_word_len.max(line_len as u16);
        Ok(())
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
    // Reconstruct a candidate slice from the packed storage.
    // Offsets and lengths are guaranteed to have matching indices by the batch
    // builders, so any `None` here indicates a bug or invalid GPU result.
    pub(crate) fn word(&self, local_index: usize) -> Option<&[u8]> {
        let start = *self.offsets_slice().get(local_index)? as usize;
        let len = *self.lengths_slice().get(local_index)? as usize;
        self.word_bytes_slice().get(start..start + len)
    }

    // Human-readable reconstruction for final reporting. We keep this lossy to
    // avoid crashing on non-UTF-8 wordlist entries while still printing a key.
    pub(crate) fn word_string_lossy(&self, local_index: usize) -> Option<String> {
        Some(String::from_utf8_lossy(self.word(local_index)?).into_owned())
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

        let base_idx = self.candidate_count;
        let mut wb_cursor = self.word_bytes_len;

        // SAFETY: all capacity checks were performed at plan time via
        // `batch_shape_can_fit`. The plan guarantees candidate_count and
        // word_bytes_len will not exceed the batch caps.
        unsafe {
            let offsets_base = (self.word_offsets_ptr as *mut u32).add(base_idx);
            let lengths_base = (self.word_lengths_ptr as *mut u16).add(base_idx);
            let wb_base = self.word_bytes_ptr as *mut u8;

            for i in 0..count {
                let src_idx = line_start + i;
                let len = *chunk_lengths.get_unchecked(src_idx) as usize;
                let src_offset =
                    chunk_start + *chunk_offsets_rel.get_unchecked(src_idx) as usize;

                *offsets_base.add(i) = wb_cursor as u32;
                *lengths_base.add(i) = len as u16;
                std::ptr::copy_nonoverlapping(
                    mmap.as_ptr().add(src_offset),
                    wb_base.add(wb_cursor),
                    len,
                );
                wb_cursor += len;
            }
        }

        // Update max_word_len by scanning the segment's lengths.
        let seg_max = chunk_lengths[line_start..line_end]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        self.max_word_len = self.max_word_len.max(seg_max);
        self.candidate_count += count;
        self.word_bytes_len = wb_cursor;
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
pub(crate) struct DispatchBatchView<'a> {
    // Minimal metadata + buffer references required to dispatch a full batch or
    // sampled prefix batch without owning the storage.
    pub(crate) candidate_count: usize,
    pub(crate) max_word_len: u16,
    pub(crate) word_bytes_buf: &'a metal::Buffer,
    pub(crate) word_offsets_buf: &'a metal::Buffer,
    pub(crate) word_lengths_buf: &'a metal::Buffer,
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
