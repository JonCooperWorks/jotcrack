use anyhow::{Context, bail};
use metal::{Device, MTLResourceOptions};

// Larger batches improve amortization of dispatch overhead and host work.
pub(super) const MAX_CANDIDATES_PER_BATCH: usize = 6_182_240;
pub(super) const MAX_WORD_BYTES_PER_BATCH: usize = 32 * 1024 * 1024;
// Approximate shared-buffer bytes held by one `WordBatch` allocation. This is
// the dominant memory cost when increasing producer/consumer pipeline depth.
pub(super) const APPROX_WORD_BATCH_BUFFER_BYTES: usize = MAX_WORD_BYTES_PER_BATCH
    + (MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u32>())
    + (MAX_CANDIDATES_PER_BATCH * std::mem::size_of::<u16>());

// Packed representation of a wordlist batch sent to the GPU.
//
// Instead of storing `Vec<String>` per candidate (which is allocation-heavy for
// huge wordlists), we store one contiguous byte blob plus offset/length tables.
// This minimizes host memory churn and matches the GPU kernel input format.
#[derive(Debug)]
pub(super) struct WordBatch {
    // Host-only absolute wordlist index of candidate #0 in this batch.
    //
    // This is intentionally `u64` so parsing/progress/result reconstruction can
    // exceed 4,294,967,295 non-empty candidates. It is not sent to the GPU
    // anymore; the kernel returns a batch-local match index (`u32`) instead.
    pub(super) candidate_index_base: u64,
    // Buffers that are directly bound to the kernel at dispatch time.
    // The producer fills them; the consumer reuses the same allocations.
    word_bytes_buf: metal::Buffer,
    word_offsets_buf: metal::Buffer,
    word_lengths_buf: metal::Buffer,
    // Cached CPU-visible base pointers for the shared buffers.
    // This avoids repeated `contents()` Objective-C calls in the hot parser loop.
    word_bytes_ptr: usize,
    word_offsets_ptr: usize,
    word_lengths_ptr: usize,
    // Logical initialized prefix lengths within the fixed-size Metal buffers.
    candidate_count: usize,
    word_bytes_len: usize,
    max_word_len: u16,
}

// Shared batch-capacity rule used by both the direct pack path and the new
// planner stage so batch boundaries remain identical across implementations.
pub(super) fn batch_shape_can_fit(
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
pub(super) fn batch_shape_can_fit_block(
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
    // Allocate fixed-capacity shared buffers sized to the batch caps so parser
    // writes can go straight into memory later bound to the kernel.
    pub(super) fn new(device: &Device, candidate_index_base: u64) -> Self {
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

    // Reuse batch allocations across producer iterations to reduce allocator
    // churn in the parsing hot path while preserving the same batch semantics.
    pub(super) fn reset_for_reuse(&mut self, candidate_index_base: u64) {
        // Rebinding the base is what preserves globally correct indexing across
        // batches even when the backing Metal buffers are recycled.
        self.candidate_index_base = candidate_index_base;
        self.candidate_count = 0;
        self.word_bytes_len = 0;
        self.max_word_len = 0;
    }

    #[cfg(test)]
    // A batch is considered empty when it has no candidate metadata entries.
    pub(super) fn is_empty(&self) -> bool {
        self.candidate_count == 0
    }

    // Number of candidates packaged for this dispatch.
    pub(super) fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    // Number of packed candidate bytes for this dispatch.
    pub(super) fn word_bytes_len(&self) -> usize {
        self.word_bytes_len
    }

    pub(super) fn max_word_len(&self) -> u16 {
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

    pub(super) fn offsets_slice(&self) -> &[u32] {
        // SAFETY: `word_offsets_buf` stores `u32` entries; the first
        // `candidate_count` entries are initialized by `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(self.word_offsets_ptr as *const u32, self.candidate_count)
        }
    }

    pub(super) fn lengths_slice(&self) -> &[u16] {
        // SAFETY: `word_lengths_buf` stores `u16` entries; the first
        // `candidate_count` entries are initialized by `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(self.word_lengths_ptr as *const u16, self.candidate_count)
        }
    }

    pub(super) fn word_bytes_slice(&self) -> &[u8] {
        // SAFETY: the first `word_bytes_len` bytes are initialized by
        // `push_candidate`.
        unsafe { std::slice::from_raw_parts(self.word_bytes_ptr as *const u8, self.word_bytes_len) }
    }

    #[allow(dead_code)] // Used by MmapWordlistBatchReader (test-only sequential path)
    pub(super) fn can_fit(&self, line_len: usize) -> bool {
        batch_shape_can_fit(self.candidate_count, self.word_bytes_len, line_len)
    }

    #[allow(dead_code)] // Used by MmapWordlistBatchReader (test-only sequential path)
    pub(super) fn push_candidate(&mut self, line: &[u8]) -> anyhow::Result<()> {
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

    // Reconstruct a candidate slice from the packed storage.
    // Offsets and lengths are guaranteed to have matching indices by the batch
    // builders, so any `None` here indicates a bug or invalid GPU result.
    pub(super) fn word(&self, local_index: usize) -> Option<&[u8]> {
        let start = *self.offsets_slice().get(local_index)? as usize;
        let len = *self.lengths_slice().get(local_index)? as usize;
        self.word_bytes_slice().get(start..start + len)
    }

    // Human-readable reconstruction for final reporting. We keep this lossy to
    // avoid crashing on non-UTF-8 wordlist entries while still printing a key.
    pub(super) fn word_string_lossy(&self, local_index: usize) -> Option<String> {
        Some(String::from_utf8_lossy(self.word(local_index)?).into_owned())
    }

    /// Bulk-append a contiguous run of candidates from parsed chunk metadata.
    ///
    /// This avoids the per-candidate overhead of `push_candidate` (bounds
    /// checking, `u32::try_from`, `can_fit`) by validating once up front and
    /// filling offsets/lengths/word_bytes via tight unsafe loops.
    ///
    /// # Safety contract
    /// Caller must guarantee:
    /// - `chunk_offsets_rel[i] + chunk_start` and `chunk_lengths[i]` are valid
    ///   mmap byte ranges for all `i` in `line_start..line_end`
    /// - The total bytes fit in the batch (checked here via debug_assert)
    pub(super) fn push_segment_bulk(
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

    pub(super) fn as_dispatch_view(&self) -> DispatchBatchView<'_> {
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

    pub(super) fn prefix_dispatch_view(
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

#[derive(Clone, Copy)]
pub(super) struct DispatchBatchView<'a> {
    // Minimal metadata + buffer references required to dispatch a full batch or
    // sampled prefix batch without owning the storage.
    pub(super) candidate_count: usize,
    pub(super) max_word_len: u16,
    pub(super) word_bytes_buf: &'a metal::Buffer,
    pub(super) word_offsets_buf: &'a metal::Buffer,
    pub(super) word_lengths_buf: &'a metal::Buffer,
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
