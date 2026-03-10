use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use memchr::memchr;
use memmap2::{Advice, Mmap, MmapOptions};
use super::gpu::GpuDevice;

use super::args::ParserConfig;
use super::batch::{WordBatch, batch_shape_can_fit, batch_shape_can_fit_block};

#[cfg(test)]
use std::io::BufRead;

#[derive(Debug, Clone, Copy, Default)]
// Snapshot of parser-side counters/configuration reported by the producer.
//
// This is copied into `ProducerMessage` so the consumer can print final `STATS`
// without reaching into parser internals or sharing mutable state across
// threads.
pub(crate) struct ParserStats {
    pub(crate) parser_threads: usize,
    pub(crate) parser_chunk_bytes: usize,
    pub(crate) parser_chunks: u64,
    pub(crate) parser_skipped_oversize: u64,
}

// Shared helper used by parser tests to verify CRLF/LF parity.
#[cfg(test)]
pub(crate) fn trim_line_endings(bytes: &[u8]) -> &[u8] {
    let mut end = bytes.len();
    while end > 0 && (bytes[end - 1] == b'\n' || bytes[end - 1] == b'\r') {
        end -= 1;
    }
    &bytes[..end]
}

// Test-only streaming reader used for parser unit tests. Runtime uses mmap-only.
//
// Keeping this path test-only lets us use `Cursor` for parser behavior tests
// without carrying a slower runtime fallback that would skew performance tuning.
#[cfg(test)]
pub(crate) struct BufferedWordlistBatchReader<R: BufRead> {
    device: GpuDevice,
    reader: R,
    line_buf: Vec<u8>,
    pending_line: Option<Vec<u8>>,
    // Absolute index of the next non-empty candidate line the reader will emit.
    // `u64` avoids the old >4.29B-line overflow on giant wordlists.
    next_index: u64,
    done: bool,
}

#[cfg(test)]
impl<R: BufRead> BufferedWordlistBatchReader<R> {
    // Construct a buffered batch reader over any `BufRead` source.
    pub(crate) fn new(device: GpuDevice, reader: R) -> Self {
        Self {
            device,
            reader,
            line_buf: Vec::new(),
            pending_line: None,
            next_index: 0,
            done: false,
        }
    }

    // Build the next GPU batch by packing candidates into a contiguous byte
    // buffer plus offset/length tables. This is where most host-side parsing
    // time is spent, so allocation and copies are kept minimal.
    #[cfg(test)]
    pub(crate) fn next_batch(&mut self) -> anyhow::Result<Option<WordBatch>> {
        self.next_batch_reusing(None)
    }

    pub(crate) fn next_batch_reusing(
        &mut self,
        recycled: Option<WordBatch>,
    ) -> anyhow::Result<Option<WordBatch>> {
        if self.done {
            return Ok(None);
        }

        let candidate_index_base = self.next_index;
        let mut batch = if let Some(mut batch) = recycled {
            // Reuse the producer-owned allocation when available. This avoids
            // re-allocating large Metal buffers on every batch and preserves the
            // cached base pointers stored in `WordBatch`.
            //
            // Important invariant: recycled batches are only returned after the
            // consumer has finished GPU dispatch for that batch, so no aliasing
            // exists while we mutate the buffers again.
            batch.reset_for_reuse(candidate_index_base);
            batch
        } else {
            // Fresh allocation on startup or when the recycle queue is empty/full.
            WordBatch::new(&self.device, candidate_index_base)
        };

        loop {
            let line = if let Some(line) = self.pending_line.take() {
                // A line that did not fit in the previous batch is retried as
                // the first candidate of the next batch.
                line
            } else {
                self.line_buf.clear();
                let bytes_read = self
                    .reader
                    // Byte-based parsing avoids UTF-8 validation and temporary
                    // `String` allocations in the hot path.
                    .read_until(b'\n', &mut self.line_buf)
                    .context("failed to read wordlist")?;
                if bytes_read == 0 {
                    self.done = true;
                    break;
                }
                let trimmed = trim_line_endings(&self.line_buf);
                if trimmed.is_empty() {
                    continue;
                }
                trimmed.to_vec()
            };

            if line.len() > u16::MAX as usize {
                bail!(
                    "wordlist entry at candidate index {} exceeds {} bytes",
                    self.next_index,
                    u16::MAX
                );
            }

            if !batch.can_fit(line.len()) {
                // Preserve the line for the next call so batch boundaries do
                // not drop or duplicate candidates.
                //
                // Reuse does not change this rule: we still split on the same
                // boundaries as before, only the backing allocation differs.
                self.pending_line = Some(line);
                break;
            }

            batch.push_candidate(&line)?;
            self.next_index = self
                .next_index
                .checked_add(1)
                .ok_or_else(|| anyhow!("candidate index overflow"))?;
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }
}

// mmap-backed reader for high-throughput local files.
//
// mmap removes per-line read syscalls. We still copy candidate bytes once into
// the packed batch buffer because the kernel expects a contiguous payload.
//
// Runtime now uses the parallel parser. We keep this sequential reader as a
// test-only reference implementation so parser behavior can be compared in
// unit tests (especially ordering and line-trimming semantics).
#[cfg(test)]
pub(crate) struct MmapWordlistBatchReader {
    device: GpuDevice,
    mmap: Mmap,
    cursor: usize,
    // Same semantic as the test-only buffered reader: absolute index of the next
    // non-empty candidate line, tracked in `u64` for huge wordlists.
    next_index: u64,
}

#[cfg(test)]
impl MmapWordlistBatchReader {
    // Wrap a mapped wordlist file for sequential batch extraction.
    pub(crate) fn new(device: GpuDevice, mmap: Mmap) -> Self {
        Self {
            device,
            mmap,
            cursor: 0,
            next_index: 0,
        }
    }

    // Scan the mmap for newline-delimited candidates while preserving the same
    // batching semantics as the buffered reader.
    pub(crate) fn next_batch_reusing(
        &mut self,
        recycled: Option<WordBatch>,
    ) -> anyhow::Result<Option<WordBatch>> {
        let bytes = self.mmap.as_ref();
        if self.cursor >= bytes.len() {
            return Ok(None);
        }

        let mut batch = if let Some(mut batch) = recycled {
            // Same reuse strategy as the buffered reader: preserve capacity,
            // reset logical contents, and repopulate using identical rules.
            batch.reset_for_reuse(self.next_index);
            batch
        } else {
            WordBatch::new(&self.device, self.next_index)
        };
        while self.cursor < bytes.len() {
            // Find the next line without allocating intermediate strings.
            let line_start = self.cursor;
            let remaining = &bytes[line_start..];
            let (line_end, next_cursor) = match memchr(b'\n', remaining) {
                Some(rel_newline) => {
                    let line_end = line_start + rel_newline;
                    (line_end, line_end + 1)
                }
                None => (bytes.len(), bytes.len()),
            };
            // Match the buffered reader's CRLF behavior by trimming a trailing
            // carriage return before packaging the candidate.
            let mut trimmed_end = line_end;
            if trimmed_end > line_start && bytes[trimmed_end - 1] == b'\r' {
                trimmed_end -= 1;
            }
            let line_len = trimmed_end - line_start;

            if line_len == 0 {
                self.cursor = next_cursor;
                continue;
            }
            if line_len > u16::MAX as usize {
                bail!(
                    "wordlist entry at candidate index {} exceeds {} bytes",
                    self.next_index,
                    u16::MAX
                );
            }

            if !batch.can_fit(line_len) {
                // Unlike the buffered reader there is no `pending_line`: we can
                // simply leave `cursor` at the current line start and retry on
                // the next call.
                //
                // This preserves byte-for-byte behavior with the non-reuse path.
                break;
            }

            batch.push_candidate(&bytes[line_start..trimmed_end])?;
            self.cursor = next_cursor;
            self.next_index = self
                .next_index
                .checked_add(1)
                .ok_or_else(|| anyhow!("candidate index overflow"))?;
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }
}

#[derive(Debug, Clone, Copy)]
// Immutable work item describing one newline-aligned byte range in the mmap.
//
// Each worker receives a `ChunkJob`, scans that slice independently, and
// returns only metadata (offsets/lengths) so the coordinator can preserve
struct ChunkJob {
    chunk_id: u64,
    start: usize,
    end: usize,
}

// Optimization: block-summary planner
//
// The serial planner decides batch boundaries by checking whether each
// candidate line fits into the current batch (count cap + byte cap).
// With ~6.2M candidates per batch and ~70 batches/sec this was 434M
// loop iterations/sec on a single thread — a measurable bottleneck.
//
// Block summaries let the planner skip 4096 lines at a time.  During
// parsing we precompute (line_count, total_bytes, max_len) for every
// 4096-line block.  The planner first tries to consume whole blocks in
// O(1) each, falling back to per-line scanning only at the batch
// boundary.  This reduces inner-loop iterations from ~6.2M to ~5,600
// per batch (~1100× reduction).
//
// Correctness: the block check is conservative — if candidate_count +
// block.line_count and word_bytes + block.total_bytes both fit, then
// every intermediate state during line-by-line consumption also fits.
// So batch boundaries are identical to the per-line-only path.
const PLANNER_BLOCK_STRIDE: usize = 4096;

#[derive(Debug, Clone, Copy)]
struct BlockSummary {
    line_count: u32,
    total_bytes: u32,
    max_len: u16,
}

#[derive(Debug)]
// Parsed metadata for one chunk.
//
// Records line offsets/lengths relative to the chunk's start in the mmap so
// the packer can later copy candidate bytes into a `WordBatch`.
struct ParsedChunk {
    chunk_id: u64,
    chunk_start: usize,
    offsets_rel: Vec<u32>,
    lengths: Vec<u16>,
    skipped_oversize: u64,
    block_summaries: Vec<BlockSummary>,
}

#[derive(Debug)]
// Messages from parser workers back to the producer/coordinator thread.
//
// We send strings for errors (instead of `anyhow::Error`) to keep the channel
// payload simple and `Send` without additional wrappers.
enum ParserWorkerMessage {
    Parsed(ParsedChunk),
    Error { chunk_id: u64, message: String },
}

#[derive(Debug, Clone)]
// A contiguous range of candidate lines from one parsed chunk used to
// reconstruct a planned GPU batch without copying parser metadata.
struct BatchPlanSegment {
    parsed_chunk: Arc<ParsedChunk>,
    line_start: usize,
    line_end: usize,
}

#[derive(Debug, Clone)]
// Host-side batch plan produced by the ordered parser/planner stage and later
// materialized into a `WordBatch` by a packer worker.
pub(crate) struct BatchPlan {
    pub(crate) seq_no: u64,
    candidate_index_base: u64,
    candidate_count: usize,
    word_bytes_len: usize,
    max_word_len: u16,
    segments: Vec<BatchPlanSegment>,
    pub(crate) parser_stats: ParserStats,
    pub(crate) plan_time: Duration,
}

// Split the mapped file into newline-aligned chunks for parallel scanning.
//
// Important property: every byte belongs to exactly one chunk and chunk ends
// are extended to include the newline that terminates the last line in that
// chunk. This removes partial-line ambiguity across workers.
fn plan_mmap_chunks(bytes: &[u8], chunk_bytes: usize) -> anyhow::Result<Vec<ChunkJob>> {
    if chunk_bytes == 0 {
        bail!("parser chunk size must be > 0");
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut chunk_id = 0u64;
    while start < bytes.len() {
        let target_end = (start + chunk_bytes).min(bytes.len());
        let end = if target_end >= bytes.len() {
            bytes.len()
        } else {
            match memchr(b'\n', &bytes[target_end..]) {
                Some(rel_newline) => target_end + rel_newline + 1,
                None => bytes.len(),
            }
        };

        if end < start {
            bail!("invalid chunk planner state: chunk end before start");
        }

        chunks.push(ChunkJob {
            chunk_id,
            start,
            end,
        });
        chunk_id = chunk_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("parser chunk id overflow"))?;
        start = end;
    }
    Ok(chunks)
}

// Parse one newline-aligned chunk into compact line metadata.
//
// The parser trims a trailing `\\r`, skips empty lines, and (intentionally for
// throughput) skips oversized entries instead of failing the entire run.
fn parse_mmap_chunk(bytes: &[u8], job: ChunkJob) -> anyhow::Result<ParsedChunk> {
    if job.start > job.end || job.end > bytes.len() {
        bail!(
            "invalid parser chunk bounds {}..{} for file len {}",
            job.start,
            job.end,
            bytes.len()
        );
    }

    // Optimization: pre-allocated metadata Vecs
    //
    // Without with_capacity, pushing ~3.2M entries into an empty Vec causes
    // ~22 reallocations (doubling from 0 → 1 → 2 → 4 → ... → 4M), each
    // copying the entire vec.  Estimating capacity up front (chunk_bytes / 6
    // for ~5-byte avg candidates + newline) eliminates all growth copies.
    // Over-estimating by 2× wastes some temporary memory; under-estimating
    // by 2× causes a single extra reallocation — both are cheap compared to
    // 22 cascading copies.
    let chunk = &bytes[job.start..job.end];
    let estimated_lines = chunk.len() / 6;
    let mut offsets_rel = Vec::with_capacity(estimated_lines);
    let mut lengths: Vec<u16> = Vec::with_capacity(estimated_lines);
    let mut skipped_oversize = 0u64;
    let mut cursor = 0usize;

    while cursor < chunk.len() {
        let line_start = cursor;
        let remaining = &chunk[line_start..];
        let (line_end, next_cursor) = match memchr(b'\n', remaining) {
            Some(rel_newline) => {
                let line_end = line_start + rel_newline;
                (line_end, line_end + 1)
            }
            None => (chunk.len(), chunk.len()),
        };

        let mut trimmed_end = line_end;
        if trimmed_end > line_start && chunk[trimmed_end - 1] == b'\r' {
            trimmed_end -= 1;
        }
        let line_len = trimmed_end - line_start;

        if line_len == 0 {
            cursor = next_cursor;
            continue;
        }

        if line_len > u16::MAX as usize {
            skipped_oversize = skipped_oversize.saturating_add(1);
            cursor = next_cursor;
            continue;
        }

        offsets_rel.push(
            u32::try_from(line_start)
                .context("parser chunk relative offset exceeds u32 while recording line")?,
        );
        lengths.push(line_len as u16);
        cursor = next_cursor;
    }

    // Build block summaries for the fast planner (see "Optimization: block-
    // summary planner" above).  This is an O(n) pass over the lengths vec,
    // running on parallel parser worker threads, so it adds negligible cost
    // relative to the memchr-based line scanning that precedes it.
    let num_blocks = (lengths.len() + PLANNER_BLOCK_STRIDE - 1) / PLANNER_BLOCK_STRIDE;
    let mut block_summaries = Vec::with_capacity(num_blocks);
    for block_start in (0..lengths.len()).step_by(PLANNER_BLOCK_STRIDE) {
        let block_end = (block_start + PLANNER_BLOCK_STRIDE).min(lengths.len());
        let slice = &lengths[block_start..block_end];
        let line_count = slice.len() as u32;
        let total_bytes: u32 = slice.iter().map(|&l| l as u32).sum();
        let max_len: u16 = slice.iter().copied().max().unwrap_or(0);
        block_summaries.push(BlockSummary {
            line_count,
            total_bytes,
            max_len,
        });
    }

    Ok(ParsedChunk {
        chunk_id: job.chunk_id,
        chunk_start: job.start,
        offsets_rel,
        lengths,
        skipped_oversize,
        block_summaries,
    })
}

// Materialize a planned batch into an owned `WordBatch` allocation.
//
// This helper is used by both the direct reader path (tests/reference behavior)
// and the producer packer workers so packing logic stays identical.
pub(crate) fn pack_batch_plan_into_batch(
    mmap: &Mmap,
    plan: &BatchPlan,
    batch: &mut WordBatch,
) -> anyhow::Result<()> {
    batch.reset_for_reuse(plan.candidate_index_base);
    let bytes = mmap.as_ref();

    for segment in &plan.segments {
        let chunk = segment.parsed_chunk.as_ref();
        if segment.line_start > segment.line_end || segment.line_end > chunk.lengths.len() {
            bail!(
                "invalid batch plan segment {}..{} for chunk {} len {}",
                segment.line_start,
                segment.line_end,
                chunk.chunk_id,
                chunk.lengths.len()
            );
        }

        batch.push_segment_bulk(
            bytes,
            chunk.chunk_start,
            &chunk.offsets_rel,
            &chunk.lengths,
            segment.line_start,
            segment.line_end,
        );
    }

    if batch.candidate_count() != plan.candidate_count {
        bail!(
            "batch plan candidate count mismatch: packed {} planned {}",
            batch.candidate_count(),
            plan.candidate_count
        );
    }
    // With the bulk-copy optimization, word_bytes_len includes inter-candidate
    // newline gap bytes (the GPU ignores them via offset/length indexing).
    // The plan only counts pure candidate bytes, so packed >= planned.
    if batch.word_bytes_len() < plan.word_bytes_len {
        bail!(
            "batch plan packed byte length mismatch: packed {} planned {}",
            batch.word_bytes_len(),
            plan.word_bytes_len
        );
    }
    if batch.max_word_len() != plan.max_word_len {
        bail!(
            "batch plan max word len mismatch: packed {} planned {}",
            batch.max_word_len(),
            plan.max_word_len
        );
    }

    Ok(())
}

// Runtime wordlist reader that parallelizes newline scanning over an mmap file
// while keeping batch emission ordered and deterministic.
//
// Design overview:
// - worker threads parse chunks into metadata (`ParsedChunk`)
// - the producer thread (coordinator) reorders results by `chunk_id`
// - the producer thread alone packs bytes into `WordBatch` GPU buffers
//
// This keeps concurrent code focused on pure parsing and preserves the existing
// GPU dispatch contract (`WordBatch`) without parallel writes into Metal memory.
pub(crate) struct ParallelMmapWordlistBatchReader {
    #[cfg(test)]
    device: GpuDevice,
    mmap: Arc<Mmap>,
    // Precomputed newline-aligned chunk plan for the whole file.
    chunks: Vec<ChunkJob>,
    // Job scheduling cursors / limits.
    next_job_to_send: usize,
    next_chunk_to_emit: u64,
    in_flight_jobs: usize,
    max_in_flight_jobs: usize,
    // Coordinator -> workers and workers -> coordinator channels.
    job_tx: Option<crossbeam_channel::Sender<ChunkJob>>,
    result_rx: crossbeam_channel::Receiver<ParserWorkerMessage>,
    worker_handles: Vec<JoinHandle<()>>,
    // Optimization: pending ring for chunk reordering
    //
    // Parser workers return chunks out of order but we must emit them in
    // file order.  The old BTreeMap<u64, Arc<ParsedChunk>> had O(log n)
    // insert/remove and allocated a tree node per entry.  Because chunk
    // IDs are monotonically increasing and at most `max_in_flight_jobs`
    // can be in-flight simultaneously, we can use a fixed-size ring
    // indexed by `chunk_id % capacity`.  This gives O(1) insert/lookup,
    // zero heap allocations, and better cache locality for the small
    // window sizes used in practice (typically 8–60 entries).
    //
    // Safety: no two in-flight chunks map to the same slot because
    // capacity ≥ max_in_flight_jobs and chunk IDs are unique and
    // consumed in order, so the slot is always empty when a new chunk
    // arrives.
    pending_ring: Vec<Option<Arc<ParsedChunk>>>,
    pending_ring_capacity: usize,
    // Current parsed chunk being drained into a `WordBatch`.
    active_chunk: Option<Arc<ParsedChunk>>,
    active_chunk_line_cursor: usize,
    // Absolute candidate index of the next accepted (non-empty, non-oversize)
    // line to be emitted into a batch.
    next_index: u64,
    parser_stats: ParserStats,
}

impl ParallelMmapWordlistBatchReader {
    // Build the parser subsystem:
    // 1) map+plan chunks (already done before this call)
    // 2) create bounded channels (backpressure)
    // 3) spawn worker threads that parse chunk metadata only
    pub(crate) fn new(device: GpuDevice, mmap: Mmap, config: ParserConfig) -> anyhow::Result<Self> {
        #[cfg(not(test))]
        let _ = device;
        let mmap = Arc::new(mmap);
        let chunks = plan_mmap_chunks(mmap.as_ref(), config.chunk_bytes)?;
        let (job_tx, job_rx) = crossbeam_channel::bounded::<ChunkJob>(config.queue_capacity);
        let (result_tx, result_rx) = crossbeam_channel::bounded::<ParserWorkerMessage>(config.queue_capacity);
        let mut worker_handles = Vec::with_capacity(config.parser_threads);

        for _ in 0..config.parser_threads {
            let mmap_for_worker = Arc::clone(&mmap);
            let job_rx_for_worker = job_rx.clone();
            let result_tx_for_worker = result_tx.clone();
            worker_handles.push(thread::spawn(move || {
                // Each worker is a simple parse loop: recv job -> parse -> send
                // metadata. No shared mutable state is touched here.
                while let Ok(job) = job_rx_for_worker.recv() {
                    let chunk_len = job.end.saturating_sub(job.start);
                    // Prefetch chunk pages asynchronously before scanning.
                    // Best-effort: ignore failures (advisory only).
                    if chunk_len > 0 {
                        let _ = mmap_for_worker.advise_range(
                            Advice::WillNeed,
                            job.start,
                            chunk_len,
                        );
                    }
                    let message = match parse_mmap_chunk(mmap_for_worker.as_ref(), job) {
                        Ok(chunk) => ParserWorkerMessage::Parsed(chunk),
                        Err(err) => ParserWorkerMessage::Error {
                            chunk_id: job.chunk_id,
                            message: format!("{err:#}"),
                        },
                    };
                    if result_tx_for_worker.send(message).is_err() {
                        return;
                    }
                }
            }));
        }
        drop(job_rx);
        drop(result_tx);

        Ok(Self {
            #[cfg(test)]
            device,
            mmap,
            chunks,
            next_job_to_send: 0,
            next_chunk_to_emit: 0,
            in_flight_jobs: 0,
            max_in_flight_jobs: config.queue_capacity.max(1),
            job_tx: Some(job_tx),
            result_rx,
            worker_handles,
            pending_ring: (0..config.queue_capacity.max(1)).map(|_| None).collect(),
            pending_ring_capacity: config.queue_capacity.max(1),
            active_chunk: None,
            active_chunk_line_cursor: 0,
            next_index: 0,
            parser_stats: ParserStats {
                parser_threads: config.parser_threads,
                parser_chunk_bytes: config.chunk_bytes,
                ..ParserStats::default()
            },
        })
    }

    // Cheap snapshot consumed by the producer so progress/final reporting can
    // include parser counters without shared atomics or locks.
    pub(crate) fn parser_stats(&self) -> ParserStats {
        self.parser_stats
    }

    pub(crate) fn shared_mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }

    // Submit more jobs until we hit the in-flight limit or exhaust the chunk
    // plan. The bounded channel keeps memory growth predictable.
    //
    // In addition to the in-flight cap, we ensure that all outstanding chunk
    // IDs (in-flight + received-but-not-yet-consumed) fit within the pending
    // ring capacity. Without this, out-of-order results can wrap around and
    // collide with unconsumed ring slots, causing either silent mis-ordering
    // or a "duplicate parsed chunk id" bail.
    fn pump_jobs(&mut self) -> anyhow::Result<()> {
        while self.in_flight_jobs < self.max_in_flight_jobs
            && self.next_job_to_send < self.chunks.len()
            && (self.next_job_to_send as u64).saturating_sub(self.next_chunk_to_emit)
                < self.pending_ring_capacity as u64
        {
            let job = self.chunks[self.next_job_to_send];
            let tx = self
                .job_tx
                .as_ref()
                .ok_or_else(|| anyhow!("parser worker job queue already closed"))?;
            tx.send(job)
                .map_err(|_| anyhow!("parser worker job queue closed unexpectedly"))?;
            self.next_job_to_send += 1;
            self.in_flight_jobs = self.in_flight_jobs.saturating_add(1);
        }

        if self.next_job_to_send >= self.chunks.len() {
            let _ = self.job_tx.take();
        }
        Ok(())
    }

    // Merge one worker message into coordinator state.
    //
    // Successful chunks are recorded in `pending_results` and become eligible
    // for ordered emission once all earlier chunk ids have been drained.
    fn handle_worker_message(&mut self, message: ParserWorkerMessage) -> anyhow::Result<()> {
        self.in_flight_jobs = self.in_flight_jobs.saturating_sub(1);
        match message {
            ParserWorkerMessage::Parsed(chunk) => {
                self.parser_stats.parser_chunks = self.parser_stats.parser_chunks.saturating_add(1);
                self.parser_stats.parser_skipped_oversize = self
                    .parser_stats
                    .parser_skipped_oversize
                    .saturating_add(chunk.skipped_oversize);
                let chunk_id = chunk.chunk_id;
                let ring_idx = (chunk_id as usize) % self.pending_ring_capacity;
                if self.pending_ring[ring_idx].is_some() {
                    bail!("duplicate parsed chunk id received from parser workers");
                }
                self.pending_ring[ring_idx] = Some(Arc::new(chunk));
                Ok(())
            }
            ParserWorkerMessage::Error { chunk_id, message } => {
                bail!("parser worker failed on chunk {chunk_id}: {message}")
            }
        }
    }

    // Promote the next in-order parsed chunk (if already available) into the
    // active-drain slot used by `next_batch_reusing`.
    fn try_activate_ready_chunk(&mut self) -> bool {
        if self.active_chunk.is_some() {
            return true;
        }
        let ring_idx = (self.next_chunk_to_emit as usize) % self.pending_ring_capacity;
        if let Some(chunk) = self.pending_ring[ring_idx].take() {
            debug_assert_eq!(
                chunk.chunk_id, self.next_chunk_to_emit,
                "ring slot contained chunk {} but expected chunk {}",
                chunk.chunk_id, self.next_chunk_to_emit
            );
            self.active_chunk = Some(chunk);
            self.active_chunk_line_cursor = 0;
            return true;
        }
        false
    }

    // Ensure there is an active parsed chunk ready to drain, blocking on worker
    // results only when necessary.
    //
    // Returns `Ok(false)` only when every chunk has been emitted and no more
    // parser work is in flight.
    fn ensure_active_chunk(&mut self) -> anyhow::Result<bool> {
        loop {
            if self.try_activate_ready_chunk() {
                return Ok(true);
            }

            self.pump_jobs()?;
            if self.try_activate_ready_chunk() {
                return Ok(true);
            }

            let all_chunks_emitted = usize::try_from(self.next_chunk_to_emit)
                .ok()
                .is_some_and(|idx| idx >= self.chunks.len());
            if all_chunks_emitted && self.in_flight_jobs == 0 {
                return Ok(false);
            }

            let message = self
                .result_rx
                .recv()
                .map_err(|_| anyhow!("parser workers terminated unexpectedly"))?;
            self.handle_worker_message(message)?;
        }
    }

    // Build the next GPU batch by draining parsed chunk metadata in file order.
    //
    // Two-phase approach (see "Optimization: block-summary planner"):
    //
    //   Phase 1 — coarse:  Walk the block_summaries array.  For each block,
    //   check whether the entire block (line_count candidates, total_bytes
    //   bytes) fits in the remaining batch capacity.  If yes, advance the
    //   cursor by 4096 lines in one step.  If not, stop — the batch boundary
    //   falls somewhere inside this block.
    //
    //   Phase 2 — fine:  Scan the remaining lines one at a time using the
    //   original per-line capacity check until the batch is full or the chunk
    //   is exhausted.  At most PLANNER_BLOCK_STRIDE (4096) lines are scanned
    //   here, compared to ~6.2M in the old all-fine path.
    //
    // Edge case: the first candidate in a batch is exempt from the byte cap
    // (allows a single oversized candidate to form its own batch).  We handle
    // this by entering Phase 1 only after candidate_count > 0.
    pub(crate) fn next_batch_plan(&mut self) -> anyhow::Result<Option<BatchPlan>> {
        let plan_started = Instant::now();
        let candidate_index_base = self.next_index;
        let mut segments = Vec::new();
        let mut candidate_count = 0usize;
        let mut word_bytes_len = 0usize;
        let mut max_word_len = 0u16;

        loop {
            if !self.ensure_active_chunk()? {
                break;
            }

            let chunk = self
                .active_chunk
                .as_ref()
                .cloned()
                .ok_or_else(|| anyhow!("parser active chunk missing"))?;
            let segment_start = self.active_chunk_line_cursor;
            let mut line_cursor = segment_start;

            // Phase 1 (coarse): Skip entire blocks using precomputed summaries.
            // Only applies when the batch already has at least one candidate
            // (the first-candidate byte-cap exemption requires the fine path).
            if candidate_count > 0 {
                let mut block_idx = line_cursor / PLANNER_BLOCK_STRIDE;
                // Start from the first full block boundary at or after line_cursor.
                let mut block_line_start = block_idx * PLANNER_BLOCK_STRIDE;
                // If line_cursor is mid-block, skip to the next full block.
                if block_line_start < line_cursor {
                    block_idx += 1;
                    block_line_start = block_idx * PLANNER_BLOCK_STRIDE;
                }
                // Scan ahead with per-line checks to reach the next block boundary.
                while line_cursor < block_line_start && line_cursor < chunk.lengths.len() {
                    let line_len = chunk.lengths[line_cursor] as usize;
                    if !batch_shape_can_fit(candidate_count, word_bytes_len, line_len) {
                        break;
                    }
                    candidate_count += 1;
                    word_bytes_len += line_len;
                    max_word_len = max_word_len.max(line_len as u16);
                    line_cursor += 1;
                    self.next_index = self
                        .next_index
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("candidate index overflow"))?;
                }
                // Now consume whole blocks.
                if line_cursor == block_line_start {
                    while block_idx < chunk.block_summaries.len() {
                        let summary = chunk.block_summaries[block_idx];
                        if !batch_shape_can_fit_block(
                            candidate_count,
                            word_bytes_len,
                            summary.line_count as usize,
                            summary.total_bytes as usize,
                        ) {
                            break;
                        }
                        candidate_count += summary.line_count as usize;
                        word_bytes_len += summary.total_bytes as usize;
                        max_word_len = max_word_len.max(summary.max_len);
                        line_cursor += summary.line_count as usize;
                        self.next_index = self
                            .next_index
                            .checked_add(summary.line_count as u64)
                            .ok_or_else(|| anyhow!("candidate index overflow"))?;
                        block_idx += 1;
                    }
                }
            }

            // Phase 2 (fine): Per-line scan for the remainder / boundary block.
            while line_cursor < chunk.lengths.len() {
                let line_len = chunk.lengths[line_cursor] as usize;
                if !batch_shape_can_fit(candidate_count, word_bytes_len, line_len) {
                    break;
                }
                candidate_count += 1;
                word_bytes_len += line_len;
                max_word_len = max_word_len.max(line_len as u16);
                line_cursor += 1;
                self.next_index = self
                    .next_index
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("candidate index overflow"))?;
            }

            if line_cursor > segment_start {
                segments.push(BatchPlanSegment {
                    parsed_chunk: chunk.clone(),
                    line_start: segment_start,
                    line_end: line_cursor,
                });
            }

            self.active_chunk_line_cursor = line_cursor;
            let chunk_exhausted = self.active_chunk_line_cursor >= chunk.lengths.len();

            if chunk_exhausted {
                self.active_chunk = None;
                self.active_chunk_line_cursor = 0;
                self.next_chunk_to_emit = self
                    .next_chunk_to_emit
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("parser chunk index overflow"))?;
                continue;
            }

            break;
        }

        if candidate_count == 0 {
            Ok(None)
        } else {
            Ok(Some(BatchPlan {
                seq_no: 0,
                candidate_index_base,
                candidate_count,
                word_bytes_len,
                max_word_len,
                segments,
                parser_stats: self.parser_stats(),
                plan_time: plan_started.elapsed(),
            }))
        }
    }

    /// Fill the next GPU batch directly, merging plan + pack into one pass.
    ///
    /// This eliminates the intermediate `BatchPlan` allocation and avoids
    /// a second iteration over the parsed chunk metadata. The same block-
    /// level capacity checks are used, but segments are packed into the
    /// `WordBatch` immediately as they are determined.
    #[cfg(target_os = "linux")]
    pub(crate) fn fill_next_batch(
        &mut self,
        mmap: &memmap2::Mmap,
        batch: &mut WordBatch,
    ) -> anyhow::Result<Option<ParserStats>> {
        batch.reset_for_reuse(self.next_index);
        let mut candidate_count = 0usize;
        let mut word_bytes_len = 0usize;
        let mut max_word_len = 0u16;

        loop {
            if !self.ensure_active_chunk()? {
                break;
            }

            let chunk = self
                .active_chunk
                .as_ref()
                .cloned()
                .ok_or_else(|| anyhow!("parser active chunk missing"))?;
            let segment_start = self.active_chunk_line_cursor;
            let mut line_cursor = segment_start;

            // Phase 1 (coarse): Skip entire blocks using precomputed summaries.
            if candidate_count > 0 {
                let mut block_idx = line_cursor / PLANNER_BLOCK_STRIDE;
                let mut block_line_start = block_idx * PLANNER_BLOCK_STRIDE;
                if block_line_start < line_cursor {
                    block_idx += 1;
                    block_line_start = block_idx * PLANNER_BLOCK_STRIDE;
                }
                while line_cursor < block_line_start && line_cursor < chunk.lengths.len() {
                    let line_len = chunk.lengths[line_cursor] as usize;
                    if !batch_shape_can_fit(candidate_count, word_bytes_len, line_len) {
                        break;
                    }
                    candidate_count += 1;
                    word_bytes_len += line_len;
                    max_word_len = max_word_len.max(line_len as u16);
                    line_cursor += 1;
                    self.next_index = self
                        .next_index
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("candidate index overflow"))?;
                }
                if line_cursor == block_line_start {
                    while block_idx < chunk.block_summaries.len() {
                        let summary = chunk.block_summaries[block_idx];
                        if !batch_shape_can_fit_block(
                            candidate_count,
                            word_bytes_len,
                            summary.line_count as usize,
                            summary.total_bytes as usize,
                        ) {
                            break;
                        }
                        candidate_count += summary.line_count as usize;
                        word_bytes_len += summary.total_bytes as usize;
                        max_word_len = max_word_len.max(summary.max_len);
                        line_cursor += summary.line_count as usize;
                        self.next_index = self
                            .next_index
                            .checked_add(summary.line_count as u64)
                            .ok_or_else(|| anyhow!("candidate index overflow"))?;
                        block_idx += 1;
                    }
                }
            }

            // Phase 2 (fine): Per-line scan for the remainder / boundary block.
            while line_cursor < chunk.lengths.len() {
                let line_len = chunk.lengths[line_cursor] as usize;
                if !batch_shape_can_fit(candidate_count, word_bytes_len, line_len) {
                    break;
                }
                candidate_count += 1;
                word_bytes_len += line_len;
                max_word_len = max_word_len.max(line_len as u16);
                line_cursor += 1;
                self.next_index = self
                    .next_index
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("candidate index overflow"))?;
            }

            // Inline pack: copy this segment directly to the batch.
            if line_cursor > segment_start {
                batch.push_segment_bulk(
                    mmap.as_ref(),
                    chunk.chunk_start,
                    &chunk.offsets_rel,
                    &chunk.lengths,
                    segment_start,
                    line_cursor,
                );
            }

            self.active_chunk_line_cursor = line_cursor;
            let chunk_exhausted = self.active_chunk_line_cursor >= chunk.lengths.len();

            if chunk_exhausted {
                self.active_chunk = None;
                self.active_chunk_line_cursor = 0;
                self.next_chunk_to_emit = self
                    .next_chunk_to_emit
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("parser chunk index overflow"))?;
                continue;
            }

            break;
        }

        if candidate_count == 0 {
            Ok(None)
        } else {
            batch.set_plan_metadata(word_bytes_len, max_word_len);
            Ok(Some(self.parser_stats()))
        }
    }

    // Build the next GPU batch by first planning deterministic boundaries and
    // then materializing the planned payload into a `WordBatch`.
    #[cfg(test)]
    pub(crate) fn next_batch_reusing(
        &mut self,
        recycled: Option<WordBatch>,
    ) -> anyhow::Result<Option<WordBatch>> {
        let Some(plan) = self.next_batch_plan()? else {
            return Ok(None);
        };
        let mut batch = if let Some(mut batch) = recycled {
            batch.reset_for_reuse(plan.candidate_index_base);
            batch
        } else {
            WordBatch::new(&self.device, plan.candidate_index_base)
        };
        pack_batch_plan_into_batch(self.mmap.as_ref(), &plan, &mut batch)?;
        Ok(Some(batch))
    }
}

impl Drop for ParallelMmapWordlistBatchReader {
    fn drop(&mut self) {
        // Closing the job sender causes workers blocked on `recv()` to exit.
        let _ = self.job_tx.take();
        // Join workers so parser-thread panics do not silently outlive the
        // reader and to avoid detached threads during early shutdown paths.
        while let Some(handle) = self.worker_handles.pop() {
            let _ = handle.join();
        }
    }
}

// Small abstraction so the producer thread can remain agnostic to reader
// details. Runtime currently has a single concrete path (parallel mmap parser).
pub(crate) enum AnyWordlistBatchReader {
    ParallelMmap(ParallelMmapWordlistBatchReader),
}

impl AnyWordlistBatchReader {
    // Runtime path is mmap-only. If mapping fails, return an error instead of
    // silently switching to a slower path (which would hide perf regressions).
    pub(crate) fn new(
        device: GpuDevice,
        path: &PathBuf,
        parser_config: ParserConfig,
    ) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open wordlist {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap wordlist {}", path.display()))?;
        // Best-effort OS hint: chunk workers still perform forward scans over
        // their assigned slices. Ignore failures so advisory support
        // differences do not affect correctness.
        let _ = mmap.advise(Advice::Sequential);
        Ok(Self::ParallelMmap(ParallelMmapWordlistBatchReader::new(
            device,
            mmap,
            parser_config,
        )?))
    }

    pub(crate) fn parser_stats(&self) -> ParserStats {
        match self {
            Self::ParallelMmap(reader) => reader.parser_stats(),
        }
    }

    pub(crate) fn shared_mmap(&self) -> Arc<Mmap> {
        match self {
            Self::ParallelMmap(reader) => reader.shared_mmap(),
        }
    }

    pub(crate) fn next_batch_plan(&mut self) -> anyhow::Result<Option<BatchPlan>> {
        match self {
            Self::ParallelMmap(reader) => reader.next_batch_plan(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;

    use super::super::test_support::{
        collect_all_words_from_mmap_reader, collect_all_words_from_parallel_reader,
        mmap_reader_from_temp_file, parallel_mmap_reader_from_temp_file, test_device,
    };
    use super::*;

    #[test]
    fn wordlist_batch_reader_packs_offsets_and_lengths() {
        let data = b"alpha\r\n\nbe\ncharlie\n";
        let mut reader =
            BufferedWordlistBatchReader::new(test_device(), Cursor::new(data.as_slice()));
        let batch = reader.next_batch().unwrap().unwrap();

        assert_eq!(batch.candidate_index_base, 0);
        assert_eq!(batch.offsets_slice(), &[0, 5, 7]);
        assert_eq!(batch.lengths_slice(), &[5, 2, 7]);
        assert_eq!(batch.word_bytes_slice(), b"alphabecharlie");
        assert_eq!(batch.max_word_len(), 7);
        assert_eq!(batch.word(0).unwrap(), b"alpha");
        assert_eq!(batch.word(1).unwrap(), b"be");
        assert_eq!(batch.word(2).unwrap(), b"charlie");
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    fn wordlist_batch_reader_rejects_too_long_line() {
        let oversized = "a".repeat((u16::MAX as usize) + 1);
        let data = format!("{oversized}\n");
        let mut reader =
            BufferedWordlistBatchReader::new(test_device(), Cursor::new(data.into_bytes()));
        let err = reader.next_batch().unwrap_err();
        assert!(format!("{err:#}").contains("exceeds"));
    }

    #[test]
    fn wordlist_batch_reader_preserves_absolute_indices_across_batches() {
        let total = super::super::batch::MAX_CANDIDATES_PER_BATCH + 2;
        let mut data = Vec::with_capacity(total * 2);
        for _ in 0..total {
            data.extend_from_slice(b"x\n");
        }
        let mut reader = BufferedWordlistBatchReader::new(test_device(), Cursor::new(data));

        let first = reader.next_batch().unwrap().unwrap();
        let second = reader.next_batch().unwrap().unwrap();

        assert_eq!(first.candidate_index_base, 0);
        assert_eq!(
            first.candidate_count(),
            super::super::batch::MAX_CANDIDATES_PER_BATCH
        );
        assert_eq!(
            second.candidate_index_base,
            super::super::batch::MAX_CANDIDATES_PER_BATCH as u64
        );
        assert_eq!(second.candidate_count(), 2);
    }

    #[test]
    fn mmap_wordlist_batch_reader_packs_offsets_lengths_and_trims_crlf() {
        let (mut reader, path) = mmap_reader_from_temp_file(b"alpha\r\n\nbe\ncharlie\n");
        let batch = reader.next_batch_reusing(None).unwrap().unwrap();

        assert_eq!(batch.candidate_index_base, 0);
        assert_eq!(batch.offsets_slice(), &[0, 5, 7]);
        assert_eq!(batch.lengths_slice(), &[5, 2, 7]);
        assert_eq!(batch.word_bytes_slice(), b"alphabecharlie");
        assert_eq!(batch.max_word_len(), 7);
        assert_eq!(batch.word(0).unwrap(), b"alpha");
        assert_eq!(batch.word(1).unwrap(), b"be");
        assert_eq!(batch.word(2).unwrap(), b"charlie");
        assert!(reader.next_batch_reusing(None).unwrap().is_none());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn mmap_wordlist_batch_reader_handles_final_line_without_newline() {
        let (mut reader, path) = mmap_reader_from_temp_file(b"alpha\nbeta");
        let batch = reader.next_batch_reusing(None).unwrap().unwrap();

        assert_eq!(batch.candidate_count(), 2);
        assert_eq!(batch.word(0).unwrap(), b"alpha");
        assert_eq!(batch.word(1).unwrap(), b"beta");
        assert!(reader.next_batch_reusing(None).unwrap().is_none());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parallel_mmap_reader_packs_offsets_lengths_and_trims_crlf() {
        let (mut reader, path) =
            parallel_mmap_reader_from_temp_file(b"alpha\r\n\nbe\ncharlie\n", 2, 8);
        let mmap = reader.shared_mmap();
        let batch = reader.next_batch_reusing(None).unwrap().unwrap();

        assert_eq!(batch.candidate_index_base, 0);
        assert_eq!(batch.candidate_count(), 3);
        assert_eq!(batch.lengths_slice(), &[5, 2, 7]);
        assert_eq!(batch.max_word_len(), 7);
        let _ = &mmap;
        assert_eq!(batch.word(0).unwrap(), b"alpha");
        assert_eq!(batch.word(1).unwrap(), b"be");
        assert_eq!(batch.word(2).unwrap(), b"charlie");
        assert!(reader.next_batch_reusing(None).unwrap().is_none());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parallel_mmap_reader_handles_final_line_without_newline() {
        let (mut reader, path) = parallel_mmap_reader_from_temp_file(b"alpha\nbeta", 2, 5);
        let mmap = reader.shared_mmap();
        let batch = reader.next_batch_reusing(None).unwrap().unwrap();

        assert_eq!(batch.candidate_count(), 2);
        let _ = &mmap;
        assert_eq!(batch.word(0).unwrap(), b"alpha");
        assert_eq!(batch.word(1).unwrap(), b"beta");
        assert!(reader.next_batch_reusing(None).unwrap().is_none());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parallel_mmap_reader_matches_sequential_reference_with_one_worker() {
        let data = b"alpha\r\n\nbeta\ngamma-delta\nx\nfinal";
        let (mut sequential, seq_path) = mmap_reader_from_temp_file(data);
        let (mut parallel, par_path) = parallel_mmap_reader_from_temp_file(data, 1, 7);

        let seq_words = collect_all_words_from_mmap_reader(&mut sequential);
        let par_words = collect_all_words_from_parallel_reader(&mut parallel);

        assert_eq!(par_words, seq_words);

        let _ = fs::remove_file(seq_path);
        let _ = fs::remove_file(par_path);
    }

    #[test]
    fn parallel_mmap_reader_matches_sequential_reference_with_multiple_workers() {
        let data = b"alpha\r\n\nverylongcandidate1234567890\nbe\ncharlie\nzzz\r\nomega";
        let (mut sequential, seq_path) = mmap_reader_from_temp_file(data);
        let (mut parallel, par_path) = parallel_mmap_reader_from_temp_file(data, 4, 9);

        let seq_words = collect_all_words_from_mmap_reader(&mut sequential);
        let par_words = collect_all_words_from_parallel_reader(&mut parallel);

        assert_eq!(par_words, seq_words);

        let _ = fs::remove_file(seq_path);
        let _ = fs::remove_file(par_path);
    }

    #[test]
    fn parallel_mmap_reader_preserves_absolute_indices_across_batches() {
        let total = super::super::batch::MAX_CANDIDATES_PER_BATCH + 2;
        let mut data = Vec::with_capacity(total * 2);
        for _ in 0..total {
            data.extend_from_slice(b"x\n");
        }

        let (mut reader, path) = parallel_mmap_reader_from_temp_file(&data, 4, 1024);
        let first = reader.next_batch_reusing(None).unwrap().unwrap();
        let second = reader.next_batch_reusing(None).unwrap().unwrap();

        assert_eq!(first.candidate_index_base, 0);
        assert_eq!(
            first.candidate_count(),
            super::super::batch::MAX_CANDIDATES_PER_BATCH
        );
        assert_eq!(
            second.candidate_index_base,
            super::super::batch::MAX_CANDIDATES_PER_BATCH as u64
        );
        assert_eq!(second.candidate_count(), 2);
        assert!(reader.next_batch_reusing(None).unwrap().is_none());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parallel_mmap_reader_skips_oversized_lines_and_counts_them() {
        let oversized = "a".repeat((u16::MAX as usize) + 1);
        let data = format!("alpha\n{oversized}\nbeta\n");
        let (mut reader, path) = parallel_mmap_reader_from_temp_file(data.as_bytes(), 2, 64);

        let words = collect_all_words_from_parallel_reader(&mut reader);
        let words = words
            .into_iter()
            .map(|w| String::from_utf8(w).unwrap())
            .collect::<Vec<_>>();
        let stats = reader.parser_stats();

        assert_eq!(words, vec!["alpha".to_string(), "beta".to_string()]);
        assert_eq!(stats.parser_skipped_oversize, 1);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parallel_mmap_reader_next_batch_plan_matches_direct_batch_pack() {
        let data = b"alpha\r\n\nbeta\ngamma-delta\nx\nfinal";
        let (mut planned_reader, planned_path) = parallel_mmap_reader_from_temp_file(data, 3, 7);
        let (mut direct_reader, direct_path) = parallel_mmap_reader_from_temp_file(data, 3, 7);
        let mmap = planned_reader.shared_mmap();
        let device = test_device();

        loop {
            let planned = planned_reader.next_batch_plan().expect("batch plan");
            let direct = direct_reader
                .next_batch_reusing(None)
                .expect("direct batch");
            match (planned, direct) {
                (None, None) => break,
                (Some(plan), Some(direct_batch)) => {
                    let mut packed = WordBatch::new(&device, 0);
                    pack_batch_plan_into_batch(mmap.as_ref(), &plan, &mut packed)
                        .expect("pack planned batch");

                    assert_eq!(
                        packed.candidate_index_base,
                        direct_batch.candidate_index_base
                    );
                    assert_eq!(packed.candidate_count(), direct_batch.candidate_count());
                    assert_eq!(packed.max_word_len(), direct_batch.max_word_len());
                    assert_eq!(packed.lengths_slice(), direct_batch.lengths_slice());
                    for i in 0..packed.candidate_count() {
                        assert_eq!(packed.word(i), direct_batch.word(i),
                            "candidate {} mismatch", i);
                    }
                }
                (lhs, rhs) => panic!(
                    "planned/direct batch stream length mismatch: planned={} direct={}",
                    lhs.is_some(),
                    rhs.is_some()
                ),
            }
        }

        let _ = fs::remove_file(planned_path);
        let _ = fs::remove_file(direct_path);
    }

    #[test]
    fn batch_planner_preserves_absolute_indices_across_batches() {
        let total = super::super::batch::MAX_CANDIDATES_PER_BATCH + 2;
        let mut data = Vec::with_capacity(total * 2);
        for _ in 0..total {
            data.extend_from_slice(b"x\n");
        }

        let (mut reader, path) = parallel_mmap_reader_from_temp_file(&data, 4, 1024);
        let first = reader.next_batch_plan().unwrap().unwrap();
        let second = reader.next_batch_plan().unwrap().unwrap();

        assert_eq!(first.candidate_index_base, 0);
        assert_eq!(
            first.candidate_count,
            super::super::batch::MAX_CANDIDATES_PER_BATCH
        );
        assert_eq!(
            second.candidate_index_base,
            super::super::batch::MAX_CANDIDATES_PER_BATCH as u64
        );
        assert_eq!(second.candidate_count, 2);
        assert!(reader.next_batch_plan().unwrap().is_none());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn wordlist_batch_reader_returns_none_for_all_empty_lines() {
        let mut reader =
            BufferedWordlistBatchReader::new(test_device(), Cursor::new(b"\n\r\n".as_slice()));
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    fn wordlist_batch_reader_handles_final_line_without_newline() {
        let mut reader =
            BufferedWordlistBatchReader::new(test_device(), Cursor::new(b"alpha\nbeta".as_slice()));
        let batch = reader.next_batch().unwrap().unwrap();
        assert_eq!(batch.candidate_count(), 2);
        assert_eq!(batch.word(0).unwrap(), b"alpha");
        assert_eq!(batch.word(1).unwrap(), b"beta");
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    fn trim_line_endings_trims_crlf_and_lf() {
        assert_eq!(trim_line_endings(b"abc\r\n"), b"abc");
        assert_eq!(trim_line_endings(b"abc\n"), b"abc");
        assert_eq!(trim_line_endings(b"abc"), b"abc");
    }
}
