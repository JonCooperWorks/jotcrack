//! HS256 JWT wordlist cracking command (Metal-backed).
//!
//! Educational overview of the runtime pipeline:
//! 1. Parse and validate the JWT (`parse_hs256_jwt`) and extract the signing
//!    input + target HS256 signature bytes.
//! 2. Initialize the GPU runtime (`GpuHs256BruteForcer`) once, including Metal
//!    kernels and small persistent shared buffers (params/message/result).
//! 3. Spawn a producer thread that coordinates a parallel mmap parser and packs
//!    candidates into GPU-ready batches (`WordBatch`).
//! 4. On the main thread, consume batches, optionally autotune threadgroup
//!    width, dispatch GPU work, and track timing buckets.
//! 5. Report progress using windowed (interval) `RATE` lines, then print final
//!    cumulative `STATS` when a key is found or the wordlist is exhausted.
//!
//! The file is intentionally organized in the same order as the runtime data
//! flow (batch packing -> producer pipeline -> GPU dispatch -> command `run()`).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use clap::Args;
use memchr::memchr;
use memmap2::{Advice, Mmap, MmapOptions};
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use serde::Deserialize;
#[cfg(test)]
use std::io::BufRead;

// ---- Runtime configuration / tuning knobs ---------------------------------
// These constants define how much work we package into one GPU dispatch and how
// aggressively the host pipeline overlaps parsing with GPU execution.
pub const DEFAULT_WORDLIST_PATH: &str = "breach.txt";
// Embed the Metal source into the binary so release builds do not depend on a
// runtime-relative source file path.
const METAL_SOURCE_EMBEDDED: &str = include_str!("../kernels/hs256_wordlist.metal");
// We keep both kernels loaded and choose per batch based on candidate lengths.
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
const RESULT_NOT_FOUND_SENTINEL: u32 = u32::MAX;
// Larger batches improve amortization of dispatch overhead and host work.
const MAX_CANDIDATES_PER_BATCH: usize = 518_224;
const MAX_WORD_BYTES_PER_BATCH: usize = 32 * 1024 * 1024;
// Small bounded queue: enough to overlap CPU/GPU without letting parsed data
// accumulate unbounded in memory when the GPU is slower.
const DEFAULT_PIPELINE_DEPTH: usize = 2;
// Parser workers scan mmap chunks in parallel and return line metadata for
// ordered batch assembly on the producer thread.
const DEFAULT_PARSER_CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// CLI arguments for the `hs256wordlist` subcommand.
///
/// This struct is intentionally close to runtime concepts (JWT, wordlist path,
/// threadgroup width, autotune toggle) so the dispatch layer can pass it
/// straight into `run()` with minimal translation.
#[derive(Debug, Clone, Args)]
pub struct Hs256WordlistArgs {
    pub jwt: String,
    #[arg(long, default_value = DEFAULT_WORDLIST_PATH)]
    pub wordlist: PathBuf,
    #[arg(long)]
    pub threads_per_group: Option<usize>,
    #[arg(long, value_parser = parse_positive_usize)]
    pub parser_threads: Option<usize>,
    #[arg(long)]
    pub autotune: bool,
}

#[derive(Debug, Clone, Copy)]
// Host-side parser tuning resolved once at command startup.
//
// Keeping this as a plain struct (instead of reading CLI args everywhere)
// makes the producer/parser code easier to test and keeps policy decisions
// (auto thread count, chunk size, queue depth) in one place.
struct ParserConfig {
    parser_threads: usize,
    chunk_bytes: usize,
    queue_capacity: usize,
}

impl ParserConfig {
    fn from_args(args: &Hs256WordlistArgs) -> Self {
        let auto_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let parser_threads = args.parser_threads.unwrap_or(auto_threads);
        let queue_capacity = parser_threads.saturating_mul(4).max(1);
        Self {
            parser_threads,
            chunk_bytes: DEFAULT_PARSER_CHUNK_BYTES,
            queue_capacity,
        }
    }
}

// Clap parser helper used for `--parser-threads`.
//
// We validate here so CLI errors are reported before any GPU/parser startup
// work begins, and the rest of the code can assume a strictly positive value.
fn parse_positive_usize(input: &str) -> Result<usize, String> {
    let parsed = input
        .parse::<usize>()
        .map_err(|_| format!("invalid integer value: {input}"))?;
    if parsed == 0 {
        return Err("must be > 0".to_string());
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Copy, Default)]
// Snapshot of parser-side counters/configuration reported by the producer.
//
// This is copied into `ProducerMessage` so the consumer can print final `STATS`
// without reaching into parser internals or sharing mutable state across
// threads.
struct ParserStats {
    parser_threads: usize,
    parser_chunk_bytes: usize,
    parser_chunks: u64,
    parser_skipped_oversize: u64,
}

#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

// Host -> Metal parameter block. `#[repr(C)]` is required because Rust and the
// Metal kernel must agree on the exact field order and byte layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs256BruteForceParams {
    target_signature: [u8; 32],
    // `message_length` is precomputed on the host so the kernel does not need
    // to infer it from buffers or rely on implicit buffer metadata.
    message_length: u32,
    candidate_count: u32,
    // Reserved/padding field kept so the Rust and Metal structs stay the same
    // size/alignment as before (4 x 32-bit fields after the 32-byte digest).
    //
    // We previously used this slot for `candidate_index_base` (absolute wordlist
    // index of candidate #0 in the batch). That overflowed at `u32::MAX` on very
    // large wordlists, so the GPU now reports a batch-local index and the host
    // reconstructs the absolute index from `WordBatch::candidate_index_base`.
    //
    // Keeping the field avoids unnecessary ABI churn and makes future kernel-side
    // metadata additions straightforward without changing buffer sizes today.
    reserved0: u32,
}

// Packed representation of a wordlist batch sent to the GPU.
//
// Instead of storing `Vec<String>` per candidate (which is allocation-heavy for
// huge wordlists), we store one contiguous byte blob plus offset/length tables.
// This minimizes host memory churn and matches the GPU kernel input format.
#[derive(Debug)]
struct WordBatch {
    // Host-only absolute wordlist index of candidate #0 in this batch.
    //
    // This is intentionally `u64` so parsing/progress/result reconstruction can
    // exceed 4,294,967,295 non-empty candidates. It is not sent to the GPU
    // anymore; the kernel returns a batch-local match index (`u32`) instead.
    candidate_index_base: u64,
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

impl WordBatch {
    // Allocate fixed-capacity shared buffers sized to the batch caps so parser
    // writes can go straight into memory later bound to the kernel.
    fn new(device: &Device, candidate_index_base: u64) -> Self {
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
    fn reset_for_reuse(&mut self, candidate_index_base: u64) {
        // Rebinding the base is what preserves globally correct indexing across
        // batches even when the backing Metal buffers are recycled.
        self.candidate_index_base = candidate_index_base;
        self.candidate_count = 0;
        self.word_bytes_len = 0;
        self.max_word_len = 0;
    }

    // A batch is considered empty when it has no candidate metadata entries.
    fn is_empty(&self) -> bool {
        self.candidate_count == 0
    }

    // Number of candidates packaged for this dispatch.
    fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    // Number of packed candidate bytes for this dispatch.
    fn word_bytes_len(&self) -> usize {
        self.word_bytes_len
    }

    fn max_word_len(&self) -> u16 {
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

    fn offsets_slice(&self) -> &[u32] {
        // SAFETY: `word_offsets_buf` stores `u32` entries; the first
        // `candidate_count` entries are initialized by `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(self.word_offsets_ptr as *const u32, self.candidate_count)
        }
    }

    fn lengths_slice(&self) -> &[u16] {
        // SAFETY: `word_lengths_buf` stores `u16` entries; the first
        // `candidate_count` entries are initialized by `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(self.word_lengths_ptr as *const u16, self.candidate_count)
        }
    }

    fn word_bytes_slice(&self) -> &[u8] {
        // SAFETY: the first `word_bytes_len` bytes are initialized by
        // `push_candidate`.
        unsafe { std::slice::from_raw_parts(self.word_bytes_ptr as *const u8, self.word_bytes_len) }
    }

    fn can_fit(&self, line_len: usize) -> bool {
        // Preserve the original batching rule:
        // - always enforce candidate-count cap
        // - only enforce byte-cap after the first candidate is present
        let would_exceed_count = self.candidate_count >= MAX_CANDIDATES_PER_BATCH;
        let would_exceed_bytes = self.candidate_count != 0
            && (self.word_bytes_len + line_len > MAX_WORD_BYTES_PER_BATCH);
        !(would_exceed_count || would_exceed_bytes)
    }

    fn push_candidate(&mut self, line: &[u8]) -> anyhow::Result<()> {
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
    fn word(&self, local_index: usize) -> Option<&[u8]> {
        let start = *self.offsets_slice().get(local_index)? as usize;
        let len = *self.lengths_slice().get(local_index)? as usize;
        self.word_bytes_slice().get(start..start + len)
    }

    // Human-readable reconstruction for final reporting. We keep this lossy to
    // avoid crashing on non-UTF-8 wordlist entries while still printing a key.
    fn word_string_lossy(&self, local_index: usize) -> Option<String> {
        Some(String::from_utf8_lossy(self.word(local_index)?).into_owned())
    }

    fn as_dispatch_view(&self) -> DispatchBatchView<'_> {
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

    fn prefix_dispatch_view(&self, sample_count: usize) -> Option<DispatchBatchView<'_>> {
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
struct DispatchBatchView<'a> {
    // Minimal metadata + buffer references required to dispatch a full batch or
    // sampled prefix batch without owning the storage.
    candidate_count: usize,
    max_word_len: u16,
    word_bytes_buf: &'a metal::Buffer,
    word_offsets_buf: &'a metal::Buffer,
    word_lengths_buf: &'a metal::Buffer,
}

// Shared helper used by parser tests to verify CRLF/LF parity.
#[cfg(test)]
fn trim_line_endings(bytes: &[u8]) -> &[u8] {
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
struct BufferedWordlistBatchReader<R: BufRead> {
    device: Device,
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
    fn new(device: Device, reader: R) -> Self {
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
    fn next_batch(&mut self) -> anyhow::Result<Option<WordBatch>> {
        self.next_batch_reusing(None)
    }

    fn next_batch_reusing(
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
struct MmapWordlistBatchReader {
    device: Device,
    mmap: Mmap,
    cursor: usize,
    // Same semantic as the test-only buffered reader: absolute index of the next
    // non-empty candidate line, tracked in `u64` for huge wordlists.
    next_index: u64,
}

#[cfg(test)]
impl MmapWordlistBatchReader {
    // Wrap a mapped wordlist file for sequential batch extraction.
    fn new(device: Device, mmap: Mmap) -> Self {
        Self {
            device,
            mmap,
            cursor: 0,
            next_index: 0,
        }
    }

    // Scan the mmap for newline-delimited candidates while preserving the same
    // batching semantics as the buffered reader.
    fn next_batch_reusing(
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
// file-order emission while still parallelizing scanning.
struct ChunkJob {
    chunk_id: u64,
    start: usize,
    end: usize,
}

#[derive(Debug)]
// Parsed metadata for one chunk.
//
// The worker deliberately does *not* copy candidate bytes into GPU buffers.
// Instead it records relative offsets and lengths so the producer thread can
// later pack candidates into `WordBatch` in deterministic file order.
struct ParsedChunk {
    chunk_id: u64,
    chunk_start: usize,
    offsets_rel: Vec<u32>,
    lengths: Vec<u16>,
    skipped_oversize: u64,
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

    let chunk = &bytes[job.start..job.end];
    let mut offsets_rel = Vec::new();
    let mut lengths = Vec::new();
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

    Ok(ParsedChunk {
        chunk_id: job.chunk_id,
        chunk_start: job.start,
        offsets_rel,
        lengths,
        skipped_oversize,
    })
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
struct ParallelMmapWordlistBatchReader {
    device: Device,
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
    // Out-of-order worker results buffered until `next_chunk_to_emit` arrives.
    pending_results: BTreeMap<u64, ParsedChunk>,
    // Current parsed chunk being drained into a `WordBatch`.
    active_chunk: Option<ParsedChunk>,
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
    fn new(device: Device, mmap: Mmap, config: ParserConfig) -> anyhow::Result<Self> {
        let mmap = Arc::new(mmap);
        let chunks = plan_mmap_chunks(mmap.as_ref(), config.chunk_bytes)?;
        let (job_tx, job_rx) = crossbeam_channel::bounded(config.queue_capacity);
        let (result_tx, result_rx) = crossbeam_channel::bounded(config.queue_capacity);
        let mut worker_handles = Vec::with_capacity(config.parser_threads);

        for _ in 0..config.parser_threads {
            let mmap_for_worker = Arc::clone(&mmap);
            let job_rx_for_worker = job_rx.clone();
            let result_tx_for_worker = result_tx.clone();
            worker_handles.push(thread::spawn(move || {
                // Each worker is a simple parse loop: recv job -> parse -> send
                // metadata. No shared mutable state is touched here.
                while let Ok(job) = job_rx_for_worker.recv() {
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
            pending_results: BTreeMap::new(),
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
    fn parser_stats(&self) -> ParserStats {
        self.parser_stats
    }

    // Submit more jobs until we hit the in-flight limit or exhaust the chunk
    // plan. The bounded channel keeps memory growth predictable.
    fn pump_jobs(&mut self) -> anyhow::Result<()> {
        while self.in_flight_jobs < self.max_in_flight_jobs
            && self.next_job_to_send < self.chunks.len()
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
                if self.pending_results.insert(chunk.chunk_id, chunk).is_some() {
                    bail!("duplicate parsed chunk id received from parser workers");
                }
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
        if let Some(chunk) = self.pending_results.remove(&self.next_chunk_to_emit) {
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
    // Batch boundaries remain governed by the same `WordBatch::can_fit` rules as
    // before; the redesign changes *how* we discover lines, not how we package
    // them for the GPU.
    fn next_batch_reusing(
        &mut self,
        recycled: Option<WordBatch>,
    ) -> anyhow::Result<Option<WordBatch>> {
        let candidate_index_base = self.next_index;
        let mut batch = if let Some(mut batch) = recycled {
            batch.reset_for_reuse(candidate_index_base);
            batch
        } else {
            WordBatch::new(&self.device, candidate_index_base)
        };

        loop {
            if !self.ensure_active_chunk()? {
                break;
            }

            let chunk_exhausted = {
                let chunk = self
                    .active_chunk
                    .as_ref()
                    .ok_or_else(|| anyhow!("parser active chunk missing"))?;
                let bytes = self.mmap.as_ref();

                // Drain as many lines from the current parsed chunk as will fit
                // in this batch, leaving the cursor parked if the batch fills.
                while self.active_chunk_line_cursor < chunk.lengths.len() {
                    let line_len = chunk.lengths[self.active_chunk_line_cursor] as usize;
                    if !batch.can_fit(line_len) {
                        break;
                    }

                    let line_start = chunk.chunk_start
                        + chunk.offsets_rel[self.active_chunk_line_cursor] as usize;
                    let line_end = line_start + line_len;
                    batch.push_candidate(&bytes[line_start..line_end])?;
                    self.active_chunk_line_cursor += 1;
                    self.next_index = self
                        .next_index
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("candidate index overflow"))?;
                }

                self.active_chunk_line_cursor >= chunk.lengths.len()
            };

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

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
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
enum AnyWordlistBatchReader {
    ParallelMmap(ParallelMmapWordlistBatchReader),
}

impl AnyWordlistBatchReader {
    // Runtime path is mmap-only. If mapping fails, return an error instead of
    // silently switching to a slower path (which would hide perf regressions).
    fn new(device: Device, path: &PathBuf, parser_config: ParserConfig) -> anyhow::Result<Self> {
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

    fn parser_stats(&self) -> ParserStats {
        match self {
            Self::ParallelMmap(reader) => reader.parser_stats(),
        }
    }

    // Forward batch production to the selected reader strategy.
    fn next_batch_reusing(
        &mut self,
        recycled: Option<WordBatch>,
    ) -> anyhow::Result<Option<WordBatch>> {
        // `recycled` is moved into exactly one concrete reader branch, so the
        // batch allocation has a single owner throughout the rebuild step.
        match self {
            Self::ParallelMmap(reader) => reader.next_batch_reusing(recycled),
        }
    }
}

// Messages sent from the producer thread to the consumer (`run()`).
// We include build time with each batch so end-to-end timing can report how
// much host parsing work is overlapped vs exposed.
enum ProducerMessage {
    Batch {
        batch: WordBatch,
        build_time: Duration,
        parser_stats: ParserStats,
    },
    Eof {
        parser_stats: ParserStats,
    },
    Error(String),
}

// Owns the producer thread and channel endpoints used to overlap wordlist
// parsing with GPU dispatch.
struct WordlistProducer {
    rx: Option<Receiver<ProducerMessage>>,
    // Reverse direction channel used only for allocation reuse:
    // consumer -> producer returns fully-owned `WordBatch` values after dispatch.
    recycle_tx: SyncSender<WordBatch>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl WordlistProducer {
    // Start the producer thread immediately so parsing can begin while the
    // consumer is initializing autotune / first dispatch state.
    fn spawn(wordlist_path: PathBuf, device: Device, parser_config: ParserConfig) -> Self {
        let (tx, rx) = sync_channel(DEFAULT_PIPELINE_DEPTH);
        // Match the forward pipeline depth so we can recycle a small pool of
        // batch allocations without letting memory usage grow unbounded.
        let (recycle_tx, recycle_rx) = sync_channel(DEFAULT_PIPELINE_DEPTH);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);
        let tx_err = tx.clone();
        let handle = thread::spawn(move || {
            if let Err(err) = run_wordlist_producer(
                wordlist_path,
                device,
                parser_config,
                stop_for_thread,
                tx,
                recycle_rx,
            ) {
                // Best effort: the receiver may already be dropped on early exit.
                let _ = tx_err.send(ProducerMessage::Error(format!("{err:#}")));
            }
        });
        Self {
            rx: Some(rx),
            recycle_tx,
            stop,
            handle: Some(handle),
        }
    }

    // Cooperative stop signal checked between batch builds.
    fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    // Receive the next batch (or terminal message) from the producer. A closed
    // channel is treated as an error because the consumer expects a clean EOF.
    fn recv(&self) -> anyhow::Result<ProducerMessage> {
        let rx = self
            .rx
            .as_ref()
            .ok_or_else(|| anyhow!("wordlist producer receiver already closed"))?;
        rx.recv()
            .map_err(|_| anyhow!("wordlist producer terminated unexpectedly"))
    }

    // Dropping the receiver is an important shutdown step when exiting early on
    // a match: it prevents the producer from blocking forever on a bounded send.
    fn close_receiver(&mut self) {
        let _ = self.rx.take();
    }

    // Best-effort recycle path: do not block the consumer/GPU path if the
    // producer is busy or already shutting down.
    fn recycle(&self, batch: WordBatch) {
        // `try_send` is intentional: the consumer is on the critical path for
        // throughput, so we prefer dropping a recyclable allocation over
        // stalling GPU dispatch progress.
        let _ = self.recycle_tx.try_send(batch);
    }

    // Join the background thread so shutdown is deterministic and panics are
    // surfaced as regular errors.
    fn join(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("wordlist producer thread panicked"))?;
        }
        Ok(())
    }
}

// Producer-thread main loop: parse batches and send them to the consumer until
// EOF, stop requested, or the receiver is dropped.
fn run_wordlist_producer(
    wordlist_path: PathBuf,
    device: Device,
    parser_config: ParserConfig,
    stop: Arc<AtomicBool>,
    tx: SyncSender<ProducerMessage>,
    // Optional returned batches from the consumer. The producer owns the next
    // parse/build step, so this is where allocation reuse naturally fits.
    recycle_rx: Receiver<WordBatch>,
) -> anyhow::Result<()> {
    let mut reader = AnyWordlistBatchReader::new(device, &wordlist_path, parser_config)?;
    while !stop.load(Ordering::Relaxed) {
        // The producer reports build time per batch so the consumer can measure
        // how much parsing work is hidden behind GPU compute.
        let batch_started = Instant::now();
        let recycled = match recycle_rx.try_recv() {
            Ok(batch) => Some(batch),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        };
        // `next_batch_reusing` preserves batching semantics; only allocation
        // source changes (fresh vs recycled).
        let maybe_batch = reader.next_batch_reusing(recycled)?;
        let build_time = batch_started.elapsed();
        let parser_stats = reader.parser_stats();
        let message = match maybe_batch {
            Some(batch) => ProducerMessage::Batch {
                batch,
                build_time,
                parser_stats,
            },
            None => {
                let _ = tx.send(ProducerMessage::Eof { parser_stats });
                return Ok(());
            }
        };
        if tx.send(message).is_err() {
            return Ok(());
        }
    }
    Ok(())
}

// Owns Metal objects and the small persistent buffers reused across dispatches.
//
// Large per-batch payload buffers now live in `WordBatch` and are recycled
// producer<->consumer, which removes the old extra host copy before dispatch.
struct GpuHs256BruteForcer {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline_mixed: metal::ComputePipelineState,
    pipeline_short_keys: metal::ComputePipelineState,
    params_buf: metal::Buffer,
    msg_buf: metal::Buffer,
    result_buf: metal::Buffer,
    message_length: u32,
    threadgroup_width: usize,
}

// Per-dispatch timing buckets used to separate host prep/encode overhead from
// time spent waiting for GPU completion.
#[derive(Debug, Clone, Copy, Default)]
struct BatchDispatchTimings {
    host_prep: Duration,
    command_encode: Duration,
    gpu_wait: Duration,
    result_readback: Duration,
    total: Duration,
}

// Aggregated run timings and batch stats printed in the final report.
#[derive(Debug, Default)]
struct RunTimings {
    wordlist_batch_build: Duration,
    host_prep: Duration,
    command_encode: Duration,
    gpu_wait: Duration,
    result_readback: Duration,
    dispatch_total: Duration,
    consumer_idle_wait: Duration,
    batch_count: u64,
    total_batch_candidates: u64,
    total_batch_word_bytes: u64,
    parser_threads: usize,
    parser_chunk_bytes: usize,
    parser_chunks: u64,
    parser_skipped_oversize: u64,
}

impl RunTimings {
    // Copy the latest parser-side counters/config into the reporting snapshot.
    //
    // The producer includes this with each message so the consumer can print a
    // complete final report even if the run ends early (match found) or at EOF.
    fn apply_parser_stats(&mut self, parser_stats: ParserStats) {
        self.parser_threads = parser_stats.parser_threads;
        self.parser_chunk_bytes = parser_stats.parser_chunk_bytes;
        self.parser_chunks = parser_stats.parser_chunks;
        self.parser_skipped_oversize = parser_stats.parser_skipped_oversize;
    }
}

// Snapshot of cumulative counters/timings at a periodic progress report.
// Deltas between snapshots produce interval ("windowed") rates and wait times.
#[derive(Debug, Clone, Copy)]
struct RateReportSnapshot {
    reported_at: Instant,
    candidates_tested: u64,
    gpu_wait: Duration,
    wordlist_batch_build: Duration,
    consumer_idle_wait: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
struct RateReportDelta {
    wall_time: Duration,
    candidates_tested: u64,
    gpu_wait: Duration,
    wordlist_batch_build: Duration,
    consumer_idle_wait: Duration,
}

impl RateReportSnapshot {
    fn capture(reported_at: Instant, candidates_tested: u64, timings: &RunTimings) -> Self {
        Self {
            reported_at,
            candidates_tested,
            gpu_wait: timings.gpu_wait,
            wordlist_batch_build: timings.wordlist_batch_build,
            consumer_idle_wait: timings.consumer_idle_wait,
        }
    }

    fn delta_since(self, previous: Self) -> RateReportDelta {
        RateReportDelta {
            wall_time: self
                .reported_at
                .checked_duration_since(previous.reported_at)
                .unwrap_or_default(),
            candidates_tested: self
                .candidates_tested
                .saturating_sub(previous.candidates_tested),
            gpu_wait: saturating_duration_sub(self.gpu_wait, previous.gpu_wait),
            wordlist_batch_build: saturating_duration_sub(
                self.wordlist_batch_build,
                previous.wordlist_batch_build,
            ),
            consumer_idle_wait: saturating_duration_sub(
                self.consumer_idle_wait,
                previous.consumer_idle_wait,
            ),
        }
    }
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

impl GpuHs256BruteForcer {
    // Compile the Metal kernels, create the command queue, and allocate the
    // small persistent buffers shared by every dispatch (params/JWT/result).
    fn new(signing_input: &[u8]) -> anyhow::Result<Self> {
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
            device.new_buffer(std::mem::size_of::<Hs256BruteForceParams>() as u64, options);
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
    // dispatch view is <= 64 bytes; otherwise fall back to the mixed kernel.
    // `DispatchBatchView` lets autotune reuse this logic on a sampled prefix.
    fn active_pipeline_for_view(
        &self,
        batch: DispatchBatchView<'_>,
    ) -> &metal::ComputePipelineState {
        if batch.max_word_len <= 64 {
            &self.pipeline_short_keys
        } else {
            &self.pipeline_mixed
        }
    }

    // Validate and apply a user-provided threadgroup width override.
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

    // Accessor used by logging and periodic rate reports.
    fn current_threadgroup_width(&self) -> usize {
        self.threadgroup_width
    }

    // Conservative execution width used for autotune candidate generation.
    fn thread_execution_width(&self) -> usize {
        (self.pipeline_mixed.thread_execution_width() as usize)
            .min(self.pipeline_short_keys.thread_execution_width() as usize)
    }

    // Conservative threadgroup cap valid for both kernels.
    fn max_total_threads_per_threadgroup(&self) -> usize {
        (self.pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(self.pipeline_short_keys.max_total_threads_per_threadgroup() as usize)
    }

    // Human-readable device name for startup logging.
    fn device_name(&self) -> &str {
        self.device.name()
    }

    // Dispatch one batch using the currently selected threadgroup width.
    fn try_batch(
        &self,
        target_signature: [u8; 32],
        batch: &WordBatch,
    ) -> anyhow::Result<(Option<u32>, BatchDispatchTimings)> {
        self.dispatch_batch_view(
            target_signature,
            batch.as_dispatch_view(),
            self.threadgroup_width,
        )
    }

    // Encode and execute a single GPU dispatch, then read back the match index.
    // Timing is split so we can distinguish host overhead from actual GPU wait.
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

        let params = Hs256BruteForceParams {
            target_signature,
            message_length: self.message_length,
            candidate_count,
            // The kernel no longer needs the absolute wordlist base. It only
            // needs `candidate_count` and writes back the earliest matching
            // batch-local index (`gid`) to the shared result slot.
            reserved0: 0,
        };

        // Host prep phase: update params and reset the result slot.
        //
        // The large offsets/lengths/payload buffers are already populated in
        // shared memory by the producer thread and are bound directly below.
        let prep_started = Instant::now();
        copy_value_to_buffer(&self.params_buf, &params);
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        timings.host_prep = prep_started.elapsed();

        // Encode phase: bind the selected kernel and batch-owned shared buffers,
        // then schedule exactly one thread per candidate.
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
        timings.command_encode = encode_started.elapsed();

        // GPU wait measures the blocking portion seen by the host after commit.
        let gpu_wait_started = Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        timings.gpu_wait = gpu_wait_started.elapsed();

        // Read back the result from shared memory after completion.
        //
        // Result contract:
        // - `RESULT_NOT_FOUND_SENTINEL` => no match in this batch
        // - any other value => earliest matching batch-local candidate index
        //
        // Host code combines that local index with `batch.candidate_index_base`
        // (tracked as `u64`) when it needs a globally meaningful position.
        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        // SAFETY: `result_buf` is a 4-byte shared buffer that stays alive for the dispatch lifetime.
        let result = unsafe { *result_ptr };
        timings.result_readback = readback_started.elapsed();

        timings.total = started.elapsed();
        if result == RESULT_NOT_FOUND_SENTINEL {
            Ok((None, timings))
        } else {
            Ok((Some(result), timings))
        }
    }

    // Benchmark a small sample from the first batch across several candidate
    // threadgroup widths, then keep the fastest width for the rest of the run.
    //
    // We optimize for `gpu_wait` (not end-to-end time) because autotune is only
    // choosing a GPU execution parameter, not host parsing behavior.
    fn autotune_threadgroup_width(
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

/// End-to-end HS256 cracking flow:
/// 1) parse the JWT,
/// 2) initialize Metal + reusable buffers,
/// 3) overlap wordlist parsing with GPU dispatch,
/// 4) report the first matching secret (or NOT FOUND).
pub fn run(args: Hs256WordlistArgs) -> anyhow::Result<bool> {
    let (signing_input, target_signature) = parse_hs256_jwt(&args.jwt)?;
    let mut gpu = GpuHs256BruteForcer::new(&signing_input)?;
    let parser_config = ParserConfig::from_args(&args);
    if let Some(tpg) = args.threads_per_group {
        gpu.set_threadgroup_width(tpg)?;
    }
    eprintln!(
        "GPU device={} tew={} max_tpg={} selected_tpg={}",
        gpu.device_name(),
        gpu.thread_execution_width(),
        gpu.max_total_threads_per_threadgroup(),
        gpu.current_threadgroup_width()
    );

    // Start the parser thread before entering the main loop so batch
    // construction can overlap with GPU setup and dispatch wait time.
    let mut producer =
        WordlistProducer::spawn(args.wordlist.clone(), gpu.device.clone(), parser_config);
    let run_result = (|| -> anyhow::Result<bool> {
        let started_at = Instant::now();
        let mut candidates_tested: u64 = 0;
        let mut autotune_done = !args.autotune;
        let mut timings = RunTimings {
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
                    parser_stats,
                } => {
                    // Producer snapshots parser counters after each batch build.
                    // We refresh the consumer-side copy here so both periodic
                    // and final reports reflect the latest parser state.
                    timings.apply_parser_stats(parser_stats);
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

// Parse and validate an HS256 JWT, returning:
// - the signing input bytes (`base64url(header) + "." + base64url(payload)`)
// - the decoded 32-byte target signature to compare against GPU results
fn parse_hs256_jwt(jwt: &str) -> anyhow::Result<(Vec<u8>, [u8; 32])> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        bail!("malformed JWT: expected 3 dot-separated segments");
    }

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .context("invalid base64url JWT header")?;
    let header: JwtHeader =
        serde_json::from_slice(&header_bytes).context("invalid JWT header JSON")?;
    if header.alg != "HS256" {
        bail!("unsupported JWT alg: expected HS256, got {}", header.alg);
    }

    let signature_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .context("invalid base64url JWT signature")?;
    if signature_bytes.len() != 32 {
        bail!(
            "invalid HS256 signature length: expected 32 bytes, got {}",
            signature_bytes.len()
        );
    }
    let mut target_signature = [0u8; 32];
    target_signature.copy_from_slice(&signature_bytes);

    let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();

    Ok((signing_input, target_signature))
}

// Helper used by both periodic and final reporting. Returns 0 when no time has
// elapsed yet to avoid division-by-zero during startup.
fn saturating_duration_sub(current: Duration, previous: Duration) -> Duration {
    // Defensive helper for "delta from cumulative totals" math used by the
    // progress reporter. Underflow should not happen in normal execution, but
    // returning zero keeps tests and future refactors safe.
    current.checked_sub(previous).unwrap_or_default()
}

fn rate_per_second(candidates_tested: u64, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        candidates_tested as f64 / elapsed.as_secs_f64()
    }
}

fn format_duration_millis(duration: Duration) -> String {
    // Keep progress logs compact and stable: milliseconds are easy to compare
    // across runs when diagnosing host-side stalls.
    format!("{:.1}", duration.as_secs_f64() * 1_000.0)
}

// Render large counts/rates in a stable human-readable format so benchmark logs
// remain easy to scan while still preserving approximate magnitude.
fn format_human_count(value: f64) -> String {
    const UNITS: [(&str, f64); 4] = [
        ("trillion", 1_000_000_000_000.0),
        ("billion", 1_000_000_000.0),
        ("million", 1_000_000.0),
        ("thousand", 1_000.0),
    ];

    for (name, scale) in UNITS {
        if value >= scale {
            let scaled = value / scale;
            let precision = if scaled >= 100.0 {
                0
            } else if scaled >= 10.0 {
                1
            } else {
                2
            };
            return format!("{scaled:.precision$} {name}");
        }
    }

    format!("{value:.0}")
}

// Print the final timing breakdown and batch statistics. This intentionally
// separates end-to-end throughput from GPU-only throughput to highlight host
// bottlenecks (parsing, prep, synchronization).
fn print_final_stats(
    candidates_tested: u64,
    elapsed: Duration,
    rate_end_to_end: f64,
    rate_gpu_only: f64,
    timings: &RunTimings,
) {
    let avg_candidates_per_batch = if timings.batch_count == 0 {
        0.0
    } else {
        timings.total_batch_candidates as f64 / timings.batch_count as f64
    };
    let avg_word_bytes_per_batch = if timings.batch_count == 0 {
        0.0
    } else {
        timings.total_batch_word_bytes as f64 / timings.batch_count as f64
    };

    eprintln!("STATS");
    eprintln!("  tested: {}", format_human_count(candidates_tested as f64));
    eprintln!("  elapsed: {:.2}s", elapsed.as_secs_f64());
    eprintln!(
        "  rate_end_to_end: {}/s",
        format_human_count(rate_end_to_end)
    );
    eprintln!("  rate_gpu_only: {}/s", format_human_count(rate_gpu_only));
    eprintln!("  batches: {}", timings.batch_count);
    eprintln!(
        "  avg_candidates_per_batch: {}",
        format_human_count(avg_candidates_per_batch)
    );
    eprintln!(
        "  avg_word_bytes_per_batch: {} bytes",
        format_human_count(avg_word_bytes_per_batch)
    );
    eprintln!("  parser_threads: {}", timings.parser_threads);
    eprintln!(
        "  parser_chunk_bytes: {} bytes",
        format_human_count(timings.parser_chunk_bytes as f64)
    );
    eprintln!("  parser_chunks: {}", timings.parser_chunks);
    eprintln!(
        "  parser_skipped_oversize: {}",
        timings.parser_skipped_oversize
    );
    eprintln!(
        "  timing.wordlist_batch_build: {:.3}s",
        timings.wordlist_batch_build.as_secs_f64()
    );
    eprintln!(
        "  timing.consumer_idle_wait: {:.3}s",
        timings.consumer_idle_wait.as_secs_f64()
    );
    eprintln!(
        "  timing.host_prep: {:.3}s",
        timings.host_prep.as_secs_f64()
    );
    eprintln!(
        "  timing.command_encode: {:.3}s",
        timings.command_encode.as_secs_f64()
    );
    eprintln!("  timing.gpu_wait: {:.3}s", timings.gpu_wait.as_secs_f64());
    eprintln!(
        "  timing.result_readback: {:.3}s",
        timings.result_readback.as_secs_f64()
    );
    eprintln!(
        "  timing.dispatch_total: {:.3}s",
        timings.dispatch_total.as_secs_f64()
    );
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
    use std::fs::{self, File as StdFile};
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_WORDLIST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_device() -> Device {
        Device::system_default().expect("Metal device is required for hs256wordlist tests")
    }

    fn write_temp_wordlist(bytes: &[u8]) -> std::path::PathBuf {
        let unique = TEMP_WORDLIST_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "jotcrack-hs256wordlist-test-{}-{nanos}-{unique}.txt",
            std::process::id()
        ));
        fs::write(&path, bytes).expect("failed to write temp wordlist");
        path
    }

    fn mmap_reader_from_temp_file(bytes: &[u8]) -> (MmapWordlistBatchReader, std::path::PathBuf) {
        let path = write_temp_wordlist(bytes);
        let file = StdFile::open(&path).expect("failed to open temp wordlist");
        let mmap = unsafe { MmapOptions::new().map(&file) }.expect("failed to mmap temp wordlist");
        (MmapWordlistBatchReader::new(test_device(), mmap), path)
    }

    fn test_parser_config(parser_threads: usize, chunk_bytes: usize) -> ParserConfig {
        ParserConfig {
            parser_threads,
            chunk_bytes,
            queue_capacity: parser_threads.saturating_mul(4).max(1),
        }
    }

    fn parallel_mmap_reader_from_temp_file(
        bytes: &[u8],
        parser_threads: usize,
        chunk_bytes: usize,
    ) -> (ParallelMmapWordlistBatchReader, std::path::PathBuf) {
        let path = write_temp_wordlist(bytes);
        let file = StdFile::open(&path).expect("failed to open temp wordlist");
        let mmap = unsafe { MmapOptions::new().map(&file) }.expect("failed to mmap temp wordlist");
        let reader = ParallelMmapWordlistBatchReader::new(
            test_device(),
            mmap,
            test_parser_config(parser_threads, chunk_bytes),
        )
        .expect("failed to build parallel mmap reader");
        (reader, path)
    }

    fn collect_all_words_from_mmap_reader(reader: &mut MmapWordlistBatchReader) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(batch) = reader
            .next_batch_reusing(None)
            .expect("sequential mmap batch")
        {
            for i in 0..batch.candidate_count() {
                out.push(batch.word(i).expect("candidate").to_vec());
            }
        }
        out
    }

    fn collect_all_words_from_parallel_reader(
        reader: &mut ParallelMmapWordlistBatchReader,
    ) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(batch) = reader
            .next_batch_reusing(None)
            .expect("parallel mmap batch")
        {
            for i in 0..batch.candidate_count() {
                out.push(batch.word(i).expect("candidate").to_vec());
            }
        }
        out
    }

    // ---- Layout / host-kernel contract tests --------------------------------
    #[test]
    fn hs256_params_round_trip_size_matches() {
        let params = Hs256BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs256BruteForceParams>());
    }

    // ---- Reporting helper behavior ------------------------------------------
    #[test]
    fn rate_report_snapshot_delta_computes_window_metrics() {
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_secs(2);
        let previous = RateReportSnapshot {
            reported_at: t0,
            candidates_tested: 100,
            gpu_wait: Duration::from_millis(400),
            wordlist_batch_build: Duration::from_millis(50),
            consumer_idle_wait: Duration::from_millis(10),
        };
        let current = RateReportSnapshot {
            reported_at: t1,
            candidates_tested: 300,
            gpu_wait: Duration::from_millis(900),
            wordlist_batch_build: Duration::from_millis(90),
            consumer_idle_wait: Duration::from_millis(25),
        };

        let delta = current.delta_since(previous);
        assert_eq!(delta.wall_time, Duration::from_secs(2));
        assert_eq!(delta.candidates_tested, 200);
        assert_eq!(delta.gpu_wait, Duration::from_millis(500));
        assert_eq!(delta.wordlist_batch_build, Duration::from_millis(40));
        assert_eq!(delta.consumer_idle_wait, Duration::from_millis(15));

        let end_to_end_rate = rate_per_second(delta.candidates_tested, delta.wall_time);
        let gpu_only_rate = rate_per_second(delta.candidates_tested, delta.gpu_wait);
        assert!((end_to_end_rate - 100.0).abs() < 0.000_001);
        assert!((gpu_only_rate - 400.0).abs() < 0.000_001);
    }

    #[test]
    fn rate_per_second_returns_zero_for_zero_elapsed() {
        assert_eq!(rate_per_second(123, Duration::ZERO), 0.0);
    }

    #[test]
    fn saturating_duration_sub_returns_zero_on_underflow() {
        assert_eq!(
            saturating_duration_sub(Duration::from_millis(10), Duration::from_millis(25)),
            Duration::ZERO
        );
    }

    // ---- JWT parsing validation ----------------------------------------------
    #[test]
    fn parse_hs256_jwt_success() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#);
        let sig = URL_SAFE_NO_PAD.encode([7u8; 32]);
        let jwt = format!("{header}.{payload}.{sig}");
        let parsed = parse_hs256_jwt(&jwt).unwrap();

        assert!(parsed.0.windows(1).count() > 0);
        assert_eq!(parsed.1.len(), 32);
        assert!(String::from_utf8(parsed.0).unwrap().contains('.'));
    }

    #[test]
    fn parse_hs256_jwt_rejects_wrong_segment_count() {
        let err = parse_hs256_jwt("abc.def").unwrap_err();
        assert!(format!("{err:#}").contains("3 dot-separated"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_non_hs256() {
        let header_json = br#"{"alg":"HS384","typ":"JWT"}"#;
        let payload_json = br#"{"sub":"alice"}"#;
        let header = URL_SAFE_NO_PAD.encode(header_json);
        let payload = URL_SAFE_NO_PAD.encode(payload_json);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 32]);
        let jwt = format!("{header}.{payload}.{sig}");

        let err = parse_hs256_jwt(&jwt).unwrap_err();
        assert!(format!("{err:#}").contains("HS256"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_wrong_signature_length() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 31]);
        let err = parse_hs256_jwt(&format!("{header}.{payload}.{sig}")).unwrap_err();
        assert!(format!("{err:#}").contains("32 bytes"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_bad_signature_base64() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let err = parse_hs256_jwt(&format!("{header}.{payload}.***")).unwrap_err();
        assert!(format!("{err:#}").contains("signature"));
    }

    // ---- Wordlist batching correctness ---------------------------------------
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
    // Verifies that the absolute wordlist index continues across batch splits,
    // which is required for deterministic GPU result reporting.
    fn wordlist_batch_reader_preserves_absolute_indices_across_batches() {
        let lines = (0..(MAX_CANDIDATES_PER_BATCH + 2))
            .map(|i| format!("pw{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let data = format!("{lines}\n");
        let mut reader =
            BufferedWordlistBatchReader::new(test_device(), Cursor::new(data.into_bytes()));

        let first = reader.next_batch().unwrap().unwrap();
        let second = reader.next_batch().unwrap().unwrap();

        assert_eq!(first.candidate_index_base, 0);
        assert_eq!(first.candidate_count(), MAX_CANDIDATES_PER_BATCH);
        assert_eq!(second.candidate_index_base, MAX_CANDIDATES_PER_BATCH as u64);
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
    fn parallel_mmap_reader_handles_final_line_without_newline() {
        let (mut reader, path) = parallel_mmap_reader_from_temp_file(b"alpha\nbeta", 2, 5);
        let batch = reader.next_batch_reusing(None).unwrap().unwrap();

        assert_eq!(batch.candidate_count(), 2);
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
        let total = MAX_CANDIDATES_PER_BATCH + 2;
        let mut data = Vec::with_capacity(total * 2);
        for _ in 0..total {
            data.extend_from_slice(b"x\n");
        }

        let (mut reader, path) = parallel_mmap_reader_from_temp_file(&data, 4, 1024);
        let first = reader.next_batch_reusing(None).unwrap().unwrap();
        let second = reader.next_batch_reusing(None).unwrap().unwrap();

        assert_eq!(first.candidate_index_base, 0);
        assert_eq!(first.candidate_count(), MAX_CANDIDATES_PER_BATCH);
        assert_eq!(second.candidate_index_base, MAX_CANDIDATES_PER_BATCH as u64);
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
    fn wordlist_producer_parallel_path_emits_batches_and_joins_cleanly() {
        let path = write_temp_wordlist(b"alpha\nbeta\n");
        let mut producer =
            WordlistProducer::spawn(path.clone(), test_device(), test_parser_config(2, 8));

        let first = producer.recv().expect("producer batch");
        match first {
            ProducerMessage::Batch {
                batch,
                parser_stats,
                ..
            } => {
                assert_eq!(batch.candidate_count(), 2);
                assert!(parser_stats.parser_threads >= 1);
                producer.recycle(batch);
            }
            ProducerMessage::Eof { .. } => panic!("unexpected EOF before first batch"),
            ProducerMessage::Error(err) => panic!("unexpected producer error: {err}"),
        }

        let second = producer.recv().expect("producer eof");
        match second {
            ProducerMessage::Eof { parser_stats } => {
                assert!(parser_stats.parser_chunks >= 1);
            }
            ProducerMessage::Batch { .. } => panic!("unexpected second batch"),
            ProducerMessage::Error(err) => panic!("unexpected producer error: {err}"),
        }

        producer.stop();
        producer.close_receiver();
        producer.join().expect("producer join");
        let _ = fs::remove_file(path);
    }

    // ---- Edge cases: empty input, missing newline, CRLF/LF normalization -----
    #[test]
    fn wordlist_batch_reader_returns_none_for_all_empty_lines() {
        let mut reader =
            BufferedWordlistBatchReader::new(test_device(), Cursor::new(b"\n\r\n".as_slice()));
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    // Final line handling is a common parser edge case when files do not end in
    // `\n`; behavior should match the normal newline-terminated case.
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
    // Shared helper used by both readers; this test locks CRLF/LF parity.
    fn trim_line_endings_trims_crlf_and_lf() {
        assert_eq!(trim_line_endings(b"abc\r\n"), b"abc");
        assert_eq!(trim_line_endings(b"abc\n"), b"abc");
        assert_eq!(trim_line_endings(b"abc"), b"abc");
    }
}
