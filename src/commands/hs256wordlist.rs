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
use memmap2::{Mmap, MmapOptions};
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
const MAX_CANDIDATES_PER_BATCH: usize = 1_048_576;
const MAX_WORD_BYTES_PER_BATCH: usize = 64 * 1024 * 1024;
// Small bounded queue: enough to overlap CPU/GPU without letting parsed data
// accumulate unbounded in memory when the GPU is slower.
const DEFAULT_PIPELINE_DEPTH: usize = 2;

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
    #[arg(long)]
    pub autotune: bool,
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
            std::slice::from_raw_parts(
                self.word_offsets_ptr as *const u32,
                self.candidate_count,
            )
        }
    }

    fn lengths_slice(&self) -> &[u16] {
        // SAFETY: `word_lengths_buf` stores `u16` entries; the first
        // `candidate_count` entries are initialized by `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(
                self.word_lengths_ptr as *const u16,
                self.candidate_count,
            )
        }
    }

    fn word_bytes_slice(&self) -> &[u8] {
        // SAFETY: the first `word_bytes_len` bytes are initialized by
        // `push_candidate`.
        unsafe {
            std::slice::from_raw_parts(
                self.word_bytes_ptr as *const u8,
                self.word_bytes_len,
            )
        }
    }

    fn can_fit(&self, line_len: usize) -> bool {
        // Preserve the original batching rule:
        // - always enforce candidate-count cap
        // - only enforce byte-cap after the first candidate is present
        let would_exceed_count = self.candidate_count >= MAX_CANDIDATES_PER_BATCH;
        let would_exceed_bytes =
            self.candidate_count != 0 && (self.word_bytes_len + line_len > MAX_WORD_BYTES_PER_BATCH);
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
struct MmapWordlistBatchReader {
    device: Device,
    mmap: Mmap,
    cursor: usize,
    // Same semantic as the test-only buffered reader: absolute index of the next
    // non-empty candidate line, tracked in `u64` for huge wordlists.
    next_index: u64,
}

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
            let mut line_end = line_start;
            while line_end < bytes.len() && bytes[line_end] != b'\n' {
                line_end += 1;
            }
            // Match the buffered reader's CRLF behavior by trimming a trailing
            // carriage return before packaging the candidate.
            let mut trimmed_end = line_end;
            if trimmed_end > line_start && bytes[trimmed_end - 1] == b'\r' {
                trimmed_end -= 1;
            }
            let next_cursor = if line_end < bytes.len() {
                line_end + 1
            } else {
                line_end
            };
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

// Small abstraction so the producer thread can remain agnostic to reader
// details. Runtime currently has a single concrete path (mmap).
enum AnyWordlistBatchReader {
    Mmap(MmapWordlistBatchReader),
}

impl AnyWordlistBatchReader {
    // Runtime path is mmap-only. If mapping fails, return an error instead of
    // silently switching to a slower path (which would hide perf regressions).
    fn new(device: Device, path: &PathBuf) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open wordlist {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap wordlist {}", path.display()))?;
        Ok(Self::Mmap(MmapWordlistBatchReader::new(device, mmap)))
    }

    // Forward batch production to the selected reader strategy.
    fn next_batch_reusing(
        &mut self,
        recycled: Option<WordBatch>,
    ) -> anyhow::Result<Option<WordBatch>> {
        // `recycled` is moved into exactly one concrete reader branch, so the
        // batch allocation has a single owner throughout the rebuild step.
        match self {
            Self::Mmap(reader) => reader.next_batch_reusing(recycled),
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
    },
    Eof,
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
    fn spawn(wordlist_path: PathBuf, device: Device) -> Self {
        let (tx, rx) = sync_channel(DEFAULT_PIPELINE_DEPTH);
        // Match the forward pipeline depth so we can recycle a small pool of
        // batch allocations without letting memory usage grow unbounded.
        let (recycle_tx, recycle_rx) = sync_channel(DEFAULT_PIPELINE_DEPTH);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);
        let tx_err = tx.clone();
        let handle = thread::spawn(move || {
            if let Err(err) =
                run_wordlist_producer(wordlist_path, device, stop_for_thread, tx, recycle_rx)
            {
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
    stop: Arc<AtomicBool>,
    tx: SyncSender<ProducerMessage>,
    // Optional returned batches from the consumer. The producer owns the next
    // parse/build step, so this is where allocation reuse naturally fits.
    recycle_rx: Receiver<WordBatch>,
) -> anyhow::Result<()> {
    let mut reader = AnyWordlistBatchReader::new(device, &wordlist_path)?;
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
        let message = match maybe_batch {
            Some(batch) => ProducerMessage::Batch { batch, build_time },
            None => {
                let _ = tx.send(ProducerMessage::Eof);
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
    fn active_pipeline_for_view(&self, batch: DispatchBatchView<'_>) -> &metal::ComputePipelineState {
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
        self.dispatch_batch_view(target_signature, batch.as_dispatch_view(), self.threadgroup_width)
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
            let (_result, timings) = self.dispatch_batch_view(target_signature, sample_view, width)?;
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
    let mut producer = WordlistProducer::spawn(args.wordlist.clone(), gpu.device.clone());
    let run_result = (|| -> anyhow::Result<bool> {
        let started_at = Instant::now();
        let mut last_rate_report_at = started_at;
        let mut candidates_tested: u64 = 0;
        let mut autotune_done = !args.autotune;
        let mut timings = RunTimings::default();

        loop {
            // Time how long the consumer waits on the producer. This measures
            // exposed parser latency (work not hidden behind GPU execution).
            let recv_started = Instant::now();
            let message = producer.recv()?;
            timings.consumer_idle_wait += recv_started.elapsed();

            let (batch, build_time) = match message {
                ProducerMessage::Batch { batch, build_time } => (batch, build_time),
                ProducerMessage::Eof => break,
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
                    .ok_or_else(|| anyhow!("candidate index overflow while reconstructing result"))?;
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
            if now.duration_since(last_rate_report_at) >= Duration::from_secs(1) {
                // Periodic progress uses cumulative counts so the rate smooths
                // out per-batch timing variance.
                let elapsed = now.duration_since(started_at);
                let rate_end_to_end = rate_per_second(candidates_tested, elapsed);
                let rate_gpu_only = rate_per_second(candidates_tested, timings.gpu_wait);
                eprintln!(
                    "RATE end_to_end={}/s gpu_only={}/s ({} tested in {:.2}s, tpg={})",
                    format_human_count(rate_end_to_end),
                    format_human_count(rate_gpu_only),
                    format_human_count(candidates_tested as f64),
                    elapsed.as_secs_f64(),
                    gpu.current_threadgroup_width()
                );
                last_rate_report_at = now;
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
fn rate_per_second(candidates_tested: u64, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        candidates_tested as f64 / elapsed.as_secs_f64()
    }
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
    use std::io::Cursor;

    fn test_device() -> Device {
        Device::system_default().expect("Metal device is required for hs256wordlist tests")
    }

    // ---- Layout / host-kernel contract tests --------------------------------
    #[test]
    fn hs256_params_round_trip_size_matches() {
        let params = Hs256BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs256BruteForceParams>());
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
        let mut reader = BufferedWordlistBatchReader::new(test_device(), Cursor::new(data.as_slice()));
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
        let mut reader = BufferedWordlistBatchReader::new(test_device(), Cursor::new(data.into_bytes()));
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
        let mut reader = BufferedWordlistBatchReader::new(test_device(), Cursor::new(data.into_bytes()));

        let first = reader.next_batch().unwrap().unwrap();
        let second = reader.next_batch().unwrap().unwrap();

        assert_eq!(first.candidate_index_base, 0);
        assert_eq!(first.candidate_count(), MAX_CANDIDATES_PER_BATCH);
        assert_eq!(second.candidate_index_base, MAX_CANDIDATES_PER_BATCH as u64);
        assert_eq!(second.candidate_count(), 2);
    }

    // ---- Edge cases: empty input, missing newline, CRLF/LF normalization -----
    #[test]
    fn wordlist_batch_reader_returns_none_for_all_empty_lines() {
        let mut reader = BufferedWordlistBatchReader::new(test_device(), Cursor::new(b"\n\r\n".as_slice()));
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
