//! # Producer pipeline for GPU batch dispatch
//!
//! This module implements the **producer/consumer pattern** that keeps the GPU
//! fed with batches of password candidates while the CPU parses the wordlist.
//!
//! ## Architecture
//!
//! On **macOS** (Metal, unified memory), the producer uses a multi-stage pipeline:
//! parser threads scan the mmap in parallel, a planner thread groups lines into
//! `BatchPlan` objects, and a coordinator packs them into `WordBatch` GPU buffers.
//!
//! On **Linux** (CUDA, discrete GPU), the entire mmap is uploaded to GPU VRAM
//! once at startup, so "packing" is just writing offset/length metadata to pinned
//! host memory. Multiple worker threads each scan a newline-aligned region of the
//! mmap with SIMD `memchr` and write offset/length metadata directly to GPU batch
//! buffers — no intermediate `ParsedChunk`, `BatchPlan`, or coordinator thread.
//!
//! ## WordBatch recycling (object pool pattern)
//!
//! Allocating GPU buffers is expensive. Instead of allocating a new `WordBatch`
//! for each batch, we recycle them:
//!
//! ```text
//! Producer ──[filled batch]──> Consumer (GPU dispatch)
//!    ↑                              │
//!    └───[empty batch]──────────────┘  (recycle channel)
//! ```
//!
//! After the GPU finishes with a batch, the consumer sends it back via a
//! separate "recycle" channel. The producer calls `reset_for_reuse()` and
//! fills it with new data. This keeps the steady-state allocation count
//! equal to `pipeline_depth` — no garbage collection, no allocator churn.

#[cfg(not(target_os = "linux"))]
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::anyhow;
#[cfg(not(target_os = "linux"))]
use anyhow::bail;

use super::gpu::GpuDevice;

use super::args::ParserConfig;
use super::batch::WordBatch;
use super::parser::ParserStats;

#[cfg(not(target_os = "linux"))]
use std::sync::mpsc::TryRecvError;
#[cfg(not(target_os = "linux"))]
use super::parser::{AnyWordlistBatchReader, BatchPlan, pack_batch_plan_into_batch};

// ---------------------------------------------------------------------------
// macOS: multi-stage pipeline (parser threads → planner → coordinator)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
enum PlannerWorkerMessage {
    Plan(BatchPlan),
    Eof { parser_stats: ParserStats },
    Error(String),
}

#[cfg(not(target_os = "linux"))]
fn planner_worker_main(
    mut reader: AnyWordlistBatchReader,
    stop: Arc<AtomicBool>,
    plan_tx: crossbeam_channel::Sender<PlannerWorkerMessage>,
) {
    let mut next_seq_no = 0u64;
    while !stop.load(Ordering::Relaxed) {
        match reader.next_batch_plan() {
            Ok(Some(mut plan)) => {
                plan.seq_no = next_seq_no;
                next_seq_no = match next_seq_no.checked_add(1) {
                    Some(v) => v,
                    None => {
                        let _ = plan_tx.send(PlannerWorkerMessage::Error(
                            "batch sequence overflow".to_string(),
                        ));
                        return;
                    }
                };
                if plan_tx.send(PlannerWorkerMessage::Plan(plan)).is_err() {
                    return;
                }
            }
            Ok(None) => {
                let _ = plan_tx.send(PlannerWorkerMessage::Eof {
                    parser_stats: reader.parser_stats(),
                });
                return;
            }
            Err(err) => {
                let _ = plan_tx.send(PlannerWorkerMessage::Error(format!("{err:#}")));
                return;
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn handle_planner_message(
    message: PlannerWorkerMessage,
    pending_plans: &mut VecDeque<BatchPlan>,
    planner_eof: &mut bool,
    latest_parser_stats: &mut ParserStats,
) -> anyhow::Result<()> {
    match message {
        PlannerWorkerMessage::Plan(plan) => {
            *latest_parser_stats = plan.parser_stats;
            pending_plans.push_back(plan);
            Ok(())
        }
        PlannerWorkerMessage::Eof { parser_stats } => {
            *latest_parser_stats = parser_stats;
            *planner_eof = true;
            Ok(())
        }
        PlannerWorkerMessage::Error(message) => bail!("planner worker failed: {message}"),
    }
}

/// Messages sent from the producer pipeline to the consumer (GPU dispatch thread).
pub(crate) enum ProducerMessage {
    Batch {
        batch: WordBatch,
        build_time: Duration,
        plan_time: Duration,
        pack_time: Duration,
        parser_stats: ParserStats,
    },
    Eof {
        parser_stats: ParserStats,
    },
    Error(String),
}

/// Owns the producer thread and channel endpoints used to overlap wordlist
/// parsing with GPU dispatch.
pub(crate) struct WordlistProducer {
    rx: Option<Receiver<ProducerMessage>>,
    recycle_tx: SyncSender<WordBatch>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl WordlistProducer {
    /// Start the producer pipeline.
    pub(crate) fn spawn(
        wordlist_path: PathBuf,
        device: GpuDevice,
        parser_config: ParserConfig,
        pipeline_depth: usize,
    ) -> Self {
        let pipeline_depth = pipeline_depth.max(1);
        let (tx, rx) = sync_channel(pipeline_depth);
        let (recycle_tx, recycle_rx) = sync_channel(pipeline_depth);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);
        let tx_err = tx.clone();
        let handle = thread::spawn(move || {
            if let Err(err) = run_wordlist_producer(
                wordlist_path,
                device,
                parser_config,
                pipeline_depth,
                stop_for_thread,
                tx,
                recycle_rx,
            ) {
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

    /// Signal the producer to stop after finishing the current batch.
    pub(crate) fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Receive the next batch (or terminal message) from the producer.
    pub(crate) fn recv(&self) -> anyhow::Result<ProducerMessage> {
        let rx = self
            .rx
            .as_ref()
            .ok_or_else(|| anyhow!("wordlist producer receiver already closed"))?;
        rx.recv()
            .map_err(|_| anyhow!("wordlist producer terminated unexpectedly"))
    }

    /// Drop the receiving end to unblock a potentially stuck producer.
    pub(crate) fn close_receiver(&mut self) {
        let _ = self.rx.take();
    }

    /// Return a used batch to the producer for recycling (best-effort).
    pub(crate) fn recycle(&self, batch: WordBatch) {
        let _ = self.recycle_tx.try_send(batch);
    }

    /// Wait for the background thread to finish and propagate any panics.
    pub(crate) fn join(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("wordlist producer thread panicked"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Linux: parallel direct-write producer (no coordinator bottleneck)
//
// Multiple worker threads each scan a newline-aligned region of the mmap
// with memchr and write offset/length metadata directly to GPU batch
// buffers (pinned host memory). This eliminates all intermediate data
// structures (ParsedChunk, BatchPlan, block summaries) and avoids
// cross-core cache transfers — each thread reads its mmap region and
// writes to its own batch without touching other threads' data.
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn run_wordlist_producer(
    wordlist_path: PathBuf,
    device: GpuDevice,
    parser_config: ParserConfig,
    pipeline_depth: usize,
    stop: Arc<AtomicBool>,
    tx: SyncSender<ProducerMessage>,
    recycle_rx: Receiver<WordBatch>,
) -> anyhow::Result<()> {
    use memmap2::{Advice, MmapOptions};

    let pipeline_depth = pipeline_depth.max(1);
    let num_workers = parser_config.parser_threads;

    // Preallocate the steady-state batch pool into a shared channel.
    let (batch_pool_tx, batch_pool_rx) =
        crossbeam_channel::bounded::<WordBatch>(pipeline_depth + num_workers);
    for _ in 0..pipeline_depth {
        batch_pool_tx
            .send(WordBatch::new(&device, 0))
            .map_err(|_| anyhow!("batch pool send failed"))?;
    }

    // Open and mmap the wordlist.
    let file = std::fs::File::open(&wordlist_path)
        .map_err(|e| anyhow!("failed to open wordlist {:?}: {e}", wordlist_path))?;
    let mmap = Arc::new(
        unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| anyhow!("failed to mmap wordlist {:?}: {e}", wordlist_path))?,
    );
    let _ = mmap.advise(Advice::Sequential);

    // Plan newline-aligned regions (reuse the same logic as plan_mmap_chunks).
    let bytes: &[u8] = &mmap;
    let region_size = if num_workers == 0 {
        bytes.len()
    } else {
        (bytes.len() / num_workers).max(1)
    };
    let mut regions: Vec<(usize, usize)> = Vec::with_capacity(num_workers);
    let mut start = 0usize;
    while start < bytes.len() {
        let target_end = (start + region_size).min(bytes.len());
        let end = if target_end >= bytes.len() {
            bytes.len()
        } else {
            match memchr::memchr(b'\n', &bytes[target_end..]) {
                Some(rel) => target_end + rel + 1,
                None => bytes.len(),
            }
        };
        regions.push((start, end));
        start = end;
    }

    let total_workers = regions.len();
    let workers_done = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Spawn worker threads.
    let mut handles = Vec::with_capacity(total_workers);
    for (region_start, region_end) in regions {
        let mmap_ref = Arc::clone(&mmap);
        let stop_w = Arc::clone(&stop);
        let tx_w = tx.clone();
        let pool_rx = batch_pool_rx.clone();
        let done_counter = Arc::clone(&workers_done);
        let total = total_workers;
        let file_len = bytes.len();

        handles.push(thread::spawn(move || {
            let result = scan_region_direct(
                &mmap_ref,
                region_start,
                region_end,
                file_len,
                &stop_w,
                &tx_w,
                &pool_rx,
            );

            // If this is the last worker, send EOF.
            let finished = done_counter.fetch_add(1, Ordering::AcqRel) + 1;
            if finished == total {
                let _ = tx_w.send(ProducerMessage::Eof {
                    parser_stats: ParserStats {
                        parser_threads: total,
                        parser_chunk_bytes: file_len,
                        parser_chunks: total as u64,
                        parser_skipped_oversize: 0,
                    },
                });
            }

            if let Err(e) = result {
                let _ = tx_w.send(ProducerMessage::Error(format!("{e:#}")));
            }
        }));
    }

    // Drop our copies so channels close when workers finish.
    drop(tx);
    drop(batch_pool_rx);

    // This thread forwards recycled batches from the consumer back to the
    // worker pool. It exits when either all workers finish (pool receivers
    // dropped → send fails) or the stop signal is set.
    loop {
        if workers_done.load(Ordering::Acquire) >= total_workers {
            break;
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
        // Use recv_timeout to avoid blocking forever when the consumer
        // stops recycling (e.g., after finding a match).
        match recycle_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(b) => {
                // Best-effort: if all pool receivers are gone, just drop the batch.
                let _ = batch_pool_tx.try_send(b);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    drop(batch_pool_tx);

    for h in handles {
        if h.join().is_err() {
            return Err(anyhow!("producer worker thread panicked"));
        }
    }

    Ok(())
}

/// One worker thread: scan a region of the mmap and write batches directly.
#[cfg(target_os = "linux")]
fn scan_region_direct(
    mmap: &[u8],
    region_start: usize,
    region_end: usize,
    _file_len: usize,
    stop: &AtomicBool,
    tx: &SyncSender<ProducerMessage>,
    pool_rx: &crossbeam_channel::Receiver<WordBatch>,
) -> anyhow::Result<()> {
    use super::batch::{MAX_CANDIDATES_PER_BATCH, MAX_WORD_BYTES_PER_BATCH};

    let region = &mmap[region_start..region_end];

    // Local staging buffers — accumulate in L1-cached memory, bulk-flush
    // to pinned GPU memory when a batch boundary is hit.
    let mut local_offsets: Vec<u32> = Vec::with_capacity(MAX_CANDIDATES_PER_BATCH);
    let mut local_lengths: Vec<u16> = Vec::with_capacity(MAX_CANDIDATES_PER_BATCH);
    let mut batch_word_bytes_len: usize = 0;
    let mut batch_max_word_len: u16 = 0;

    let mut batch = match pool_rx.recv() {
        Ok(b) => b,
        Err(_) => return Ok(()),
    };
    batch.reset_for_reuse(0);
    let mut batch_start = Instant::now();

    let mut next_line_start: usize = 0;
    for newline_pos in memchr::memchr_iter(b'\n', region) {
        let this_line_start = next_line_start;
        next_line_start = newline_pos + 1;

        // Trim trailing \r for CRLF.
        let trimmed_end = if newline_pos > this_line_start && region[newline_pos - 1] == b'\r' {
            newline_pos - 1
        } else {
            newline_pos
        };
        let line_len = trimmed_end - this_line_start;

        if line_len == 0 || line_len > u16::MAX as usize {
            continue;
        }

        // Check batch boundary.
        let count = local_offsets.len();
        if count > 0
            && (count >= MAX_CANDIDATES_PER_BATCH
                || batch_word_bytes_len + line_len > MAX_WORD_BYTES_PER_BATCH)
        {
            // Flush and send.
            flush_staged_to_batch(
                &mut batch,
                &local_offsets,
                &local_lengths,
                batch_word_bytes_len,
                batch_max_word_len,
            );
            let build_time = batch_start.elapsed();
            if tx
                .send(ProducerMessage::Batch {
                    batch,
                    build_time,
                    plan_time: Duration::ZERO,
                    pack_time: Duration::ZERO,
                    parser_stats: ParserStats {
                        parser_threads: 1,
                        parser_chunk_bytes: region.len(),
                        parser_chunks: 1,
                        parser_skipped_oversize: 0,
                    },
                })
                .is_err()
            {
                return Ok(());
            }

            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }

            batch = match pool_rx.recv() {
                Ok(b) => b,
                Err(_) => return Ok(()),
            };
            batch.reset_for_reuse(0);
            batch_start = Instant::now();
            local_offsets.clear();
            local_lengths.clear();
            batch_word_bytes_len = 0;
            batch_max_word_len = 0;
        }

        // Stage: absolute mmap offset.
        local_offsets.push((region_start + this_line_start) as u32);
        local_lengths.push(line_len as u16);
        batch_word_bytes_len += line_len;
        batch_max_word_len = batch_max_word_len.max(line_len as u16);
    }

    // Handle final line without trailing newline.
    if next_line_start < region.len() {
        let this_line_start = next_line_start;
        let trimmed_end = if region[region.len() - 1] == b'\r' {
            region.len() - 1
        } else {
            region.len()
        };
        let line_len = trimmed_end - this_line_start;
        if line_len > 0 && line_len <= u16::MAX as usize {
            let count = local_offsets.len();
            if count > 0
                && (count >= MAX_CANDIDATES_PER_BATCH
                    || batch_word_bytes_len + line_len > MAX_WORD_BYTES_PER_BATCH)
            {
                flush_staged_to_batch(
                    &mut batch,
                    &local_offsets,
                    &local_lengths,
                    batch_word_bytes_len,
                    batch_max_word_len,
                );
                let build_time = batch_start.elapsed();
                if tx
                    .send(ProducerMessage::Batch {
                        batch,
                        build_time,
                        plan_time: Duration::ZERO,
                        pack_time: Duration::ZERO,
                        parser_stats: ParserStats {
                            parser_threads: 1,
                            parser_chunk_bytes: region.len(),
                            parser_chunks: 1,
                            parser_skipped_oversize: 0,
                        },
                    })
                    .is_err()
                {
                    return Ok(());
                }
                batch = match pool_rx.recv() {
                    Ok(b) => b,
                    Err(_) => return Ok(()),
                };
                batch.reset_for_reuse(0);
                batch_start = Instant::now();
                local_offsets.clear();
                local_lengths.clear();
                batch_word_bytes_len = 0;
                batch_max_word_len = 0;
            }
            local_offsets.push((region_start + this_line_start) as u32);
            local_lengths.push(line_len as u16);
            batch_word_bytes_len += line_len;
            batch_max_word_len = batch_max_word_len.max(line_len as u16);
        }
    }

    // Emit final partial batch.
    if !local_offsets.is_empty() {
        flush_staged_to_batch(
            &mut batch,
            &local_offsets,
            &local_lengths,
            batch_word_bytes_len,
            batch_max_word_len,
        );
        let build_time = batch_start.elapsed();
        let _ = tx.send(ProducerMessage::Batch {
            batch,
            build_time,
            plan_time: Duration::ZERO,
            pack_time: Duration::ZERO,
            parser_stats: ParserStats {
                parser_threads: 1,
                parser_chunk_bytes: region.len(),
                parser_chunks: 1,
                parser_skipped_oversize: 0,
            },
        });
    }

    Ok(())
}

/// Bulk-copy staged offsets/lengths from local (cached) buffers to pinned
/// GPU memory. Offsets are already absolute mmap positions.
#[cfg(target_os = "linux")]
fn flush_staged_to_batch(
    batch: &mut WordBatch,
    offsets: &[u32],
    lengths: &[u16],
    word_bytes_len: usize,
    max_word_len: u16,
) {
    let count = offsets.len();
    if count == 0 {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            offsets.as_ptr(),
            batch.word_offsets_ptr_mut().add(batch.candidate_count()),
            count,
        );
        std::ptr::copy_nonoverlapping(
            lengths.as_ptr(),
            batch.word_lengths_ptr_mut().add(batch.candidate_count()),
            count,
        );
    }
    batch.set_staged_counts(count, word_bytes_len, max_word_len);
}

// ---------------------------------------------------------------------------
// macOS: multi-stage pipeline (planner → coordinator with inline pack)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "linux"))]
fn run_wordlist_producer(
    wordlist_path: PathBuf,
    device: GpuDevice,
    parser_config: ParserConfig,
    pipeline_depth: usize,
    stop: Arc<AtomicBool>,
    tx: SyncSender<ProducerMessage>,
    recycle_rx: Receiver<WordBatch>,
) -> anyhow::Result<()> {
    let pipeline_depth = pipeline_depth.max(1);

    // Preallocate the steady-state batch pool up front.
    let mut preallocated_batches = Vec::with_capacity(pipeline_depth);
    for _ in 0..pipeline_depth {
        preallocated_batches.push(WordBatch::new(&device, 0));
    }
    let reader = AnyWordlistBatchReader::new(device, &wordlist_path, parser_config)?;
    let shared_mmap = reader.shared_mmap();

    let (plan_tx, plan_rx) = crossbeam_channel::bounded::<PlannerWorkerMessage>(pipeline_depth);
    let stop_for_planner = Arc::clone(&stop);
    let planner_handle =
        thread::spawn(move || planner_worker_main(reader, stop_for_planner, plan_tx));

    let coordinator_result = (|| -> anyhow::Result<()> {
        let mut available_batches = preallocated_batches;
        let mut pending_plans = VecDeque::<BatchPlan>::new();
        let mut planner_eof = false;
        let mut latest_parser_stats = ParserStats {
            parser_threads: parser_config.parser_threads,
            parser_chunk_bytes: parser_config.chunk_bytes,
            ..ParserStats::default()
        };

        loop {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }

            let mut made_progress = false;

            // Pull back any batches the consumer has finished using.
            loop {
                match recycle_rx.try_recv() {
                    Ok(batch) => {
                        available_batches.push(batch);
                        made_progress = true;
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                }
            }

            // Pull planner results without blocking.
            loop {
                match plan_rx.try_recv() {
                    Ok(message) => {
                        handle_planner_message(
                            message,
                            &mut pending_plans,
                            &mut planner_eof,
                            &mut latest_parser_stats,
                        )?;
                        made_progress = true;
                    }
                    Err(crossbeam_channel::TryRecvError::Empty) => break,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        if !planner_eof && !stop.load(Ordering::Relaxed) {
                            bail!("planner worker terminated unexpectedly");
                        }
                        planner_eof = true;
                        break;
                    }
                }
            }

            // Inline-pack pending plans when batches are available.
            while !pending_plans.is_empty() {
                let mut batch = match available_batches.pop() {
                    Some(b) => b,
                    None => break, // no batch available, will block on recycle below
                };
                let plan = pending_plans.pop_front().unwrap();
                let pack_start = Instant::now();
                pack_batch_plan_into_batch(shared_mmap.as_ref(), &plan, &mut batch)?;
                let pack_time = pack_start.elapsed();

                latest_parser_stats = plan.parser_stats;
                let build_time = plan
                    .plan_time
                    .checked_add(pack_time)
                    .unwrap_or(plan.plan_time + pack_time);

                let msg = ProducerMessage::Batch {
                    batch,
                    build_time,
                    plan_time: plan.plan_time,
                    pack_time,
                    parser_stats: plan.parser_stats,
                };
                if tx.send(msg).is_err() {
                    return Ok(());
                }
                made_progress = true;
            }

            if planner_eof && pending_plans.is_empty() {
                let _ = tx.send(ProducerMessage::Eof {
                    parser_stats: latest_parser_stats,
                });
                return Ok(());
            }

            if made_progress {
                continue;
            }

            // No progress — block on the source that can unblock the pipeline.
            if !pending_plans.is_empty() {
                // Have plans but no batches — wait for recycling.
                match recycle_rx.recv() {
                    Ok(batch) => available_batches.push(batch),
                    Err(_) => return Ok(()),
                }
                continue;
            }

            if !planner_eof {
                match plan_rx.recv() {
                    Ok(message) => {
                        handle_planner_message(
                            message,
                            &mut pending_plans,
                            &mut planner_eof,
                            &mut latest_parser_stats,
                        )?;
                    }
                    Err(_) if stop.load(Ordering::Relaxed) => return Ok(()),
                    Err(_) => bail!("planner worker terminated unexpectedly"),
                }
                continue;
            }
        }
    })();

    // Cleanup: drop channel, join planner thread.
    drop(plan_rx);

    let mut join_error: Option<anyhow::Error> = None;
    if planner_handle.join().is_err() && join_error.is_none() {
        join_error = Some(anyhow!("planner worker thread panicked"));
    }

    match (coordinator_result, join_error) {
        (Err(err), _) => Err(err),
        (Ok(_), Some(err)) => Err(err),
        (Ok(()), None) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::super::args::DEFAULT_PIPELINE_DEPTH;
    use super::super::test_support::{test_device, test_parser_config, write_temp_wordlist};
    use super::*;

    /// Drain all batches from the producer, returning the total candidate count.
    fn drain_all_batches(producer: &mut WordlistProducer) -> (usize, usize) {
        let mut total_candidates = 0usize;
        let mut batch_count = 0usize;
        loop {
            let msg = producer.recv().expect("producer message");
            match msg {
                ProducerMessage::Batch { batch, .. } => {
                    total_candidates += batch.candidate_count();
                    batch_count += 1;
                    producer.recycle(batch);
                }
                ProducerMessage::Eof { .. } => break,
                ProducerMessage::Error(err) => panic!("unexpected producer error: {err}"),
            }
        }
        (total_candidates, batch_count)
    }

    #[test]
    fn wordlist_producer_parallel_path_emits_batches_and_joins_cleanly() {
        let path = write_temp_wordlist(b"alpha\nbeta\n");
        let mut producer = WordlistProducer::spawn(
            path.clone(),
            test_device(),
            test_parser_config(2, 8),
            DEFAULT_PIPELINE_DEPTH,
        );

        let (total, batch_count) = drain_all_batches(&mut producer);
        assert_eq!(total, 2, "expected 2 total candidates");
        assert!(batch_count >= 1, "expected at least 1 batch");

        producer.stop();
        producer.close_receiver();
        producer.join().expect("producer join");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn wordlist_producer_emits_batches_from_larger_wordlist() {
        let path = write_temp_wordlist(b"alpha\nbeta\ngamma\ndelta\n");
        let mut producer =
            WordlistProducer::spawn(path.clone(), test_device(), test_parser_config(2, 8), 4);

        let (total, batch_count) = drain_all_batches(&mut producer);
        assert_eq!(total, 4, "expected 4 total candidates");
        assert!(batch_count >= 1, "expected at least 1 batch");

        producer.stop();
        producer.close_receiver();
        producer.join().expect("producer join");
        let _ = fs::remove_file(path);
    }
}
