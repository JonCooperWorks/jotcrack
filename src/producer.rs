//! # Producer pipeline for GPU batch dispatch
//!
//! This module implements the **producer/consumer pattern** that keeps the GPU
//! fed with batches of password candidates while the CPU parses the wordlist
//! in parallel.
//!
//! ## The core problem: CPU and GPU are independent processors
//!
//! Without pipelining, the workflow would be:
//!
//! ```text
//! CPU: [parse batch 1]                  [parse batch 2]                  [parse batch 3]
//! GPU:                 [hash batch 1]                  [hash batch 2]                  ...
//!                      ↑ GPU idle here                 ↑ GPU idle here
//! ```
//!
//! Each processor waits for the other. With pipelining:
//!
//! ```text
//! CPU: [parse batch 1][parse batch 2][parse batch 3][parse batch 4]...
//! GPU:                [hash batch 1][hash batch 2][hash batch 3]...
//!                     ↑ overlapped! No idle time (ideally)
//! ```
//!
//! The producer thread parses batches ahead of what the GPU is currently
//! processing. A bounded channel (`sync_channel`) connects them, with a
//! configurable depth (how many batches can be "in flight" between producer
//! and consumer). Deeper pipelines tolerate more variance in batch processing
//! time, at the cost of more memory (each in-flight batch holds GPU buffers).
//!
//! ## Two-stage pipeline: Planner -> Coordinator (inline pack)
//!
//! The producer uses two stages:
//!
//! 1. **Planner thread** — Scans the memory-mapped wordlist and determines
//!    batch boundaries (which lines go in which batch) without copying data.
//!    Produces `BatchPlan` objects.
//!
//! 2. **Coordinator** (runs on the producer thread) — Receives plans, packs
//!    them inline into `WordBatch` buffers, and sends filled batches to the
//!    consumer (GPU dispatch thread) via the output channel. Packing is done
//!    inline because it's fast enough (metadata memcpy + offset adjustment)
//!    that the overhead of a separate thread pool outweighs the work itself.
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

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail};

use super::gpu::GpuDevice;

use super::args::ParserConfig;
use super::batch::WordBatch;
use super::parser::{AnyWordlistBatchReader, BatchPlan, ParserStats, pack_batch_plan_into_batch};

/// Messages sent from the planner thread to the coordinator.
///
/// Three variants model the planner's lifecycle:
/// - `Plan` — "here's the next batch of work"
/// - `Eof` — "I've reached the end of the wordlist" (with final stats)
/// - `Error` — "something went wrong during parsing"
#[derive(Debug)]
enum PlannerWorkerMessage {
    Plan(BatchPlan),
    Eof { parser_stats: ParserStats },
    Error(String),
}

/// Main loop for the planner thread.
///
/// The planner scans the memory-mapped wordlist and determines batch boundaries
/// (which ranges of lines form each batch) without copying any candidate data.
///
/// ## Cooperative shutdown via `AtomicBool`
///
/// The `stop` flag is checked between batches. `Ordering::Relaxed` is sufficient
/// here because we don't need happens-before guarantees — we just need the
/// planner to eventually see the flag.
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

/// Process a single message from the planner thread.
///
/// Each planner message is handled exactly once — `Plan` is queued for packing,
/// `Eof` sets a flag, and `Error` short-circuits the coordinator.
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
///
/// - **`Batch`** — A filled `WordBatch` ready for GPU dispatch, with timing metadata.
/// - **`Eof`** — End of the wordlist. All candidates have been dispatched.
/// - **`Error`** — An unrecoverable error occurred in the producer pipeline.
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
    ///
    /// Uses a bounded `sync_channel(pipeline_depth)` to limit memory usage.
    /// The producer packs batches inline (no separate packer threads) — the
    /// packing work (metadata memcpy + offset adjustment) is fast enough
    /// that thread synchronization overhead would be a net negative.
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

/// Producer-thread main loop: plan batches, pack inline, and emit to consumer.
///
/// The coordinator receives plans from the planner thread, packs them inline
/// into `WordBatch` buffers (no separate packer threads), and sends filled
/// batches to the consumer via the output channel.
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

    #[test]
    fn wordlist_producer_parallel_path_emits_batches_and_joins_cleanly() {
        let path = write_temp_wordlist(b"alpha\nbeta\n");
        let mut producer = WordlistProducer::spawn(
            path.clone(),
            test_device(),
            test_parser_config(2, 8),
            DEFAULT_PIPELINE_DEPTH,
        );

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

    #[test]
    fn wordlist_producer_emits_batches_from_larger_wordlist() {
        let path = write_temp_wordlist(b"alpha\nbeta\ngamma\ndelta\n");
        let mut producer =
            WordlistProducer::spawn(path.clone(), test_device(), test_parser_config(2, 8), 4);

        let first = producer.recv().expect("producer first message");
        match first {
            ProducerMessage::Batch { batch, .. } => {
                assert_eq!(batch.candidate_count(), 4);
                producer.recycle(batch);
            }
            ProducerMessage::Eof { .. } => panic!("unexpected eof before batch"),
            ProducerMessage::Error(err) => panic!("unexpected producer error: {err}"),
        }

        let second = producer.recv().expect("producer second message");
        match second {
            ProducerMessage::Eof { parser_stats } => {
                assert!(parser_stats.parser_chunks >= 1);
            }
            ProducerMessage::Batch { .. } => panic!("unexpected extra batch"),
            ProducerMessage::Error(err) => panic!("unexpected producer error: {err}"),
        }

        producer.stop();
        producer.close_receiver();
        producer.join().expect("producer join");
        let _ = fs::remove_file(path);
    }
}
