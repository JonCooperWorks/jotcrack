//! # Multi-threaded producer pipeline for GPU batch dispatch
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
//! time, at the cost of more memory (each in-flight batch holds ~67 MiB of
//! Metal buffers).
//!
//! ## Three-stage pipeline: Planner -> Packers -> Coordinator
//!
//! The producer itself is internally pipelined into three stages:
//!
//! 1. **Planner thread** — Scans the memory-mapped wordlist and determines
//!    batch boundaries (which lines go in which batch) without copying data.
//!    Produces `BatchPlan` objects.
//!
//! 2. **Packer worker threads** — Take a `BatchPlan` + empty `WordBatch` and
//!    copy the actual candidate bytes into the Metal shared buffers. Multiple
//!    packer threads can work in parallel on different batches.
//!
//! 3. **Coordinator** (runs on the producer thread) — Orchestrates the planner
//!    and packers, manages the batch pool, and sends filled batches to the
//!    consumer (GPU dispatch thread) via the output channel.
//!
//! ## WordBatch recycling (object pool pattern)
//!
//! Allocating Metal shared buffers is expensive (Objective-C FFI + kernel VM
//! operations). Instead of allocating a new `WordBatch` for each batch, we
//! recycle them:
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
use memmap2::Mmap;
use metal::Device;

use super::args::ParserConfig;
use super::batch::WordBatch;
use super::parser::{AnyWordlistBatchReader, BatchPlan, ParserStats, pack_batch_plan_into_batch};

/// A unit of work sent to a packer worker thread: "fill this batch using this plan."
///
/// The `plan` describes which lines from the wordlist go into this batch (byte
/// ranges, line counts). The `batch` is an empty (or recycled) `WordBatch` with
/// pre-allocated Metal buffers ready to be filled.
struct PackerJob {
    plan: BatchPlan,
    batch: WordBatch,
}

/// The result of a packer worker completing a job.
///
/// Contains the now-filled `WordBatch` plus timing data for performance monitoring.
#[derive(Debug)]
struct PackerResult {
    batch: WordBatch,
    plan_time: Duration,
    pack_time: Duration,
    parser_stats: ParserStats,
}

/// Messages sent from packer workers back to the coordinator.
///
/// This is a simple enum-based protocol: either the packing succeeded
/// (`Packed`) or it failed (`Error`). Using an enum here instead of
/// `Result<PackerResult, String>` makes pattern matching more explicit
/// and lets us include additional context like `seq_no` in the error case.
#[derive(Debug)]
enum PackerWorkerMessage {
    Packed(PackerResult),
    Error { seq_no: u64, message: String },
}

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

/// Main loop for a packer worker thread.
///
/// Each packer worker runs this loop: receive a job, pack the candidates from
/// the mmap into the Metal batch buffers, send the result back. Workers are
/// stateless — they share an `Arc<Mmap>` for read-only access to the wordlist
/// file and communicate exclusively through channels.
///
/// The `crossbeam_channel` crate is used here (instead of `std::sync::mpsc`)
/// because crossbeam channels support multiple consumers (`Receiver` is
/// `Clone`), which lets us fan out work to multiple packer threads from a
/// single job queue. `std::sync::mpsc::Receiver` is NOT `Clone`.
fn packer_worker_main(
    mmap: Arc<Mmap>,
    job_rx: crossbeam_channel::Receiver<PackerJob>,
    result_tx: crossbeam_channel::Sender<PackerWorkerMessage>,
) {
    while let Ok(PackerJob { plan, mut batch }) = job_rx.recv() {
        let pack_started = Instant::now();
        match pack_batch_plan_into_batch(mmap.as_ref(), &plan, &mut batch) {
            Ok(()) => {
                let message = PackerWorkerMessage::Packed(PackerResult {
                    batch,
                    plan_time: plan.plan_time,
                    pack_time: pack_started.elapsed(),
                    parser_stats: plan.parser_stats,
                });
                if result_tx.send(message).is_err() {
                    return;
                }
            }
            Err(err) => {
                let _ = result_tx.send(PackerWorkerMessage::Error {
                    seq_no: plan.seq_no,
                    message: format!("{err:#}"),
                });
                return;
            }
        }
    }
}

/// Main loop for the planner thread.
///
/// The planner scans the memory-mapped wordlist and determines batch boundaries
/// (which ranges of lines form each batch) without copying any candidate data.
/// This is a lightweight operation compared to packing — it just tracks counts
/// and byte totals.
///
/// ## Why separate planning from packing?
///
/// Planning is sequential (it must scan the wordlist in order to determine
/// where batches begin and end), but packing is embarrassingly parallel (each
/// batch can be filled independently). By separating the two, we let the
/// planner run ahead while multiple packer threads fill batches concurrently.
///
/// ## Cooperative shutdown via `AtomicBool`
///
/// The `stop` flag is checked between batches. `Ordering::Relaxed` is sufficient
/// here because we don't need happens-before guarantees — we just need the
/// planner to eventually see the flag. A few extra batches planned after the
/// stop signal are harmless (they'll be discarded by the coordinator).
fn planner_worker_main(
    mut reader: AnyWordlistBatchReader,
    stop: Arc<AtomicBool>,
    plan_tx: crossbeam_channel::Sender<PlannerWorkerMessage>,
) {
    let mut next_seq_no = 0u64;
    // Planner owns sequential plan numbering so downstream stages can keep
    // deterministic emission and error messages without shared counters.
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
                // A disconnected receiver means the coordinator is already
                // exiting, so this worker should stop quietly as well.
                if plan_tx.send(PlannerWorkerMessage::Plan(plan)).is_err() {
                    return;
                }
            }
            Ok(None) => {
                // Forward final parser stats snapshot with EOF so the consumer
                // can print accurate cumulative parser counters.
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
/// Extracted as a helper so the coordinator loop stays readable. Each planner
/// message is handled exactly once — `Plan` is queued for packing, `Eof` sets
/// a flag, and `Error` short-circuits the coordinator.
///
/// Note that `latest_parser_stats` is replaced (not accumulated) on each
/// message. This works because `ParserStats` is a cumulative snapshot — each
/// later snapshot includes all work from earlier ones.
fn handle_planner_message(
    message: PlannerWorkerMessage,
    pending_plans: &mut VecDeque<BatchPlan>,
    planner_eof: &mut bool,
    latest_parser_stats: &mut ParserStats,
) -> anyhow::Result<()> {
    match message {
        PlannerWorkerMessage::Plan(plan) => {
            // Stats are cumulative snapshots; replacing is correct because each
            // later snapshot dominates earlier ones.
            *latest_parser_stats = plan.parser_stats;
            pending_plans.push_back(plan);
            Ok(())
        }
        PlannerWorkerMessage::Eof { parser_stats } => {
            // EOF from planner does not imply no pack jobs are left; caller
            // still waits for queued work to drain before sending producer EOF.
            *latest_parser_stats = parser_stats;
            *planner_eof = true;
            Ok(())
        }
        PlannerWorkerMessage::Error(message) => bail!("planner worker failed: {message}"),
    }
}

/// Process a single message from a packer worker.
///
/// Returns `Some(ProducerMessage::Batch{..})` when a batch is ready to be
/// sent to the consumer, or `None` if the message was handled internally.
/// The `in_flight_pack_jobs` counter is decremented here to track how many
/// packer jobs are still outstanding — this is used by the coordinator to
/// know when all work is done.
fn handle_packer_message(
    message: PackerWorkerMessage,
    in_flight_pack_jobs: &mut usize,
    latest_parser_stats: &mut ParserStats,
) -> anyhow::Result<Option<ProducerMessage>> {
    // Every received packer message completes exactly one scheduled pack job.
    *in_flight_pack_jobs = in_flight_pack_jobs.saturating_sub(1);
    match message {
        PackerWorkerMessage::Packed(result) => {
            *latest_parser_stats = result.parser_stats;
            // Producer "build" time is intentionally modeled as plan + pack so
            // final reporting can compare host-side build work vs GPU wait.
            let build_time = result
                .plan_time
                .checked_add(result.pack_time)
                .unwrap_or(result.plan_time + result.pack_time);
            Ok(Some(ProducerMessage::Batch {
                batch: result.batch,
                build_time,
                plan_time: result.plan_time,
                pack_time: result.pack_time,
                parser_stats: result.parser_stats,
            }))
        }
        PackerWorkerMessage::Error { seq_no, message } => {
            bail!("packer worker failed on batch seq {seq_no}: {message}")
        }
    }
}

/// Messages sent from the producer pipeline to the consumer (GPU dispatch thread).
///
/// This enum defines the public protocol between the producer and consumer.
/// The consumer calls `producer.recv()` in a loop and pattern-matches on these
/// variants:
///
/// - **`Batch`** — A filled `WordBatch` ready for GPU dispatch. Includes timing
///   metadata so the consumer can report performance statistics (e.g., "batch
///   build took X ms, GPU hashing took Y ms"). The consumer should call
///   `producer.recycle(batch)` after the GPU finishes to return the allocation.
///
/// - **`Eof`** — End of the wordlist. All candidates have been dispatched.
///   Includes final cumulative parser statistics.
///
/// - **`Error`** — An unrecoverable error occurred in the producer pipeline.
///   The consumer should report the error and shut down.
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
///
/// ## Struct fields — a channel topology
///
/// ```text
///                    ┌──────────────────────────────┐
///                    │   Producer thread (internal)  │
///                    │                               │
///   recycle_tx ─────>│  recycle_rx ──> batch pool    │
///   (consumer side)  │                    │          │
///                    │              fill batches     │
///                    │                    │          │
///                    │                    v          │
///                    │  tx ──────────────────────────┼──> rx (consumer side)
///                    └──────────────────────────────┘
///                              ↑
///                         stop (AtomicBool)
/// ```
///
/// - `rx` — Consumer reads filled batches from here (`Option` so we can `take()` it during shutdown)
/// - `recycle_tx` — Consumer sends empty batches back for reuse
/// - `stop` — Cooperative shutdown flag (shared with the producer thread)
/// - `handle` — `JoinHandle` for the background thread (`Option` so we can `take()` it in `join()`)
pub(crate) struct WordlistProducer {
    rx: Option<Receiver<ProducerMessage>>,
    // Reverse direction channel used only for allocation reuse:
    // consumer -> producer returns fully-owned `WordBatch` values after dispatch.
    recycle_tx: SyncSender<WordBatch>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl WordlistProducer {
    /// Start the producer pipeline immediately so parsing can begin while the
    /// consumer is initializing autotune / first dispatch state.
    ///
    /// ## Bounded channels (`sync_channel`)
    ///
    /// We use `std::sync::mpsc::sync_channel(pipeline_depth)` rather than an
    /// unbounded channel. This is critical for memory control:
    ///
    /// - **Unbounded channel**: Producer could parse the entire 10 GB wordlist
    ///   into memory before the GPU processes a single batch. Memory usage: unbounded.
    /// - **Bounded channel**: Producer blocks when `pipeline_depth` batches are
    ///   already queued, creating natural backpressure. Memory usage: bounded to
    ///   `pipeline_depth * ~67 MiB`.
    ///
    /// The `pipeline_depth` parameter controls the tradeoff:
    /// - Depth 1: minimal memory, but GPU may stall waiting for the next batch
    /// - Depth 4-8: good overlap, the producer stays ahead of the GPU
    /// - Depth 16+: diminishing returns, just wastes memory
    ///
    /// ## Why `thread::spawn` instead of `async`
    ///
    /// The producer does blocking I/O (mmap reads) and CPU-bound work (parsing).
    /// OS threads are the right abstraction here — `async` would just add
    /// complexity with no benefit since we're not doing network I/O or managing
    /// thousands of concurrent tasks.
    pub(crate) fn spawn(
        wordlist_path: PathBuf,
        device: Device,
        parser_config: ParserConfig,
        pipeline_depth: usize,
        packer_threads: usize,
    ) -> Self {
        // CLI validation enforces >0; keep a defensive floor for internal
        // callers/tests so we never accidentally create a zero-capacity queue.
        let pipeline_depth = pipeline_depth.max(1);
        let (tx, rx) = sync_channel(pipeline_depth);
        // Match the forward pipeline depth so we can recycle a small pool of
        // batch allocations without letting memory usage grow unbounded.
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
                packer_threads,
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

    /// Signal the producer to stop after finishing the current batch.
    ///
    /// ## The shutdown protocol (3 steps)
    ///
    /// Shutting down a multi-threaded pipeline cleanly is one of the trickiest
    /// parts of concurrent programming. Here's the protocol:
    ///
    /// 1. **`stop()`** — Sets the `AtomicBool` flag. The producer checks this
    ///    between batches and will stop producing new ones. This is cooperative:
    ///    the producer may finish its current batch before noticing.
    ///
    /// 2. **`close_receiver()`** — Drops the receiving end of the channel.
    ///    This is crucial: if the producer is blocked on `tx.send()` (because
    ///    the bounded channel is full), dropping the receiver unblocks it with
    ///    a `SendError`, which the producer interprets as "consumer is gone,
    ///    time to exit."
    ///
    /// 3. **`join()`** — Waits for the background thread to actually finish.
    ///    This ensures deterministic shutdown and surfaces any thread panics
    ///    as a regular `anyhow::Error`.
    ///
    /// Calling these in the wrong order can cause deadlocks! For example, if
    /// you call `join()` without `close_receiver()`, and the producer is
    /// blocked on a full channel, you'll wait forever.
    // Cooperative stop signal checked between batch builds.
    pub(crate) fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Receive the next batch (or terminal message) from the producer.
    ///
    /// This is a blocking call — the consumer thread will sleep here until
    /// the producer sends a batch. A closed channel is treated as an error
    /// because the consumer expects a clean `ProducerMessage::Eof`.
    // Receive the next batch (or terminal message) from the producer. A closed
    // channel is treated as an error because the consumer expects a clean EOF.
    pub(crate) fn recv(&self) -> anyhow::Result<ProducerMessage> {
        let rx = self
            .rx
            .as_ref()
            .ok_or_else(|| anyhow!("wordlist producer receiver already closed"))?;
        rx.recv()
            .map_err(|_| anyhow!("wordlist producer terminated unexpectedly"))
    }

    /// Drop the receiving end of the channel to unblock a potentially stuck producer.
    ///
    /// `rx` is wrapped in `Option` specifically so we can `take()` it here.
    /// Dropping the `Receiver` causes any `tx.send()` call on the producer side
    /// to return `Err(SendError)`, which the producer interprets as a signal
    /// to exit its loop.
    // Dropping the receiver is an important shutdown step when exiting early on
    // a match: it prevents the producer from blocking forever on a bounded send.
    pub(crate) fn close_receiver(&mut self) {
        let _ = self.rx.take();
    }

    /// Return a used batch to the producer for recycling (best-effort).
    ///
    /// ## Why `try_send` instead of `send`?
    ///
    /// `send()` would block if the recycle channel is full. That would stall
    /// the GPU dispatch thread — the most latency-sensitive thread in the
    /// pipeline. `try_send()` returns immediately; if the channel is full,
    /// the batch is simply dropped (its Metal buffers are freed). The producer
    /// will allocate a new one when needed.
    ///
    /// In practice, the recycle channel is rarely full because the producer
    /// consumes recycled batches as fast as the consumer returns them. But
    /// under bursty workloads, this non-blocking behavior prevents a
    /// priority inversion where the GPU stalls waiting on the CPU.
    // Best-effort recycle path: do not block the consumer/GPU path if the
    // producer is busy or already shutting down.
    pub(crate) fn recycle(&self, batch: WordBatch) {
        // `try_send` is intentional: the consumer is on the critical path for
        // throughput, so we prefer dropping a recyclable allocation over
        // stalling GPU dispatch progress.
        let _ = self.recycle_tx.try_send(batch);
    }

    /// Wait for the background thread to finish and propagate any panics.
    ///
    /// `JoinHandle::join()` returns `Err` if the thread panicked. We convert
    /// this to an `anyhow::Error` so the caller gets a clean error message
    /// instead of an unwinding panic propagating to the main thread.
    ///
    /// The `handle` is wrapped in `Option` so we can `take()` it — Rust
    /// requires ownership of the `JoinHandle` to call `join()`, but we only
    /// have `&mut self`. `take()` moves the handle out of `self` and replaces
    /// it with `None`, transferring ownership.
    // Join the background thread so shutdown is deterministic and panics are
    // surfaced as regular errors.
    pub(crate) fn join(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("wordlist producer thread panicked"))?;
        }
        Ok(())
    }
}

/// Producer-thread main loop: orchestrates planning, packing, and batch emission.
///
/// This is the most complex function in the module. It sets up the internal
/// three-stage pipeline (planner -> packers -> coordinator) and runs the
/// coordinator loop that ties everything together.
///
/// ## Memory strategy: preallocate, then recycle
///
/// We allocate `pipeline_depth` `WordBatch` objects up front. This means:
/// - The initial pipeline fill is fast (no allocation in the hot loop)
/// - Steady-state memory usage is predictable and bounded
/// - If recycling fails occasionally (consumer too slow), we fall back to
///   allocating a fresh batch, but this should be rare
///
/// ## Channel topology inside this function
///
/// ```text
///  planner_thread ──[PlannerWorkerMessage]──> coordinator
///                                                │
///                        ┌───────────────────────┤
///                        │                       │
///                        v                       v
///  packer_thread_1 <──[PackerJob]──> coordinator ──[ProducerMessage]──> consumer
///  packer_thread_2 <──┘              ^
///                                    │
///              [PackerWorkerMessage]──┘
/// ```
// Producer-thread main loop: parse batches and send them to the consumer until
// EOF, stop requested, or the receiver is dropped.
fn run_wordlist_producer(
    wordlist_path: PathBuf,
    device: Device,
    parser_config: ParserConfig,
    pipeline_depth: usize,
    packer_threads: usize,
    stop: Arc<AtomicBool>,
    tx: SyncSender<ProducerMessage>,
    // Optional returned batches from the consumer. The producer owns the next
    // parse/build step, so this is where allocation reuse naturally fits.
    recycle_rx: Receiver<WordBatch>,
) -> anyhow::Result<()> {
    let pipeline_depth = pipeline_depth.max(1);
    let packer_threads = packer_threads.max(1).min(pipeline_depth);
    // `pipeline_depth` is both throughput overlap target and memory cap for
    // owned `WordBatch` allocations.
    // Preallocate the steady-state batch pool up front so the initial pipeline
    // fill does not pay large Metal buffer allocation costs on the hot path.
    let mut preallocated_batches = Vec::with_capacity(pipeline_depth);
    for _ in 0..pipeline_depth {
        preallocated_batches.push(WordBatch::new(&device, 0));
    }
    let batch_alloc_device = device.clone();
    let reader = AnyWordlistBatchReader::new(device, &wordlist_path, parser_config)?;
    let shared_mmap = reader.shared_mmap();

    let (plan_tx, plan_rx) = crossbeam_channel::bounded::<PlannerWorkerMessage>(pipeline_depth);
    let stop_for_planner = Arc::clone(&stop);
    let planner_handle =
        thread::spawn(move || planner_worker_main(reader, stop_for_planner, plan_tx));

    let (pack_job_tx, pack_job_rx) = crossbeam_channel::bounded::<PackerJob>(pipeline_depth);
    let (pack_result_tx, pack_result_rx) =
        crossbeam_channel::bounded::<PackerWorkerMessage>(pipeline_depth);
    let mut packer_handles = Vec::with_capacity(packer_threads);
    for _ in 0..packer_threads {
        let mmap_for_worker = Arc::clone(&shared_mmap);
        let job_rx_for_worker = pack_job_rx.clone();
        let result_tx_for_worker = pack_result_tx.clone();
        packer_handles.push(thread::spawn(move || {
            packer_worker_main(mmap_for_worker, job_rx_for_worker, result_tx_for_worker)
        }));
    }
    drop(pack_job_rx);
    drop(pack_result_tx);

    // The coordinator loop is wrapped in a closure so we can use `?` for
    // error propagation while still guaranteeing cleanup (channel drops and
    // thread joins) runs afterward regardless of success or failure.
    //
    // This is a common Rust pattern: wrap fallible logic in a closure that
    // returns `Result`, call it, then do cleanup unconditionally after.
    let coordinator_result = (|| -> anyhow::Result<()> {
        // Coordinator owns all mutable pipeline state so workers can stay
        // "pure" (parse or pack) and communicate only via channels.
        let mut available_batches = preallocated_batches;
        let mut pending_plans = VecDeque::<BatchPlan>::new();
        let mut in_flight_pack_jobs = 0usize;
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

            // Pull planner results without blocking so planning can progress
            // independently from packer throughput and emission cadence.
            loop {
                match plan_rx.try_recv() {
                    Ok(message) => {
                        // Planner can run ahead of packers; stash plans in FIFO
                        // order so pack scheduling remains deterministic.
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

            // Harvest completed pack jobs without blocking.
            loop {
                match pack_result_rx.try_recv() {
                    Ok(message) => {
                        if let Some(producer_message) = handle_packer_message(
                            message,
                            &mut in_flight_pack_jobs,
                            &mut latest_parser_stats,
                        )? {
                            // If consumer is gone (early match/exit), producer
                            // stops without treating it as an error.
                            if tx.send(producer_message).is_err() {
                                return Ok(());
                            }
                            made_progress = true;
                        }
                    }
                    Err(crossbeam_channel::TryRecvError::Empty) => break,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        if in_flight_pack_jobs > 0 {
                            bail!("packer workers terminated unexpectedly");
                        }
                        break;
                    }
                }
            }

            // Keep packers busy by dispatching queued plans.
            while in_flight_pack_jobs < pipeline_depth {
                let Some(plan) = pending_plans.pop_front() else {
                    break;
                };
                let batch = available_batches
                    .pop()
                    // Recycling is best-effort; if the recycle queue is empty,
                    // allocate a fresh batch so packers keep flowing.
                    .unwrap_or_else(|| WordBatch::new(&batch_alloc_device, 0));
                pack_job_tx
                    .send(PackerJob { plan, batch })
                    .map_err(|_| anyhow!("packer worker job queue closed unexpectedly"))?;
                in_flight_pack_jobs = in_flight_pack_jobs.saturating_add(1);
                made_progress = true;
            }

            if planner_eof && pending_plans.is_empty() && in_flight_pack_jobs == 0 {
                // EOF is emitted exactly once only after all queued plans and
                // in-flight pack jobs are fully drained.
                let _ = tx.send(ProducerMessage::Eof {
                    parser_stats: latest_parser_stats,
                });
                return Ok(());
            }

            if made_progress {
                continue;
            }

            // No immediate progress: block on the stream that can unblock the
            // pipeline next (packer completions first, otherwise new plans).
            if in_flight_pack_jobs > 0 {
                let message = pack_result_rx
                    .recv()
                    .map_err(|_| anyhow!("packer workers terminated unexpectedly"))?;
                if let Some(producer_message) = handle_packer_message(
                    message,
                    &mut in_flight_pack_jobs,
                    &mut latest_parser_stats,
                )? {
                    if tx.send(producer_message).is_err() {
                        return Ok(());
                    }
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

    // --- Cleanup: channel drops + thread joins ---
    //
    // This section runs regardless of whether the coordinator loop succeeded
    // or failed. The order matters:
    //
    // 1. Drop the SENDER sides of channels first. This causes workers' `recv()`
    //    calls to return `Err(RecvError)`, which they interpret as "time to exit."
    //    If we joined threads first without dropping channels, workers blocked on
    //    `recv()` would never wake up → deadlock.
    //
    // 2. Join all threads to ensure they've fully exited. This surfaces panics
    //    and ensures no dangling threads outlive this function.
    //
    // 3. Combine coordinator errors with join errors. The coordinator error
    //    takes priority (it's the root cause), but if the coordinator succeeded
    //    and a thread panicked, we still report that.
    // Closing sender sides first guarantees worker `recv()` loops can observe
    // channel closure and exit before we join them.
    drop(pack_job_tx);
    drop(pack_result_rx);
    drop(plan_rx);

    let mut join_error: Option<anyhow::Error> = None;
    if planner_handle.join().is_err() && join_error.is_none() {
        join_error = Some(anyhow!("planner worker thread panicked"));
    }
    while let Some(handle) = packer_handles.pop() {
        if handle.join().is_err() && join_error.is_none() {
            join_error = Some(anyhow!("packer worker thread panicked"));
        }
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
            2,
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
    fn wordlist_producer_parallel_path_emits_batches_with_multiple_packer_workers() {
        let path = write_temp_wordlist(b"alpha\nbeta\ngamma\ndelta\n");
        let mut producer =
            WordlistProducer::spawn(path.clone(), test_device(), test_parser_config(2, 8), 4, 2);

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
