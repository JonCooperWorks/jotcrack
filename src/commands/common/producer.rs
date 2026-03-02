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

struct PackerJob {
    plan: BatchPlan,
    batch: WordBatch,
}

#[derive(Debug)]
struct PackerResult {
    batch: WordBatch,
    plan_time: Duration,
    pack_time: Duration,
    parser_stats: ParserStats,
}

#[derive(Debug)]
enum PackerWorkerMessage {
    Packed(PackerResult),
    Error { seq_no: u64, message: String },
}

#[derive(Debug)]
enum PlannerWorkerMessage {
    Plan(BatchPlan),
    Eof { parser_stats: ParserStats },
    Error(String),
}

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

// Owns the producer thread and channel endpoints used to overlap wordlist
// parsing with GPU dispatch.
pub(crate) struct WordlistProducer {
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

    // Cooperative stop signal checked between batch builds.
    pub(crate) fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

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

    // Dropping the receiver is an important shutdown step when exiting early on
    // a match: it prevents the producer from blocking forever on a bounded send.
    pub(crate) fn close_receiver(&mut self) {
        let _ = self.rx.take();
    }

    // Best-effort recycle path: do not block the consumer/GPU path if the
    // producer is busy or already shutting down.
    pub(crate) fn recycle(&self, batch: WordBatch) {
        // `try_send` is intentional: the consumer is on the critical path for
        // throughput, so we prefer dropping a recyclable allocation over
        // stalling GPU dispatch progress.
        let _ = self.recycle_tx.try_send(batch);
    }

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
