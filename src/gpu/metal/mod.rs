use anyhow::{Context, anyhow, bail};
use metal::{CompileOptions, MTLSize};
use std::time::{Duration, Instant};

use crate::batch::{DispatchBatchView, WordBatch};
use crate::stats::{BatchDispatchTimings, format_human_count};

use super::backend::{GpuBruteForcer, HmacVariant};
use super::{GpuCommandHandle, GpuDevice, default_device};

// ---------------------------------------------------------------------------
// Embedded shader sources
// ---------------------------------------------------------------------------

const METAL_SOURCE_HS256: &str = include_str!("hs256_wordlist.metal");

const METAL_SOURCE_HS512: &str = include_str!("hs512_wordlist.metal");

// ---------------------------------------------------------------------------
// Sentinel value for "no match in this batch"
// ---------------------------------------------------------------------------

const RESULT_NOT_FOUND_SENTINEL: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Metal buffer binding indices — match [[buffer(N)]] slots in the .metal shaders.
// ---------------------------------------------------------------------------

const BUF_PARAMS: u64 = 0;
const BUF_MESSAGE: u64 = 1;
const BUF_WORD_BYTES: u64 = 2;
const BUF_WORD_OFFSETS: u64 = 3;
const BUF_WORD_LENGTHS: u64 = 4;
const BUF_RESULT: u64 = 5;

// ---------------------------------------------------------------------------
// Host -> GPU parameter blocks (#[repr(C)] for Metal struct layout parity)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs256BruteForceParams {
    target_signature: [u32; 8],
    message_length: u32,
    candidate_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs512BruteForceParams {
    target_signature: [u64; 8],
    message_length: u32,
    candidate_count: u32,
}

// ---------------------------------------------------------------------------
// Shared buffer-write helpers
// ---------------------------------------------------------------------------

fn copy_bytes_to_buffer(buffer: &metal::Buffer, data: &[u8]) {
    debug_assert!(buffer.length() as usize >= data.len());
    if data.is_empty() {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast::<u8>(), data.len());
    }
}

fn copy_value_to_buffer<T>(buffer: &metal::Buffer, value: &T) {
    copy_bytes_to_buffer(buffer, bytes_of(value));
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

// ---------------------------------------------------------------------------
// MetalBruteForcer — consolidated GPU backend for all HMAC variants
// ---------------------------------------------------------------------------

pub(crate) struct MetalBruteForcer {
    variant: HmacVariant,
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

impl MetalBruteForcer {
    pub(crate) fn new(variant: HmacVariant, signing_input: &[u8]) -> anyhow::Result<Self> {
        let device = default_device()?;
        let compile_options = CompileOptions::new();

        let (source, fn_mixed, fn_short) = match variant {
            HmacVariant::Hs256 => (
                METAL_SOURCE_HS256,
                "hs256_wordlist",
                "hs256_wordlist_short_keys",
            ),
            HmacVariant::Hs384 => (
                METAL_SOURCE_HS512,
                "hs384_wordlist",
                "hs384_wordlist_short_keys",
            ),
            HmacVariant::Hs512 => (
                METAL_SOURCE_HS512,
                "hs512_wordlist",
                "hs512_wordlist_short_keys",
            ),
        };

        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| anyhow!("failed to compile Metal kernel: {e}"))?;

        let mixed_function = library
            .get_function(fn_mixed, None)
            .map_err(|e| anyhow!("failed to get Metal function {fn_mixed}: {e}"))?;
        let short_function = library
            .get_function(fn_short, None)
            .map_err(|e| anyhow!("failed to get Metal function {fn_short}: {e}"))?;

        let pipeline_mixed = device
            .new_compute_pipeline_state_with_function(&mixed_function)
            .map_err(|e| anyhow!("failed to create mixed compute pipeline: {e}"))?;
        let pipeline_short_keys = device
            .new_compute_pipeline_state_with_function(&short_function)
            .map_err(|e| anyhow!("failed to create short-key compute pipeline: {e}"))?;

        let queue = device.new_command_queue();
        let max_threads = (pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(pipeline_short_keys.max_total_threads_per_threadgroup() as usize);
        let threadgroup_width = 256usize.min(max_threads.max(1));

        let message_length =
            u32::try_from(signing_input.len()).context("JWT signing input too long")?;

        let options = metal::MTLResourceOptions::StorageModeShared;

        let params_size = match variant {
            HmacVariant::Hs256 => std::mem::size_of::<Hs256BruteForceParams>(),
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                std::mem::size_of::<Hs512BruteForceParams>()
            }
        };
        let params_buf = device.new_buffer(params_size as u64, options);
        let msg_buf = device.new_buffer(signing_input.len().max(1) as u64, options);
        copy_bytes_to_buffer(&msg_buf, signing_input);

        let result_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, options);
        copy_value_to_buffer(&result_buf, &RESULT_NOT_FOUND_SENTINEL);

        Ok(Self {
            variant,
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

    // -----------------------------------------------------------------------
    // Pipeline selection
    // -----------------------------------------------------------------------

    fn short_key_threshold(&self) -> u16 {
        match self.variant {
            HmacVariant::Hs256 => 64,
            HmacVariant::Hs384 | HmacVariant::Hs512 => 128,
        }
    }

    fn active_pipeline_for_view(
        &self,
        batch: DispatchBatchView<'_>,
    ) -> &metal::ComputePipelineState {
        if batch.max_word_len <= self.short_key_threshold() {
            &self.pipeline_short_keys
        } else {
            &self.pipeline_mixed
        }
    }

    // -----------------------------------------------------------------------
    // Params preparation (variant-specific signature conversion)
    // -----------------------------------------------------------------------

    fn write_params(&self, target_signature: &[u8], candidate_count: u32) -> anyhow::Result<()> {
        let expected_len = self.variant.signature_len();
        if target_signature.len() != expected_len {
            bail!(
                "{} signature must be {} bytes, got {}",
                self.variant.label(),
                expected_len,
                target_signature.len()
            );
        }

        match self.variant {
            HmacVariant::Hs256 => {
                let mut target_words = [0u32; 8];
                for i in 0..8 {
                    let off = i * 4;
                    target_words[i] =
                        u32::from_be_bytes(target_signature[off..off + 4].try_into().unwrap());
                }
                let params = Hs256BruteForceParams {
                    target_signature: target_words,
                    message_length: self.message_length,
                    candidate_count,
                };
                copy_value_to_buffer(&self.params_buf, &params);
            }
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                let word_count = expected_len / 8;
                let mut target_words = [0u64; 8];
                for i in 0..word_count {
                    let off = i * 8;
                    target_words[i] =
                        u64::from_be_bytes(target_signature[off..off + 8].try_into().unwrap());
                }
                let params = Hs512BruteForceParams {
                    target_signature: target_words,
                    message_length: self.message_length,
                    candidate_count,
                };
                copy_value_to_buffer(&self.params_buf, &params);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Core dispatch helpers
    // -----------------------------------------------------------------------

    fn encode_and_commit_view(
        &self,
        target_signature: &[u8],
        batch: DispatchBatchView<'_>,
        threadgroup_width: usize,
    ) -> anyhow::Result<(metal::CommandBuffer, Duration, Duration)> {
        let candidate_count =
            u32::try_from(batch.candidate_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            bail!("cannot encode empty batch for async dispatch");
        }

        let prep_started = Instant::now();
        self.write_params(target_signature, candidate_count)?;
        copy_value_to_buffer(&self.result_buf, &RESULT_NOT_FOUND_SENTINEL);
        let host_prep = prep_started.elapsed();

        let encode_started = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = self.active_pipeline_for_view(batch);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(BUF_PARAMS, Some(&self.params_buf), 0);
        encoder.set_buffer(BUF_MESSAGE, Some(&self.msg_buf), 0);
        encoder.set_buffer(BUF_WORD_BYTES, Some(batch.word_bytes_buf), 0);
        encoder.set_buffer(BUF_WORD_OFFSETS, Some(batch.word_offsets_buf), 0);
        encoder.set_buffer(BUF_WORD_LENGTHS, Some(batch.word_lengths_buf), 0);
        encoder.set_buffer(BUF_RESULT, Some(&self.result_buf), 0);

        let threads_per_group = MTLSize::new(threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(candidate_count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
        let command_encode = encode_started.elapsed();

        command_buffer.commit();

        let owned = command_buffer.to_owned();
        Ok((owned, host_prep, command_encode))
    }

    fn wait_and_readback_impl(
        &self,
        cmd_buf: &metal::CommandBufferRef,
    ) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        cmd_buf.wait_until_completed();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let result_ptr = self.result_buf.contents().cast::<u32>();
        // SAFETY: result_buf is a 4-byte shared buffer; GPU has completed.
        let result = unsafe { *result_ptr };
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    fn dispatch_batch_view(
        &self,
        target_signature: &[u8],
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

        let (cmd_buf, host_prep, command_encode) =
            self.encode_and_commit_view(target_signature, batch, threadgroup_width)?;
        timings.host_prep = host_prep;
        timings.command_encode = command_encode;

        let (maybe_match, gpu_wait, result_readback) = self.wait_and_readback_impl(&cmd_buf);
        timings.gpu_wait = gpu_wait;
        timings.result_readback = result_readback;
        timings.total = started.elapsed();
        Ok((maybe_match, timings))
    }
}

// ---------------------------------------------------------------------------
// GpuBruteForcer trait implementation
// ---------------------------------------------------------------------------

impl GpuBruteForcer for MetalBruteForcer {
    fn device(&self) -> &GpuDevice {
        &self.device
    }

    fn device_name(&self) -> &str {
        self.device.name()
    }

    fn thread_execution_width(&self) -> usize {
        (self.pipeline_mixed.thread_execution_width() as usize)
            .min(self.pipeline_short_keys.thread_execution_width() as usize)
    }

    fn max_total_threads_per_threadgroup(&self) -> usize {
        (self.pipeline_mixed.max_total_threads_per_threadgroup() as usize)
            .min(self.pipeline_short_keys.max_total_threads_per_threadgroup() as usize)
    }

    fn current_threadgroup_width(&self) -> usize {
        self.threadgroup_width
    }

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

    fn encode_and_commit(
        &self,
        target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)> {
        self.encode_and_commit_view(
            target_signature,
            batch.as_dispatch_view(),
            self.threadgroup_width,
        )
    }

    fn wait_and_readback(
        &self,
        handle: &GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration) {
        self.wait_and_readback_impl(handle)
    }

    fn autotune_threadgroup_width(
        &mut self,
        target_signature: &[u8],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hs256_params_round_trip_size_matches() {
        let params = Hs256BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs256BruteForceParams>());
    }

    #[test]
    fn hs512_params_round_trip_size_matches() {
        let params = Hs512BruteForceParams::default();
        let bytes = bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<Hs512BruteForceParams>());
    }
}
