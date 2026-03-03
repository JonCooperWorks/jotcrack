use std::time::Duration;

use crate::common::batch::WordBatch;

use super::{GpuCommandHandle, GpuDevice};

/// Which HMAC algorithm variant to crack.
///
/// Determines the Metal/CUDA kernel entry points, the signature byte length,
/// the short-key threshold, and the params struct layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HmacVariant {
    Hs256,
    Hs384,
    Hs512,
}

impl HmacVariant {
    /// Expected byte length of the HMAC signature for this variant.
    pub(crate) fn signature_len(self) -> usize {
        match self {
            Self::Hs256 => 32,
            Self::Hs384 => 48,
            Self::Hs512 => 64,
        }
    }

    /// Human-readable label for result output (e.g. "HS256 key:").
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Hs256 => "HS256",
            Self::Hs384 => "HS384",
            Self::Hs512 => "HS512",
        }
    }

    /// The JWT `alg` header string for this variant.
    pub(crate) fn alg_string(self) -> &'static str {
        match self {
            Self::Hs256 => "HS256",
            Self::Hs384 => "HS384",
            Self::Hs512 => "HS512",
        }
    }
}

/// Platform-agnostic interface for GPU brute-force dispatch.
///
/// Implementors handle device initialization, shader compilation, pipeline
/// state creation, and the encode -> commit -> wait lifecycle. The trait
/// uses `&[u8]` for the target signature so a single interface covers all
/// HMAC variants; implementations validate the slice length.
pub(crate) trait GpuBruteForcer {
    /// The underlying GPU device handle (for buffer allocation by the producer).
    fn device(&self) -> &GpuDevice;

    /// Human-readable GPU device name (e.g. "Apple M4 Max").
    fn device_name(&self) -> &str;

    /// SIMD lane count (Metal thread execution width / CUDA warp size).
    fn thread_execution_width(&self) -> usize;

    /// Hardware maximum threads per threadgroup / block.
    fn max_total_threads_per_threadgroup(&self) -> usize;

    /// Current threadgroup width setting.
    fn current_threadgroup_width(&self) -> usize;

    /// Validate and apply a threadgroup width override.
    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()>;

    /// Encode GPU commands for a batch and submit asynchronously.
    ///
    /// `target_signature` must be exactly `variant.signature_len()` bytes.
    /// Returns the opaque command handle plus (host_prep, encode) durations.
    fn encode_and_commit(
        &self,
        target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)>;

    /// Block until the given command handle completes, then read the result.
    ///
    /// Returns `Some(batch_local_index)` on match, `None` if no match.
    /// Also returns (gpu_wait, readback) durations.
    fn wait_and_readback(
        &self,
        handle: &GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration);

    /// Benchmark threadgroup widths on a sample and keep the fastest.
    fn autotune_threadgroup_width(
        &mut self,
        target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<()>;
}
