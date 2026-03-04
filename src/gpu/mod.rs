use std::time::Duration;

use crate::batch::WordBatch;

#[cfg(target_os = "macos")]
pub(crate) mod metal;

#[cfg(target_os = "linux")]
mod cuda;

// ---------------------------------------------------------------------------
// HmacVariant
// ---------------------------------------------------------------------------

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

    /// Human-readable algorithm name (e.g. "HS256").
    /// Used for output labels, JWT `alg` header matching, and error messages.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Hs256 => "HS256",
            Self::Hs384 => "HS384",
            Self::Hs512 => "HS512",
        }
    }
}

// ---------------------------------------------------------------------------
// CrackVariant
// ---------------------------------------------------------------------------

/// Top-level dispatch type covering both JWT (HMAC) and JWE (Key Wrap) attacks.
///
/// `HmacVariant` continues to drive GPU kernel selection for HMAC attacks.
/// `CrackVariant` sits above it as the unified discriminant used by the
/// runner and CLI layers to route tokens to the correct compute backend:
///
/// - **`Hmac(hv)`** → Metal/CUDA GPU shader (HMAC-SHA per candidate)
/// - **`JweA128kw`** → Metal/CUDA GPU shader (software AES-128 Key Wrap per candidate)
///
/// Both paths run on the GPU. AES Key Wrap uses software S-box lookup tables
/// in constant memory — the GPU's massive parallelism (thousands of concurrent
/// threads) outweighs the per-thread cost of software AES vs hardware AESD,
/// benchmarking at ~120M/s GPU vs ~5.3M/s CPU (22× faster on Apple M4 Max).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CrackVariant {
    /// HMAC-SHA JWT cracking via GPU compute.
    Hmac(HmacVariant),
    /// JWE A128KW (AES-128 Key Wrap, RFC 3394) cracking via GPU compute.
    JweA128kw,
}

impl CrackVariant {
    /// Human-readable algorithm label for output (e.g. "HS256", "A128KW").
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Hmac(hv) => hv.label(),
            Self::JweA128kw => "A128KW",
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBruteForcer trait
// ---------------------------------------------------------------------------

/// Platform-agnostic interface for GPU brute-force dispatch.
///
/// Implementors handle device initialization, shader compilation, pipeline
/// state creation, and the encode → commit → wait lifecycle. The trait
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

// ---------------------------------------------------------------------------
// Platform type aliases
//
// Shared code (batch.rs, producer.rs, parser.rs) uses these aliases instead
// of importing Metal or CUDA types directly. This keeps platform-specific
// concerns confined to this module and the backend implementations.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub(crate) type GpuBuffer = ::metal::Buffer;

#[cfg(target_os = "macos")]
pub(crate) type GpuDevice = ::metal::Device;

#[cfg(target_os = "macos")]
pub(crate) type GpuCommandHandle = ::metal::CommandBuffer;

#[cfg(target_os = "linux")]
compile_error!("CUDA backend not yet implemented — see src/gpu/cuda.rs");

// ---------------------------------------------------------------------------
// Platform-gated buffer helpers
//
// These thin wrappers keep Metal API calls out of batch.rs / producer.rs so
// those files compile unchanged on any supported platform.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub(crate) fn alloc_shared_buffer(device: &GpuDevice, size: usize) -> GpuBuffer {
    device.new_buffer(size as u64, ::metal::MTLResourceOptions::StorageModeShared)
}

#[cfg(target_os = "macos")]
pub(crate) fn buffer_host_ptr(buf: &GpuBuffer) -> *mut u8 {
    buf.contents().cast::<u8>()
}

#[cfg(target_os = "macos")]
pub(crate) fn default_device() -> anyhow::Result<GpuDevice> {
    ::metal::Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device available"))
}
