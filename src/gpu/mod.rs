use std::time::Duration;

use crate::batch::WordBatch;

#[cfg(target_os = "macos")]
pub(crate) mod metal;

#[cfg(target_os = "linux")]
pub(crate) mod cuda;

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
// AesKwVariant
// ---------------------------------------------------------------------------

/// Which AES Key Wrap variant to crack (RFC 7518 §4.4).
///
/// All three variants use RFC 3394 AES Key Wrap with different AES key sizes.
/// The underlying algorithm is identical — only the AES block cipher changes:
///
/// | Variant | AES Key | Rounds | Round Keys | FIPS 197 §5.2 |
/// |---------|---------|--------|------------|---------------|
/// | A128KW  | 16 bytes (Nk=4) | 10 (Nr=10) | 44 words | Standard |
/// | A192KW  | 24 bytes (Nk=6) | 12 (Nr=12) | 52 words | Standard |
/// | A256KW  | 32 bytes (Nk=8) | 14 (Nr=14) | 60 words | Extra SubWord at i%8==4 |
///
/// The Metal shader is compiled three times with different `AES_KEY_BYTES`
/// preprocessor defines, producing fully specialised kernels for each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AesKwVariant {
    /// AES-128 Key Wrap (16-byte key, 10 rounds).
    A128kw,
    /// AES-192 Key Wrap (24-byte key, 12 rounds).
    A192kw,
    /// AES-256 Key Wrap (32-byte key, 14 rounds).
    A256kw,
}

impl AesKwVariant {
    /// Human-readable algorithm name matching the JWE `alg` header value.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::A128kw => "A128KW",
            Self::A192kw => "A192KW",
            Self::A256kw => "A256KW",
        }
    }

    /// AES key size in bytes for this variant.
    ///
    /// This determines the key expansion schedule, round count, and the
    /// short-key threshold (candidates ≤ key_bytes are zero-padded;
    /// longer candidates are SHA-256 hashed and truncated).
    pub(crate) fn key_bytes(self) -> usize {
        match self {
            Self::A128kw => 16,
            Self::A192kw => 24,
            Self::A256kw => 32,
        }
    }
}

// ---------------------------------------------------------------------------
// CrackVariant
// ---------------------------------------------------------------------------

/// Top-level dispatch type covering both JWT (HMAC) and JWE (Key Wrap) attacks.
///
/// `HmacVariant` drives GPU kernel selection for HMAC attacks.
/// `AesKwVariant` drives GPU kernel selection for AES Key Wrap attacks.
/// `CrackVariant` sits above them as the unified discriminant used by the
/// runner and CLI layers to route tokens to the correct compute backend:
///
/// - **`Hmac(hv)`** → Metal/CUDA GPU shader (HMAC-SHA per candidate)
/// - **`JweAesKw(akv)`** → Metal/CUDA GPU shader (software AES Key Wrap per candidate)
///
/// Both paths run on the GPU. AES Key Wrap uses software S-box lookup tables
/// in constant memory — the GPU's massive parallelism (thousands of concurrent
/// threads) outweighs the per-thread cost of software AES vs hardware AESD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CrackVariant {
    /// HMAC-SHA JWT cracking via GPU compute.
    Hmac(HmacVariant),
    /// JWE AES Key Wrap (RFC 3394) cracking via GPU compute.
    /// Supports A128KW, A192KW, and A256KW.
    JweAesKw(AesKwVariant),
}

impl CrackVariant {
    /// Human-readable algorithm label for output (e.g. "HS256", "A128KW").
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Hmac(hv) => hv.label(),
            Self::JweAesKw(akv) => akv.label(),
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
pub(crate) type GpuBuffer = cuda::CudaBuffer;

#[cfg(target_os = "linux")]
pub(crate) type GpuDevice = cuda::CudaDeviceHandle;

#[cfg(target_os = "linux")]
pub(crate) type GpuCommandHandle = cuda::CudaCommandHandle;

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

#[cfg(target_os = "linux")]
pub(crate) fn alloc_shared_buffer(_device: &GpuDevice, size: usize) -> GpuBuffer {
    cuda::alloc_shared_buffer(_device, size)
}

#[cfg(target_os = "linux")]
pub(crate) fn buffer_host_ptr(buf: &GpuBuffer) -> *mut u8 {
    cuda::buffer_host_ptr(buf)
}

#[cfg(target_os = "linux")]
#[allow(dead_code)] // Available for tests and future direct usage
pub(crate) fn default_device() -> anyhow::Result<GpuDevice> {
    cuda::default_device()
}
