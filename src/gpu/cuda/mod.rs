//! # CUDA GPU backend for brute-force dispatch
//!
//! This module implements the Linux CUDA compute backend. It compiles
//! embedded `.cu` kernel sources at runtime via NVRTC (NVIDIA Runtime
//! Compilation), manages device memory allocations, and drives the
//! copy → launch → sync → readback lifecycle for each batch of
//! password candidates.
//!
//! ## Two compute backends
//!
//! - **`CudaBruteForcer`** — HMAC-SHA JWT cracking (HS256/384/512).
//!   Uses GPU-optimised software SHA implementations.
//!
//! - **`CudaAesKwBruteForcer`** — JWE AES Key Wrap cracking (A128KW,
//!   A192KW, A256KW) via software AES in a CUDA compute kernel.
//!   The kernel source is compiled three times with different
//!   `AES_KEY_BYTES` defines, producing specialised kernels for each
//!   key size.
//!
//! Both implement the `GpuBruteForcer` trait so the runner can use
//! a single generic dispatch loop.
//!
//! ## Two-kernel strategy
//!
//! Each backend exposes two kernel entry points:
//!
//! - **`mixed`** — handles keys of any length.
//! - **`short_keys`** — optimised fast path for short keys.
//!
//! ## Memory model: explicit host→device transfers
//!
//! Unlike Metal's unified memory (`StorageModeShared`), CUDA on
//! discrete GPUs requires explicit host→device copies across PCIe.
//! Device buffers are pre-allocated to maximum batch sizes and reused
//! across dispatches. Host data (from `CudaBuffer` / `Vec<u8>`) is
//! copied to device before each kernel launch via `memcpy_htod`.
//!
//! ## Interior mutability via `UnsafeCell`
//!
//! The `GpuBruteForcer` trait's `encode_and_commit` method takes
//! `&self`, but CUDA dispatch requires mutable access to device buffers
//! (for `memcpy_htod`). We use `UnsafeCell` for these buffers.
//!
//! Safety invariant: the runner's double-buffer pattern guarantees
//! that only one `encode_and_commit` → `wait_and_readback` cycle is
//! in flight at a time, and `wait_and_readback` synchronises the
//! stream before buffers can be reused by the next dispatch.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{self, CompileOptions};

use crate::batch::{MAX_CANDIDATES_PER_BATCH, WordBatch};
use crate::stats::{BatchDispatchTimings, format_human_count};

use super::{AesKwVariant, GpuBruteForcer, GpuMarkovBruteForcer, HmacVariant};

// ---------------------------------------------------------------------------
// Embedded kernel sources
//
// `include_str!` embeds the `.cu` source as a `&'static str` in the
// binary at compile time. NVRTC compiles it to PTX on first use,
// which adds ~200–500ms to startup but avoids shipping pre-compiled
// binaries and keeps the deployment self-contained.
//
// HS384 reuses the HS512 kernel source — the same file contains
// entry points for both (hs384_wordlist, hs512_wordlist) since
// SHA-384 is a truncated SHA-512 with different initial hash values.
// ---------------------------------------------------------------------------

const CUDA_SOURCE_HS256: &str = include_str!("hs256_wordlist.cu");

const CUDA_SOURCE_HS512: &str = include_str!("hs512_wordlist.cu");

const CUDA_SOURCE_AESKW: &str = include_str!("aeskw_wordlist.cu");

// ---------------------------------------------------------------------------
// Sentinel value for "no match in this batch"
//
// The GPU kernel writes the batch-local index of a matching candidate
// into the result buffer via `atomicMin`. If no candidate matches,
// the buffer retains this sentinel value. `u32::MAX` is safe because
// the maximum batch size (~6.2M) is well below 2^32.
// ---------------------------------------------------------------------------

const RESULT_NOT_FOUND_SENTINEL: u32 = u32::MAX;

/// NVIDIA warp size — always 32 threads on all NVIDIA architectures.
/// Analogous to Metal's "thread execution width" (SIMD lane count).
const WARP_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Host → GPU parameter blocks
//
// These structs are copied byte-for-byte to device memory and read
// by the GPU kernel via a pointer. `#[repr(C)]` guarantees a
// deterministic field layout matching the CUDA struct definition.
//
// `unsafe impl DeviceRepr` tells cudarc these types can be safely
// represented in GPU memory. This is sound because they are `Copy`,
// contain only primitive numeric fields, and have no padding on
// standard architectures (all fields are naturally aligned).
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs256BruteForceParams {
    target_signature: [u32; 8],
    message_length: u32,
    candidate_count: u32,
}
unsafe impl DeviceRepr for Hs256BruteForceParams {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs512BruteForceParams {
    target_signature: [u64; 8],
    message_length: u32,
    candidate_count: u32,
}
unsafe impl DeviceRepr for Hs512BruteForceParams {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AesKwBruteForceParams {
    encrypted_key_len: u32,
    n_blocks: u32,
    candidate_count: u32,
}
unsafe impl DeviceRepr for AesKwBruteForceParams {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs256MarkovParams {
    target_signature: [u32; 8],
    message_length: u32,
    candidate_count: u32,
    pw_length: u32,
    threshold: u32,
    offset: u64,
}
unsafe impl DeviceRepr for Hs256MarkovParams {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hs512MarkovParams {
    target_signature: [u64; 8],
    message_length: u32,
    candidate_count: u32,
    pw_length: u32,
    threshold: u32,
    offset: u64,
}
unsafe impl DeviceRepr for Hs512MarkovParams {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AesKwMarkovParams {
    encrypted_key_len: u32,
    n_blocks: u32,
    candidate_count: u32,
    pw_length: u32,
    threshold: u32,
    _pad: u32,
    offset: u64,
}
unsafe impl DeviceRepr for AesKwMarkovParams {}

// ---------------------------------------------------------------------------
// Platform types — used by batch.rs, producer.rs via type aliases
// ---------------------------------------------------------------------------

/// Host-side staging buffer for GPU batch data, backed by CUDA
/// **pinned (page-locked) memory**.
///
/// On CUDA (discrete GPU), there is no unified memory like Metal's
/// `StorageModeShared`. Instead, batch data lives in host memory
/// and is explicitly copied to device memory before each kernel launch.
///
/// ## Why pinned memory?
///
/// Regular `Vec<u8>` memory is *pageable* — the OS can swap pages out.
/// When `cuMemcpyHtoD` copies pageable memory, the CUDA driver must
/// first copy it to an internal pinned staging buffer, then DMA that
/// to the GPU. With pinned memory, the DMA engine reads directly from
/// the host buffer, eliminating one copy and roughly doubling PCIe
/// throughput (measured ~2x improvement on discrete NVIDIA GPUs).
///
/// ## Safety
///
/// The buffer is allocated via `cuMemHostAlloc` and freed via
/// `cuMemFreeHost` on drop. The raw pointer is valid for the
/// lifetime of the struct. `Send` and `Sync` are safe because
/// the memory is owned exclusively and never aliased.
pub(crate) struct CudaBuffer {
    ptr: *mut u8,
    pub(crate) len: usize,
}

// SAFETY: CudaBuffer owns its pinned memory exclusively.
// No aliasing occurs — batch.rs writes through buffer_host_ptr()
// and the CUDA driver reads during memcpy_htod, but these are
// sequenced by the pipeline (pack completes before dispatch).
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl std::fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("len", &self.len)
            .field("pinned", &true)
            .finish()
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was allocated by cuMemHostAlloc in alloc_shared_buffer.
            unsafe {
                let _ = cudarc::driver::result::free_host(self.ptr.cast());
            }
        }
    }
}

impl CudaBuffer {
    /// Return a slice view of the pinned buffer contents.
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes, allocated by cuMemHostAlloc.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Wrapper around a CUDA context, serving as the device handle.
///
/// `Clone` is required because `producer.rs` clones the device handle
/// to distribute across worker threads. `Arc<CudaContext>` is already
/// thread-safe (Send + Sync), so cloning is cheap (reference count bump).
#[derive(Clone)]
pub(crate) struct CudaDeviceHandle(pub(crate) Arc<CudaContext>);

/// Opaque handle for an in-flight GPU dispatch.
///
/// In the Metal backend, this wraps a `CommandBuffer` with
/// `wait_until_completed()`. In CUDA, synchronisation is done via
/// the stream owned by the brute forcer, so this handle is a
/// zero-size marker. The runner stores it in `InFlightBatch` to
/// satisfy the trait contract.
pub(crate) struct CudaCommandHandle;

// ---------------------------------------------------------------------------
// Buffer helper functions — called by batch.rs and producer.rs
// via the platform-gated wrappers in gpu/mod.rs
// ---------------------------------------------------------------------------

/// Allocate a pinned (page-locked) host-side staging buffer.
///
/// Uses `cuMemHostAlloc` for DMA-friendly memory that transfers
/// to the GPU at full PCIe bandwidth (no intermediate staging copy).
///
/// The `device` parameter provides the CUDA context needed for
/// pinned allocation. `cuMemHostAlloc` requires an active context
/// on the calling thread, so we push it before allocating.
pub(crate) fn alloc_shared_buffer(device: &CudaDeviceHandle, size: usize) -> CudaBuffer {
    if size == 0 {
        return CudaBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
    }
    // Push the CUDA context onto the calling thread's context stack.
    // This is needed because packer worker threads (spawned by the
    // producer pipeline) don't inherit the main thread's context.
    // SAFETY: cu_ctx() returns the valid context from CudaContext::new().
    unsafe {
        let _ = cudarc::driver::result::ctx::set_current(device.0.cu_ctx());
    }
    // SAFETY: We request `size` bytes of pinned host memory.
    // Flag 0 = default (portable, not mapped to device address space).
    let ptr = unsafe { cudarc::driver::result::malloc_host(size, 0) }
        .expect("cuMemHostAlloc failed — out of pinned memory?");
    // Zero-initialize to match Vec<u8> behaviour.
    unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, size) };
    CudaBuffer {
        ptr: ptr.cast::<u8>(),
        len: size,
    }
}

/// Return a raw mutable pointer to the buffer's pinned host memory.
///
/// The caller (batch.rs) writes candidate data directly through this
/// pointer. The pointer remains valid as long as the `CudaBuffer` is
/// not dropped — guaranteed because `WordBatch` owns the buffer.
pub(crate) fn buffer_host_ptr(buf: &CudaBuffer) -> *mut u8 {
    buf.ptr
}

/// Create a CUDA context on GPU device 0 (the default).
#[allow(dead_code)] // Used via gpu::default_device() on Linux
pub(crate) fn default_device() -> anyhow::Result<CudaDeviceHandle> {
    let ctx =
        CudaContext::new(0).map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
    Ok(CudaDeviceHandle(ctx))
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Reinterpret a `#[repr(C)]` struct as a byte slice for copying to device.
fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

/// Compile CUDA source to PTX via NVRTC, load into a module, and
/// extract the named kernel functions.
fn compile_and_load(
    ctx: &Arc<CudaContext>,
    source: &str,
    fn_mixed: &str,
    fn_short: &str,
    extra_opts: &[String],
) -> anyhow::Result<(Arc<cudarc::driver::CudaModule>, CudaFunction, CudaFunction)> {
    let mut opts = CompileOptions {
        options: extra_opts.to_vec(),
        ..Default::default()
    };
    // Detect compute capability for architecture-specific optimisation.
    if let Ok((major, minor)) = ctx.compute_capability() {
        opts.arch = None; // We set it via the options string below.
        opts.options
            .push(format!("--gpu-architecture=compute_{major}{minor}"));
    }
    let ptx = nvrtc::compile_ptx_with_opts(source, opts)
        .map_err(|e| anyhow!("NVRTC compilation failed: {e}"))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| anyhow!("failed to load CUDA module: {e}"))?;
    let func_mixed = module
        .load_function(fn_mixed)
        .map_err(|e| anyhow!("failed to load CUDA function '{fn_mixed}': {e}"))?;
    let func_short = module
        .load_function(fn_short)
        .map_err(|e| anyhow!("failed to load CUDA function '{fn_short}': {e}"))?;
    Ok((module, func_mixed, func_short))
}

/// Compile CUDA source to PTX and load a single named kernel function.
/// Used for Markov kernels which have only one entry point (no mixed/short split).
fn compile_and_load_single(
    ctx: &Arc<CudaContext>,
    source: &str,
    fn_name: &str,
    extra_opts: &[String],
) -> anyhow::Result<(Arc<cudarc::driver::CudaModule>, CudaFunction)> {
    let mut opts = CompileOptions {
        options: extra_opts.to_vec(),
        ..Default::default()
    };
    if let Ok((major, minor)) = ctx.compute_capability() {
        opts.arch = None;
        opts.options
            .push(format!("--gpu-architecture=compute_{major}{minor}"));
    }
    let ptx = nvrtc::compile_ptx_with_opts(source, opts)
        .map_err(|e| anyhow!("NVRTC compilation failed: {e}"))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| anyhow!("failed to load CUDA module: {e}"))?;
    let func = module
        .load_function(fn_name)
        .map_err(|e| anyhow!("failed to load CUDA function '{fn_name}': {e}"))?;
    Ok((module, func))
}

// ---------------------------------------------------------------------------
// CudaBruteForcer — HMAC-SHA JWT cracking
// ---------------------------------------------------------------------------

/// CUDA compute backend for HMAC-SHA brute-force cracking.
///
/// Owns the CUDA context, stream, compiled kernels, and pre-allocated
/// device buffers.
///
/// ## Zero-copy vs per-batch dispatch
///
/// At construction time, we query available VRAM via `cuMemGetInfo` and
/// decide which of two strategies to use:
///
/// 1. **Zero-copy** (wordlist fits in VRAM): the entire mmap is uploaded
///    once to `d_mmap`. Per-batch dispatch only copies metadata (offsets
///    and lengths — ~30 MiB). The kernel reads candidate bytes directly
///    from VRAM using absolute mmap offsets. This is the fast path.
///
/// 2. **Per-batch copy** (wordlist exceeds VRAM): `d_mmap` is a small
///    (~44 MiB) reusable buffer. Before each kernel launch, word bytes
///    from the batch's pinned host buffer are copied to `d_mmap` via
///    `memcpy_htod`. Offsets in this mode are batch-relative (starting
///    from 0), not absolute mmap positions. This avoids OOM crashes at
///    the cost of PCIe transfer overhead (~2 ms per 32 MiB batch on
///    PCIe Gen4).
///
/// The `zero_copy` field records which path was chosen. The kernel source
/// is identical for both modes — it simply reads `word_bytes[offsets[i]]`,
/// and whether those offsets are absolute or batch-relative is determined
/// by how the producer packed the batch.
pub(crate) struct CudaBruteForcer {
    variant: HmacVariant,
    #[allow(dead_code)] // Kept alive to prevent CUDA context from being dropped
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<cudarc::driver::CudaModule>,
    func_mixed: CudaFunction,
    func_short_keys: CudaFunction,
    // --- Pre-allocated device buffers ---
    //
    // All device buffers are wrapped in `UnsafeCell` to allow interior
    // mutability through `&self` (required by the `GpuBruteForcer` trait).
    // See the module-level safety comment for why this is sound.
    d_params: UnsafeCell<CudaSlice<u8>>,
    d_message: CudaSlice<u8>,
    /// Dual-purpose device buffer whose role depends on `zero_copy`:
    ///
    /// - **zero_copy = true**: holds the entire wordlist mmap in VRAM.
    ///   Offsets from the batch are absolute mmap positions; the kernel
    ///   reads `d_mmap[offset..offset+len]` to get each candidate.
    ///
    /// - **zero_copy = false**: a reusable ~44 MiB staging buffer. Before
    ///   each kernel launch, `encode_and_commit_impl` copies the batch's
    ///   word bytes here via `memcpy_htod`. Offsets are batch-relative.
    ///
    /// The `UnsafeCell` wrapper is needed because per-batch mode mutates
    /// this buffer during `encode_and_commit` (which takes `&self`).
    d_mmap: UnsafeCell<CudaSlice<u8>>,
    d_word_offsets: UnsafeCell<CudaSlice<u8>>,
    d_word_lengths: UnsafeCell<CudaSlice<u8>>,
    d_result: UnsafeCell<CudaSlice<u32>>,
    message_length: u32,
    threadgroup_width: usize,
    device_name: String,
    max_threads_per_block: usize,
    device_handle: CudaDeviceHandle,
    /// Runtime toggle: `true` = mmap in VRAM (fast), `false` = per-batch copy.
    ///
    /// Determined at construction by comparing the wordlist size against
    /// available VRAM (with 10% headroom). Once set, this flag controls:
    ///   - Whether `encode_and_commit_impl` copies word bytes per-batch
    ///   - Whether the producer writes absolute or batch-relative offsets
    ///   - Whether `handle_match` reads from the mmap or the batch buffer
    zero_copy: bool,
}

impl CudaBruteForcer {
    /// Compile the CUDA kernel, load functions, upload mmap to VRAM,
    /// and allocate device buffers for the given HMAC variant.
    ///
    /// `mmap_bytes` is the entire wordlist file content. It is uploaded
    /// to GPU VRAM once here; per-batch dispatch only copies metadata.
    pub(crate) fn new(variant: HmacVariant, signing_input: &[u8], mmap_bytes: &[u8]) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("failed to create CUDA stream: {e}"))?;

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".into());

        // Select source and kernel entry point names.
        let (source, fn_mixed, fn_short) = match variant {
            HmacVariant::Hs256 => (
                CUDA_SOURCE_HS256,
                "hs256_wordlist",
                "hs256_wordlist_short_keys",
            ),
            HmacVariant::Hs384 => (
                CUDA_SOURCE_HS512,
                "hs384_wordlist",
                "hs384_wordlist_short_keys",
            ),
            HmacVariant::Hs512 => (
                CUDA_SOURCE_HS512,
                "hs512_wordlist",
                "hs512_wordlist_short_keys",
            ),
        };

        let (module, func_mixed, func_short_keys) =
            compile_and_load(&ctx, source, fn_mixed, fn_short, &[])?;

        let max_threads_per_block = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            )
            .map(|v| v as usize)
            .unwrap_or(1024);
        let threadgroup_width = 256usize.min(max_threads_per_block.max(1));

        let message_length =
            u32::try_from(signing_input.len()).context("JWT signing input too long")?;

        // Allocate device buffers.
        let params_size = match variant {
            HmacVariant::Hs256 => std::mem::size_of::<Hs256BruteForceParams>(),
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                std::mem::size_of::<Hs512BruteForceParams>()
            }
        };
        let d_params = stream
            .alloc_zeros::<u8>(params_size)
            .map_err(|e| anyhow!("failed to alloc params buffer: {e}"))?;

        // Message buffer: copy signing input once, reuse for all dispatches.
        // Handle empty messages by allocating at least 1 byte.
        let d_message = if signing_input.is_empty() {
            stream
                .alloc_zeros::<u8>(1)
                .map_err(|e| anyhow!("failed to alloc message buffer: {e}"))?
        } else {
            stream
                .clone_htod(signing_input)
                .map_err(|e| anyhow!("failed to copy message to device: {e}"))?
        };

        // ## VRAM capacity check — zero-copy vs per-batch fallback
        //
        // This is the key OOM-prevention logic. We query the GPU's free VRAM
        // via `cuMemGetInfo` and compare it to the wordlist file size:
        //
        //   - If the file fits (with 10% headroom for other buffers), we
        //     upload the entire mmap to VRAM once. This is the zero-copy
        //     path: the kernel reads candidate bytes directly from VRAM.
        //
        //   - If it doesn't fit, we allocate a small reusable buffer
        //     (~44 MiB = WORD_BYTES_ALLOC_SIZE) and copy word bytes from
        //     pinned host memory before each kernel launch. This is the
        //     per-batch copy path, identical to how macOS Metal works.
        //
        // The 10% headroom (`free_vram / 10`) reserves space for the
        // metadata buffers (offsets, lengths, params, result) that are
        // also allocated in VRAM. Without headroom, a file that barely
        // fits would OOM when allocating those buffers.
        let mmap_len = mmap_bytes.len();
        let (d_mmap, zero_copy) = if mmap_len == 0 {
            // Edge case: empty wordlist. Allocate 1-byte placeholder to
            // satisfy the kernel argument contract (non-null device pointer).
            let buf = stream
                .alloc_zeros::<u8>(1)
                .map_err(|e| anyhow!("failed to alloc placeholder mmap buffer: {e}"))?;
            (buf, true) // Empty mmap is trivially "zero-copy".
        } else {
            let (free_vram, _total_vram) = cudarc::driver::result::mem_get_info()
                .map_err(|e| anyhow!("failed to query VRAM: {e}"))?;
            let headroom = free_vram / 10;
            if mmap_len < free_vram.saturating_sub(headroom) {
                // Fast path: upload entire mmap to VRAM.
                // `clone_htod` performs a single contiguous DMA transfer.
                eprintln!(
                    "CUDA: uploading {} byte mmap to GPU VRAM ({} free)",
                    format_human_count(mmap_len as f64),
                    format_human_count(free_vram as f64),
                );
                let buf = stream
                    .clone_htod(mmap_bytes)
                    .map_err(|e| anyhow!("failed to upload mmap to device ({mmap_len} bytes): {e}"))?;
                (buf, true)
            } else {
                // Fallback path: wordlist too large for VRAM.
                // Allocate a per-batch staging buffer sized to the maximum
                // word bytes payload. This buffer is reused across all batches
                // — its contents are overwritten before each kernel launch.
                eprintln!(
                    "CUDA: mmap too large for VRAM ({} > {} free), using per-batch copy",
                    format_human_count(mmap_len as f64),
                    format_human_count(free_vram as f64),
                );
                let buf = stream
                    .alloc_zeros::<u8>(crate::batch::WORD_BYTES_ALLOC_SIZE)
                    .map_err(|e| anyhow!("failed to alloc word_bytes device buffer: {e}"))?;
                (buf, false)
            }
        };
        let d_word_offsets = stream
            .alloc_zeros::<u8>(MAX_CANDIDATES_PER_BATCH * 4)
            .map_err(|e| anyhow!("failed to alloc word_offsets buffer: {e}"))?;
        let d_word_lengths = stream
            .alloc_zeros::<u8>(MAX_CANDIDATES_PER_BATCH * 2)
            .map_err(|e| anyhow!("failed to alloc word_lengths buffer: {e}"))?;
        let d_result = stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow!("failed to alloc result buffer: {e}"))?;

        stream
            .synchronize()
            .map_err(|e| anyhow!("failed to sync after init: {e}"))?;

        let device_handle = CudaDeviceHandle(ctx.clone());

        Ok(Self {
            variant,
            ctx,
            stream,
            _module: module,
            func_mixed,
            func_short_keys,
            d_params: UnsafeCell::new(d_params),
            d_message,
            d_mmap: UnsafeCell::new(d_mmap),
            d_word_offsets: UnsafeCell::new(d_word_offsets),
            d_word_lengths: UnsafeCell::new(d_word_lengths),
            d_result: UnsafeCell::new(d_result),
            message_length,
            threadgroup_width,
            device_name,
            max_threads_per_block,
            device_handle,
            zero_copy,
        })
    }

    /// The byte length threshold for the short-key fast path.
    fn short_key_threshold(&self) -> u16 {
        match self.variant {
            HmacVariant::Hs256 => 64,
            HmacVariant::Hs384 | HmacVariant::Hs512 => 128,
        }
    }

    /// Choose the kernel function based on the batch's longest candidate.
    fn active_func(&self, max_word_len: u16) -> &CudaFunction {
        if max_word_len <= self.short_key_threshold() {
            &self.func_short_keys
        } else {
            &self.func_mixed
        }
    }

    /// Write the HMAC params struct to device memory.
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

        let param_bytes: Vec<u8> = match self.variant {
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
                bytes_of(&params).to_vec()
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
                bytes_of(&params).to_vec()
            }
        };

        // SAFETY: Single-threaded access — see module-level safety comment.
        let d_params = unsafe { &mut *self.d_params.get() };
        self.stream
            .memcpy_htod(&param_bytes, d_params)
            .map_err(|e| anyhow!("failed to copy params to device: {e}"))?;
        Ok(())
    }

    /// Copy batch metadata to device, launch kernel, return command handle.
    ///
    /// In zero-copy mode, only the small metadata arrays (offsets and lengths)
    /// are copied per batch. In per-batch mode, word bytes are also copied.
    ///
    /// `candidate_count_override` allows autotuning to dispatch a
    /// smaller prefix of the batch without copying all metadata.
    fn encode_and_commit_impl(
        &self,
        target_signature: &[u8],
        batch: &WordBatch,
        threadgroup_width: usize,
        candidate_count_override: Option<usize>,
    ) -> anyhow::Result<(CudaCommandHandle, Duration, Duration)> {
        let actual_count = candidate_count_override.unwrap_or_else(|| batch.candidate_count());
        let candidate_count =
            u32::try_from(actual_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            bail!("cannot encode empty batch for dispatch");
        }

        let view = batch.as_dispatch_view();

        // Phase 1: Host prep — write params and copy metadata to device.
        let prep_started = Instant::now();
        self.write_params(target_signature, candidate_count)?;

        // Reset result sentinel.
        let d_result = unsafe { &mut *self.d_result.get() };
        self.stream
            .memcpy_htod(&[RESULT_NOT_FOUND_SENTINEL], d_result)
            .map_err(|e| anyhow!("failed to reset result sentinel: {e}"))?;

        // Copy metadata (offsets + lengths) from host to device.
        let d_word_offsets = unsafe { &mut *self.d_word_offsets.get() };
        let d_word_lengths = unsafe { &mut *self.d_word_lengths.get() };

        let offsets_bytes = actual_count * 4;
        if offsets_bytes > 0 {
            self.stream
                .memcpy_htod(
                    &view.word_offsets_buf.as_slice()[..offsets_bytes],
                    d_word_offsets,
                )
                .map_err(|e| anyhow!("failed to copy word_offsets to device: {e}"))?;
        }
        let lengths_bytes = actual_count * 2;
        if lengths_bytes > 0 {
            self.stream
                .memcpy_htod(
                    &view.word_lengths_buf.as_slice()[..lengths_bytes],
                    d_word_lengths,
                )
                .map_err(|e| anyhow!("failed to copy word_lengths to device: {e}"))?;
        }

        // ## Per-batch word bytes copy (fallback mode)
        //
        // In zero-copy mode, `d_mmap` already holds the entire wordlist
        // in VRAM, so we skip this step entirely — a major performance win.
        //
        // In per-batch mode, the producer packed word bytes into the batch's
        // pinned host buffer with batch-relative offsets (starting at 0).
        // We copy just the valid prefix (`word_bytes_len` bytes, typically
        // ~32 MiB) to the reusable `d_mmap` device buffer. The kernel then
        // reads `d_mmap[offset..offset+len]` using those batch-relative offsets.
        //
        // Cost: ~2 ms per 32 MiB on PCIe Gen4. This is the price we pay
        // for supporting wordlists larger than GPU VRAM. For comparison,
        // the GPU computation for a 6M-candidate HMAC-SHA256 batch takes
        // ~3–5 ms, so the PCIe copy adds roughly 40–60% overhead.
        if !self.zero_copy {
            let d_mmap = unsafe { &mut *self.d_mmap.get() };
            let word_bytes_len = view.word_bytes_len;
            if word_bytes_len > 0 {
                self.stream
                    .memcpy_htod(
                        &view.word_bytes_buf.as_slice()[..word_bytes_len],
                        d_mmap,
                    )
                    .map_err(|e| anyhow!("failed to copy word_bytes to device: {e}"))?;
            }
        }

        let host_prep = prep_started.elapsed();

        // Phase 2: Kernel launch.
        //
        // The kernel function, grid dimensions, and argument layout are
        // identical for both zero-copy and per-batch modes. The kernel
        // doesn't know (or care) whether `d_mmap` holds the full wordlist
        // or just one batch's worth of word bytes — it simply reads
        // `word_bytes[offsets[tid]]` for `lengths[tid]` bytes.
        let encode_started = Instant::now();
        let func = self.active_func(view.max_word_len);
        let grid_dim = (candidate_count + threadgroup_width as u32 - 1) / threadgroup_width as u32;
        let config = LaunchConfig {
            grid_dim: (grid_dim.max(1), 1, 1),
            block_dim: (threadgroup_width as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel arguments match the CUDA kernel signature:
        //   params*, message*, word_bytes*, word_offsets*, word_lengths*, result*
        // The buffer contents are valid because:
        //   - d_params was just written by write_params()
        //   - d_message was written once in new()
        //   - d_mmap holds either the full mmap (zero-copy) or this batch's
        //     word bytes (per-batch), both valid for the current offsets
        //   - d_word_offsets/d_word_lengths were just copied above
        //   - d_result was reset to RESULT_NOT_FOUND_SENTINEL
        let d_params = unsafe { &*self.d_params.get() };
        let d_mmap = unsafe { &*self.d_mmap.get() };
        let d_word_offsets = unsafe { &*self.d_word_offsets.get() };
        let d_word_lengths = unsafe { &*self.d_word_lengths.get() };
        let d_result = unsafe { &mut *self.d_result.get() };

        unsafe {
            self.stream
                .launch_builder(func)
                .arg(d_params)
                .arg(&self.d_message)
                .arg(d_mmap)
                .arg(d_word_offsets)
                .arg(d_word_lengths)
                .arg(d_result)
                .launch(config)
                .map_err(|e| anyhow!("kernel launch failed: {e}"))?;
        }
        let command_encode = encode_started.elapsed();

        Ok((CudaCommandHandle, host_prep, command_encode))
    }

    /// Block until GPU execution completes, then read the result buffer.
    fn wait_and_readback_impl(&self) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        let _ = self.stream.synchronize();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let d_result = unsafe { &*self.d_result.get() };
        let result_vec = self
            .stream
            .clone_dtoh(d_result)
            .unwrap_or_else(|_| vec![RESULT_NOT_FOUND_SENTINEL]);
        let result = result_vec.first().copied().unwrap_or(RESULT_NOT_FOUND_SENTINEL);
        let _ = self.stream.synchronize();
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    /// Synchronous dispatch for autotune benchmarking.
    fn dispatch_sync(
        &self,
        target_signature: &[u8],
        batch: &WordBatch,
        threadgroup_width: usize,
        sample_count: Option<usize>,
    ) -> anyhow::Result<(Option<u32>, BatchDispatchTimings)> {
        let started = Instant::now();
        let mut timings = BatchDispatchTimings::default();
        if batch.candidate_count() == 0 {
            timings.total = started.elapsed();
            return Ok((None, timings));
        }

        let (_handle, host_prep, command_encode) =
            self.encode_and_commit_impl(target_signature, batch, threadgroup_width, sample_count)?;
        timings.host_prep = host_prep;
        timings.command_encode = command_encode;

        let (maybe_match, gpu_wait, result_readback) = self.wait_and_readback_impl();
        timings.gpu_wait = gpu_wait;
        timings.result_readback = result_readback;
        timings.total = started.elapsed();
        Ok((maybe_match, timings))
    }
}

impl GpuBruteForcer for CudaBruteForcer {
    fn device(&self) -> &super::GpuDevice {
        &self.device_handle
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn zero_copy(&self) -> bool {
        self.zero_copy
    }

    fn thread_execution_width(&self) -> usize {
        WARP_SIZE
    }

    fn max_total_threads_per_threadgroup(&self) -> usize {
        self.max_threads_per_block
    }

    fn current_threadgroup_width(&self) -> usize {
        self.threadgroup_width
    }

    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()> {
        if requested == 0 {
            bail!("--threads-per-group must be > 0");
        }
        if requested > self.max_threads_per_block {
            bail!(
                "--threads-per-group {} exceeds max {}",
                requested,
                self.max_threads_per_block
            );
        }
        self.threadgroup_width = requested;
        Ok(())
    }

    fn encode_and_commit(
        &self,
        target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<(super::GpuCommandHandle, Duration, Duration)> {
        self.encode_and_commit_impl(target_signature, batch, self.threadgroup_width, None)
    }

    fn wait_and_readback(
        &self,
        _handle: &super::GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration) {
        self.wait_and_readback_impl()
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

        let mut candidates = vec![32, 64, 128, 256, 512, 1024];
        candidates.retain(|&v| v > 0 && v <= self.max_threads_per_block);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(());
        }

        eprintln!(
            "AUTOTUNE: benchmarking {} candidates across {} block sizes",
            sample_count,
            candidates.len()
        );

        let mut best: Option<(usize, f64)> = None;
        for width in candidates {
            let result =
                self.dispatch_sync(target_signature, batch, width, Some(sample_count));
            let (_match_result, timings) = match result {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("AUTOTUNE: block_size={width} failed (too many resources), skipping");
                    continue;
                }
            };
            if timings.gpu_wait.is_zero() {
                continue;
            }
            let rate = sample_count as f64 / timings.gpu_wait.as_secs_f64();
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

// ---------------------------------------------------------------------------
// CudaAesKwBruteForcer — AES Key Wrap JWE cracking
// ---------------------------------------------------------------------------

/// CUDA compute backend for AES Key Wrap JWE cracking (A128KW/A192KW/A256KW).
///
/// The kernel source is compiled with `#define AES_KEY_BYTES N` to
/// produce a fully specialised binary for each AES key size.
///
/// Supports both zero-copy mmap dispatch and per-batch fallback,
/// using the same VRAM capacity check as `CudaBruteForcer`. See
/// that struct's documentation for a detailed explanation of the
/// two dispatch modes.
pub(crate) struct CudaAesKwBruteForcer {
    #[allow(dead_code)]
    variant: AesKwVariant,
    #[allow(dead_code)] // Kept alive to prevent CUDA context from being dropped
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<cudarc::driver::CudaModule>,
    func_mixed: CudaFunction,
    func_short_keys: CudaFunction,
    d_params: UnsafeCell<CudaSlice<u8>>,
    d_encrypted_key: CudaSlice<u8>,
    /// See `CudaBruteForcer::d_mmap` for documentation on the dual-purpose
    /// nature of this buffer (whole mmap vs per-batch staging).
    d_mmap: UnsafeCell<CudaSlice<u8>>,
    d_word_offsets: UnsafeCell<CudaSlice<u8>>,
    d_word_lengths: UnsafeCell<CudaSlice<u8>>,
    d_result: UnsafeCell<CudaSlice<u32>>,
    n_blocks: u32,
    short_key_threshold: u16,
    threadgroup_width: usize,
    device_name: String,
    max_threads_per_block: usize,
    device_handle: CudaDeviceHandle,
    /// Runtime toggle — see `CudaBruteForcer::zero_copy` for documentation.
    zero_copy: bool,
}

impl CudaAesKwBruteForcer {
    /// Compile the AESKW CUDA kernel with the appropriate AES_KEY_BYTES
    /// define, load functions, upload mmap to VRAM, and allocate device buffers.
    pub(crate) fn new(
        variant: AesKwVariant,
        encrypted_key: &[u8],
        n_blocks: usize,
        mmap_bytes: &[u8],
    ) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("failed to create CUDA stream: {e}"))?;

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".into());

        let key_bytes_define = format!("-DAES_KEY_BYTES={}", variant.key_bytes());
        let (module, func_mixed, func_short_keys) = compile_and_load(
            &ctx,
            CUDA_SOURCE_AESKW,
            "aeskw_wordlist",
            "aeskw_wordlist_short_keys",
            &[key_bytes_define],
        )?;

        let max_threads_per_block = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            )
            .map(|v| v as usize)
            .unwrap_or(1024);
        let threadgroup_width = 256usize.min(max_threads_per_block.max(1));

        let n_blocks_u32 =
            u32::try_from(n_blocks).context("n_blocks exceeds u32")?;

        let d_params = stream
            .alloc_zeros::<u8>(std::mem::size_of::<AesKwBruteForceParams>())
            .map_err(|e| anyhow!("failed to alloc params buffer: {e}"))?;
        let d_encrypted_key = stream
            .clone_htod(encrypted_key)
            .map_err(|e| anyhow!("failed to copy encrypted_key to device: {e}"))?;
        // VRAM capacity check — same logic as CudaBruteForcer::new().
        // See the HMAC backend above for a detailed explanation of the
        // zero-copy vs per-batch copy decision and the 10% headroom rule.
        let mmap_len = mmap_bytes.len();
        let (d_mmap, zero_copy) = if mmap_len == 0 {
            let buf = stream
                .alloc_zeros::<u8>(1)
                .map_err(|e| anyhow!("failed to alloc placeholder mmap buffer: {e}"))?;
            (buf, true)
        } else {
            let (free_vram, _total_vram) = cudarc::driver::result::mem_get_info()
                .map_err(|e| anyhow!("failed to query VRAM: {e}"))?;
            let headroom = free_vram / 10;
            if mmap_len < free_vram.saturating_sub(headroom) {
                eprintln!(
                    "CUDA: uploading {} byte mmap to GPU VRAM ({} free)",
                    format_human_count(mmap_len as f64),
                    format_human_count(free_vram as f64),
                );
                let buf = stream
                    .clone_htod(mmap_bytes)
                    .map_err(|e| anyhow!("failed to upload mmap to device ({mmap_len} bytes): {e}"))?;
                (buf, true)
            } else {
                eprintln!(
                    "CUDA: mmap too large for VRAM ({} > {} free), using per-batch copy",
                    format_human_count(mmap_len as f64),
                    format_human_count(free_vram as f64),
                );
                let buf = stream
                    .alloc_zeros::<u8>(crate::batch::WORD_BYTES_ALLOC_SIZE)
                    .map_err(|e| anyhow!("failed to alloc word_bytes device buffer: {e}"))?;
                (buf, false)
            }
        };
        let d_word_offsets = stream
            .alloc_zeros::<u8>(MAX_CANDIDATES_PER_BATCH * 4)
            .map_err(|e| anyhow!("failed to alloc word_offsets buffer: {e}"))?;
        let d_word_lengths = stream
            .alloc_zeros::<u8>(MAX_CANDIDATES_PER_BATCH * 2)
            .map_err(|e| anyhow!("failed to alloc word_lengths buffer: {e}"))?;
        let d_result = stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow!("failed to alloc result buffer: {e}"))?;

        stream
            .synchronize()
            .map_err(|e| anyhow!("failed to sync after init: {e}"))?;

        let device_handle = CudaDeviceHandle(ctx.clone());
        let short_key_threshold = variant.key_bytes() as u16;

        Ok(Self {
            variant,
            ctx,
            stream,
            _module: module,
            func_mixed,
            func_short_keys,
            d_params: UnsafeCell::new(d_params),
            d_encrypted_key,
            d_mmap: UnsafeCell::new(d_mmap),
            d_word_offsets: UnsafeCell::new(d_word_offsets),
            d_word_lengths: UnsafeCell::new(d_word_lengths),
            d_result: UnsafeCell::new(d_result),
            n_blocks: n_blocks_u32,
            short_key_threshold,
            threadgroup_width,
            device_name,
            max_threads_per_block,
            device_handle,
            zero_copy,
        })
    }

    fn active_func(&self, max_word_len: u16) -> &CudaFunction {
        if max_word_len <= self.short_key_threshold {
            &self.func_short_keys
        } else {
            &self.func_mixed
        }
    }

    fn write_params(&self, candidate_count: u32) -> anyhow::Result<()> {
        let params = AesKwBruteForceParams {
            encrypted_key_len: self.d_encrypted_key.len() as u32,
            n_blocks: self.n_blocks,
            candidate_count,
        };
        let param_bytes = bytes_of(&params).to_vec();
        let d_params = unsafe { &mut *self.d_params.get() };
        self.stream
            .memcpy_htod(&param_bytes, d_params)
            .map_err(|e| anyhow!("failed to copy params to device: {e}"))?;
        Ok(())
    }

    fn encode_and_commit_impl(
        &self,
        _target_data: &[u8],
        batch: &WordBatch,
        threadgroup_width: usize,
        candidate_count_override: Option<usize>,
    ) -> anyhow::Result<(CudaCommandHandle, Duration, Duration)> {
        let actual_count = candidate_count_override.unwrap_or_else(|| batch.candidate_count());
        let candidate_count =
            u32::try_from(actual_count).context("candidate count exceeds u32")?;
        if candidate_count == 0 {
            bail!("cannot encode empty batch");
        }

        let view = batch.as_dispatch_view();

        let prep_started = Instant::now();
        self.write_params(candidate_count)?;

        let d_result = unsafe { &mut *self.d_result.get() };
        self.stream
            .memcpy_htod(&[RESULT_NOT_FOUND_SENTINEL], d_result)
            .map_err(|e| anyhow!("failed to reset result sentinel: {e}"))?;

        // Copy metadata (offsets + lengths) from host to device.
        let d_word_offsets = unsafe { &mut *self.d_word_offsets.get() };
        let d_word_lengths = unsafe { &mut *self.d_word_lengths.get() };

        let offsets_bytes = actual_count * 4;
        if offsets_bytes > 0 {
            self.stream
                .memcpy_htod(
                    &view.word_offsets_buf.as_slice()[..offsets_bytes],
                    d_word_offsets,
                )
                .map_err(|e| anyhow!("failed to copy word_offsets to device: {e}"))?;
        }
        let lengths_bytes = actual_count * 2;
        if lengths_bytes > 0 {
            self.stream
                .memcpy_htod(
                    &view.word_lengths_buf.as_slice()[..lengths_bytes],
                    d_word_lengths,
                )
                .map_err(|e| anyhow!("failed to copy word_lengths to device: {e}"))?;
        }

        // Per-batch word bytes copy — see CudaBruteForcer::encode_and_commit_impl
        // for a detailed explanation of the zero-copy vs per-batch copy decision.
        if !self.zero_copy {
            let d_mmap = unsafe { &mut *self.d_mmap.get() };
            let word_bytes_len = view.word_bytes_len;
            if word_bytes_len > 0 {
                self.stream
                    .memcpy_htod(
                        &view.word_bytes_buf.as_slice()[..word_bytes_len],
                        d_mmap,
                    )
                    .map_err(|e| anyhow!("failed to copy word_bytes to device: {e}"))?;
            }
        }

        let host_prep = prep_started.elapsed();

        // Kernel launch — identical for both zero-copy and per-batch modes.
        let encode_started = Instant::now();
        let func = self.active_func(view.max_word_len);
        let grid_dim = (candidate_count + threadgroup_width as u32 - 1) / threadgroup_width as u32;
        let config = LaunchConfig {
            grid_dim: (grid_dim.max(1), 1, 1),
            block_dim: (threadgroup_width as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_params = unsafe { &*self.d_params.get() };
        let d_mmap = unsafe { &*self.d_mmap.get() };
        let d_word_offsets = unsafe { &*self.d_word_offsets.get() };
        let d_word_lengths = unsafe { &*self.d_word_lengths.get() };
        let d_result = unsafe { &mut *self.d_result.get() };

        unsafe {
            self.stream
                .launch_builder(func)
                .arg(d_params)
                .arg(&self.d_encrypted_key)
                .arg(d_mmap)
                .arg(d_word_offsets)
                .arg(d_word_lengths)
                .arg(d_result)
                .launch(config)
                .map_err(|e| anyhow!("kernel launch failed: {e}"))?;
        }
        let command_encode = encode_started.elapsed();

        Ok((CudaCommandHandle, host_prep, command_encode))
    }

    fn wait_and_readback_impl(&self) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        let _ = self.stream.synchronize();
        let gpu_wait = gpu_wait_started.elapsed();

        let readback_started = Instant::now();
        let d_result = unsafe { &*self.d_result.get() };
        let result_vec = self
            .stream
            .clone_dtoh(d_result)
            .unwrap_or_else(|_| vec![RESULT_NOT_FOUND_SENTINEL]);
        let result = result_vec.first().copied().unwrap_or(RESULT_NOT_FOUND_SENTINEL);
        let _ = self.stream.synchronize();
        let result_readback = readback_started.elapsed();

        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL {
            None
        } else {
            Some(result)
        };
        (maybe_match, gpu_wait, result_readback)
    }

    fn dispatch_sync(
        &self,
        target_data: &[u8],
        batch: &WordBatch,
        threadgroup_width: usize,
        sample_count: Option<usize>,
    ) -> anyhow::Result<(Option<u32>, BatchDispatchTimings)> {
        let started = Instant::now();
        let mut timings = BatchDispatchTimings::default();
        if batch.candidate_count() == 0 {
            timings.total = started.elapsed();
            return Ok((None, timings));
        }

        let (_handle, host_prep, command_encode) =
            self.encode_and_commit_impl(target_data, batch, threadgroup_width, sample_count)?;
        timings.host_prep = host_prep;
        timings.command_encode = command_encode;

        let (maybe_match, gpu_wait, result_readback) = self.wait_and_readback_impl();
        timings.gpu_wait = gpu_wait;
        timings.result_readback = result_readback;
        timings.total = started.elapsed();
        Ok((maybe_match, timings))
    }
}

impl GpuBruteForcer for CudaAesKwBruteForcer {
    fn device(&self) -> &super::GpuDevice {
        &self.device_handle
    }

    fn zero_copy(&self) -> bool {
        self.zero_copy
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn thread_execution_width(&self) -> usize {
        WARP_SIZE
    }

    fn max_total_threads_per_threadgroup(&self) -> usize {
        self.max_threads_per_block
    }

    fn current_threadgroup_width(&self) -> usize {
        self.threadgroup_width
    }

    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()> {
        if requested == 0 {
            bail!("--threads-per-group must be > 0");
        }
        if requested > self.max_threads_per_block {
            bail!(
                "--threads-per-group {} exceeds max {}",
                requested,
                self.max_threads_per_block
            );
        }
        self.threadgroup_width = requested;
        Ok(())
    }

    fn encode_and_commit(
        &self,
        target_data: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<(super::GpuCommandHandle, Duration, Duration)> {
        self.encode_and_commit_impl(target_data, batch, self.threadgroup_width, None)
    }

    fn wait_and_readback(
        &self,
        _handle: &super::GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration) {
        self.wait_and_readback_impl()
    }

    fn autotune_threadgroup_width(
        &mut self,
        target_data: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<()> {
        let sample_count = batch.candidate_count().min(16_384);
        if sample_count == 0 {
            return Ok(());
        }

        let mut candidates = vec![32, 64, 128, 256, 512, 1024];
        candidates.retain(|&v| v > 0 && v <= self.max_threads_per_block);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(());
        }

        eprintln!(
            "AUTOTUNE: benchmarking {} candidates across {} block sizes",
            sample_count,
            candidates.len()
        );

        let mut best: Option<(usize, f64)> = None;
        for width in candidates {
            let result =
                self.dispatch_sync(target_data, batch, width, Some(sample_count));
            let (_match_result, timings) = match result {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("AUTOTUNE: block_size={width} failed (too many resources), skipping");
                    continue;
                }
            };
            if timings.gpu_wait.is_zero() {
                continue;
            }
            let rate = sample_count as f64 / timings.gpu_wait.as_secs_f64();
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

// ---------------------------------------------------------------------------
// CudaMarkovHmacBruteForcer — Markov + HMAC-SHA
// ---------------------------------------------------------------------------

/// CUDA compute backend for Markov chain + HMAC-SHA cracking.
///
/// Each GPU thread decodes a global keyspace index into per-position rank
/// selections, walks an order-1 Markov chain using the pre-trained lookup
/// table, and immediately computes HMAC-SHA on the generated candidate.
/// The candidate never leaves registers — no intermediate buffer is needed.
///
/// ## Interior mutability via `UnsafeCell`
///
/// Same pattern as `CudaBruteForcer`: the `GpuMarkovBruteForcer` trait's
/// `encode_and_commit_markov` takes `&self`, but CUDA dispatch requires
/// mutable access to device buffers (for `memcpy_htod`). Safety is
/// guaranteed by the runner's single-dispatch-at-a-time invariant.
pub(crate) struct CudaMarkovHmacBruteForcer {
    variant: HmacVariant,
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<cudarc::driver::CudaModule>,
    func: CudaFunction,
    d_params: UnsafeCell<CudaSlice<u8>>,
    d_message: CudaSlice<u8>,
    d_markov_table: CudaSlice<u8>,
    d_result: UnsafeCell<CudaSlice<u32>>,
    message_length: u32,
    threadgroup_width: usize,
    device_name: String,
    max_threads_per_block: usize,
}

unsafe impl Send for CudaMarkovHmacBruteForcer {}

impl CudaMarkovHmacBruteForcer {
    pub(crate) fn new(
        variant: HmacVariant,
        signing_input: &[u8],
        markov_table: &[u8],
    ) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("failed to create CUDA stream: {e}"))?;
        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".into());

        let (source, fn_name) = match variant {
            HmacVariant::Hs256 => (CUDA_SOURCE_HS256, "hs256_markov"),
            HmacVariant::Hs384 => (CUDA_SOURCE_HS512, "hs384_markov"),
            HmacVariant::Hs512 => (CUDA_SOURCE_HS512, "hs512_markov"),
        };
        let (module, func) = compile_and_load_single(&ctx, source, fn_name, &[])?;

        let max_threads_per_block = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map(|v| v as usize)
            .unwrap_or(1024);
        let threadgroup_width = 256usize.min(max_threads_per_block.max(1));
        let message_length =
            u32::try_from(signing_input.len()).context("JWT signing input too long")?;

        let params_size = match variant {
            HmacVariant::Hs256 => std::mem::size_of::<Hs256MarkovParams>(),
            HmacVariant::Hs384 | HmacVariant::Hs512 => std::mem::size_of::<Hs512MarkovParams>(),
        };
        let d_params = stream.alloc_zeros::<u8>(params_size)
            .map_err(|e| anyhow!("failed to alloc params buffer: {e}"))?;
        let d_message = stream.clone_htod(signing_input)
            .map_err(|e| anyhow!("failed to copy signing input to device: {e}"))?;
        let d_markov_table = stream.clone_htod(markov_table)
            .map_err(|e| anyhow!("failed to copy Markov table to device: {e}"))?;
        let d_result = stream.alloc_zeros::<u32>(1)
            .map_err(|e| anyhow!("failed to alloc result buffer: {e}"))?;

        Ok(Self {
            variant, ctx, stream, _module: module, func,
            d_params: UnsafeCell::new(d_params),
            d_message, d_markov_table,
            d_result: UnsafeCell::new(d_result),
            message_length, threadgroup_width, device_name, max_threads_per_block,
        })
    }

    fn write_markov_params(&self, target_signature: &[u8], candidate_count: u32,
                           pw_length: u32, threshold: u32, offset: u64) -> anyhow::Result<()> {
        let d_params = unsafe { &mut *self.d_params.get() };
        match self.variant {
            HmacVariant::Hs256 => {
                let mut target_words = [0u32; 8];
                for i in 0..8 {
                    let off = i * 4;
                    target_words[i] = u32::from_be_bytes(target_signature[off..off + 4].try_into().unwrap());
                }
                let params = Hs256MarkovParams {
                    target_signature: target_words, message_length: self.message_length,
                    candidate_count, pw_length, threshold, offset,
                };
                self.stream.memcpy_htod(bytes_of(&params), d_params)
                    .map_err(|e| anyhow!("failed to copy Markov params: {e}"))?;
            }
            HmacVariant::Hs384 | HmacVariant::Hs512 => {
                let wc = if self.variant == HmacVariant::Hs384 { 6 } else { 8 };
                let mut target_words = [0u64; 8];
                for i in 0..wc {
                    let off = i * 8;
                    target_words[i] = u64::from_be_bytes(target_signature[off..off + 8].try_into().unwrap());
                }
                let params = Hs512MarkovParams {
                    target_signature: target_words, message_length: self.message_length,
                    candidate_count, pw_length, threshold, offset,
                };
                self.stream.memcpy_htod(bytes_of(&params), d_params)
                    .map_err(|e| anyhow!("failed to copy Markov params: {e}"))?;
            }
        }
        Ok(())
    }
}

impl GpuMarkovBruteForcer for CudaMarkovHmacBruteForcer {
    fn device_name(&self) -> &str { &self.device_name }
    fn thread_execution_width(&self) -> usize { 32 }
    fn max_total_threads_per_threadgroup(&self) -> usize { self.max_threads_per_block }
    fn current_threadgroup_width(&self) -> usize { self.threadgroup_width }

    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()> {
        if requested == 0 { bail!("--threads-per-group must be > 0"); }
        if requested > self.max_threads_per_block {
            bail!("--threads-per-group {} exceeds max {}", requested, self.max_threads_per_block);
        }
        self.threadgroup_width = requested;
        Ok(())
    }

    fn encode_and_commit_markov(&self, target_data: &[u8], length: u32, threshold: u32,
                                 offset: u64, count: u32)
        -> anyhow::Result<(super::GpuCommandHandle, Duration, Duration)>
    {
        if count == 0 { bail!("cannot encode empty Markov batch"); }
        let prep_started = Instant::now();
        self.write_markov_params(target_data, count, length, threshold, offset)?;
        let d_result = unsafe { &mut *self.d_result.get() };
        self.stream.memcpy_htod(&[RESULT_NOT_FOUND_SENTINEL], d_result)
            .map_err(|e| anyhow!("failed to reset result: {e}"))?;
        let host_prep = prep_started.elapsed();

        let encode_started = Instant::now();
        let grid_dim = (count + self.threadgroup_width as u32 - 1) / self.threadgroup_width as u32;
        let config = LaunchConfig {
            grid_dim: (grid_dim.max(1), 1, 1),
            block_dim: (self.threadgroup_width as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_params = unsafe { &*self.d_params.get() };
        let d_result = unsafe { &mut *self.d_result.get() };
        unsafe {
            self.stream.launch_builder(&self.func)
                .arg(d_params).arg(&self.d_message)
                .arg(&self.d_markov_table).arg(d_result)
                .launch(config)?;
        }
        let command_encode = encode_started.elapsed();
        Ok((CudaCommandHandle, host_prep, command_encode))
    }

    fn wait_and_readback(&self, _handle: &super::GpuCommandHandle) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        let _ = self.stream.synchronize();
        let gpu_wait = gpu_wait_started.elapsed();
        let readback_started = Instant::now();
        let d_result = unsafe { &*self.d_result.get() };
        let result_vec = self.stream.clone_dtoh(d_result)
            .unwrap_or_else(|_| vec![RESULT_NOT_FOUND_SENTINEL]);
        let result = result_vec.first().copied().unwrap_or(RESULT_NOT_FOUND_SENTINEL);
        let _ = self.stream.synchronize();
        let result_readback = readback_started.elapsed();
        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL { None } else { Some(result) };
        (maybe_match, gpu_wait, result_readback)
    }

    fn autotune_markov(&mut self, target_data: &[u8], length: u32, threshold: u32) -> anyhow::Result<()> {
        const SAMPLE_COUNT: u32 = 16_384;
        let max_threads = self.max_threads_per_block;
        let mut candidates = vec![32, 64, 128, 256, 512, 1024];
        candidates.retain(|&v| v > 0 && v <= max_threads);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() { return Ok(()); }

        eprintln!("AUTOTUNE: benchmarking {} Markov candidates across {} block sizes", SAMPLE_COUNT, candidates.len());
        let mut best: Option<(usize, f64)> = None;
        for width in &candidates {
            self.threadgroup_width = *width;
            let (handle, _, _) = self.encode_and_commit_markov(target_data, length, threshold, 0, SAMPLE_COUNT)?;
            let (_, gpu_wait, _) = self.wait_and_readback(&handle);
            if gpu_wait.is_zero() { continue; }
            let rate = SAMPLE_COUNT as f64 / gpu_wait.as_secs_f64();
            match best {
                Some((_, best_rate)) if rate <= best_rate => {}
                _ => best = Some((*width, rate)),
            }
        }
        if let Some((best_width, best_rate)) = best {
            self.threadgroup_width = best_width;
            eprintln!("AUTOTUNE: selected --threads-per-group {} ({}/s sample)", best_width, format_human_count(best_rate));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CudaMarkovAesKwBruteForcer — Markov + AES Key Wrap
// ---------------------------------------------------------------------------

/// CUDA compute backend for Markov chain + AES Key Wrap cracking.
///
/// Same fused-kernel approach as `CudaMarkovHmacBruteForcer` but for JWE
/// tokens using A128KW/A192KW/A256KW. The shader is compiled with
/// `-DAES_KEY_BYTES=N` for compile-time specialisation per variant.
///
/// The encrypted key is uploaded once at construction. Subsequent dispatches
/// only update the params struct (length, offset, count, threshold).
pub(crate) struct CudaMarkovAesKwBruteForcer {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<cudarc::driver::CudaModule>,
    func: CudaFunction,
    d_params: UnsafeCell<CudaSlice<u8>>,
    d_encrypted_key: CudaSlice<u8>,
    d_markov_table: CudaSlice<u8>,
    d_result: UnsafeCell<CudaSlice<u32>>,
    encrypted_key_len: u32,
    n_blocks: u32,
    threadgroup_width: usize,
    device_name: String,
    max_threads_per_block: usize,
}

unsafe impl Send for CudaMarkovAesKwBruteForcer {}

impl CudaMarkovAesKwBruteForcer {
    pub(crate) fn new(variant: AesKwVariant, encrypted_key: &[u8], n_blocks: usize,
                      markov_table: &[u8]) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
        let stream = ctx.new_stream()
            .map_err(|e| anyhow!("failed to create CUDA stream: {e}"))?;
        let device_name = ctx.name().unwrap_or_else(|_| "unknown CUDA device".into());
        let key_bytes_define = format!("-DAES_KEY_BYTES={}", variant.key_bytes());
        let (module, func) = compile_and_load_single(&ctx, CUDA_SOURCE_AESKW, "aeskw_markov", &[key_bytes_define])?;
        let max_threads_per_block = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map(|v| v as usize).unwrap_or(1024);
        let threadgroup_width = 256usize.min(max_threads_per_block.max(1));
        let encrypted_key_len = u32::try_from(encrypted_key.len()).context("encrypted_key too long")?;
        let n_blocks_u32 = u32::try_from(n_blocks).context("n_blocks exceeds u32")?;

        let d_params = stream.alloc_zeros::<u8>(std::mem::size_of::<AesKwMarkovParams>())
            .map_err(|e| anyhow!("failed to alloc params: {e}"))?;
        let d_encrypted_key = stream.clone_htod(encrypted_key)
            .map_err(|e| anyhow!("failed to copy encrypted_key: {e}"))?;
        let d_markov_table = stream.clone_htod(markov_table)
            .map_err(|e| anyhow!("failed to copy Markov table: {e}"))?;
        let d_result = stream.alloc_zeros::<u32>(1)
            .map_err(|e| anyhow!("failed to alloc result: {e}"))?;

        Ok(Self {
            ctx, stream, _module: module, func,
            d_params: UnsafeCell::new(d_params), d_encrypted_key, d_markov_table,
            d_result: UnsafeCell::new(d_result),
            encrypted_key_len, n_blocks: n_blocks_u32,
            threadgroup_width, device_name, max_threads_per_block,
        })
    }
}

impl GpuMarkovBruteForcer for CudaMarkovAesKwBruteForcer {
    fn device_name(&self) -> &str { &self.device_name }
    fn thread_execution_width(&self) -> usize { 32 }
    fn max_total_threads_per_threadgroup(&self) -> usize { self.max_threads_per_block }
    fn current_threadgroup_width(&self) -> usize { self.threadgroup_width }

    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()> {
        if requested == 0 { bail!("--threads-per-group must be > 0"); }
        if requested > self.max_threads_per_block {
            bail!("--threads-per-group {} exceeds max {}", requested, self.max_threads_per_block);
        }
        self.threadgroup_width = requested;
        Ok(())
    }

    fn encode_and_commit_markov(&self, _target_data: &[u8], length: u32, threshold: u32,
                                 offset: u64, count: u32)
        -> anyhow::Result<(super::GpuCommandHandle, Duration, Duration)>
    {
        if count == 0 { bail!("cannot encode empty Markov batch"); }
        let prep_started = Instant::now();
        let params = AesKwMarkovParams {
            encrypted_key_len: self.encrypted_key_len, n_blocks: self.n_blocks,
            candidate_count: count, pw_length: length, threshold, _pad: 0, offset,
        };
        let d_params = unsafe { &mut *self.d_params.get() };
        self.stream.memcpy_htod(bytes_of(&params), d_params)
            .map_err(|e| anyhow!("failed to copy params: {e}"))?;
        let d_result = unsafe { &mut *self.d_result.get() };
        self.stream.memcpy_htod(&[RESULT_NOT_FOUND_SENTINEL], d_result)
            .map_err(|e| anyhow!("failed to reset result: {e}"))?;
        let host_prep = prep_started.elapsed();

        let encode_started = Instant::now();
        let grid_dim = (count + self.threadgroup_width as u32 - 1) / self.threadgroup_width as u32;
        let config = LaunchConfig {
            grid_dim: (grid_dim.max(1), 1, 1),
            block_dim: (self.threadgroup_width as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_params = unsafe { &*self.d_params.get() };
        let d_result = unsafe { &mut *self.d_result.get() };
        unsafe {
            self.stream.launch_builder(&self.func)
                .arg(d_params).arg(&self.d_encrypted_key)
                .arg(&self.d_markov_table).arg(d_result)
                .launch(config)?;
        }
        let command_encode = encode_started.elapsed();
        Ok((CudaCommandHandle, host_prep, command_encode))
    }

    fn wait_and_readback(&self, _handle: &super::GpuCommandHandle) -> (Option<u32>, Duration, Duration) {
        let gpu_wait_started = Instant::now();
        let _ = self.stream.synchronize();
        let gpu_wait = gpu_wait_started.elapsed();
        let readback_started = Instant::now();
        let d_result = unsafe { &*self.d_result.get() };
        let result_vec = self.stream.clone_dtoh(d_result)
            .unwrap_or_else(|_| vec![RESULT_NOT_FOUND_SENTINEL]);
        let result = result_vec.first().copied().unwrap_or(RESULT_NOT_FOUND_SENTINEL);
        let _ = self.stream.synchronize();
        let result_readback = readback_started.elapsed();
        let maybe_match = if result == RESULT_NOT_FOUND_SENTINEL { None } else { Some(result) };
        (maybe_match, gpu_wait, result_readback)
    }

    fn autotune_markov(&mut self, _target_data: &[u8], length: u32, threshold: u32) -> anyhow::Result<()> {
        const SAMPLE_COUNT: u32 = 16_384;
        let max_threads = self.max_threads_per_block;
        let mut candidates = vec![32, 64, 128, 256, 512, 1024];
        candidates.retain(|&v| v > 0 && v <= max_threads);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() { return Ok(()); }

        eprintln!("AUTOTUNE: benchmarking {} Markov candidates across {} block sizes", SAMPLE_COUNT, candidates.len());
        let mut best: Option<(usize, f64)> = None;
        let dummy_target: &[u8] = &[];
        for width in &candidates {
            self.threadgroup_width = *width;
            let (handle, _, _) = self.encode_and_commit_markov(dummy_target, length, threshold, 0, SAMPLE_COUNT)?;
            let (_, gpu_wait, _) = self.wait_and_readback(&handle);
            if gpu_wait.is_zero() { continue; }
            let rate = SAMPLE_COUNT as f64 / gpu_wait.as_secs_f64();
            match best {
                Some((_, best_rate)) if rate <= best_rate => {}
                _ => best = Some((*width, rate)),
            }
        }
        if let Some((best_width, best_rate)) = best {
            self.threadgroup_width = best_width;
            eprintln!("AUTOTUNE: selected --threads-per-group {} ({}/s sample)", best_width, format_human_count(best_rate));
        }
        Ok(())
    }
}
