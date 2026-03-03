mod backend;

#[cfg(target_os = "macos")]
pub(crate) mod metal;

#[cfg(target_os = "linux")]
mod cuda;

pub(crate) use backend::{GpuBruteForcer, HmacVariant};

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
