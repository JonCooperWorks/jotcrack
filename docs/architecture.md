# Jotcrack Architecture: GPU Abstraction & Pipeline

Jotcrack is a GPU-accelerated JWT/JWE secret cracker for macOS (Metal) and Linux (CUDA). Two attack modes: **wordlist** — processes massive wordlists (100+ GB) by memory-mapping files, parsing in parallel, and double-buffering GPU dispatch; **Markov** — trains a character-level bigram model and enumerates candidates directly on the GPU with fused generate+hash kernels.

**Wordlist performance:** 793M candidates/s for HS256 on NVIDIA RTX PRO 6000 (CUDA), 446M/s on Apple M4 Max (Metal). 145M/s for A128KW on RTX PRO 6000, 120M/s on M4 Max.

**Markov performance (M4 Max, T=30):** 394M/s for HS256, 128M/s for A128KW. Expands guess space ~47,000× beyond rockyou.txt while remaining feasible in minutes.

---

## End-to-End Data Flow

```
  Wordlist (disk)
       │
       ▼
  ┌──────────────┐
  │  Memory-Map   │  mmap() — zero-copy, OS manages paging
  └──────┬───────┘
         │
         ├──────────────────────────────────────────────┐
         │  macOS (Metal)                               │  Linux (CUDA)
         │                                              │
         ▼                                              ▼
  ┌──────────────────────┐                  ┌─────────────────────────┐
  │  Parser Threads (N)  │                  │  Worker Threads (N)     │
  │  (parser.rs)         │                  │  (producer.rs)          │
  │  Scan ~16 MiB chunks │                  │  Scan newline-aligned   │
  │  for line boundaries │                  │  regions with memchr    │
  └──────┬───────────────┘                  │  Write offset/length    │
         │                                  │  metadata directly to   │
         ▼                                  │  pinned host memory     │
  ┌──────────────────┐                      └──────┬──────────────────┘
  │  Planner Thread  │                             │
  │  (producer.rs)   │                             │
  │  Group lines     │                             │
  │  into BatchPlans │                             │
  └──────┬───────────┘                             │
         │                                         │
         ▼                                         │
  ┌──────────────────────┐                         │
  │  Coordinator         │                         │
  │  (producer.rs)       │                         │
  │  Inline-pack plans   │                         │
  │  into GPU buffers    │                         │
  └──────┬───────────────┘                         │
         │                                         │
         └────────────────┬────────────────────────┘
                          │
                          ▼
  ┌──────────────────────────────┐
  │  Consumer Loop (runner.rs)   │  Double-buffered dispatch:
  │                              │    While GPU runs batch N,
  │                              │    CPU prepares batch N+1
  └──────┬───────────────────────┘
         │
         ▼
  ┌──────────────────────────────┐
  │  GPU Kernel                  │  One thread per candidate:
  │  (Metal / CUDA)              │    compute hash/unwrap → compare → atomic write
  └──────────────────────────────┘
```

---

## 1. GPU Abstraction Layer (`src/gpu/mod.rs`)

The GPU abstraction defines a **platform-agnostic trait** that decouples the cracking algorithm from the GPU backend. Two backends exist today: Apple Metal (macOS) and NVIDIA CUDA (Linux).

### The `GpuBruteForcer` Trait

```rust
pub(crate) trait GpuBruteForcer {
    fn device(&self) -> &GpuDevice;
    fn device_name(&self) -> &str;
    fn thread_execution_width(&self) -> usize;
    fn max_total_threads_per_threadgroup(&self) -> usize;
    fn current_threadgroup_width(&self) -> usize;
    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()>;
    fn encode_and_commit(
        &self,
        target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)>;
    fn wait_and_readback(
        &self,
        handle: &GpuCommandHandle,
    ) -> (Option<u32>, Duration, Duration);
    fn autotune_threadgroup_width(
        &mut self,
        target_signature: &[u8],
        batch: &WordBatch,
    ) -> anyhow::Result<()>;
}
```

| Method | Purpose |
|--------|---------|
| `device()` | Return the underlying GPU device handle (for buffer allocation by the producer). |
| `device_name()` | Human-readable GPU name (e.g. "Apple M4 Max", "NVIDIA RTX PRO 6000"). |
| `thread_execution_width()` | SIMD lane count — Metal thread execution width or CUDA warp size (always 32). |
| `max_total_threads_per_threadgroup()` | Hardware maximum threads per threadgroup (Metal) or block (CUDA). |
| `set_threadgroup_width()` | Validate and apply a threadgroup/block width override. Returns an error for invalid values. |
| `encode_and_commit()` | Encode GPU commands and submit asynchronously. Returns a command handle and (host_prep, encode) timing data. The GPU begins executing immediately. |
| `wait_and_readback()` | Block until the GPU finishes, then read the result buffer. Returns `Some(match_index)` if a candidate matched, or `None`. Also returns (gpu_wait, readback) durations. |
| `autotune_threadgroup_width()` | Benchmark different threadgroup/block widths on the first batch to find the optimal configuration for the current device. |

### The `GpuMarkovBruteForcer` Trait

The Markov mode uses a separate trait because there's no `WordBatch` — candidates are generated on the GPU from keyspace indices:

```rust
pub(crate) trait GpuMarkovBruteForcer {
    fn device_name(&self) -> &str;
    fn current_threadgroup_width(&self) -> usize;
    fn set_threadgroup_width(&mut self, requested: usize) -> anyhow::Result<()>;
    fn encode_and_commit_markov(
        &self, target: &[u8], length: u32, offset: u64, count: u32,
    ) -> anyhow::Result<(GpuCommandHandle, Duration, Duration)>;
    fn wait_and_readback(&self, handle: &GpuCommandHandle) -> (Option<u32>, Duration, Duration);
    fn autotune_markov(
        &mut self, target_data: &[u8], length: u32, threshold: u32,
    ) -> anyhow::Result<()>;
}
```

| Method | Purpose |
|--------|---------|
| `encode_and_commit_markov()` | Dispatch a batch of `count` candidates starting at `offset` in the keyspace for a given `length`. Each GPU thread decodes its index, walks the Markov chain, hashes, and compares. |
| `autotune_markov()` | Benchmark threadgroup/block widths on a small 16K-candidate Markov dispatch. |

### Platform Type Aliases

```rust
// macOS (Metal)
#[cfg(target_os = "macos")]
pub(crate) type GpuDevice = ::metal::Device;
pub(crate) type GpuBuffer = ::metal::Buffer;
pub(crate) type GpuCommandHandle = ::metal::CommandBuffer;

// Linux (CUDA)
#[cfg(target_os = "linux")]
pub(crate) type GpuBuffer = cuda::CudaBuffer;
pub(crate) type GpuDevice = cuda::CudaDeviceHandle;
pub(crate) type GpuCommandHandle = cuda::CudaCommandHandle;
```

These aliases let all code outside `gpu/metal/` and `gpu/cuda/` remain platform-independent. The `alloc_shared_buffer()` and `buffer_host_ptr()` helper functions in `gpu/mod.rs` provide platform-gated wrappers for buffer allocation and host-pointer access.

### Algorithm Enums

The `CrackVariant` enum is the top-level discriminant that routes through the entire system:

```rust
pub(crate) enum CrackVariant {
    Hmac(HmacVariant),        // JWT HMAC-SHA path (HS256, HS384, HS512)
    JweAesKw(AesKwVariant),   // JWE AES Key Wrap path (A128KW, A192KW, A256KW)
}
```

`HmacVariant` and `AesKwVariant` carry algorithm-specific metadata (signature length, key size, labels for output).

---

## 2. Metal Backend (`src/gpu/metal/mod.rs`)

Two structs implement `GpuBruteForcer` on macOS:

### MetalBruteForcer (HMAC-SHA JWT)

Holds compiled Metal pipelines, persistent GPU buffers, and the current threadgroup width:

```rust
pub(crate) struct MetalBruteForcer {
    variant: HmacVariant,
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline_mixed: metal::ComputePipelineState,      // General path (all key lengths)
    pipeline_short_keys: metal::ComputePipelineState, // Fast path (keys < block size)
    params_buf: metal::Buffer,    // Algorithm parameters (target signature, message length, etc.)
    msg_buf: metal::Buffer,       // JWT signing input (header.payload)
    result_buf: metal::Buffer,    // Single u32: match index or u32::MAX sentinel
    threadgroup_width: usize,
}
```

**Two pipelines** are compiled from the same `.metal` source: `pipeline_short_keys` skips the key-hashing step when all candidates in a batch are shorter than the SHA block size (64 bytes for SHA-256, 128 bytes for SHA-512). The `active_pipeline_for_view()` method selects the right one based on `max_word_len` in the current batch.

### MetalAesKwBruteForcer (JWE AES Key Wrap)

Similar structure, but targets AES Key Wrap (RFC 3394) instead of HMAC:

```rust
pub(crate) struct MetalAesKwBruteForcer {
    variant: AesKwVariant,
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline_mixed: metal::ComputePipelineState,
    pipeline_short_keys: metal::ComputePipelineState,
    params_buf: metal::Buffer,
    encrypted_key_buf: metal::Buffer,  // RFC 3394 wrapped CEK
    result_buf: metal::Buffer,
    threadgroup_width: usize,
    n: u32,  // Number of 64-bit CEK blocks
}
```

### MetalMarkovHmacBruteForcer / MetalMarkovAesKwBruteForcer

Markov mode backends for Metal. Each holds a single compiled pipeline (one Markov entry point per algorithm) plus a `markov_table_buf` containing the pre-trained lookup table uploaded once at init. No word batch buffers are needed — candidates are generated in-kernel from keyspace indices.

The dispatch loop passes `(length, offset, count)` parameters. Each GPU thread decodes `batch_offset + gid` into per-position rank selections via repeated `idx % threshold` / `idx / threshold`, walks the Markov chain in registers, and immediately hashes the result.

### Buffer Slot Assignments (Metal)

Both Metal backends bind six Metal buffers to fixed slot indices:

| Slot | Buffer | Contents |
|------|--------|----------|
| 0 | `params_buf` | Algorithm parameters struct |
| 1 | `msg_buf` / `encrypted_key_buf` | JWT signing input or wrapped CEK |
| 2 | `word_bytes_buf` | Packed candidate bytes (from WordBatch) |
| 3 | `word_offsets_buf` | Per-candidate byte offsets (u32) |
| 4 | `word_lengths_buf` | Per-candidate byte lengths (u16) |
| 5 | `result_buf` | Match result (u32::MAX = no match) |

---

## 3. CUDA Backend (`src/gpu/cuda/mod.rs`)

Two structs implement `GpuBruteForcer` on Linux:

### CudaBruteForcer (HMAC-SHA JWT)

Uses NVRTC to compile embedded `.cu` kernel sources to PTX at startup. The wordlist mmap is uploaded to GPU VRAM once at construction, so per-batch dispatch only transfers offset/length metadata:

- **Device memory**: Pre-allocated to maximum batch sizes and reused across dispatches.
- **Pinned host memory**: Host-side batch buffers use `cuMemHostAlloc` (page-locked memory), enabling DMA transfers at full PCIe bandwidth without the driver's internal staging copy.
- **Interior mutability via `UnsafeCell`**: The `GpuBruteForcer` trait's `encode_and_commit` method takes `&self`, but CUDA dispatch requires mutable access to device buffers (for `memcpy_htod`). Safety is guaranteed by the runner's double-buffer pattern — only one dispatch cycle is in flight at a time.

### CudaAesKwBruteForcer (JWE AES Key Wrap)

Same architecture as `CudaBruteForcer`, but for AES Key Wrap. The kernel source is compiled three times with different `AES_KEY_BYTES` defines, producing specialised kernels for each key size.

### CudaMarkovHmacBruteForcer / CudaMarkovAesKwBruteForcer

CUDA Markov backends. Same fused-kernel architecture as Metal — each thread decodes a keyspace index, walks the Markov chain, hashes, and compares. The Markov table is uploaded to global memory with `const __restrict__` (L2 cache handles it; the table may exceed the 64 KB `__constant__` limit). Uses `UnsafeCell` for interior mutability (same pattern as `CudaBruteForcer`).

### Two-Kernel Strategy (Both Backends)

Each backend (Metal and CUDA) exposes two kernel entry points per algorithm:

- **`mixed`** — handles keys of any length.
- **`short_keys`** — optimised fast path for short keys (skips the key-hashing step when all candidates are shorter than the hash block size).

### Memory Model Differences

| Aspect | Metal (macOS) | CUDA (Linux) |
|--------|--------------|--------------|
| **Memory** | Unified — CPU and GPU share physical RAM | Discrete — explicit host→device copies across PCIe |
| **Batch buffers** | `StorageModeShared` — zero-copy | Pinned host memory + `memcpy_htod` DMA |
| **Wordlist data** | Candidate bytes copied into batch buffers by packer threads | Entire mmap uploaded to GPU VRAM once at startup; batches carry only offset/length metadata |
| **Kernel compilation** | Metal runtime shader compilation at pipeline creation | NVRTC compiles `.cu` to PTX at startup (~200–500ms); compute capability is auto-detected |

---

## 4. Batch System (`src/batch.rs`)

### The Problem

The GPU needs flat, contiguous buffers. A naive `Vec<String>` causes per-candidate allocation overhead that dominates at 400M+ candidates/second.

### The Solution: Struct-of-Arrays Packing

`WordBatch` stores candidates in three parallel arrays packed into pre-allocated GPU buffers:

```
Candidates: "alpha", "be", "cat"

word_bytes:   [a][l][p][h][a][b][e][c][a][t]   ← concatenated, no separators
word_offsets: [0, 5, 7]                          ← byte offset of each candidate
word_lengths: [5, 2, 3]                          ← byte length of each candidate
```

```rust
pub(crate) struct WordBatch {
    candidate_index_base: u64,       // Global index of first candidate in this batch
    word_bytes_buf: GpuBuffer,       // GPU buffer — visible to CPU for writing
    word_offsets_buf: GpuBuffer,
    word_lengths_buf: GpuBuffer,
    word_bytes_ptr: *mut u8,         // Cached raw pointers (avoids ObjC message sends on Metal)
    word_offsets_ptr: *mut u32,
    word_lengths_ptr: *mut u16,
    candidate_count: usize,
    word_bytes_len: usize,
    max_word_len: u16,
}
```

On macOS, `GpuBuffer` is `metal::Buffer` with `StorageModeShared` (unified memory). On Linux, `GpuBuffer` is a pinned host memory allocation (`CudaBuffer`) backed by `cuMemHostAlloc`. In both cases, the cached raw pointers let the packer write candidate data without per-access FFI overhead.

### Batch Capacity Limits

- **MAX_CANDIDATES_PER_BATCH**: 6,182,240 (~6.2M candidates)
- **MAX_WORD_BYTES_PER_BATCH**: 32 MiB of packed candidate bytes
- **APPROX_WORD_BATCH_BUFFER_BYTES**: ~80 MiB total memory per batch

### Object Pool Recycling

Allocating GPU buffers involves FFI calls and kernel VM operations. Instead of allocating/freeing per batch, jotcrack allocates a fixed number of batches (`pipeline_depth`, default 10) and recycles them:

1. Consumer finishes with a batch → sends it back via `producer.recycle(batch)`
2. Producer calls `batch.reset_for_reuse()` → clears logical cursors, keeps allocation
3. Packer thread fills the batch again with new candidates

This amortizes allocation cost to near zero.

### Zero-Copy GPU Dispatch (macOS)

`WordBatch` allocates Metal `StorageModeShared` buffers, meaning the CPU and GPU share the same physical memory on Apple Silicon. When the packer writes candidate bytes into `word_bytes_ptr`, the GPU reads them directly — no copy or transfer step.

### DMA GPU Dispatch (Linux)

On CUDA, the wordlist mmap is uploaded to GPU VRAM once at construction. Each batch's offset/length metadata is written to pinned host memory buffers, then transferred to device memory via `memcpy_htod` at dispatch time. This achieves full PCIe bandwidth without intermediate staging copies.

`as_dispatch_view()` (macOS) returns a `DispatchBatchView` — a zero-copy borrow of the batch's buffers and metadata — that gets passed to `gpu.encode_and_commit()`. On Linux, `encode_and_commit()` reads directly from the `WordBatch`.

---

## 5. Parser System (`src/parser.rs`)

### The Problem

A 112 GB wordlist cannot be loaded into memory. Even streaming line-by-line is too slow for 400M+ candidates/second.

### The Solution: Mmap + Parallel Chunk Parsing

**Step 1: Memory-map the file.** The OS maps disk pages into virtual memory on demand. No data is copied until accessed.

**Step 2: Divide into chunks.** The mmap is split into ~16 MiB chunks. Each chunk is processed independently by a parser thread.

**Step 3: Scan for line boundaries.** Each parser thread uses `memchr` to find newline characters within its chunk, recording per-line offsets and lengths:

```rust
pub(crate) struct MmapChunk {
    chunk_start: usize,           // Byte offset in mmap
    chunk_offsets_rel: Vec<u32>,  // Per-line offsets relative to chunk_start
    chunk_lengths: Vec<u16>,      // Per-line byte lengths
}
```

**Step 4: Plan batches.** The planner thread consumes parsed chunks and groups lines into `BatchPlan`s that respect the GPU batch limits. An optimization checks entire blocks (~4000 lines) at once instead of per-line, reducing planning overhead by orders of magnitude:

```rust
pub(crate) fn batch_shape_can_fit_block(
    candidate_count: usize,
    word_bytes_len: usize,
    block_candidates: usize,
    block_bytes: usize,
) -> bool
```

**Step 5: Pack into GPU buffers.** Packer threads receive a `BatchPlan` and call `pack_batch_plan_into_batch()`, which copies candidate bytes directly from the mmap into GPU buffers via `push_segment_bulk()`.

> **Note:** On Linux (CUDA), steps 3–5 are replaced by a simpler direct-write path — see [Producer Pipeline](#6-producer-pipeline-srcproducerrs) below.

---

## 6. Producer Pipeline (`src/producer.rs`)

The producer orchestrates wordlist parsing and batch preparation. Its implementation differs significantly between platforms.

### macOS: Multi-Stage Pipeline

On macOS (Metal, unified memory), the producer uses a three-stage pipeline:

```
┌──────────────────────────────────────────────────┐
│              Producer Thread                      │
│                                                  │
│  ┌──────────┐    ┌───────────┐                   │
│  │ Planner  │───→│ Job Queue │                   │
│  │          │    │           │                   │
│  └──────────┘    └─────┬─────┘                   │
│                        │                         │
│                        ▼                         │
│                  ┌───────────┐                   │
│                  │Coordinator│                   │
│                  │(inline    │                   │
│                  │ pack)     │                   │
│                  └─────┬─────┘                   │
│                        │                         │
│  ┌──────────────┐      │                         │
│  │ Batch Pool   │◄── recycle ────────────────┐   │
│  │ (recycled    │                            │   │
│  │  WordBatches)│                            │   │
│  └──────────────┘                            │   │
│                        │                     │   │
│                        ▼                     │   │
│              tx ─────────────────────── rx   │   │
└──────────────────────────────────────────────┘   │
                                          │        │
                                          ▼        │
                                Consumer (runner.rs)
```

Parser threads scan the mmap in parallel, a planner thread groups lines into `BatchPlan` objects, and a coordinator thread inline-packs them into `WordBatch` GPU buffers (Metal shared memory). The coordinator pulls recycled batches from the pool, packs the next plan, and sends the filled batch downstream.

### Linux: Parallel Direct-Write

On Linux (CUDA, discrete GPU), the entire mmap is uploaded to GPU VRAM once at startup. "Packing" is just writing offset/length metadata to pinned host memory — no candidate bytes need to be copied per batch.

Multiple worker threads each scan a newline-aligned region of the mmap with SIMD `memchr` and write offset/length metadata directly to GPU batch buffers. This eliminates all intermediate data structures (`ParsedChunk`, `BatchPlan`, block summaries) and avoids the coordinator bottleneck:

```
┌──────────────────────────────────────────────┐
│           Worker Threads (N)                  │
│                                              │
│  Worker 1: scan region → stage locally →     │
│            flush offsets/lengths to batch     │
│  Worker 2: scan region → stage locally →     │
│            flush offsets/lengths to batch     │
│  Worker N: ...                               │
│                                              │
│  Each worker:                                │
│    1. Receives an empty batch from the pool  │
│    2. Scans its mmap region with memchr      │
│    3. Stages offsets/lengths in L1-cached     │
│       local buffers                          │
│    4. Bulk-flushes to pinned host memory     │
│       when a batch boundary is hit           │
│    5. Sends the filled batch to the consumer │
│    6. Last worker to finish sends EOF        │
└──────────────────────────────────────────────┘
```

### Bounded Channel Backpressure

The producer-to-consumer channel uses `sync_channel(pipeline_depth)`. This prevents the producer from parsing the entire wordlist into memory while the GPU is still processing early batches. Memory usage is bounded to `pipeline_depth × ~80 MiB`.

### ProducerMessage

```rust
pub(crate) enum ProducerMessage {
    Batch { batch: WordBatch, build_time, plan_time, pack_time, parser_stats },
    Eof { parser_stats },
    Error(String),
}
```

---

## 7. Runner & Double-Buffered Dispatch (`src/runner.rs`)

The runner ties everything together with a generic dispatch loop:

```rust
fn run_gpu_crack<B: GpuBruteForcer>(
    variant: CrackVariant,
    mut gpu: B,
    target_data: &[u8],
    args: WordlistArgs,
    #[cfg(target_os = "linux")] mmap_source: &[u8],
) -> anyhow::Result<bool>
```

On Linux, the runner also opens its own mmap of the wordlist file (sharing OS page cache pages with the producer's mmap) for two purposes: uploading the wordlist to GPU VRAM, and reconstructing the matching candidate after the GPU finds a hit.

### Double-Buffering Pattern

The key to high throughput is **never letting the GPU go idle**:

```
Time ──────────────────────────────────────────────────►

CPU:  [prepare batch 1] [prepare batch 2] [prepare batch 3] ...
GPU:                     [process batch 1] [process batch 2] ...
```

The consumer loop:

1. **Receive** the next batch from the producer (`producer.recv()`)
2. **Wait** for the previous GPU dispatch to finish (`gpu.wait_and_readback()`)
3. **Check** for a match — if found, report and exit
4. **Recycle** the finished batch (`producer.recycle(batch)`)
5. **Autotune** threadgroup width on the first batch (one-time)
6. **Dispatch** the new batch to the GPU (`gpu.encode_and_commit()`)
7. **Loop** — the GPU now executes while the CPU returns to step 1

This means the GPU is always working on one batch while the CPU prepares the next. Neither processor waits for the other (under ideal conditions).

### Token Parsing (`src/jwt.rs`)

Before the pipeline starts, the runner parses the input token:

- **JWT (3-part):** Validates structure, decodes base64url, extracts signing input (`header.payload`) and target signature
- **JWE (5-part):** Validates structure, extracts the encrypted key (the only part needed for Key Wrap cracking)
- **Auto-detection:** `detect_token_variant()` inspects the header's `alg` field to route to the correct `CrackVariant`

---

## 8. GPU Shader Kernels

Both Metal and CUDA backends implement the same algorithmic structure. The CUDA ports preserve the same optimizations as the Metal originals.

### HMAC-SHA Kernels (`hs256_wordlist.{metal,cu}`, `hs512_wordlist.{metal,cu}`)

Each GPU thread (identified by `gid`) processes one candidate:

1. Read candidate bytes using `word_bytes[offsets[gid]..offsets[gid]+lengths[gid]]`
2. Derive the HMAC key (zero-pad if short, SHA-hash if longer than block size)
3. Compute HMAC: `SHA(key⊕opad ‖ SHA(key⊕ipad ‖ message))`
4. Compare against `target_signature` (precomputed as native-endian words on the host)
5. On match: `atomic_fetch_min(result_buf, gid)` / `atomicMin(result_buf, gid)` — lowest matching index wins

**Optimizations:**
- Rolling message schedule (`w[16]` instead of `w[64]`) reduces register pressure
- Flat HMAC implementation avoids streaming context overhead
- Host pre-converts signature to native-endian u32/u64 words
- SHA-256 constant K values cached across SIMD groups/warps

### AES Key Wrap Kernel (`aeskw_wordlist.{metal,cu}`)

Compiled three times with different `#define AES_KEY_BYTES` values (16, 24, 32):

1. Derive AES key from candidate (zero-pad if short, SHA-256 + truncate if long)
2. Expand AES key schedule (10/12/14 rounds depending on key size)
3. RFC 3394 unwrap: 6×n AES-ECB decryptions
4. Check integrity value == `0xA6A6A6A6A6A6A6A6`
5. On match: atomic write to result buffer

**Optimizations:**
- Compile-time specialization lets the compiler unroll key-size-dependent loops
- S-boxes in `constant` (Metal) / `__constant__` (CUDA) address space (hardware-cached, broadcast across SIMD groups/warps)
- GF(2^8) arithmetic via `xtime` helper avoids extra lookup tables

### Markov Kernels (fused entry points in the same `.metal`/`.cu` files)

Each kernel file contains both wordlist and Markov entry points. The Markov kernels reuse the same hash/unwrap primitives but replace the wordlist-lookup candidate loading with in-register Markov chain generation:

1. Decode global keyspace index via repeated `idx % T` / `idx / T`
2. Walk the Markov chain: `candidate[pos] = table[(pos*256 + prev)*T + rank]`
3. Load key words (using a `thread`-address-space variant of `load_key_words` on Metal — the only new helper needed)
4. Hash/unwrap and compare — same as wordlist kernels from this point

The Markov table (200 KB for T=50, L=16) is bound as a separate buffer. On Metal it uses `device const` address space; on CUDA it uses global memory with `const __restrict__` for L2 caching.

### CUDA-Specific Details

- **Unified address space**: CUDA's flat global memory model eliminates the need for Metal-style duplicate helper functions across `device`/`constant`/`threadgroup` address spaces.
- **`__constant__` memory**: SHA round constants and AES S-box tables use CUDA's dedicated 64KB constant cache, which broadcasts reads to all 32 threads in a warp simultaneously.
- **NVRTC runtime compilation**: Kernel sources are compiled to PTX at startup via NVIDIA's Runtime Compilation library. Compute capability is auto-detected for architecture-specific codegen.
- **Markov table in global memory**: The table may exceed the 64 KB `__constant__` limit, so it uses global memory with `const __restrict__` — L2 cache handles the repeated lookups efficiently.

---

## 9. Statistics & Timing (`src/stats.rs`)

Every batch records a detailed timing breakdown:

| Metric | What It Measures |
|--------|-----------------|
| `plan_time` | Planner grouping lines into a BatchPlan |
| `pack_time` | Packer copying bytes into GPU buffers |
| `host_prep` | Writing params, resetting sentinel |
| `command_encode` | GPU API calls to encode commands |
| `gpu_wait` | Actual GPU compute time |
| `result_readback` | Reading the result buffer |
| `consumer_idle_wait` | Time consumer blocks waiting for the producer |

Windowed rate reporting prints throughput every second. The final report shows both end-to-end and GPU-only throughput, highlighting whether the CPU or GPU is the bottleneck.

---

## 10. Markov Chain Mode (`src/markov.rs`, `src/runner.rs`)

The Markov mode bypasses the entire wordlist pipeline (parser, producer, batch system, double-buffering) and instead enumerates candidates directly on the GPU.

### Markov Model (Order-1 Bigram)

**Training** (`train_from_file`): Single pass over the wordlist counting `freq[position][prev_char][next_char]`. Position 0 uses `prev_char=0` as the start-of-word sentinel. The raw frequency counts are stored as a flat `u32[max_positions][256][256]` array.

**Auto-caching**: On first run, raw frequency counts are saved to `<wordlist>.markov` (4 MB for max_len=16). Subsequent runs load the cache in ~0ms. Changing `--threshold` or `--max-len` re-derives the ranked table from cached counts — no retraining needed. `--retrain` forces a re-scan.

**GPU table**: `ranked_table(threshold)` sorts each (position, prev_char) bucket by frequency, keeps the top-T characters, and packs them into a flat `u8` array indexed as `table[(pos * 256 + prev_char) * T + rank]`. For T=50, max_len=16: 200 KB — fits easily in GPU cache.

### Fused Generate+Hash Kernels

Each GPU thread generates and evaluates one candidate entirely in registers:

```
gid → global keyspace index
prev = 0  (start-of-word sentinel)
for each position:
    rank = idx % threshold
    idx /= threshold
    candidate[pos] = table[(pos*256 + prev)*T + rank]
    prev = candidate[pos]
→ hash candidate → compare with target → atomic write on match
```

No intermediate candidate buffer, no memory traffic for generated passwords. The only GPU memory reads are the Markov table lookups (cached in L2/constant cache) and the algorithm-specific constants (SHA round constants, AES S-boxes).

### Dispatch Loop

Unlike wordlist mode's double-buffered producer-consumer pipeline, Markov dispatch is a simple synchronous loop:

```rust
for length in min_len..=max_len {
    let total = threshold.pow(length);
    for offset in (0..total).step_by(6_000_000) {
        let count = min(6M, total - offset);
        gpu.encode_and_commit_markov(target, length, offset, count);
        gpu.wait_and_readback();
    }
}
```

Host prep is ~32 bytes (params struct), taking microseconds — so double-buffering would add complexity without meaningful benefit.

### Progress Reporting

Because the total keyspace is known upfront (`T^min_len + T^(min_len+1) + ... + T^max_len`), Markov mode reports real-time progress percentage and ETA alongside the windowed rate. The wordlist path passes `None` for `total_keyspace` since the wordlist size isn't known in the streaming pipeline.

### Result Reconstruction

On a GPU match, the CPU calls `model.candidate_from_index(length, offset + local_index)` to recover the password string — the same Markov walk logic as the kernel, just in Rust instead of shader code.

---

## Summary: Why It's Fast

| Technique | Benefit |
|-----------|---------|
| **Memory-mapped I/O** | Process 100+ GB files without loading into RAM |
| **Parallel chunk parsing** | Multiple threads scan independent file regions concurrently |
| **Block-level batch planning** | O(n/4000) capacity checks instead of O(n) per-line (macOS) |
| **Direct-write parsing** | Eliminate intermediate data structures entirely (Linux) |
| **Struct-of-Arrays packing** | Minimal overhead, matches GPU memory layout exactly |
| **Zero-copy shared buffers** | CPU writes directly to GPU-visible memory (Apple unified memory) |
| **Pinned host memory + DMA** | Full PCIe bandwidth transfers without staging copies (CUDA) |
| **One-time VRAM upload** | Wordlist uploaded to GPU memory once; batches carry only metadata (CUDA) |
| **Cached raw pointers** | Avoids Objective-C message sends in the hot loop (Metal) |
| **Object pool recycling** | Allocate GPU buffers once, reuse indefinitely |
| **Double-buffered dispatch** | GPU never idles — always processing while CPU prepares next batch |
| **Bounded backpressure** | Memory usage capped at `pipeline_depth × ~80 MiB` |
| **Compile-time shader specialization** | Dead-code elimination and exact register allocation per algorithm |
| **Autotune threadgroup/block width** | Benchmark optimal GPU parallelism on first batch |
| **Fused Markov kernels** | Generate + hash + compare in registers — no memory traffic for candidates |
| **Markov model auto-caching** | 4 MB binary cache loads in ~0ms; re-derive ranked tables for any threshold without retraining |
| **Synchronous Markov dispatch** | Zero host overhead (32-byte params struct) eliminates need for pipeline complexity |
| **Known keyspace → progress/ETA** | Real-time completion tracking since T^L is computed upfront |
