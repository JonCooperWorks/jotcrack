# Jotcrack Architecture: GPU Abstraction & Pipeline

Jotcrack is a GPU-accelerated JWT/JWE secret cracker for macOS using Apple Metal compute shaders. It processes massive wordlists (100+ GB) by memory-mapping files, parsing in parallel, and double-buffering GPU dispatch so the CPU and GPU never wait on each other.

**Performance:** 446M candidates/s for HS256, 120M/s for A128KW on Apple M4 Max.

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
         ▼
  ┌──────────────────────┐
  │  Parser Threads (N)  │  Scan ~16 MiB chunks for line boundaries
  │  (parser.rs)         │  Produce MmapChunk metadata (offsets + lengths)
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────┐
  │  Planner Thread  │  Group lines into BatchPlans respecting GPU limits
  │  (producer.rs)   │  (~6.2M candidates / 32 MiB per batch)
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────────┐
  │  Packer Threads (N)  │  Copy candidate bytes from mmap into Metal buffers
  │  (producer.rs)       │  Direct raw memcpy — no intermediate allocations
  └──────┬───────────────┘
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
  │  GPU Kernel (Metal shaders)  │  One thread per candidate:
  │  HMAC-SHA or AES Key Wrap   │    compute hash/unwrap → compare → atomic write
  └──────────────────────────────┘
```

---

## 1. GPU Abstraction Layer (`src/gpu/mod.rs`)

The GPU abstraction defines a **platform-agnostic trait** that decouples the cracking algorithm from the GPU backend. Today the only backend is Apple Metal, but the trait boundary makes it possible to add OpenCL, Vulkan, or CUDA backends.

### The `GpuBruteForcer` Trait

```rust
pub(crate) trait GpuBruteForcer {
    fn device(&self) -> &GpuDevice;
    fn encode_and_commit(&mut self, target: &[u8], view: &DispatchBatchView)
        -> Result<(GpuCommandHandle, Duration, Duration)>;
    fn wait_and_readback(&self, handle: &GpuCommandHandle)
        -> (Option<u32>, Duration, Duration);
    fn autotune_threadgroup_width(&mut self, target: &[u8], batch: &WordBatch) -> Result<()>;
    fn set_threadgroup_width(&mut self, width: usize);
    fn current_threadgroup_width(&self) -> usize;
}
```

| Method | Purpose |
|--------|---------|
| `encode_and_commit()` | Encode GPU commands and submit asynchronously. Returns a command handle and timing data. The GPU begins executing immediately. |
| `wait_and_readback()` | Block until the GPU finishes, then read the result buffer. Returns `Some(match_index)` if a candidate matched, or `None`. |
| `autotune_threadgroup_width()` | Benchmark different threadgroup widths on the first batch to find the optimal configuration for the current device. |

### Platform Type Aliases

```rust
#[cfg(target_os = "macos")]
pub(crate) type GpuDevice = ::metal::Device;
pub(crate) type GpuBuffer = ::metal::Buffer;
pub(crate) type GpuCommandHandle = ::metal::CommandBuffer;
```

These aliases let all code outside `gpu/metal/` remain platform-independent.

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

Two structs implement `GpuBruteForcer`:

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

### Buffer Slot Assignments (Both Backends)

Both backends bind six Metal buffers to fixed slot indices:

| Slot | Buffer | Contents |
|------|--------|----------|
| 0 | `params_buf` | Algorithm parameters struct |
| 1 | `msg_buf` / `encrypted_key_buf` | JWT signing input or wrapped CEK |
| 2 | `word_bytes_buf` | Packed candidate bytes (from WordBatch) |
| 3 | `word_offsets_buf` | Per-candidate byte offsets (u32) |
| 4 | `word_lengths_buf` | Per-candidate byte lengths (u16) |
| 5 | `result_buf` | Match result (u32::MAX = no match) |

---

## 3. Batch System (`src/batch.rs`)

### The Problem

The GPU needs flat, contiguous buffers. A naive `Vec<String>` causes per-candidate allocation overhead that dominates at 400M+ candidates/second.

### The Solution: Struct-of-Arrays Packing

`WordBatch` stores candidates in three parallel arrays packed into pre-allocated Metal shared buffers:

```
Candidates: "alpha", "be", "cat"

word_bytes:   [a][l][p][h][a][b][e][c][a][t]   ← concatenated, no separators
word_offsets: [0, 5, 7]                          ← byte offset of each candidate
word_lengths: [5, 2, 3]                          ← byte length of each candidate
```

```rust
pub(crate) struct WordBatch {
    candidate_index_base: u64,       // Global index of first candidate in this batch
    word_bytes_buf: GpuBuffer,       // Metal shared buffer — visible to both CPU and GPU
    word_offsets_buf: GpuBuffer,
    word_lengths_buf: GpuBuffer,
    word_bytes_ptr: *mut u8,         // Cached raw pointers (avoids ObjC message sends)
    word_offsets_ptr: *mut u32,
    word_lengths_ptr: *mut u16,
    candidate_count: usize,
    word_bytes_len: usize,
    max_word_len: u16,
}
```

### Batch Capacity Limits

- **MAX_CANDIDATES_PER_BATCH**: 6,182,240 (~6.2M candidates)
- **MAX_WORD_BYTES_PER_BATCH**: 32 MiB of packed candidate bytes
- **APPROX_WORD_BATCH_BUFFER_BYTES**: ~67 MiB total memory per batch

### Object Pool Recycling

Allocating Metal buffers involves FFI calls and kernel VM operations. Instead of allocating/freeing per batch, jotcrack allocates a fixed number of batches (`pipeline_depth`, default 10) and recycles them:

1. Consumer finishes with a batch → sends it back via `producer.recycle(batch)`
2. Producer calls `batch.reset_for_reuse()` → clears logical cursors, keeps allocation
3. Packer thread fills the batch again with new candidates

This amortizes allocation cost to near zero.

### Zero-Copy GPU Dispatch

`WordBatch` allocates Metal `StorageModeShared` buffers, meaning the CPU and GPU share the same physical memory on Apple Silicon. When the packer writes candidate bytes into `word_bytes_ptr`, the GPU reads them directly — no copy or transfer step.

`as_dispatch_view()` returns a `DispatchBatchView` — a zero-copy borrow of the batch's buffers and metadata — that gets passed to `gpu.encode_and_commit()`.

---

## 4. Parser System (`src/parser.rs`)

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

**Step 5: Pack into Metal buffers.** Packer threads receive a `BatchPlan` and call `pack_batch_plan_into_batch()`, which copies candidate bytes directly from the mmap into Metal shared buffers via `push_segment_bulk()`.

---

## 5. Producer Pipeline (`src/producer.rs`)

The producer orchestrates the parser, planner, and packer threads into a three-stage pipeline:

```
┌──────────────────────────────────────────────────┐
│              Producer Thread                      │
│                                                  │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐ │
│  │ Planner  │───→│ Job Queue │───→│ Packer 1  │ │
│  │          │    │           │    │ Packer 2  │ │
│  └──────────┘    └───────────┘    │ Packer N  │ │
│                                   └─────┬─────┘ │
│                                         │       │
│  ┌──────────────┐                       │       │
│  │ Batch Pool   │◄── recycle ───────────┤       │
│  │ (recycled    │                       │       │
│  │  WordBatches)│                       │       │
│  └──────┬───────┘                       │       │
│         │                               │       │
│         └───► Coordinator ◄─────────────┘       │
│                    │                             │
│                    ▼                             │
│              tx ──────────────────────────── rx  │
└──────────────────────────────────────────────────┘
                                              │
                                              ▼
                                    Consumer (runner.rs)
```

### Bounded Channel Backpressure

The producer-to-consumer channel uses `sync_channel(pipeline_depth)`. This prevents the producer from parsing the entire wordlist into memory while the GPU is still processing early batches. Memory usage is bounded to `pipeline_depth × 67 MiB`.

### ProducerMessage

```rust
pub(crate) enum ProducerMessage {
    Batch { batch: WordBatch, build_time, plan_time, pack_time, parser_stats },
    Eof { parser_stats },
    Error(String),
}
```

---

## 6. Runner & Double-Buffered Dispatch (`src/runner.rs`)

The runner ties everything together with a generic dispatch loop:

```rust
fn run_gpu_crack<B: GpuBruteForcer>(
    variant: CrackVariant,
    mut gpu: B,
    target_data: &[u8],
    args: WordlistArgs,
) -> anyhow::Result<bool>
```

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

## 7. Metal Shader Kernels

### HMAC-SHA Kernels (`hs256_wordlist.metal`, `hs512_wordlist.metal`)

Each GPU thread (identified by `gid`) processes one candidate:

1. Read candidate bytes using `word_bytes[offsets[gid]..offsets[gid]+lengths[gid]]`
2. Derive the HMAC key (zero-pad if short, SHA-hash if longer than block size)
3. Compute HMAC: `SHA(key⊕opad ‖ SHA(key⊕ipad ‖ message))`
4. Compare against `target_signature` (precomputed as native-endian words on the host)
5. On match: `atomic_fetch_min(result_buf, gid)` — lowest matching index wins

**Optimizations:**
- Rolling message schedule (`w[16]` instead of `w[64]`) reduces register pressure
- Flat HMAC implementation avoids streaming context overhead
- Host pre-converts signature to native-endian u32/u64 words
- SHA-256 constant K values cached across SIMD groups

### AES Key Wrap Kernel (`aeskw_wordlist.metal`)

Compiled three times with different `#define AES_KEY_BYTES` values (16, 24, 32):

1. Derive AES key from candidate (zero-pad if short, SHA-256 + truncate if long)
2. Expand AES key schedule (10/12/14 rounds depending on key size)
3. RFC 3394 unwrap: 6×n AES-ECB decryptions
4. Check integrity value == `0xA6A6A6A6A6A6A6A6`
5. On match: atomic write to result buffer

**Optimizations:**
- Compile-time specialization lets Metal unroll key-size-dependent loops
- S-boxes in `constant` address space (hardware-cached)
- GF(2^8) arithmetic via `xtime` helper avoids extra lookup tables

---

## 8. Statistics & Timing (`src/stats.rs`)

Every batch records a detailed timing breakdown:

| Metric | What It Measures |
|--------|-----------------|
| `plan_time` | Planner grouping lines into a BatchPlan |
| `pack_time` | Packer copying bytes into Metal buffers |
| `host_prep` | Writing params, resetting sentinel |
| `command_encode` | Metal API calls to encode GPU commands |
| `gpu_wait` | Actual GPU compute time |
| `result_readback` | Reading the result buffer |
| `consumer_idle_wait` | Time consumer blocks waiting for the producer |

Windowed rate reporting prints throughput every second. The final report shows both end-to-end and GPU-only throughput.

---

## Summary: Why It's Fast

| Technique | Benefit |
|-----------|---------|
| **Memory-mapped I/O** | Process 100+ GB files without loading into RAM |
| **Parallel chunk parsing** | Multiple threads scan independent file regions concurrently |
| **Block-level batch planning** | O(n/4000) capacity checks instead of O(n) per-line |
| **Struct-of-Arrays packing** | Minimal overhead, matches GPU memory layout exactly |
| **Zero-copy shared buffers** | CPU writes directly to GPU-visible memory (Apple unified memory) |
| **Cached raw pointers** | Avoids Objective-C message sends in the hot loop |
| **Object pool recycling** | Allocate Metal buffers once, reuse indefinitely |
| **Double-buffered dispatch** | GPU never idles — always processing while CPU prepares next batch |
| **Bounded backpressure** | Memory usage capped at `pipeline_depth × 67 MiB` |
| **Compile-time shader specialization** | Dead-code elimination and exact register allocation per algorithm |
| **Autotune threadgroup width** | Benchmark optimal GPU parallelism on first batch |
