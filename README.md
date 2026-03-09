# jotcrack

GPU-accelerated JWT/JWE secret cracking on macOS (Metal) and Linux (CUDA).

Supports **HS256**, **HS384**, **HS512** (HMAC-SHA JWT signing) and **A128KW**, **A192KW**, **A256KW** (AES Key Wrap JWE key management) with automatic detection.

## How it works

### JWT (HMAC-SHA)

JWTs signed with HMAC (HS256/HS384/HS512) use a shared secret key. If you have a JWT and a wordlist of candidate secrets, `jotcrack` tries each candidate on the GPU in parallel and reports the first match.

### JWE (AES Key Wrap)

JWE tokens encrypted with A128KW/A192KW/A256KW wrap the Content Encryption Key (CEK) using AES Key Wrap (RFC 3394). The attack exploits the deterministic integrity check value (0xA6A6A6A6A6A6A6A6) — no known plaintext is needed. Each GPU thread unwraps the CEK with a candidate key and checks the integrity value.

### Attack modes

**Wordlist mode** (default) — tries every line from a wordlist file:

1. **Parse** the wordlist using memory-mapped I/O with parallel chunk scanning
2. **Pack** candidates into GPU-friendly batches (contiguous byte arrays with offset/length tables)
3. **Dispatch** computation on the GPU via Metal (macOS) or CUDA (Linux) — one thread per candidate
4. **Double-buffer** so the next batch is being prepared while the GPU processes the current one

**Markov mode** (`markov` subcommand) — trains an order-1 (bigram) character model on a wordlist and enumerates candidates on the GPU:

1. **Train** (or load from cache) a positional bigram frequency model from the wordlist
2. **Rank** each (position, previous-character) context by frequency, keeping the top-T characters
3. **Enumerate** the full keyspace (T^length candidates per length) with fused GPU kernels — each thread decodes an index into a Markov chain walk, hashes the result, and compares, all in registers
4. **Report** progress with real-time percentage and ETA (total keyspace is known upfront)

## Requirements

**macOS:**
- Apple Silicon (Metal GPU required)
- Rust toolchain (`cargo`)

**Linux:**
- NVIDIA GPU with CUDA 11.4+ (tested on RTX PRO 6000 with CUDA 13.1 on [vast.ai](https://cloud.vast.ai/?ref_id=394548))
- NVIDIA driver with CUDA runtime and NVRTC libraries
- Rust toolchain (`cargo`)

**Both platforms:**
- A wordlist file (default: `breach.txt`)

## Build

```bash
cargo build --release
```

## Quick start

The easiest way to crack a token is `autocrack` — it reads the algorithm from the token header automatically:

```bash
./target/release/jotcrack autocrack '<token>' --wordlist <path>
```

It distinguishes JWT (3-part) from JWE (5-part) compact tokens and routes to the correct GPU backend.

## Subcommands

| Command | Algorithm | Type | Mode | Metal kernel | CUDA kernel |
|---------|-----------|------|------|-------------|-------------|
| `autocrack` | Auto-detect from header | JWT or JWE | Wordlist | Routes to the right one | Routes to the right one |
| `markov` | Auto-detect from header | JWT or JWE | Markov | Fused generate+hash kernels | Fused generate+hash kernels |
| `hs256wordlist` | HMAC-SHA256 | JWT | Wordlist | `hs256_wordlist.metal` | `hs256_wordlist.cu` |
| `hs384wordlist` | HMAC-SHA384 | JWT | Wordlist | `hs512_wordlist.metal` (shared) | `hs512_wordlist.cu` (shared) |
| `hs512wordlist` | HMAC-SHA512 | JWT | Wordlist | `hs512_wordlist.metal` | `hs512_wordlist.cu` |
| `jwe-a128kw` | AES-128 Key Wrap | JWE | Wordlist | `aeskw_wordlist.metal` (AES_KEY_BYTES=16) | `aeskw_wordlist.cu` (AES_KEY_BYTES=16) |
| `jwe-a192kw` | AES-192 Key Wrap | JWE | Wordlist | `aeskw_wordlist.metal` (AES_KEY_BYTES=24) | `aeskw_wordlist.cu` (AES_KEY_BYTES=24) |
| `jwe-a256kw` | AES-256 Key Wrap | JWE | Wordlist | `aeskw_wordlist.metal` (AES_KEY_BYTES=32) | `aeskw_wordlist.cu` (AES_KEY_BYTES=32) |

`autocrack` is the recommended default for wordlist attacks. `markov` expands the search space beyond any static wordlist by generating candidates from learned character patterns.

## CLI options

### Wordlist mode (`autocrack` and algorithm-specific commands)

```
ARGS:
  <JWT>    Token in compact form (JWT: header.payload.signature, JWE: header.ek.iv.ct.tag)

OPTIONS:
  --wordlist <PATH>           Wordlist file path [default: breach.txt]
  --threads-per-group <N>     Fixed GPU threadgroup/block width override
  --parser-threads <N>        Number of parallel wordlist parser workers
  --pipeline-depth <N>        Max in-flight batches between parser and GPU
  --packer-threads <N>        Number of host batch packing workers
  --autotune                  Benchmark threadgroup widths on first batch
```

### Markov mode (`markov`)

```
ARGS:
  <JWT>    Token in compact form

OPTIONS:
  --wordlist <PATH>           Training wordlist for the Markov model [default: breach.txt]
  --threshold <N>             Top-N characters per (position, prev_char) context [default: 30]
  --min-len <N>               Minimum candidate length to enumerate [default: 1]
  --max-len <N>               Maximum candidate length to enumerate [default: 8]
  --threads-per-group <N>     Fixed GPU threadgroup/block width override
  --autotune                  Benchmark threadgroup widths before dispatch
  --retrain                   Force re-scan of wordlist (ignore cached .markov file)
```

The Markov model is auto-cached to `<wordlist>.markov` on first run. Subsequent runs load the cache in ~0ms. Changing `--threshold` or `--max-len` doesn't require retraining — the ranked table is re-derived from cached frequency counts.

## Testing

We tested all algorithms end-to-end with our own wordlist — cracking known secrets (e.g. `"password"`) and running full-throughput stress tests with random keys that don't appear in the wordlist (to force a complete scan and measure sustained GPU speed).

The repo does not include a wordlist. You'll need to build or source your own — one candidate secret per line, plain text. Any wordlist works; bigger lists give you a better throughput picture since the GPU stays saturated longer.

## Wordlist format

- Plain text file, one candidate secret per line
- Empty lines are ignored
- Lines over 65,535 bytes are skipped (counted in final `STATS`)
- Supports both LF and CRLF line endings

## Output and exit codes

Output:
- `HS256 key: <secret>` / `A128KW key: <secret>` / etc. — when found
- `NOT FOUND` — when no candidate matches
- `ERROR: ...` — for invalid input or runtime failures

Exit codes:
- `0` = found
- `1` = not found
- `2` = error

## Performance

### HMAC-SHA (JWT)

**NVIDIA RTX PRO 6000** (Blackwell, 96 GB VRAM, CUDA 13.1) with ~12 billion candidates:

| Algorithm | End-to-End | GPU-Only |
|-----------|-----------|----------|
| **HS256** | **793 M/s** | 16.0 B/s |
| **HS384** | **795 M/s** | 5.18 B/s |
| **HS512** | **795 M/s** | 5.10 B/s |

All three HMAC algorithms hit the same ~795 M/s end-to-end because the pipeline is CPU-bound — the GPU finishes each batch before the next one is ready. GPU-only rates show NVIDIA's native 64-bit ALUs handle SHA-512 much better than Apple Silicon's 32-bit emulation.

**Apple M4 Max** (40-core GPU, 64 GB RAM) with a 112 GB wordlist (16.4 billion candidates):

| Algorithm | End-to-End | GPU-Only | vs HS256 |
|-----------|-----------|----------|----------|
| **HS256** | **446 M/s** | 441 M/s | 1.0x |
| **HS384** | **70 M/s** | 96 M/s | 6.3x slower |
| **HS512** | **73 M/s** | 91 M/s | 6.1x slower |

**Why HS384/HS512 are slower on Apple Silicon**: SHA-512 uses 64-bit words, but Apple GPUs have 32-bit ALUs — every 64-bit operation is emulated as multiple 32-bit instructions. SHA-512 also uses 128-byte blocks and 80 rounds (vs SHA-256's 64-byte blocks and 64 rounds). NVIDIA GPUs have native 64-bit integer support, so all three variants are CPU-bound at the same rate.

### AES Key Wrap (JWE)

**NVIDIA RTX PRO 6000** with ~13.2 billion candidates:

| Algorithm | End-to-End | GPU-Only | AES Rounds | Key Size |
|-----------|-----------|----------|------------|----------|
| **A128KW** | **145 M/s** | 146 M/s | 10 | 16 bytes |
| **A192KW** | **120 M/s** | 121 M/s | 12 | 24 bytes |
| **A256KW** | **104 M/s** | 104 M/s | 14 | 32 bytes |

**Apple M4 Max** with breach.txt (~296M candidates):

| Algorithm | End-to-End | GPU-Only | AES Rounds | Key Size |
|-----------|-----------|----------|------------|----------|
| **A128KW** | **120 M/s** | 125 M/s | 10 | 16 bytes |
| **A192KW** | **106 M/s** | 110 M/s | 12 | 24 bytes |
| **A256KW** | **90.5 M/s** | 93.4 M/s | 14 | 32 bytes |

Performance scales linearly with AES round count — each additional 2 rounds adds ~12% overhead. The AES Key Wrap shader uses software AES with S-box lookup tables in GPU constant memory, broadcast-cached across SIMD groups/warps.

### Markov Mode

**Apple M4 Max** (40-core GPU, 64 GB RAM) with threshold=30:

| Algorithm | Speed | Keyspace (T=30, L=6) |
|-----------|-------|----------------------|
| **HS256** | **394 M/s** | 754M candidates |
| **HS384** | **114 M/s** | 754M candidates |
| **HS512** | **112 M/s** | 754M candidates |
| **A128KW** | **128 M/s** | 754M candidates |
| **A192KW** | **107 M/s** | 754M candidates |
| **A256KW** | **90.1 M/s** | 754M candidates |

HS256 sustained rate with T=30, L=7 (21.9B candidates): **344 M/s** (~335 M/s steady state).

Markov mode eliminates the wordlist parsing pipeline entirely — the GPU generates and hashes candidates in a single fused kernel. Throughput is purely GPU-bound.

**Keyspace scaling** (threshold=30):

| Max Length | Candidates | HS256 M4 Max ETA |
|------------|------------|-------------------|
| 6 | 754M | ~2 s |
| 7 | 21.9B | ~65 s |
| 8 | 638B | ~32 min |

### Benchmarking

```bash
cargo build --release

# Crack a known secret (should find it quickly)
./target/release/jotcrack autocrack '<your-token>' --wordlist my_wordlist.txt

# Full throughput benchmark — use a token with a key NOT in your wordlist
# so the entire wordlist is scanned and you get sustained GPU throughput numbers
./target/release/jotcrack autocrack '<stress-token>' --wordlist my_wordlist.txt 2>bench.log
```

Tips:
- Run each scenario at least 3 times and compare median throughput
- Keep the machine in a similar thermal/power state
- `RATE` lines in stderr are windowed (interval) snapshots, not cumulative averages
- Final `STATS` block shows cumulative totals for the entire run

## Architecture

```
src/
  main.rs               Entry point (thin — delegates to commands::run())
  commands.rs            CLI parsing (clap) + subcommand dispatch
  args.rs                Shared CLI args, ParserConfig, constants
  jwt.rs                 JWT/JWE token parsing and algorithm detection
  runner.rs              Unified cracking pipeline (wordlist + Markov dispatch loops)
  markov.rs              Markov model training, caching, candidate reconstruction
  batch.rs               WordBatch — packed GPU batch format
  parser.rs              Parallel mmap wordlist parsing
  producer.rs            Multi-threaded producer pipeline
  stats.rs               Timing, rate reporting, ETA, final stats
  test_support.rs        Test helpers (temp wordlists, JWT/JWE generation)
  gpu/
    mod.rs               GpuBruteForcer + GpuMarkovBruteForcer traits + platform type aliases
    metal/               macOS Metal backend
      mod.rs             MetalBruteForcer, MetalAesKwBruteForcer, MetalMarkovHmacBruteForcer, MetalMarkovAesKwBruteForcer
      hs256_wordlist.metal   SHA-256 Metal kernel (wordlist + Markov entry points)
      hs512_wordlist.metal   SHA-512 Metal kernel (wordlist + Markov entry points, shared by HS384)
      aeskw_wordlist.metal   AES Key Wrap Metal kernel (wordlist + Markov entry points)
    cuda/                Linux CUDA backend
      mod.rs             CudaBruteForcer, CudaAesKwBruteForcer, CudaMarkovHmacBruteForcer, CudaMarkovAesKwBruteForcer
      hs256_wordlist.cu  SHA-256 CUDA kernel (wordlist + Markov entry points)
      hs512_wordlist.cu  SHA-512 CUDA kernel (wordlist + Markov entry points, shared by HS384)
      aeskw_wordlist.cu  AES Key Wrap CUDA kernel (wordlist + Markov entry points)
```

### Key design decisions

- **GPU kernels are embedded** via `include_str!()` — the binary is fully self-contained. Metal shaders are compiled at pipeline creation; CUDA kernels are compiled to PTX at startup via NVRTC
- **Platform abstraction**: the `GpuBruteForcer` trait and platform-gated type aliases (`GpuBuffer`, `GpuDevice`, `GpuCommandHandle`) keep Metal and CUDA concerns isolated — `runner.rs`, `batch.rs`, and `producer.rs` compile unchanged on both platforms
- **Double-buffered dispatch**: the next batch is parsed and packed while the GPU processes the current one
- **Zero-copy batches** (Metal): wordlist parser writes directly into Metal shared-memory buffers. On CUDA, **pinned (page-locked) host memory** enables DMA transfers at full PCIe bandwidth without intermediate staging copies
- **Two kernel variants per algorithm**: a "short-key" specialization avoids the long-key hash-first path
- **Compile-time AES specialisation**: the AES Key Wrap kernel is compiled three times with different `#define AES_KEY_BYTES` values, producing fully specialised kernels per variant (no runtime branching, exact register allocation)
- **Autotune**: `--autotune` benchmarks several threadgroup/block widths on the first batch and picks the fastest
- **Fused Markov kernels**: Each GPU thread decodes a keyspace index → walks a Markov chain in registers → hashes → compares. The candidate never leaves registers — no intermediate buffer, no memory traffic for generated passwords
- **Markov model auto-caching**: Frequency counts are saved to `<wordlist>.markov` on first training. Subsequent runs load the 4 MB cache in ~0ms. Changing `--threshold` or `--max-len` re-derives the ranked table from cached counts without retraining
- **Markov progress & ETA**: Total keyspace is known upfront (T^L per length), enabling real-time progress percentage and ETA reporting

## Kernel optimizations

These optimizations apply to both Metal and CUDA kernels — the CUDA ports preserve the same algorithmic structure.

### HMAC-SHA kernels

1. **Rolling 16-word message schedule**: `w[16]` replaces `w[64]` in SHA compression, saving register pressure. Rounds 16+ update `w[i&15]` in-place.
2. **Flat HMAC**: Bare state arrays replace streaming hash contexts, eliminating block buffers from the hot path. HMAC inner/outer passes are manually finalized.
3. **Precomputed target words**: Host converts signature bytes to native-endian words once, avoiding per-thread byte-swapping in the comparison loop.

### AES Key Wrap kernel

1. **Compile-time specialisation**: `AES_KEY_BYTES` controls Nk (key words), Nr (rounds), and round key array size at compile time — the compiler unrolls loops and allocates exact registers.
2. **Constant-memory S-boxes**: Forward and inverse S-box tables in constant memory (`constant` on Metal, `__constant__` on CUDA) — broadcast-cached across SIMD groups/warps, avoiding per-thread register spills.
3. **Arithmetic GF(2^8)**: MixColumns/InvMixColumns use inline `xtime`/`mul` arithmetic instead of lookup tables, reducing constant memory pressure.
4. **SHA-256 key derivation**: Long candidates (> key_bytes) are SHA-256 hashed and truncated; short candidates are zero-padded — matching JOSE key derivation semantics.

### CUDA-specific details

- **Unified address space**: CUDA has a flat global memory model, so the Metal pattern of duplicating helper functions for `device`/`constant`/`threadgroup` address spaces collapses to single implementations.
- **`__constant__` memory**: SHA round constants and AES S-box tables use CUDA's dedicated 64KB constant cache, which broadcasts reads to all 32 threads in a warp simultaneously.
- **Pinned host memory**: Host-side batch buffers use `cuMemHostAlloc` (page-locked memory), enabling DMA transfers at full PCIe bandwidth and eliminating the driver's internal staging copy.
- **NVRTC runtime compilation**: Kernel sources are compiled to PTX at startup via NVIDIA's Runtime Compilation library, matching Metal's runtime shader compilation approach. Compute capability is auto-detected for architecture-specific codegen.

## Tests

```bash
cargo test
```

Tests cover:
- JWT parsing for each HMAC algorithm (success, wrong alg, wrong sig length, bad base64)
- JWE parsing for each AES-KW variant (success, wrong alg, wrong part count, short encrypted key)
- GPU params struct size round-trips
- End-to-end wordlist cracking for HS256, HS384, HS512 (finds "password" in a temp wordlist)
- End-to-end wordlist cracking for A128KW, A192KW, A256KW (finds "password" in a temp wordlist)
- End-to-end Markov cracking for HS256, HS384, HS512, A128KW, A192KW, A256KW
- Not-found cases for all algorithms (both wordlist and Markov modes)
- Autodetect routing (autocrack dispatches to each algorithm correctly, both JWT and JWE)
- Autodetect rejection of unsupported algorithms (e.g., RS256, A128GCMKW)
- Markov model training correctness, candidate_from_index roundtrip, keyspace calculation
- Markov cache save/load, ETA formatting
- CLI contract tests (accepted flags, defaults, validation)
- Wordlist parser correctness (batch boundaries, CRLF handling, oversized lines)
- Producer pipeline lifecycle (spawn, produce, join)
