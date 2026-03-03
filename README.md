# jotcrack

GPU-accelerated HMAC-SHA JWT secret cracking on macOS using Apple Metal.

Supports **HS256**, **HS384**, and **HS512** algorithms with automatic detection.

## How it works

JWTs signed with HMAC (HS256/HS384/HS512) use a shared secret key. If you have a JWT and a wordlist of candidate secrets, `jotcrack` tries each candidate on the GPU in parallel and reports the first match.

The pipeline:
1. **Parse** the wordlist using memory-mapped I/O with parallel chunk scanning
2. **Pack** candidates into GPU-friendly batches (contiguous byte arrays with offset/length tables)
3. **Dispatch** HMAC-SHA computation on the Metal GPU (one thread per candidate)
4. **Double-buffer** so the next batch is being prepared while the GPU processes the current one

## Requirements

- macOS with Apple Silicon (Metal GPU required)
- Rust toolchain (`cargo`)
- A wordlist file (default: `breach.txt`)

## Build

```bash
cargo build --release
```

## Quick start

The easiest way to crack a JWT is `autocrack` — it reads the algorithm from the JWT header automatically:

```bash
./target/release/jotcrack autocrack '<jwt>' --wordlist <path>
```

Example:

```bash
# Crack any HMAC-SHA JWT — autocrack reads the algorithm from the header
./target/release/jotcrack autocrack 'eyJhbGciOiJIUzI1NiIs...' --wordlist my_wordlist.txt
```

## Subcommands

| Command | Algorithm | Signature size | GPU kernel |
|---------|-----------|---------------|------------|
| `autocrack` | Auto-detect from JWT header | — | Routes to the right one |
| `hs256wordlist` | HMAC-SHA256 | 32 bytes | `hs256_wordlist.metal` |
| `hs384wordlist` | HMAC-SHA384 | 48 bytes | `hs512_wordlist.metal` (shared from hs512wordlist/) |
| `hs512wordlist` | HMAC-SHA512 | 64 bytes | `hs512_wordlist.metal` |

`autocrack` is the recommended default. Use the algorithm-specific commands if you want to skip the auto-detection step.

## CLI options

All subcommands accept the same flags:

```
ARGS:
  <JWT>    JWT in compact form (header.payload.signature)

OPTIONS:
  --wordlist <PATH>           Wordlist file path [default: breach.txt]
  --threads-per-group <N>     Fixed Metal threadgroup width override
  --parser-threads <N>        Number of parallel wordlist parser workers
  --pipeline-depth <N>        Max in-flight batches between parser and GPU
  --packer-threads <N>        Number of host batch packing workers
  --autotune                  Benchmark threadgroup widths on first batch
```

## Testing

We tested all three algorithms end-to-end with our own wordlist — cracking known secrets (e.g. `"password"`) and running full-throughput stress tests with random 32-character keys that don't appear in the wordlist (to force a complete scan and measure sustained GPU speed).

The repo does not include a wordlist. You'll need to build or source your own — one candidate secret per line, plain text. Any wordlist works; bigger lists give you a better throughput picture since the GPU stays saturated longer.

## Wordlist format

- Plain text file, one candidate secret per line
- Empty lines are ignored
- Lines over 65,535 bytes are skipped (counted in final `STATS`)
- Supports both LF and CRLF line endings

## Output and exit codes

Output:
- `HS256 key: <secret>` / `HS384 key: <secret>` / `HS512 key: <secret>` — when found
- `NOT FOUND` — when no candidate matches
- `ERROR: ...` — for invalid input or runtime failures

Exit codes:
- `0` = found
- `1` = not found
- `2` = error

## Performance

Benchmarked on Apple M4 Max (40-core GPU, 64 GB RAM) with a 112 GB wordlist (16.4 billion candidates):

| Algorithm | End-to-End | GPU-Only | vs HS256 |
|-----------|-----------|----------|----------|
| **HS256** | **446 M/s** | 441 M/s | 1.0x |
| **HS384** | **70 M/s** | 96 M/s | 6.3x slower |
| **HS512** | **73 M/s** | 91 M/s | 6.1x slower |

**Why HS384/HS512 are slower**: SHA-512 uses 64-bit words, but Apple GPUs have 32-bit ALUs — every 64-bit operation is emulated as multiple 32-bit instructions. SHA-512 also uses 128-byte blocks and 80 rounds (vs SHA-256's 64-byte blocks and 64 rounds).

**Steady-state GPU rates** (outside the long-word region of the wordlist): ~105 M/s for both HS384/HS512, which is ~4.2x slower than HS256 — matching the expected cost of 64-bit emulation.

### Benchmarking

```bash
cargo build --release

# Crack a known secret (should find it quickly)
./target/release/jotcrack autocrack '<your-jwt>' --wordlist my_wordlist.txt

# Full throughput benchmark — use a JWT signed with a key NOT in your wordlist
# so the entire wordlist is scanned and you get sustained GPU throughput numbers
./target/release/jotcrack autocrack '<stress-jwt>' --wordlist my_wordlist.txt 2>bench.log
```

Tips:
- Run each scenario at least 3 times and compare median throughput
- Keep the machine in a similar thermal/power state
- `RATE` lines in stderr are windowed (interval) snapshots, not cumulative averages
- Final `STATS` block shows cumulative totals for the entire run

## Architecture

```
src/
  main.rs                           Entry point (thin — delegates to commands::run())
  commands/
    mod.rs                          CLI parsing (clap) + subcommand dispatch
    common/
      args.rs                       Shared ParserConfig, constants
      batch.rs                      WordBatch — packed GPU batch format
      parser.rs                     Parallel mmap wordlist parsing (~1300 lines)
      producer.rs                   Multi-threaded producer pipeline (~580 lines)
      stats.rs                      Timing, rate reporting, final stats
      test_support.rs               Test helpers (temp wordlists, JWT generation)
    hs256wordlist/
      args.rs, jwt.rs, gpu.rs       HS256-specific parsing + GPU setup
      command.rs                    HS256 cracking pipeline
      hs256_wordlist.metal          SHA-256 GPU kernel
    hs384wordlist/                  HS384-specific (uses hs512wordlist's kernel)
    hs512wordlist/
      args.rs, jwt.rs, gpu.rs       HS512-specific parsing + GPU setup
      command.rs                    HS512 cracking pipeline
      hs512_wordlist.metal          SHA-512 GPU kernel (shared by HS384)
    autocrack/                    Auto-detect algorithm + dispatch
```

### Key design decisions

- **Metal kernels are embedded** via `include_str!()` — the binary is fully self-contained
- **Double-buffered dispatch**: the next batch is parsed and packed while the GPU processes the current one
- **Zero-copy batches**: wordlist parser writes directly into Metal shared-memory buffers
- **Two kernel variants per algorithm**: a "short-key" specialization (keys <= 64/128 bytes) avoids the long-key hash-first path
- **Autotune**: `--autotune` benchmarks several threadgroup widths on the first batch and picks the fastest

## Kernel optimizations

Three GPU kernel techniques (applied to all algorithms):

1. **Rolling 16-word message schedule**: `w[16]` replaces `w[64]` in SHA compression, saving register pressure. Rounds 16+ update `w[i&15]` in-place.
2. **Flat HMAC**: Bare state arrays replace streaming hash contexts, eliminating block buffers from the hot path. HMAC inner/outer passes are manually finalized.
3. **Precomputed target words**: Host converts signature bytes to native-endian words once, avoiding per-thread byte-swapping in the comparison loop.

## Tests

```bash
cargo test
```

Tests cover:
- JWT parsing for each algorithm (success, wrong alg, wrong sig length, bad base64)
- GPU params struct size round-trips
- End-to-end cracking for HS256, HS384, HS512 (finds "password" in a temp wordlist)
- Not-found cases
- Autodetect routing (autocrack dispatches to each algorithm correctly)
- Autodetect rejection of unsupported algorithms (e.g., RS256)
- CLI contract tests (accepted flags, defaults, validation)
- Wordlist parser correctness (batch boundaries, CRLF handling, oversized lines)
- Producer pipeline lifecycle (spawn, produce, join)
