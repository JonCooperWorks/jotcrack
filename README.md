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
# Crack an HS256 JWT using the auto-detector
./target/release/jotcrack autocrack "$(cat jwt_hs256_password)" --wordlist wordlists.txt

# It also works with HS384 and HS512 — same command
./target/release/jotcrack autocrack "$(cat jwt_hs384_password)" --wordlist wordlists.txt
./target/release/jotcrack autocrack "$(cat jwt_hs512_password)" --wordlist wordlists.txt
```

## Subcommands

| Command | Algorithm | Signature size | GPU kernel |
|---------|-----------|---------------|------------|
| `autocrack` | Auto-detect from JWT header | — | Routes to the right one |
| `hs256wordlist` | HMAC-SHA256 | 32 bytes | `hs256_wordlist.metal` |
| `hs384wordlist` | HMAC-SHA384 | 48 bytes | `hs512_wordlist.metal` (shared) |
| `hs512wordlist` | HMAC-SHA512 | 64 bytes | `hs512_wordlist.metal` (shared) |

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

## JWT test files

The repo includes pre-generated JWT files for testing and benchmarking:

| File | Algorithm | Secret | Purpose |
|------|-----------|--------|---------|
| `jwt_hs256_password` | HS256 | `password` | Quick crack test (should find it) |
| `jwt_hs384_password` | HS384 | `password` | Quick crack test |
| `jwt_hs512_password` | HS512 | `password` | Quick crack test |
| `jwt_hs256_stress` | HS256 | random 32-char | Throughput benchmark (NOT FOUND) |
| `jwt_hs384_stress` | HS384 | random 32-char | Throughput benchmark |
| `jwt_hs512_stress` | HS512 | random 32-char | Throughput benchmark |

Generate fresh ones with:

```bash
python3 -c "
import hmac, hashlib, base64, json, secrets, string
def make_jwt(alg, payload, secret):
    header = json.dumps({'alg': alg, 'typ': 'JWT'}, separators=(',',':'))
    h = base64.urlsafe_b64encode(header.encode()).rstrip(b'=').decode()
    p = base64.urlsafe_b64encode(json.dumps(payload, separators=(',',':')).encode()).rstrip(b'=').decode()
    si = f'{h}.{p}'
    hf = {'HS256': hashlib.sha256, 'HS384': hashlib.sha384, 'HS512': hashlib.sha512}[alg]
    sig = base64.urlsafe_b64encode(hmac.new(secret.encode(), si.encode(), hf).digest()).rstrip(b'=').decode()
    return f'{si}.{sig}'
stress = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
for alg in ['HS256', 'HS384', 'HS512']:
    for name, key in [('password', 'password'), ('stress', stress)]:
        with open(f'jwt_{alg.lower()}_{name}', 'w') as f:
            f.write(make_jwt(alg, {'sub': 'jotcrack-test'}, key))
print(f'Stress key: {stress}')
"
```

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
# Build once
cargo build --release

# Quick crack test (should find "password")
./target/release/jotcrack autocrack "$(cat jwt_hs256_password)" --wordlist wordlists.txt

# Full throughput benchmark (NOT FOUND — processes entire wordlist)
./target/release/jotcrack autocrack "$(cat jwt_hs256_stress)" --wordlist wordlists.txt 2>bench.log
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
      hs512_wordlist.metal          SHA-512 GPU kernel (shared by HS384/HS512)
    hs256wordlist/
      args.rs, jwt.rs, gpu.rs       HS256-specific parsing + GPU setup
      command.rs                    HS256 cracking pipeline
      hs256_wordlist.metal          SHA-256 GPU kernel
    hs384wordlist/                  HS384-specific (uses common SHA-512 kernel)
    hs512wordlist/                  HS512-specific (uses common SHA-512 kernel)
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
