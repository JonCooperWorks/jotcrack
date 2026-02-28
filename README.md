# jotcrack

GPU-assisted HS256 JWT secret brute forcing on macOS using Metal.

## What it does

`jotcrack` takes:

- a JWT (must use `alg=HS256`)
- a wordlist file (one candidate secret per line)

It computes HMAC-SHA256 candidates on the GPU and prints the first matching secret if found.

## Requirements

- macOS (Apple Metal is required)
- Rust toolchain (`cargo`)
- A wordlist file (default: `breach.txt`)

## Build

```bash
cargo build --release
```

## Usage

```bash
cargo run --release -- hs256wordlist '<jwt>' [--wordlist <path>] [--parser-threads <n>] [--pipeline-depth <n>] [--packer-threads <n>]
```

Or run the built binary:

```bash
./target/release/jotcrack hs256wordlist '<jwt>' [--wordlist <path>] [--parser-threads <n>] [--pipeline-depth <n>] [--packer-threads <n>]
```

## Examples

Use the sample JWT in `jwt_example.txt`:

```bash
JWT="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.kXSdJhhUKTJemgs8O0rfIJmUaxoSIDdClL_OPmaC7Eo"
cargo run --release -- hs256wordlist "$JWT" --wordlist ./my_wordlist.txt
```

If you omit `--wordlist`, it uses `breach.txt` in the project root:

```bash
cargo run --release -- hs256wordlist "$JWT"
```

## Wordlist format

- Plain text file
- One candidate secret per line
- Empty lines are ignored
- Very long lines (over 65535 bytes) are skipped and counted in final `STATS`

## Output and exit codes

Output:

- `FOUND <secret>` when a match is found
- `NOT FOUND` when no candidate matches
- `ERROR: ...` for invalid input / runtime failures

Exit codes:

- `0` = found
- `1` = not found
- `2` = error

## JWT constraints

- JWT must have exactly 3 dot-separated segments
- Header must decode as JSON with `"alg": "HS256"`
- Signature must be valid base64url and decode to 32 bytes

## Notes

- Currently available subcommand: `hs256wordlist` (uses the `hs256_wordlist` Metal kernel).
- `--wordlist` is currently supported on `hs256wordlist`.
- `--parser-threads` optionally overrides the auto-selected parallel wordlist parser worker count.
- `--pipeline-depth` optionally increases prefetched `WordBatch` queue depth (higher RAM use, lower GPU idle risk).
- `--packer-threads` optionally controls parallel host batch packing workers (default auto: 2).
- The default `breach.txt` is ignored by git (not included in the repo).

## Performance benchmarking (before/after kernel changes)

Use the same JWT and wordlist ordering for every run so `RATE .../s` numbers are comparable.

Notes on throughput logs:

- Periodic `RATE` lines report windowed (interval) throughput, not cumulative averages.
- Final `STATS` lines report aggregate/cumulative throughput and timing totals.
- Final `STATS` also include parser settings/counters (`parser_threads`, `parser_chunk_bytes`, `parser_chunks`, `parser_skipped_oversize`).

Recommended scenarios:

- Short signing input (`header.payload` < 128 bytes), mostly short candidate secrets
- Medium signing input (~512 bytes), mostly short candidate secrets
- Large signing input (> 1 KB), mostly short candidate secrets

Example workflow:

```bash
# Build once in release mode
cargo build --release

# Run and capture throughput logs (stderr includes RATE/STATS lines)
JWT="$eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.kXSdJhhUKTJemgs8O0rfIJmUaxoSIDdClL_OPmaC7Eo"
./target/release/jotcrack hs256wordlist "$JWT" --wordlist ./my_wordlist.txt 2>bench.log

# Extract throughput snapshots
rg '^RATE |^STATS|^  rate:' bench.log
```

Benchmarking guidance:

- Run each scenario at least 3 times and compare median throughput.
- Keep the machine in a similar thermal/power state.
- Record candidate count and elapsed time along with the reported rate.

## Documented tuned run (2026-02-27)

Exact command used (112GB wordlist):

```bash
cargo run --release -- hs256wordlist --threads-per-group 256 --wordlist wordlists.txt "$(cat jwt_example.txt)" --pipeline-depth 10 --packer-threads 6
```

### Hardware used

- MacBook Pro (Mac16,6)
- Apple M4 Max
- 16-core CPU (12 Performance + 4 Efficiency)
- 40-core GPU
- 64 GB RAM
- macOS 26.3 (Build 25D125)

### Final run summary

```text
STATS
  tested: 16.4 billion
  elapsed: 40.66s
  rate_end_to_end: 404 million/s
  rate_gpu_only: 408 million/s
  batches: 2720
  avg_candidates_per_batch: 6.05 million
  avg_word_bytes_per_batch: 31.0 million bytes
  pipeline_depth: 10
  packer_threads: 6
  parser_threads: 15
  parser_chunk_bytes: 16.8 million bytes
  parser_chunks: 6961
  parser_skipped_oversize: 0
  timing.wordlist_batch_build: 137.189s
  timing.wordlist_batch_plan: 32.340s
  timing.wordlist_batch_pack: 104.849s
  timing.consumer_idle_wait: 0.141s
  timing.host_prep: 0.000s
  timing.command_encode: 0.087s
  timing.gpu_wait: 40.303s
  timing.result_readback: 0.003s
  timing.dispatch_total: 40.394s
NOT FOUND
```

### Interpretation

- `RATE` lines are interval/windowed snapshots (delta since the previous report), not cumulative averages.
- Final `STATS` rates are cumulative over the full run.
- Early high windows (for example 400M+/s) can lift the cumulative average significantly: if the first 1 billion candidates land in about 2.x seconds instead of 4.x seconds, the overall average remains higher even if later windows settle lower.
