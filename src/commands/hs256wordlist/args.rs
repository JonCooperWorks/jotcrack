use std::path::PathBuf;

use clap::Args;

// ---- Runtime configuration / tuning knobs ---------------------------------
// These constants define how much work we package into one GPU dispatch and how
// aggressively the host pipeline overlaps parsing with GPU execution.
pub const DEFAULT_WORDLIST_PATH: &str = "breach.txt";
// Small bounded queue: enough to overlap CPU/GPU and smooth producer jitter
// without letting parsed data accumulate unbounded in memory.
pub(super) const DEFAULT_PIPELINE_DEPTH: usize = 10;
// Parser workers scan mmap chunks in parallel and return line metadata for
// ordered batch assembly on the producer thread.
pub(super) const DEFAULT_PARSER_CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// CLI arguments for the `hs256wordlist` subcommand.
///
/// This struct is intentionally close to runtime concepts (JWT, wordlist path,
/// threadgroup width, autotune toggle) so the dispatch layer can pass it
/// straight into `run()` with minimal translation.
#[derive(Debug, Clone, Args)]
pub struct Hs256WordlistArgs {
    /// JWT in compact form (`header.payload.signature`) with `alg=HS256`.
    pub jwt: String,
    /// Wordlist file path (one candidate secret per line).
    #[arg(long, default_value = DEFAULT_WORDLIST_PATH)]
    pub wordlist: PathBuf,
    /// Fixed Metal threadgroup width override (default picks a safe value).
    #[arg(long)]
    pub threads_per_group: Option<usize>,
    /// Number of parser worker threads for mmap chunk scanning.
    #[arg(long, value_parser = parse_positive_usize)]
    pub parser_threads: Option<usize>,
    /// Max number of in-flight batches queued between parser and GPU consumer.
    #[arg(long, value_parser = parse_positive_usize)]
    pub pipeline_depth: Option<usize>,
    /// Number of host packer workers that materialize planned batches.
    #[arg(long, value_parser = parse_positive_usize)]
    pub packer_threads: Option<usize>,
    /// Benchmark several threadgroup widths on the first batch before steady-state dispatch.
    #[arg(long)]
    pub autotune: bool,
}

#[derive(Debug, Clone, Copy)]
// Host-side parser tuning resolved once at command startup.
//
// Keeping this as a plain struct (instead of reading CLI args everywhere)
// makes the producer/parser code easier to test and keeps policy decisions
// (auto thread count, chunk size, queue depth) in one place.
pub(super) struct ParserConfig {
    pub(super) parser_threads: usize,
    pub(super) chunk_bytes: usize,
    pub(super) queue_capacity: usize,
}

impl ParserConfig {
    pub(super) fn from_args(args: &Hs256WordlistArgs) -> Self {
        let auto_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let parser_threads = args.parser_threads.unwrap_or(auto_threads);
        let queue_capacity = parser_threads.saturating_mul(4).max(1);
        Self {
            parser_threads,
            chunk_bytes: DEFAULT_PARSER_CHUNK_BYTES,
            queue_capacity,
        }
    }
}

// Clap parser helper used for `--parser-threads`.
//
// We validate here so CLI errors are reported before any GPU/parser startup
// work begins, and the rest of the code can assume a strictly positive value.
fn parse_positive_usize(input: &str) -> Result<usize, String> {
    let parsed = input
        .parse::<usize>()
        .map_err(|_| format!("invalid integer value: {input}"))?;
    if parsed == 0 {
        return Err("must be > 0".to_string());
    }
    Ok(parsed)
}
