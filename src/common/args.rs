use std::path::PathBuf;
use clap::Args;

pub const DEFAULT_WORDLIST_PATH: &str = "breach.txt";

pub(crate) const DEFAULT_PIPELINE_DEPTH: usize = 10;

/// Shared CLI arguments for all wordlist cracking subcommands.
#[derive(Debug, Clone, Args)]
pub struct WordlistArgs {
    /// JWT in compact form (`header.payload.signature`).
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

impl WordlistArgs {
    pub(crate) fn parser_config(&self) -> ParserConfig {
        ParserConfig::resolve(self.parser_threads)
    }
}

/// Size of each chunk the parallel parser reads from the mmap'd wordlist.
/// 16 MiB is large enough to amortize per-chunk overhead but small enough
/// that `parser_threads` chunks fit comfortably in L3 cache on Apple Silicon.
pub(crate) const DEFAULT_PARSER_CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// Resolved configuration for the parallel wordlist parser.
///
/// ## Why `Clone` + `Copy`?
///
/// `ParserConfig` is small (three `usize` fields = 24 bytes on 64-bit). Deriving
/// `Copy` lets it be passed by value without moves, which is simpler and avoids
/// lifetime annotations. A good rule of thumb: if your struct is smaller than a
/// few pointers and contains no heap-allocated data, `Copy` is a win.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParserConfig {
    pub(crate) parser_threads: usize,
    pub(crate) chunk_bytes: usize,
    /// How many parsed batches can be queued before the parser threads block.
    /// Sized as a multiple of `parser_threads` so each thread can have several
    /// batches in flight, smoothing out variance in batch-build time.
    pub(crate) queue_capacity: usize,
}

impl ParserConfig {
    /// Build a complete config, filling in auto-detected defaults for any
    /// values the user did not specify.
    ///
    /// ## The `Option` pattern for CLI flags
    ///
    /// Clap stores optional flags as `Option<T>`. This method uses
    /// `unwrap_or(auto_detected_value)` to merge user-supplied and default
    /// values in one place. This is better than scattering default logic
    /// across clap annotations because it can use runtime information
    /// (like `available_parallelism()`).
    pub(crate) fn resolve(parser_threads: Option<usize>) -> Self {
        // `available_parallelism()` returns the number of logical CPUs.
        // We subtract 1 to leave a core free for the GPU consumer thread,
        // then clamp to at least 1. `.map()` operates on the `Ok` variant;
        // `.unwrap_or(1)` handles the case where the OS cannot report core count.
        let auto_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let parser_threads = parser_threads.unwrap_or(auto_threads);
        // `saturating_mul` avoids panic on overflow — returns `usize::MAX`
        // instead. Defensive arithmetic matters in config paths where user
        // input can be arbitrarily large.
        let queue_capacity = parser_threads.saturating_mul(4).max(1);
        Self {
            parser_threads,
            chunk_bytes: DEFAULT_PARSER_CHUNK_BYTES,
            queue_capacity,
        }
    }
}

/// Custom clap `value_parser` that rejects zero.
///
/// ## Why a custom parser instead of clap's `value_parser!(1..)`?
///
/// Clap's built-in range parsers work for simple cases, but a function gives us
/// control over the exact error message. Clap calls this function with the raw
/// string from the CLI; if it returns `Err(String)`, clap shows the error and
/// exits with a `ValueValidation` error kind.
///
/// Usage in a clap struct field:
/// ```ignore
/// #[arg(long, value_parser = parse_positive_usize)]
/// parser_threads: Option<usize>,
/// ```
pub fn parse_positive_usize(input: &str) -> Result<usize, String> {
    let parsed = input
        .parse::<usize>()
        // `.map_err()` converts the `ParseIntError` into a `String` that clap
        // can display. The `?` operator then short-circuits on error.
        .map_err(|_| format!("invalid integer value: {input}"))?;
    if parsed == 0 {
        return Err("must be > 0".to_string());
    }
    Ok(parsed)
}
