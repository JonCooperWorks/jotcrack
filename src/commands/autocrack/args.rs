//! CLI argument definition for the `autocrack` subcommand.
//!
//! The args mirror the algorithm-specific commands but accept any HS256/HS384/HS512
//! JWT — the algorithm is detected at runtime from the JWT header.

use std::path::PathBuf;

use clap::Args;

use crate::common::args::{DEFAULT_WORDLIST_PATH, parse_positive_usize};

/// Automatically detect the HMAC-SHA algorithm from the JWT header and crack it.
///
/// Supported algorithms: HS256, HS384, HS512. The JWT header's `alg` field is
/// read to determine which GPU kernel to use. If the algorithm is unsupported,
/// we print a clear error and exit.
#[derive(Debug, Clone, Args)]
pub struct AutocrackArgs {
    /// JWT in compact form (`header.payload.signature`) with `alg` set to
    /// HS256, HS384, or HS512.
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

    /// Benchmark several threadgroup widths on the first batch before
    /// steady-state dispatch.
    #[arg(long)]
    pub autotune: bool,
}
