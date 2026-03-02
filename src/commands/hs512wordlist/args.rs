//! CLI argument definitions for the `hs512-wordlist` subcommand.
//!
//! # Learning note: structural repetition across subcommands
//!
//! This file is nearly identical to `hs384wordlist/args.rs` and
//! `hs256wordlist/args.rs`.  Each subcommand gets its own args struct so
//! that clap can generate distinct help text and the type system prevents
//! accidentally passing HS256 args to the HS512 runner.
//!
//! The `value_parser = parse_positive_usize` attribute tells clap to reject
//! zero or negative values at parse time, before our code ever sees them.
//! This is a good pattern: push validation as close to the input boundary
//! as possible so inner code can assume valid data.

use std::path::PathBuf;

use clap::Args;

use crate::commands::common::args::{
    DEFAULT_WORDLIST_PATH, ParserConfig, parse_positive_usize,
};

pub(crate) const DEFAULT_PIPELINE_DEPTH: usize =
    crate::commands::common::args::DEFAULT_PIPELINE_DEPTH;

/// Command-line arguments for the HS512 wordlist attack.
///
/// Same structure as the HS384 variant; only the JWT algorithm name differs.
#[derive(Debug, Clone, Args)]
pub struct Hs512WordlistArgs {
    /// JWT in compact form (`header.payload.signature`) with `alg=HS512`.
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

impl Hs512WordlistArgs {
    pub(super) fn parser_config(&self) -> ParserConfig {
        ParserConfig::resolve(self.parser_threads)
    }
}
