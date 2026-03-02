//! CLI argument definitions for the `hs384-wordlist` subcommand.
//!
//! # Learning note: how `clap` derive works
//!
//! The `clap` crate is Rust's most popular command-line argument parser.
//! Instead of building a parser by hand, we use its **derive** feature:
//! annotate a plain struct with `#[derive(Args)]` and clap generates all
//! the parsing, help text, and validation code at compile time.
//!
//! Each `///` doc-comment on a field becomes the `--help` description for
//! that flag.  `#[arg(...)]` attributes control details like default values,
//! custom parsers (`value_parser`), and whether the flag is positional or named.
//!
//! This struct is identical in shape to the HS256 and HS512 variants -- only
//! the JWT algorithm name and struct/type names differ.  In a larger project
//! you might use generics or a macro to deduplicate, but explicit structs keep
//! the per-subcommand help text clear and greppable.

use std::path::PathBuf;

use clap::Args;

use crate::commands::common::args::{
    DEFAULT_WORDLIST_PATH, ParserConfig, parse_positive_usize,
};

pub(crate) const DEFAULT_PIPELINE_DEPTH: usize =
    crate::commands::common::args::DEFAULT_PIPELINE_DEPTH;

/// Command-line arguments for the HS384 wordlist attack.
///
/// Clap's `#[derive(Args)]` macro turns this struct into a fully-featured
/// CLI parser.  Each field annotated with `#[arg(...)]` becomes a named
/// flag; bare fields (like `jwt`) become positional arguments.
#[derive(Debug, Clone, Args)]
pub struct Hs384WordlistArgs {
    /// JWT in compact form (`header.payload.signature`) with `alg=HS384`.
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

impl Hs384WordlistArgs {
    pub(super) fn parser_config(&self) -> ParserConfig {
        ParserConfig::resolve(self.parser_threads)
    }
}
