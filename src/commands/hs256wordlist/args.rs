//! CLI argument definitions for the HS256 wordlist cracking command.
//!
//! # Clap derive pattern
//!
//! Instead of manually building CLI parsers, Rust's `clap` crate lets you
//! define a struct and annotate it with `#[derive(Args)]`. The derive macro
//! generates all the parsing, validation, and help text code at compile time.
//!
//! This is called the **derive pattern** (as opposed to the "builder pattern"
//! where you chain method calls). Benefits:
//! - Your argument definitions are a single source of truth (struct fields).
//! - Adding a new flag is just adding a new field + doc comment.
//! - `clap` automatically generates `--help` text from `///` doc comments.
//!
//! # Why `Option<usize>` instead of `usize`?
//!
//! Fields like `threads_per_group` are `Option<usize>` because they are
//! *optional* flags. When the user omits them, the program picks a sensible
//! default at runtime (e.g., querying the GPU for its hardware limits).
//! If they were plain `usize`, `clap` would require a `default_value`, and
//! we would lose the ability to distinguish "user explicitly set this" from
//! "use the smart default".

use std::path::PathBuf;

use clap::Args;

use crate::common::args::{
    DEFAULT_WORDLIST_PATH, ParserConfig, parse_positive_usize,
};

pub(crate) const DEFAULT_PIPELINE_DEPTH: usize =
    crate::common::args::DEFAULT_PIPELINE_DEPTH;

/// Command-line arguments for the `hs256-wordlist` subcommand.
///
/// `#[derive(Args)]` tells `clap` to generate a CLI parser from this struct.
/// Each field becomes a CLI flag, and `///` doc comments become the help text
/// shown by `--help`. The `#[arg(...)]` attributes configure parsing behavior
/// like default values and custom validation functions.
///
/// `#[derive(Debug, Clone)]` are standard Rust derives:
/// - `Debug` lets you `println!("{:?}", args)` for logging.
/// - `Clone` lets you duplicate the struct (needed because we pass it around).
#[derive(Debug, Clone, Args)]
pub struct Hs256WordlistArgs {
    /// JWT in compact form (`header.payload.signature`) with `alg=HS256`.
    // This is a *positional* argument (no `#[arg(long)]`), so the user
    // provides it directly: `jotcrack hs256-wordlist "eyJ..."`.
    pub jwt: String,
    /// Wordlist file path (one candidate secret per line).
    // `#[arg(long)]` makes this a named flag: `--wordlist /path/to/file`.
    // `default_value` uses the constant so there is a fallback if omitted.
    #[arg(long, default_value = DEFAULT_WORDLIST_PATH)]
    pub wordlist: PathBuf,
    /// Fixed Metal threadgroup width override (default picks a safe value).
    // Metal dispatches work in groups of threads called "threadgroups".
    // The GPU has a hardware maximum (e.g., 1024). This flag lets the user
    // experiment or force a specific width for benchmarking.
    #[arg(long)]
    pub threads_per_group: Option<usize>,
    /// Number of parser worker threads for mmap chunk scanning.
    // `value_parser = parse_positive_usize` provides custom validation:
    // clap will reject 0 or negative values with a helpful error message,
    // instead of silently accepting invalid input.
    #[arg(long, value_parser = parse_positive_usize)]
    pub parser_threads: Option<usize>,
    /// Max number of in-flight batches queued between parser and GPU consumer.
    // Controls how many batches the producer can prepare ahead of the GPU
    // consumer. Higher depth = more memory usage but better overlap between
    // CPU parsing and GPU execution. Think of it like a conveyor belt buffer.
    #[arg(long, value_parser = parse_positive_usize)]
    pub pipeline_depth: Option<usize>,
    /// Number of host packer workers that materialize planned batches.
    #[arg(long, value_parser = parse_positive_usize)]
    pub packer_threads: Option<usize>,
    /// Benchmark several threadgroup widths on the first batch before steady-state dispatch.
    // Boolean flags like this are `false` by default and become `true` when
    // the user passes `--autotune`. No value needed after the flag.
    #[arg(long)]
    pub autotune: bool,
}

impl Hs256WordlistArgs {
    /// Convert the user-specified (or defaulted) parser thread count into a
    /// resolved `ParserConfig`. The `pub(super)` visibility means this method
    /// is accessible within the parent module (`hs256wordlist`) but not outside
    /// it -- a Rust visibility pattern that keeps implementation details private.
    pub(super) fn parser_config(&self) -> ParserConfig {
        ParserConfig::resolve(self.parser_threads)
    }
}
