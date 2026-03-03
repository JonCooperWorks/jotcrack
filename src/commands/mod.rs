//! # Subcommand dispatch and CLI definition
//!
//! This module owns the top-level CLI shape (via [clap]) and maps each
//! subcommand to its implementation. The pattern is:
//!
//! 1. **`Cli`** — a derive-based clap struct that represents the root parser.
//! 2. **`Commands`** — an enum whose variants carry per-subcommand argument structs.
//! 3. **`run()`** — parses args, dispatches, and converts `Result<bool>` into [`ExitCode`].
//!
//! ## Why an enum for subcommands?
//!
//! Clap's `#[derive(Subcommand)]` turns an enum into a CLI subcommand tree.
//! Each variant automatically gets its own `--help` page and argument set.
//! Dispatch is then a simple `match` — the compiler guarantees every variant is
//! handled, so adding a new cracking mode means adding a variant and a match arm.
//!
//! ## Exit-code contract
//!
//! | Code | Meaning        |
//! |------|----------------|
//! | `0`  | Key found      |
//! | `1`  | Key not found  |
//! | `2`  | Runtime error  |
//!
//! This three-way split lets shell scripts distinguish "searched everything, no
//! match" from "something went wrong".

use clap::{Parser, Subcommand};
use std::process::ExitCode;

// `pub(crate)` means "public within this crate but invisible to external users."
// `common` holds shared infrastructure (args, stats, parsers) that every
// subcommand needs but that we do not want to expose as a public API.
// These are `pub` (not `pub(crate)`) because integration tests or benchmarks
// in `tests/` may need to reference their argument types. If you only ever
// access them from within the crate, `pub(crate)` would be more appropriate.
pub mod hs256wordlist;
pub mod hs384wordlist;
pub mod hs512wordlist;
pub mod autocrack;

// Top-level CLI parser. Clap derives the argument parsing implementation from
// this type so the rest of the program can work with structured values.
//
// `#[derive(Parser)]` generates an `impl Cli` with a `parse()` method that
// reads `std::env::args()` and returns a populated `Cli` struct — or prints
// help/error and exits. `try_parse_from()` does the same but returns a
// `Result`, which is what the tests below use.
#[derive(Debug, Parser)]
#[command(name = "jotcrack", about = "GPU-assisted JWT cracking with Metal")]
struct Cli {
    // `jotcrack` is organized as subcommands so new cracking modes can be added
    // later without changing the top-level invocation shape.
    //
    // The `#[command(subcommand)]` attribute tells clap that this field should
    // be populated by one of the variants in the `Commands` enum below.
    #[command(subcommand)]
    command: Commands,
}

// Enum of all supported subcommands. Each variant carries the argument struct
// for that subcommand so dispatch is just a `match`.
//
// This is the "enum dispatch" pattern: each variant wraps a struct that
// implements `clap::Args`. Clap automatically generates a subcommand for each
// variant, using the variant name (lowercased) as the CLI command name.
//
// Adding a new cracking mode is: (1) create a new module, (2) add a variant
// here, (3) add a match arm in `run()`. The compiler enforces exhaustiveness
// so you cannot forget step 3.
#[derive(Debug, Subcommand)]
enum Commands {
    Hs256wordlist(hs256wordlist::Hs256WordlistArgs),
    Hs384wordlist(hs384wordlist::Hs384WordlistArgs),
    Hs512wordlist(hs512wordlist::Hs512WordlistArgs),
    Autocrack(autocrack::AutocrackArgs),
}

/// Parse CLI arguments, dispatch the selected subcommand, and map the command
/// result into process exit codes (`0` found, `1` not found, `2` error).
///
/// Each subcommand's `run()` returns `anyhow::Result<bool>` — `Ok(true)` means
/// "key found", `Ok(false)` means "exhausted wordlist, no match". This function
/// converts that three-state result into numeric exit codes for the shell.
///
/// ## Why `anyhow::Result` instead of a custom error type?
///
/// For a CLI tool, `anyhow` gives us automatic error chaining (via `.context()`)
/// and human-readable formatting (via `{:#}` which prints the full chain). A
/// library would use a typed error enum, but for top-level CLI error reporting,
/// anyhow is more ergonomic.
pub fn run() -> ExitCode {
    // `Cli::parse()` reads from `std::env::args()`. On parse failure it prints
    // a clap-formatted error and calls `std::process::exit(2)` — this never
    // returns. In tests we use `try_parse_from()` which returns `Result` instead.
    let cli = Cli::parse();

    // Centralized dispatch keeps `main()` simple and makes behavior easy to
    // test without shelling out to the binary.
    //
    // Each arm destructures the enum variant to extract the args struct, then
    // passes it to the subcommand's `run()` function. Rust's pattern matching
    // is exhaustive: if you add a `Commands` variant and forget a match arm,
    // the compiler will refuse to build.
    let result = match cli.command {
        Commands::Hs256wordlist(args) => hs256wordlist::run(args),
        Commands::Hs384wordlist(args) => hs384wordlist::run(args),
        Commands::Hs512wordlist(args) => hs512wordlist::run(args),
        Commands::Autocrack(args) => autocrack::run(args),
    };

    // Map the three-state Result<bool> into exit codes.
    // This is the single place that decides what the process returns to the OS.
    match result {
        Ok(true) => ExitCode::from(0),  // Key found
        Ok(false) => ExitCode::from(1), // Exhausted wordlist, no match
        Err(err) => {
            // Errors go to stderr so stdout remains reserved for tool output
            // such as "HS256 key: ..." in scripting scenarios.
            //
            // `{err:#}` uses the "alternate" Display format, which for anyhow
            // errors prints the full chain: "outer context: inner cause".
            eprintln!("ERROR: {err:#}");
            ExitCode::from(2)
        }
    }
}

// `#[cfg(test)]` means this entire module is compiled only when running
// `cargo test`. It does not exist in release builds, so test-only imports
// and helpers add zero cost to the final binary.
#[cfg(test)]
mod tests {
    // `use super::*;` imports everything from the parent module (this file's
    // non-test items) into the test module's scope. This is the standard
    // pattern for test modules in Rust.
    use super::*;
    use clap::error::ErrorKind;
    use std::path::PathBuf;

    // These tests lock the CLI contract: accepted flags, defaults, and
    // argument-validation failures should stay stable across refactors.
    //
    // Why test CLI parsing separately? Because `Cli::parse()` calls
    // `std::process::exit` on failure, making it untestable. Using
    // `Cli::try_parse_from()` returns a `Result` so we can assert on both
    // success and specific error kinds (e.g., `MissingRequiredArgument`).
    //
    // These tests act as a "contract test" — if someone renames a flag or
    // changes a default, the test will fail, preventing accidental breakage
    // for users who depend on the CLI interface.
    #[test]
    fn clap_cli_parses_hs256wordlist_with_default_wordlist() {
        let cli = Cli::try_parse_from(["jotcrack", "hs256wordlist", "abc.def.ghi"]).unwrap();

        // `let ... else` is a Rust pattern for "irrefutable destructuring with
        // a diverging else branch." If the enum variant does not match, the
        // `else` block must diverge (panic, return, break, etc.). This is
        // cleaner than a full `match` when you only care about one variant.
        let Commands::Hs256wordlist(args) = cli.command else {
            panic!("expected Hs256wordlist");
        };
        assert_eq!(args.jwt, "abc.def.ghi");
        // Verify the default wordlist path is applied when `--wordlist` is omitted.
        assert_eq!(
            args.wordlist,
            PathBuf::from(hs256wordlist::DEFAULT_WORDLIST_PATH)
        );
        // `None` means "user did not provide this flag" — the subcommand's
        // `run()` will apply its own defaults via `ParserConfig::resolve()`.
        assert_eq!(args.threads_per_group, None);
        assert_eq!(args.parser_threads, None);
        assert_eq!(args.pipeline_depth, None);
        assert_eq!(args.packer_threads, None);
        assert!(!args.autotune);
    }

    #[test]
    fn clap_cli_rejects_missing_subcommand() {
        let err = Cli::try_parse_from(["jotcrack", "abc.def.ghi"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn clap_cli_rejects_unknown_subcommand() {
        let err = Cli::try_parse_from(["jotcrack", "bogus", "abc.def.ghi"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn clap_cli_rejects_unknown_flag_under_hs256wordlist() {
        // Verifies Clap scopes flags to the selected subcommand instead of
        // silently accepting unsupported options.
        let err = Cli::try_parse_from(["jotcrack", "hs256wordlist", "--bogus", "abc.def.ghi"])
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn clap_cli_rejects_missing_jwt_for_hs256wordlist() {
        let err = Cli::try_parse_from(["jotcrack", "hs256wordlist"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn clap_cli_accepts_custom_wordlist() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "hs256wordlist",
            "abc.def.ghi",
            "--wordlist",
            "custom.txt",
        ])
        .unwrap();

        let Commands::Hs256wordlist(args) = cli.command else {
            panic!("expected Hs256wordlist");
        };
        assert_eq!(args.wordlist, PathBuf::from("custom.txt"));
    }

    #[test]
    fn clap_cli_accepts_autotune_and_threadgroup_flags() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "hs256wordlist",
            "abc.def.ghi",
            "--threads-per-group",
            "512",
            "--parser-threads",
            "3",
            "--autotune",
        ])
        .unwrap();

        let Commands::Hs256wordlist(args) = cli.command else {
            panic!("expected Hs256wordlist");
        };
        assert_eq!(args.threads_per_group, Some(512));
        assert_eq!(args.parser_threads, Some(3));
        assert_eq!(args.pipeline_depth, None);
        assert_eq!(args.packer_threads, None);
        assert!(args.autotune);
    }

    #[test]
    fn clap_cli_accepts_pipeline_depth() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "hs256wordlist",
            "abc.def.ghi",
            "--pipeline-depth",
            "10",
        ])
        .unwrap();

        let Commands::Hs256wordlist(args) = cli.command else {
            panic!("expected Hs256wordlist");
        };
        assert_eq!(args.pipeline_depth, Some(10));
    }

    #[test]
    fn clap_cli_accepts_packer_threads() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "hs256wordlist",
            "abc.def.ghi",
            "--packer-threads",
            "3",
        ])
        .unwrap();

        let Commands::Hs256wordlist(args) = cli.command else {
            panic!("expected Hs256wordlist");
        };
        assert_eq!(args.packer_threads, Some(3));
    }

    #[test]
    fn clap_cli_rejects_zero_parser_threads() {
        let err = Cli::try_parse_from([
            "jotcrack",
            "hs256wordlist",
            "abc.def.ghi",
            "--parser-threads",
            "0",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn clap_cli_rejects_zero_pipeline_depth() {
        let err = Cli::try_parse_from([
            "jotcrack",
            "hs256wordlist",
            "abc.def.ghi",
            "--pipeline-depth",
            "0",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn clap_cli_parses_autocrack_with_default_wordlist() {
        let cli = Cli::try_parse_from(["jotcrack", "autocrack", "abc.def.ghi"]).unwrap();

        let Commands::Autocrack(args) = cli.command else {
            panic!("expected Autocrack");
        };
        assert_eq!(args.jwt, "abc.def.ghi");
        assert_eq!(
            args.wordlist,
            PathBuf::from(autocrack::DEFAULT_WORDLIST_PATH)
        );
        assert_eq!(args.threads_per_group, None);
        assert_eq!(args.parser_threads, None);
        assert_eq!(args.pipeline_depth, None);
        assert_eq!(args.packer_threads, None);
        assert!(!args.autotune);
    }

    #[test]
    fn clap_cli_rejects_zero_packer_threads() {
        let err = Cli::try_parse_from([
            "jotcrack",
            "hs256wordlist",
            "abc.def.ghi",
            "--packer-threads",
            "0",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ValueValidation);
    }
}
