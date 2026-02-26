use clap::{Parser, Subcommand};
use std::process::ExitCode;

pub mod hs256wordlist;

// Top-level CLI parser. Clap derives the argument parsing implementation from
// this type so the rest of the program can work with structured values.
#[derive(Debug, Parser)]
#[command(name = "jotcrack", about = "GPU-assisted JWT cracking with Metal")]
struct Cli {
    // `jotcrack` is organized as subcommands so new cracking modes can be added
    // later without changing the top-level invocation shape.
    #[command(subcommand)]
    command: Commands,
}

// Enum of all supported subcommands. Each variant carries the argument struct
// for that subcommand so dispatch is just a `match`.
#[derive(Debug, Subcommand)]
enum Commands {
    Hs256wordlist(hs256wordlist::Hs256WordlistArgs),
}

/// Parse CLI arguments, dispatch the selected subcommand, and map the command
/// result into process exit codes (`0` found, `1` not found, `2` error).
pub fn run() -> ExitCode {
    let cli = Cli::parse();

    // Centralized dispatch keeps `main()` simple and makes behavior easy to
    // test without shelling out to the binary.
    let result = match cli.command {
        Commands::Hs256wordlist(args) => hs256wordlist::run(args),
    };

    match result {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(err) => {
            // Errors go to stderr so stdout remains reserved for tool output
            // such as "HS256 key: ..." in scripting scenarios.
            eprintln!("ERROR: {err:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;
    use std::path::PathBuf;

    // These tests lock the CLI contract: accepted flags, defaults, and
    // argument-validation failures should stay stable across refactors.
    #[test]
    fn clap_cli_parses_hs256wordlist_with_default_wordlist() {
        let cli = Cli::try_parse_from(["jotcrack", "hs256wordlist", "abc.def.ghi"]).unwrap();

        let Commands::Hs256wordlist(args) = cli.command;
        assert_eq!(args.jwt, "abc.def.ghi");
        assert_eq!(
            args.wordlist,
            PathBuf::from(hs256wordlist::DEFAULT_WORDLIST_PATH)
        );
        assert_eq!(args.threads_per_group, None);
        assert_eq!(args.parser_threads, None);
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

        let Commands::Hs256wordlist(args) = cli.command;
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

        let Commands::Hs256wordlist(args) = cli.command;
        assert_eq!(args.threads_per_group, Some(512));
        assert_eq!(args.parser_threads, Some(3));
        assert!(args.autotune);
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
}
