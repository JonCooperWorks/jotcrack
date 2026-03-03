use clap::{Parser, Subcommand};
use std::process::ExitCode;

use crate::common::args::WordlistArgs;
use crate::common::gpu::HmacVariant;

#[derive(Debug, Parser)]
#[command(name = "jotcrack", about = "GPU-assisted JWT cracking with Metal")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Crack an HS256-signed JWT using a wordlist attack.
    Hs256wordlist(WordlistArgs),
    /// Crack an HS384-signed JWT using a wordlist attack.
    Hs384wordlist(WordlistArgs),
    /// Crack an HS512-signed JWT using a wordlist attack.
    Hs512wordlist(WordlistArgs),
    /// Auto-detect the HMAC-SHA algorithm from the JWT header and crack it.
    Autocrack(WordlistArgs),
}

/// Parse CLI arguments, dispatch the selected subcommand, and map the result
/// into process exit codes (`0` found, `1` not found, `2` error).
pub fn run() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Hs256wordlist(args) => {
            crate::common::runner::run_wordlist_crack(HmacVariant::Hs256, args)
        }
        Commands::Hs384wordlist(args) => {
            crate::common::runner::run_wordlist_crack(HmacVariant::Hs384, args)
        }
        Commands::Hs512wordlist(args) => {
            crate::common::runner::run_wordlist_crack(HmacVariant::Hs512, args)
        }
        Commands::Autocrack(args) => match crate::common::jwt::detect_variant(&args.jwt) {
            Ok(variant) => {
                eprintln!("AUTODETECT: JWT algorithm is {}", variant.label());
                crate::common::runner::run_wordlist_crack(variant, args)
            }
            Err(e) => Err(e),
        },
    };

    match result {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(err) => {
            eprintln!("ERROR: {err:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::args::DEFAULT_WORDLIST_PATH;
    use clap::error::ErrorKind;
    use std::path::PathBuf;

    #[test]
    fn clap_cli_parses_hs256wordlist_with_default_wordlist() {
        let cli = Cli::try_parse_from(["jotcrack", "hs256wordlist", "abc.def.ghi"]).unwrap();
        let Commands::Hs256wordlist(args) = cli.command else {
            panic!("expected Hs256wordlist");
        };
        assert_eq!(args.jwt, "abc.def.ghi");
        assert_eq!(args.wordlist, PathBuf::from(DEFAULT_WORDLIST_PATH));
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
        assert_eq!(args.wordlist, PathBuf::from(DEFAULT_WORDLIST_PATH));
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
