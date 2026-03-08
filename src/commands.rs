use clap::{Parser, Subcommand};
use std::process::ExitCode;

use crate::args::WordlistArgs;
use crate::gpu::{AesKwVariant, CrackVariant, HmacVariant};

#[derive(Debug, Parser)]
#[command(name = "jotcrack", about = "GPU-assisted JWT/JWE cracking with Metal/CUDA")]
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
    /// Crack a JWE encrypted with A128KW (AES-128 Key Wrap) using a wordlist attack.
    ///
    /// Uses GPU compute (Metal/CUDA) with software AES-128 (10 rounds, 16-byte key)
    /// via S-box lookup tables in constant memory. The RFC 3394 integrity check
    /// value 0xA6A6A6A6A6A6A6A6 serves as the oracle — no known plaintext is needed.
    JweA128kw(WordlistArgs),
    /// Crack a JWE encrypted with A192KW (AES-192 Key Wrap) using a wordlist attack.
    ///
    /// Uses GPU compute (Metal/CUDA) with software AES-192 (12 rounds, 24-byte key)
    /// via S-box lookup tables in constant memory. The RFC 3394 integrity check
    /// value 0xA6A6A6A6A6A6A6A6 serves as the oracle — no known plaintext is needed.
    JweA192kw(WordlistArgs),
    /// Crack a JWE encrypted with A256KW (AES-256 Key Wrap) using a wordlist attack.
    ///
    /// Uses GPU compute (Metal/CUDA) with software AES-256 (14 rounds, 32-byte key)
    /// via S-box lookup tables in constant memory. The RFC 3394 integrity check
    /// value 0xA6A6A6A6A6A6A6A6 serves as the oracle — no known plaintext is needed.
    JweA256kw(WordlistArgs),
    /// Auto-detect the algorithm from the token header and crack it.
    ///
    /// Distinguishes JWT (3-part) from JWE (5-part) compact tokens,
    /// reads the `alg` header field, and routes to the appropriate backend.
    Autocrack(WordlistArgs),
}

/// Parse CLI arguments, dispatch the selected subcommand, and map the result
/// into process exit codes (`0` found, `1` not found, `2` error).
pub fn run() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Hs256wordlist(args) => {
            crate::runner::run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs256), args)
        }
        Commands::Hs384wordlist(args) => {
            crate::runner::run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs384), args)
        }
        Commands::Hs512wordlist(args) => {
            crate::runner::run_wordlist_crack(CrackVariant::Hmac(HmacVariant::Hs512), args)
        }
        Commands::JweA128kw(args) => {
            crate::runner::run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A128kw), args)
        }
        Commands::JweA192kw(args) => {
            crate::runner::run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A192kw), args)
        }
        Commands::JweA256kw(args) => {
            crate::runner::run_wordlist_crack(CrackVariant::JweAesKw(AesKwVariant::A256kw), args)
        }
        Commands::Autocrack(args) => match crate::jwt::detect_token_variant(&args.jwt) {
            Ok(variant) => {
                eprintln!("AUTODETECT: token algorithm is {}", variant.label());
                crate::runner::run_wordlist_crack(variant, args)
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
    use crate::args::DEFAULT_WORDLIST_PATH;
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

    /// The jwe-a128kw subcommand parses correctly and routes to JWE mode.
    #[test]
    fn clap_cli_parses_jwe_a128kw() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "jwe-a128kw",
            "a.b.c.d.e",
        ])
        .unwrap();
        let Commands::JweA128kw(args) = cli.command else {
            panic!("expected JweA128kw");
        };
        assert_eq!(args.jwt, "a.b.c.d.e");
        assert_eq!(args.wordlist, PathBuf::from(DEFAULT_WORDLIST_PATH));
    }

    /// The jwe-a192kw subcommand parses correctly and routes to JWE mode.
    #[test]
    fn clap_cli_parses_jwe_a192kw() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "jwe-a192kw",
            "a.b.c.d.e",
        ])
        .unwrap();
        let Commands::JweA192kw(args) = cli.command else {
            panic!("expected JweA192kw");
        };
        assert_eq!(args.jwt, "a.b.c.d.e");
        assert_eq!(args.wordlist, PathBuf::from(DEFAULT_WORDLIST_PATH));
    }

    /// The jwe-a256kw subcommand parses correctly and routes to JWE mode.
    #[test]
    fn clap_cli_parses_jwe_a256kw() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "jwe-a256kw",
            "a.b.c.d.e",
        ])
        .unwrap();
        let Commands::JweA256kw(args) = cli.command else {
            panic!("expected JweA256kw");
        };
        assert_eq!(args.jwt, "a.b.c.d.e");
        assert_eq!(args.wordlist, PathBuf::from(DEFAULT_WORDLIST_PATH));
    }
}
