use clap::{Parser, Subcommand};
use std::process::ExitCode;

pub mod hs256crack;

#[derive(Debug, Parser)]
#[command(name = "jotcrack", about = "GPU-assisted JWT cracking with Metal")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Hs256crack(hs256crack::Hs256CrackArgs),
}

pub fn run() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Hs256crack(args) => hs256crack::run(args),
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
    use clap::error::ErrorKind;
    use std::path::PathBuf;

    #[test]
    fn clap_cli_parses_hs256crack_with_default_wordlist() {
        let cli = Cli::try_parse_from(["jotcrack", "hs256crack", "abc.def.ghi"]).unwrap();

        let Commands::Hs256crack(args) = cli.command;
        assert_eq!(args.jwt, "abc.def.ghi");
        assert_eq!(
            args.wordlist,
            PathBuf::from(hs256crack::DEFAULT_WORDLIST_PATH)
        );
        assert_eq!(args.threads_per_group, None);
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
    fn clap_cli_rejects_unknown_flag_under_hs256crack() {
        let err =
            Cli::try_parse_from(["jotcrack", "hs256crack", "--bogus", "abc.def.ghi"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn clap_cli_rejects_missing_jwt_for_hs256crack() {
        let err = Cli::try_parse_from(["jotcrack", "hs256crack"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn clap_cli_accepts_custom_wordlist() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "hs256crack",
            "abc.def.ghi",
            "--wordlist",
            "custom.txt",
        ])
        .unwrap();

        let Commands::Hs256crack(args) = cli.command;
        assert_eq!(args.wordlist, PathBuf::from("custom.txt"));
    }

    #[test]
    fn clap_cli_accepts_autotune_and_threadgroup_flags() {
        let cli = Cli::try_parse_from([
            "jotcrack",
            "hs256crack",
            "abc.def.ghi",
            "--threads-per-group",
            "512",
            "--autotune",
        ])
        .unwrap();

        let Commands::Hs256crack(args) = cli.command;
        assert_eq!(args.threads_per_group, Some(512));
        assert!(args.autotune);
    }
}
