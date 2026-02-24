use std::process::ExitCode;

mod commands;

fn main() -> ExitCode {
    commands::run()
}
