use std::process::ExitCode;

mod commands;

// Keep `main` intentionally thin: argument parsing, subcommand dispatch, and
// exit-code policy live in `commands::run()` so they can be tested separately.
fn main() -> ExitCode {
    // Returning `ExitCode` directly makes success / not-found / error mapping
    // explicit at the top level without calling `std::process::exit`.
    commands::run()
}
