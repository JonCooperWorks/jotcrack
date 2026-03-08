//! # jotcrack — GPU-assisted JWT cracking with Metal
//!
//! This is the binary entry point. It is intentionally thin: all argument parsing,
//! subcommand dispatch, and exit-code policy live in [`commands::run()`] so they
//! can be unit-tested without spawning a child process.
//!
//! ## Why is `main` so small?
//!
//! In Rust, `main` can return types that implement [`std::process::Termination`].
//! By returning [`ExitCode`] directly we get clean process exit codes without
//! ever calling [`std::process::exit`], which would skip destructors and flush
//! operations. Keeping logic out of `main` also means tests can call
//! `commands::run()` (or its helpers) as a regular function, making CI faster
//! and assertions easier.

use std::process::ExitCode;

mod args;
mod batch;
mod commands;
mod gpu;
mod jwt;
#[cfg_attr(all(target_os = "linux", not(test)), allow(dead_code))]
mod parser;
mod producer;
mod runner;
mod stats;

#[cfg(test)]
mod test_support;

// Keep `main` intentionally thin: argument parsing, subcommand dispatch, and
// exit-code policy live in `commands::run()` so they can be tested separately.
//
// Returning `ExitCode` instead of `Result<(), ...>` gives us control over the
// exact numeric exit code — important for scripting (`0` = found, `1` = not
// found, `2` = error). If we returned `Result`, Rust would print the Debug
// representation and always exit with code 1 on `Err`.
fn main() -> ExitCode {
    // Returning `ExitCode` directly makes success / not-found / error mapping
    // explicit at the top level without calling `std::process::exit`.
    commands::run()
}
