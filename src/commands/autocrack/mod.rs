//! Auto-detecting HMAC-SHA JWT cracker.
//!
//! This module provides a convenience subcommand (`autocrack`) that reads the
//! JWT header's `alg` field and automatically dispatches to the correct
//! algorithm-specific cracker (HS256, HS384, or HS512).
//!
//! It contains no GPU code of its own — it's a thin routing layer that reuses
//! the existing algorithm modules. If the algorithm is not supported, it prints
//! a clear error message and exits.

mod args;
mod command;

pub use args::AutocrackArgs;
pub use command::run;

#[allow(dead_code)]
pub const DEFAULT_WORDLIST_PATH: &str = crate::common::args::DEFAULT_WORDLIST_PATH;
