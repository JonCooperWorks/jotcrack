//! HS256 JWT wordlist cracking command (Metal-backed).
//!
//! Runtime pipeline:
//! 1. Parse and validate JWT.
//! 2. Initialize Metal runtime.
//! 3. Produce GPU batches from the wordlist in parallel.
//! 4. Dispatch on GPU and report progress/final stats.

mod args;
mod batch;
mod command;
mod gpu;
mod jwt;
mod parser;
mod producer;
mod stats;

#[cfg(test)]
mod test_support;

pub use args::Hs256WordlistArgs;
#[allow(dead_code)]
pub const DEFAULT_WORDLIST_PATH: &str = args::DEFAULT_WORDLIST_PATH;
pub use command::run;
