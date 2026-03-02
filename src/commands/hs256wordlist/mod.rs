//! HS256 JWT wordlist cracking command (Metal-backed).
//!
//! # Module architecture
//!
//! This module is organized following a common Rust pattern for non-trivial
//! commands: split the concern into small, focused files and re-export only
//! the public surface from `mod.rs`.
//!
//! ## Why split into multiple files?
//!
//! Each file owns a single responsibility in the cracking pipeline:
//!
//! - **`args.rs`** -- CLI argument parsing via `clap` derive macros.
//! - **`jwt.rs`** -- JWT parsing and cryptographic validation (pure CPU logic).
//! - **`gpu.rs`** -- Metal GPU setup, kernel compilation, dispatch mechanics.
//! - **`command.rs`** -- The main run loop that ties everything together with a
//!   double-buffered producer/consumer pattern for maximum throughput.
//!
//! This separation keeps each file short enough to reason about independently
//! and makes it easy to unit-test parsing, GPU dispatch, and orchestration
//! in isolation.
//!
//! ## Runtime pipeline
//!
//! 1. Parse and validate JWT (ensure it is HS256, extract signing input + target signature).
//! 2. Initialize Metal runtime (compile shaders, allocate shared buffers).
//! 3. Produce GPU batches from the wordlist in parallel (mmap + multi-threaded parsing).
//! 4. Dispatch on GPU and report progress/final stats (double-buffered overlap).
//!
//! ## Re-exports
//!
//! Only the argument struct (`Hs256WordlistArgs`) and the entry point (`run`)
//! are public. Everything else is `pub(super)` or private, keeping the API
//! surface minimal -- a good Rust practice that prevents accidental coupling
//! between unrelated parts of the codebase.

mod args;
mod command;
mod gpu;
mod jwt;

pub use args::Hs256WordlistArgs;
#[allow(dead_code)]
pub const DEFAULT_WORDLIST_PATH: &str = crate::commands::common::args::DEFAULT_WORDLIST_PATH;
pub use command::run;
