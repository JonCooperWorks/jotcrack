//! # Shared infrastructure for all subcommands
//!
//! This module re-exports the building blocks that every cracking subcommand
//! shares: argument parsing/resolution, GPU batch types, wordlist parsers,
//! producer/consumer pipeline helpers, and timing/statistics reporting.
//!
//! ## Module organization
//!
//! Rust's module system maps directly to the filesystem. This file
//! (`common/mod.rs`) is the "root" of the `common` module. Each `pub(crate) mod`
//! declaration below makes a child module visible to the rest of the crate.
//!
//! ## Visibility: `pub(crate)` vs `pub`
//!
//! - **`pub(crate)`** — visible to all code in this crate, but not to external
//!   consumers (other crates, integration tests in `tests/`). Use this when a
//!   type is an internal implementation detail that multiple modules need.
//! - **`pub`** — visible to everyone, including external crates. We use this
//!   sparingly: only on items that integration tests or benchmarks need.
//! - **`pub(super)`** — visible only to the parent module. Useful for helpers
//!   that should not leak further up the module tree.
//!
//! The `common` module itself is declared `pub(crate)` in its parent
//! (`commands/mod.rs`), so even though items here are `pub(crate)`, they are
//! still invisible outside the crate.

// Each child module provides one piece of the pipeline:
//   args     — CLI argument types, defaults, and resolution logic
//   batch    — GPU batch buffer types (the data that crosses the CPU-GPU boundary)
//   parser   — Wordlist file reading and parallel chunked parsing
//   producer — Producer/consumer pipeline that feeds parsed batches to the GPU
//   stats    — Timing structs and human-readable reporting
pub(crate) mod args;
pub(crate) mod batch;
pub(crate) mod parser;
pub(crate) mod producer;
pub(crate) mod stats;

// `#[cfg(test)]` means this module only exists in test builds. It provides
// helpers like `test_device()`, `write_temp_wordlist()`, and `make_test_jwt()`
// that multiple test modules across the crate need.
//
// The `pub(crate)` visibility lets test modules in other files (e.g.,
// `hs256wordlist/tests.rs`) import these helpers, while `#[cfg(test)]` ensures
// they add zero overhead to release builds.
#[cfg(test)]
pub(crate) mod test_support;
