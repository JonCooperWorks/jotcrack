//! HS384 JWT wordlist cracking command (Metal-backed).
//!
//! # Learning note: why a separate module for HS384?
//!
//! HS384 (HMAC-SHA384) is *not* a distinct algorithm from HS512 (HMAC-SHA512)
//! -- it is the same SHA-512 compression function with two differences:
//!   1. Different initial hash values (IV), defined in FIPS 180-4 section 5.3.4.
//!   2. The output is truncated to 48 bytes (384 bits) instead of 64 bytes.
//!
//! Despite sharing the GPU kernel source file (`hs512wordlist/hs512_wordlist.metal`),
//! this module exists as its own subcommand because:
//!   - The JWT validation enforces `alg=HS384` and expects a 48-byte signature.
//!   - The host-side byte-to-word conversion fills only 6 of 8 u64 words.
//!   - The GPU comparison loop checks 6 words instead of 8.
//!
//! The module follows the same four-file layout as every other subcommand:
//!   `args.rs` (CLI), `jwt.rs` (parsing), `gpu.rs` (Metal bridge), `command.rs` (orchestration).
//!
//! Runtime pipeline:
//! 1. Parse and validate JWT.
//! 2. Initialize Metal runtime.
//! 3. Produce GPU batches from the wordlist in parallel.
//! 4. Dispatch on GPU and report progress/final stats.

mod args;
mod command;
mod jwt;

pub use args::Hs384WordlistArgs;
#[allow(dead_code)]
pub const DEFAULT_WORDLIST_PATH: &str = crate::common::args::DEFAULT_WORDLIST_PATH;
pub use command::run;
