//! HS512 JWT wordlist cracking command (Metal-backed).
//!
//! # Learning note: HS512 = HMAC-SHA512
//!
//! HS512 uses the full SHA-512 hash: all 8 state words are kept, producing a
//! 64-byte (512-bit) HMAC signature.  Compared to HS256 (SHA-256 based):
//!   - SHA-512 operates on 64-bit words (vs. 32-bit), uses 80 rounds (vs. 64),
//!     and has a 128-byte block size (vs. 64 bytes).
//!   - The HMAC block size is 128 bytes, so keys <= 128 bytes are zero-padded
//!     directly; keys > 128 bytes are hashed first (RFC 2104).
//!
//! This module shares the Metal kernel source with HS384 via
//! `common/hs512_wordlist.metal`.  The kernel file contains entry points for
//! both `hs384_wordlist*` and `hs512_wordlist*`; each Rust-side GPU wrapper
//! selects its own kernel function name and signature byte count.
//!
//! Runtime pipeline:
//! 1. Parse and validate JWT.
//! 2. Initialize Metal runtime.
//! 3. Produce GPU batches from the wordlist in parallel.
//! 4. Dispatch on GPU and report progress/final stats.

mod args;
mod command;
mod gpu;
mod jwt;

pub use args::Hs512WordlistArgs;
#[allow(dead_code)]
pub const DEFAULT_WORDLIST_PATH: &str = crate::commands::common::args::DEFAULT_WORDLIST_PATH;
pub use command::run;
