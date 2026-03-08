//! # Test helpers shared across all subcommand tests
//!
//! This module exists only in test builds (`#![cfg(test)]` at the file level).
//! It provides reusable utilities that multiple test modules need:
//!
//! - **`test_device()`** — Obtain the system Metal GPU (panics if unavailable).
//! - **`write_temp_wordlist()`** — Create a unique temp file from bytes.
//! - **`mmap_reader_from_temp_file()`** — Write bytes, mmap them, return a reader.
//! - **`make_test_jwt()`** — Construct a valid HMAC-signed JWT from scratch.
//!
//! ## `#![cfg(test)]` vs `#[cfg(test)]`
//!
//! The `!` makes this a *crate-level* (inner) attribute that applies to the
//! entire file. It is equivalent to wrapping every item in the file with
//! `#[cfg(test)]`. This is the standard idiom for test-only modules.
//!
//! ## Why are `hmac`, `sha2`, and `base64` used here but not in production code?
//!
//! These crates are listed under `[dev-dependencies]` in `Cargo.toml`. Rust
//! only links dev-dependencies into test/bench builds, so they add zero bytes
//! to the release binary. The production GPU code computes HMAC-SHA via Metal
//! shaders, but tests need a CPU-side reference implementation to create valid
//! JWTs for end-to-end testing.

// `#![cfg(test)]` — this entire file is compiled only when running `cargo test`.
#![cfg(test)]

use std::fs::{self, File as StdFile};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

// `memmap2` provides cross-platform memory-mapped file I/O. Mmap lets us treat
// a file as a byte slice without reading it all into a `Vec<u8>`, which is
// important for large wordlists (hundreds of MBs).
use memmap2::MmapOptions;
use super::gpu::{AesKwVariant, GpuDevice};

use super::args::ParserConfig;
use super::parser::{MmapWordlistBatchReader, ParallelMmapWordlistBatchReader};

// A global atomic counter used to generate unique temp file names across tests
// running in parallel. `static` means one instance for the entire process.
// `AtomicU64` provides lock-free thread-safe increment via `fetch_add`.
//
// Why `Ordering::Relaxed`? We only need uniqueness, not happens-before ordering
// with respect to other memory operations. `Relaxed` is the cheapest atomic
// ordering and is sufficient for a simple counter.
static TEMP_WORDLIST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Obtain the default Metal GPU device, or panic if none is available.
///
/// Tests that need GPU access call this helper. The `expect()` provides a clear
/// panic message — much better than `unwrap()` which would just say "None".
/// In CI environments without a GPU, these tests will fail fast with an
/// understandable message.
pub(crate) fn test_device() -> GpuDevice {
    super::gpu::default_device().expect("GPU device is required for tests")
}

/// Write raw bytes to a uniquely-named temporary file and return its path.
///
/// Uniqueness comes from three sources combined:
/// - `std::process::id()` — unique per process (parallel `cargo test` workers)
/// - nanosecond timestamp — unique per instant
/// - atomic counter — unique per call within the same nanosecond
///
/// This triple guarantee avoids collisions even under aggressive parallelism
/// (`cargo test` runs tests on multiple threads by default).
///
/// ## Lifetime note
///
/// The `bytes: &[u8]` parameter is a borrowed slice — this function does not
/// take ownership of the data. The slice only needs to live until `fs::write`
/// copies it to disk, which happens before this function returns. No lifetime
/// annotation is needed because Rust's elision rules handle single-reference
/// parameters automatically.
pub(crate) fn write_temp_wordlist(bytes: &[u8]) -> std::path::PathBuf {
    let unique = TEMP_WORDLIST_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "jotcrack-hs256wordlist-test-{}-{nanos}-{unique}.txt",
        std::process::id()
    ));
    fs::write(&path, bytes).expect("failed to write temp wordlist");
    path
}

/// Create a temp wordlist file, memory-map it, and return a sequential reader.
///
/// ## Why `unsafe` for mmap?
///
/// `MmapOptions::new().map(&file)` is `unsafe` because memory-mapped files
/// have a fundamental soundness issue: another process (or even another thread)
/// could modify or truncate the underlying file while we hold the mmap, causing
/// undefined behavior (the mapped memory could become invalid). Rust's type
/// system cannot prevent this because the danger comes from *outside* the
/// process.
///
/// In practice, our temp files are written once and never modified, so the
/// `unsafe` is safe here. But the `unsafe` block documents the assumption:
/// "we promise no one else will modify this file while the mmap exists."
pub(crate) fn mmap_reader_from_temp_file(
    bytes: &[u8],
) -> (MmapWordlistBatchReader, std::path::PathBuf) {
    let path = write_temp_wordlist(bytes);
    let file = StdFile::open(&path).expect("failed to open temp wordlist");
    let mmap = unsafe { MmapOptions::new().map(&file) }.expect("failed to mmap temp wordlist");
    (MmapWordlistBatchReader::new(test_device(), mmap), path)
}

/// Build a `ParserConfig` with explicit values (no auto-detection).
///
/// Tests use this instead of `ParserConfig::resolve()` to get deterministic,
/// reproducible configurations. `resolve()` auto-detects CPU count, which
/// varies across machines and would make tests flaky.
pub(crate) fn test_parser_config(parser_threads: usize, chunk_bytes: usize) -> ParserConfig {
    ParserConfig {
        parser_threads,
        chunk_bytes,
        queue_capacity: parser_threads.saturating_mul(4).max(1),
    }
}

/// Like `mmap_reader_from_temp_file`, but returns the parallel (multi-threaded)
/// reader variant. The `parser_threads` and `chunk_bytes` parameters let tests
/// exercise edge cases like single-threaded parsing or tiny chunk sizes.
pub(crate) fn parallel_mmap_reader_from_temp_file(
    bytes: &[u8],
    parser_threads: usize,
    chunk_bytes: usize,
) -> (ParallelMmapWordlistBatchReader, std::path::PathBuf) {
    let path = write_temp_wordlist(bytes);
    let file = StdFile::open(&path).expect("failed to open temp wordlist");
    // Same `unsafe` rationale as `mmap_reader_from_temp_file` above.
    let mmap = unsafe { MmapOptions::new().map(&file) }.expect("failed to mmap temp wordlist");
    let reader = ParallelMmapWordlistBatchReader::new(
        test_device(),
        mmap,
        test_parser_config(parser_threads, chunk_bytes),
    )
    .expect("failed to build parallel mmap reader");
    (reader, path)
}

/// Drain all words from a sequential reader into a `Vec<Vec<u8>>`.
///
/// This is a test-only convenience. Production code processes batches one at a
/// time to bound memory usage; tests collect everything to make assertions
/// easier (e.g., "the reader produced exactly these words in this order").
///
/// ## Pattern: `while let Some(...)`
///
/// `while let` is Rust sugar for "loop, pattern-match each iteration, break
/// when the pattern does not match." It is the idiomatic way to drain an
/// iterator-like API that returns `Option`.
pub(crate) fn collect_all_words_from_mmap_reader(
    reader: &mut MmapWordlistBatchReader,
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(batch) = reader
        .next_batch_reusing(None)
        .expect("sequential mmap batch")
    {
        for i in 0..batch.candidate_count() {
            // `.to_vec()` copies the borrowed `&[u8]` slice into an owned
            // `Vec<u8>`. We must copy because the batch buffer will be reused
            // or dropped on the next iteration.
            out.push(batch.word(i).expect("candidate").to_vec());
        }
    }
    out
}

/// Construct a valid HMAC-signed JWT for testing.
///
/// A JWT has three base64url-encoded parts separated by dots:
///   `<header>.<payload>.<signature>`
///
/// The signature is `HMAC-SHA(secret, "<header>.<payload>")`. The GPU shader
/// computes this same HMAC for each candidate password and compares the result
/// to the JWT's signature. This helper builds a JWT with a known secret so
/// tests can verify that the GPU shader produces a match.
///
/// ## Why `use` inside the function body?
///
/// The `use base64::...` and `use hmac::...` statements are scoped to this
/// function. This is legal in Rust and keeps these dev-dependency imports
/// from cluttering the module namespace. It also makes it visually clear
/// that these crates are only needed for JWT construction.
///
/// ## HMAC construction pattern
///
/// 1. `Hmac::<Sha256>::new_from_slice(secret)` — creates a MAC instance keyed
///    with the secret. Returns `Result` because keys can be rejected (though
///    HMAC accepts any key length, so `unwrap()` is safe here).
/// 2. `.update(data)` — feeds data into the MAC. Can be called multiple times.
/// 3. `.finalize().into_bytes()` — produces the fixed-size authentication tag.
///    `.into_bytes()` converts the `CtOutput` wrapper into a `GenericArray`.
pub(crate) fn make_test_jwt(alg: &str, payload_json: &str, secret: &[u8]) -> String {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use hmac::{Hmac, Mac};

    // Build the JWT header. `r#"..."#` is a raw string literal — backslashes
    // and quotes inside are treated literally, which avoids escaping hell
    // when writing JSON.
    let header = format!(r#"{{"alg":"{}","typ":"JWT"}}"#, alg);
    // JWTs use base64url encoding (URL_SAFE) without padding (NO_PAD), which
    // differs from standard base64 (uses `-` and `_` instead of `+` and `/`).
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    // The "signing input" is the part that gets HMAC'd — header.payload
    // concatenated with a literal dot.
    let signing_input = format!("{}.{}", header_b64, payload_b64);

    // Compute the HMAC signature using the CPU-side reference implementation.
    // The GPU shader must produce the exact same bytes for a match to be found.
    let signature = match alg {
        "HS256" => {
            let mut mac = Hmac::<sha2::Sha256>::new_from_slice(secret).unwrap();
            mac.update(signing_input.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        "HS384" => {
            let mut mac = Hmac::<sha2::Sha384>::new_from_slice(secret).unwrap();
            mac.update(signing_input.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        "HS512" => {
            let mut mac = Hmac::<sha2::Sha512>::new_from_slice(secret).unwrap();
            mac.update(signing_input.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        _ => panic!("unsupported algorithm: {}", alg),
    };

    // Assemble the final JWT: header.payload.signature
    format!("{}.{}", signing_input, URL_SAFE_NO_PAD.encode(&signature))
}

/// Construct a valid JWE compact token for any AES Key Wrap variant.
///
/// Supports A128KW, A192KW, and A256KW. The token wraps a randomly-generated
/// 128-bit Content Encryption Key (CEK) using AES Key Wrap (RFC 3394) with a
/// key derived from `secret`:
///
/// - `secret` ≤ N bytes → zero-padded to N bytes
/// - `secret` > N bytes → SHA-256 hashed and truncated to N bytes
///
/// Where N = `variant.key_bytes()` (16 / 24 / 32 for A128KW / A192KW / A256KW).
///
/// This matches the key derivation logic in the Metal compute shader
/// (`aeskw_wordlist.metal`). The IV, ciphertext, and tag segments are
/// dummy bytes — they are irrelevant for the Key Wrap integrity attack
/// which only checks the wrapped key's integrity check value (0xA6A6A6A6A6A6A6A6).
///
/// ## Why `aes_kw` in tests but not production code?
///
/// Production code implements AES Key *Unwrap* entirely in the Metal
/// compute shader (software AES with S-box lookup tables). Test JWE
/// generation needs Key *Wrap* to create valid tokens. The `aes_kw` crate
/// provides both directions and is listed under `[dev-dependencies]` so it
/// adds no bytes to the release binary.
pub(crate) fn make_test_jwe(variant: AesKwVariant, secret: &[u8]) -> String {
    use aes_kw::{KekAes128, KekAes192, KekAes256};
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;

    let key_bytes = variant.key_bytes();

    // Derive an AES key of the appropriate length from the secret.
    // This replicates the key derivation logic used by the Metal shader:
    //   ≤ key_bytes → zero-pad to key_bytes
    //   > key_bytes → SHA-256 hash, truncate to key_bytes
    let mut aes_key = [0u8; 32]; // max size for AES-256
    if secret.len() <= key_bytes {
        aes_key[..secret.len()].copy_from_slice(secret);
    } else {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(secret);
        aes_key[..key_bytes].copy_from_slice(&hash[..key_bytes]);
    }

    // Generate a deterministic 128-bit (16-byte) CEK for reproducibility.
    // SHA-256(b"cek-seed-" || secret) → take first 16 bytes.
    let mut cek = [0u8; 16];
    {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest([b"cek-seed-" as &[u8], secret].concat());
        cek.copy_from_slice(&hash[..16]);
    }

    // AES Key Wrap the CEK → 24 bytes (n=2 blocks, output is (n+1)*8).
    // Dispatch to the correct KEK type based on variant.
    let mut wrapped = [0u8; 24];
    match variant {
        AesKwVariant::A128kw => {
            let kek = KekAes128::from(<[u8; 16]>::try_from(&aes_key[..16]).unwrap());
            kek.wrap(&cek, &mut wrapped)
                .expect("AES-128 Key Wrap failed");
        }
        AesKwVariant::A192kw => {
            let kek = KekAes192::from(<[u8; 24]>::try_from(&aes_key[..24]).unwrap());
            kek.wrap(&cek, &mut wrapped)
                .expect("AES-192 Key Wrap failed");
        }
        AesKwVariant::A256kw => {
            let kek = KekAes256::from(<[u8; 32]>::try_from(&aes_key[..32]).unwrap());
            kek.wrap(&cek, &mut wrapped)
                .expect("AES-256 Key Wrap failed");
        }
    }

    // Build the JWE header with the correct `alg` for this variant.
    let header = format!(r#"{{"alg":"{}","enc":"A128GCM"}}"#, variant.label());
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let encrypted_key_b64 = URL_SAFE_NO_PAD.encode(&wrapped);

    // IV, ciphertext, and tag are irrelevant for the Key Wrap attack.
    let iv_b64 = URL_SAFE_NO_PAD.encode(&[0x42u8; 12]); // 12 bytes for A128GCM IV
    let ct_b64 = URL_SAFE_NO_PAD.encode(&[0x00u8; 32]); // dummy ciphertext
    let tag_b64 = URL_SAFE_NO_PAD.encode(&[0x00u8; 16]); // dummy tag

    // Assemble the 5-part JWE compact token.
    format!("{header_b64}.{encrypted_key_b64}.{iv_b64}.{ct_b64}.{tag_b64}")
}

/// Write test JWE token files to the project root for manual benchmarking.
///
/// Run with: `cargo test generate_jwe_token_files -- --ignored --nocapture`
///
/// This generates token files for all three AES Key Wrap variants:
/// - `jwe_a{128,192,256}kw_found` — JWE wrapped with "password" (present in breach.txt)
/// - `jwe_a{128,192,256}kw_notfound` — JWE wrapped with "fsdgdfsgdsfsgdf" (absent)
#[test]
#[ignore]
fn generate_jwe_token_files() {
    for (variant, tag) in [
        (AesKwVariant::A128kw, "a128kw"),
        (AesKwVariant::A192kw, "a192kw"),
        (AesKwVariant::A256kw, "a256kw"),
    ] {
        let found_token = make_test_jwe(variant, b"password");
        let notfound_token = make_test_jwe(variant, b"fsdgdfsgdsfsgdf");

        std::fs::write(format!("jwe_{tag}_found"), &found_token)
            .unwrap_or_else(|_| panic!("failed to write jwe_{tag}_found"));
        std::fs::write(format!("jwe_{tag}_notfound"), &notfound_token)
            .unwrap_or_else(|_| panic!("failed to write jwe_{tag}_notfound"));

        eprintln!("jwe_{tag}_found:    {found_token}");
        eprintln!("jwe_{tag}_notfound: {notfound_token}");
    }
    eprintln!("Files written to project root.");
}

/// Same as `collect_all_words_from_mmap_reader` but for the parallel reader.
/// Having both variants avoids test code needing to know which reader type
/// is being tested — each test picks the appropriate helper.
///
/// On Linux (zero-copy path), offsets are absolute mmap positions, so we
/// reconstruct candidates from the mmap via `word_from_source`. On macOS,
/// the batch's local word_bytes buffer is populated as usual.
pub(crate) fn collect_all_words_from_parallel_reader(
    reader: &mut ParallelMmapWordlistBatchReader,
) -> Vec<Vec<u8>> {
    let mmap = reader.shared_mmap();
    let mut out = Vec::new();
    while let Some(batch) = reader
        .next_batch_reusing(None)
        .expect("parallel mmap batch")
    {
        for i in 0..batch.candidate_count() {
            #[cfg(target_os = "linux")]
            let w = batch.word_from_source(i, mmap.as_ref()).expect("candidate");
            #[cfg(not(target_os = "linux"))]
            let w = batch.word(i).expect("candidate");
            out.push(w.to_vec());
        }
    }
    // Suppress unused variable warning on macOS.
    let _ = &mmap;
    out
}
