//! Auto-detection logic and dispatch for `autocrack`.
//!
//! The JWT header is a base64url-encoded JSON object that always contains an
//! `alg` field specifying the signing algorithm. We decode just enough of the
//! header to read that field, then hand off to the appropriate cracker module.
//!
//! This avoids duplicating any GPU or pipeline code — we simply construct the
//! algorithm-specific args struct and call the existing `run()` function.
//!
//! If the algorithm is not one we support (e.g., RS256, ES384), we print a
//! clear error message listing what IS supported and exit with an error code.

use anyhow::{bail, Context};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::Deserialize;

use super::args::AutocrackArgs;
use crate::commands::hs256wordlist::{self, Hs256WordlistArgs};
use crate::commands::hs384wordlist::{self, Hs384WordlistArgs};
use crate::commands::hs512wordlist::{self, Hs512WordlistArgs};

/// Minimal JWT header — we only need the `alg` field to decide which cracker
/// to use. The `#[serde(deny_unknown_fields)]` is intentionally NOT set so
/// we tolerate extra fields like `typ`, `kid`, etc.
#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

/// Read the JWT header, detect the algorithm, and dispatch to the right cracker.
///
/// Returns `Ok(true)` if the secret was found, `Ok(false)` if not found, or
/// `Err` if the JWT is malformed or the algorithm is unsupported.
pub fn run(args: AutocrackArgs) -> anyhow::Result<bool> {
    // Step 1: Split the JWT into its three dot-separated segments.
    //
    // A JWT always has the form: <header>.<payload>.<signature>
    // We only need the first segment (the header) to detect the algorithm.
    let header_b64 = args
        .jwt
        .split('.')
        .next()
        .context("malformed JWT: expected at least one dot-separated segment")?;

    // Step 2: Base64url-decode the header segment.
    //
    // JWTs use base64url encoding (RFC 4648 section 5) WITHOUT padding. The
    // `URL_SAFE_NO_PAD` engine handles this — it uses `-` and `_` instead of
    // `+` and `/`, and doesn't require trailing `=` characters.
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .context("invalid base64url in JWT header")?;

    // Step 3: Parse the header JSON to extract the `alg` field.
    let header: JwtHeader =
        serde_json::from_slice(&header_bytes).context("invalid JSON in JWT header")?;

    // Step 4: Log the detected algorithm so the user knows what was chosen.
    eprintln!("AUTODETECT: JWT algorithm is {}", header.alg);

    // Step 5: Dispatch to the appropriate cracker by constructing its args
    // struct from our shared fields.
    //
    // Each algorithm-specific args struct is identical in shape — they just
    // differ in the doc comments on the `jwt` field. We copy all the tuning
    // knobs through so `autocrack` supports the same flags.
    match header.alg.as_str() {
        "HS256" => {
            let specific_args = Hs256WordlistArgs {
                jwt: args.jwt,
                wordlist: args.wordlist,
                threads_per_group: args.threads_per_group,
                parser_threads: args.parser_threads,
                pipeline_depth: args.pipeline_depth,
                packer_threads: args.packer_threads,
                autotune: args.autotune,
            };
            hs256wordlist::run(specific_args)
        }
        "HS384" => {
            let specific_args = Hs384WordlistArgs {
                jwt: args.jwt,
                wordlist: args.wordlist,
                threads_per_group: args.threads_per_group,
                parser_threads: args.parser_threads,
                pipeline_depth: args.pipeline_depth,
                packer_threads: args.packer_threads,
                autotune: args.autotune,
            };
            hs384wordlist::run(specific_args)
        }
        "HS512" => {
            let specific_args = Hs512WordlistArgs {
                jwt: args.jwt,
                wordlist: args.wordlist,
                threads_per_group: args.threads_per_group,
                parser_threads: args.parser_threads,
                pipeline_depth: args.pipeline_depth,
                packer_threads: args.packer_threads,
                autotune: args.autotune,
            };
            hs512wordlist::run(specific_args)
        }
        unsupported => {
            bail!(
                "unsupported JWT algorithm: {unsupported} (expected HS256, HS384, or HS512)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::args::AutocrackArgs;
    use crate::common::test_support::{make_test_jwt, write_temp_wordlist};
    use std::path::PathBuf;

    /// Helper to construct `AutocrackArgs` with minimal config for testing.
    fn make_args(jwt: &str, wordlist: PathBuf) -> AutocrackArgs {
        AutocrackArgs {
            jwt: jwt.to_string(),
            wordlist,
            threads_per_group: None,
            parser_threads: Some(1),
            pipeline_depth: Some(2),
            packer_threads: Some(1),
            autotune: false,
        }
    }

    #[test]
    fn autocrack_routes_hs256() {
        let jwt = make_test_jwt("HS256", r#"{"sub":"auto"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let result = super::run(make_args(&jwt, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via HS256");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn autocrack_routes_hs384() {
        let jwt = make_test_jwt("HS384", r#"{"sub":"auto"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let result = super::run(make_args(&jwt, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via HS384");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn autocrack_routes_hs512() {
        let jwt = make_test_jwt("HS512", r#"{"sub":"auto"}"#, b"password");
        let path = write_temp_wordlist(b"wrong1\npassword\nwrong2\n");
        let result = super::run(make_args(&jwt, path.clone())).unwrap();
        assert!(result, "autocrack should find 'password' via HS512");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn autocrack_rejects_unsupported_alg() {
        // Craft a JWT with alg=RS256 (an asymmetric algorithm we don't support).
        // We can't use make_test_jwt because it only supports HS* algorithms,
        // so we manually construct the header.
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let header = r#"{"alg":"RS256","typ":"JWT"}"#;
        let payload = r#"{"sub":"test"}"#;
        let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
        let payload_b64 = URL_SAFE_NO_PAD.encode(payload.as_bytes());
        // Signature doesn't matter — we'll fail before signature validation.
        let fake_sig = URL_SAFE_NO_PAD.encode(b"fakesignaturebytes");
        let jwt = format!("{header_b64}.{payload_b64}.{fake_sig}");

        let path = write_temp_wordlist(b"password\n");
        let result = super::run(make_args(&jwt, path.clone()));
        let _ = std::fs::remove_file(path);

        let err = result.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("unsupported JWT algorithm: RS256"),
            "expected unsupported algorithm error, got: {msg}"
        );
    }
}
