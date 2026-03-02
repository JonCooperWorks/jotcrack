//! JWT (JSON Web Token) parsing and HS256 signature extraction.
//!
//! # What is a JWT?
//!
//! A JWT is a compact, URL-safe token format used for authentication. It has
//! three parts separated by dots: `header.payload.signature`.
//!
//! Example: `eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbGljZSJ9.XXXX`
//!
//! - **Header**: JSON like `{"alg":"HS256"}`, base64url-encoded.
//! - **Payload**: JSON claims like `{"sub":"alice"}`, base64url-encoded.
//! - **Signature**: HMAC-SHA256 of `"header.payload"` using a secret key.
//!
//! # What is HS256?
//!
//! HS256 = HMAC using SHA-256. It is a symmetric signing algorithm: the same
//! secret key is used to both create and verify the signature.
//!
//!   signature = HMAC-SHA256(secret_key, "base64url(header).base64url(payload)")
//!
//! To "crack" this, we try millions of candidate secret keys from a wordlist,
//! compute HMAC-SHA256 for each, and check if the output matches the JWT's
//! signature. If it matches, we found the secret key.
//!
//! # What is base64url?
//!
//! Regular base64 uses `+` and `/` which are not safe in URLs. Base64url
//! replaces them with `-` and `_`, and omits `=` padding. JWTs always use
//! base64url encoding (RFC 4648 section 5), which is why we use
//! `URL_SAFE_NO_PAD` from the `base64` crate.
//!
//! # Why return `(Vec<u8>, [u8; 32])`?
//!
//! The function returns two things the GPU cracker needs:
//! 1. `signing_input` (`Vec<u8>`): the raw bytes of `"header.payload"` that
//!    get fed into HMAC-SHA256 for every candidate key.
//! 2. `target_signature` (`[u8; 32]`): the 32-byte expected output. SHA-256
//!    always produces exactly 32 bytes (256 bits), hence the fixed-size array.

use anyhow::{Context, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;

/// Minimal JWT header -- we only need the `alg` field to verify this is HS256.
///
/// `#[derive(Deserialize)]` uses the `serde` crate to automatically generate
/// JSON parsing code at compile time. Rust does not have runtime reflection
/// like Python or JavaScript, so `serde` uses proc macros to generate efficient
/// deserialization code that is zero-cost at runtime.
#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

/// Parse and validate an HS256 JWT, returning:
/// - the signing input bytes (`base64url(header) + "." + base64url(payload)`)
/// - the decoded 32-byte target signature to compare against GPU results
///
/// This is a "fail-fast" function: it validates everything up front (correct
/// segment count, valid base64url, correct algorithm, correct signature length)
/// so the rest of the pipeline can assume a valid JWT without re-checking.
/// This pattern -- validate at the boundary, trust internally -- is common in
/// Rust and reduces error handling noise in hot paths.
pub(super) fn parse_hs256_jwt(jwt: &str) -> anyhow::Result<(Vec<u8>, [u8; 32])> {
    // Split on '.' to get the three JWT segments.
    // Using `collect::<Vec<&str>>()` is idiomatic for when you need random
    // access to the parts (here we need parts[0], parts[1], parts[2]).
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        bail!("malformed JWT: expected 3 dot-separated segments");
    }

    // Decode and validate the header to ensure this is an HS256 JWT.
    // We reject other algorithms (HS384, RS256, etc.) because our GPU kernel
    // only implements HMAC-SHA256.
    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .context("invalid base64url JWT header")?;
    let header: JwtHeader =
        serde_json::from_slice(&header_bytes).context("invalid JWT header JSON")?;
    if header.alg != "HS256" {
        bail!("unsupported JWT alg: expected HS256, got {}", header.alg);
    }

    // Decode the signature from base64url. For HS256 (HMAC-SHA256), the
    // signature is always exactly 32 bytes (256 bits / 8 = 32 bytes).
    // Any other length means this JWT is malformed or uses a different algorithm.
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .context("invalid base64url JWT signature")?;
    if signature_bytes.len() != 32 {
        bail!(
            "invalid HS256 signature length: expected 32 bytes, got {}",
            signature_bytes.len()
        );
    }
    // Copy into a fixed-size array `[u8; 32]`. This gives us compile-time
    // guarantees about the length, so downstream code (like the GPU params
    // struct) never needs to check the length again. Fixed-size arrays are
    // also `Copy`, which makes them cheap to pass around by value.
    let mut target_signature = [0u8; 32];
    target_signature.copy_from_slice(&signature_bytes);

    // The signing input is the raw base64url-encoded header and payload
    // separated by '.'. Note: we do NOT decode them -- HMAC signs the
    // *encoded* text, not the decoded JSON. This is part of the JWT spec.
    let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();

    Ok((signing_input, target_signature))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hs256_jwt_success() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#);
        let sig = URL_SAFE_NO_PAD.encode([7u8; 32]);
        let jwt = format!("{header}.{payload}.{sig}");
        let parsed = parse_hs256_jwt(&jwt).unwrap();

        assert!(parsed.0.windows(1).count() > 0);
        assert_eq!(parsed.1.len(), 32);
        assert!(String::from_utf8(parsed.0).unwrap().contains('.'));
    }

    #[test]
    fn parse_hs256_jwt_rejects_wrong_segment_count() {
        let err = parse_hs256_jwt("abc.def").unwrap_err();
        assert!(format!("{err:#}").contains("3 dot-separated"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_non_hs256() {
        let header_json = br#"{"alg":"HS384","typ":"JWT"}"#;
        let payload_json = br#"{"sub":"alice"}"#;
        let header = URL_SAFE_NO_PAD.encode(header_json);
        let payload = URL_SAFE_NO_PAD.encode(payload_json);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 32]);
        let jwt = format!("{header}.{payload}.{sig}");

        let err = parse_hs256_jwt(&jwt).unwrap_err();
        assert!(format!("{err:#}").contains("HS256"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_wrong_signature_length() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 31]);
        let err = parse_hs256_jwt(&format!("{header}.{payload}.{sig}")).unwrap_err();
        assert!(format!("{err:#}").contains("32 bytes"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_bad_signature_base64() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let err = parse_hs256_jwt(&format!("{header}.{payload}.***")).unwrap_err();
        assert!(format!("{err:#}").contains("signature"));
    }
}
