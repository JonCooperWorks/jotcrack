//! JWT parsing and validation for the HS384 subcommand.
//!
//! # Learning note: SHA-384 specifics
//!
//! SHA-384 is defined in FIPS 180-4 as a **truncated variant of SHA-512**.
//! It uses the exact same compression function and 80-round schedule as
//! SHA-512, but with two differences:
//!
//!   1. **Different initial values (IV).**  SHA-512 and SHA-384 start from
//!      different 8-word constants so that their outputs are domain-separated
//!      -- you cannot recover one from the other.
//!
//!   2. **Truncated output.**  SHA-512 produces a 64-byte (512-bit) digest
//!      from all 8 state words.  SHA-384 takes only the first 6 state words,
//!      yielding a 48-byte (384-bit) digest.
//!
//! For HMAC-SHA384 (HS384 in JWT parlance), the signature in the JWT is
//! therefore 48 bytes after base64url decoding, compared to 32 bytes for
//! HS256 and 64 bytes for HS512.

use anyhow::{Context, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;

/// Minimal deserialization target for the JWT header.
/// We only need the `alg` field to verify the token uses HS384.
#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

/// Parse and validate an HS384 JWT, returning:
/// - the signing input bytes (`base64url(header) + "." + base64url(payload)`)
/// - the decoded 48-byte target signature to compare against GPU results
///
/// The 48-byte signature length is the key HS384-specific detail here.
/// SHA-384 output = first 6 of 8 SHA-512 state words = 6 * 8 = 48 bytes.
pub(super) fn parse_hs384_jwt(jwt: &str) -> anyhow::Result<(Vec<u8>, [u8; 48])> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        bail!("malformed JWT: expected 3 dot-separated segments");
    }

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .context("invalid base64url JWT header")?;
    let header: JwtHeader =
        serde_json::from_slice(&header_bytes).context("invalid JWT header JSON")?;
    if header.alg != "HS384" {
        bail!("unsupported JWT alg: expected HS384, got {}", header.alg);
    }

    let signature_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .context("invalid base64url JWT signature")?;
    if signature_bytes.len() != 48 {
        bail!(
            "invalid HS384 signature length: expected 48 bytes, got {}",
            signature_bytes.len()
        );
    }
    let mut target_signature = [0u8; 48];
    target_signature.copy_from_slice(&signature_bytes);

    let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();

    Ok((signing_input, target_signature))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hs384_jwt_success() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS384","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#);
        let sig = URL_SAFE_NO_PAD.encode([7u8; 48]);
        let jwt = format!("{header}.{payload}.{sig}");
        let parsed = parse_hs384_jwt(&jwt).unwrap();

        assert!(parsed.0.windows(1).count() > 0);
        assert_eq!(parsed.1.len(), 48);
        assert!(String::from_utf8(parsed.0).unwrap().contains('.'));
    }

    #[test]
    fn parse_hs384_jwt_rejects_wrong_segment_count() {
        let err = parse_hs384_jwt("abc.def").unwrap_err();
        assert!(format!("{err:#}").contains("3 dot-separated"));
    }

    #[test]
    fn parse_hs384_jwt_rejects_non_hs384() {
        let header_json = br#"{"alg":"HS256","typ":"JWT"}"#;
        let payload_json = br#"{"sub":"alice"}"#;
        let header = URL_SAFE_NO_PAD.encode(header_json);
        let payload = URL_SAFE_NO_PAD.encode(payload_json);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 48]);
        let jwt = format!("{header}.{payload}.{sig}");

        let err = parse_hs384_jwt(&jwt).unwrap_err();
        assert!(format!("{err:#}").contains("HS384"));
    }

    #[test]
    fn parse_hs384_jwt_rejects_wrong_signature_length() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS384"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 47]);
        let err = parse_hs384_jwt(&format!("{header}.{payload}.{sig}")).unwrap_err();
        assert!(format!("{err:#}").contains("48 bytes"));
    }

    #[test]
    fn parse_hs384_jwt_rejects_bad_signature_base64() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS384"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let err = parse_hs384_jwt(&format!("{header}.{payload}.***")).unwrap_err();
        assert!(format!("{err:#}").contains("signature"));
    }
}
