//! JWT parsing and validation for the HS512 subcommand.
//!
//! # Learning note: SHA-512 signature size
//!
//! SHA-512 produces a 64-byte (512-bit) digest using all 8 of its internal
//! 64-bit state words.  Compare this to SHA-384, which truncates to 48 bytes
//! (6 words), and SHA-256, which produces 32 bytes (8 x 32-bit words).
//!
//! The HMAC-SHA512 (HS512) signature in a JWT is therefore 64 bytes after
//! base64url decoding.  On the GPU side, the host converts these 64 bytes
//! into 8 big-endian u64 words so the kernel can compare them directly
//! against the final SHA-512 state without per-thread byte shuffling.

use anyhow::{Context, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;

/// Minimal deserialization target for the JWT header.
/// We only need the `alg` field to verify the token uses HS512.
#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

/// Parse and validate an HS512 JWT, returning:
/// - the signing input bytes (`base64url(header) + "." + base64url(payload)`)
/// - the decoded 64-byte target signature to compare against GPU results
///
/// The 64-byte length reflects the full SHA-512 output: 8 state words * 8 bytes each.
pub(super) fn parse_hs512_jwt(jwt: &str) -> anyhow::Result<(Vec<u8>, [u8; 64])> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        bail!("malformed JWT: expected 3 dot-separated segments");
    }

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .context("invalid base64url JWT header")?;
    let header: JwtHeader =
        serde_json::from_slice(&header_bytes).context("invalid JWT header JSON")?;
    if header.alg != "HS512" {
        bail!("unsupported JWT alg: expected HS512, got {}", header.alg);
    }

    let signature_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .context("invalid base64url JWT signature")?;
    if signature_bytes.len() != 64 {
        bail!(
            "invalid HS512 signature length: expected 64 bytes, got {}",
            signature_bytes.len()
        );
    }
    let mut target_signature = [0u8; 64];
    target_signature.copy_from_slice(&signature_bytes);

    let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();

    Ok((signing_input, target_signature))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hs512_jwt_success() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS512","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#);
        let sig = URL_SAFE_NO_PAD.encode([7u8; 64]);
        let jwt = format!("{header}.{payload}.{sig}");
        let parsed = parse_hs512_jwt(&jwt).unwrap();

        assert!(parsed.0.windows(1).count() > 0);
        assert_eq!(parsed.1.len(), 64);
        assert!(String::from_utf8(parsed.0).unwrap().contains('.'));
    }

    #[test]
    fn parse_hs512_jwt_rejects_wrong_segment_count() {
        let err = parse_hs512_jwt("abc.def").unwrap_err();
        assert!(format!("{err:#}").contains("3 dot-separated"));
    }

    #[test]
    fn parse_hs512_jwt_rejects_non_hs512() {
        let header_json = br#"{"alg":"HS256","typ":"JWT"}"#;
        let payload_json = br#"{"sub":"alice"}"#;
        let header = URL_SAFE_NO_PAD.encode(header_json);
        let payload = URL_SAFE_NO_PAD.encode(payload_json);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 64]);
        let jwt = format!("{header}.{payload}.{sig}");

        let err = parse_hs512_jwt(&jwt).unwrap_err();
        assert!(format!("{err:#}").contains("HS512"));
    }

    #[test]
    fn parse_hs512_jwt_rejects_wrong_signature_length() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS512"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 63]);
        let err = parse_hs512_jwt(&format!("{header}.{payload}.{sig}")).unwrap_err();
        assert!(format!("{err:#}").contains("64 bytes"));
    }

    #[test]
    fn parse_hs512_jwt_rejects_bad_signature_base64() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS512"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let err = parse_hs512_jwt(&format!("{header}.{payload}.***")).unwrap_err();
        assert!(format!("{err:#}").contains("signature"));
    }
}
