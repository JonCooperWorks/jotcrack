use anyhow::{Context, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;

use super::gpu::HmacVariant;

#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

/// Parse and validate a JWT against the given HMAC variant.
///
/// Returns the signing input bytes (`base64url(header).base64url(payload)`)
/// and the decoded target signature. The signature length is validated
/// against `variant.signature_len()`.
pub(crate) fn parse_jwt(
    variant: HmacVariant,
    jwt: &str,
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        bail!("malformed JWT: expected 3 dot-separated segments");
    }

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .context("invalid base64url JWT header")?;
    let header: JwtHeader =
        serde_json::from_slice(&header_bytes).context("invalid JWT header JSON")?;

    let expected_alg = variant.label();
    if header.alg != expected_alg {
        bail!(
            "unsupported JWT alg: expected {}, got {}",
            expected_alg,
            header.alg
        );
    }

    let signature_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .context("invalid base64url JWT signature")?;
    let expected_len = variant.signature_len();
    if signature_bytes.len() != expected_len {
        bail!(
            "invalid {} signature length: expected {} bytes, got {}",
            expected_alg,
            expected_len,
            signature_bytes.len()
        );
    }

    let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();
    Ok((signing_input, signature_bytes))
}

/// Detect the HMAC variant from the JWT header's `alg` field.
pub(crate) fn detect_variant(jwt: &str) -> anyhow::Result<HmacVariant> {
    let header_b64 = jwt
        .split('.')
        .next()
        .context("malformed JWT: expected at least one dot-separated segment")?;
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .context("invalid base64url in JWT header")?;
    let header: JwtHeader =
        serde_json::from_slice(&header_bytes).context("invalid JSON in JWT header")?;

    match header.alg.as_str() {
        "HS256" => Ok(HmacVariant::Hs256),
        "HS384" => Ok(HmacVariant::Hs384),
        "HS512" => Ok(HmacVariant::Hs512),
        other => bail!(
            "unsupported JWT algorithm: {} (expected HS256, HS384, or HS512)",
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parse_success(variant: HmacVariant) {
        let alg = variant.label();
        let sig_len = variant.signature_len();
        let header = URL_SAFE_NO_PAD.encode(
            format!(r#"{{"alg":"{}","typ":"JWT"}}"#, alg).as_bytes(),
        );
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#);
        let sig = URL_SAFE_NO_PAD.encode(vec![7u8; sig_len]);
        let jwt = format!("{header}.{payload}.{sig}");
        let (signing_input, target_sig) = parse_jwt(variant, &jwt).unwrap();
        assert!(!signing_input.is_empty());
        assert_eq!(target_sig.len(), sig_len);
        assert!(String::from_utf8(signing_input).unwrap().contains('.'));
    }

    #[test]
    fn parse_hs256_success() { test_parse_success(HmacVariant::Hs256); }
    #[test]
    fn parse_hs384_success() { test_parse_success(HmacVariant::Hs384); }
    #[test]
    fn parse_hs512_success() { test_parse_success(HmacVariant::Hs512); }

    fn test_rejects_wrong_segment_count(variant: HmacVariant) {
        let err = parse_jwt(variant, "abc.def").unwrap_err();
        assert!(format!("{err:#}").contains("3 dot-separated"));
    }

    #[test]
    fn hs256_rejects_wrong_segment_count() { test_rejects_wrong_segment_count(HmacVariant::Hs256); }
    #[test]
    fn hs384_rejects_wrong_segment_count() { test_rejects_wrong_segment_count(HmacVariant::Hs384); }
    #[test]
    fn hs512_rejects_wrong_segment_count() { test_rejects_wrong_segment_count(HmacVariant::Hs512); }

    fn test_rejects_wrong_alg(variant: HmacVariant, wrong_alg: &str) {
        let header = URL_SAFE_NO_PAD.encode(
            format!(r#"{{"alg":"{}","typ":"JWT"}}"#, wrong_alg).as_bytes(),
        );
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#);
        let sig = URL_SAFE_NO_PAD.encode(vec![0u8; variant.signature_len()]);
        let jwt = format!("{header}.{payload}.{sig}");
        let err = parse_jwt(variant, &jwt).unwrap_err();
        assert!(format!("{err:#}").contains(variant.label()));
    }

    #[test]
    fn hs256_rejects_wrong_alg() { test_rejects_wrong_alg(HmacVariant::Hs256, "HS384"); }
    #[test]
    fn hs384_rejects_wrong_alg() { test_rejects_wrong_alg(HmacVariant::Hs384, "HS256"); }
    #[test]
    fn hs512_rejects_wrong_alg() { test_rejects_wrong_alg(HmacVariant::Hs512, "HS256"); }

    fn test_rejects_wrong_sig_length(variant: HmacVariant) {
        let alg = variant.label();
        let header = URL_SAFE_NO_PAD.encode(
            format!(r#"{{"alg":"{}"}}"#, alg).as_bytes(),
        );
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let sig = URL_SAFE_NO_PAD.encode(vec![0u8; variant.signature_len() - 1]);
        let err = parse_jwt(variant, &format!("{header}.{payload}.{sig}")).unwrap_err();
        assert!(format!("{err:#}").contains(&format!("{} bytes", variant.signature_len())));
    }

    #[test]
    fn hs256_rejects_wrong_sig_length() { test_rejects_wrong_sig_length(HmacVariant::Hs256); }
    #[test]
    fn hs384_rejects_wrong_sig_length() { test_rejects_wrong_sig_length(HmacVariant::Hs384); }
    #[test]
    fn hs512_rejects_wrong_sig_length() { test_rejects_wrong_sig_length(HmacVariant::Hs512); }

    fn test_rejects_bad_sig_base64(variant: HmacVariant) {
        let alg = variant.label();
        let header = URL_SAFE_NO_PAD.encode(
            format!(r#"{{"alg":"{}"}}"#, alg).as_bytes(),
        );
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let err = parse_jwt(variant, &format!("{header}.{payload}.***")).unwrap_err();
        assert!(format!("{err:#}").contains("signature"));
    }

    #[test]
    fn hs256_rejects_bad_sig_base64() { test_rejects_bad_sig_base64(HmacVariant::Hs256); }
    #[test]
    fn hs384_rejects_bad_sig_base64() { test_rejects_bad_sig_base64(HmacVariant::Hs384); }
    #[test]
    fn hs512_rejects_bad_sig_base64() { test_rejects_bad_sig_base64(HmacVariant::Hs512); }

    #[test]
    fn detect_variant_hs256() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let jwt = format!("{header}.payload.sig");
        assert_eq!(detect_variant(&jwt).unwrap(), HmacVariant::Hs256);
    }

    #[test]
    fn detect_variant_hs384() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS384"}"#);
        let jwt = format!("{header}.payload.sig");
        assert_eq!(detect_variant(&jwt).unwrap(), HmacVariant::Hs384);
    }

    #[test]
    fn detect_variant_hs512() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS512"}"#);
        let jwt = format!("{header}.payload.sig");
        assert_eq!(detect_variant(&jwt).unwrap(), HmacVariant::Hs512);
    }

    #[test]
    fn detect_variant_rejects_unsupported() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256"}"#);
        let jwt = format!("{header}.payload.sig");
        let err = detect_variant(&jwt).unwrap_err();
        assert!(format!("{err:#}").contains("unsupported JWT algorithm: RS256"));
    }
}
