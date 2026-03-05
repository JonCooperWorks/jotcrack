use anyhow::{Context, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;

use super::gpu::{AesKwVariant, CrackVariant, HmacVariant};

#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

/// JWE (JSON Web Encryption) header.
///
/// JWE headers carry both an `alg` (key management algorithm) and an `enc`
/// (content encryption algorithm). For AES Key Wrap cracking we only need
/// `alg` to confirm the key wrapping method (A128KW/A192KW/A256KW), but
/// `enc` is parsed for validation and future use (it determines the CEK
/// length and thus the wrapped key size).
#[derive(Debug, Deserialize)]
struct JweHeader {
    alg: String,
    #[allow(dead_code)]
    enc: String,
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

/// Parse and validate a JWE compact token with an AES Key Wrap algorithm.
///
/// Supports A128KW, A192KW, and A256KW — all three ECB-based AES Key Wrap
/// variants defined in RFC 7518 §4.4. The `variant` parameter determines
/// which `alg` header value is expected.
///
/// JWE compact serialisation has five dot-separated base64url segments:
///   `<header>.<encrypted_key>.<iv>.<ciphertext>.<tag>`
///
/// For Key Wrap cracking, only the `encrypted_key` matters — it contains
/// the CEK wrapped using AES Key Wrap (RFC 3394). The IV, ciphertext,
/// and tag are ignored because the attack's oracle is the RFC 3394
/// integrity check value (0xA6A6A6A6A6A6A6A6), not the decrypted content.
///
/// The encrypted_key length must be `(n+1) * 8` bytes where `n >= 2`
/// (number of 64-bit CEK blocks). Common sizes:
/// - 24 bytes → n=2 (A128GCM: 128-bit CEK)
/// - 40 bytes → n=4 (A128CBC-HS256: 256-bit CEK)
pub(crate) fn parse_jwe_aes_kw(variant: AesKwVariant, jwe: &str) -> anyhow::Result<Vec<u8>> {
    let parts: Vec<&str> = jwe.split('.').collect();
    if parts.len() != 5 {
        bail!(
            "malformed JWE: expected 5 dot-separated segments, got {}",
            parts.len()
        );
    }

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .context("invalid base64url JWE header")?;
    let header: JweHeader =
        serde_json::from_slice(&header_bytes).context("invalid JWE header JSON")?;

    let expected_alg = variant.label();
    if header.alg != expected_alg {
        bail!(
            "unsupported JWE alg: expected {}, got {}",
            expected_alg,
            header.alg
        );
    }

    let encrypted_key = URL_SAFE_NO_PAD
        .decode(parts[1])
        .context("invalid base64url JWE encrypted_key")?;

    // AES Key Wrap output is (n+1) * 8 bytes, where n is the number of
    // 64-bit blocks in the CEK. Minimum n=2 (128-bit CEK), so minimum
    // encrypted_key length is 24 bytes.
    if encrypted_key.len() < 24 || encrypted_key.len() % 8 != 0 {
        bail!(
            "invalid {} encrypted_key length: expected multiple of 8 bytes (>= 24), got {}",
            expected_alg,
            encrypted_key.len()
        );
    }

    Ok(encrypted_key)
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

/// Auto-detect whether a token is a JWT or JWE and return the appropriate
/// `CrackVariant`.
///
/// The detection is based on the number of dot-separated segments:
/// - **3 parts** → JWT (`header.payload.signature`), delegates to `detect_variant`
/// - **5 parts** → JWE (`header.encrypted_key.iv.ciphertext.tag`), parses header
///   to determine the key management algorithm
///
/// This function only inspects the header — it does not decode or validate
/// the remaining segments. Full validation happens later in `parse_jwt` or
/// `parse_jwe_a128kw`.
pub(crate) fn detect_token_variant(token: &str) -> anyhow::Result<CrackVariant> {
    let part_count = token.split('.').count();
    match part_count {
        3 => {
            let hmac_variant = detect_variant(token)?;
            Ok(CrackVariant::Hmac(hmac_variant))
        }
        5 => {
            // JWE compact serialisation — parse header to identify the alg.
            let header_b64 = token
                .split('.')
                .next()
                .context("malformed JWE: empty token")?;
            let header_bytes = URL_SAFE_NO_PAD
                .decode(header_b64)
                .context("invalid base64url in JWE header")?;
            let header: JweHeader =
                serde_json::from_slice(&header_bytes).context("invalid JSON in JWE header")?;

            match header.alg.as_str() {
                "A128KW" => Ok(CrackVariant::JweAesKw(AesKwVariant::A128kw)),
                "A192KW" => Ok(CrackVariant::JweAesKw(AesKwVariant::A192kw)),
                "A256KW" => Ok(CrackVariant::JweAesKw(AesKwVariant::A256kw)),
                other => bail!(
                    "unsupported JWE algorithm: {} (expected A128KW, A192KW, or A256KW)",
                    other
                ),
            }
        }
        _ => bail!(
            "malformed token: expected 3 (JWT) or 5 (JWE) dot-separated segments, got {}",
            part_count
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

    // -----------------------------------------------------------------------
    // JWE parsing tests
    // -----------------------------------------------------------------------

    /// A well-formed 5-part A128KW JWE token is detected as `CrackVariant::JweAesKw(A128kw)`.
    #[test]
    fn detect_token_variant_jwe_a128kw() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A128KW","enc":"A128GCM"}"#);
        let ek = URL_SAFE_NO_PAD.encode(&[0u8; 24]); // 24-byte encrypted key
        let jwe = format!("{header}.{ek}.iv.ct.tag");
        assert_eq!(
            detect_token_variant(&jwe).unwrap(),
            CrackVariant::JweAesKw(AesKwVariant::A128kw)
        );
    }

    /// A well-formed 5-part A192KW JWE token is detected as `CrackVariant::JweAesKw(A192kw)`.
    #[test]
    fn detect_token_variant_jwe_a192kw() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A192KW","enc":"A128GCM"}"#);
        let ek = URL_SAFE_NO_PAD.encode(&[0u8; 24]);
        let jwe = format!("{header}.{ek}.iv.ct.tag");
        assert_eq!(
            detect_token_variant(&jwe).unwrap(),
            CrackVariant::JweAesKw(AesKwVariant::A192kw)
        );
    }

    /// A well-formed 5-part A256KW JWE token is detected as `CrackVariant::JweAesKw(A256kw)`.
    #[test]
    fn detect_token_variant_jwe_a256kw() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A256KW","enc":"A128GCM"}"#);
        let ek = URL_SAFE_NO_PAD.encode(&[0u8; 24]);
        let jwe = format!("{header}.{ek}.iv.ct.tag");
        assert_eq!(
            detect_token_variant(&jwe).unwrap(),
            CrackVariant::JweAesKw(AesKwVariant::A256kw)
        );
    }

    /// A 3-part HS256 JWT is detected as `CrackVariant::Hmac(Hs256)` by
    /// `detect_token_variant`.
    #[test]
    fn detect_token_variant_jwt_hs256() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let jwt = format!("{header}.payload.sig");
        assert_eq!(
            detect_token_variant(&jwt).unwrap(),
            CrackVariant::Hmac(HmacVariant::Hs256)
        );
    }

    /// Tokens with segment counts other than 3 or 5 are rejected.
    #[test]
    fn detect_token_variant_rejects_bad_segment_count() {
        let err = detect_token_variant("a.b.c.d").unwrap_err();
        assert!(format!("{err:#}").contains("expected 3 (JWT) or 5 (JWE)"));
    }

    /// A JWE with an unsupported alg is rejected by `detect_token_variant`.
    #[test]
    fn detect_token_variant_rejects_unsupported_jwe_alg() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A128GCMKW","enc":"A128GCM"}"#);
        let jwe = format!("{header}.ek.iv.ct.tag");
        let err = detect_token_variant(&jwe).unwrap_err();
        assert!(format!("{err:#}").contains("unsupported JWE algorithm: A128GCMKW"));
    }

    /// `parse_jwe_aes_kw` extracts the encrypted_key from a valid A128KW JWE.
    #[test]
    fn parse_jwe_aes_kw_valid_a128kw() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A128KW","enc":"A128GCM"}"#);
        let ek_bytes = [42u8; 24];
        let ek = URL_SAFE_NO_PAD.encode(&ek_bytes);
        let iv = URL_SAFE_NO_PAD.encode(&[0u8; 12]);
        let ct = URL_SAFE_NO_PAD.encode(&[0u8; 32]);
        let tag = URL_SAFE_NO_PAD.encode(&[0u8; 16]);
        let jwe = format!("{header}.{ek}.{iv}.{ct}.{tag}");
        let result = parse_jwe_aes_kw(AesKwVariant::A128kw, &jwe).unwrap();
        assert_eq!(result, ek_bytes);
    }

    /// `parse_jwe_aes_kw` extracts the encrypted_key from a valid A192KW JWE.
    #[test]
    fn parse_jwe_aes_kw_valid_a192kw() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A192KW","enc":"A128GCM"}"#);
        let ek_bytes = [42u8; 24];
        let ek = URL_SAFE_NO_PAD.encode(&ek_bytes);
        let iv = URL_SAFE_NO_PAD.encode(&[0u8; 12]);
        let ct = URL_SAFE_NO_PAD.encode(&[0u8; 32]);
        let tag = URL_SAFE_NO_PAD.encode(&[0u8; 16]);
        let jwe = format!("{header}.{ek}.{iv}.{ct}.{tag}");
        let result = parse_jwe_aes_kw(AesKwVariant::A192kw, &jwe).unwrap();
        assert_eq!(result, ek_bytes);
    }

    /// `parse_jwe_aes_kw` extracts the encrypted_key from a valid A256KW JWE.
    #[test]
    fn parse_jwe_aes_kw_valid_a256kw() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A256KW","enc":"A128GCM"}"#);
        let ek_bytes = [42u8; 24];
        let ek = URL_SAFE_NO_PAD.encode(&ek_bytes);
        let iv = URL_SAFE_NO_PAD.encode(&[0u8; 12]);
        let ct = URL_SAFE_NO_PAD.encode(&[0u8; 32]);
        let tag = URL_SAFE_NO_PAD.encode(&[0u8; 16]);
        let jwe = format!("{header}.{ek}.{iv}.{ct}.{tag}");
        let result = parse_jwe_aes_kw(AesKwVariant::A256kw, &jwe).unwrap();
        assert_eq!(result, ek_bytes);
    }

    /// `parse_jwe_aes_kw` rejects a JWE with a non-matching algorithm.
    #[test]
    fn parse_jwe_aes_kw_rejects_wrong_alg() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A256KW","enc":"A256GCM"}"#);
        let ek = URL_SAFE_NO_PAD.encode(&[0u8; 24]);
        let jwe = format!("{header}.{ek}.iv.ct.tag");
        let err = parse_jwe_aes_kw(AesKwVariant::A128kw, &jwe).unwrap_err();
        assert!(format!("{err:#}").contains("expected A128KW"));
    }

    /// `parse_jwe_aes_kw` rejects tokens with wrong segment count.
    #[test]
    fn parse_jwe_aes_kw_rejects_wrong_part_count() {
        let err = parse_jwe_aes_kw(AesKwVariant::A128kw, "a.b.c").unwrap_err();
        assert!(format!("{err:#}").contains("5 dot-separated"));
    }

    /// `parse_jwe_aes_kw` rejects encrypted_key shorter than 24 bytes.
    #[test]
    fn parse_jwe_aes_kw_rejects_short_encrypted_key() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"A128KW","enc":"A128GCM"}"#);
        let ek = URL_SAFE_NO_PAD.encode(&[0u8; 16]); // too short
        let jwe = format!("{header}.{ek}.iv.ct.tag");
        let err = parse_jwe_aes_kw(AesKwVariant::A128kw, &jwe).unwrap_err();
        assert!(format!("{err:#}").contains("multiple of 8 bytes"));
    }
}
