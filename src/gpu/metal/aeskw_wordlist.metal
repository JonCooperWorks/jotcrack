// ===========================================================================
// Metal compute kernel for AES Key Wrap (RFC 3394) JWE cracking.
//
// COMPILE-TIME SPECIALISATION
//
// This shader is compiled three times with different AES_KEY_BYTES values:
//
//   AES_KEY_BYTES = 16  →  AES-128 (A128KW):  Nk=4,  Nr=10,  44 round keys
//   AES_KEY_BYTES = 24  →  AES-192 (A192KW):  Nk=6,  Nr=12,  52 round keys
//   AES_KEY_BYTES = 32  →  AES-256 (A256KW):  Nk=8,  Nr=14,  60 round keys
//
// The Rust host prepends `#define AES_KEY_BYTES N` before compilation,
// producing a fully specialised binary for each variant.  This approach
// lets the Metal compiler:
//   - Statically allocate exact-sized round key arrays (no wasted registers)
//   - Unroll loops with known trip counts
//   - Eliminate dead branches (SHA-256 truncation length, key expansion)
//
// DESIGN OVERVIEW
//
// Each GPU thread receives one candidate secret from a wordlist and attempts
// to unwrap a JWE encrypted key using AES Key Wrap (RFC 3394).  If the
// unwrap succeeds (integrity check value == 0xA6A6A6A6A6A6A6A6), the thread
// writes its index to an atomic result slot.
//
// WHY SOFTWARE AES ON THE GPU?
//
// Metal compute shaders cannot access Apple Silicon's hardware AES
// instructions (AESD/AESIMC).  This kernel implements AES entirely in
// software using lookup tables:
//
//   - Forward S-box (256 bytes, `constant`) for key expansion's SubWord
//   - Inverse S-box (256 bytes, `constant`) for decryption's InvSubBytes
//   - GF(2^8) arithmetic for InvMixColumns (xtime-based, no tables)
//
// The S-box tables live in Metal's `constant` address space, which is
// broadcast-cached across all threads in a SIMD group.  At 512 bytes total,
// they fit comfortably in the GPU's constant cache.
//
// Despite the per-operation cost being higher than CPU hardware AES, the
// GPU's massive parallelism (thousands of concurrent threads) outweighs
// the per-thread overhead.  Benchmarks on Apple M4 Max show ~121M/s GPU
// vs ~5.3M/s CPU for A128KW — 22× faster.
//
// PER-THREAD REGISTER BUDGET
//
//   AES-128: round_keys[44] = 176 bytes
//   AES-192: round_keys[52] = 208 bytes
//   AES-256: round_keys[60] = 240 bytes
//   A + R[0..7]:               72 bytes   (unwrap state: 8-byte A + up to 8×8 R blocks)
//   AES block:                 16 bytes   (scratch for decrypt)
//   w[16] (SHA-256):           64 bytes   (only in general kernel for long keys)
//   Total: ~264-376 bytes (variant-dependent)
//
// This is higher than HS256's ~128 bytes, so occupancy will be lower.  The
// short_keys kernel (candidates ≤ AES_KEY_BYTES) avoids SHA-256 entirely,
// reducing pressure by ~64 bytes.
//
// KEY DERIVATION
//
// AES Key Wrap requires exactly AES_KEY_BYTES raw bytes as the KEK:
//   - Candidates ≤ AES_KEY_BYTES: zero-padded
//   - Candidates > AES_KEY_BYTES: SHA-256 hashed, truncated to AES_KEY_BYTES
// Note: AES-256 uses the full 32-byte SHA-256 digest (no truncation).
//
// AES DECRYPTION OVERVIEW (FIPS 197 §5.3)
//
// AES uses Nr rounds (10/12/14 for 128/192/256-bit keys).  Each decryption
// round (except the last) applies:
//   1. InvShiftRows  — cyclic byte permutation within each row
//   2. InvSubBytes   — byte substitution via the inverse S-box
//   3. AddRoundKey   — XOR with the round key (BEFORE InvMixColumns!)
//   4. InvMixColumns — GF(2^8) matrix multiplication per column
// The last round omits InvMixColumns.
//
// RFC 3394 KEY UNWRAP
//
// The unwrap algorithm performs 6×n AES-ECB decryptions (n = number of
// 64-bit CEK blocks).  For n=2 (A128GCM): 12 decryptions per candidate.
// The algorithm is identical for all AES key sizes — only the underlying
// AES block cipher changes.
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Compile-time AES variant configuration
//
// AES_KEY_BYTES is defined by the Rust host before compilation:
//   16 → AES-128 (A128KW), 24 → AES-192 (A192KW), 32 → AES-256 (A256KW)
//
// All derived constants are computed at compile time:
//   AES_NK       = key words (4/6/8)
//   AES_NR       = round count (10/12/14)  — always Nk + 6
//   AES_RK_WORDS = total round key words   — always 4 × (Nr + 1)
// ---------------------------------------------------------------------------
#ifndef AES_KEY_BYTES
#define AES_KEY_BYTES 16
#endif

#define AES_NK        (AES_KEY_BYTES / 4)
#define AES_NR        (AES_NK + 6)
#define AES_RK_WORDS  (4 * (AES_NR + 1))

// ---------------------------------------------------------------------------
// Host-to-kernel parameter block
// ---------------------------------------------------------------------------

// Matches the Rust-side `#[repr(C)]` struct `AesKwBruteForceParams`.
// The AES key size is baked in at compile time (AES_KEY_BYTES), so it
// does not need to be in the params struct — the shader is already
// specialised for a specific variant.
struct AesKwBruteForceParams {
    uint32_t encrypted_key_len;   // Byte length of the encrypted_key in buffer(1).
    uint32_t n_blocks;            // Number of 64-bit blocks in the CEK (n ≥ 2).
    uint32_t candidate_count;     // Number of candidate secrets in this batch.
};

// ---------------------------------------------------------------------------
// AES S-box tables (FIPS 197)
//
// These tables are placed in `constant` address space so every thread in a
// SIMD group shares one cached copy.  At 256 bytes each, both fit in the
// GPU's constant cache with room to spare.
//
// The forward S-box is used only by key expansion (SubWord operation).
// The inverse S-box is used by every AES decryption round (InvSubBytes).
// ---------------------------------------------------------------------------

// Forward S-box (FIPS 197 §5.1.1): used by AES key expansion's SubWord.
constant uint8_t SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// Inverse S-box (FIPS 197 §5.3.2): used by AES decryption's InvSubBytes.
constant uint8_t INV_SBOX[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

// AES key expansion round constants (FIPS 197 §5.2).
// RCON[i] = x^(i+1) in GF(2^8), stored as the MSB of a 32-bit word
// (the other 3 bytes are always zero in AES key expansion).
// AES-256 uses at most 7 (i=0..6), but we keep all 10 for generality.
constant uint32_t RCON[10] = {
    0x01000000u, 0x02000000u, 0x04000000u, 0x08000000u, 0x10000000u,
    0x20000000u, 0x40000000u, 0x80000000u, 0x1b000000u, 0x36000000u
};

// ---------------------------------------------------------------------------
// SubWord helper — apply the forward S-box to each byte of a 32-bit word.
// Used by all three key expansion variants.
// ---------------------------------------------------------------------------
static inline uint32_t sub_word(uint32_t w) {
    return (uint32_t(SBOX[(w >> 24) & 0xffu]) << 24) |
           (uint32_t(SBOX[(w >> 16) & 0xffu]) << 16) |
           (uint32_t(SBOX[(w >> 8)  & 0xffu]) << 8)  |
            uint32_t(SBOX[ w        & 0xffu]);
}

// ---------------------------------------------------------------------------
// GF(2^8) arithmetic for InvMixColumns
//
// AES operates in GF(2^8) with the irreducible polynomial x^8 + x^4 + x^3 + x + 1
// (0x11b in binary).  InvMixColumns multiplies each column by the fixed matrix:
//
//   [0x0e  0x0b  0x0d  0x09]
//   [0x09  0x0e  0x0b  0x0d]
//   [0x0d  0x09  0x0e  0x0b]
//   [0x0b  0x0d  0x09  0x0e]
//
// We implement multiplication by these constants using repeated applications
// of `xtime` (multiply by 0x02 = left shift + conditional XOR with 0x1b).
// This avoids lookup tables at the cost of a few extra ALU ops per byte.
// ---------------------------------------------------------------------------

// GF(2^8) multiply by 2 ("xtime").
// Left-shifts and conditionally XORs with the reduction polynomial 0x1b
// if the high bit was set (polynomial reduction).
static inline uint8_t xtime(uint8_t x) {
    return (x << 1) ^ ((x & 0x80u) ? 0x1bu : 0x00u);
}

// Multiply by 0x09 = x^3 + 1 = xtime(xtime(xtime(x))) ^ x.
static inline uint8_t mul09(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ x;
}

// Multiply by 0x0b = x^3 + x + 1 = xtime(xtime(xtime(x))) ^ xtime(x) ^ x.
static inline uint8_t mul0b(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ xtime(x) ^ x;
}

// Multiply by 0x0d = x^3 + x^2 + 1 = xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ x.
static inline uint8_t mul0d(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ x;
}

// Multiply by 0x0e = x^3 + x^2 + x = xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ xtime(x).
static inline uint8_t mul0e(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ xtime(x);
}

// ---------------------------------------------------------------------------
// AES key expansion (FIPS 197 §5.2)
//
// Expands an AES_KEY_BYTES-byte key into AES_RK_WORDS round key words.
// The expansion algorithm differs by key size:
//
//   AES-128 (Nk=4, §5.2):
//     Load 4 words from key.  For i = 4..43:
//       if i % 4 == 0: W[i] = W[i-4] ^ SubWord(RotWord(W[i-1])) ^ RCON[i/4-1]
//       else:          W[i] = W[i-4] ^ W[i-1]
//
//   AES-192 (Nk=6, §5.2):
//     Load 6 words from key.  For i = 6..51:
//       if i % 6 == 0: W[i] = W[i-6] ^ SubWord(RotWord(W[i-1])) ^ RCON[i/6-1]
//       else:          W[i] = W[i-6] ^ W[i-1]
//
//   AES-256 (Nk=8, §5.2):
//     Load 8 words from key.  For i = 8..59:
//       if i % 8 == 0: W[i] = W[i-8] ^ SubWord(RotWord(W[i-1])) ^ RCON[i/8-1]
//       if i % 8 == 4: W[i] = W[i-8] ^ SubWord(W[i-1])  ← extra SubWord!
//       else:          W[i] = W[i-8] ^ W[i-1]
//
// The AES-256 schedule has an additional SubWord (without RotWord or RCON)
// at every i % 8 == 4 position.  This is unique to AES-256 and is the
// reason the three schedules cannot be trivially unified.
// ---------------------------------------------------------------------------
static inline void aes_key_expand(
    thread const uint8_t key[AES_KEY_BYTES],
    thread uint32_t rk[AES_RK_WORDS]
) {
    // Load the first AES_NK words directly from the key (big-endian).
    #pragma unroll
    for (uint i = 0; i < AES_NK; ++i) {
        rk[i] = (uint32_t(key[4*i]) << 24) |
                (uint32_t(key[4*i+1]) << 16) |
                (uint32_t(key[4*i+2]) << 8) |
                 uint32_t(key[4*i+3]);
    }

    // Expand remaining words.  The schedule depends on the key size.
    for (uint i = AES_NK; i < AES_RK_WORDS; ++i) {
        uint32_t temp = rk[i - 1];

#if AES_NK == 4
        // AES-128: SubWord + RotWord + RCON every 4th word.
        if ((i & 3u) == 0u) {
            uint32_t rotated = (temp << 8) | (temp >> 24);
            temp = sub_word(rotated) ^ RCON[i / 4u - 1u];
        }
        rk[i] = rk[i - 4u] ^ temp;

#elif AES_NK == 6
        // AES-192: SubWord + RotWord + RCON every 6th word.
        if (i % 6u == 0u) {
            uint32_t rotated = (temp << 8) | (temp >> 24);
            temp = sub_word(rotated) ^ RCON[i / 6u - 1u];
        }
        rk[i] = rk[i - 6u] ^ temp;

#elif AES_NK == 8
        // AES-256: SubWord + RotWord + RCON every 8th word,
        // plus an extra SubWord (no RotWord, no RCON) at i % 8 == 4.
        if ((i & 7u) == 0u) {
            uint32_t rotated = (temp << 8) | (temp >> 24);
            temp = sub_word(rotated) ^ RCON[i / 8u - 1u];
        } else if ((i & 7u) == 4u) {
            temp = sub_word(temp);
        }
        rk[i] = rk[i - 8u] ^ temp;
#endif
    }
}

// ---------------------------------------------------------------------------
// AES ECB decrypt (FIPS 197 §5.3, standard Inverse Cipher)
//
// Decrypts a single 16-byte block in-place using the expanded round keys.
// The state is represented as 4 × uint32_t columns in big-endian byte order.
//
// The round count is AES_NR (10/12/14), determined at compile time.
//
// Decryption order (standard Inverse Cipher, §5.3):
//   1. AddRoundKey(round AES_NR)
//   2. For round = AES_NR-1 down to 1:
//      a. InvShiftRows — cyclic right-shift of rows by 0,1,2,3 positions
//      b. InvSubBytes  — replace each byte via INV_SBOX
//      c. AddRoundKey  — XOR with round key (BEFORE InvMixColumns!)
//      d. InvMixColumns — GF(2^8) matrix multiply per column
//   3. InvShiftRows
//   4. InvSubBytes
//   5. AddRoundKey(round 0)
//
// IMPORTANT: The standard Inverse Cipher requires AddRoundKey BEFORE
// InvMixColumns for rounds (Nr-1)..1. The alternative "Equivalent Inverse
// Cipher" (§5.3.5) reverses this order but requires pre-transforming
// the round keys with InvMixColumns. Since we use standard key expansion,
// we must use the standard order: AddRoundKey then InvMixColumns.
// ---------------------------------------------------------------------------
static inline void aes_ecb_decrypt(
    thread const uint32_t rk[AES_RK_WORDS],
    thread uint8_t block[16]
) {
    // Load state as 4 columns (big-endian: byte[0] is MSB of s0).
    uint32_t s0 = (uint32_t(block[0])  << 24) | (uint32_t(block[1])  << 16) | (uint32_t(block[2])  << 8) | uint32_t(block[3]);
    uint32_t s1 = (uint32_t(block[4])  << 24) | (uint32_t(block[5])  << 16) | (uint32_t(block[6])  << 8) | uint32_t(block[7]);
    uint32_t s2 = (uint32_t(block[8])  << 24) | (uint32_t(block[9])  << 16) | (uint32_t(block[10]) << 8) | uint32_t(block[11]);
    uint32_t s3 = (uint32_t(block[12]) << 24) | (uint32_t(block[13]) << 16) | (uint32_t(block[14]) << 8) | uint32_t(block[15]);

    // AddRoundKey(round AES_NR) — XOR with the last round key.
    // Index = AES_RK_WORDS - 4 = 4 * AES_NR.
    s0 ^= rk[AES_RK_WORDS - 4]; s1 ^= rk[AES_RK_WORDS - 3];
    s2 ^= rk[AES_RK_WORDS - 2]; s3 ^= rk[AES_RK_WORDS - 1];

    // Rounds (AES_NR-1) down to 1: InvShiftRows + InvSubBytes + AddRoundKey + InvMixColumns.
    for (int round = AES_NR - 1; round >= 1; --round) {
        // Extract individual bytes from the state columns.
        // State byte layout (row, col):
        //   s0 = [0,0][1,0][2,0][3,0]  (column 0)
        //   s1 = [0,1][1,1][2,1][3,1]  (column 1)
        //   s2 = [0,2][1,2][2,2][3,2]  (column 2)
        //   s3 = [0,3][1,3][2,3][3,3]  (column 3)

        // InvShiftRows: row 0 unchanged, row 1 right-shift 1, row 2 right-shift 2, row 3 right-shift 3.
        // Combined with InvSubBytes (apply INV_SBOX to each byte after the shift).
        //
        // After InvShiftRows, byte at position (row, col) comes from (row, (col+shift) % 4):
        //   Row 0: col 0 from s0, col 1 from s1, col 2 from s2, col 3 from s3  (no shift)
        //   Row 1: col 0 from s3, col 1 from s0, col 2 from s1, col 3 from s2  (shift=1)
        //   Row 2: col 0 from s2, col 1 from s3, col 2 from s0, col 3 from s1  (shift=2)
        //   Row 3: col 0 from s1, col 1 from s2, col 2 from s3, col 3 from s0  (shift=3)

        // Column 0 after InvShiftRows + InvSubBytes:
        uint8_t b00 = INV_SBOX[(s0 >> 24) & 0xffu];  // row 0 from col 0
        uint8_t b10 = INV_SBOX[(s3 >> 16) & 0xffu];  // row 1 from col 3
        uint8_t b20 = INV_SBOX[(s2 >> 8)  & 0xffu];  // row 2 from col 2
        uint8_t b30 = INV_SBOX[ s1        & 0xffu];  // row 3 from col 1

        // Column 1 after InvShiftRows + InvSubBytes:
        uint8_t b01 = INV_SBOX[(s1 >> 24) & 0xffu];  // row 0 from col 1
        uint8_t b11 = INV_SBOX[(s0 >> 16) & 0xffu];  // row 1 from col 0
        uint8_t b21 = INV_SBOX[(s3 >> 8)  & 0xffu];  // row 2 from col 3
        uint8_t b31 = INV_SBOX[ s2        & 0xffu];  // row 3 from col 2

        // Column 2 after InvShiftRows + InvSubBytes:
        uint8_t b02 = INV_SBOX[(s2 >> 24) & 0xffu];  // row 0 from col 2
        uint8_t b12 = INV_SBOX[(s1 >> 16) & 0xffu];  // row 1 from col 1
        uint8_t b22 = INV_SBOX[(s0 >> 8)  & 0xffu];  // row 2 from col 0
        uint8_t b32 = INV_SBOX[ s3        & 0xffu];  // row 3 from col 3

        // Column 3 after InvShiftRows + InvSubBytes:
        uint8_t b03 = INV_SBOX[(s3 >> 24) & 0xffu];  // row 0 from col 3
        uint8_t b13 = INV_SBOX[(s2 >> 16) & 0xffu];  // row 1 from col 2
        uint8_t b23 = INV_SBOX[(s1 >> 8)  & 0xffu];  // row 2 from col 1
        uint8_t b33 = INV_SBOX[ s0        & 0xffu];  // row 3 from col 0

        // Reassemble columns and AddRoundKey (BEFORE InvMixColumns per §5.3).
        uint rki = uint(round) * 4u;
        s0 = ((uint32_t(b00) << 24) | (uint32_t(b10) << 16) | (uint32_t(b20) << 8) | uint32_t(b30)) ^ rk[rki];
        s1 = ((uint32_t(b01) << 24) | (uint32_t(b11) << 16) | (uint32_t(b21) << 8) | uint32_t(b31)) ^ rk[rki + 1u];
        s2 = ((uint32_t(b02) << 24) | (uint32_t(b12) << 16) | (uint32_t(b22) << 8) | uint32_t(b32)) ^ rk[rki + 2u];
        s3 = ((uint32_t(b03) << 24) | (uint32_t(b13) << 16) | (uint32_t(b23) << 8) | uint32_t(b33)) ^ rk[rki + 3u];

        // InvMixColumns: multiply each column by the inverse MixColumns matrix.
        // Operates on the AddRoundKey result (s0..s3), extracting bytes,
        // applying the GF(2^8) matrix, and reassembling into new columns.
        //
        // For each column [b0, b1, b2, b3]:
        //   r0 = 0e·b0 ^ 0b·b1 ^ 0d·b2 ^ 09·b3
        //   r1 = 09·b0 ^ 0e·b1 ^ 0b·b2 ^ 0d·b3
        //   r2 = 0d·b0 ^ 09·b1 ^ 0e·b2 ^ 0b·b3
        //   r3 = 0b·b0 ^ 0d·b1 ^ 09·b2 ^ 0e·b3

        // Extract bytes from columns after AddRoundKey.
        uint8_t m00 = uint8_t((s0 >> 24) & 0xffu); uint8_t m10 = uint8_t((s0 >> 16) & 0xffu);
        uint8_t m20 = uint8_t((s0 >> 8)  & 0xffu); uint8_t m30 = uint8_t( s0        & 0xffu);
        uint8_t m01 = uint8_t((s1 >> 24) & 0xffu); uint8_t m11 = uint8_t((s1 >> 16) & 0xffu);
        uint8_t m21 = uint8_t((s1 >> 8)  & 0xffu); uint8_t m31 = uint8_t( s1        & 0xffu);
        uint8_t m02 = uint8_t((s2 >> 24) & 0xffu); uint8_t m12 = uint8_t((s2 >> 16) & 0xffu);
        uint8_t m22 = uint8_t((s2 >> 8)  & 0xffu); uint8_t m32 = uint8_t( s2        & 0xffu);
        uint8_t m03 = uint8_t((s3 >> 24) & 0xffu); uint8_t m13 = uint8_t((s3 >> 16) & 0xffu);
        uint8_t m23 = uint8_t((s3 >> 8)  & 0xffu); uint8_t m33 = uint8_t( s3        & 0xffu);

        s0 = (uint32_t(mul0e(m00) ^ mul0b(m10) ^ mul0d(m20) ^ mul09(m30)) << 24) |
             (uint32_t(mul09(m00) ^ mul0e(m10) ^ mul0b(m20) ^ mul0d(m30)) << 16) |
             (uint32_t(mul0d(m00) ^ mul09(m10) ^ mul0e(m20) ^ mul0b(m30)) << 8)  |
              uint32_t(mul0b(m00) ^ mul0d(m10) ^ mul09(m20) ^ mul0e(m30));

        s1 = (uint32_t(mul0e(m01) ^ mul0b(m11) ^ mul0d(m21) ^ mul09(m31)) << 24) |
             (uint32_t(mul09(m01) ^ mul0e(m11) ^ mul0b(m21) ^ mul0d(m31)) << 16) |
             (uint32_t(mul0d(m01) ^ mul09(m11) ^ mul0e(m21) ^ mul0b(m31)) << 8)  |
              uint32_t(mul0b(m01) ^ mul0d(m11) ^ mul09(m21) ^ mul0e(m31));

        s2 = (uint32_t(mul0e(m02) ^ mul0b(m12) ^ mul0d(m22) ^ mul09(m32)) << 24) |
             (uint32_t(mul09(m02) ^ mul0e(m12) ^ mul0b(m22) ^ mul0d(m32)) << 16) |
             (uint32_t(mul0d(m02) ^ mul09(m12) ^ mul0e(m22) ^ mul0b(m32)) << 8)  |
              uint32_t(mul0b(m02) ^ mul0d(m12) ^ mul09(m22) ^ mul0e(m32));

        s3 = (uint32_t(mul0e(m03) ^ mul0b(m13) ^ mul0d(m23) ^ mul09(m33)) << 24) |
             (uint32_t(mul09(m03) ^ mul0e(m13) ^ mul0b(m23) ^ mul0d(m33)) << 16) |
             (uint32_t(mul0d(m03) ^ mul09(m13) ^ mul0e(m23) ^ mul0b(m33)) << 8)  |
              uint32_t(mul0b(m03) ^ mul0d(m13) ^ mul09(m23) ^ mul0e(m33));
    }

    // Final round (round 0): InvShiftRows + InvSubBytes + AddRoundKey (no InvMixColumns).
    uint8_t f00 = INV_SBOX[(s0 >> 24) & 0xffu];
    uint8_t f10 = INV_SBOX[(s3 >> 16) & 0xffu];
    uint8_t f20 = INV_SBOX[(s2 >> 8)  & 0xffu];
    uint8_t f30 = INV_SBOX[ s1        & 0xffu];

    uint8_t f01 = INV_SBOX[(s1 >> 24) & 0xffu];
    uint8_t f11 = INV_SBOX[(s0 >> 16) & 0xffu];
    uint8_t f21 = INV_SBOX[(s3 >> 8)  & 0xffu];
    uint8_t f31 = INV_SBOX[ s2        & 0xffu];

    uint8_t f02 = INV_SBOX[(s2 >> 24) & 0xffu];
    uint8_t f12 = INV_SBOX[(s1 >> 16) & 0xffu];
    uint8_t f22 = INV_SBOX[(s0 >> 8)  & 0xffu];
    uint8_t f32 = INV_SBOX[ s3        & 0xffu];

    uint8_t f03 = INV_SBOX[(s3 >> 24) & 0xffu];
    uint8_t f13 = INV_SBOX[(s2 >> 16) & 0xffu];
    uint8_t f23 = INV_SBOX[(s1 >> 8)  & 0xffu];
    uint8_t f33 = INV_SBOX[ s0        & 0xffu];

    s0 = ((uint32_t(f00) << 24) | (uint32_t(f10) << 16) | (uint32_t(f20) << 8) | uint32_t(f30)) ^ rk[0];
    s1 = ((uint32_t(f01) << 24) | (uint32_t(f11) << 16) | (uint32_t(f21) << 8) | uint32_t(f31)) ^ rk[1];
    s2 = ((uint32_t(f02) << 24) | (uint32_t(f12) << 16) | (uint32_t(f22) << 8) | uint32_t(f32)) ^ rk[2];
    s3 = ((uint32_t(f03) << 24) | (uint32_t(f13) << 16) | (uint32_t(f23) << 8) | uint32_t(f33)) ^ rk[3];

    // Store state back to block (big-endian).
    block[0]  = uint8_t((s0 >> 24) & 0xffu); block[1]  = uint8_t((s0 >> 16) & 0xffu);
    block[2]  = uint8_t((s0 >> 8)  & 0xffu); block[3]  = uint8_t( s0        & 0xffu);
    block[4]  = uint8_t((s1 >> 24) & 0xffu); block[5]  = uint8_t((s1 >> 16) & 0xffu);
    block[6]  = uint8_t((s1 >> 8)  & 0xffu); block[7]  = uint8_t( s1        & 0xffu);
    block[8]  = uint8_t((s2 >> 24) & 0xffu); block[9]  = uint8_t((s2 >> 16) & 0xffu);
    block[10] = uint8_t((s2 >> 8)  & 0xffu); block[11] = uint8_t( s2        & 0xffu);
    block[12] = uint8_t((s3 >> 24) & 0xffu); block[13] = uint8_t((s3 >> 16) & 0xffu);
    block[14] = uint8_t((s3 >> 8)  & 0xffu); block[15] = uint8_t( s3        & 0xffu);
}

// ---------------------------------------------------------------------------
// RFC 3394 AES Key Unwrap (§2.2.2)
//
// Attempts to unwrap the encrypted key using the provided round keys.
// Returns true if the integrity check passes (A == 0xA6A6A6A6A6A6A6A6).
//
// The algorithm performs 6×n iterations, where n is the number of 64-bit
// blocks in the Content Encryption Key (CEK).  Each iteration:
//   1. Compute t = n*j + i (1-based counter)
//   2. XOR t (big-endian) into register A
//   3. Concatenate (A ^ t) || R[i] into a 16-byte block
//   4. AES ECB decrypt the block
//   5. Split: A = first 8 bytes, R[i] = last 8 bytes
//
// This function is identical for all AES key sizes — only the underlying
// AES block cipher (called via aes_ecb_decrypt) changes based on the
// compile-time AES_NR constant.
// ---------------------------------------------------------------------------
static inline bool try_key_unwrap(
    thread const uint32_t rk[AES_RK_WORDS],
    constant const uint8_t* encrypted_key,
    uint n
) {
    // Load register A (first 8 bytes) and R[0..n-1] (remaining n×8 bytes).
    // Maximum n=8 covers all standard JWE `enc` algorithms.
    uint8_t a[8];
    uint8_t r[8][8]; // r[i] = 8 bytes, up to 8 blocks

    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        a[i] = encrypted_key[i];
    }
    for (uint i = 0; i < n; ++i) {
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            r[i][k] = encrypted_key[(i + 1) * 8 + k];
        }
    }

    // Unwrap loop: j counts down from 5 to 0, i counts down from n-1 to 0.
    // This reverses the wrap operation's forward iteration.
    for (uint j = 6; j > 0; --j) {
        for (uint i = n; i > 0; --i) {
            // t is 1-based: t = n * (j-1) + i
            uint64_t t = uint64_t(n) * uint64_t(j - 1) + uint64_t(i);

            // XOR t (big-endian) into A before decryption.
            uint8_t t_bytes[8];
            t_bytes[0] = uint8_t((t >> 56) & 0xffu);
            t_bytes[1] = uint8_t((t >> 48) & 0xffu);
            t_bytes[2] = uint8_t((t >> 40) & 0xffu);
            t_bytes[3] = uint8_t((t >> 32) & 0xffu);
            t_bytes[4] = uint8_t((t >> 24) & 0xffu);
            t_bytes[5] = uint8_t((t >> 16) & 0xffu);
            t_bytes[6] = uint8_t((t >> 8)  & 0xffu);
            t_bytes[7] = uint8_t( t        & 0xffu);

            // Build 16-byte AES input: (A ^ t) || R[i-1].
            uint8_t block[16];
            #pragma unroll
            for (uint k = 0; k < 8; ++k) {
                block[k] = a[k] ^ t_bytes[k];
            }
            #pragma unroll
            for (uint k = 0; k < 8; ++k) {
                block[8 + k] = r[i - 1][k];
            }

            // AES ECB decrypt — uses AES_NR rounds (10/12/14 at compile time).
            aes_ecb_decrypt(rk, block);

            // Split result: A = first 8 bytes, R[i-1] = last 8 bytes.
            #pragma unroll
            for (uint k = 0; k < 8; ++k) {
                a[k] = block[k];
            }
            #pragma unroll
            for (uint k = 0; k < 8; ++k) {
                r[i - 1][k] = block[8 + k];
            }
        }
    }

    // Integrity check: A must equal the RFC 3394 default IV (0xA6 repeated 8 times).
    // Any incorrect key produces a pseudo-random A with probability 2^-64 of
    // false positive — effectively zero for our purposes.
    return a[0] == 0xA6u && a[1] == 0xA6u && a[2] == 0xA6u && a[3] == 0xA6u &&
           a[4] == 0xA6u && a[5] == 0xA6u && a[6] == 0xA6u && a[7] == 0xA6u;
}

// ---------------------------------------------------------------------------
// SHA-256 primitives for key derivation of long candidates (> AES_KEY_BYTES)
//
// These are copied from hs256_wordlist.metal.  Each shader is self-contained
// (no cross-file includes in Metal) so the SHA-256 implementation is
// duplicated here.  Only the general kernel needs these; the short_keys
// kernel skips SHA-256 entirely.
// ---------------------------------------------------------------------------

constant uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }
static inline uint32_t ch32(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ ((~x) & z); }
static inline uint32_t maj32(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
static inline uint32_t bsig0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
static inline uint32_t bsig1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
static inline uint32_t ssig0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
static inline uint32_t ssig1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

static inline uint32_t load_be_u32_device(device const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

#define SHA256_ROUND_R(i, a, b, c, d, e, f, g, h) do { \
    uint32_t t1_ = (h) + bsig1((e)) + ch32((e), (f), (g)) + SHA256_K[(i)] + w[(i) & 15]; \
    uint32_t t2_ = bsig0((a)) + maj32((a), (b), (c)); \
    (d) += t1_; \
    (h) = t1_ + t2_; \
} while (0)

#define SHA256_EXPAND_ROUND(i, a, b, c, d, e, f, g, h) do { \
    w[(i) & 15] = ssig1(w[((i) - 2) & 15]) + w[((i) - 7) & 15] + ssig0(w[((i) - 15) & 15]) + w[(i) & 15]; \
    SHA256_ROUND_R(i, a, b, c, d, e, f, g, h); \
} while (0)

#define SHA256_ROUND_8_INIT(base, a, b, c, d, e, f, g, h) \
    SHA256_ROUND_R((base) + 0, a, b, c, d, e, f, g, h); \
    SHA256_ROUND_R((base) + 1, h, a, b, c, d, e, f, g); \
    SHA256_ROUND_R((base) + 2, g, h, a, b, c, d, e, f); \
    SHA256_ROUND_R((base) + 3, f, g, h, a, b, c, d, e); \
    SHA256_ROUND_R((base) + 4, e, f, g, h, a, b, c, d); \
    SHA256_ROUND_R((base) + 5, d, e, f, g, h, a, b, c); \
    SHA256_ROUND_R((base) + 6, c, d, e, f, g, h, a, b); \
    SHA256_ROUND_R((base) + 7, b, c, d, e, f, g, h, a)

#define SHA256_ROUND_8_ROLLING(base, a, b, c, d, e, f, g, h) \
    SHA256_EXPAND_ROUND((base) + 0, a, b, c, d, e, f, g, h); \
    SHA256_EXPAND_ROUND((base) + 1, h, a, b, c, d, e, f, g); \
    SHA256_EXPAND_ROUND((base) + 2, g, h, a, b, c, d, e, f); \
    SHA256_EXPAND_ROUND((base) + 3, f, g, h, a, b, c, d, e); \
    SHA256_EXPAND_ROUND((base) + 4, e, f, g, h, a, b, c, d); \
    SHA256_EXPAND_ROUND((base) + 5, d, e, f, g, h, a, b, c); \
    SHA256_EXPAND_ROUND((base) + 6, c, d, e, f, g, h, a, b); \
    SHA256_EXPAND_ROUND((base) + 7, b, c, d, e, f, g, h, a)

static inline void sha256_compress_rolling(thread uint32_t state[8], thread uint32_t w[16]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    SHA256_ROUND_8_INIT(0,  a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_INIT(8,  a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(16, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(24, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(32, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(40, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(48, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(56, a, b, c, d, e, f, g, h);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// ---------------------------------------------------------------------------
// SHA-256 key derivation — hash candidate and extract AES_KEY_BYTES bytes.
//
// SHA-256 produces a 32-byte (256-bit) digest.  We extract:
//   AES-128: first 16 bytes (4 words) — truncation
//   AES-192: first 24 bytes (6 words) — truncation
//   AES-256: all 32 bytes   (8 words) — full digest, no truncation
//
// This is a simplified single-purpose SHA-256: it processes the candidate
// bytes, applies padding inline, and extracts only AES_KEY_BYTES of the
// digest.
// ---------------------------------------------------------------------------

// Number of SHA-256 state words to extract for the AES key.
// AES-128: 4, AES-192: 6, AES-256: 8 (full digest).
#define SHA256_KEY_WORDS (AES_KEY_BYTES / 4)

static inline void sha256_derive_key(
    device const uint8_t* data,
    uint32_t len,
    thread uint8_t key_out[AES_KEY_BYTES]
) {
    // SHA-256 initial hash values.
    uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    // Process full 64-byte blocks.
    uint32_t offset = 0u;
    while ((len - offset) >= 64u) {
        uint32_t w[16];
        for (uint i = 0; i < 16; ++i)
            w[i] = load_be_u32_device(data + offset + i * 4u);
        sha256_compress_rolling(state, w);
        offset += 64u;
    }

    // Final block: remaining bytes + 0x80 padding + bit length.
    uint32_t rem = len - offset;
    uint64_t bit_len = uint64_t(len) * 8ull;
    {
        uint32_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = 0u;

        // Load remaining full words.
        uint32_t full_words = rem >> 2;
        for (uint32_t i = 0; i < full_words; ++i)
            w[i] = load_be_u32_device(data + offset + i * 4u);

        // Load partial trailing bytes and append 0x80.
        uint32_t tail = rem & 3u;
        uint32_t pad_word = 0u;
        for (uint32_t i = 0; i < tail; ++i)
            pad_word |= uint32_t(data[offset + full_words * 4u + i]) << (24u - 8u * i);
        pad_word |= 0x80u << (24u - 8u * tail);
        w[full_words] = pad_word;

        if (rem > 55u) {
            // Two-block finalization.
            sha256_compress_rolling(state, w);
            #pragma unroll
            for (uint i = 0; i < 16; ++i) w[i] = 0u;
        }
        w[14] = uint32_t((bit_len >> 32u) & 0xffffffffull);
        w[15] = uint32_t(bit_len & 0xffffffffull);
        sha256_compress_rolling(state, w);
    }

    // Extract first AES_KEY_BYTES of the digest (SHA256_KEY_WORDS words, big-endian).
    for (uint i = 0; i < SHA256_KEY_WORDS; ++i) {
        key_out[i * 4 + 0] = uint8_t((state[i] >> 24) & 0xffu);
        key_out[i * 4 + 1] = uint8_t((state[i] >> 16) & 0xffu);
        key_out[i * 4 + 2] = uint8_t((state[i] >> 8)  & 0xffu);
        key_out[i * 4 + 3] = uint8_t( state[i]        & 0xffu);
    }
}

// ===========================================================================
// KERNEL ENTRY POINTS
//
// Buffer bindings (identical layout to HMAC kernels):
//   buffer(0) = params       (AesKwBruteForceParams: encrypted_key_len, n_blocks, candidate_count)
//   buffer(1) = enc_key      (raw encrypted_key bytes from the JWE token)
//   buffer(2) = word_bytes   (concatenated candidate secret bytes)
//   buffer(3) = word_offsets (byte offset of each candidate in word_bytes)
//   buffer(4) = word_lengths (byte length of each candidate)
//   buffer(5) = result_index (atomic: lowest matching gid, or 0xFFFFFFFF)
//
// Kernel names are the same for all three AES variants because each variant
// is compiled into a separate Metal library.  The Rust host selects the
// correct library at construction time based on AesKwVariant.
// ===========================================================================

// General kernel: handles candidates of any length.
// For candidates > AES_KEY_BYTES, derives the AES key via SHA-256 truncation.
// For candidates ≤ AES_KEY_BYTES, zero-pads.
kernel void aeskw_wordlist(
    constant AesKwBruteForceParams& params [[buffer(0)]],
    constant const uint8_t* encrypted_key [[buffer(1)]],
    device const uint8_t* word_bytes [[buffer(2)]],
    device const uint32_t* word_offsets [[buffer(3)]],
    device const uint16_t* word_lengths [[buffer(4)]],
    device atomic_uint* result_index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.candidate_count) return;

    const uint32_t candidate_offset = word_offsets[gid];
    const uint32_t candidate_length = uint32_t(word_lengths[gid]);
    device const uint8_t* candidate = word_bytes + candidate_offset;

    // Derive AES_KEY_BYTES-byte AES key from candidate.
    uint8_t aes_key[AES_KEY_BYTES];
    if (candidate_length <= uint32_t(AES_KEY_BYTES)) {
        // Zero-pad: copy candidate bytes, fill rest with zeros.
        #pragma unroll
        for (uint i = 0; i < AES_KEY_BYTES; ++i) aes_key[i] = 0u;
        for (uint i = 0; i < candidate_length; ++i)
            aes_key[i] = candidate[i];
    } else {
        // SHA-256 hash and truncate to AES_KEY_BYTES bytes.
        sha256_derive_key(candidate, candidate_length, aes_key);
    }

    // Expand to AES_RK_WORDS round key words.
    uint32_t round_keys[AES_RK_WORDS];
    aes_key_expand(aes_key, round_keys);

    // Attempt RFC 3394 unwrap.
    if (try_key_unwrap(round_keys, encrypted_key, params.n_blocks)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}

// Short-key specialization: candidates guaranteed ≤ AES_KEY_BYTES bytes.
// Eliminates the SHA-256 branch entirely, reducing per-thread register
// pressure by ~64 bytes (no w[16] message schedule).  The host selects
// this kernel when the batch's max_word_len ≤ AES_KEY_BYTES, which is
// the common case for dictionary attacks (most passwords are under 16-32
// characters).
kernel void aeskw_wordlist_short_keys(
    constant AesKwBruteForceParams& params [[buffer(0)]],
    constant const uint8_t* encrypted_key [[buffer(1)]],
    device const uint8_t* word_bytes [[buffer(2)]],
    device const uint32_t* word_offsets [[buffer(3)]],
    device const uint16_t* word_lengths [[buffer(4)]],
    device atomic_uint* result_index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.candidate_count) return;

    const uint32_t candidate_offset = word_offsets[gid];
    const uint32_t candidate_length = uint32_t(word_lengths[gid]);
    device const uint8_t* candidate = word_bytes + candidate_offset;

    // Zero-pad candidate to AES_KEY_BYTES bytes (no SHA-256 needed).
    uint8_t aes_key[AES_KEY_BYTES];
    #pragma unroll
    for (uint i = 0; i < AES_KEY_BYTES; ++i) aes_key[i] = 0u;
    for (uint i = 0; i < candidate_length; ++i)
        aes_key[i] = candidate[i];

    // Expand to AES_RK_WORDS round key words.
    uint32_t round_keys[AES_RK_WORDS];
    aes_key_expand(aes_key, round_keys);

    // Attempt RFC 3394 unwrap.
    if (try_key_unwrap(round_keys, encrypted_key, params.n_blocks)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}
