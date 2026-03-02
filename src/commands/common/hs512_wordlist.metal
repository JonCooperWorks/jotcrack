// Metal compute kernel for HMAC-SHA384 and HMAC-SHA512 JWT cracking (wordlist mode).
// Follows the same flat, rolling-schedule design as the HS256 kernel but with 64-bit
// arithmetic, 128-byte HMAC block size, and 80 compression rounds.
#include <metal_stdlib>
using namespace metal;

// Host-to-kernel parameter block. Rust uploads this as `buffer(0)`.
// target_signature holds the expected HMAC output as big-endian uint64_t words:
//   HS384: first 6 words used (48 bytes), HS512: all 8 words used (64 bytes).
struct Hs512BruteForceParams {
    uint64_t target_signature[8];
    uint32_t message_length;
    uint32_t candidate_count;
};

// ---------------------------------------------------------------------------
// SHA-512 round constants (FIPS 180-4, section 4.2.3).
// ---------------------------------------------------------------------------
constant uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, // 0..3
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, // 4..7
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, // 8..11
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL, // 12..15
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, // 16..19
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, // 20..23
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, // 24..27
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, // 28..31
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL, // 32..35
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL, // 36..39
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, // 40..43
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, // 44..47
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, // 48..51
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, // 52..55
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL, // 56..59
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, // 60..63
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, // 64..67
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL, // 68..71
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL, // 72..75
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL, // 76..79
};

// ---------------------------------------------------------------------------
// 64-bit helper functions for SHA-384 / SHA-512.
// ---------------------------------------------------------------------------

static inline uint64_t rotr64(uint64_t x, uint64_t n) {
    return (x >> n) | (x << (64ULL - n));
}

static inline uint64_t ch64(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ ((~x) & z);
}

static inline uint64_t maj64(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

// SHA-512 big sigma 0.
static inline uint64_t bsig0_64(uint64_t x) {
    return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39);
}

// SHA-512 big sigma 1.
static inline uint64_t bsig1_64(uint64_t x) {
    return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41);
}

// SHA-512 small sigma 0 (message schedule expansion).
static inline uint64_t ssig0_64(uint64_t x) {
    return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7);
}

// SHA-512 small sigma 1 (message schedule expansion).
static inline uint64_t ssig1_64(uint64_t x) {
    return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6);
}

// ---------------------------------------------------------------------------
// Big-endian load/store for 64-bit words (thread / device / constant).
// ---------------------------------------------------------------------------

static inline uint64_t load_be_u64(thread const uint8_t* p) {
    return (uint64_t(p[0]) << 56) | (uint64_t(p[1]) << 48) |
           (uint64_t(p[2]) << 40) | (uint64_t(p[3]) << 32) |
           (uint64_t(p[4]) << 24) | (uint64_t(p[5]) << 16) |
           (uint64_t(p[6]) << 8)  |  uint64_t(p[7]);
}

static inline uint64_t load_be_u64(device const uint8_t* p) {
    return (uint64_t(p[0]) << 56) | (uint64_t(p[1]) << 48) |
           (uint64_t(p[2]) << 40) | (uint64_t(p[3]) << 32) |
           (uint64_t(p[4]) << 24) | (uint64_t(p[5]) << 16) |
           (uint64_t(p[6]) << 8)  |  uint64_t(p[7]);
}

static inline uint64_t load_be_u64(constant const uint8_t* p) {
    return (uint64_t(p[0]) << 56) | (uint64_t(p[1]) << 48) |
           (uint64_t(p[2]) << 40) | (uint64_t(p[3]) << 32) |
           (uint64_t(p[4]) << 24) | (uint64_t(p[5]) << 16) |
           (uint64_t(p[6]) << 8)  |  uint64_t(p[7]);
}

static inline void store_be_u64(thread uint8_t* p, uint64_t v) {
    p[0] = uint8_t((v >> 56) & 0xffULL);
    p[1] = uint8_t((v >> 48) & 0xffULL);
    p[2] = uint8_t((v >> 40) & 0xffULL);
    p[3] = uint8_t((v >> 32) & 0xffULL);
    p[4] = uint8_t((v >> 24) & 0xffULL);
    p[5] = uint8_t((v >> 16) & 0xffULL);
    p[6] = uint8_t((v >> 8)  & 0xffULL);
    p[7] = uint8_t(v & 0xffULL);
}

// ---------------------------------------------------------------------------
// Rolling 16-word message schedule for SHA-512 (80 rounds).
//
// Same strategy as the SHA-256 kernel: keep only w[16] and update w[i & 15]
// in-place during rounds 16-79, saving register pressure vs. a full w[80].
// ---------------------------------------------------------------------------

// Single SHA-512 round using rolling w[i & 15] indexing.
#define SHA512_ROUND_R(i, a, b, c, d, e, f, g, h) do { \
    uint64_t t1_ = (h) + bsig1_64((e)) + ch64((e), (f), (g)) + SHA512_K[(i)] + w[(i) & 15]; \
    uint64_t t2_ = bsig0_64((a)) + maj64((a), (b), (c)); \
    (d) += t1_; \
    (h) = t1_ + t2_; \
} while (0)

// Rolling expand + round: update w[i&15] from the recurrence, then compress.
#define SHA512_EXPAND_ROUND(i, a, b, c, d, e, f, g, h) do { \
    w[(i) & 15] = ssig1_64(w[((i) - 2) & 15]) + w[((i) - 7) & 15] + ssig0_64(w[((i) - 15) & 15]) + w[(i) & 15]; \
    SHA512_ROUND_R(i, a, b, c, d, e, f, g, h); \
} while (0)

// 8 consecutive rounds for the first 16 rounds (no expansion needed).
#define SHA512_ROUND_8_INIT(base, a, b, c, d, e, f, g, h) \
    SHA512_ROUND_R((base) + 0, a, b, c, d, e, f, g, h); \
    SHA512_ROUND_R((base) + 1, h, a, b, c, d, e, f, g); \
    SHA512_ROUND_R((base) + 2, g, h, a, b, c, d, e, f); \
    SHA512_ROUND_R((base) + 3, f, g, h, a, b, c, d, e); \
    SHA512_ROUND_R((base) + 4, e, f, g, h, a, b, c, d); \
    SHA512_ROUND_R((base) + 5, d, e, f, g, h, a, b, c); \
    SHA512_ROUND_R((base) + 6, c, d, e, f, g, h, a, b); \
    SHA512_ROUND_R((base) + 7, b, c, d, e, f, g, h, a)

// 8 consecutive rounds with rolling expansion (rounds 16-79).
#define SHA512_ROUND_8_ROLLING(base, a, b, c, d, e, f, g, h) \
    SHA512_EXPAND_ROUND((base) + 0, a, b, c, d, e, f, g, h); \
    SHA512_EXPAND_ROUND((base) + 1, h, a, b, c, d, e, f, g); \
    SHA512_EXPAND_ROUND((base) + 2, g, h, a, b, c, d, e, f); \
    SHA512_EXPAND_ROUND((base) + 3, f, g, h, a, b, c, d, e); \
    SHA512_EXPAND_ROUND((base) + 4, e, f, g, h, a, b, c, d); \
    SHA512_EXPAND_ROUND((base) + 5, d, e, f, g, h, a, b, c); \
    SHA512_EXPAND_ROUND((base) + 6, c, d, e, f, g, h, a, b); \
    SHA512_EXPAND_ROUND((base) + 7, b, c, d, e, f, g, h, a)

// Compress one 1024-bit block using a rolling 16-word schedule (80 rounds).
// w[16] is modified in-place by the rolling expansion (rounds 16-79).
static inline void sha512_compress_rolling(thread uint64_t state[8], thread uint64_t w[16]) {
    uint64_t a = state[0];
    uint64_t b = state[1];
    uint64_t c = state[2];
    uint64_t d = state[3];
    uint64_t e = state[4];
    uint64_t f = state[5];
    uint64_t g = state[6];
    uint64_t h = state[7];

    // Rounds 0-15: use w[i] directly (no expansion needed).
    SHA512_ROUND_8_INIT(0,  a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_INIT(8,  a, b, c, d, e, f, g, h);
    // Rounds 16-79: expand w[i&15] in-place before each round.
    SHA512_ROUND_8_ROLLING(16, a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_ROLLING(24, a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_ROLLING(32, a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_ROLLING(40, a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_ROLLING(48, a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_ROLLING(56, a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_ROLLING(64, a, b, c, d, e, f, g, h);
    SHA512_ROUND_8_ROLLING(72, a, b, c, d, e, f, g, h);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// ---------------------------------------------------------------------------
// Streaming SHA-512 context (used only for the long-key fallback path).
// ---------------------------------------------------------------------------

struct Sha512Ctx {
    uint64_t state[8];
    uint8_t block[128];            // 1024-bit block buffer.
    uint32_t block_len;
    uint64_t total_len;
};

static inline void sha512_init(thread Sha512Ctx& ctx) {
    ctx.state[0] = 0x6a09e667f3bcc908ULL;
    ctx.state[1] = 0xbb67ae8584caa73bULL;
    ctx.state[2] = 0x3c6ef372fe94f82bULL;
    ctx.state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx.state[4] = 0x510e527fade682d1ULL;
    ctx.state[5] = 0x9b05688c2b3e6c1fULL;
    ctx.state[6] = 0x1f83d9abfb41bd6bULL;
    ctx.state[7] = 0x5be0cd19137e2179ULL;
    ctx.block_len = 0;
    ctx.total_len = 0;
}

static inline void sha384_init(thread Sha512Ctx& ctx) {
    ctx.state[0] = 0xcbbb9d5dc1059ed8ULL;
    ctx.state[1] = 0x629a292a367cd507ULL;
    ctx.state[2] = 0x9159015a3070dd17ULL;
    ctx.state[3] = 0x152fecd8f70e5939ULL;
    ctx.state[4] = 0x67332667ffc00b31ULL;
    ctx.state[5] = 0x8eb44a8768581511ULL;
    ctx.state[6] = 0xdb0c2e0d64f98fa7ULL;
    ctx.state[7] = 0x47b5481dbefa4fa4ULL;
    ctx.block_len = 0;
    ctx.total_len = 0;
}

// Compress helpers for different address spaces.
static inline void sha512_compress_from_thread(thread uint64_t state[8], thread const uint8_t* block) {
    uint64_t w[16];
    for (uint i = 0; i < 16; ++i) {
        w[i] = load_be_u64(block + (i * 8u));
    }
    sha512_compress_rolling(state, w);
}

static inline void sha512_compress_from_device(thread uint64_t state[8], device const uint8_t* block) {
    uint64_t w[16];
    for (uint i = 0; i < 16; ++i) {
        w[i] = load_be_u64(block + (i * 8u));
    }
    sha512_compress_rolling(state, w);
}

// Update SHA-512 state from device memory (used for hashing long keys).
static inline void sha512_update_device(thread Sha512Ctx& ctx, device const uint8_t* data, uint32_t len) {
    if (len == 0u) return;
    ctx.total_len += uint64_t(len);

    uint32_t offset = 0u;
    if (ctx.block_len != 0u) {
        const uint32_t need = 128u - ctx.block_len;
        const uint32_t take = (len < need) ? len : need;
        for (uint32_t i = 0; i < take; ++i) {
            ctx.block[ctx.block_len + i] = data[i];
        }
        ctx.block_len += take;
        offset += take;
        if (ctx.block_len == 128u) {
            sha512_compress_from_thread(ctx.state, ctx.block);
            ctx.block_len = 0u;
        }
    }

    while ((len - offset) >= 128u) {
        sha512_compress_from_device(ctx.state, data + offset);
        offset += 128u;
    }

    for (; offset < len; ++offset) {
        ctx.block[ctx.block_len++] = data[offset];
    }
}

// Finalize SHA-512 without serializing (state[0..7] holds result as words).
static inline void sha512_finalize(thread Sha512Ctx& ctx) {
    const uint64_t bit_len = ctx.total_len * 8ULL;
    ctx.block[ctx.block_len++] = 0x80u;
    if (ctx.block_len > 112u) {
        while (ctx.block_len < 128u) ctx.block[ctx.block_len++] = 0u;
        sha512_compress_from_thread(ctx.state, ctx.block);
        ctx.block_len = 0u;
    }
    while (ctx.block_len < 112u) ctx.block[ctx.block_len++] = 0u;
    // SHA-512 uses a 128-bit (16-byte) big-endian length field.
    // Upper 64 bits are zero for messages < 2^64 bits.
    for (uint i = 0; i < 8; ++i)
        ctx.block[112u + i] = 0u;
    for (uint i = 0; i < 8; ++i)
        ctx.block[120u + i] = uint8_t((bit_len >> (56u - 8u * i)) & 0xffULL);
    sha512_compress_from_thread(ctx.state, ctx.block);
    ctx.block_len = 0u;
}

// ---------------------------------------------------------------------------
// Key loading: 128-byte HMAC block as 16 x uint64_t words, zero-padded.
// ---------------------------------------------------------------------------

static inline void load_key_words_64(device const uint8_t* key, uint32_t key_len, thread uint64_t kw[16]) {
    #pragma unroll
    for (uint i = 0; i < 16; ++i) kw[i] = 0ULL;

    const uint32_t full_words = key_len >> 3;
    for (uint32_t i = 0; i < full_words; ++i)
        kw[i] = load_be_u64(key + i * 8u);

    const uint32_t remaining = key_len & 7u;
    if (remaining > 0u) {
        uint64_t val = 0ULL;
        device const uint8_t* tail = key + (full_words * 8u);
        for (uint32_t i = 0; i < remaining; ++i)
            val |= uint64_t(tail[i]) << (56u - 8u * i);
        kw[full_words] = val;
    }
}

// ---------------------------------------------------------------------------
// Flat HMAC-SHA384: no Sha512Ctx on the hot path.
//
// Block size = 128 bytes.  ipad/opad XOR = 0x3636363636363636 / 0x5c5c5c5c5c5c5c5c.
// Inner digest = 48 bytes (first 6 state words of SHA-384).
// Outer message = 128 (opad) + 48 (digest) = 176 bytes  ->  bit_len = 1408.
// ---------------------------------------------------------------------------

static inline bool hmac_sha384_from_key_words(
    thread const uint64_t kw[16],
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint64_t target[8]
) {
    // --- Inner hash: SHA-384(ipad || message) ---
    uint64_t istate[8] = {
        0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL,
        0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
        0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL,
        0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL
    };

    // Compress ipad block: kw XOR 0x3636363636363636.
    {
        uint64_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x3636363636363636ULL;
        sha512_compress_rolling(istate, w);
    }

    // Process full 128-byte message blocks from constant memory.
    uint32_t offset = 0u;
    while ((message_len - offset) >= 128u) {
        uint64_t w[16];
        for (uint i = 0; i < 16; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }
        sha512_compress_rolling(istate, w);
        offset += 128u;
    }

    // Finalize inner hash: remaining bytes + 0x80 padding + 128-bit bit length.
    const uint32_t rem = message_len - offset;
    const uint64_t inner_bit_len = (128ULL + uint64_t(message_len)) * 8ULL;
    {
        uint64_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = 0ULL;

        // Load remaining full 8-byte words from message.
        const uint32_t full_words = rem >> 3;
        for (uint32_t i = 0; i < full_words; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }

        // Load partial trailing bytes and append 0x80 padding byte.
        const uint32_t tail = rem & 7u;
        uint64_t pad_word = 0ULL;
        constant const uint8_t* tail_ptr = message + offset + (full_words * 8u);
        for (uint32_t i = 0; i < tail; ++i) {
            pad_word |= uint64_t(tail_ptr[i]) << (56u - 8u * i);
        }
        pad_word |= uint64_t(0x80u) << (56u - 8u * tail);
        w[full_words] = pad_word;

        if (rem > 111u) {
            // Two-block finalization: data + 0x80 don't fit with the 16-byte length field.
            sha512_compress_rolling(istate, w);
            #pragma unroll
            for (uint i = 0; i < 16; ++i) w[i] = 0ULL;
        }
        // 128-bit big-endian bit length: upper 64 bits = 0, lower 64 bits = inner_bit_len.
        w[14] = 0ULL;
        w[15] = inner_bit_len;
        sha512_compress_rolling(istate, w);
    }
    // istate[0..5] now holds the 48-byte SHA-384 inner digest as words.

    // --- Outer hash: SHA-384(opad || inner_digest) ---
    uint64_t ostate[8] = {
        0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL,
        0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
        0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL,
        0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL
    };

    // Compress opad block: kw XOR 0x5c5c5c5c5c5c5c5c.
    {
        uint64_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        sha512_compress_rolling(ostate, w);
    }

    // Compress final outer block: 48-byte inner digest + padding.
    // Outer message total = 128 (opad) + 48 (digest) = 176 bytes -> bit_len = 1408.
    {
        uint64_t w[16];
        w[0] = istate[0];
        w[1] = istate[1];
        w[2] = istate[2];
        w[3] = istate[3];
        w[4] = istate[4];
        w[5] = istate[5];
        // 48 bytes = 6 words, then 0x80 padding starts at byte 48 (word 6, MSB).
        w[6] = 0x8000000000000000ULL;
        w[7]  = 0ULL;
        w[8]  = 0ULL;
        w[9]  = 0ULL;
        w[10] = 0ULL;
        w[11] = 0ULL;
        w[12] = 0ULL;
        w[13] = 0ULL;
        // 128-bit big-endian bit length = 1408.
        w[14] = 0ULL;
        w[15] = 1408ULL;
        sha512_compress_rolling(ostate, w);
    }

    // Compare first 6 words (48 bytes) against target.
    #pragma unroll
    for (uint i = 0; i < 6; ++i) {
        if (ostate[i] != target[i]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Flat HMAC-SHA512: no Sha512Ctx on the hot path.
//
// Block size = 128 bytes.  ipad/opad XOR = 0x3636363636363636 / 0x5c5c5c5c5c5c5c5c.
// Inner digest = 64 bytes (all 8 state words of SHA-512).
// Outer message = 128 (opad) + 64 (digest) = 192 bytes  ->  bit_len = 1536.
// 192 bytes = one 128-byte opad block + one 64-byte digest block.
// The digest (64 bytes) does NOT fit with padding in one block (need 64 + 1 + padding + 16 = 81+ bytes > 128),
// so the outer hash after the opad compress needs TWO more blocks:
//   Block 2: w[0..7] = inner state words (64 bytes of digest fills entire block... NO:
//   64 bytes = 8 words, which leaves 8 words for padding+length in the same 128-byte block).
//   Actually 64 bytes + 1 (0x80) + padding + 16 (length) = 81 minimum bytes, fits in 128.
//   So: w[0..7] = inner digest, w[8] = 0x80..., w[9..13] = 0, w[14..15] = bit_len.
// Wait -- 64 bytes of digest. After opad (128 bytes), next data is the 64-byte digest.
// A single 128-byte block can hold 128 bytes. 64 bytes of digest + 1 byte padding = 65 bytes,
// plus 16 bytes for length = 81 bytes. 81 <= 128. So it fits in one block.
// ---------------------------------------------------------------------------

static inline bool hmac_sha512_from_key_words(
    thread const uint64_t kw[16],
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint64_t target[8]
) {
    // --- Inner hash: SHA-512(ipad || message) ---
    uint64_t istate[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    // Compress ipad block: kw XOR 0x3636363636363636.
    {
        uint64_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x3636363636363636ULL;
        sha512_compress_rolling(istate, w);
    }

    // Process full 128-byte message blocks from constant memory.
    uint32_t offset = 0u;
    while ((message_len - offset) >= 128u) {
        uint64_t w[16];
        for (uint i = 0; i < 16; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }
        sha512_compress_rolling(istate, w);
        offset += 128u;
    }

    // Finalize inner hash: remaining bytes + 0x80 padding + 128-bit bit length.
    const uint32_t rem = message_len - offset;
    const uint64_t inner_bit_len = (128ULL + uint64_t(message_len)) * 8ULL;
    {
        uint64_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = 0ULL;

        const uint32_t full_words = rem >> 3;
        for (uint32_t i = 0; i < full_words; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }

        const uint32_t tail = rem & 7u;
        uint64_t pad_word = 0ULL;
        constant const uint8_t* tail_ptr = message + offset + (full_words * 8u);
        for (uint32_t i = 0; i < tail; ++i) {
            pad_word |= uint64_t(tail_ptr[i]) << (56u - 8u * i);
        }
        pad_word |= uint64_t(0x80u) << (56u - 8u * tail);
        w[full_words] = pad_word;

        if (rem > 111u) {
            sha512_compress_rolling(istate, w);
            #pragma unroll
            for (uint i = 0; i < 16; ++i) w[i] = 0ULL;
        }
        w[14] = 0ULL;
        w[15] = inner_bit_len;
        sha512_compress_rolling(istate, w);
    }
    // istate[0..7] now holds the 64-byte SHA-512 inner digest.

    // --- Outer hash: SHA-512(opad || inner_digest) ---
    uint64_t ostate[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    // Compress opad block: kw XOR 0x5c5c5c5c5c5c5c5c.
    {
        uint64_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        sha512_compress_rolling(ostate, w);
    }

    // Compress final outer block: 64-byte inner digest + padding + length.
    // 64 bytes digest = 8 words.  Remaining in block: 8 words (64 bytes).
    // Need: 1 byte (0x80) + padding + 16 bytes (length) = 17 bytes minimum.
    // 64 bytes available >= 17, so it fits in one block.
    // Outer message total = 128 (opad) + 64 (digest) = 192 bytes -> bit_len = 1536.
    {
        uint64_t w[16];
        w[0] = istate[0];
        w[1] = istate[1];
        w[2] = istate[2];
        w[3] = istate[3];
        w[4] = istate[4];
        w[5] = istate[5];
        w[6] = istate[6];
        w[7] = istate[7];
        // 64 bytes = 8 words, then 0x80 padding starts at byte 64 (word 8, MSB).
        w[8] = 0x8000000000000000ULL;
        w[9]  = 0ULL;
        w[10] = 0ULL;
        w[11] = 0ULL;
        w[12] = 0ULL;
        w[13] = 0ULL;
        // 128-bit big-endian bit length = 1536.
        w[14] = 0ULL;
        w[15] = 1536ULL;
        sha512_compress_rolling(ostate, w);
    }

    // Compare all 8 words (64 bytes) against target.
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        if (ostate[i] != target[i]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Full HMAC wrappers with long-key fallback (key > 128 bytes -> hash first).
// ---------------------------------------------------------------------------

// HMAC-SHA384: any key length.
static inline bool hmac_sha384_matches(
    device const uint8_t* key,
    uint32_t key_len,
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint64_t target[8]
) {
    uint64_t kw[16];

    if (key_len > 128u) {
        // Long keys: hash with SHA-384 first (produces 48 bytes = 6 words).
        #pragma unroll
        for (uint i = 0; i < 16; ++i) kw[i] = 0ULL;
        Sha512Ctx key_ctx;
        sha384_init(key_ctx);
        sha512_update_device(key_ctx, key, key_len);
        sha512_finalize(key_ctx);
        #pragma unroll
        for (uint i = 0; i < 6; ++i) kw[i] = key_ctx.state[i];
    } else {
        load_key_words_64(key, key_len, kw);
    }

    return hmac_sha384_from_key_words(kw, message, message_len, target);
}

// HMAC-SHA384: short-key specialization (keys guaranteed <= 128 bytes).
static inline bool hmac_sha384_short_key_matches(
    device const uint8_t* key,
    uint32_t key_len,
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint64_t target[8]
) {
    uint64_t kw[16];
    load_key_words_64(key, key_len, kw);
    return hmac_sha384_from_key_words(kw, message, message_len, target);
}

// HMAC-SHA512: any key length.
static inline bool hmac_sha512_matches(
    device const uint8_t* key,
    uint32_t key_len,
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint64_t target[8]
) {
    uint64_t kw[16];

    if (key_len > 128u) {
        // Long keys: hash with SHA-512 first (produces 64 bytes = 8 words).
        #pragma unroll
        for (uint i = 0; i < 16; ++i) kw[i] = 0ULL;
        Sha512Ctx key_ctx;
        sha512_init(key_ctx);
        sha512_update_device(key_ctx, key, key_len);
        sha512_finalize(key_ctx);
        #pragma unroll
        for (uint i = 0; i < 8; ++i) kw[i] = key_ctx.state[i];
    } else {
        load_key_words_64(key, key_len, kw);
    }

    return hmac_sha512_from_key_words(kw, message, message_len, target);
}

// HMAC-SHA512: short-key specialization (keys guaranteed <= 128 bytes).
static inline bool hmac_sha512_short_key_matches(
    device const uint8_t* key,
    uint32_t key_len,
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint64_t target[8]
) {
    uint64_t kw[16];
    load_key_words_64(key, key_len, kw);
    return hmac_sha512_from_key_words(kw, message, message_len, target);
}

// ---------------------------------------------------------------------------
// Kernel entry points.
// Buffer bindings match the HS256 kernel:
//   buffer(0) = params
//   buffer(1) = message (JWT header.payload)
//   buffer(2) = word_bytes (concatenated candidate secrets)
//   buffer(3) = word_offsets (byte offset of each candidate)
//   buffer(4) = word_lengths (byte length of each candidate)
//   buffer(5) = result_index (atomic min: lowest matching gid)
// ---------------------------------------------------------------------------

kernel void hs384_wordlist(
    constant Hs512BruteForceParams& params [[buffer(0)]],
    constant const uint8_t* message_bytes [[buffer(1)]],
    device const uint8_t* word_bytes [[buffer(2)]],
    device const uint32_t* word_offsets [[buffer(3)]],
    device const uint16_t* word_lengths [[buffer(4)]],
    device atomic_uint* result_index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.candidate_count) return;

    const uint32_t candidate_offset = word_offsets[gid];
    const uint32_t candidate_length = uint32_t(word_lengths[gid]);
    device const uint8_t* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha384_matches(candidate_secret, candidate_length,
                            message_bytes, params.message_length,
                            params.target_signature)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}

kernel void hs384_wordlist_short_keys(
    constant Hs512BruteForceParams& params [[buffer(0)]],
    constant const uint8_t* message_bytes [[buffer(1)]],
    device const uint8_t* word_bytes [[buffer(2)]],
    device const uint32_t* word_offsets [[buffer(3)]],
    device const uint16_t* word_lengths [[buffer(4)]],
    device atomic_uint* result_index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.candidate_count) return;

    const uint32_t candidate_offset = word_offsets[gid];
    const uint32_t candidate_length = uint32_t(word_lengths[gid]);
    device const uint8_t* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha384_short_key_matches(candidate_secret, candidate_length,
                                      message_bytes, params.message_length,
                                      params.target_signature)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}

kernel void hs512_wordlist(
    constant Hs512BruteForceParams& params [[buffer(0)]],
    constant const uint8_t* message_bytes [[buffer(1)]],
    device const uint8_t* word_bytes [[buffer(2)]],
    device const uint32_t* word_offsets [[buffer(3)]],
    device const uint16_t* word_lengths [[buffer(4)]],
    device atomic_uint* result_index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.candidate_count) return;

    const uint32_t candidate_offset = word_offsets[gid];
    const uint32_t candidate_length = uint32_t(word_lengths[gid]);
    device const uint8_t* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha512_matches(candidate_secret, candidate_length,
                            message_bytes, params.message_length,
                            params.target_signature)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}

kernel void hs512_wordlist_short_keys(
    constant Hs512BruteForceParams& params [[buffer(0)]],
    constant const uint8_t* message_bytes [[buffer(1)]],
    device const uint8_t* word_bytes [[buffer(2)]],
    device const uint32_t* word_offsets [[buffer(3)]],
    device const uint16_t* word_lengths [[buffer(4)]],
    device atomic_uint* result_index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.candidate_count) return;

    const uint32_t candidate_offset = word_offsets[gid];
    const uint32_t candidate_length = uint32_t(word_lengths[gid]);
    device const uint8_t* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha512_short_key_matches(candidate_secret, candidate_length,
                                      message_bytes, params.message_length,
                                      params.target_signature)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}
