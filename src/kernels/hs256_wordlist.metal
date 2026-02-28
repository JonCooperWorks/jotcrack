// Import Metal's standard library (types like `uint32_t`, kernel attributes, atomics, etc.).
#include <metal_stdlib>
// Pull `metal::` names into the local namespace so code can use `kernel`, `device`, `thread`, etc. directly.
using namespace metal;

// Host-to-kernel parameter block. Rust uploads this as `buffer(0)` with the exact same field order/layout.
struct Hs256BruteForceParams {
    uint8_t target_signature[32];      // Expected HS256 HMAC digest (32 bytes) we compare against.
    uint32_t message_length;           // Actual byte length of `message_bytes` (the JWT `header.payload` string).
    uint32_t candidate_count;          // Number of candidate secrets in this GPU batch.
    uint32_t reserved0;                // Reserved/padding slot kept for ABI stability with the Rust-side params struct.
                                      // Older versions used this as `candidate_index_base` (absolute wordlist index
                                      // of candidate #0). The kernel now reports batch-local match indices (`gid`)
                                      // so host-side absolute indexing can scale past `uint32_t`.
};

// Minimal SHA-256 streaming context used per GPU thread.
struct Sha256Ctx {
    uint32_t state[8];                 // Current SHA-256 state words (a..h / hash state).
    uint8_t block[64];                 // Current 512-bit block buffer being filled before compression.
    uint32_t block_len;                // Number of bytes currently stored in `block`.
    uint64_t total_len;                // Total message length processed so far, in bytes (needed for final padding length).
};

// SHA-256 round constants from the specification (FIPS 180-4).
constant uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, // Rounds 0..3
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, // Rounds 4..7
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, // Rounds 8..11
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, // Rounds 12..15
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, // Rounds 16..19
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, // Rounds 20..23
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, // Rounds 24..27
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, // Rounds 28..31
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, // Rounds 32..35
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, // Rounds 36..39
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, // Rounds 40..43
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, // Rounds 44..47
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, // Rounds 48..51
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u, // Rounds 52..55
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, // Rounds 56..59
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u, // Rounds 60..63
};

// Rotate-right helper for 32-bit values. SHA-256 uses rotations heavily.
static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n)); // Split + wrap bits from low end back to high end.
}

// SHA-256 choice function: chooses bits from y/z based on x.
static inline uint32_t ch32(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ ((~x) & z); // Equivalent to: x ? y : z, bitwise.
}

// SHA-256 majority function: bit is 1 if at least two inputs have bit 1.
static inline uint32_t maj32(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z); // Standard SHA-256 boolean function.
}

// SHA-256 big sigma 0 for the compression round math.
static inline uint32_t bsig0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); // XOR of three rotations.
}

// SHA-256 big sigma 1 for the compression round math.
static inline uint32_t bsig1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); // XOR of three rotations.
}

// SHA-256 small sigma 0 for message schedule expansion.
static inline uint32_t ssig0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); // Two rotates + one logical shift.
}

// SHA-256 small sigma 1 for message schedule expansion.
static inline uint32_t ssig1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); // Two rotates + one logical shift.
}

// Read 4 bytes in big-endian order into one 32-bit word.
static inline uint32_t load_be_u32(thread const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | // Most-significant byte.
           (uint32_t(p[1]) << 16) | // Next byte.
           (uint32_t(p[2]) << 8)  | // Next byte.
           uint32_t(p[3]);          // Least-significant byte.
}

static inline uint32_t load_be_u32(device const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | // Most-significant byte.
           (uint32_t(p[1]) << 16) | // Next byte.
           (uint32_t(p[2]) << 8)  | // Next byte.
           uint32_t(p[3]);          // Least-significant byte.
}

static inline uint32_t load_be_u32(constant const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | // Most-significant byte.
           (uint32_t(p[1]) << 16) | // Next byte.
           (uint32_t(p[2]) << 8)  | // Next byte.
           uint32_t(p[3]);          // Least-significant byte.
}

// Write one 32-bit word into 4 bytes in big-endian order.
static inline void store_be_u32(thread uint8_t* p, uint32_t v) {
    p[0] = uint8_t((v >> 24) & 0xffu); // Highest byte.
    p[1] = uint8_t((v >> 16) & 0xffu); // Next byte.
    p[2] = uint8_t((v >> 8) & 0xffu);  // Next byte.
    p[3] = uint8_t(v & 0xffu);         // Lowest byte.
}

// Initialize a SHA-256 context to the algorithm's fixed IV (initial hash values).
static inline void sha256_init(thread Sha256Ctx& ctx) {
    ctx.state[0] = 0x6a09e667u; // H0
    ctx.state[1] = 0xbb67ae85u; // H1
    ctx.state[2] = 0x3c6ef372u; // H2
    ctx.state[3] = 0xa54ff53au; // H3
    ctx.state[4] = 0x510e527fu; // H4
    ctx.state[5] = 0x9b05688cu; // H5
    ctx.state[6] = 0x1f83d9abu; // H6
    ctx.state[7] = 0x5be0cd19u; // H7
    ctx.block_len = 0;          // No bytes buffered yet.
    ctx.total_len = 0;          // No bytes processed yet.
}

// Compress exactly one 64-byte SHA-256 block into the running state.
#define SHA256_EXPAND(i) \
    w[(i)] = ssig1(w[(i) - 2]) + w[(i) - 7] + ssig0(w[(i) - 15]) + w[(i) - 16]

#define SHA256_EXPAND_16(base) \
    SHA256_EXPAND((base) + 0);  \
    SHA256_EXPAND((base) + 1);  \
    SHA256_EXPAND((base) + 2);  \
    SHA256_EXPAND((base) + 3);  \
    SHA256_EXPAND((base) + 4);  \
    SHA256_EXPAND((base) + 5);  \
    SHA256_EXPAND((base) + 6);  \
    SHA256_EXPAND((base) + 7);  \
    SHA256_EXPAND((base) + 8);  \
    SHA256_EXPAND((base) + 9);  \
    SHA256_EXPAND((base) + 10); \
    SHA256_EXPAND((base) + 11); \
    SHA256_EXPAND((base) + 12); \
    SHA256_EXPAND((base) + 13); \
    SHA256_EXPAND((base) + 14); \
    SHA256_EXPAND((base) + 15)

#define SHA256_ROUND(i, a, b, c, d, e, f, g, h) do { \
    uint32_t t1_ = (h) + bsig1((e)) + ch32((e), (f), (g)) + SHA256_K[(i)] + w[(i)]; \
    uint32_t t2_ = bsig0((a)) + maj32((a), (b), (c)); \
    (d) += t1_; \
    (h) = t1_ + t2_; \
} while (0)

#define SHA256_ROUND_8(base, a, b, c, d, e, f, g, h) \
    SHA256_ROUND((base) + 0, a, b, c, d, e, f, g, h); \
    SHA256_ROUND((base) + 1, h, a, b, c, d, e, f, g); \
    SHA256_ROUND((base) + 2, g, h, a, b, c, d, e, f); \
    SHA256_ROUND((base) + 3, f, g, h, a, b, c, d, e); \
    SHA256_ROUND((base) + 4, e, f, g, h, a, b, c, d); \
    SHA256_ROUND((base) + 5, d, e, f, g, h, a, b, c); \
    SHA256_ROUND((base) + 6, c, d, e, f, g, h, a, b); \
    SHA256_ROUND((base) + 7, b, c, d, e, f, g, h, a)

static inline void sha256_compress_words(thread Sha256Ctx& ctx, thread uint32_t w[64]) {
    SHA256_EXPAND_16(16);
    SHA256_EXPAND_16(32);
    SHA256_EXPAND_16(48);

    uint32_t a = ctx.state[0]; // Working variable a starts from current hash state.
    uint32_t b = ctx.state[1]; // Working variable b.
    uint32_t c = ctx.state[2]; // Working variable c.
    uint32_t d = ctx.state[3]; // Working variable d.
    uint32_t e = ctx.state[4]; // Working variable e.
    uint32_t f = ctx.state[5]; // Working variable f.
    uint32_t g = ctx.state[6]; // Working variable g.
    uint32_t h = ctx.state[7]; // Working variable h.

    SHA256_ROUND_8(0,  a, b, c, d, e, f, g, h);
    SHA256_ROUND_8(8,  a, b, c, d, e, f, g, h);
    SHA256_ROUND_8(16, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8(24, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8(32, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8(40, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8(48, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8(56, a, b, c, d, e, f, g, h);

    ctx.state[0] += a; // Feed-forward: add working variables back into the hash state.
    ctx.state[1] += b; // Feed-forward for state word 1.
    ctx.state[2] += c; // Feed-forward for state word 2.
    ctx.state[3] += d; // Feed-forward for state word 3.
    ctx.state[4] += e; // Feed-forward for state word 4.
    ctx.state[5] += f; // Feed-forward for state word 5.
    ctx.state[6] += g; // Feed-forward for state word 6.
    ctx.state[7] += h; // Feed-forward for state word 7.
}

static inline void sha256_compress(thread Sha256Ctx& ctx, thread const uint8_t* block) {
    uint32_t w[64]; // Message schedule array for this block.
    for (uint i = 0; i < 16; ++i) { // First 16 words come directly from the 64-byte input block.
        w[i] = load_be_u32(block + (i * 4u)); // Convert each 4-byte chunk to a 32-bit word.
    }
    sha256_compress_words(ctx, w);
}

static inline void sha256_compress_device(thread Sha256Ctx& ctx, device const uint8_t* block) {
    uint32_t w[64];
    for (uint i = 0; i < 16; ++i) {
        w[i] = load_be_u32(block + (i * 4u));
    }
    sha256_compress_words(ctx, w);
}

static inline void sha256_compress_constant(thread Sha256Ctx& ctx, constant const uint8_t* block) {
    uint32_t w[64];
    for (uint i = 0; i < 16; ++i) {
        w[i] = load_be_u32(block + (i * 4u));
    }
    sha256_compress_words(ctx, w);
}

// Update SHA-256 state from thread-local memory (`thread` address space).
// We use this for small local arrays like ipad/opad and inner digest.
static inline void sha256_update_thread(thread Sha256Ctx& ctx, thread const uint8_t* data, uint32_t len) {
    if (len == 0u) {
        return;
    }
    ctx.total_len += uint64_t(len);             // Count total bytes for final bit-length encoding.

    uint32_t offset = 0u;
    if (ctx.block_len != 0u) {                  // Fill pending partial block first.
        const uint32_t need = 64u - ctx.block_len;
        const uint32_t take = (len < need) ? len : need;
        for (uint32_t i = 0; i < take; ++i) {
            ctx.block[ctx.block_len + i] = data[i];
        }
        ctx.block_len += take;
        offset += take;
        if (ctx.block_len == 64u) {
            sha256_compress(ctx, ctx.block);
            ctx.block_len = 0u;
        }
    }

    while ((len - offset) >= 64u) {             // Compress whole source blocks directly.
        sha256_compress(ctx, data + offset);
        offset += 64u;
    }

    for (; offset < len; ++offset) {            // Buffer only trailing bytes.
        ctx.block[ctx.block_len++] = data[offset];
    }
}

// Update SHA-256 state from device memory (`device` address space).
// We use this for thread-specific GPU buffers uploaded by the host (candidate secrets).
static inline void sha256_update_device(thread Sha256Ctx& ctx, device const uint8_t* data, uint32_t len) {
    if (len == 0u) {
        return;
    }
    ctx.total_len += uint64_t(len);             // Track total length in bytes once per call.

    uint32_t offset = 0u;
    if (ctx.block_len != 0u) {                  // Fill pending partial block first.
        const uint32_t need = 64u - ctx.block_len;
        const uint32_t take = (len < need) ? len : need;
        for (uint32_t i = 0; i < take; ++i) {
            ctx.block[ctx.block_len + i] = data[i];
        }
        ctx.block_len += take;
        offset += take;
        if (ctx.block_len == 64u) {
            sha256_compress(ctx, ctx.block);
            ctx.block_len = 0u;
        }
    }

    while ((len - offset) >= 64u) {             // Compress whole device blocks directly.
        sha256_compress_device(ctx, data + offset);
        offset += 64u;
    }

    for (; offset < len; ++offset) {            // Buffer only the tail bytes.
        ctx.block[ctx.block_len++] = data[offset];
    }
}

// Update SHA-256 state from constant memory (`constant` address space).
// Shared JWT message bytes are read by all threads, so `constant` helps on Apple GPUs.
static inline void sha256_update_constant(thread Sha256Ctx& ctx, constant const uint8_t* data, uint32_t len) {
    if (len == 0u) {
        return;
    }
    ctx.total_len += uint64_t(len);

    uint32_t offset = 0u;
    if (ctx.block_len != 0u) {
        const uint32_t need = 64u - ctx.block_len;
        const uint32_t take = (len < need) ? len : need;
        for (uint32_t i = 0; i < take; ++i) {
            ctx.block[ctx.block_len + i] = data[i];
        }
        ctx.block_len += take;
        offset += take;
        if (ctx.block_len == 64u) {
            sha256_compress(ctx, ctx.block);
            ctx.block_len = 0u;
        }
    }

    while ((len - offset) >= 64u) {
        sha256_compress_constant(ctx, data + offset);
        offset += 64u;
    }

    for (; offset < len; ++offset) {
        ctx.block[ctx.block_len++] = data[offset];
    }
}

// Finalize SHA-256: add padding + length, compress final block(s), and write 32-byte digest.
static inline void sha256_final(thread Sha256Ctx& ctx, thread uint8_t out[32]) {
    const uint64_t bit_len = ctx.total_len * 8ull;

    ctx.block[ctx.block_len++] = 0x80u;
    if (ctx.block_len > 56u) {
        while (ctx.block_len < 64u) ctx.block[ctx.block_len++] = 0u;
        sha256_compress(ctx, ctx.block);
        ctx.block_len = 0;
    }
    while (ctx.block_len < 56u) ctx.block[ctx.block_len++] = 0u;
    for (uint i = 0; i < 8; ++i)
        ctx.block[56u + i] = uint8_t((bit_len >> (56u - 8u * i)) & 0xffull);
    sha256_compress(ctx, ctx.block);
    ctx.block_len = 0;

    for (uint i = 0; i < 8; ++i)
        store_be_u32(out + (i * 4u), ctx.state[i]);
}

// Finalize SHA-256 without serializing output to bytes.
// After this call, ctx.state[0..7] contains the digest as big-endian uint32_t words.
// This avoids the store_be_u32 output loop when the digest will be consumed as words.
static inline void sha256_finalize(thread Sha256Ctx& ctx) {
    const uint64_t bit_len = ctx.total_len * 8ull;
    ctx.block[ctx.block_len++] = 0x80u;
    if (ctx.block_len > 56u) {
        while (ctx.block_len < 64u) ctx.block[ctx.block_len++] = 0u;
        sha256_compress(ctx, ctx.block);
        ctx.block_len = 0u;
    }
    while (ctx.block_len < 56u) ctx.block[ctx.block_len++] = 0u;
    for (uint i = 0; i < 8; ++i)
        ctx.block[56u + i] = uint8_t((bit_len >> (56u - 8u * i)) & 0xffull);
    sha256_compress(ctx, ctx.block);
    ctx.block_len = 0u;
}

// Load an HMAC key from device memory into 16 big-endian uint32_t words (zero-padded).
// Eliminates the byte-level key_block[64] array in favor of word-level operations.
static inline void load_key_words(device const uint8_t* key, uint32_t key_len, thread uint32_t kw[16]) {
    #pragma unroll
    for (uint i = 0; i < 16; ++i) kw[i] = 0u;

    const uint32_t full_words = key_len >> 2;
    for (uint32_t i = 0; i < full_words; ++i)
        kw[i] = load_be_u32(key + i * 4u);

    const uint32_t remaining = key_len & 3u;
    if (remaining > 0u) {
        uint32_t val = 0u;
        device const uint8_t* tail = key + (full_words * 4u);
        for (uint32_t i = 0; i < remaining; ++i)
            val |= uint32_t(tail[i]) << (24u - 8u * i);
        kw[full_words] = val;
    }
}

// Core word-level HMAC-SHA256 computation from pre-loaded key words.
// Eliminates key_block[64], ipad[64], opad[64], inner_digest[32], and digest[32]
// byte arrays. Works entirely with uint32_t words and compares ctx.state directly.
// Returns true if the HMAC matches `target`.
static inline bool hmac_sha256_from_key_words(
    thread const uint32_t kw[16],
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint8_t target[32]
) {
    // Inner hash: compress (kw XOR 0x36363636) as the ipad block, then message.
    Sha256Ctx inner;
    sha256_init(inner);
    {
        uint32_t w[64];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x36363636u;
        sha256_compress_words(inner, w);
    }
    inner.total_len = 64ull;
    inner.block_len = 0u;
    sha256_update_constant(inner, message, message_len);
    sha256_finalize(inner);
    // inner.state[0..7] now holds the inner digest as words.

    // Outer hash: compress (kw XOR 0x5c5c5c5c) as the opad block,
    // then finalize with inner digest as a 32-byte tail.
    Sha256Ctx outer;
    sha256_init(outer);
    {
        uint32_t w[64];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x5c5c5c5cu;
        sha256_compress_words(outer, w);
    }
    outer.total_len = 64ull;
    outer.block_len = 0u;

    // Build the final outer block directly from inner digest words.
    // Block layout: [inner_digest(32 bytes)][0x80][zeros][bit_length(8 bytes)]
    {
        uint32_t w[64];
        for (uint i = 0; i < 8; ++i) w[i] = inner.state[i];
        w[8] = 0x80000000u;
        w[9] = 0u; w[10] = 0u; w[11] = 0u; w[12] = 0u; w[13] = 0u;
        const uint64_t bit_len = (64ull + 32ull) * 8ull;
        w[14] = uint32_t((bit_len >> 32u) & 0xffffffffull);
        w[15] = uint32_t(bit_len & 0xffffffffull);
        sha256_compress_words(outer, w);
    }

    // Compare outer hash state words directly against the target signature.
    // No digest serialization needed — saves 8x store_be_u32 + 8x load_be_u32.
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        if (outer.state[i] != load_be_u32(target + i * 4u)) return false;
    }
    return true;
}

// HMAC-SHA256 that handles any key length and returns match result directly.
// For keys > 64 bytes, hashes the key first (RFC 2104), then uses word-level path.
static inline bool hmac_sha256_matches(
    device const uint8_t* key,
    uint32_t key_len,
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint8_t target[32]
) {
    uint32_t kw[16];

    if (key_len > 64u) {
        // Long keys: hash first, then use the 32-byte hash as the key.
        #pragma unroll
        for (uint i = 0; i < 16; ++i) kw[i] = 0u;
        Sha256Ctx key_ctx;
        sha256_init(key_ctx);
        sha256_update_device(key_ctx, key, key_len);
        sha256_finalize(key_ctx);
        #pragma unroll
        for (uint i = 0; i < 8; ++i) kw[i] = key_ctx.state[i];
    } else {
        load_key_words(key, key_len, kw);
    }

    return hmac_sha256_from_key_words(kw, message, message_len, target);
}

// Short-key specialization: keys guaranteed <= 64 bytes, no hash-long-key branch.
static inline bool hmac_sha256_short_key_matches(
    device const uint8_t* key,
    uint32_t key_len,
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint8_t target[32]
) {
    uint32_t kw[16];
    load_key_words(key, key_len, kw);
    return hmac_sha256_from_key_words(kw, message, message_len, target);
}

// Main compute kernel: one GPU thread tests exactly one candidate secret.
kernel void hs256_wordlist(
    constant Hs256BruteForceParams& params [[buffer(0)]],
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

    if (hmac_sha256_matches(candidate_secret, candidate_length,
                            message_bytes, params.message_length,
                            params.target_signature)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}

// Short-key specialization. Host dispatches this when every key in the batch is <= 64 bytes.
kernel void hs256_wordlist_short_keys(
    constant Hs256BruteForceParams& params [[buffer(0)]],
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

    if (hmac_sha256_short_key_matches(candidate_secret, candidate_length,
                                      message_bytes, params.message_length,
                                      params.target_signature)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}
