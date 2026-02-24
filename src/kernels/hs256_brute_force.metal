// Import Metal's standard library (types like `uint32_t`, kernel attributes, atomics, etc.).
#include <metal_stdlib>
// Pull `metal::` names into the local namespace so code can use `kernel`, `device`, `thread`, etc. directly.
using namespace metal;

// Host-to-kernel parameter block. Rust uploads this as `buffer(0)` with the exact same field order/layout.
struct Hs256BruteForceParams {
    uint8_t target_signature[32];      // Expected HS256 HMAC digest (32 bytes) we compare against.
    uint32_t message_length;           // Actual byte length of `message_bytes` (the JWT `header.payload` string).
    uint32_t candidate_count;          // Number of candidate secrets in this GPU batch.
    uint32_t candidate_index_base;     // Absolute wordlist index of candidate #0 in this batch (for deterministic reporting).
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
static inline void sha256_compress(thread Sha256Ctx& ctx, thread const uint8_t* block) {
    uint32_t w[64]; // Message schedule array for this block.

    for (uint i = 0; i < 16; ++i) { // First 16 words come directly from the 64-byte input block.
        w[i] = load_be_u32(block + (i * 4u)); // Convert each 4-byte chunk to a 32-bit word.
    }
    for (uint i = 16; i < 64; ++i) { // Remaining words are expanded from previous schedule words.
        w[i] = ssig1(w[i - 2]) + w[i - 7] + ssig0(w[i - 15]) + w[i - 16]; // Standard SHA-256 schedule formula.
    }

    uint32_t a = ctx.state[0]; // Working variable a starts from current hash state.
    uint32_t b = ctx.state[1]; // Working variable b.
    uint32_t c = ctx.state[2]; // Working variable c.
    uint32_t d = ctx.state[3]; // Working variable d.
    uint32_t e = ctx.state[4]; // Working variable e.
    uint32_t f = ctx.state[5]; // Working variable f.
    uint32_t g = ctx.state[6]; // Working variable g.
    uint32_t h = ctx.state[7]; // Working variable h.

    for (uint i = 0; i < 64; ++i) { // 64 SHA-256 rounds.
        uint32_t t1 = h + bsig1(e) + ch32(e, f, g) + SHA256_K[i] + w[i]; // Main round sum #1.
        uint32_t t2 = bsig0(a) + maj32(a, b, c);                          // Main round sum #2.
        h = g;               // Shift working registers down.
        g = f;               // Shift working registers down.
        f = e;               // Shift working registers down.
        e = d + t1;          // New e includes previous d + t1.
        d = c;               // Shift working registers down.
        c = b;               // Shift working registers down.
        b = a;               // Shift working registers down.
        a = t1 + t2;         // New a combines both round sums.
    }

    ctx.state[0] += a; // Feed-forward: add working variables back into the hash state.
    ctx.state[1] += b; // Feed-forward for state word 1.
    ctx.state[2] += c; // Feed-forward for state word 2.
    ctx.state[3] += d; // Feed-forward for state word 3.
    ctx.state[4] += e; // Feed-forward for state word 4.
    ctx.state[5] += f; // Feed-forward for state word 5.
    ctx.state[6] += g; // Feed-forward for state word 6.
    ctx.state[7] += h; // Feed-forward for state word 7.
}

// Update SHA-256 state from thread-local memory (`thread` address space).
// We use this for small local arrays like ipad/opad and inner digest.
static inline void sha256_update_thread(thread Sha256Ctx& ctx, thread const uint8_t* data, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) {        // Consume one byte at a time (simple, correct implementation).
        ctx.block[ctx.block_len++] = data[i];   // Append byte into current 64-byte block buffer.
        ctx.total_len += 1;                     // Count total bytes for final bit-length encoding.
        if (ctx.block_len == 64u) {             // If block buffer is full,
            sha256_compress(ctx, ctx.block);    // compress it now,
            ctx.block_len = 0;                  // then reset buffer length for the next block.
        }
    }
}

// Update SHA-256 state from device memory (`device` address space).
// We use this for GPU buffers uploaded by the host (candidate secrets, JWT message bytes).
static inline void sha256_update_device(thread Sha256Ctx& ctx, device const uint8_t* data, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) {        // Consume one byte at a time from device memory.
        ctx.block[ctx.block_len++] = data[i];   // Append to the local 64-byte block buffer.
        ctx.total_len += 1;                     // Track total length in bytes.
        if (ctx.block_len == 64u) {             // Compress whenever one full block is available.
            sha256_compress(ctx, ctx.block);    // Run the SHA-256 compression function.
            ctx.block_len = 0;                  // Start filling a fresh block.
        }
    }
}

// Finalize SHA-256: add padding + length, compress final block(s), and write 32-byte digest.
static inline void sha256_final(thread Sha256Ctx& ctx, thread uint8_t out[32]) {
    const uint64_t bit_len = ctx.total_len * 8ull; // SHA-256 appends message length in bits, not bytes.

    ctx.block[ctx.block_len++] = 0x80u;            // Append the required single '1' bit (10000000 byte).
    if (ctx.block_len > 56u) {                     // If not enough room for 8-byte length field,
        while (ctx.block_len < 64u) {              // pad this block with zeros,
            ctx.block[ctx.block_len++] = 0u;       // zero-fill remaining bytes.
        }
        sha256_compress(ctx, ctx.block);           // Compress padded block.
        ctx.block_len = 0;                         // Start a new final block.
    }

    while (ctx.block_len < 56u) {                  // Zero-pad until byte 56,
        ctx.block[ctx.block_len++] = 0u;           // leaving bytes 56..63 for the 64-bit length.
    }

    for (uint i = 0; i < 8; ++i) {                 // Write `bit_len` in big-endian order.
        ctx.block[56u + i] = uint8_t((bit_len >> (56u - 8u * i)) & 0xffull);
    }
    sha256_compress(ctx, ctx.block);               // Compress the final block.
    ctx.block_len = 0;                             // Reset buffer length (not strictly needed after finalize).

    for (uint i = 0; i < 8; ++i) {                 // Convert 8 state words into 32 output bytes.
        store_be_u32(out + (i * 4u), ctx.state[i]);
    }
}

// Compute HMAC-SHA256(key, message) into `out`.
// `key` and `message` both live in device memory because they come from host-uploaded buffers.
static inline void hmac_sha256(
    device const uint8_t* key,                  // Candidate secret bytes (wordlist entry) for this thread.
    uint32_t key_len,                           // Candidate secret length in bytes.
    device const uint8_t* message,              // Shared JWT signing input bytes (`header.payload`).
    uint32_t message_len,                       // Length of the JWT signing input.
    thread uint8_t out[32]                      // Output digest buffer in thread-local memory.
) {
    uint8_t key_block[64];                      // HMAC uses a 64-byte block for SHA-256 keys.
    for (uint i = 0; i < 64; ++i) {             // Start with all zeros (key padding if key is short).
        key_block[i] = 0u;
    }

    if (key_len > 64u) {                        // Long HMAC keys must be hashed first (RFC 2104 behavior).
        Sha256Ctx key_ctx;                      // Temporary SHA-256 context to hash the long key.
        uint8_t hashed_key[32];                 // Hashed key becomes 32 bytes.
        sha256_init(key_ctx);                   // Initialize SHA-256 context.
        sha256_update_device(key_ctx, key, key_len); // Feed original key bytes.
        sha256_final(key_ctx, hashed_key);      // Finalize into `hashed_key`.
        for (uint i = 0; i < 32; ++i) {         // Copy hashed key into the 64-byte HMAC key block.
            key_block[i] = hashed_key[i];       // Remaining bytes stay zero-padded.
        }
    } else {
        for (uint32_t i = 0; i < key_len; ++i) { // Short keys are copied directly into the key block.
            key_block[i] = key[i];               // Remaining bytes stay zero (right-padding with zeros).
        }
    }

    uint8_t ipad[64];                            // Inner padding block (key xor 0x36).
    uint8_t opad[64];                            // Outer padding block (key xor 0x5c).
    for (uint i = 0; i < 64; ++i) {              // Build both pads byte-by-byte.
        ipad[i] = key_block[i] ^ 0x36u;          // HMAC inner pad constant.
        opad[i] = key_block[i] ^ 0x5cu;          // HMAC outer pad constant.
    }

    Sha256Ctx inner;                             // SHA-256 context for the inner hash.
    uint8_t inner_digest[32];                    // Output of SHA256(ipad || message).
    sha256_init(inner);                          // Initialize inner SHA-256.
    sha256_update_thread(inner, ipad, 64u);      // Feed 64-byte ipad from thread memory.
    sha256_update_device(inner, message, message_len); // Feed shared JWT signing input from device memory.
    sha256_final(inner, inner_digest);           // Finalize inner hash.

    Sha256Ctx outer;                             // SHA-256 context for the outer hash.
    sha256_init(outer);                          // Initialize outer SHA-256.
    sha256_update_thread(outer, opad, 64u);      // Feed 64-byte opad.
    sha256_update_thread(outer, inner_digest, 32u); // Feed inner digest bytes.
    sha256_final(outer, out);                    // Final HMAC digest = SHA256(opad || inner_digest).
}

// Compare two 32-byte digests byte-by-byte.
static inline bool digest_matches(thread const uint8_t digest[32], constant const uint8_t target[32]) {
    for (uint i = 0; i < 32; ++i) {              // Check every digest byte.
        if (digest[i] != target[i]) {            // Mismatch means this candidate secret is wrong.
            return false;                        // Exit early on first mismatch.
        }
    }
    return true;                                 // All bytes matched.
}

// Main compute kernel: one GPU thread tests exactly one candidate secret from the wordlist batch.
kernel void hs256_brute_force(
    constant Hs256BruteForceParams& params [[buffer(0)]], // Small shared params uploaded by Rust.
    device const uint8_t* message_bytes [[buffer(1)]],    // JWT signing input bytes (`header.payload`).
    device const uint8_t* word_bytes [[buffer(2)]],       // Packed wordlist bytes for this batch.
    device const uint32_t* word_offsets [[buffer(3)]],    // Start offset of each candidate into `word_bytes`.
    device const uint16_t* word_lengths [[buffer(4)]],    // Length of each candidate secret.
    device atomic_uint* result_index [[buffer(5)]],       // Shared output slot (absolute wordlist index, or sentinel).
    uint gid [[thread_position_in_grid]]                  // Global thread id = candidate index within this batch.
) {
    if (gid >= params.candidate_count) {                  // Guard against over-dispatch from rounded grid sizes.
        return;                                           // Thread has no real candidate assigned.
    }

    const uint32_t candidate_offset = word_offsets[gid];  // Read this candidate's start offset.
    const uint32_t candidate_length = uint32_t(word_lengths[gid]); // Read and widen this candidate's length.
    device const uint8_t* candidate_secret = word_bytes + candidate_offset; // Pointer to candidate bytes.

    uint8_t digest[32];                                   // Thread-local output buffer for computed HMAC digest.
    hmac_sha256(candidate_secret,                         // Candidate secret acts as the HS256 key.
                candidate_length,                         // Candidate key length.
                message_bytes,                            // Shared JWT signing input bytes.
                params.message_length,                    // Shared JWT signing input length.
                digest);                                  // Receive computed HMAC-SHA256 digest here.

    if (digest_matches(digest, params.target_signature)) { // If this candidate produced the target signature,
        const uint32_t absolute_index = params.candidate_index_base + gid; // convert batch-local index to absolute line index.
        atomic_fetch_min_explicit(result_index, absolute_index, memory_order_relaxed); // Publish earliest match deterministically.
    }
}
