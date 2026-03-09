// ===========================================================================
// Metal compute kernel for HMAC-SHA256 JWT cracking (wordlist mode).
//
// LEARNING OVERVIEW -- read this first if you're new to GPU programming.
//
// What this file does:
//   Each GPU thread receives one candidate secret from a wordlist and computes
//   HMAC-SHA256(secret, JWT_signing_input).  If the result matches the target
//   signature, the thread writes its index to an atomic result slot.
//
// Key concepts for understanding this kernel:
//
//   1. KERNEL DISPATCH MODEL
//      Metal dispatches a "grid" of threads.  Each thread gets a unique
//      `gid` (global thread ID) via `[[thread_position_in_grid]]`.  The host
//      sets grid size = number of candidates and threadgroup size = a tunable
//      width (e.g. 256).  Metal's scheduler maps threadgroups onto the GPU's
//      SIMD units (Apple calls them "execution width" groups of 32 threads).
//
//   2. SHA-256 COMPRESSION
//      SHA-256 processes data in 512-bit (64-byte) blocks.  Each block goes
//      through 64 "rounds" of mixing using bitwise rotations, XORs, additions
//      mod 2^32, and a set of round constants (SHA256_K[64]).  The internal
//      state is eight 32-bit words (a..h) that accumulate the hash.
//
//   3. HMAC CONSTRUCTION (RFC 2104)
//      HMAC wraps a hash function with a key:
//        HMAC(K, M) = H((K ^ opad) || H((K ^ ipad) || M))
//      where ipad = 0x36 repeated, opad = 0x5c repeated, and K is zero-padded
//      to one block (64 bytes for SHA-256).  This requires two full SHA-256
//      computations: an "inner hash" and an "outer hash".
//
//   4. ADDRESS SPACES
//      Metal has distinct pointer types: `thread` (per-thread registers/stack),
//      `device` (GPU VRAM, read-write), `constant` (read-only, cached, shared
//      by all threads).  The JWT message uses `constant` because every thread
//      reads the same data; candidate secrets use `device` because each thread
//      reads a different slice.
//
//   5. ATOMIC RESULT REPORTING
//      When a thread finds a match, it uses `atomic_fetch_min_explicit` to
//      write its `gid` to the result slot.  The `min` ensures that if
//      multiple threads match (shouldn't happen with correct JWTs), the
//      lowest-indexed match wins deterministically.
// ===========================================================================

// Import Metal's standard library (types like `uint32_t`, kernel attributes, atomics, etc.).
#include <metal_stdlib>
// Pull `metal::` names into the local namespace so code can use `kernel`, `device`, `thread`, etc. directly.
using namespace metal;

// Host-to-kernel parameter block. Rust uploads this as `buffer(0)` with the exact same field order/layout.
// The struct must match the Rust-side `#[repr(C)]` struct field-for-field, byte-for-byte.
struct Hs256BruteForceParams {
    // Optimization: target stored as uint32_t[8] (precomputed on host) instead of
    // uint8_t[32].  Eliminates per-thread load_be_u32 calls in the comparison loop
    // — the final check becomes a straight word-level `ostate[i] != target[i]`.
    uint32_t target_signature[8];
    uint32_t message_length;           // Actual byte length of `message_bytes` (the JWT `header.payload` string).
    uint32_t candidate_count;          // Number of candidate secrets in this GPU batch.
};

// Minimal SHA-256 streaming context used per GPU thread.
//
// Learning note: why is this so large?
// Each GPU thread needs its own copy of this struct (76+ bytes) because
// threads execute independently.  The "flat" HMAC path below avoids this
// struct entirely on the hot path, using bare uint32_t arrays instead,
// which dramatically reduces register pressure and improves GPU occupancy
// (= more threads can be in flight simultaneously).
struct Sha256Ctx {
    uint32_t state[8];                 // Current SHA-256 state words (a..h / hash state).
    uint8_t block[64];                 // Current 512-bit block buffer being filled before compression.
    uint32_t block_len;                // Number of bytes currently stored in `block`.
    uint64_t total_len;                // Total message length processed so far, in bytes (needed for final padding length).
};

// SHA-256 round constants from the specification (FIPS 180-4, section 4.2.2).
// These are the first 32 bits of the fractional parts of the cube roots of
// the first 64 prime numbers.  They are placed in `constant` address space
// so every thread shares one cached copy (saves VRAM bandwidth vs. `device`).
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
// These constants are the first 32 bits of the fractional parts of the square
// roots of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19).
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

// ---------------------------------------------------------------------------
// Optimization: rolling 16-word message schedule.
//
// Standard SHA-256 pre-expands all 64 message words into w[64] (256 bytes)
// before compression.  The rolling schedule keeps only w[16] (64 bytes) and
// updates w[i & 15] in-place during rounds 16-63, saving ~192 bytes of
// register pressure per sha256_compress call.  On Apple GPUs this directly
// improves occupancy (more threads in flight per SIMD group).
//
// Rounds 0-15:  use the caller-loaded w[i] directly (SHA256_ROUND_R).
// Rounds 16-63: recurrence w[i&15] = σ1(w[i-2]) + w[i-7] + σ0(w[i-15])
//               + w[i-16], then round (SHA256_EXPAND_ROUND).
// ---------------------------------------------------------------------------

// SHA-256 round using rolling w[i & 15] indexing.
#define SHA256_ROUND_R(i, a, b, c, d, e, f, g, h) do { \
    uint32_t t1_ = (h) + bsig1((e)) + ch32((e), (f), (g)) + SHA256_K[(i)] + w[(i) & 15]; \
    uint32_t t2_ = bsig0((a)) + maj32((a), (b), (c)); \
    (d) += t1_; \
    (h) = t1_ + t2_; \
} while (0)

// Rolling expand + round: update w[i&15] in-place from the recurrence, then compress.
// w[(i-16)&15] == w[i&15] still holds the value from 16 rounds ago.
#define SHA256_EXPAND_ROUND(i, a, b, c, d, e, f, g, h) do { \
    w[(i) & 15] = ssig1(w[((i) - 2) & 15]) + w[((i) - 7) & 15] + ssig0(w[((i) - 15) & 15]) + w[(i) & 15]; \
    SHA256_ROUND_R(i, a, b, c, d, e, f, g, h); \
} while (0)

// 8 consecutive rounds for the first 16 rounds (no expansion needed).
#define SHA256_ROUND_8_INIT(base, a, b, c, d, e, f, g, h) \
    SHA256_ROUND_R((base) + 0, a, b, c, d, e, f, g, h); \
    SHA256_ROUND_R((base) + 1, h, a, b, c, d, e, f, g); \
    SHA256_ROUND_R((base) + 2, g, h, a, b, c, d, e, f); \
    SHA256_ROUND_R((base) + 3, f, g, h, a, b, c, d, e); \
    SHA256_ROUND_R((base) + 4, e, f, g, h, a, b, c, d); \
    SHA256_ROUND_R((base) + 5, d, e, f, g, h, a, b, c); \
    SHA256_ROUND_R((base) + 6, c, d, e, f, g, h, a, b); \
    SHA256_ROUND_R((base) + 7, b, c, d, e, f, g, h, a)

// 8 consecutive rounds with rolling expansion (rounds 16-63).
#define SHA256_ROUND_8_ROLLING(base, a, b, c, d, e, f, g, h) \
    SHA256_EXPAND_ROUND((base) + 0, a, b, c, d, e, f, g, h); \
    SHA256_EXPAND_ROUND((base) + 1, h, a, b, c, d, e, f, g); \
    SHA256_EXPAND_ROUND((base) + 2, g, h, a, b, c, d, e, f); \
    SHA256_EXPAND_ROUND((base) + 3, f, g, h, a, b, c, d, e); \
    SHA256_EXPAND_ROUND((base) + 4, e, f, g, h, a, b, c, d); \
    SHA256_EXPAND_ROUND((base) + 5, d, e, f, g, h, a, b, c); \
    SHA256_EXPAND_ROUND((base) + 6, c, d, e, f, g, h, a, b); \
    SHA256_EXPAND_ROUND((base) + 7, b, c, d, e, f, g, h, a)

// Compress one 512-bit block using a rolling 16-word message schedule.
// Takes bare state[8] and w[16] (caller loads the first 16 message words).
// NOTE: w[16] is modified in-place by the rolling expansion (rounds 16-63).
//
// Learning note: SHA-256 compression step by step
//
// 1. Copy current state into working variables a..h.
// 2. Rounds 0-15:  use the 16 message words directly with the round function.
//    Each round updates one working variable using: rotations of e (bsig1),
//    a choice function on e,f,g, the round constant K[i], and the message word w[i].
// 3. Rounds 16-63: first expand w[i&15] using a recurrence relation:
//      w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16]
//    then apply the same round function.
// 4. Add the working variables back to the state (this is the "Davies-Meyer" pattern).
//
// After processing all blocks, state[0..7] holds the final hash digest.
static inline void sha256_compress_rolling(thread uint32_t state[8], thread uint32_t w[16]) {
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    // Rounds 0-15: use w[i] directly (no expansion needed).
    SHA256_ROUND_8_INIT(0,  a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_INIT(8,  a, b, c, d, e, f, g, h);
    // Rounds 16-63: expand w[i&15] in-place before each round.
    SHA256_ROUND_8_ROLLING(16, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(24, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(32, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(40, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(48, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(56, a, b, c, d, e, f, g, h);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

static inline void sha256_compress(thread Sha256Ctx& ctx, thread const uint8_t* block) {
    uint32_t w[16];
    for (uint i = 0; i < 16; ++i) {
        w[i] = load_be_u32(block + (i * 4u));
    }
    sha256_compress_rolling(ctx.state, w);
}

static inline void sha256_compress_device(thread Sha256Ctx& ctx, device const uint8_t* block) {
    uint32_t w[16];
    for (uint i = 0; i < 16; ++i) {
        w[i] = load_be_u32(block + (i * 4u));
    }
    sha256_compress_rolling(ctx.state, w);
}

static inline void sha256_compress_constant(thread Sha256Ctx& ctx, constant const uint8_t* block) {
    uint32_t w[16];
    for (uint i = 0; i < 16; ++i) {
        w[i] = load_be_u32(block + (i * 4u));
    }
    sha256_compress_rolling(ctx.state, w);
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
//
// Learning note: SHA-256 padding rules
// 1. Append a single 0x80 byte (a 1-bit followed by zeros).
// 2. Pad with zero bytes until the block is 56 bytes long (leaving room for 8 bytes of length).
//    If appending 0x80 already pushed past byte 56, finish this block and start a new one.
// 3. Append the total message length in bits as a 64-bit big-endian integer in the last 8 bytes.
// 4. Compress the final block.  state[0..7] is the digest.
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

// ---------------------------------------------------------------------------
// Optimization: flat specialized HMAC (no Sha256Ctx on the hot path).
//
// Learning note: why two HMAC implementations?
//
// The generic streaming Sha256Ctx carries 76 bytes of per-thread state
// (block[64] + block_len + total_len) that the HMAC hot path never needs:
// ipad/opad blocks are always exactly 64 bytes, message length is known, and
// finalization layout is deterministic.  This flat version uses bare
// uint32_t istate[8] / ostate[8] (32 bytes each) and constructs each 512-bit
// block in a local w[16] directly from constant memory with manual padding,
// eliminating ~152 bytes of register pressure across the inner+outer hashes.
//
// On GPUs, register pressure is the primary bottleneck for occupancy.  Fewer
// registers per thread = more threads can execute simultaneously = higher
// throughput.  This optimization alone can yield a 1.5-2x speedup.
//
// Sha256Ctx + the streaming update/finalize API are kept for the long-key
// fallback path (key_len > 64 -> hash-then-HMAC per RFC 2104).
//
// HMAC-SHA256 conceptually:
//   inner = SHA256( (key XOR ipad) || message )
//   outer = SHA256( (key XOR opad) || inner_digest )
// where ipad = 0x36 repeated to block size, opad = 0x5c repeated to block size.
// ---------------------------------------------------------------------------
static inline bool hmac_sha256_from_key_words(
    thread const uint32_t kw[16],
    constant const uint8_t* message,
    uint32_t message_len,
    constant const uint32_t target[8]
) {
    // --- Inner hash (flat, no Sha256Ctx) ---
    uint32_t istate[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    // Compress ipad block: kw XOR 0x36363636.
    {
        uint32_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x36363636u;
        sha256_compress_rolling(istate, w);
    }

    // Process full 64-byte message blocks directly from constant memory.
    uint32_t offset = 0u;
    while ((message_len - offset) >= 64u) {
        uint32_t w[16];
        for (uint i = 0; i < 16; ++i) {
            w[i] = load_be_u32(message + offset + i * 4u);
        }
        sha256_compress_rolling(istate, w);
        offset += 64u;
    }

    // Finalize inner hash: remaining bytes + 0x80 padding + bit length.
    const uint32_t rem = message_len - offset;
    const uint64_t inner_bit_len = (64ull + uint64_t(message_len)) * 8ull;
    {
        uint32_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = 0u;

        // Load remaining full words from message.
        const uint32_t full_words = rem >> 2;
        for (uint32_t i = 0; i < full_words; ++i) {
            w[i] = load_be_u32(message + offset + i * 4u);
        }

        // Load partial trailing bytes and append 0x80 padding byte.
        const uint32_t tail = rem & 3u;
        uint32_t pad_word = 0u;
        constant const uint8_t* tail_ptr = message + offset + (full_words * 4u);
        for (uint32_t i = 0; i < tail; ++i) {
            pad_word |= uint32_t(tail_ptr[i]) << (24u - 8u * i);
        }
        pad_word |= 0x80u << (24u - 8u * tail);
        w[full_words] = pad_word;

        if (rem > 55u) {
            // Two-block finalization: data+0x80 don't fit with the length field.
            sha256_compress_rolling(istate, w);
            #pragma unroll
            for (uint i = 0; i < 16; ++i) w[i] = 0u;
        }
        w[14] = uint32_t((inner_bit_len >> 32u) & 0xffffffffull);
        w[15] = uint32_t(inner_bit_len & 0xffffffffull);
        sha256_compress_rolling(istate, w);
    }
    // istate[0..7] now holds the inner digest as words.

    // --- Outer hash (flat, no Sha256Ctx) ---
    uint32_t ostate[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    // Compress opad block: kw XOR 0x5c5c5c5c.
    {
        uint32_t w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x5c5c5c5cu;
        sha256_compress_rolling(ostate, w);
    }

    // Compress final outer block: inner digest + padding.
    // Outer message = 64 (opad) + 32 (inner digest) = 96 bytes total.
    {
        uint32_t w[16];
        for (uint i = 0; i < 8; ++i) w[i] = istate[i];
        w[8] = 0x80000000u;
        w[9] = 0u; w[10] = 0u; w[11] = 0u; w[12] = 0u; w[13] = 0u;
        const uint64_t outer_bit_len = (64ull + 32ull) * 8ull;
        w[14] = uint32_t((outer_bit_len >> 32u) & 0xffffffffull);
        w[15] = uint32_t(outer_bit_len & 0xffffffffull);
        sha256_compress_rolling(ostate, w);
    }

    // Compare against precomputed target words.
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        if (ostate[i] != target[i]) return false;
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
    constant const uint32_t target[8]
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
    constant const uint32_t target[8]
) {
    uint32_t kw[16];
    load_key_words(key, key_len, kw);
    return hmac_sha256_from_key_words(kw, message, message_len, target);
}

// ===========================================================================
// KERNEL ENTRY POINTS
//
// Learning note: Metal kernel dispatch model
//
// The host (Rust code) calls `dispatch_threads(grid_size, threadgroup_size)`.
// Metal creates `grid_size` threads total, organized into groups of
// `threadgroup_size`.  Each thread receives `gid` = its global index.
//
// For this kernel:
//   grid_size      = number of candidate secrets in the batch
//   threadgroup_size = tunable (e.g. 256), chosen by autotune
//   gid            = which candidate this thread should test
//
// Buffer bindings (set by the Rust encoder):
//   buffer(0) = params     (target signature, message length, candidate count)
//   buffer(1) = message    (JWT header.payload bytes, shared by all threads)
//   buffer(2) = word_bytes (concatenated candidate secret bytes)
//   buffer(3) = word_offsets (byte offset of each candidate in word_bytes)
//   buffer(4) = word_lengths (byte length of each candidate)
//   buffer(5) = result_index (atomic: lowest matching gid, or 0xFFFFFFFF)
// ===========================================================================

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
// Learning note: by guaranteeing key_len <= 64, this kernel eliminates the
// hash-the-long-key branch, saving one conditional and one full SHA-256
// computation per thread.  The host checks max_word_len at batch level and
// selects this kernel when safe.  This is a common GPU optimization: branch
// elimination via kernel specialization.
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

// ===========================================================================
// Markov chain candidate generation + HMAC-SHA256 (fused kernel).
//
// Each GPU thread decodes a global keyspace index into per-position rank
// selections via modular arithmetic, walks an order-1 Markov chain using a
// pre-trained lookup table, and immediately computes HMAC-SHA256 on the
// generated candidate — no intermediate candidate buffer or memory traffic.
//
// The Markov table is in `constant` address space: all threads read the same
// table, and Metal's constant cache broadcasts reads efficiently.
// ===========================================================================

struct Hs256MarkovParams {
    uint32_t target_signature[8];
    uint32_t message_length;
    uint32_t candidate_count;
    uint32_t pw_length;        // password length for this batch (all candidates same length)
    uint32_t threshold;        // T: number of ranked successors per (pos, prev_char) context
    uint64_t offset;           // starting index in the keyspace for this batch
};

// Thread-address-space variant of load_key_words.
// Identical logic to load_key_words but reads from `thread` instead of `device`.
static inline void load_key_words_thread(thread const uint8_t* key, uint32_t key_len, thread uint32_t kw[16]) {
    #pragma unroll
    for (uint i = 0; i < 16; ++i) kw[i] = 0u;

    const uint32_t full_words = key_len >> 2;
    for (uint32_t i = 0; i < full_words; ++i)
        kw[i] = load_be_u32(key + i * 4u);

    const uint32_t remaining = key_len & 3u;
    if (remaining > 0u) {
        uint32_t val = 0u;
        thread const uint8_t* tail = key + (full_words * 4u);
        for (uint32_t i = 0; i < remaining; ++i)
            val |= uint32_t(tail[i]) << (24u - 8u * i);
        kw[full_words] = val;
    }
}

kernel void hs256_markov(
    constant Hs256MarkovParams& params [[buffer(0)]],
    constant const uint8_t* message_bytes [[buffer(1)]],
    constant const uint8_t* markov_table [[buffer(2)]],
    device atomic_uint* result_index [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.candidate_count) return;

    const uint32_t T = params.threshold;
    const uint32_t len = params.pw_length;
    uint64_t idx = params.offset + uint64_t(gid);

    // Generate candidate in thread-local registers.
    uint8_t candidate[64];
    uint8_t prev = 0; // start-of-word sentinel
    for (uint32_t pos = 0; pos < len; ++pos) {
        uint32_t rank = uint32_t(idx % uint64_t(T));
        idx /= uint64_t(T);
        uint32_t table_idx = (pos * 256u + uint32_t(prev)) * T + rank;
        candidate[pos] = markov_table[table_idx];
        prev = candidate[pos];
    }

    // Hash and compare using the flat HMAC path (always short-key since len <= 64).
    uint32_t kw[16];
    load_key_words_thread(candidate, len, kw);
    if (hmac_sha256_from_key_words(kw, message_bytes, params.message_length,
                                    params.target_signature)) {
        atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed);
    }
}
