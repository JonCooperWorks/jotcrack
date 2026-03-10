// ===========================================================================
// CUDA compute kernel for HMAC-SHA384 and HMAC-SHA512 JWT cracking (wordlist mode).
//
// LEARNING OVERVIEW: SHA-512 vs SHA-256
//
// This kernel implements the same HMAC-then-compare pattern as the HS256
// kernel but using SHA-512 family hashing.  Key differences from SHA-256:
//
//   SHA-256                          SHA-512
//   -------                          -------
//   32-bit words (uint32_t)          64-bit words (uint64_t)
//   64-byte (512-bit) blocks         128-byte (1024-bit) blocks
//   64 compression rounds            80 compression rounds
//   8 x uint32_t state (32 bytes)    8 x uint64_t state (64 bytes)
//   64 round constants               80 round constants
//   HMAC block size: 64 bytes        HMAC block size: 128 bytes
//   Short-key threshold: 64 bytes    Short-key threshold: 128 bytes
//
// SHA-384 is NOT a separate algorithm.  It is SHA-512 with:
//   1. Different initial values (IV) -- defined in FIPS 180-4 section 5.3.4.
//   2. Output truncated to 48 bytes (first 6 of 8 state words).
//
// This single .cu file contains kernel entry points for BOTH HS384 and HS512.
// The Rust host code in hs384wordlist/gpu.rs and hs512wordlist/gpu.rs both
// load this file and select the appropriate kernel function names.
//
// ROLLING MESSAGE SCHEDULE
//   Like the SHA-256 kernel, we use a rolling 16-word schedule (w[16]) instead
//   of pre-expanding all 80 words (w[80]).  This saves 80*8 - 16*8 = 512 bytes
//   of register pressure per thread -- critical for GPU occupancy.
//   The recurrence is: w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16]
//   which only needs the last 16 values, so w[i & 15] is updated in-place.
//
// CUDA vs METAL PORTING NOTES
//
// This file is a direct port of the Metal shader hs512_wordlist.metal.
// Key syntax transformations applied:
//
//   Metal                                  CUDA
//   -----                                  ----
//   #include <metal_stdlib>                (removed -- not needed in CUDA)
//   using namespace metal;                 (removed)
//   constant uint64_t K[80]                __constant__ unsigned long long K[80]
//   static inline fn(...)                  __device__ __forceinline__ fn(...)
//   kernel void fn(...)                    extern "C" __global__ void fn(...)
//   constant T& params                     const T* __restrict__ params
//   device T* ptr                          const T* ptr  (or T* for read-write)
//   constant const uint8_t*                const unsigned char*
//   uint gid [[thread_position_in_grid]]   unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
//   atomic_fetch_min_explicit(...)          atomicMin(...)
//   thread uint64_t state[8]               unsigned long long state[8]  (local / register)
//   params.field                           params->field
//
// In CUDA, `__constant__` memory resides in a dedicated 64 KB cache on each
// SM (streaming multiprocessor), similar to Metal's `constant` address space.
// Both provide fast broadcast reads when all threads in a warp/SIMD-group
// access the same address.
//
// `__device__ __forceinline__` is the CUDA equivalent of Metal's
// `static inline`: the function lives on the GPU and the compiler is
// strongly encouraged to inline it.  Unlike Metal, CUDA also supports
// `__noinline__` for debugging, but we always want inlining here.
//
// `extern "C" __global__` marks a kernel entry point callable from the host.
// `extern "C"` disables C++ name mangling so the Rust host can look up
// the kernel by its plain string name (e.g., "hs512_wordlist").
// ===========================================================================

// ---------------------------------------------------------------------------
// We use unsigned long long throughout for 64-bit values to ensure
// portability across CUDA toolchains.  NVRTC does support <stdint.h>,
// but unsigned long long is guaranteed to be 64-bit on all CUDA platforms.
// We typedef for readability.
// ---------------------------------------------------------------------------
typedef unsigned long long uint64;
typedef unsigned int       uint32;
typedef unsigned short     uint16;
typedef unsigned char      uint8;

// Host-to-kernel parameter block.  Rust uploads this to device memory.
// target_signature holds the expected HMAC output as big-endian uint64 words:
//   HS384: first 6 words used (48 bytes), HS512: all 8 words used (64 bytes).
//
// Learning note: the host pre-converts target bytes to uint64 words so that
// each GPU thread can do a simple word-level comparison (ostate[i] != target[i])
// instead of reconstructing words from bytes.  This is the same optimization
// used in the HS256 kernel with uint32 words.
//
// CUDA note: unlike Metal, where `constant Hs512BruteForceParams& params`
// passes the struct by reference in constant address space, CUDA passes
// a `const Hs512BruteForceParams* __restrict__ params` pointer.  The
// `__restrict__` hint tells the compiler this pointer does not alias any
// other pointer, enabling more aggressive optimization.
struct Hs512BruteForceParams {
    uint64 target_signature[8];
    uint32 message_length;
    uint32 candidate_count;
};

// ---------------------------------------------------------------------------
// SHA-512 round constants (FIPS 180-4, section 4.2.3).
// 80 constants (vs. 64 for SHA-256), each 64 bits wide.
// These are the first 64 bits of the fractional parts of the cube roots of
// the first 80 prime numbers.
//
// CUDA note: `__constant__` places these in dedicated constant memory (64 KB
// per SM), which is cached and optimized for broadcast reads -- all threads
// in a warp reading the same address get the value in a single transaction.
// This is the CUDA equivalent of Metal's `constant` address space qualifier.
// ---------------------------------------------------------------------------
__constant__ uint64 SHA512_K[80] = {
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
//
// Learning note: these are the exact same logical operations as SHA-256's
// helpers (rotr, ch, maj, big/small sigma) but operating on 64-bit words
// with different rotation amounts specified in FIPS 180-4 section 4.1.3.
//
// CUDA note: `__device__ __forceinline__` is the CUDA equivalent of Metal's
// `static inline`.  `__device__` means the function runs on the GPU (not
// callable from host code).  `__forceinline__` strongly requests the
// compiler to inline the function body at every call site, eliminating
// function-call overhead -- critical for these tiny bitwise helpers that
// are called billions of times during SHA-512 compression.
// ---------------------------------------------------------------------------

// Rotate-right for 64-bit values.  Same concept as rotr32 but for uint64.
__device__ __forceinline__ uint64 rotr64(uint64 x, uint64 n) {
    return (x >> n) | (x << (64ULL - n));
}

__device__ __forceinline__ uint64 ch64(uint64 x, uint64 y, uint64 z) {
    return (x & y) ^ ((~x) & z);
}

__device__ __forceinline__ uint64 maj64(uint64 x, uint64 y, uint64 z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

// SHA-512 big sigma 0.
__device__ __forceinline__ uint64 bsig0_64(uint64 x) {
    return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39);
}

// SHA-512 big sigma 1.
__device__ __forceinline__ uint64 bsig1_64(uint64 x) {
    return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41);
}

// SHA-512 small sigma 0 (message schedule expansion).
__device__ __forceinline__ uint64 ssig0_64(uint64 x) {
    return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7);
}

// SHA-512 small sigma 1 (message schedule expansion).
__device__ __forceinline__ uint64 ssig1_64(uint64 x) {
    return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6);
}

// ---------------------------------------------------------------------------
// Big-endian load/store for 64-bit words.
//
// CUDA note: unlike Metal, which requires separate overloads for each
// address space qualifier (thread, device, constant), CUDA uses a flat
// memory model where all pointers are in the same unified address space.
// A single function handles reads from any source (global memory, local
// variables, etc.).  This is one of the simplifications CUDA offers over
// Metal's explicit address space qualifiers.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64 load_be_u64(const uint8* p) {
    return ((uint64)(p[0]) << 56) | ((uint64)(p[1]) << 48) |
           ((uint64)(p[2]) << 40) | ((uint64)(p[3]) << 32) |
           ((uint64)(p[4]) << 24) | ((uint64)(p[5]) << 16) |
           ((uint64)(p[6]) << 8)  |  (uint64)(p[7]);
}

__device__ __forceinline__ void store_be_u64(uint8* p, uint64 v) {
    p[0] = (uint8)((v >> 56) & 0xffULL);
    p[1] = (uint8)((v >> 48) & 0xffULL);
    p[2] = (uint8)((v >> 40) & 0xffULL);
    p[3] = (uint8)((v >> 32) & 0xffULL);
    p[4] = (uint8)((v >> 24) & 0xffULL);
    p[5] = (uint8)((v >> 16) & 0xffULL);
    p[6] = (uint8)((v >> 8)  & 0xffULL);
    p[7] = (uint8)(v & 0xffULL);
}

// ---------------------------------------------------------------------------
// Rolling 16-word message schedule for SHA-512 (80 rounds).
//
// Same strategy as the SHA-256 kernel: keep only w[16] and update w[i & 15]
// in-place during rounds 16-79, saving register pressure vs. a full w[80].
// ---------------------------------------------------------------------------

// Single SHA-512 round using rolling w[i & 15] indexing.
#define SHA512_ROUND_R(i, a, b, c, d, e, f, g, h) do { \
    uint64 t1_ = (h) + bsig1_64((e)) + ch64((e), (f), (g)) + SHA512_K[(i)] + w[(i) & 15]; \
    uint64 t2_ = bsig0_64((a)) + maj64((a), (b), (c)); \
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
//
// Learning note: SHA-512 compression vs SHA-256
//
// The structure is identical to sha256_compress_rolling:
//   1. Copy state into working variables a..h.
//   2. Rounds 0-15: use the 16 message words directly.
//   3. Rounds 16-79: expand w[i&15] via the recurrence, then apply round function.
//   4. Add working variables back to state.
//
// The differences are:
//   - 80 rounds instead of 64 (so rounds 16-79 = 64 expansion rounds, vs 48).
//   - All arithmetic is 64-bit (uint64 instead of uint32).
//   - Different rotation amounts in the sigma functions.
//   - 1024-bit (128-byte) blocks instead of 512-bit (64-byte).
//
// The extra 16 rounds plus doubled word size make SHA-512 roughly 2x slower
// per byte than SHA-256 on 32-bit hardware, but on 64-bit hardware (like
// modern NVIDIA GPUs with native 64-bit integer ALUs) the gap is smaller.
//
// CUDA note: this function is `__device__ __forceinline__` rather than
// Metal's `static inline`.  Both achieve the same effect: the function
// is compiled for the GPU and inlined at every call site to avoid the
// overhead of a function call in the hot compression loop.
__device__ __noinline__ void sha512_compress_rolling(uint64 state[8], uint64 w[16]) {
    uint64 a = state[0];
    uint64 b = state[1];
    uint64 c = state[2];
    uint64 d = state[3];
    uint64 e = state[4];
    uint64 f = state[5];
    uint64 g = state[6];
    uint64 h = state[7];

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
//
// Learning note: this struct is much larger than its SHA-256 counterpart:
//   - block[128] vs block[64]  (128-byte blocks for SHA-512)
//   - state[8] as uint64 vs uint32  (64 bytes vs 32 bytes)
// Total: ~200 bytes per thread.  This is why we avoid it on the hot path
// and use bare uint64 arrays instead (the "flat" HMAC implementation).
//
// CUDA note: in Metal, this struct lives in `thread` address space (per-thread
// registers/stack).  In CUDA, local variables are automatically placed in
// registers when possible, or spilled to "local memory" (which is actually
// global memory with per-thread addressing, cached in L1/L2).  The compiler
// decides placement based on register pressure -- another reason to keep
// this struct off the hot path.
// ---------------------------------------------------------------------------

struct Sha512Ctx {
    uint64 state[8];
    uint8  block[128];            // 1024-bit block buffer.
    uint32 block_len;
    uint64 total_len;
};

// SHA-512 initial values: first 64 bits of fractional parts of square roots
// of the first 8 primes.  Note these are DIFFERENT from SHA-384's IVs.
__device__ __forceinline__ void sha512_init(Sha512Ctx& ctx) {
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

// SHA-384 initial values: first 64 bits of fractional parts of square roots
// of the 9th through 16th primes (23, 29, 31, 37, 41, 43, 47, 53).
// These are deliberately different from SHA-512's IVs so that SHA-384 and
// SHA-512 produce completely different digests for the same input, even though
// they use the same compression function.
__device__ __forceinline__ void sha384_init(Sha512Ctx& ctx) {
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

// Compress helper: load a 128-byte block from a byte pointer into 16 uint64
// words and compress into state.
//
// CUDA note: in Metal, separate overloads are needed for `thread const uint8_t*`
// vs `device const uint8_t*` vs `constant const uint8_t*` because Metal
// enforces address space qualifiers at the type level.  In CUDA, there is
// a single unified address space for device code, so one function suffices.
__device__ __forceinline__ void sha512_compress_from_bytes(uint64 state[8], const uint8* block) {
    uint64 w[16];
    for (uint32 i = 0; i < 16; ++i) {
        w[i] = load_be_u64(block + (i * 8u));
    }
    sha512_compress_rolling(state, w);
}

// Update SHA-512 state from device memory (used for hashing long keys).
__device__ __forceinline__ void sha512_update_device(Sha512Ctx& ctx, const uint8* data, uint32 len) {
    if (len == 0u) return;
    ctx.total_len += (uint64)(len);

    uint32 offset = 0u;
    if (ctx.block_len != 0u) {
        const uint32 need = 128u - ctx.block_len;
        const uint32 take = (len < need) ? len : need;
        for (uint32 i = 0; i < take; ++i) {
            ctx.block[ctx.block_len + i] = data[i];
        }
        ctx.block_len += take;
        offset += take;
        if (ctx.block_len == 128u) {
            sha512_compress_from_bytes(ctx.state, ctx.block);
            ctx.block_len = 0u;
        }
    }

    while ((len - offset) >= 128u) {
        sha512_compress_from_bytes(ctx.state, data + offset);
        offset += 128u;
    }

    for (; offset < len; ++offset) {
        ctx.block[ctx.block_len++] = data[offset];
    }
}

// Finalize SHA-512 without serializing (state[0..7] holds result as words).
__device__ __forceinline__ void sha512_finalize(Sha512Ctx& ctx) {
    const uint64 bit_len = ctx.total_len * 8ULL;
    ctx.block[ctx.block_len++] = 0x80u;
    if (ctx.block_len > 112u) {
        while (ctx.block_len < 128u) ctx.block[ctx.block_len++] = 0u;
        sha512_compress_from_bytes(ctx.state, ctx.block);
        ctx.block_len = 0u;
    }
    while (ctx.block_len < 112u) ctx.block[ctx.block_len++] = 0u;
    // SHA-512 uses a 128-bit (16-byte) big-endian length field.
    // Upper 64 bits are zero for messages < 2^64 bits.
    for (uint32 i = 0; i < 8; ++i)
        ctx.block[112u + i] = 0u;
    for (uint32 i = 0; i < 8; ++i)
        ctx.block[120u + i] = (uint8)((bit_len >> (56u - 8u * i)) & 0xffULL);
    sha512_compress_from_bytes(ctx.state, ctx.block);
    ctx.block_len = 0u;
}

// ---------------------------------------------------------------------------
// Key loading: 128-byte HMAC block as 16 x uint64 words, zero-padded.
//
// Learning note: HMAC-SHA512 block size = 128 bytes = 16 x uint64 words.
// This is double the 64-byte block size of HMAC-SHA256 (16 x uint32).
// Keys shorter than 128 bytes are zero-padded to fill one block.
// Keys longer than 128 bytes are hashed first (handled by the caller).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void load_key_words_64(const uint8* key, uint32 key_len, uint64 kw[16]) {
    #pragma unroll
    for (uint32 i = 0; i < 16; ++i) kw[i] = 0ULL;

    const uint32 full_words = key_len >> 3;
    for (uint32 i = 0; i < full_words; ++i)
        kw[i] = load_be_u64(key + i * 8u);

    const uint32 remaining = key_len & 7u;
    if (remaining > 0u) {
        uint64 val = 0ULL;
        const uint8* tail = key + (full_words * 8u);
        for (uint32 i = 0; i < remaining; ++i)
            val |= (uint64)(tail[i]) << (56u - 8u * i);
        kw[full_words] = val;
    }
}

// ---------------------------------------------------------------------------
// Flat HMAC-SHA384: no Sha512Ctx on the hot path.
//
// Learning note: HMAC-SHA384 step by step
//
// 1. INNER HASH:  SHA-384( (key XOR ipad_128) || message )
//    - ipad is 0x36 repeated to 128 bytes (= 16 x 0x3636363636363636 as u64 words).
//    - Compress the ipad block, then process message blocks, then finalize.
//    - SHA-384 output = first 6 of 8 state words = 48 bytes.
//
// 2. OUTER HASH:  SHA-384( (key XOR opad_128) || inner_digest )
//    - opad is 0x5c repeated to 128 bytes.
//    - Compress the opad block, then compress one block containing the
//      48-byte inner digest + padding + bit-length.
//    - SHA-384 output = first 6 state words = 48 bytes.
//
// 3. COMPARE:  check the first 6 words against the target.
//    (Words 6-7 of target are zeroed by the host; the kernel ignores them.)
//
// Block size = 128 bytes.  ipad/opad XOR = 0x3636363636363636 / 0x5c5c5c5c5c5c5c5c.
// Inner digest = 48 bytes (first 6 state words of SHA-384).
// Outer message = 128 (opad) + 48 (digest) = 176 bytes  ->  bit_len = 1408.
// ---------------------------------------------------------------------------

__device__ __noinline__ bool hmac_sha384_from_key_words(
    const uint64 kw[16],
    const uint8* message,
    uint32 message_len,
    const uint64 target[8]
) {
    // --- Inner hash: SHA-384(ipad || message) ---
    // Learning note: these are the SHA-384 IVs (different from SHA-512 IVs).
    // Using SHA-384 IVs here is what makes this HMAC-SHA384 and not HMAC-SHA512.
    uint64 istate[8] = {
        0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL,
        0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
        0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL,
        0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL
    };

    // Compress ipad block: kw XOR 0x3636363636363636.
    // Learning note: the ipad XOR constant is 0x36 repeated for each byte,
    // which becomes 0x3636363636363636 as a uint64 word.  This is the
    // HMAC inner padding step from RFC 2104.
    {
        uint64 w[16];
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x3636363636363636ULL;
        sha512_compress_rolling(istate, w);
    }

    // Process full 128-byte message blocks from global memory.
    // Learning note: message blocks are 128 bytes for SHA-512 (vs 64 for SHA-256).
    //
    // CUDA note: in Metal, `message` is in `constant` address space (cached,
    // broadcast-optimized).  In CUDA, this pointer is in global memory.
    // Since all threads in a warp read the same message bytes at the same
    // offsets, the L1/L2 cache provides similar broadcast behavior.
    // For even better performance, the host could place the message in
    // CUDA `__constant__` memory, but that is limited to 64 KB.
    uint32 offset = 0u;
    while ((message_len - offset) >= 128u) {
        uint64 w[16];
        for (uint32 i = 0; i < 16; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }
        sha512_compress_rolling(istate, w);
        offset += 128u;
    }

    // Finalize inner hash: remaining bytes + 0x80 padding + 128-bit bit length.
    const uint32 rem = message_len - offset;
    const uint64 inner_bit_len = (128ULL + (uint64)(message_len)) * 8ULL;
    {
        uint64 w[16];
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) w[i] = 0ULL;

        // Load remaining full 8-byte words from message.
        const uint32 full_words = rem >> 3;
        for (uint32 i = 0; i < full_words; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }

        // Load partial trailing bytes and append 0x80 padding byte.
        const uint32 tail = rem & 7u;
        uint64 pad_word = 0ULL;
        const uint8* tail_ptr = message + offset + (full_words * 8u);
        for (uint32 i = 0; i < tail; ++i) {
            pad_word |= (uint64)(tail_ptr[i]) << (56u - 8u * i);
        }
        pad_word |= (uint64)(0x80u) << (56u - 8u * tail);
        w[full_words] = pad_word;

        if (rem > 111u) {
            // Two-block finalization: data + 0x80 don't fit with the 16-byte length field.
            sha512_compress_rolling(istate, w);
            #pragma unroll
            for (uint32 i = 0; i < 16; ++i) w[i] = 0ULL;
        }
        // 128-bit big-endian bit length: upper 64 bits = 0, lower 64 bits = inner_bit_len.
        w[14] = 0ULL;
        w[15] = inner_bit_len;
        sha512_compress_rolling(istate, w);
    }
    // istate[0..5] now holds the 48-byte SHA-384 inner digest as words.

    // --- Outer hash: SHA-384(opad || inner_digest) ---
    uint64 ostate[8] = {
        0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL,
        0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
        0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL,
        0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL
    };

    // Compress opad block: kw XOR 0x5c5c5c5c5c5c5c5c.
    {
        uint64 w[16];
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        sha512_compress_rolling(ostate, w);
    }

    // Compress final outer block: 48-byte inner digest + padding.
    // Outer message total = 128 (opad) + 48 (digest) = 176 bytes -> bit_len = 1408.
    {
        uint64 w[16];
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
    // Learning note: SHA-384 truncates to 6 words.  The target's words 6-7
    // are zeroed by the host, and we simply don't compare them.  This is
    // the key difference from HS512, which compares all 8 words below.
    #pragma unroll
    for (uint32 i = 0; i < 6; ++i) {
        if (ostate[i] != target[i]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Flat HMAC-SHA512: no Sha512Ctx on the hot path.
//
// Learning note: HMAC-SHA512 step by step
//
// Same structure as HMAC-SHA384 above, but with two differences:
//   1. Uses SHA-512 IVs (not SHA-384 IVs).
//   2. Inner digest = 64 bytes (all 8 state words, not just 6).
//   3. Comparison checks all 8 words (not 6).
//
// Outer hash layout:
//   Block 1: opad (128 bytes = one full block).
//   Block 2: 64-byte inner digest + 0x80 + zero padding + 16-byte bit-length.
//            = 64 + 1 + 47 + 16 = 128 bytes -> fits in one block.
//   Total outer message = 128 + 64 = 192 bytes -> 1536 bits.
//
// Block size = 128 bytes.  ipad/opad XOR = 0x3636363636363636 / 0x5c5c5c5c5c5c5c5c.
// Inner digest = 64 bytes (all 8 state words of SHA-512).
// Outer message = 128 (opad) + 64 (digest) = 192 bytes  ->  bit_len = 1536.
// ---------------------------------------------------------------------------

__device__ __noinline__ bool hmac_sha512_from_key_words(
    const uint64 kw[16],
    const uint8* message,
    uint32 message_len,
    const uint64 target[8]
) {
    // --- Inner hash: SHA-512(ipad || message) ---
    // Learning note: these are the SHA-512 IVs (different from SHA-384 IVs above).
    uint64 istate[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    // Compress ipad block: kw XOR 0x3636363636363636.
    {
        uint64 w[16];
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x3636363636363636ULL;
        sha512_compress_rolling(istate, w);
    }

    // Process full 128-byte message blocks from global memory.
    uint32 offset = 0u;
    while ((message_len - offset) >= 128u) {
        uint64 w[16];
        for (uint32 i = 0; i < 16; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }
        sha512_compress_rolling(istate, w);
        offset += 128u;
    }

    // Finalize inner hash: remaining bytes + 0x80 padding + 128-bit bit length.
    const uint32 rem = message_len - offset;
    const uint64 inner_bit_len = (128ULL + (uint64)(message_len)) * 8ULL;
    {
        uint64 w[16];
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) w[i] = 0ULL;

        const uint32 full_words = rem >> 3;
        for (uint32 i = 0; i < full_words; ++i) {
            w[i] = load_be_u64(message + offset + i * 8u);
        }

        const uint32 tail = rem & 7u;
        uint64 pad_word = 0ULL;
        const uint8* tail_ptr = message + offset + (full_words * 8u);
        for (uint32 i = 0; i < tail; ++i) {
            pad_word |= (uint64)(tail_ptr[i]) << (56u - 8u * i);
        }
        pad_word |= (uint64)(0x80u) << (56u - 8u * tail);
        w[full_words] = pad_word;

        if (rem > 111u) {
            sha512_compress_rolling(istate, w);
            #pragma unroll
            for (uint32 i = 0; i < 16; ++i) w[i] = 0ULL;
        }
        w[14] = 0ULL;
        w[15] = inner_bit_len;
        sha512_compress_rolling(istate, w);
    }
    // istate[0..7] now holds the 64-byte SHA-512 inner digest.

    // --- Outer hash: SHA-512(opad || inner_digest) ---
    uint64 ostate[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    // Compress opad block: kw XOR 0x5c5c5c5c5c5c5c5c.
    {
        uint64 w[16];
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        sha512_compress_rolling(ostate, w);
    }

    // Compress final outer block: 64-byte inner digest + padding + length.
    // 64 bytes digest = 8 words.  Remaining in block: 8 words (64 bytes).
    // Need: 1 byte (0x80) + padding + 16 bytes (length) = 17 bytes minimum.
    // 64 bytes available >= 17, so it fits in one block.
    // Outer message total = 128 (opad) + 64 (digest) = 192 bytes -> bit_len = 1536.
    {
        uint64 w[16];
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
    // Learning note: this is the key difference from HS384 above, which only
    // compares 6 words.  HS512 uses the full SHA-512 output (all 8 state words).
    #pragma unroll
    for (uint32 i = 0; i < 8; ++i) {
        if (ostate[i] != target[i]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Full HMAC wrappers with long-key fallback (key > 128 bytes -> hash first).
//
// Learning note: per RFC 2104, if the key is longer than the hash block size
// (128 bytes for SHA-512 family), it must be hashed first to reduce it.
// For HMAC-SHA384, long keys are hashed with SHA-384 (48-byte result).
// For HMAC-SHA512, long keys are hashed with SHA-512 (64-byte result).
// The "short key" specializations below skip this branch entirely when the
// host guarantees all candidates in the batch are <= 128 bytes.
// ---------------------------------------------------------------------------

// HMAC-SHA384: any key length.
__device__ __forceinline__ bool hmac_sha384_matches(
    const uint8* key,
    uint32 key_len,
    const uint8* message,
    uint32 message_len,
    const uint64 target[8]
) {
    uint64 kw[16];

    if (key_len > 128u) {
        // Long keys: hash with SHA-384 first (produces 48 bytes = 6 words).
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) kw[i] = 0ULL;
        Sha512Ctx key_ctx;
        sha384_init(key_ctx);
        sha512_update_device(key_ctx, key, key_len);
        sha512_finalize(key_ctx);
        #pragma unroll
        for (uint32 i = 0; i < 6; ++i) kw[i] = key_ctx.state[i];
    } else {
        load_key_words_64(key, key_len, kw);
    }

    return hmac_sha384_from_key_words(kw, message, message_len, target);
}

// HMAC-SHA384: short-key specialization (keys guaranteed <= 128 bytes).
__device__ __forceinline__ bool hmac_sha384_short_key_matches(
    const uint8* key,
    uint32 key_len,
    const uint8* message,
    uint32 message_len,
    const uint64 target[8]
) {
    uint64 kw[16];
    load_key_words_64(key, key_len, kw);
    return hmac_sha384_from_key_words(kw, message, message_len, target);
}

// HMAC-SHA512: any key length.
__device__ __forceinline__ bool hmac_sha512_matches(
    const uint8* key,
    uint32 key_len,
    const uint8* message,
    uint32 message_len,
    const uint64 target[8]
) {
    uint64 kw[16];

    if (key_len > 128u) {
        // Long keys: hash with SHA-512 first (produces 64 bytes = 8 words).
        #pragma unroll
        for (uint32 i = 0; i < 16; ++i) kw[i] = 0ULL;
        Sha512Ctx key_ctx;
        sha512_init(key_ctx);
        sha512_update_device(key_ctx, key, key_len);
        sha512_finalize(key_ctx);
        #pragma unroll
        for (uint32 i = 0; i < 8; ++i) kw[i] = key_ctx.state[i];
    } else {
        load_key_words_64(key, key_len, kw);
    }

    return hmac_sha512_from_key_words(kw, message, message_len, target);
}

// HMAC-SHA512: short-key specialization (keys guaranteed <= 128 bytes).
__device__ __forceinline__ bool hmac_sha512_short_key_matches(
    const uint8* key,
    uint32 key_len,
    const uint8* message,
    uint32 message_len,
    const uint64 target[8]
) {
    uint64 kw[16];
    load_key_words_64(key, key_len, kw);
    return hmac_sha512_from_key_words(kw, message, message_len, target);
}

// ---------------------------------------------------------------------------
// Kernel entry points.
//
// Learning note: this single .cu file provides FOUR kernel entry points:
//   - hs384_wordlist              (any key length, compares 6 words)
//   - hs384_wordlist_short_keys   (keys <= 128 bytes, compares 6 words)
//   - hs512_wordlist              (any key length, compares 8 words)
//   - hs512_wordlist_short_keys   (keys <= 128 bytes, compares 8 words)
//
// The Rust host selects which kernel to dispatch:
//   - hs384wordlist/gpu.rs uses the hs384_* entry points.
//   - hs512wordlist/gpu.rs uses the hs512_* entry points.
//   - Within each, the "short_keys" variant is chosen when the batch's
//     max_word_len <= 128, eliminating the long-key hash-first branch.
//
// CUDA note on kernel parameters:
//   Metal uses numbered buffer bindings ([[buffer(0)]], [[buffer(1)]], etc.)
//   and passes structs by reference in `constant` address space.  CUDA
//   passes kernel arguments as regular function parameters -- pointers to
//   device memory allocated and filled by the host via cudaMalloc/cudaMemcpy.
//
//   Metal gets the thread ID via `uint gid [[thread_position_in_grid]]`.
//   CUDA computes it manually: gid = blockIdx.x * blockDim.x + threadIdx.x.
//   This is the standard 1D grid indexing pattern where:
//     - blockIdx.x  = index of this thread block within the grid
//     - blockDim.x  = number of threads per block
//     - threadIdx.x = index of this thread within its block
//   The bounds check `if (gid >= candidate_count) return;` handles the case
//   where the total number of threads launched (rounded up to a multiple of
//   blockDim.x) exceeds the actual number of candidates.
//
//   Metal uses `atomic_fetch_min_explicit(ptr, val, memory_order_relaxed)`.
//   CUDA uses `atomicMin(ptr, val)`, which provides the same relaxed-ordering
//   atomic minimum.  Both ensure that if multiple threads find a match, the
//   lowest candidate index wins.
//
//   `extern "C"` disables C++ name mangling so the Rust host can find the
//   kernel by its plain string name when using the CUDA driver API
//   (cuModuleGetFunction / cuLaunchKernel).
//
// Buffer layout (matching the Metal kernel):
//   params         = Hs512BruteForceParams (target signature + lengths)
//   message_bytes  = JWT header.payload bytes
//   word_bytes     = concatenated candidate secrets
//   word_offsets   = byte offset of each candidate in word_bytes
//   word_lengths   = byte length of each candidate
//   result_index   = atomic min: lowest matching gid
// ---------------------------------------------------------------------------

extern "C" __global__ void hs384_wordlist(
    const Hs512BruteForceParams* __restrict__ params,
    const uint8* message_bytes,
    const uint8* word_bytes,
    const uint32* word_offsets,
    const uint16* word_lengths,
    uint32* result_index
) {
    // Compute global thread ID -- CUDA equivalent of Metal's
    // `uint gid [[thread_position_in_grid]]`.
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->candidate_count) return;

    const uint32 candidate_offset = word_offsets[gid];
    const uint32 candidate_length = (uint32)(word_lengths[gid]);
    const uint8* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha384_matches(candidate_secret, candidate_length,
                            message_bytes, params->message_length,
                            params->target_signature)) {
        // CUDA note: atomicMin provides relaxed-ordering atomic minimum,
        // equivalent to Metal's atomic_fetch_min_explicit with memory_order_relaxed.
        // If multiple threads find a match, the lowest gid wins.
        atomicMin(result_index, gid);
    }
}

extern "C" __global__ void hs384_wordlist_short_keys(
    const Hs512BruteForceParams* __restrict__ params,
    const uint8* message_bytes,
    const uint8* word_bytes,
    const uint32* word_offsets,
    const uint16* word_lengths,
    uint32* result_index
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->candidate_count) return;

    const uint32 candidate_offset = word_offsets[gid];
    const uint32 candidate_length = (uint32)(word_lengths[gid]);
    const uint8* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha384_short_key_matches(candidate_secret, candidate_length,
                                      message_bytes, params->message_length,
                                      params->target_signature)) {
        atomicMin(result_index, gid);
    }
}

extern "C" __global__ void hs512_wordlist(
    const Hs512BruteForceParams* __restrict__ params,
    const uint8* message_bytes,
    const uint8* word_bytes,
    const uint32* word_offsets,
    const uint16* word_lengths,
    uint32* result_index
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->candidate_count) return;

    const uint32 candidate_offset = word_offsets[gid];
    const uint32 candidate_length = (uint32)(word_lengths[gid]);
    const uint8* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha512_matches(candidate_secret, candidate_length,
                            message_bytes, params->message_length,
                            params->target_signature)) {
        atomicMin(result_index, gid);
    }
}

extern "C" __global__ void hs512_wordlist_short_keys(
    const Hs512BruteForceParams* __restrict__ params,
    const uint8* message_bytes,
    const uint8* word_bytes,
    const uint32* word_offsets,
    const uint16* word_lengths,
    uint32* result_index
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->candidate_count) return;

    const uint32 candidate_offset = word_offsets[gid];
    const uint32 candidate_length = (uint32)(word_lengths[gid]);
    const uint8* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha512_short_key_matches(candidate_secret, candidate_length,
                                      message_bytes, params->message_length,
                                      params->target_signature)) {
        atomicMin(result_index, gid);
    }
}

// ===========================================================================
// Markov chain candidate generation + HMAC-SHA384/512 (fused kernels).
// ===========================================================================

struct Hs512MarkovParams {
    uint64 target_signature[8];
    uint32 message_length;
    uint32 candidate_count;
    uint32 pw_length;
    uint32 threshold;
    uint64 offset;
};

extern "C" __global__ void hs384_markov(
    const Hs512MarkovParams* __restrict__ params,
    const uint8* __restrict__ message_bytes,
    const uint8* __restrict__ markov_table,
    uint32* __restrict__ result_index
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->candidate_count) return;

    const uint32 T = params->threshold;
    const uint32 len = params->pw_length;
    uint64 idx = params->offset + (uint64)(gid);

    uint8 candidate[128];
    uint8 prev = 0;
    for (uint32 pos = 0; pos < len; ++pos) {
        uint32 rank = (uint32)(idx % (uint64)(T));
        idx /= (uint64)(T);
        uint32 table_idx = (pos * 256u + (uint32)(prev)) * T + rank;
        candidate[pos] = markov_table[table_idx];
        prev = candidate[pos];
    }

    uint64 kw[16];
    load_key_words_64(candidate, len, kw);
    if (hmac_sha384_from_key_words(kw, message_bytes, params->message_length,
                                    params->target_signature)) {
        atomicMin(result_index, gid);
    }
}

extern "C" __global__ void hs512_markov(
    const Hs512MarkovParams* __restrict__ params,
    const uint8* __restrict__ message_bytes,
    const uint8* __restrict__ markov_table,
    uint32* __restrict__ result_index
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->candidate_count) return;

    const uint32 T = params->threshold;
    const uint32 len = params->pw_length;
    uint64 idx = params->offset + (uint64)(gid);

    uint8 candidate[128];
    uint8 prev = 0;
    for (uint32 pos = 0; pos < len; ++pos) {
        uint32 rank = (uint32)(idx % (uint64)(T));
        idx /= (uint64)(T);
        uint32 table_idx = (pos * 256u + (uint32)(prev)) * T + rank;
        candidate[pos] = markov_table[table_idx];
        prev = candidate[pos];
    }

    uint64 kw[16];
    load_key_words_64(candidate, len, kw);
    if (hmac_sha512_from_key_words(kw, message_bytes, params->message_length,
                                    params->target_signature)) {
        atomicMin(result_index, gid);
    }
}
